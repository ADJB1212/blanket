use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use pyo3::exceptions::{PyMemoryError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyList;

use super::PixelMode;
use super::validation::{expected_len, validate_raw_data};

/// Allocate a zeroed pixel buffer of `len` bytes, returning a memory error on failure.
pub fn alloc_pixels(len: usize) -> PyResult<Vec<u8>> {
    if len == 0 {
        return Ok(Vec::new());
    }
    let layout = std::alloc::Layout::array::<u8>(len).map_err(|_| PyMemoryError::new_err("cannot allocate image"))?;
    // SAFETY: The nonzero layout matches Vec<u8>; all bytes are initialized to zero.
    let pointer = unsafe { std::alloc::alloc_zeroed(layout) };
    if pointer.is_null() {
        return Err(PyMemoryError::new_err("cannot allocate image"));
    }
    // SAFETY: Transfer the global allocation with its exact size and alignment.
    Ok(unsafe { Vec::from_raw_parts(pointer, len, len) })
}

/// Immutable external pixel storage; writers must first detach into owned pixels.
pub trait SharedPixels: std::any::Any + Send + Sync + pyo3::marker::Ungil {
    fn bytes(&self) -> &[u8];
    fn as_any(&self) -> &dyn std::any::Any;
}

#[derive(Debug)]
pub(super) struct SampleValidation(AtomicBool);

impl SampleValidation {
    const fn new() -> Self {
        Self(AtomicBool::new(true))
    }
}

impl Clone for SampleValidation {
    fn clone(&self) -> Self {
        Self(AtomicBool::new(self.0.load(Ordering::Relaxed)))
    }
}

#[pyclass(name = "_Image", module = "blanket._blanket")]
#[derive(Clone)]
pub struct Image {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) mode: PixelMode,
    // Owned snapshots share their allocation until a mutable borrow.
    pub(super) pixels: Option<Arc<Vec<u8>>>,
    pub(crate) shared_pixels: Option<std::sync::Arc<dyn SharedPixels>> = None,
    pub(crate) bit_depth: u8 = 8,
    pub format: Option<String> = None,
    pub(crate) palette: Option<(PixelMode, Vec<u8>)> = None,
    pub(super) sample_validation: SampleValidation = SampleValidation::new(),
}

impl std::fmt::Debug for Image {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Image")
            .field("size", &self.size())
            .field("mode", &self.mode)
            .field("bit_depth", &self.bit_depth)
            .field("format", &self.format)
            .finish_non_exhaustive()
    }
}

impl Image {
    pub fn pixel_mode(&self) -> PixelMode {
        self.mode
    }

    pub fn palette(&self) -> &Option<(PixelMode, Vec<u8>)> {
        &self.palette
    }

    pub fn copy_palette_from(&mut self, image: &Self) {
        self.palette.clone_from(&image.palette);
    }

    pub fn set_palette_data(&mut self, palette: Option<(PixelMode, Vec<u8>)>) -> PyResult<()> {
        if let Some((mode, data)) = &palette
            && (!matches!(mode, PixelMode::Rgb | PixelMode::Rgba)
                || data.len() > 256 * mode.channels()
                || !data.len().is_multiple_of(mode.channels()))
        {
            return Err(PyValueError::new_err("invalid palette"));
        }
        self.palette = palette;
        Ok(())
    }

    /// Mutable owned samples, without allowing the storage length to change.
    /// Modified 10/12-bit samples are validated on their next read.
    pub fn pixels_mut(&mut self) -> Option<&mut [u8]> {
        if matches!(self.bit_depth, 10 | 12) {
            self.sample_validation.0.store(false, Ordering::Relaxed);
        }
        self.pixels.as_mut().map(|pixels| Arc::make_mut(pixels).as_mut_slice())
    }

    pub fn shared_pixels(&self) -> Option<&dyn SharedPixels> {
        self.shared_pixels.as_deref()
    }

    pub fn into_pixels(mut self) -> PyResult<Vec<u8>> {
        self.raw_data()?;
        self.ensure_owned()?;
        let pixels = self.pixels.take().ok_or_else(|| PyValueError::new_err("operation on closed image"))?;
        match Arc::try_unwrap(pixels) {
            Ok(pixels) => Ok(pixels),
            Err(shared) => crate::buffers::copy(&shared),
        }
    }

    /// Replace samples and their layout while preserving this image's metadata.
    pub fn replace_storage(&mut self, image: Self) -> PyResult<()> {
        if self.size() != image.size() {
            return Err(PyValueError::new_err("replacement image dimensions do not match"));
        }
        self.mode = image.mode;
        self.bit_depth = image.bit_depth;
        self.pixels = image.pixels;
        self.shared_pixels = image.shared_pixels;
        self.sample_validation = image.sample_validation;
        Ok(())
    }

    /// Reinterpret existing samples only when their length and depth fit the new mode.
    pub fn reinterpret_mode(&mut self, mode: PixelMode) -> PyResult<()> {
        validate_raw_data(self.width, self.height, mode, self.bit_depth, self.raw_data()?)?;
        self.mode = mode;
        Ok(())
    }

    /// Construct owned storage using the same mode/depth checks as typed constructors.
    pub fn from_raw_bytes(width: u32, height: u32, mode: PixelMode, pixels: Vec<u8>, bit_depth: u8) -> PyResult<Self> {
        if mode.is_float() && bit_depth == 32 {
            Self::from_color_float_bytes(width, height, mode, pixels)
        } else if mode.is_integer() && bit_depth as usize == mode.sample_bytes() * 8 {
            Self::from_integer_bytes(width, height, mode, pixels)
        } else if bit_depth == 8 {
            Self::from_pixels(width, height, mode, pixels, None)
        } else {
            Self::from_sample_bytes(width, height, mode, pixels, bit_depth, None)
        }
    }

    /// Construct an image backed by immutable external storage.
    pub fn from_shared(width: u32, height: u32, mode: PixelMode, bit_depth: u8, storage: std::sync::Arc<dyn SharedPixels>) -> PyResult<Self> {
        validate_raw_data(width, height, mode, bit_depth, storage.bytes())?;
        Ok(Self { width, height, mode, pixels: None, shared_pixels: Some(storage), bit_depth, .. })
    }

    pub fn ensure_owned(&mut self) -> PyResult<()> {
        if self.pixels.is_none() && self.shared_pixels.is_some() {
            let pixels = crate::buffers::copy(self.raw_data()?)?;
            self.pixels = Some(Arc::new(pixels));
            self.shared_pixels = None;
        }
        Ok(())
    }

    pub(super) fn write_pixels(&mut self, py: Python<'_>) -> PyResult<&mut [u8]> {
        let pixels = self.pixels.as_mut().expect("validated owned image");
        if Arc::strong_count(pixels) > 1 {
            let copy = py.detach(|| crate::buffers::copy(pixels.as_slice()))?;
            *pixels = Arc::new(copy);
        }
        Ok(Arc::make_mut(pixels).as_mut_slice())
    }
    pub fn from_pixels(width: u32, height: u32, mode: PixelMode, pixels: Vec<u8>, format: Option<String>) -> PyResult<Self> {
        if mode.is_wide_scalar() {
            return Err(PyValueError::new_err("wide scalar modes require 32-bit or 16-bit sample storage"));
        }
        let expected = expected_len(width, height, mode)?;
        if pixels.len() != expected {
            return Err(PyValueError::new_err(format!("not enough image data: expected {expected} bytes, got {}", pixels.len())));
        }
        Ok(Self { width, height, mode, pixels: Some(Arc::new(pixels)), format, .. })
    }

    pub fn pixel_data(&self) -> PyResult<&[u8]> {
        if self.bit_depth != 8 || self.mode.is_wide_scalar() {
            return Err(PyValueError::new_err("this operation requires 8-bit pixels; use convert(..., bit_depth=8) explicitly"));
        }
        self.raw_data()
    }

    pub fn from_integer_bytes(width: u32, height: u32, mode: PixelMode, pixels: Vec<u8>) -> PyResult<Self> {
        let expected = expected_len(width, height, mode)?
            .checked_mul(mode.sample_bytes())
            .ok_or_else(|| PyValueError::new_err("image dimensions are too large"))?;
        if !mode.is_integer() || pixels.len() != expected {
            return Err(PyValueError::new_err("wrong amount of image data"));
        }
        Ok(Self { width, height, mode, pixels: Some(Arc::new(pixels)), bit_depth: (mode.sample_bytes() << 3) as u8, .. })
    }

    pub fn from_float_bytes(width: u32, height: u32, pixels: Vec<u8>) -> PyResult<Self> {
        Self::from_color_float_bytes(width, height, PixelMode::F, pixels)
    }

    pub fn from_color_float_bytes(width: u32, height: u32, mode: PixelMode, pixels: Vec<u8>) -> PyResult<Self> {
        let expected = expected_len(width, height, mode)?.checked_mul(4).ok_or_else(|| PyValueError::new_err("image dimensions are too large"))?;
        if !mode.is_float() || pixels.len() != expected {
            return Err(PyValueError::new_err("wrong amount of image data"));
        }
        Ok(Self { width, height, mode, pixels: Some(Arc::new(pixels)), bit_depth: 32, .. })
    }

    pub fn raw_data(&self) -> PyResult<&[u8]> {
        if let Some(pixels) = &self.pixels {
            if !self.sample_validation.0.load(Ordering::Relaxed) {
                validate_raw_data(self.width, self.height, self.mode, self.bit_depth, pixels)?;
                self.sample_validation.0.store(true, Ordering::Relaxed);
            }
            return Ok(pixels);
        }
        let pixels = self.shared_pixels.as_ref().ok_or_else(|| PyValueError::new_err("operation on closed image"))?.bytes();
        validate_raw_data(self.width, self.height, self.mode, self.bit_depth, pixels)?;
        Ok(pixels)
    }

    /// Bytes occupied by one pixel in raw storage: accounts for wide scalars,
    /// 16-bit high-bit-depth samples, and ordinary 8-bit packed pixels.
    pub fn raw_bytes_per_pixel(&self) -> usize {
        self.mode.channels()
            * if self.mode.is_wide_scalar() {
                self.mode.sample_bytes()
            } else if self.bit_depth > 8 {
                2
            } else {
                1
            }
    }

    pub fn from_samples(width: u32, height: u32, mode: PixelMode, samples: Vec<u16>, bit_depth: u8, format: Option<String>) -> PyResult<Self> {
        let pixels = crate::samples::encode(&samples, false)?;
        Self::from_sample_bytes(width, height, mode, pixels, bit_depth, format)
    }

    pub fn from_sample_bytes(width: u32, height: u32, mode: PixelMode, pixels: Vec<u8>, bit_depth: u8, format: Option<String>) -> PyResult<Self> {
        if !matches!(mode, PixelMode::L | PixelMode::Rgb | PixelMode::Rgba) {
            return Err(PyValueError::new_err("this mode requires 8-bit pixels"));
        }
        if !matches!(bit_depth, 10 | 12 | 16) {
            return Err(PyValueError::new_err("bit_depth must be 8, 10, 12, or 16"));
        }
        if !pixels.len().is_multiple_of(2)
            || pixels.len() / 2 != expected_len(width, height, mode)?
            || bit_depth != 16 && !crate::pixels::samples_in_range(&pixels, bit_depth)
        {
            return Err(PyValueError::new_err("invalid sample count or sample outside bit_depth range"));
        }
        Ok(Self { width, height, mode, pixels: Some(Arc::new(pixels)), bit_depth, format, .. })
    }

    pub(super) fn color_count_list<'py>(&self, py: Python<'py>, entries: impl IntoIterator<Item = (usize, usize)>) -> PyResult<Bound<'py, PyList>> {
        let result = PyList::empty(py);
        for (count, i) in entries {
            let pixel = self.getpixel(py, ((i % self.width as usize) as i64, (i / self.width as usize) as i64))?;
            result.append((count, pixel))?;
        }
        Ok(result)
    }

    pub fn alpha_palette(&self, alpha: Option<&[u8]>, index: Option<u8>) -> PyResult<Vec<u8>> {
        let (mode, entries) = self.palette.as_ref().ok_or_else(|| PyValueError::new_err("image has no palette"))?;
        if alpha.is_some_and(|alpha| alpha.len() > 256) {
            return Err(PyValueError::new_err("invalid transparency table"));
        }
        let mut output = crate::buffers::reserved(256 * 4)?;
        for i in 0..256 {
            let start = i * mode.channels();
            let entry = entries.get(start..start + mode.channels()).unwrap_or_else(|| &[0, 0, 0, 255][..mode.channels()]);
            let a = if let Some(index) = index {
                if i == usize::from(index) { 0 } else { entry.get(3).copied().unwrap_or(255) }
            } else {
                alpha.and_then(|alpha| alpha.get(i)).copied().unwrap_or(255)
            };
            output.extend_from_slice(&[entry[0], entry[1], entry[2], a]);
        }
        Ok(output)
    }
}
