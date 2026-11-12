use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{PyIndexError, PyMemoryError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyByteArray, PyBytes, PyDict, PyInt, PyList, PyMemoryView, PySlice, PyTuple};
use rustc_hash::FxHashMap;

// Golden-ratio multipliers spread packed pixel keys across the high hash bits.
const PIXEL_HASH_MULTIPLIER: u32 = 0x9e37_79b9;
const COLOR_HASH_MULTIPLIER: u64 = 0x9e37_79b9_7f4a_7c15;
// A direct-mapped cache reuses Python tuples for repeated pixel values.
const PIXEL_CACHE_BITS: u32 = 8;
const PIXEL_CACHE_SIZE: usize = 1 << PIXEL_CACHE_BITS;
// Twice the 256-color palette limit keeps linear probing at most half full.
const COLOR_TABLE_BITS: u32 = 9;
const COLOR_TABLE_SIZE: usize = 1 << COLOR_TABLE_BITS;
// Row-aligned bilevel tasks target 128 KiB of unpacked samples.
const BILEVEL_CHUNK_BYTES: usize = 128 * 1024;
// Packing and unpacking use separate thresholds because their output sizes differ.
const PACK_PARALLEL_BYTES: usize = 512 * 1024;
const UNPACK_PARALLEL_BYTES: usize = 4 * 1024 * 1024;

const fn bilevel_bytes() -> [u64; 256] {
    let mut table = [0; 256];
    let mut value = 0;
    while value < 256 {
        let mut bit = 0;
        while bit < 8 {
            if value & (128 >> bit) != 0 {
                table[value] |= 255_u64 << (bit * 8);
            }
            bit += 1;
        }
        value += 1;
    }
    table
}

const BILEVEL_BYTES: [u64; 256] = bilevel_bytes();

pub fn unpack_bilevel(data: &[u8], width: usize, height: usize) -> Vec<u8> {
    unpack_bilevel_bytes::<false>(data, width, height)
}

pub fn unpack_bilevel_inverted(data: &[u8], width: usize, height: usize) -> Vec<u8> {
    unpack_bilevel_bytes::<true>(data, width, height)
}

fn unpack_bilevel_bytes<const INVERT: bool>(data: &[u8], width: usize, height: usize) -> Vec<u8> {
    let len = width * height;
    let mut pixels = Vec::with_capacity(len);
    if width == 0 {
        return pixels;
    }
    let row_bytes = width.div_ceil(8);
    let rows_per_chunk = (BILEVEL_CHUNK_BYTES / width).max(1);
    crate::parallel::chunks_mut_above(&mut pixels.spare_capacity_mut()[..len], rows_per_chunk * width, UNPACK_PARALLEL_BYTES, |chunk, rows| {
        for (row_index, row) in rows.chunks_mut(width).enumerate() {
            let y = chunk * rows_per_chunk + row_index;
            let packed = &data[y * row_bytes..(y + 1) * row_bytes];
            let (groups, tail) = row.as_chunks_mut::<8>();
            for (byte, dst) in packed.iter().zip(groups) {
                let value = BILEVEL_BYTES[*byte as usize];
                dst.write_copy_of_slice(&(if INVERT { !value } else { value }).to_le_bytes());
            }
            for (x, dst) in tail.iter_mut().enumerate() {
                dst.write(if (packed[width >> 3] & (128 >> x) != 0) != INVERT { 255 } else { 0 });
            }
        }
    });
    // SAFETY: Every reserved byte, including partial rows, was initialized above.
    unsafe { pixels.set_len(len) };
    pixels
}

pub fn pack_bilevel(pixels: &[u8], width: usize, height: usize) -> Vec<u8> {
    let row_bytes = width.div_ceil(8);
    let mut packed = vec![0; row_bytes * height];
    pack_bilevel_into(pixels, width, &mut packed);
    packed
}

pub fn pack_bilevel_into(pixels: &[u8], width: usize, packed: &mut [u8]) {
    if packed.is_empty() {
        return;
    }
    let row_bytes = width.div_ceil(8);
    let rows_per_chunk = (BILEVEL_CHUNK_BYTES / width).max(1);
    crate::parallel::chunks_mut_above(packed, rows_per_chunk * row_bytes, PACK_PARALLEL_BYTES, |chunk, rows| {
        for (row_index, row) in rows.chunks_mut(row_bytes).enumerate() {
            let y = chunk * rows_per_chunk + row_index;
            crate::pixels::pack_bilevel::<false>(&pixels[y * width..(y + 1) * width], row);
        }
    });
}

fn parse_pixel(value: &Bound<'_, PyAny>, channels: usize) -> PyResult<[u8; 4]> {
    if !value.is_exact_instance_of::<PyTuple>()
        && let Ok(number) = value.extract::<i64>()
    {
        return Ok(if channels == 1 { [number.clamp(0, 255) as u8, 0, 0, 0] } else { std::array::from_fn(|i| ((number >> (i * 8)) & 255) as u8) });
    }
    let tuple = value.cast::<PyTuple>().map_err(|_| PyTypeError::new_err("color must be int or tuple"))?;
    let count = tuple.len();
    if !(count == channels || channels >= 3 && matches!(count, 3 | 4)) {
        return Err(PyTypeError::new_err("wrong number of color components"));
    }
    let mut pixel = [0, 0, 0, 255];
    for (i, component) in pixel.iter_mut().enumerate().take(channels.min(count)) {
        *component = tuple.get_borrowed_item(i)?.extract::<i64>()?.clamp(0, 255) as u8;
    }
    Ok(pixel)
}

type ColorCounts = [(u64, usize, usize); COLOR_TABLE_SIZE];

fn pixel_sequence<'py>(
    py: Python<'py>, items: impl ExactSizeIterator<Item = PyResult<Bound<'py, PyAny>>>, tuple: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let len = isize::try_from(items.len()).map_err(|_| PyMemoryError::new_err("too many pixels"))?;
    // SAFETY: The container stays private while its initially null slots are filled.  CPython safely releases partially filled containers if conversion fails.
    let result = unsafe { Bound::from_owned_ptr_or_err(py, if tuple { pyo3::ffi::PyTuple_New(len) } else { pyo3::ffi::PyList_New(len) }) }?;
    for (i, item) in items.enumerate() {
        let item = item?;
        // SAFETY: Each in-bounds slot is assigned once, stealing the owned reference.
        unsafe {
            if tuple {
                pyo3::ffi::PyTuple_SET_ITEM(result.as_ptr(), i as isize, item.into_ptr());
            } else {
                pyo3::ffi::PyList_SET_ITEM(result.as_ptr(), i as isize, item.into_ptr());
            }
        }
    }
    Ok(result)
}

fn packed_pixel_data<'py, const C: usize>(py: Python<'py>, data: &[u8], tuple: bool) -> PyResult<Bound<'py, PyAny>> {
    let values: [_; 256] = std::array::from_fn(|v| (v as u8).into_pyobject(py).unwrap());
    let mut cache: [Option<([u8; C], Bound<'py, PyAny>)>; PIXEL_CACHE_SIZE] = std::array::from_fn(|_| None);
    let items = data.as_chunks::<C>().0.iter().map(|pixel| {
        let mut key = [0; 4];
        key[..C].copy_from_slice(pixel);
        let slot = (u32::from_le_bytes(key).wrapping_mul(PIXEL_HASH_MULTIPLIER) >> (u32::BITS - PIXEL_CACHE_BITS)) as usize;
        let item = if let Some((_, item)) = cache[slot].as_ref().filter(|(key, _)| key == pixel) {
            item.clone()
        } else {
            let item = PyTuple::new(py, pixel.iter().map(|&value| &values[value as usize]))?.into_any();
            cache[slot] = Some((*pixel, item.clone()));
            item
        };
        Ok(item)
    });
    pixel_sequence(py, items, tuple)
}

fn wide_pixel_data<'py>(py: Python<'py>, image: &Image, tuple: bool) -> PyResult<Bound<'py, PyAny>> {
    let data = image.raw_data()?;
    let channels = image.mode.channels();
    let bytes = if image.mode.is_float() { 4 } else { 2 };
    pixel_sequence(
        py,
        data.chunks_exact(channels * bytes).map(|pixel| {
            if image.mode.is_float() {
                Ok(PyTuple::new(py, pixel.as_chunks::<4>().0.iter().map(|bytes| float_value(bytes)))?.into_any())
            } else if channels == 1 {
                Ok(u16::from_le_bytes([pixel[0], pixel[1]]).into_pyobject(py)?.into_any())
            } else {
                Ok(PyTuple::new(py, pixel.as_chunks::<2>().0.iter().map(|&bytes| u16::from_le_bytes(bytes)))?.into_any())
            }
        }),
        tuple,
    )
}

fn bounding_box_l(pixels: &[u8], width: usize) -> Option<(usize, usize, usize, usize)> {
    let top = pixels.chunks_exact(width).position(|row| row.iter().any(|&v| v != 0))?;
    let bottom = pixels.chunks_exact(width).rposition(|row| row.iter().any(|&v| v != 0)).unwrap();
    let first = &pixels[top * width..(top + 1) * width];
    let mut left = first.iter().position(|&v| v != 0).unwrap();
    let mut right = first.iter().rposition(|&v| v != 0).unwrap() + 1;
    let remaining = &pixels[(top + 1) * width..(bottom + 1) * width];
    if left > 8 || width - right > 8 {
        // Wide empty margins are cheaper to scan contiguously by row.
        for row in remaining.chunks_exact(width) {
            if let Some(x) = row[..left].iter().position(|&v| v != 0) {
                left = x;
            }
            if let Some(x) = row[right..].iter().rposition(|&v| v != 0) {
                right += x + 1;
            }
            if left == 0 && right == width {
                break;
            }
        }
        return Some((left, top, right, bottom + 1));
    }
    // Search only columns outside the first occupied row's bounds. Separate
    // edges avoid revisiting an already complete right edge on every row.
    for x in 0..left {
        if remaining.chunks_exact(width).any(|row| row[x] != 0) {
            left = x;
            break;
        }
    }
    for x in (right..width).rev() {
        if remaining.chunks_exact(width).any(|row| row[x] != 0) {
            right = x + 1;
            break;
        }
    }
    Some((left, top, right, bottom + 1))
}

fn extrema<const C: usize>(data: &[u8]) -> Vec<(u16, u16)> {
    use std::simd::Simd;
    use std::simd::cmp::SimdOrd;
    use std::simd::num::SimdUint;

    use crate::pixels::{Bytes, load};

    let mut minimum = [255_u8; C];
    let mut maximum = [0_u8; C];
    for block in data.as_chunks::<C>().0.chunks(256) {
        // The scalar RGB loop already vectorizes efficiently.
        let done = if C == 3 {
            0
        } else {
            let bytes = block.as_flattened();
            let done = bytes.len() / 64 * 64;
            let mut low = Simd::<u8, 64>::splat(255);
            let mut high = Simd::<u8, 64>::splat(0);
            for values in bytes[..done].as_chunks::<64>().0 {
                let values = Simd::from_array(*values);
                low = low.simd_min(values);
                high = high.simd_max(values);
            }
            if C == 1 {
                minimum[0] = minimum[0].min(low.reduce_min());
                maximum[0] = maximum[0].max(high.reduce_max());
            } else {
                let low = low.to_array();
                let high = high.to_array();
                let mut low_channels = minimum.map(Bytes::splat);
                let mut high_channels = maximum.map(Bytes::splat);
                for i in (0..64).step_by(16 * C) {
                    let low = load::<C>(&low[i..]);
                    let high = load::<C>(&high[i..]);
                    for c in 0..C {
                        low_channels[c] = low_channels[c].simd_min(low[c]);
                        high_channels[c] = high_channels[c].simd_max(high[c]);
                    }
                }
                minimum = low_channels.map(SimdUint::reduce_min);
                maximum = high_channels.map(SimdUint::reduce_max);
            }
            done / C
        };
        for pixel in &block[done..] {
            for c in 0..C {
                minimum[c] = minimum[c].min(pixel[c]);
                maximum[c] = maximum[c].max(pixel[c]);
            }
        }
        if minimum == [0; C] && maximum == [255; C] {
            break;
        }
    }
    minimum.into_iter().zip(maximum).map(|(a, b)| (u16::from(a), u16::from(b))).collect()
}

fn count_colors<const N: usize>(data: &[u8], maxcolors: usize) -> Option<ColorCounts> {
    const CHUNK: usize = 128 << 10;
    if maxcolors >= 64 && crate::parallel::should_parallel(data.len() / N, CHUNK, CHUNK * 2) {
        use rayon::prelude::*;
        let tables: Option<Vec<_>> = data.par_chunks(CHUNK * N).map(|chunk| count_colors_serial::<N>(chunk, maxcolors)).collect();
        let mut merged = [(0, 0, 0); COLOR_TABLE_SIZE];
        let mut used = 0;
        for (chunk, table) in tables?.into_iter().enumerate() {
            for (key, count, index) in table.into_iter().filter(|entry| entry.1 != 0) {
                let mut slot = (key.wrapping_mul(COLOR_HASH_MULTIPLIER) >> (u64::BITS - COLOR_TABLE_BITS)) as usize;
                loop {
                    let entry = &mut merged[slot];
                    if entry.1 == 0 {
                        used += 1;
                        if used > maxcolors {
                            return None;
                        }
                        *entry = (key, count, chunk * CHUNK + index);
                        break;
                    }
                    if entry.0 == key {
                        entry.1 += count;
                        break;
                    }
                    slot = (slot + 1) & (COLOR_TABLE_SIZE - 1);
                }
            }
        }
        return Some(merged);
    }
    count_colors_serial::<N>(data, maxcolors)
}

fn count_colors_serial<const N: usize>(data: &[u8], maxcolors: usize) -> Option<ColorCounts> {
    let mut table = [(0_u64, 0_usize, 0_usize); COLOR_TABLE_SIZE];
    let mut used = 0;
    for (i, pixel) in data.as_chunks::<N>().0.iter().enumerate() {
        let mut bytes = [0; 8];
        bytes[..N].copy_from_slice(pixel);
        let key = u64::from_le_bytes(bytes);
        let mut slot = (key.wrapping_mul(COLOR_HASH_MULTIPLIER) >> (u64::BITS - COLOR_TABLE_BITS)) as usize;
        loop {
            let entry = &mut table[slot];
            if entry.1 == 0 {
                used += 1;
                if used > maxcolors {
                    return None;
                }
                *entry = (key, 1, i);
                break;
            }
            if entry.0 == key {
                entry.1 += 1;
                break;
            }
            slot = (slot + 1) & (COLOR_TABLE_SIZE - 1);
        }
    }
    Some(table)
}

fn put_sample_pixel(pixels: &mut [u8], value: &Bound<'_, PyAny>, channels: usize, maximum: u16, scale: f64, offset: f64) -> PyResult<()> {
    let mut samples = [0_u16; 4];
    if channels > 1 && value.len()? != channels {
        return Err(PyValueError::new_err("wrong number of color components"));
    }
    for (c, sample) in samples[..channels].iter_mut().enumerate() {
        let number = if channels == 1 { value.extract::<f64>()? } else { value.get_item(c)?.extract::<f64>()? };
        *sample = (number * scale + offset).clamp(0.0, f64::from(maximum)) as u16;
    }
    for (dst, sample) in pixels.as_chunks_mut::<2>().0.iter_mut().zip(samples) {
        *dst = sample.to_le_bytes();
    }
    Ok(())
}

fn put_pixels<const C: usize>(pixels: &mut [u8], data: &Bound<'_, PyAny>, length: usize, scale: f64, offset: f64) -> PyResult<()> {
    let mut output = pixels.as_chunks_mut::<C>().0.iter_mut();
    let mut write = |value: &Bound<'_, PyAny>| -> PyResult<()> {
        let pixel = if C == 1 {
            let number = if value.is_exact_instance_of::<PyInt>() {
                // SAFETY: Avoid a temporary Python float. The exact integer stays alive.
                let number = unsafe { pyo3::ffi::PyLong_AsDouble(value.as_ptr()) };
                if number == -1.0
                    && let Some(error) = PyErr::take(value.py())
                {
                    return Err(error);
                }
                number
            } else {
                value.extract::<f64>()?
            };
            [(number * scale + offset).clamp(0.0, 255.0) as u8, 0, 0, 0]
        } else {
            parse_pixel(value, C)?
        };
        *output.next().expect("validated data length") = pixel[..C].try_into().unwrap();
        Ok(())
    };
    if let Ok(list) = data.cast_exact::<PyList>() {
        for i in 0..length {
            write(&list.get_item(i)?)?;
        }
    } else if let Ok(tuple) = data.cast_exact::<PyTuple>() {
        for value in tuple.iter_borrowed() {
            write(&value)?;
        }
    } else {
        for i in 0..length {
            write(&data.get_item(i)?)?;
        }
    }
    Ok(())
}

pub use blanket_modes::PixelMode;

fn float_value(bytes: &[u8]) -> f32 {
    f32::from_le_bytes(bytes.try_into().unwrap())
}

pub fn integer_value(mode: PixelMode, bytes: &[u8]) -> i64 {
    match mode {
        PixelMode::I => i64::from(i32::from_le_bytes(bytes.try_into().unwrap())),
        PixelMode::I16B => i64::from(u16::from_be_bytes(bytes.try_into().unwrap())),
        PixelMode::I16N => i64::from(u16::from_ne_bytes(bytes.try_into().unwrap())),
        PixelMode::I16 | PixelMode::I16L => i64::from(u16::from_le_bytes(bytes.try_into().unwrap())),
        _ => unreachable!(),
    }
}

pub fn integer_bytes(mode: PixelMode, value: i64) -> [u8; 4] {
    let mut bytes = [0; 4];
    match mode {
        PixelMode::I => bytes.copy_from_slice(&(value as i32).to_le_bytes()),
        PixelMode::I16B => bytes[..2].copy_from_slice(&(value as u16).to_be_bytes()),
        PixelMode::I16N => bytes[..2].copy_from_slice(&(value as u16).to_ne_bytes()),
        PixelMode::I16 | PixelMode::I16L => bytes[..2].copy_from_slice(&(value as u16).to_le_bytes()),
        _ => unreachable!(),
    }
    bytes
}

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
struct SampleValidation(AtomicBool);

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
    pixels: Option<Arc<Vec<u8>>>,
    pub(crate) shared_pixels: Option<std::sync::Arc<dyn SharedPixels>> = None,
    pub(crate) bit_depth: u8 = 8,
    pub format: Option<String> = None,
    pub(crate) palette: Option<(PixelMode, Vec<u8>)> = None,
    sample_validation: SampleValidation = SampleValidation::new(),
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

    fn write_pixels(&mut self, py: Python<'_>) -> PyResult<&mut [u8]> {
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

    fn color_count_list<'py>(&self, py: Python<'py>, entries: impl IntoIterator<Item = (usize, usize)>) -> PyResult<Bound<'py, PyList>> {
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

#[pymethods]
impl Image {
    pub fn getextrema_float(&self, py: Python<'_>) -> PyResult<Vec<(f32, f32)>> {
        if !matches!(self.mode, PixelMode::RgbF | PixelMode::RgbaF) {
            return Err(PyValueError::new_err("float color image required"));
        }
        let data = self.raw_data()?;
        let channels = self.mode.channels();
        Ok(py.detach(|| {
            if data.is_empty() {
                return Vec::new();
            }
            let mut ranges = vec![(f32::NAN, f32::NAN); channels];
            for pixel in data.chunks_exact(channels * 4) {
                for (range, bytes) in ranges.iter_mut().zip(pixel.as_chunks::<4>().0) {
                    let value = float_value(bytes);
                    if range.0.is_nan() && range.1.is_nan() {
                        *range = (value, value);
                    } else {
                        if value < range.0 {
                            range.0 = value;
                        }
                        if value > range.1 {
                            range.1 = value;
                        }
                    }
                }
            }
            ranges
        }))
    }

    pub fn getextrema_scalar(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        if !self.mode.is_wide_scalar() || self.mode.channels() != 1 {
            return Err(PyValueError::new_err("scalar extrema require a wide scalar mode"));
        }
        let data = self.raw_data()?;
        if data.is_empty() {
            return Ok(None);
        }
        if self.mode == PixelMode::F {
            let (lo, hi) = py.detach(|| {
                let values = data.as_chunks::<4>().0.iter().map(|bytes| float_value(bytes));
                values.fold((f32::NAN, f32::NAN), |(lo, hi), v| {
                    if lo.is_nan() && hi.is_nan() {
                        return (v, v);
                    }
                    (if v < lo { v } else { lo }, if v > hi { v } else { hi })
                })
            });
            return Ok(Some((lo, hi).into_pyobject(py)?.into_any().unbind()));
        }
        let mode = self.mode;
        let (lo, hi) = py.detach(|| {
            data.chunks_exact(mode.sample_bytes())
                .map(|bytes| integer_value(mode, bytes))
                .fold((i64::MAX, i64::MIN), |(lo, hi), v| (lo.min(v), hi.max(v)))
        });
        Ok(Some((lo, hi).into_pyobject(py)?.into_any().unbind()))
    }

    pub fn getextrema(&self, py: Python<'_>) -> PyResult<Vec<(u16, u16)>> {
        if self.mode.is_integer() {
            return Err(PyValueError::new_err("integer extrema require getdata"));
        }
        let data = self.raw_data()?;
        let channels = self.mode.channels();
        Ok(py.detach(|| {
            if data.is_empty() {
                return Vec::new();
            }
            if self.bit_depth == 8 {
                // Once every band spans the entire domain, no later pixel
                // can change the answer. Check only once per block.
                return match self.mode {
                    PixelMode::One | PixelMode::L => extrema::<1>(data),
                    PixelMode::La | PixelMode::PremulLa | PixelMode::Pa => extrema::<2>(data),
                    PixelMode::Rgb | PixelMode::Hsv | PixelMode::YCbCr | PixelMode::Lab => extrema::<3>(data),
                    PixelMode::Rgba | PixelMode::PremulRgba | PixelMode::Rgbx | PixelMode::Cmyk => extrema::<4>(data),
                    _ => unreachable!(),
                };
            }
            let mut ranges = vec![(u16::MAX, 0); channels];
            for (i, bytes) in data.as_chunks::<2>().0.iter().enumerate() {
                let value = u16::from_le_bytes(*bytes);
                let range = &mut ranges[i % channels];
                range.0 = range.0.min(value);
                range.1 = range.1.max(value);
            }
            ranges
        }))
    }

    pub fn getchannel(&self, py: Python<'_>, channel: usize) -> PyResult<Self> {
        if self.mode.is_float() && self.mode.channels() > 1 {
            if channel >= self.mode.channels() {
                return Err(PyValueError::new_err("band index out of range"));
            }
            let data = self.raw_data()?;
            let pixels = py.detach(|| {
                data.chunks_exact(self.raw_bytes_per_pixel())
                    .map(|p| <[u8; 4]>::try_from(&p[channel * 4..channel * 4 + 4]).unwrap())
                    .collect::<Vec<_>>()
                    .into_flattened()
            });
            return Self::from_float_bytes(self.width, self.height, pixels);
        }
        if self.mode.is_wide_scalar() {
            return if channel == 0 { self.copy(py) } else { Err(PyValueError::new_err("band index out of range")) };
        }
        let data = self.raw_data()?;
        let channels = self.mode.channels();
        if channel >= channels {
            return Err(PyValueError::new_err("band index out of range"));
        }
        if matches!(self.mode, PixelMode::One | PixelMode::L) {
            let mut band = self.copy(py)?;
            band.mode = PixelMode::L;
            band.palette = None;
            return Ok(band);
        }
        py.detach(|| {
            if self.bit_depth == 8 {
                let pixels = match self.mode {
                    PixelMode::La | PixelMode::PremulLa | PixelMode::Pa => crate::pixels::extract_two_channel(data, channel),
                    PixelMode::Rgb | PixelMode::Hsv | PixelMode::YCbCr | PixelMode::Lab => crate::pixels::extract_channel::<3>(data, channel),
                    PixelMode::Rgba | PixelMode::PremulRgba | PixelMode::Rgbx | PixelMode::Cmyk => crate::pixels::extract_channel::<4>(data, channel),
                    _ => unreachable!(),
                };
                Self::from_pixels(self.width, self.height, PixelMode::L, pixels, None)
            } else {
                let pixels = crate::pixels::extract_sample_channel(data, channels, channel);
                Self::from_sample_bytes(self.width, self.height, PixelMode::L, pixels, self.bit_depth, None)
            }
        })
    }

    pub fn getdata(&self, py: Python<'_>) -> PyResult<Py<PyList>> {
        let data = self.raw_data()?;
        if self.bit_depth == 8 && self.mode.channels() > 1 {
            let result = match self.mode.channels() {
                2 => packed_pixel_data::<2>(py, data, false)?,
                3 => packed_pixel_data::<3>(py, data, false)?,
                4 => packed_pixel_data::<4>(py, data, false)?,
                _ => unreachable!(),
            };
            return Ok(result.cast_into::<PyList>()?.unbind());
        }
        if self.mode == PixelMode::F {
            return Ok(PyList::new(py, data.as_chunks::<4>().0.iter().map(|bytes| float_value(bytes)))?.unbind());
        }
        if self.mode.is_integer() {
            return Ok(PyList::new(py, data.chunks_exact(self.mode.sample_bytes()).map(|bytes| integer_value(self.mode, bytes)))?.unbind());
        }
        if self.bit_depth == 8 {
            return Ok(PyList::new(py, data.iter().copied())?.unbind());
        }
        Ok(wide_pixel_data(py, self, false)?.cast_into::<PyList>()?.unbind())
    }

    pub fn get_flattened_data(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        let data = self.raw_data()?;
        if self.bit_depth == 8 && self.mode.channels() > 1 {
            let result = match self.mode.channels() {
                2 => packed_pixel_data::<2>(py, data, true)?,
                3 => packed_pixel_data::<3>(py, data, true)?,
                4 => packed_pixel_data::<4>(py, data, true)?,
                _ => unreachable!(),
            };
            return Ok(result.cast_into::<PyTuple>()?.unbind());
        }
        if self.mode == PixelMode::F {
            return Ok(PyTuple::new(py, data.as_chunks::<4>().0.iter().map(|bytes| float_value(bytes)))?.unbind());
        }
        if self.mode.is_integer() {
            return Ok(PyTuple::new(py, data.chunks_exact(self.mode.sample_bytes()).map(|bytes| integer_value(self.mode, bytes)))?.unbind());
        }
        if self.bit_depth == 8 {
            return Ok(PyTuple::new(py, data.iter().copied())?.unbind());
        }
        Ok(wide_pixel_data(py, self, true)?.cast_into::<PyTuple>()?.unbind())
    }

    pub fn getbanddata<'py>(&self, py: Python<'py>, channel: usize, tuple: bool) -> PyResult<Bound<'py, PyAny>> {
        let data = self.raw_data()?;
        if channel >= self.mode.channels() {
            return Err(PyValueError::new_err("band index out of range"));
        }
        let stride = self.raw_bytes_per_pixel();
        let sample_bytes = stride / self.mode.channels();
        let items = data.chunks_exact(stride).map(|pixel| {
            let sample = &pixel[channel * sample_bytes..(channel + 1) * sample_bytes];
            if self.mode.is_float() {
                Ok(float_value(sample).into_pyobject(py)?.into_any())
            } else if self.mode.is_integer() {
                Ok(integer_value(self.mode, sample).into_pyobject(py)?.into_any())
            } else if sample_bytes == 2 {
                Ok(u16::from_le_bytes([sample[0], sample[1]]).into_pyobject(py)?.into_any())
            } else {
                Ok(sample[0].into_pyobject(py)?.into_any())
            }
        });
        pixel_sequence(py, items, tuple)
    }

    pub fn getcolors(&self, py: Python<'_>, maxcolors: usize) -> PyResult<Option<Py<PyList>>> {
        let data = self.raw_data()?;
        let stride = self.mode.channels()
            * if self.mode.is_wide_scalar() {
                self.mode.sample_bytes()
            } else if self.bit_depth == 8 {
                1
            } else {
                2
            };
        if stride == 1 {
            let counts = py.detach(|| {
                let mut counts = [0_usize; 256];
                for &value in data {
                    counts[value as usize] += 1;
                }
                counts
            });
            if counts.iter().filter(|&&n| n != 0).count() > maxcolors {
                return Ok(None);
            }
            return Ok(Some(PyList::new(py, counts.into_iter().enumerate().filter(|&(_, n)| n != 0).map(|(v, n)| (n, v)))?.unbind()));
        }
        if maxcolors <= 256 && stride <= 8 {
            // A bounded open-addressed table avoids hashing a byte slice and
            // allocating a map entry for each distinct small pixel value.
            let counts = py.detach(|| match stride {
                2 => count_colors::<2>(data, maxcolors),
                3 => count_colors::<3>(data, maxcolors),
                4 => count_colors::<4>(data, maxcolors),
                6 => count_colors::<6>(data, maxcolors),
                8 => count_colors::<8>(data, maxcolors),
                _ => return None,
            });
            let Some(counts) = counts else { return Ok(None) };
            let entries = counts.into_iter().filter(|entry| entry.1 != 0).map(|(_, count, i)| (count, i));
            return Ok(Some(self.color_count_list(py, entries)?.unbind()));
        }
        let counts = py.detach(|| {
            let mut counts = FxHashMap::<&[u8], (usize, usize)>::default();
            for (i, pixel) in data.chunks_exact(stride).enumerate() {
                let entry = counts.entry(pixel).or_insert((0, i));
                entry.0 += 1;
                if counts.len() > maxcolors {
                    return None;
                }
            }
            Some(counts)
        });
        let Some(counts) = counts else { return Ok(None) };
        Ok(Some(self.color_count_list(py, counts.into_values())?.unbind()))
    }

    pub fn putpixel(&mut self, xy: (i64, i64), value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.ensure_owned()?;
        self.raw_data()?;
        let x = if xy.0 < 0 { xy.0 + i64::from(self.width) } else { xy.0 };
        let y = if xy.1 < 0 { xy.1 + i64::from(self.height) } else { xy.1 };
        if x < 0 || y < 0 || x >= i64::from(self.width) || y >= i64::from(self.height) {
            return Err(PyIndexError::new_err("image index out of range"));
        }
        if self.mode.is_float() && self.mode.channels() > 1 {
            let values: Vec<f32> = value.extract()?;
            if values.len() != self.mode.channels() {
                return Err(PyValueError::new_err("wrong number of color components"));
            }
            let start = (y as usize * self.width as usize + x as usize) * self.raw_bytes_per_pixel();
            let pixels = self.write_pixels(value.py())?;
            for (sample, value) in pixels[start..start + values.len() * 4].as_chunks_mut::<4>().0.iter_mut().zip(values) {
                *sample = value.to_le_bytes();
            }
            return Ok(());
        }
        if self.mode == PixelMode::F {
            let start = (y as usize * self.width as usize + x as usize) * 4;
            let sample = value.extract::<f64>()? as f32;
            self.write_pixels(value.py())?[start..start + 4].copy_from_slice(&sample.to_le_bytes());
            return Ok(());
        }
        if self.mode.is_integer() {
            let bytes = integer_bytes(self.mode, value.extract::<i64>()?);
            let stride = self.mode.sample_bytes();
            let offset = (y as usize * self.width as usize + x as usize) * stride;
            self.write_pixels(value.py())?[offset..offset + stride].copy_from_slice(&bytes[..stride]);
            return Ok(());
        }
        if crate::samples::supported(self) {
            let channels = self.mode.channels();
            let maximum = crate::samples::maximum(self);
            let start = (y as usize * self.width as usize + x as usize) * channels * 2;
            return put_sample_pixel(&mut self.write_pixels(value.py())?[start..start + channels * 2], value, channels, maximum, 1.0, 0.0);
        }
        self.pixel_data()?;
        let channels = self.mode.channels();
        let pixel = if self.mode == PixelMode::One { [value.extract::<i64>()?.clamp(0, 255) as u8, 0, 0, 0] } else { parse_pixel(value, channels)? };
        let offset = (y as usize * self.width as usize + x as usize) * channels;
        self.write_pixels(value.py())?[offset..offset + channels].copy_from_slice(&pixel[..channels]);
        Ok(())
    }

    pub fn putdata(&mut self, data: &Bound<'_, PyAny>, scale: f64, offset: f64) -> PyResult<()> {
        self.ensure_owned()?;
        self.raw_data()?;
        let length = data.len()?;
        if length > self.width as usize * self.height as usize {
            return Err(PyTypeError::new_err("too many data entries"));
        }
        if length == 0 && !self.mode.is_float() && !self.mode.is_integer() {
            return Ok(());
        }
        if self.bit_depth == 8
            && self.mode.channels() == 1
            && scale == 1.0
            && offset == 0.0
            && let Ok(bytes) = data.cast_exact::<PyBytes>()
        {
            let source = bytes.as_bytes();
            let pixels = self.pixels.as_mut().unwrap();
            if length == pixels.len() && Arc::strong_count(pixels) > 1 {
                *pixels = Arc::new(data.py().detach(|| crate::buffers::copy(source))?);
            } else {
                let target = &mut self.write_pixels(data.py())?[..length];
                data.py().detach(|| target.copy_from_slice(source));
            }
            return Ok(());
        }
        if crate::samples::supported(self) && !self.mode.is_integer() {
            let channels = self.mode.channels();
            let maximum = crate::samples::maximum(self);
            let pixels = self.write_pixels(data.py())?;
            for i in 0..length {
                let start = i * channels * 2;
                put_sample_pixel(&mut pixels[start..start + channels * 2], &data.get_item(i)?, channels, maximum, scale, offset)?;
            }
            return Ok(());
        }
        if self.mode.is_float() && self.mode.channels() > 1 {
            let channels = self.mode.channels();
            let stride = channels * 4;
            let pixels = self.write_pixels(data.py())?;
            for (i, item) in data.try_iter()?.enumerate() {
                if i >= length {
                    return Err(PyTypeError::new_err("too many data entries"));
                }
                let item = item?;
                let mut values = [0.0; 4];
                let values = &mut values[..channels];
                if let Ok(tuple) = item.cast_exact::<PyTuple>()
                    && tuple.len() == channels
                {
                    for (c, value) in values.iter_mut().enumerate() {
                        *value = tuple.get_borrowed_item(c)?.extract::<f64>()?;
                    }
                } else {
                    let samples: Vec<f64> = item.extract()?;
                    if samples.len() != channels {
                        return Err(PyValueError::new_err("wrong number of color components"));
                    }
                    values.copy_from_slice(&samples);
                }
                for (&value, dst) in values.iter().zip(pixels[i * stride..(i + 1) * stride].as_chunks_mut::<4>().0) {
                    *dst = ((value * scale + offset) as f32).to_le_bytes();
                }
            }
            return Ok(());
        }
        if self.mode == PixelMode::F {
            let pixels = self.write_pixels(data.py())?;
            for (i, item) in data.try_iter()?.enumerate() {
                if i >= length {
                    return Err(PyTypeError::new_err("too many data entries"));
                }
                let value = (item?.extract::<f64>()? * scale + offset) as f32;
                pixels[i * 4..i * 4 + 4].copy_from_slice(&value.to_le_bytes());
            }
            return Ok(());
        }
        if self.mode.is_integer() {
            let mode = self.mode;
            let stride = mode.sample_bytes();
            let pixels = self.write_pixels(data.py())?;
            for (i, item) in data.try_iter()?.enumerate() {
                if i >= length {
                    return Err(PyTypeError::new_err("too many data entries"));
                }
                let value = (item?.extract::<f64>()? * scale + offset) as i64;
                let bytes = integer_bytes(mode, value);
                pixels[i * stride..(i + 1) * stride].copy_from_slice(&bytes[..stride]);
            }
            return Ok(());
        }
        self.pixel_data()?;
        let mode = self.mode;
        let pixels = self.write_pixels(data.py())?;
        match mode {
            PixelMode::One | PixelMode::L => put_pixels::<1>(pixels, data, length, scale, offset),
            PixelMode::La | PixelMode::PremulLa | PixelMode::Pa => put_pixels::<2>(pixels, data, length, scale, offset),
            PixelMode::Rgb | PixelMode::Hsv | PixelMode::YCbCr | PixelMode::Lab => put_pixels::<3>(pixels, data, length, scale, offset),
            PixelMode::Rgba | PixelMode::PremulRgba | PixelMode::Rgbx | PixelMode::Cmyk => put_pixels::<4>(pixels, data, length, scale, offset),
            _ => unreachable!(),
        }
    }

    pub fn point_float(&self, function: &Bound<'_, PyAny>) -> PyResult<Self> {
        if self.mode != PixelMode::F {
            return Err(PyValueError::new_err("point operation requires F mode"));
        }
        let source = self.raw_data()?;
        let mut pixels = crate::buffers::reserved(source.len())?;
        for bytes in source.as_chunks::<4>().0 {
            let value = function.call1((float_value(bytes),))?.extract::<f64>()? as f32;
            pixels.extend_from_slice(&value.to_le_bytes());
        }
        Self::from_float_bytes(self.width, self.height, pixels)
    }

    pub fn getbbox(&self, py: Python<'_>, alpha_only: bool) -> PyResult<Option<(u32, u32, u32, u32)>> {
        let pixels = self.raw_data()?;
        let channels = self.mode.channels();
        let sample_bytes = if self.mode.is_wide_scalar() {
            self.mode.sample_bytes()
        } else if self.bit_depth == 8 {
            1
        } else {
            2
        };
        Ok(py.detach(|| {
            if self.width == 0 || self.height == 0 {
                return None;
            }
            let stride = channels * sample_bytes;
            if stride == 1 {
                return bounding_box_l(pixels, self.width as usize)
                    .map(|(left, top, right, bottom)| (left as u32, top as u32, right as u32, bottom as u32));
            }
            let row_bytes = self.width as usize * stride;
            let occupied = |pixel: &[u8]| {
                let values = if alpha_only
                    && matches!(
                        self.mode,
                        PixelMode::Rgba | PixelMode::PremulRgba | PixelMode::RgbaF | PixelMode::La | PixelMode::PremulLa | PixelMode::Pa
                    ) {
                    &pixel[(channels - 1) * sample_bytes..]
                } else {
                    pixel
                };
                if matches!(self.mode, PixelMode::RgbF | PixelMode::RgbaF) {
                    values.as_chunks::<4>().0.iter().any(|bytes| float_value(bytes) != 0.0)
                } else {
                    values.iter().any(|&v| v != 0)
                }
            };
            let mut rows = pixels.chunks_exact(row_bytes);
            let top = rows.position(|row| row.chunks_exact(stride).any(occupied))?;
            let bottom = pixels.chunks_exact(row_bytes).rposition(|row| row.chunks_exact(stride).any(occupied)).unwrap();
            let mut left = self.width as usize;
            let mut right = 0;
            for row in pixels[top * row_bytes..(bottom + 1) * row_bytes].chunks_exact(row_bytes) {
                if let Some(x) = row[..left * stride].chunks_exact(stride).position(occupied) {
                    left = x;
                }
                if let Some(x) = row[right * stride..].chunks_exact(stride).rposition(occupied) {
                    right += x + 1;
                }
                if left == 0 && right == self.width as usize {
                    break;
                }
            }
            Some((left as u32, top as u32, right as u32, bottom as u32 + 1))
        }))
    }

    #[getter]
    pub fn bit_depth(&self) -> u8 {
        self.bit_depth
    }

    #[getter]
    pub fn mode(&self) -> &'static str {
        if self.mode == PixelMode::Pa {
            "PA"
        } else if self.palette.is_some() {
            "P"
        } else {
            self.mode.as_str()
        }
    }

    #[getter]
    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    #[getter]
    pub fn width(&self) -> u32 {
        self.width
    }

    #[getter]
    pub fn height(&self) -> u32 {
        self.height
    }

    #[getter]
    pub fn format(&self) -> Option<&str> {
        self.format.as_deref()
    }

    #[getter]
    pub fn info(&self, py: Python<'_>) -> Py<PyDict> {
        PyDict::new(py).unbind()
    }

    pub fn load(&self) -> PyResult<()> {
        self.raw_data().map(|_| ())
    }

    #[staticmethod]
    fn _mapped_layout(mode: &str, size: (u32, u32)) -> PyResult<Self> {
        // The Python mapping supplies pixels separately.
        let palette = (mode == "P").then(|| (PixelMode::Rgb, Vec::new()));
        let mode = if mode == "P" { PixelMode::L } else { PixelMode::parse(mode)? };
        let bit_depth = (mode.sample_bytes() * 8) as u8;
        expected_len(size.0, size.1, mode)?
            .checked_mul(mode.sample_bytes())
            .ok_or_else(|| PyValueError::new_err("image dimensions are too large"))?;
        Ok(Self { width: size.0, height: size.1, mode, pixels: None, bit_depth, palette, .. })
    }

    fn _write_mapped_rows(&self, data: &Bound<'_, PyMemoryView>, mode: &str, stride: usize, orientation: i8) -> PyResult<()> {
        let source = self.raw_data()?;
        if self.mode.as_str() != mode && !(self.mode == PixelMode::L && mode == "P") {
            return Err(PyValueError::new_err("mapped buffer mode does not match"));
        }
        let row_bytes = self.width as usize * self.raw_bytes_per_pixel();
        let height = self.height as usize;
        let required = stride.checked_mul(height).ok_or_else(|| PyValueError::new_err("image dimensions are too large"))?;
        let buffer = PyBuffer::<u8>::get(data.as_any())?;
        if stride < row_bytes
            || !matches!(orientation, -1 | 1)
            || !buffer.is_c_contiguous()
            || buffer.dimensions() != 1
            || buffer.len_bytes() != required
        {
            return Err(PyValueError::new_err("invalid mapped buffer layout"));
        }
        if source.is_empty() {
            return Ok(());
        }
        let py = data.py();
        if stride == row_bytes && orientation == 1 {
            return buffer.copy_from_slice(py, source);
        }
        for (row, source) in source.chunks_exact(row_bytes).enumerate() {
            let row = if orientation == 1 { row } else { height - 1 - row };
            let offset = row * stride;
            let view = data.get_item(PySlice::new(py, offset as isize, (offset + row_bytes) as isize, 1))?;
            PyBuffer::<u8>::get(&view)?.copy_from_slice(py, source)?;
        }
        Ok(())
    }

    fn _replace_pixels(&mut self, mut replacement: PyRefMut<'_, Self>) -> PyResult<()> {
        if self.size() != replacement.size() || self.mode != replacement.mode || self.bit_depth != replacement.bit_depth {
            return Err(PyValueError::new_err("replacement image layout does not match"));
        }
        self.raw_data()?;
        replacement.raw_data()?;
        std::mem::swap(&mut self.pixels, &mut replacement.pixels);
        std::mem::swap(&mut self.shared_pixels, &mut replacement.shared_pixels);
        std::mem::swap(&mut self.sample_validation, &mut replacement.sample_validation);
        Ok(())
    }

    fn _copy_palette(&mut self, source: &Self) {
        self.copy_palette_from(source);
    }

    fn _bilevel_array_bytes(&self, py: Python<'_>) -> PyResult<Py<PyBytes>> {
        let source = self.raw_data()?;
        PyBytes::new_with(py, source.len(), |output| {
            py.detach(|| crate::pixels::bilevel_to_l_into(source, output));
            Ok(())
        })
        .map(Bound::unbind)
    }

    fn _writable_samples<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyByteArray>> {
        Ok(PyByteArray::new(py, self.raw_data()?))
    }

    pub fn pixels_equal(&self, py: Python<'_>, other: &Self) -> PyResult<bool> {
        let first = self.raw_data()?;
        let second = other.raw_data()?;
        if self.mode != other.mode || self.size() != other.size() || self.bit_depth != other.bit_depth {
            return Ok(false);
        }
        if std::ptr::eq(first, second) {
            return Ok(true);
        }
        Ok(py.detach(|| if self.mode == PixelMode::One { first.iter().zip(second).all(|(&a, &b)| (a != 0) == (b != 0)) } else { first == second }))
    }

    pub fn copy(&self, py: Python<'_>) -> PyResult<Self> {
        let source = self.raw_data()?;
        let pixels = if let Some(pixels) = &self.pixels {
            py.detach(|| Arc::clone(pixels))
        } else {
            let mut pixels = crate::buffers::reserved::<u8>(source.len())?;
            let chunk = 256 * 1024;
            py.detach(|| {
                if source.len() < 4 << 20 {
                    pixels.extend_from_slice(source);
                    return;
                }
                crate::parallel::chunks_mut_above(&mut pixels.spare_capacity_mut()[..source.len()], chunk, 4 << 20, |i, dst| {
                    dst.write_copy_of_slice(&source[i * chunk..i * chunk + dst.len()]);
                });
            });
            // SAFETY: Every byte is initialized above, including any partial final partition.
            unsafe { pixels.set_len(source.len()) };
            Arc::new(pixels)
        };
        Ok(Self {
            width: self.width,
            height: self.height,
            mode: self.mode,
            pixels: Some(pixels),
            bit_depth: self.bit_depth,
            palette: self.palette.as_ref().map(|(mode, data)| crate::buffers::copy(data).map(|data| (*mode, data))).transpose()?,
            ..
        })
    }

    pub fn getpixel(&self, py: Python<'_>, xy: (i64, i64)) -> PyResult<Py<PyAny>> {
        let pixels = self.raw_data()?;
        let x = if xy.0 < 0 { xy.0 + i64::from(self.width) } else { xy.0 };
        let y = if xy.1 < 0 { xy.1 + i64::from(self.height) } else { xy.1 };
        if x < 0 || y < 0 || x >= i64::from(self.width) || y >= i64::from(self.height) {
            return Err(PyIndexError::new_err("image index out of range"));
        }
        let channels = self.mode.channels();
        let offset = (y as usize * self.width as usize + x as usize) * channels;
        if self.mode.is_float() && channels > 1 {
            return Ok(PyTuple::new(py, (0..channels).map(|c| float_value(&pixels[(offset + c) * 4..(offset + c + 1) * 4])))?.into_any().unbind());
        }
        if self.mode == PixelMode::F {
            return Ok(float_value(&pixels[offset * 4..offset * 4 + 4]).into_pyobject(py)?.into_any().unbind());
        }
        if self.mode.is_integer() {
            let start = offset * self.mode.sample_bytes();
            return Ok(integer_value(self.mode, &pixels[start..start + self.mode.sample_bytes()]).into_pyobject(py)?.into_any().unbind());
        }
        if self.bit_depth > 8 {
            let sample = |i: usize| u16::from_le_bytes([pixels[2 * i], pixels[2 * i + 1]]);
            return Ok(match self.mode {
                PixelMode::One | PixelMode::L => sample(offset).into_pyobject(py)?.into_any().unbind(),
                PixelMode::La | PixelMode::PremulLa | PixelMode::Pa => (sample(offset), sample(offset + 1)).into_pyobject(py)?.into_any().unbind(),
                PixelMode::Rgb | PixelMode::Hsv | PixelMode::YCbCr | PixelMode::Lab => {
                    (sample(offset), sample(offset + 1), sample(offset + 2)).into_pyobject(py)?.into_any().unbind()
                }
                PixelMode::Rgba | PixelMode::PremulRgba | PixelMode::Rgbx | PixelMode::Cmyk => {
                    (sample(offset), sample(offset + 1), sample(offset + 2), sample(offset + 3)).into_pyobject(py)?.into_any().unbind()
                }
                _ => unreachable!(),
            });
        }
        Ok(match self.mode {
            PixelMode::One | PixelMode::L => pixels[offset].into_pyobject(py)?.into_any().unbind(),
            PixelMode::La | PixelMode::PremulLa | PixelMode::Pa => (pixels[offset], pixels[offset + 1]).into_pyobject(py)?.into_any().unbind(),
            PixelMode::Rgb | PixelMode::Hsv | PixelMode::YCbCr | PixelMode::Lab => {
                (pixels[offset], pixels[offset + 1], pixels[offset + 2]).into_pyobject(py)?.into_any().unbind()
            }
            PixelMode::Rgba | PixelMode::PremulRgba | PixelMode::Rgbx | PixelMode::Cmyk => {
                (pixels[offset], pixels[offset + 1], pixels[offset + 2], pixels[offset + 3]).into_pyobject(py)?.into_any().unbind()
            }
            _ => unreachable!(),
        })
    }

    pub fn palette_data<'py>(&self, py: Python<'py>) -> Option<(&'static str, Bound<'py, PyBytes>)> {
        self.palette.as_ref().map(|(mode, data)| (mode.as_str(), PyBytes::new(py, data)))
    }

    pub fn set_palette(&mut self, mode: &str, data: std::borrow::Cow<'_, [u8]>) -> PyResult<()> {
        self.pixel_data()?;
        let mode = PixelMode::parse(mode)?;
        if !matches!(self.mode, PixelMode::L | PixelMode::Pa)
            || !matches!(mode, PixelMode::Rgb | PixelMode::Rgba)
            || data.len() > 256 * mode.channels()
            || !data.len().is_multiple_of(mode.channels())
        {
            return Err(PyValueError::new_err("invalid palette"));
        }
        if let Some((old_mode, old_data)) = &mut self.palette {
            *old_mode = mode;
            if old_data.as_slice() != data.as_ref() {
                old_data.clear();
                old_data.extend_from_slice(&data);
            }
        } else {
            self.palette = Some((mode, data.into_owned()));
        }
        Ok(())
    }

    #[pyo3(signature = (alpha=None, index=None))]
    pub fn set_palette_alpha(&mut self, alpha: Option<std::borrow::Cow<'_, [u8]>>, index: Option<u8>) -> PyResult<()> {
        self.pixel_data()?;
        let output = self.alpha_palette(alpha.as_deref(), index)?;
        self.palette = Some((PixelMode::Rgba, output));
        Ok(())
    }

    pub fn close(&mut self) {
        self.pixels = None;
        self.shared_pixels = None;
    }

    pub fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    pub fn __exit__(&mut self, _exc_type: &Bound<'_, PyAny>, _exc_value: &Bound<'_, PyAny>, _traceback: &Bound<'_, PyAny>) {
        self.close();
    }

    pub fn __repr__(&self) -> String {
        format!("<blanket.Image.Image image mode={} size={}x{}>", self.mode.as_str(), self.width, self.height)
    }
}

fn validate_raw_data(width: u32, height: u32, mode: PixelMode, bit_depth: u8, pixels: &[u8]) -> PyResult<()> {
    let sample_bytes = if mode.is_wide_scalar() {
        if bit_depth as usize != mode.sample_bytes() * 8 {
            return Err(PyValueError::new_err("invalid sample bit depth"));
        }
        mode.sample_bytes()
    } else if bit_depth == 8 {
        1
    } else if matches!(mode, PixelMode::L | PixelMode::Rgb | PixelMode::Rgba) && matches!(bit_depth, 10 | 12 | 16) {
        2
    } else {
        return Err(PyValueError::new_err("invalid sample bit depth"));
    };
    let expected =
        expected_len(width, height, mode)?.checked_mul(sample_bytes).ok_or_else(|| PyValueError::new_err("image dimensions are too large"))?;
    if pixels.len() != expected || matches!(bit_depth, 10 | 12) && !crate::pixels::samples_in_range(pixels, bit_depth) {
        return Err(PyValueError::new_err("invalid image data"));
    }
    Ok(())
}

fn expected_len(width: u32, height: u32, mode: PixelMode) -> PyResult<usize> {
    let pixels = (width as usize).checked_mul(height as usize).ok_or_else(|| PyValueError::new_err("image dimensions are too large"))?;
    pixels.checked_mul(mode.channels()).ok_or_else(|| PyValueError::new_err("image dimensions are too large"))
}

pub fn lab_raw_bytes(source: &[u8]) -> Vec<u8> {
    let mut output = source.to_vec();
    lab_raw_bytes_mut(&mut output);
    output
}

pub fn lab_raw_bytes_mut(pixels: &mut [u8]) {
    const MASK: [u8; 48] = {
        let mut mask = [128; 48];
        let mut i = 0;
        while i < 48 {
            mask[i] = 0;
            i += 3;
        }
        mask
    };
    let (blocks, tail) = pixels.as_chunks_mut::<48>();
    for block in blocks {
        *block = std::array::from_fn(|i| block[i] ^ MASK[i]);
    }
    for (value, mask) in tail.iter_mut().zip(MASK) {
        *value ^= mask;
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn mutable_high_depth_samples_are_revalidated_and_cloned_independently() {
        use super::{Image, PixelMode};

        for mode in [PixelMode::L, PixelMode::Rgb, PixelMode::Rgba] {
            for depth in [10, 12] {
                let mut image = Image::from_samples(1, 1, mode, vec![0; mode.channels()], depth, None).unwrap();
                image.pixels_mut().unwrap()[..2].copy_from_slice(&u16::MAX.to_le_bytes());
                let mut cloned = image.clone();
                assert!(image.raw_data().is_err());
                assert!(cloned.raw_data().is_err());
                cloned.pixels_mut().unwrap()[..2].copy_from_slice(&1_u16.to_le_bytes());
                assert!(cloned.raw_data().is_ok());
                assert!(image.raw_data().is_err());
                image.pixels_mut().unwrap()[..2].copy_from_slice(&0_u16.to_le_bytes());
                assert!(image.raw_data().is_ok());
                image.pixels_mut().unwrap()[..2].copy_from_slice(&(1_u16 << depth).to_le_bytes());
                assert!(image.raw_data().is_err());
            }
        }
    }

    struct Storage(Vec<u8>);

    impl super::SharedPixels for Storage {
        fn bytes(&self) -> &[u8] {
            &self.0
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    #[test]
    fn shared_storage_validates_layout_without_copying() {
        Python::initialize();
        for (mode, depth) in [
            (PixelMode::One, 8),
            (PixelMode::L, 8),
            (PixelMode::Rgb, 10),
            (PixelMode::Rgb, 12),
            (PixelMode::Rgba, 16),
            (PixelMode::I, 32),
            (PixelMode::I16, 16),
            (PixelMode::I16B, 16),
            (PixelMode::F, 32),
            (PixelMode::RgbF, 32),
            (PixelMode::RgbaF, 32),
        ] {
            let length = 6 * mode.channels() * usize::from(depth).div_ceil(8);
            let storage = std::sync::Arc::new(Storage(vec![0; length]));
            let pointer = storage.0.as_ptr();
            let image = Image::from_shared(3, 2, mode, depth, storage).unwrap();
            assert_eq!(image.raw_data().unwrap().as_ptr(), pointer);
            assert_eq!(image.raw_bytes_per_pixel(), length / 6);
            assert_eq!(image.into_pixels().unwrap(), vec![0; length]);
            for invalid in [length - 1, length + 1] {
                assert!(Image::from_shared(3, 2, mode, depth, std::sync::Arc::new(Storage(vec![0; invalid]))).is_err());
            }
        }
        for (mode, depth) in [(PixelMode::Rgb, 32), (PixelMode::F, 8), (PixelMode::I, 16), (PixelMode::La, 16), (PixelMode::L, 9)] {
            assert!(Image::from_shared(1, 1, mode, depth, std::sync::Arc::new(Storage(vec![0; 4]))).is_err());
            assert!(Image::from_raw_bytes(1, 1, mode, vec![0; 4], depth).is_err());
        }
        for depth in [10, 12] {
            let data = (1_u16 << depth).to_le_bytes().to_vec();
            assert!(Image::from_shared(1, 1, PixelMode::L, depth, std::sync::Arc::new(Storage(data))).is_err());
        }
        assert!(Image::from_shared(u32::MAX, u32::MAX, PixelMode::RgbaF, 32, std::sync::Arc::new(Storage(Vec::new()))).is_err());
    }

    #[test]
    fn storage_replacement_and_reinterpretation_preserve_invariants() {
        Python::initialize();
        let mut image = Image::from_pixels(2, 1, PixelMode::L, vec![1, 2], Some("PNG".into())).unwrap();
        let replacement = Image::from_pixels(1, 2, PixelMode::L, vec![3, 4], None).unwrap();
        assert!(image.replace_storage(replacement).is_err());
        assert_eq!(image.raw_data().unwrap(), [1, 2]);
        assert!(image.reinterpret_mode(PixelMode::Rgb).is_err());
        assert_eq!(image.pixel_mode(), PixelMode::L);
        let replacement = Image::from_raw_bytes(2, 1, PixelMode::L, vec![0, 1, 0, 2], 16).unwrap();
        image.replace_storage(replacement).unwrap();
        image.reinterpret_mode(PixelMode::I16).unwrap();
        assert_eq!(image.bit_depth(), 16);
        assert_eq!(image.format(), Some("PNG"));
        assert_eq!(image.raw_data().unwrap(), [0, 1, 0, 2]);
    }

    #[test]
    fn changing_shared_storage_cannot_bypass_validation_on_detach() {
        struct ChangingStorage(std::sync::atomic::AtomicBool);

        impl SharedPixels for ChangingStorage {
            fn bytes(&self) -> &[u8] {
                if self.0.load(std::sync::atomic::Ordering::Relaxed) { &[1] } else { &[1, 2] }
            }

            fn as_any(&self) -> &dyn std::any::Any {
                self
            }
        }

        Python::initialize();
        let storage = std::sync::Arc::new(ChangingStorage(std::sync::atomic::AtomicBool::new(false)));
        let shared = std::sync::Arc::clone(&storage);
        let mut image = Image::from_shared(2, 1, PixelMode::L, 8, shared).unwrap();
        storage.0.store(true, std::sync::atomic::Ordering::Relaxed);
        assert!(image.raw_data().is_err());
        assert!(image.ensure_owned().is_err());
        assert!(image.pixels_mut().is_none());
        storage.0.store(false, std::sync::atomic::Ordering::Relaxed);
        image.ensure_owned().unwrap();
        assert_eq!(image.raw_data().unwrap(), [1, 2]);
    }

    #[test]
    fn shared_pixels_detach_on_write_and_close_independently() {
        let mut image = Image::from_shared(3, 1, PixelMode::L, 8, std::sync::Arc::new(Storage(vec![1, 2, 3]))).unwrap();
        let mut sibling = image.clone();
        assert_eq!(image.raw_data().unwrap(), [1, 2, 3]);
        image.ensure_owned().unwrap();
        image.pixels_mut().unwrap()[0] = 9;
        assert_eq!(image.raw_data().unwrap(), [9, 2, 3]);
        assert_eq!(sibling.raw_data().unwrap(), [1, 2, 3]);
        assert!(image.shared_pixels.is_none());
        sibling.close();
        assert!(sibling.shared_pixels.is_none());
        assert_eq!(image.raw_data().unwrap(), [9, 2, 3]);
    }

    #[test]
    fn owned_raster_copy_shares_storage_until_mutated() {
        Python::initialize();
        let mut image = Image::from_pixels(3, 1, PixelMode::L, vec![1, 2, 3], Some("PNG".into())).unwrap();
        let mut sibling = Python::attach(|py| image.copy(py)).unwrap();
        assert!(std::ptr::eq(image.raw_data().unwrap(), sibling.raw_data().unwrap()));
        assert_eq!(sibling.format(), None);
        image.pixels_mut().unwrap()[0] = 9;
        assert_eq!(image.raw_data().unwrap(), [9, 2, 3]);
        assert_eq!(sibling.raw_data().unwrap(), [1, 2, 3]);
        sibling.close();
        assert_eq!(image.into_pixels().unwrap(), [9, 2, 3]);
    }

    #[test]
    fn zeroed_pixel_allocations_support_vec_mutation_and_growth() {
        for len in [0, 1, 15, 16, 17, 4095, 4096, 4097, 1024 * 1024] {
            let mut pixels = super::alloc_pixels(len).unwrap();
            assert_eq!(pixels.len(), len);
            assert!(pixels.iter().all(|&value| value == 0));
            pixels.fill(173);
            pixels.push(91);
            assert_eq!(pixels[len], 91);
            assert!(pixels[..len].iter().all(|&value| value == 173));
        }
        assert!(super::alloc_pixels(usize::MAX).is_err());
    }

    use super::*;

    #[test]
    fn lab_conversion_bounds_capacity_and_reuses_owned_storage() {
        for count in [0, 1, 15, 16, 17, 1048577] {
            let source: Vec<u8> = (0..count * 3).map(|i| (i * 71) as u8).collect();
            let output = lab_raw_bytes(&source);
            assert_eq!(output.len(), source.len());
            assert!(output.capacity() <= source.len() + 47);
            let mut owned = output;
            let pointer = owned.as_ptr();
            lab_raw_bytes_mut(&mut owned);
            assert_eq!(owned, source);
            assert_eq!(owned.as_ptr(), pointer);
        }
    }

    #[test]
    fn bilevel_rows_preserve_partial_bytes_and_parallel_partitions() {
        for threads in [1, 4] {
            rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap().install(|| {
                for width in [0_usize, 1, 7, 8, 9, 15, 16, 17, 63, 64, 65, 127, 128, 129, 255, 256, 257, 800, 1023, 4097] {
                    for height in [0, 1, 3, 129, 1025] {
                        let row_bytes = width.div_ceil(8);
                        let packed: Vec<_> = (0..row_bytes * height).map(|i| (i * 37 + 0x95) as u8).collect();
                        let pixels = unpack_bilevel(&packed, width, height);
                        let inverted = unpack_bilevel_inverted(&packed, width, height);
                        assert_eq!(pixels.len(), width * height);
                        for y in 0..height {
                            for x in 0..width {
                                let value = if packed[y * row_bytes + x / 8] & (128 >> (x % 8)) != 0 { 255 } else { 0 };
                                assert_eq!(pixels[y * width + x], value);
                                assert_eq!(inverted[y * width + x], !value);
                            }
                        }
                        let mut expected = packed;
                        if width % 8 != 0 {
                            for row in expected.chunks_mut(row_bytes) {
                                row[row_bytes - 1] &= 0xff << (8 - width % 8);
                            }
                        }
                        assert_eq!(pack_bilevel(&pixels, width, height), expected);
                    }
                }
            });
        }
    }

    #[test]
    fn extrema_matches_scalar_for_unaligned_buffers_and_tails() {
        fn check<const C: usize>() {
            for offset in 0..16 {
                for count in [0, 1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 521] {
                    for fill in [None, Some(0), Some(255)] {
                        let mut data: Vec<u8> = (0..offset + count * C).map(|i| fill.unwrap_or(((i * 37 + i / 11) % 192 + 32) as u8)).collect();
                        if fill.is_none() && count >= 2 {
                            for c in 0..C {
                                data[offset + (count - 1) * C + c] = c as u8;
                                data[offset + (count - 2) * C + c] = 255 - c as u8;
                            }
                        }
                        let data = &data[offset..];
                        let expected: Vec<_> = (0..C)
                            .map(|c| {
                                data.as_chunks::<C>()
                                    .0
                                    .iter()
                                    .fold((255_u16, 0_u16), |(low, high), p| (low.min(u16::from(p[c])), high.max(u16::from(p[c]))))
                            })
                            .collect();
                        assert_eq!(extrema::<C>(data), expected, "C={C}, offset={offset}, count={count}");
                    }
                }
            }
        }
        check::<1>();
        check::<2>();
        check::<3>();
        check::<4>();
    }

    #[test]
    pub(super) fn validates_buffer_length() {
        Python::initialize();
        Python::attach(|_| {
            let error = Image::from_pixels(2, 2, PixelMode::Rgb, vec![0; 11], None).expect_err("invalid buffer should fail");
            assert!(error.to_string().contains("expected 12 bytes"));
        });
    }
}
