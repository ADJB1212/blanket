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

fn pack_bilevel_into(pixels: &[u8], width: usize, packed: &mut [u8]) {
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

pub(crate) fn integer_value(mode: PixelMode, bytes: &[u8]) -> i64 {
    match mode {
        PixelMode::I => i64::from(i32::from_le_bytes(bytes.try_into().unwrap())),
        PixelMode::I16B => i64::from(u16::from_be_bytes(bytes.try_into().unwrap())),
        PixelMode::I16N => i64::from(u16::from_ne_bytes(bytes.try_into().unwrap())),
        PixelMode::I16 | PixelMode::I16L => i64::from(u16::from_le_bytes(bytes.try_into().unwrap())),
        _ => unreachable!(),
    }
}

pub(crate) fn integer_bytes(mode: PixelMode, value: i64) -> [u8; 4] {
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
            || bit_depth != 16 && !crate::simd::samples_in_range(&pixels, bit_depth)
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

    fn alpha_palette(&self, alpha: Option<&[u8]>, index: Option<u8>) -> PyResult<Vec<u8>> {
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
                    PixelMode::La | PixelMode::PremulLa | PixelMode::Pa => crate::simd::extract_two_channel(data, channel),
                    PixelMode::Rgb | PixelMode::Hsv | PixelMode::YCbCr | PixelMode::Lab => crate::simd::extract_channel::<3>(data, channel),
                    PixelMode::Rgba | PixelMode::PremulRgba | PixelMode::Rgbx | PixelMode::Cmyk => crate::simd::extract_channel::<4>(data, channel),
                    _ => unreachable!(),
                };
                Self::from_pixels(self.width, self.height, PixelMode::L, pixels, None)
            } else {
                let pixels = crate::simd::extract_sample_channel(data, channels, channel);
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
            py.detach(|| crate::simd::bilevel_to_l_into(source, output));
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

    #[pyo3(signature = (mode, alpha=None, index=None, bit_depth=None))]
    pub fn convert_palette_alpha(
        &self, py: Python<'_>, mode: &str, alpha: Option<std::borrow::Cow<'_, [u8]>>, index: Option<u8>, bit_depth: Option<u8>,
    ) -> PyResult<Self> {
        if self.mode != PixelMode::L || !matches!(mode, "LA" | "RGBA") {
            return Err(PyValueError::new_err("indexed alpha conversion requires LA or RGBA output"));
        }
        let indices = self.pixel_data()?;
        let palette = self.alpha_palette(alpha.as_deref(), index)?;
        let destination = PixelMode::parse(mode)?;
        let pixels = py.detach(|| {
            if destination == PixelMode::La {
                let table = convert_pixels(&palette, PixelMode::Rgba, PixelMode::La);
                expand_palette::<2>(indices, &table)
            } else {
                expand_palette::<4>(indices, &palette)
            }
        })?;
        let result = Self::from_pixels(self.width, self.height, destination, pixels, None)?;
        if bit_depth.is_some_and(|depth| depth != 8) { result.convert(py, mode, bit_depth) } else { Ok(result) }
    }

    pub fn close(&mut self) {
        self.pixels = None;
        self.shared_pixels = None;
    }

    pub fn convert_matrix(&self, py: Python<'_>, mode: &str, matrix: Vec<f32>) -> PyResult<Self> {
        if !matches!(mode, "L" | "RGB") {
            return Err(PyValueError::new_err("illegal conversion"));
        }
        if !matches!(matrix.len(), 4 | 12) {
            return Err(PyTypeError::new_err("matrix must be tuple of length 4 or 12"));
        }
        if self.mode != PixelMode::Rgb || self.bit_depth != 8 {
            return Err(PyValueError::new_err("image has wrong mode"));
        }
        let destination = PixelMode::parse(mode)?;
        let channels = destination.channels();
        if matrix.len() < channels * 4 {
            return Err(PyTypeError::new_err("matrix must be tuple of length 12"));
        }
        let source = self.pixel_data()?;
        let length = source.len() / 3 * channels;
        let mut pixels = crate::buffers::reserved(length)?;
        py.detach(|| {
            crate::parallel::chunks_mut_above(&mut pixels.spare_capacity_mut()[..length], 16 * 1024 * channels, 256 * 1024, |chunk, dst| {
                let start = chunk * 16 * 1024 * 3;
                for (pixel, output) in source[start..].as_chunks::<3>().0.iter().zip(dst.chunks_exact_mut(channels)) {
                    for (channel, value) in output.iter_mut().enumerate() {
                        let m = &matrix[channel * 4..channel * 4 + 4];
                        value.write(
                            (f64::from(m[0] * f32::from(pixel[0]) + m[1] * f32::from(pixel[1]) + m[2] * f32::from(pixel[2]) + m[3]) + 0.5)
                                .clamp(0.0, 255.0) as u8,
                        );
                    }
                }
            });
            // SAFETY: Every reserved byte is initialized by its pixel's matrix row.
            unsafe { pixels.set_len(length) };
        });
        Self::from_pixels(self.width, self.height, destination, pixels, None)
    }

    pub fn convert_transparent_alpha(&self, py: Python<'_>, mode: &str, key: Vec<u8>) -> PyResult<Self> {
        if !matches!(self.mode, PixelMode::L | PixelMode::Rgb) || key.len() != self.mode.channels() || self.bit_depth != 8 {
            return Err(PyValueError::new_err("invalid transparency color"));
        }
        let destination = if matches!(mode, "LA" | "La") { "LA" } else { "RGBA" };
        let mut image = self.convert(py, destination, None)?;
        let channels = image.mode.channels();
        let source = self.pixel_data()?;
        let output = image.pixels_mut().unwrap();
        py.detach(|| {
            for (pixel, output) in source.chunks_exact(key.len()).zip(output.chunks_exact_mut(channels)) {
                output[channels - 1] = if pixel == key { 0 } else { 255 };
                if pixel == key && self.mode == PixelMode::Rgb && destination == "LA" {
                    output[0] = key[0];
                }
            }
        });
        if mode == destination { Ok(image) } else { image.convert(py, mode, None) }
    }

    pub fn convert_dithered_bilevel(&self, py: Python<'_>) -> PyResult<Self> {
        if !matches!(self.mode, PixelMode::L | PixelMode::Rgb) || self.bit_depth != 8 {
            return Err(PyValueError::new_err("conversion not supported"));
        }
        let source = self.pixel_data()?;
        let channels = self.mode.channels();
        let width = self.width as usize;
        let mut pixels = alloc_pixels(width * self.height as usize)?;
        let mut errors = vec![0i32; width + 1];
        py.detach(|| {
            if width == 0 {
                return;
            }
            for (row, output) in source.chunks_exact(width * channels).zip(pixels.chunks_exact_mut(width)) {
                let (mut right, mut below, mut below_right) = (0, 0, 0);
                for (x, (pixel, out)) in row.chunks_exact(channels).zip(output).enumerate() {
                    let gray = if channels == 1 {
                        i32::from(pixel[0])
                    } else {
                        (299 * i32::from(pixel[0]) + 587 * i32::from(pixel[1]) + 114 * i32::from(pixel[2])) / 1000
                    };
                    let adjusted = (gray + (right + errors[x + 1]) / 16).clamp(0, 255);
                    *out = if adjusted > 128 { 255 } else { 0 };
                    let error = adjusted - i32::from(*out);
                    errors[x] = 3 * error + below;
                    below = 5 * error + below_right;
                    below_right = error;
                    right = 7 * error;
                }
                errors[width] = below;
            }
        });
        Self::from_pixels(self.width, self.height, PixelMode::One, pixels, None)
    }

    #[pyo3(signature = (mode, bit_depth=None))]
    pub fn convert(&self, py: Python<'_>, mode: &str, bit_depth: Option<u8>) -> PyResult<Self> {
        if mode == "P" && self.palette.is_some() {
            return self.copy(py);
        }
        let destination = PixelMode::parse(mode)?;
        if matches!(self.mode, PixelMode::PremulLa | PixelMode::PremulRgba | PixelMode::Rgbx)
            || matches!(destination, PixelMode::PremulLa | PixelMode::PremulRgba | PixelMode::Rgbx)
        {
            if matches!(destination, PixelMode::PremulLa | PixelMode::PremulRgba | PixelMode::Rgbx) && bit_depth.is_some_and(|depth| depth != 8) {
                return Err(PyValueError::new_err("this mode requires 8-bit pixels"));
            }
            if self.mode == destination {
                return self.copy(py);
            }
            if matches!((self.mode, destination), (PixelMode::Rgbx, PixelMode::Rgba) | (PixelMode::Rgba, PixelMode::Rgbx))
                && self.bit_depth == 8
                && bit_depth.is_none_or(|depth| depth == 8)
            {
                let source = self.raw_data()?;
                let pixels = py.detach(|| crate::simd::opaque_rgba(source));
                return Self::from_pixels(self.width, self.height, destination, pixels, None);
            }
            if matches!(self.mode, PixelMode::PremulLa | PixelMode::PremulRgba) {
                let straight = if self.mode == PixelMode::PremulLa { PixelMode::La } else { PixelMode::Rgba };
                if self.mode == PixelMode::PremulLa && destination != PixelMode::La {
                    return Err(PyValueError::new_err("conversion from La requires LA"));
                }
                let source = self.raw_data()?;
                let pixels = py.detach(|| {
                    let mut pixels = vec![0; source.len()];
                    if straight == PixelMode::La {
                        crate::alpha::convert::<2>(source, &mut pixels, false);
                    } else {
                        crate::alpha::convert::<4>(source, &mut pixels, false);
                    }
                    pixels
                });
                let image = Self::from_pixels(self.width, self.height, straight, pixels, None)?;
                return if destination == straight {
                    if bit_depth.is_none_or(|depth| depth == 8) { Ok(image) } else { image.convert(py, mode, bit_depth) }
                } else {
                    image.convert(py, "RGB", None)?.convert(py, mode, bit_depth)
                };
            }
            if matches!(destination, PixelMode::PremulLa | PixelMode::PremulRgba) {
                let straight = if destination == PixelMode::PremulLa { PixelMode::La } else { PixelMode::Rgba };
                let converted;
                let image = if self.mode == straight {
                    self
                } else if self.mode.channels() >= 3 || self.palette.is_some() {
                    converted = self.convert(py, "RGB", Some(8))?.convert(py, straight.as_str(), None)?;
                    &converted
                } else {
                    return Err(PyValueError::new_err("unsupported premultiplied conversion"));
                };
                let source = image.raw_data()?;
                let pixels = py.detach(|| {
                    let mut pixels = vec![0; source.len()];
                    if straight == PixelMode::La {
                        crate::alpha::convert::<2>(source, &mut pixels, true);
                    } else {
                        crate::alpha::convert::<4>(source, &mut pixels, true);
                    }
                    pixels
                });
                return Self::from_pixels(self.width, self.height, destination, pixels, None);
            }
            if self.mode == PixelMode::Rgbx {
                let source = self.raw_data()?;
                if bit_depth.is_none_or(|depth| depth == 8) && matches!(destination, PixelMode::L | PixelMode::Rgb) {
                    let pixels = py.detach(|| crate::simd::convert(source, PixelMode::Rgba, destination));
                    return Self::from_pixels(self.width, self.height, destination, pixels, None);
                }
                let pixels = py.detach(|| crate::simd::convert(source, PixelMode::Rgba, PixelMode::Rgb));
                let image = Self::from_pixels(self.width, self.height, PixelMode::Rgb, pixels, None)?;
                return image.convert(py, mode, bit_depth);
            }
            let mut image = if self.mode == PixelMode::La || self.mode == PixelMode::Rgb && self.bit_depth == 8 {
                self.convert(py, "RGBA", None)?
            } else {
                self.convert(py, "RGB", Some(8))?.convert(py, "RGBA", None)?
            };
            image.mode = destination;
            return Ok(image);
        }
        if matches!(self.mode, PixelMode::RgbF | PixelMode::RgbaF) || matches!(destination, PixelMode::RgbF | PixelMode::RgbaF) {
            return py.detach(|| crate::float::convert(self, destination, bit_depth));
        }
        if self.mode == PixelMode::Lab && destination != PixelMode::Lab {
            let source = self.pixel_data()?;
            let pixels = py.detach(|| convert_lab(source, false))?;
            let rgb = Self::from_pixels(self.width, self.height, PixelMode::Rgb, pixels, None)?;
            return if destination == PixelMode::Rgb && bit_depth.is_none_or(|depth| depth == 8) {
                Ok(rgb)
            } else {
                rgb.convert(py, mode, bit_depth)
            };
        }
        if destination == PixelMode::Lab && self.mode != PixelMode::Lab {
            if bit_depth.is_some_and(|depth| depth != 8) {
                return Err(PyValueError::new_err("this mode requires 8-bit pixels"));
            }
            let converted;
            let rgb = if self.mode == PixelMode::Rgb && self.bit_depth == 8 {
                self
            } else {
                converted = self.convert(py, "RGB", Some(8))?;
                &converted
            };
            let source = rgb.pixel_data()?;
            let pixels = py.detach(|| convert_lab(source, true))?;
            return Self::from_pixels(self.width, self.height, PixelMode::Lab, pixels, None);
        }
        if destination == PixelMode::F
            && self.bit_depth == 8
            && matches!(self.mode, PixelMode::Rgb | PixelMode::Rgba | PixelMode::Cmyk | PixelMode::YCbCr)
        {
            if bit_depth.is_some_and(|depth| depth != 32) {
                return Err(PyValueError::new_err("bit_depth does not match float mode"));
            }
            let source = self.pixel_data()?;
            let pixels = py.detach(|| {
                let luma = |r, g, b| ((f64::from(r) * 299.0 + f64::from(g) * 587.0 + f64::from(b) * 114.0) as f32 / 1000.0).to_le_bytes();
                if self.mode == PixelMode::Rgb {
                    return map_pixels::<3, 4>(source, |p| luma(p[0], p[1], p[2]));
                }
                if self.mode == PixelMode::Rgba {
                    return map_pixels::<4, 4>(source, |p| luma(p[0], p[1], p[2]));
                }
                let rgb = convert_pixels(source, self.mode, PixelMode::Rgb);
                rgb.as_chunks::<3>()
                    .0
                    .iter()
                    .flat_map(|p| ((f64::from(p[0]) * 299.0 + f64::from(p[1]) * 587.0 + f64::from(p[2]) * 114.0) as f32 / 1000.0).to_le_bytes())
                    .collect()
            });
            return Self::from_float_bytes(self.width, self.height, pixels);
        }
        if self.mode == PixelMode::F || destination == PixelMode::F {
            if destination == PixelMode::F {
                if bit_depth.is_some_and(|depth| depth != 32) {
                    return Err(PyValueError::new_err("bit_depth does not match float mode"));
                }
                if self.mode == PixelMode::F {
                    return self.copy(py);
                }
                let converted;
                let source = if self.mode.is_integer() || self.mode == PixelMode::L && self.bit_depth > 8 {
                    self
                } else {
                    converted = self.convert(py, "L", Some(8))?;
                    &converted
                };
                let data = source.raw_data()?;
                let pixels = py.detach(|| {
                    if source.mode.is_integer() {
                        crate::simd::integer_to_float(data, source.mode)
                    } else if source.bit_depth > 8 {
                        crate::simd::integer_to_float(data, PixelMode::I16)
                    } else {
                        crate::simd::gray_to_float(data)
                    }
                });
                return Self::from_float_bytes(self.width, self.height, pixels);
            }
            if destination == PixelMode::L && bit_depth == Some(16) {
                let source = self.raw_data()?;
                let pixels = py.detach(|| crate::simd::float_to_u16(source, false));
                return Self::from_sample_bytes(self.width, self.height, PixelMode::L, pixels, 16, None);
            }
            let data = self.raw_data()?;
            if destination == PixelMode::I {
                let pixels = py.detach(|| crate::simd::float_to_integer(data));
                return Self::from_integer_bytes(self.width, self.height, destination, pixels);
            }
            if destination.is_integer() {
                let pixels = py.detach(|| crate::simd::float_to_u16(data, destination.is_big_endian()));
                return Self::from_integer_bytes(self.width, self.height, destination, pixels);
            }
            let pixels = py.detach(|| crate::simd::float_to_l(data));
            let gray = Self::from_pixels(self.width, self.height, PixelMode::L, pixels, None)?;
            if destination == PixelMode::L && bit_depth.is_none_or(|depth| depth == 8) {
                return Ok(gray);
            }
            return gray.convert(py, mode, bit_depth);
        }
        if matches!(self.mode, PixelMode::Cmyk | PixelMode::YCbCr) && destination != self.mode {
            let source = self.pixel_data()?;
            if self.mode == PixelMode::YCbCr && matches!(destination, PixelMode::L | PixelMode::La) {
                let luma = py.detach(|| crate::simd::extract_channel::<3>(source, 0));
                let gray = Self::from_pixels(self.width, self.height, PixelMode::L, luma, None)?;
                return if destination == PixelMode::L && bit_depth.is_none_or(|depth| depth == 8) {
                    Ok(gray)
                } else {
                    gray.convert(py, mode, bit_depth)
                };
            }
            if bit_depth.is_none_or(|depth| depth == 8) && matches!(destination, PixelMode::L | PixelMode::La | PixelMode::Rgb | PixelMode::Rgba) {
                let pixels = py.detach(|| convert_pixels(source, self.mode, destination));
                return Self::from_pixels(self.width, self.height, destination, pixels, None);
            }
            let pixels = py.detach(|| convert_pixels(source, self.mode, PixelMode::Rgb));
            if destination == PixelMode::One {
                if bit_depth.is_some_and(|depth| depth != 8) {
                    return Err(PyValueError::new_err("this mode requires 8-bit pixels"));
                }
                let pixels = py.detach(|| {
                    pixels
                        .as_chunks::<3>()
                        .0
                        .iter()
                        .map(|p| if u32::from(p[0]) * 299 + u32::from(p[1]) * 587 + u32::from(p[2]) * 114 >= 128000 { 255 } else { 0 })
                        .collect()
                });
                return Self::from_pixels(self.width, self.height, destination, pixels, None);
            }
            let rgb = Self::from_pixels(self.width, self.height, PixelMode::Rgb, pixels, None)?;
            if destination == PixelMode::Rgb && bit_depth.is_none_or(|depth| depth == 8) {
                return Ok(rgb);
            }
            return rgb.convert(py, mode, bit_depth);
        }
        if matches!(destination, PixelMode::Cmyk | PixelMode::YCbCr) && self.bit_depth > 8 && !self.mode.is_wide_scalar() {
            let intermediate = if self.mode == PixelMode::L { "L" } else { "RGB" };
            return self.convert(py, intermediate, Some(8))?.convert(py, mode, bit_depth);
        }
        if self.mode == PixelMode::Hsv && destination != PixelMode::Hsv {
            if bit_depth.is_none_or(|depth| depth == 8) && matches!(destination, PixelMode::L | PixelMode::Rgb | PixelMode::Rgba) {
                let source = self.pixel_data()?;
                let pixels = py.detach(|| convert_pixels(source, PixelMode::Hsv, destination));
                return Self::from_pixels(self.width, self.height, destination, pixels, None);
            }
            let source = self.pixel_data()?;
            let pixels = py.detach(|| hsv_to_rgb_bytes(source));
            let rgb = Self::from_pixels(self.width, self.height, PixelMode::Rgb, pixels, None)?;
            return if destination == PixelMode::Rgb { Ok(rgb) } else { rgb.convert(py, mode, bit_depth) };
        }
        if destination == PixelMode::Hsv && self.bit_depth > 8 {
            return self.convert(py, "RGB", Some(8))?.convert(py, "HSV", None);
        }
        if self.mode.is_integer() || destination.is_integer() {
            if let Some(depth) = bit_depth
                && depth != (destination.sample_bytes() << 3) as u8
                && destination.is_integer()
            {
                return Err(PyValueError::new_err("bit_depth does not match integer mode"));
            }
            if self.mode == destination {
                return self.copy(py);
            }
            if destination.is_integer() {
                let converted;
                let source = if self.palette.is_none() && (self.mode.is_integer() || self.mode == PixelMode::L) {
                    self
                } else {
                    converted = self.convert(py, "L", Some(8))?;
                    &converted
                };
                let data = source.raw_data()?;
                if source.mode.is_integer() {
                    let output = py.detach(|| crate::simd::convert_integer(data, source.mode, destination));
                    return Self::from_integer_bytes(self.width, self.height, destination, output);
                }
                let output = py.detach(|| {
                    if source.bit_depth > 8 {
                        crate::simd::convert_integer(data, PixelMode::I16, destination)
                    } else {
                        crate::simd::gray_to_integer(data, destination)
                    }
                });
                return Self::from_integer_bytes(self.width, self.height, destination, output);
            }
            let data = self.raw_data()?;
            if destination == PixelMode::L && bit_depth.is_some_and(|depth| depth > 8) {
                let pixels = py.detach(|| crate::simd::convert_integer(data, self.mode, PixelMode::I16));
                let gray = Self::from_sample_bytes(self.width, self.height, PixelMode::L, pixels, 16, None)?;
                return gray.convert(py, mode, bit_depth);
            }
            if matches!(destination, PixelMode::Rgb | PixelMode::Rgba) {
                let pixels = py.detach(|| crate::simd::integer_to_color(data, self.mode, destination.channels()));
                return Self::from_pixels(self.width, self.height, destination, pixels, None);
            }
            let clipped = py.detach(|| crate::simd::integer_to_l(data, self.mode));
            let grayscale = Self::from_pixels(self.width, self.height, PixelMode::L, clipped, None)?;
            if destination == PixelMode::L && bit_depth.is_none_or(|depth| depth == 8) {
                return Ok(grayscale);
            }
            return grayscale.convert(py, mode, bit_depth);
        }
        let depth = bit_depth.unwrap_or(self.bit_depth);
        if !matches!(depth, 8 | 10 | 12 | 16) {
            return Err(PyValueError::new_err("bit_depth must be 8, 10, 12, or 16"));
        }
        if (self.mode != PixelMode::L && self.mode != PixelMode::Rgb && self.mode != PixelMode::Rgba
            || destination != PixelMode::L && destination != PixelMode::Rgb && destination != PixelMode::Rgba)
            && depth != 8
        {
            return Err(PyValueError::new_err("this mode requires 8-bit pixels"));
        }
        if self.bit_depth > 8 || depth > 8 {
            if self.palette.is_some() {
                let expanded = self.convert(py, mode, Some(8))?;
                return expanded.convert(py, mode, Some(depth));
            }
            let bytes = self.raw_data()?;
            if self.mode == destination {
                let output = py.detach(|| crate::simd::rescale_samples(bytes, self.bit_depth, depth));
                return if depth == 8 {
                    Self::from_pixels(self.width, self.height, destination, output, None)
                } else {
                    Self::from_sample_bytes(self.width, self.height, destination, output, depth, None)
                };
            }
            let output = py.detach(|| crate::simd::convert_samples(bytes, self.mode, destination, self.bit_depth, depth));
            return if depth == 8 {
                Self::from_pixels(self.width, self.height, destination, output, None)
            } else {
                Self::from_sample_bytes(self.width, self.height, destination, output, depth, None)
            };
        }
        if let Some((palette_mode, palette)) = &self.palette {
            let indices = self.pixel_data()?;
            let converted = py.detach(|| {
                if self.mode == PixelMode::Pa {
                    if matches!(destination, PixelMode::L | PixelMode::Rgb | PixelMode::Rgba) {
                        return expand_pa(indices, *palette_mode, palette, destination);
                    }
                    let len = indices.len().checked_mul(2).ok_or_else(|| PyMemoryError::new_err("cannot allocate image"))?;
                    let mut rgba = crate::buffers::reserved(len)?;
                    for pair in indices.as_chunks::<2>().0 {
                        let entry = pair[0] as usize * palette_mode.channels();
                        let rgb = palette.get(entry..entry + 3).unwrap_or(&[0, 0, 0]);
                        rgba.extend_from_slice(rgb);
                        rgba.push(pair[1]);
                    }
                    return Ok(convert_pixels(&rgba, PixelMode::Rgba, destination));
                }
                convert_palette(indices, *palette_mode, palette, destination)
            })?;
            return Self::from_pixels(self.width, self.height, destination, converted, None);
        }
        let source = self.mode;
        let pixels = self.pixel_data()?;
        let converted = py.detach(|| convert_pixels(pixels, source, destination));
        Self::from_pixels(self.width, self.height, destination, converted, None)
    }

    #[pyo3(signature = (rawmode=None, stride=0, orientation=1))]
    pub fn tobytes(&self, py: Python<'_>, rawmode: Option<&str>, stride: usize, orientation: i8) -> PyResult<Py<PyBytes>> {
        let pixels = self.raw_data()?;
        if stride == 0 && orientation == 1 && rawmode.is_none_or(|mode| mode == self.mode.as_str() || self.mode == PixelMode::L && mode == "P") {
            if self.mode == PixelMode::One {
                let width = self.width as usize;
                return PyBytes::new_with(py, width.div_ceil(8) * self.height as usize, |output| {
                    py.detach(|| pack_bilevel_into(pixels, width, output));
                    Ok(())
                })
                .map(Bound::unbind);
            }
            if self.mode == PixelMode::Lab {
                return PyBytes::new_with(py, pixels.len(), |output| {
                    py.detach(|| {
                        output.copy_from_slice(pixels);
                        lab_raw_bytes_mut(output);
                    });
                    Ok(())
                })
                .map(Bound::unbind);
            }
            return Ok(PyBytes::new(py, pixels).unbind());
        }
        let packed;
        let source = if self.mode == PixelMode::Lab {
            packed = py.detach(|| lab_raw_bytes(pixels));
            packed.as_slice()
        } else if self.mode == PixelMode::One {
            packed = py.detach(|| pack_bilevel(pixels, self.width as usize, self.height as usize));
            packed.as_slice()
        } else {
            pixels
        };
        crate::raw::encode(py, source, self.mode, self.size(), self.bit_depth, rawmode, stride, orientation).map(Bound::unbind)
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

#[pyfunction(signature = (mode, size, data, bit_depth=8, rawmode=None, stride=0, orientation=1, decoder="raw"))]
pub fn frombytes(
    py: Python<'_>, mode: &str, size: (u32, u32), data: &[u8], bit_depth: u8, rawmode: Option<&str>, stride: usize, orientation: i8, decoder: &str,
) -> PyResult<Image> {
    let mode = PixelMode::parse(mode)?;
    py.detach(|| {
        let input = crate::raw::decode(data, decoder)?;
        from_raw_input(mode, size, input, bit_depth, rawmode, stride, orientation)
    })
}

#[pyfunction(signature = (mode, size, data, bit_depth=8, rawmode=None, stride=0, orientation=1))]
pub fn frombuffer_copy(
    py: Python<'_>, mode: &str, size: (u32, u32), data: &Bound<'_, PyAny>, bit_depth: u8, rawmode: Option<&str>, stride: usize, orientation: i8,
) -> PyResult<Image> {
    let mode = PixelMode::parse(mode)?;
    let input = std::borrow::Cow::Owned(array_bytes(data)?);
    py.detach(|| from_raw_input(mode, size, input, bit_depth, rawmode, stride, orientation))
}

fn from_raw_input(
    mode: PixelMode, size: (u32, u32), input: std::borrow::Cow<'_, [u8]>, bit_depth: u8, rawmode: Option<&str>, stride: usize, orientation: i8,
) -> PyResult<Image> {
    let decoded = crate::raw::transform(&input, mode, size, bit_depth, rawmode, stride, orientation, false)?;
    if mode == PixelMode::One {
        if bit_depth != 8 {
            return Err(PyValueError::new_err("mode 1 requires 8-bit pixels"));
        }
        let row_bytes = (size.0 as usize).div_ceil(8);
        if decoded.len() != row_bytes * size.1 as usize {
            return Err(PyValueError::new_err("wrong amount of image data"));
        }
        let pixels = unpack_bilevel(&decoded, size.0 as usize, size.1 as usize);
        return Image::from_pixels(size.0, size.1, mode, pixels, None);
    }
    let mut data = match decoded {
        std::borrow::Cow::Owned(data) => {
            drop(input);
            data
        }
        std::borrow::Cow::Borrowed(_) => input.into_owned(),
    };
    if mode.is_float() {
        if bit_depth != 8 && bit_depth != 32 {
            return Err(PyValueError::new_err("bit_depth does not match float mode"));
        }
        return Image::from_color_float_bytes(size.0, size.1, mode, data);
    }
    if mode.is_integer() {
        if bit_depth != 8 && bit_depth != (mode.sample_bytes() << 3) as u8 {
            return Err(PyValueError::new_err("bit_depth does not match integer mode"));
        }
        return Image::from_integer_bytes(size.0, size.1, mode, data);
    }
    if matches!(mode, PixelMode::La | PixelMode::PremulLa | PixelMode::Pa | PixelMode::Hsv | PixelMode::Cmyk | PixelMode::YCbCr | PixelMode::Lab)
        && bit_depth != 8
    {
        return Err(PyValueError::new_err("this mode requires 8-bit pixels"));
    }
    if bit_depth != 8 {
        if !data.len().is_multiple_of(2) {
            return Err(PyValueError::new_err("16-bit storage requires an even byte count"));
        }
        return Image::from_sample_bytes(size.0, size.1, mode, data, bit_depth, None);
    }
    if mode == PixelMode::Lab {
        lab_raw_bytes_mut(&mut data);
    }
    Image::from_pixels(size.0, size.1, mode, data, None)
}

#[pyfunction]
pub fn mapped_getpixel(
    py: Python<'_>, data: &Bound<'_, PyAny>, mode: &str, size: (u32, u32), stride: usize, orientation: i8, xy: (i64, i64),
) -> PyResult<Py<PyAny>> {
    let mode = PixelMode::parse(if mode == "P" { "L" } else { mode })?;
    if !matches!(mode, PixelMode::L | PixelMode::Rgbx | PixelMode::Rgba | PixelMode::Cmyk | PixelMode::I16 | PixelMode::I16L | PixelMode::I16B)
        || !matches!(orientation, -1 | 1)
    {
        return Err(PyValueError::new_err("invalid mapped pixel layout"));
    }
    let (x, y) = (if xy.0 < 0 { xy.0 + i64::from(size.0) } else { xy.0 }, if xy.1 < 0 { xy.1 + i64::from(size.1) } else { xy.1 });
    if x < 0 || y < 0 || x >= i64::from(size.0) || y >= i64::from(size.1) {
        return Err(PyIndexError::new_err("image index out of range"));
    }
    let pixel_bytes = mode.channels() * mode.sample_bytes();
    let required = stride.checked_mul(size.1 as usize).ok_or_else(|| PyValueError::new_err("image dimensions are too large"))?;
    if stride < size.0 as usize * pixel_bytes {
        return Err(PyValueError::new_err("invalid mapped pixel stride"));
    }
    data.cast::<PyMemoryView>()?;
    let buffer = PyBuffer::<u8>::get(data)?;
    if !buffer.is_c_contiguous() || buffer.dimensions() != 1 {
        return Err(PyValueError::new_err("mapped buffer must be contiguous and one-dimensional"));
    }
    if buffer.len_bytes() < required {
        return Err(PyValueError::new_err("buffer is not large enough"));
    }
    let row = if orientation == 1 { y as usize } else { size.1 as usize - 1 - y as usize };
    let offset = row * stride + x as usize * pixel_bytes;
    let view = data.get_item(PySlice::new(py, offset as isize, (offset + pixel_bytes) as isize, 1))?;
    let mut pixel = [0; 4];
    PyBuffer::<u8>::get(&view)?.copy_to_slice(py, &mut pixel[..pixel_bytes])?;
    if mode.is_integer() {
        return Ok(integer_value(mode, &pixel[..pixel_bytes]).into_pyobject(py)?.into_any().unbind());
    }
    if mode == PixelMode::L {
        return Ok(pixel[0].into_pyobject(py)?.into_any().unbind());
    }
    Ok(PyTuple::new(py, pixel)?.into_any().unbind())
}

fn array_bytes(obj: &Bound<'_, PyAny>) -> PyResult<Vec<u8>> {
    if let Ok(bytes) = obj.cast::<PyBytes>() {
        return crate::buffers::copy(bytes.as_bytes());
    }
    if let Ok(buffer) = PyBuffer::<u8>::get(obj) {
        return buffer.to_vec(obj.py());
    }
    if let Ok(view) = PyMemoryView::from(obj) {
        if let Ok(raw) = view.call_method1("cast", ("B",)) {
            return PyBuffer::<u8>::get(&raw)?.to_vec(obj.py());
        }
        macro_rules! copy_typed {
            ($($sample:ty),+ $(,)?) => {
                $(if let Ok(buffer) = PyBuffer::<$sample>::get(obj) {
                    let mut pixels = alloc_pixels(buffer.len_bytes())?;
                    // SAFETY: Numeric samples accept every bit pattern. Copying retains the source byte order for the conversion that follows.
                    let (prefix, values, suffix) = unsafe { pixels.align_to_mut::<$sample>() };
                    if prefix.is_empty() && suffix.is_empty() {
                        buffer.copy_to_slice(obj.py(), values)?;
                        return Ok(pixels);
                    }
                })+
            };
        }
        copy_typed!(u16, i32, f32, f64);
        let bytes = view.call_method0("tobytes")?;
        return crate::buffers::copy(bytes.cast::<PyBytes>()?.as_bytes());
    }
    let bytes = obj.call_method0("tobytes")?;
    crate::buffers::copy(bytes.cast::<PyBytes>()?.as_bytes())
}

#[pyfunction(signature = (obj, mode = None, bit_depth = None, interface = None))]
pub fn fromarray(
    py: Python<'_>, obj: &Bound<'_, PyAny>, mode: Option<&str>, bit_depth: Option<u8>, interface: Option<Bound<'_, PyAny>>,
) -> PyResult<Image> {
    let interface = interface.map_or_else(|| obj.getattr("__array_interface__"), Ok)?;
    let shape: Vec<usize> = interface.get_item("shape")?.extract()?;
    let typestr: String = interface.get_item("typestr")?.extract()?;
    let wide = matches!(typestr.as_str(), "<u2" | ">u2" | "=u2");
    let inferred_mode = array_mode(&shape, if wide && bit_depth.is_some() { "|u1" } else { &typestr })?;
    let mode = mode.map_or(Ok(inferred_mode), PixelMode::parse)?;

    let maximum_dimensions = match mode {
        PixelMode::One | PixelMode::L | PixelMode::I | PixelMode::F | PixelMode::I16 | PixelMode::I16L | PixelMode::I16B | PixelMode::I16N => 2,
        PixelMode::La | PixelMode::PremulLa | PixelMode::Pa => 3,
        PixelMode::Rgb | PixelMode::RgbF | PixelMode::Hsv | PixelMode::YCbCr | PixelMode::Lab => 3,
        PixelMode::Rgba | PixelMode::PremulRgba | PixelMode::Rgbx | PixelMode::RgbaF | PixelMode::Cmyk => 4,
    };
    if shape.len() > maximum_dimensions {
        return Err(PyValueError::new_err(format!("too many dimensions: {} > {maximum_dimensions}", shape.len())));
    }

    let (width, height) = match shape.as_slice() {
        [] => {
            return Err(PyValueError::new_err("array must have at least one dimension"));
        }
        [height] => (1, *height),
        [height, width, ..] => (*width, *height),
    };
    let width = u32::try_from(width).map_err(|_| PyValueError::new_err("image dimensions are too large"))?;
    let height = u32::try_from(height).map_err(|_| PyValueError::new_err("image dimensions are too large"))?;

    if mode.is_float() {
        if bit_depth.is_some_and(|depth| depth != 32) {
            return Err(PyValueError::new_err("bit_depth does not match float mode"));
        }
        let mut raw = array_bytes(obj)?;
        return py.detach(|| {
            let pixels = match typestr.as_str() {
                "<f4" | "=f4" => raw,
                ">f4" => {
                    for bytes in raw.as_chunks_mut::<4>().0 {
                        bytes.reverse();
                    }
                    raw
                }
                "<f8" | "=f8" => {
                    raw.as_chunks::<8>().0.iter().map(|bytes| (f64::from_le_bytes(*bytes) as f32).to_le_bytes()).collect::<Vec<_>>().into_flattened()
                }
                ">f8" => {
                    raw.as_chunks::<8>().0.iter().map(|bytes| (f64::from_be_bytes(*bytes) as f32).to_le_bytes()).collect::<Vec<_>>().into_flattened()
                }
                _ => return Err(PyTypeError::new_err("float mode requires float32 or float64 array")),
            };
            Image::from_color_float_bytes(width, height, mode, pixels)
        });
    }
    if mode.is_integer() {
        let source = match typestr.as_str() {
            "<i4" | "=i4" => PixelMode::I,
            "<u2" => PixelMode::I16L,
            "=u2" => PixelMode::I16N,
            ">u2" => PixelMode::I16B,
            _ => return Err(PyTypeError::new_err("integer mode requires int32 or uint16 array")),
        };
        if let Some(depth) = bit_depth
            && depth != (mode.sample_bytes() * 8) as u8
        {
            return Err(PyValueError::new_err("bit_depth does not match integer mode"));
        }
        let raw = array_bytes(obj)?;
        if source != PixelMode::I && mode != PixelMode::I {
            let mut raw = raw;
            if source.is_big_endian() != mode.is_big_endian() {
                for bytes in raw.as_chunks_mut::<2>().0 {
                    bytes.swap(0, 1);
                }
            }
            return Image::from_integer_bytes(width, height, mode, raw);
        }
        let input = Image::from_integer_bytes(width, height, source, raw)?;
        return if source == mode { Ok(input) } else { input.convert(py, mode.as_str(), None) };
    }

    let depth = bit_depth.unwrap_or(if wide { 16 } else { 8 });
    if wide {
        let mut raw = array_bytes(obj)?;
        if typestr == ">u2" {
            for bytes in raw.as_chunks_mut::<2>().0 {
                bytes.swap(0, 1);
            }
        }
        return Image::from_sample_bytes(width, height, mode, raw, depth, None);
    }
    if depth != 8 {
        return Err(PyValueError::new_err("high-bit-depth arrays must use uint16 samples"));
    }
    let mut pixels = array_bytes(obj)?;
    if mode == PixelMode::Lab {
        lab_raw_bytes_mut(&mut pixels);
    }
    Image::from_pixels(width, height, mode, pixels, None)
}

fn array_mode(shape: &[usize], typestr: &str) -> PyResult<PixelMode> {
    if matches!(typestr, "<f4" | "=f4" | ">f4" | "<f8" | "=f8" | ">f8") {
        match shape {
            [_, _, 3] => return Ok(PixelMode::RgbF),
            [_, _, 4] => return Ok(PixelMode::RgbaF),
            _ => {}
        }
    }
    if matches!(shape, [_] | [_, _]) {
        match typestr {
            "<i4" | "=i4" => return Ok(PixelMode::I),
            "<f4" | "=f4" | ">f4" | "<f8" | "=f8" | ">f8" => return Ok(PixelMode::F),
            "<u2" => return Ok(PixelMode::I16),
            "=u2" => return Ok(PixelMode::I16N),
            ">u2" => return Ok(PixelMode::I16B),
            _ => {}
        }
    }
    if matches!(typestr, "|u1" | "<u2" | ">u2" | "=u2") {
        match shape {
            [_] | [_, _] => return Ok(PixelMode::L),
            [_, _, 2] if typestr == "|u1" => return Ok(PixelMode::La),
            [_, _, 3] => return Ok(PixelMode::Rgb),
            [_, _, 4] => return Ok(PixelMode::Rgba),
            _ => {}
        }
    }

    let type_shape = match shape {
        [] | [_] | [_, _] => vec![1, 1],
        [_, _, channels, ..] => vec![1, 1, *channels],
    };
    Err(PyTypeError::new_err(format!("cannot handle this data type: {type_shape:?}, {typestr}")))
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
    if pixels.len() != expected || matches!(bit_depth, 10 | 12) && !crate::simd::samples_in_range(pixels, bit_depth) {
        return Err(PyValueError::new_err("invalid image data"));
    }
    Ok(())
}

fn expected_len(width: u32, height: u32, mode: PixelMode) -> PyResult<usize> {
    let pixels = (width as usize).checked_mul(height as usize).ok_or_else(|| PyValueError::new_err("image dimensions are too large"))?;
    pixels.checked_mul(mode.channels()).ok_or_else(|| PyValueError::new_err("image dimensions are too large"))
}

fn convert_pixels(source: &[u8], from: PixelMode, to: PixelMode) -> Vec<u8> {
    if from == to {
        return source.to_vec();
    }
    if matches!(from, PixelMode::Cmyk | PixelMode::YCbCr) {
        if matches!(to, PixelMode::L | PixelMode::La | PixelMode::Rgb | PixelMode::Rgba) {
            return if from == PixelMode::Cmyk {
                map_rgb_pixels::<4>(source, to, cmyk_to_rgb)
            } else {
                map_rgb_pixels::<3>(source, to, ycbcr_to_rgb)
            };
        }
        let rgb = if from == PixelMode::Cmyk { map_pixels::<4, 3>(source, cmyk_to_rgb) } else { map_pixels::<3, 3>(source, ycbcr_to_rgb) };
        return if to == PixelMode::Rgb { rgb } else { convert_pixels(&rgb, PixelMode::Rgb, to) };
    }
    if matches!(to, PixelMode::Cmyk | PixelMode::YCbCr) {
        if matches!(from, PixelMode::One | PixelMode::L | PixelMode::La) {
            return match (from, to) {
                (PixelMode::La, PixelMode::Cmyk) => map_pixels::<2, 4>(source, |p| [0, 0, 0, 255 - p[0]]),
                (PixelMode::La, _) => map_pixels::<2, 3>(source, |p| [p[0], 128, 128]),
                (PixelMode::One, PixelMode::Cmyk) => map_pixels::<1, 4>(source, |p| [0, 0, 0, if p[0] == 0 { 255 } else { 0 }]),
                (PixelMode::One, _) => map_pixels::<1, 3>(source, |p| [if p[0] == 0 { 0 } else { 255 }, 128, 128]),
                (_, PixelMode::Cmyk) => crate::simd::gray_to_cmyk(source),
                _ => crate::simd::gray_to_ycbcr(source),
            };
        }
        if from == PixelMode::Rgba {
            return if to == PixelMode::Cmyk {
                crate::simd::color_to_cmyk::<4>(source)
            } else {
                map_pixels::<4, 3>(source, |p| rgb_to_ycbcr([p[0], p[1], p[2]]))
            };
        }
        if from == PixelMode::Rgb {
            return if to == PixelMode::Cmyk { crate::simd::color_to_cmyk::<3>(source) } else { map_pixels::<3, 3>(source, rgb_to_ycbcr) };
        }
        let rgb = convert_pixels(source, from, PixelMode::Rgb);
        return if to == PixelMode::Cmyk {
            map_pixels::<3, 4>(&rgb, |p| [255 - p[0], 255 - p[1], 255 - p[2], 0])
        } else {
            map_pixels::<3, 3>(&rgb, rgb_to_ycbcr)
        };
    }
    if from == PixelMode::One {
        if matches!(to, PixelMode::L | PixelMode::F | PixelMode::Rgb | PixelMode::Rgba) {
            return crate::simd::convert_bilevel(source, to);
        }
        let gray = crate::simd::bilevel_to_l(source);
        return if to == PixelMode::L { gray } else { convert_pixels(&gray, PixelMode::L, to) };
    }
    if from == PixelMode::La && matches!(to, PixelMode::L | PixelMode::Rgb | PixelMode::Rgba) {
        return crate::simd::convert_la(source, to);
    }
    if from == PixelMode::Hsv {
        if to == PixelMode::L {
            return map_pixels::<3, 1>(source, |p| {
                let [r, g, b] = hsv_to_rgb(p[0], p[1], p[2]);
                [crate::simd::pillow_luma(r, g, b)]
            });
        }
        if to == PixelMode::Rgba {
            return map_pixels::<3, 4>(source, |p| {
                let [r, g, b] = hsv_to_rgb(p[0], p[1], p[2]);
                [r, g, b, 255]
            });
        }
        let rgb = hsv_to_rgb_bytes(source);
        return if to == PixelMode::Rgb { rgb } else { convert_pixels(&rgb, PixelMode::Rgb, to) };
    }
    if to == PixelMode::Hsv {
        return match from {
            PixelMode::One | PixelMode::L => crate::simd::gray_to_hsv::<1>(source),
            PixelMode::La => crate::simd::gray_to_hsv::<2>(source),
            PixelMode::Rgb => map_pixels::<3, 3>(source, |p| rgb_to_hsv(p[0], p[1], p[2])),
            PixelMode::Rgba => map_pixels::<4, 3>(source, |p| rgb_to_hsv(p[0], p[1], p[2])),
            _ => {
                let rgb = convert_pixels(source, from, PixelMode::Rgb);
                convert_pixels(&rgb, PixelMode::Rgb, PixelMode::Hsv)
            }
        };
    }
    if matches!(from, PixelMode::L | PixelMode::Rgb | PixelMode::Rgba) && matches!(to, PixelMode::L | PixelMode::Rgb | PixelMode::Rgba) {
        return crate::simd::convert(source, from, to);
    }
    convert_pixels_generic(source, from, to)
}

pub fn lab_raw_bytes(source: &[u8]) -> Vec<u8> {
    let mut output = source.to_vec();
    lab_raw_bytes_mut(&mut output);
    output
}

fn lab_raw_bytes_mut(pixels: &mut [u8]) {
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

fn convert_lab(source: &[u8], to_lab: bool) -> PyResult<Vec<u8>> {
    use lcms2::{CIExyY, Flags, Intent, PixelFormat, Profile, ThreadContext, Transform};
    let context = ThreadContext::new();
    let srgb = Profile::new_srgb_context(&context);
    let lab = Profile::new_lab4_context(&context, CIExyY::d50()).map_err(|error| PyValueError::new_err(error.to_string()))?;
    let (input, input_format, output, output_format) =
        if to_lab { (&srgb, PixelFormat::RGB_8, &lab, PixelFormat::Lab_8) } else { (&lab, PixelFormat::Lab_8, &srgb, PixelFormat::RGB_8) };
    // NO_CACHE makes the transform Sync; the cache only memoizes repeated pixels.
    let transform: Transform<u8, u8, ThreadContext, _> =
        Transform::new_flags_context(&context, input, input_format, output, output_format, Intent::Perceptual, Flags::NO_CACHE)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let mut pixels = vec![0_u8; source.len()];
    let chunk = crate::parallel::CHUNK_PIXELS * 3;
    crate::parallel::chunks_mut_above(&mut pixels, chunk, 64 * 1024, |i, dst| {
        transform.transform_pixels(&source[i * chunk..i * chunk + dst.len()], dst);
    });
    Ok(pixels)
}

fn rgb_to_ycbcr(p: [u8; 3]) -> [u8; 3] {
    // Pillow rounds each contribution at six fractional bits before summing.
    const fn table(coefficient: f64) -> [i32; 256] {
        let mut values = [0; 256];
        let mut i = 0;
        while i < 256 {
            values[i] = (coefficient * i as f64 * 64.0 + 0.5) as i32;
            i += 1;
        }
        values
    }
    static TERMS: [[i32; 256]; 8] =
        [table(0.299), table(0.587), table(0.114), table(-0.16874), table(-0.33126), table(0.5), table(-0.41869), table(-0.08131)];
    let [r, g, b] = p.map(usize::from);
    [
        ((TERMS[0][r] + TERMS[1][g] + TERMS[2][b]) >> 6) as u8,
        (((TERMS[3][r] + TERMS[4][g] + TERMS[5][b]) >> 6) + 128) as u8,
        (((TERMS[5][r] + TERMS[6][g] + TERMS[7][b]) >> 6) + 128) as u8,
    ]
}

fn ycbcr_to_rgb(p: [u8; 3]) -> [u8; 3] {
    const fn table(coefficient: f64) -> [i32; 256] {
        let mut values = [0; 256];
        let mut i = 0;
        while i < 256 {
            values[i] = (coefficient * (i as f64 - 128.0) * 64.0 + 0.5) as i32;
            i += 1;
        }
        values
    }
    static TERMS: [[i32; 256]; 4] = [table(1.402), table(-0.34414), table(-0.71414), table(1.772)];
    let [y, cb, cr] = p;
    let (y, cb, cr) = (i32::from(y), usize::from(cb), usize::from(cr));
    [clip8(y + (TERMS[0][cr] >> 6)), clip8(y + ((TERMS[1][cb] + TERMS[2][cr]) >> 6)), clip8(y + (TERMS[3][cb] >> 6))]
}

fn clip8(value: i32) -> u8 {
    if value <= 0 {
        0
    } else if value >= 256 {
        255
    } else {
        value as u8
    }
}

fn map_pixels<const S: usize, const D: usize>(source: &[u8], convert: impl Fn([u8; S]) -> [u8; D] + Sync + Send) -> Vec<u8> {
    let mut output = vec![0; source.len() / S * D];
    crate::parallel::chunks_mut(&mut output, crate::parallel::CHUNK_PIXELS * D, |chunk, dst| {
        let start = chunk * crate::parallel::CHUNK_PIXELS * S;
        for (src, dst) in source[start..].as_chunks::<S>().0.iter().zip(dst.as_chunks_mut::<D>().0) {
            *dst = convert(*src);
        }
    });
    output
}

fn map_rgb_pixels<const S: usize>(source: &[u8], to: PixelMode, rgb: impl Fn([u8; S]) -> [u8; 3] + Sync + Send) -> Vec<u8> {
    match to {
        PixelMode::L => map_pixels::<S, 1>(source, |p| {
            let [r, g, b] = rgb(p);
            [crate::simd::pillow_luma(r, g, b)]
        }),
        PixelMode::La => map_pixels::<S, 2>(source, |p| {
            let [r, g, b] = rgb(p);
            [crate::simd::pillow_luma(r, g, b), 255]
        }),
        PixelMode::Rgb => map_pixels::<S, 3>(source, rgb),
        PixelMode::Rgba => map_pixels::<S, 4>(source, |p| {
            let [r, g, b] = rgb(p);
            [r, g, b, 255]
        }),
        _ => unreachable!(),
    }
}

fn cmyk_to_rgb(pixel: [u8; 4]) -> [u8; 3] {
    std::array::from_fn(|i| {
        let value = u32::from(255 - pixel[i]) * u32::from(255 - pixel[3]) + 128;
        (((value >> 8) + value) >> 8) as u8
    })
}

fn rgb_to_hsv(r: u8, g: u8, b: u8) -> [u8; 3] {
    let maxc = r.max(g).max(b);
    let minc = r.min(g).min(b);
    if minc == maxc {
        return [0, 0, maxc];
    }
    let cr = f32::from(maxc - minc);
    let s = cr / f32::from(maxc);
    let rc = f32::from(maxc - r) / cr;
    let gc = f32::from(maxc - g) / cr;
    let bc = f32::from(maxc - b) / cr;
    let h = if r == maxc {
        bc - gc
    } else if g == maxc {
        (2.0 + f64::from(rc) - f64::from(bc)) as f32
    } else {
        (4.0 + f64::from(gc) - f64::from(rc)) as f32
    };
    let shifted = f64::from(h) / 6.0 + 1.0;
    let wrapped = (shifted - shifted.floor()) as f32;
    [clip8((f64::from(wrapped) * 255.0) as i32), clip8((f64::from(s) * 255.0) as i32), maxc]
}

fn hsv_to_rgb_bytes(source: &[u8]) -> Vec<u8> {
    map_pixels::<3, 3>(source, |p| hsv_to_rgb(p[0], p[1], p[2]))
}

fn hsv_to_rgb(h: u8, s: u8, v: u8) -> [u8; 3] {
    if s == 0 {
        return [v, v, v];
    }
    let sector = f32::from(h) * 6.0 / 255.0;
    let i = sector.floor();
    let f = sector - i;
    let fs = f32::from(s) / 255.0;
    let p = (f32::from(v) * (1.0 - fs)).round() as i32;
    let q = (f32::from(v) * (1.0 - fs * f)).round() as i32;
    let t = (f32::from(v) * (1.0 - fs * (1.0 - f))).round() as i32;
    let (r, g, b) = match (i as i32).rem_euclid(6) {
        0 => (i32::from(v), t, p),
        1 => (q, i32::from(v), p),
        2 => (p, i32::from(v), t),
        3 => (p, q, i32::from(v)),
        4 => (t, p, i32::from(v)),
        _ => (i32::from(v), p, q),
    };
    [clip8(r), clip8(g), clip8(b)]
}

fn convert_pixels_generic(source: &[u8], from: PixelMode, to: PixelMode) -> Vec<u8> {
    let (source_channels, target_channels) = (from.channels(), to.channels());
    let mut result = vec![0; source.len() / source_channels * target_channels];
    let chunk_pixels = crate::parallel::CHUNK_PIXELS;
    crate::parallel::chunks_mut(&mut result, chunk_pixels * target_channels, |chunk, dst| {
        let src = &source[chunk * chunk_pixels * source_channels..];
        for (pixel, out) in src.chunks_exact(source_channels).zip(dst.chunks_exact_mut(target_channels)) {
            convert_pixel_generic(pixel, out, from, to);
        }
    });
    result
}

#[inline(always)]
fn convert_pixel_generic(pixel: &[u8], out: &mut [u8], from: PixelMode, to: PixelMode) {
    let (r, g, b, a) = match from {
        PixelMode::One | PixelMode::L => (pixel[0], pixel[0], pixel[0], 255),
        PixelMode::La => (pixel[0], pixel[0], pixel[0], pixel[1]),
        PixelMode::Rgb => (pixel[0], pixel[1], pixel[2], 255),
        PixelMode::Rgba => (pixel[0], pixel[1], pixel[2], pixel[3]),
        PixelMode::Pa => (pixel[0], pixel[0], pixel[0], pixel[1]),
        _ => unreachable!("integer modes handled before byte conversion"),
    };
    let gray = crate::simd::pillow_luma(r, g, b);
    match to {
        PixelMode::One => out[0] = if gray >= 128 { 255 } else { 0 },
        PixelMode::L => out[0] = gray,
        PixelMode::La | PixelMode::Pa => out.copy_from_slice(&[gray, a]),
        PixelMode::Rgb => out.copy_from_slice(&[r, g, b]),
        PixelMode::Rgba => out.copy_from_slice(&[r, g, b, a]),
        _ => unreachable!("integer modes handled before byte conversion"),
    }
}

/// Resolve palette indices through a complete 256-entry table. Missing
/// entries read as opaque black, as before.
fn convert_palette(indices: &[u8], palette_mode: PixelMode, palette: &[u8], destination: PixelMode) -> PyResult<Vec<u8>> {
    let table;
    let palette = if palette_mode == destination {
        palette
    } else {
        let channels = palette_mode.channels();
        let mut complete = crate::buffers::reserved(256 * channels)?;
        for index in 0..256 {
            complete.extend_from_slice(palette.get(index * channels..(index + 1) * channels).unwrap_or(&[0, 0, 0, 255][..channels]));
        }
        table = convert_pixels(&complete, palette_mode, destination);
        &table
    };
    match destination.channels() {
        1 => expand_palette::<1>(indices, palette),
        2 => expand_palette::<2>(indices, palette),
        3 => expand_palette::<3>(indices, palette),
        4 => expand_palette::<4>(indices, palette),
        _ => unreachable!("validated palette destination"),
    }
}

fn expand_palette<const C: usize>(indices: &[u8], palette: &[u8]) -> PyResult<Vec<u8>> {
    let table: [[u8; C]; 256] = std::array::from_fn(|slot| {
        let mut color = [0; C];
        color.copy_from_slice(palette.get(slot * C..slot * C + C).unwrap_or(&[0, 0, 0, 255][..C]));
        color
    });
    let mut expanded = crate::buffers::reserved::<[u8; C]>(indices.len())?;
    crate::parallel::chunks_mut_above(
        &mut expanded.spare_capacity_mut()[..indices.len()],
        crate::parallel::CHUNK_PIXELS,
        crate::parallel::MIN_PARALLEL_BYTES,
        |chunk, dst| {
            let start = chunk * crate::parallel::CHUNK_PIXELS;
            for (pixel, &slot) in dst.iter_mut().zip(&indices[start..]) {
                pixel.write(table[usize::from(slot)]);
            }
        },
    );
    // SAFETY: Each reserved pixel was initialized by its palette lookup.
    unsafe { expanded.set_len(indices.len()) };
    Ok(expanded.into_flattened())
}

fn expand_pa(indices: &[u8], palette_mode: PixelMode, palette: &[u8], destination: PixelMode) -> PyResult<Vec<u8>> {
    let table: [[u8; 3]; 256] = std::array::from_fn(|slot| {
        let entry = slot * palette_mode.channels();
        palette.get(entry..entry + 3).unwrap_or(&[0, 0, 0]).try_into().unwrap()
    });
    let channels = destination.channels();
    let len = (indices.len() / 2).checked_mul(channels).ok_or_else(|| PyMemoryError::new_err("cannot allocate image"))?;
    let mut output = crate::buffers::reserved::<u8>(len)?;
    crate::parallel::chunks_mut_above(
        &mut output.spare_capacity_mut()[..len],
        crate::parallel::CHUNK_PIXELS * channels,
        crate::parallel::MIN_PARALLEL_BYTES,
        |chunk, dst| {
            let start = chunk * crate::parallel::CHUNK_PIXELS * 2;
            let source = &indices[start..start + dst.len() / channels * 2];
            for (pair, pixel) in source.as_chunks::<2>().0.iter().zip(dst.chunks_mut(channels)) {
                let rgb = table[pair[0] as usize];
                match destination {
                    PixelMode::L => {
                        pixel[0].write(crate::simd::pillow_luma(rgb[0], rgb[1], rgb[2]));
                    }
                    PixelMode::Rgb => {
                        pixel.write_copy_of_slice(&rgb);
                    }
                    PixelMode::Rgba => {
                        pixel.write_copy_of_slice(&[rgb[0], rgb[1], rgb[2], pair[1]]);
                    }
                    _ => unreachable!(),
                }
            }
        },
    );
    // SAFETY: Each reserved output byte was initialized above, including alpha.
    unsafe { output.set_len(len) };
    Ok(output)
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
    fn palette_table_conversion_matches_expanded_pixels() {
        let indices: Vec<u8> = (0..259).map(|i| (i * 71) as u8).collect();
        for mode in [PixelMode::L, PixelMode::Rgb, PixelMode::Rgba] {
            for count in [0, 1, 7, 256] {
                let palette: Vec<u8> = (0..count * mode.channels()).map(|i| (i * 37) as u8).collect();
                let expanded = match mode {
                    PixelMode::L => expand_palette::<1>(&indices, &palette),
                    PixelMode::Rgb => expand_palette::<3>(&indices, &palette),
                    _ => expand_palette::<4>(&indices, &palette),
                }
                .unwrap();
                for target in [
                    PixelMode::One,
                    PixelMode::L,
                    PixelMode::La,
                    PixelMode::Pa,
                    PixelMode::Rgb,
                    PixelMode::Rgba,
                    PixelMode::Hsv,
                    PixelMode::Cmyk,
                    PixelMode::YCbCr,
                ] {
                    assert_eq!(
                        convert_palette(&indices, mode, &palette, target).unwrap(),
                        convert_pixels(&expanded, mode, target),
                        "mode={mode:?}, target={target:?}, count={count}"
                    );
                }
            }
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
    fn lab_encoding_and_reference_colors() {
        let raw = [0, 0, 255, 255, 128, 127];
        assert_eq!(lab_raw_bytes(&raw), [0, 128, 127, 255, 0, 255]);
        assert_eq!(lab_raw_bytes(&lab_raw_bytes(&raw)), raw);
        assert_eq!(convert_lab(&[0, 0, 0, 255, 255, 255, 255, 0, 0], true).unwrap(), [0, 128, 128, 255, 128, 128, 138, 209, 198]);
        for count in [0, 1, 15, 16, 17, 33] {
            let raw: Vec<u8> = (0..count * 3 + 1).map(|i| (i * 37) as u8).collect();
            let expected: Vec<u8> = raw[1..].iter().enumerate().map(|(i, &v)| if i % 3 == 0 { v } else { v ^ 128 }).collect();
            assert_eq!(lab_raw_bytes(&raw[1..]), expected);
        }
    }

    #[test]
    fn parallel_lab_and_generic_conversion_match_serial_pixels() {
        for threads in [1, 4] {
            rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap().install(|| {
                let count = crate::parallel::CHUNK_PIXELS * 5 + 7;
                let rgb: Vec<u8> = (0..count * 3).map(|i| (i * 37 + i / 7) as u8).collect();
                for to_lab in [true, false] {
                    let converted = convert_lab(&rgb, to_lab).unwrap();
                    for (i, pixel) in rgb.as_chunks::<3>().0.iter().enumerate().step_by(4099) {
                        assert_eq!(convert_lab(pixel, to_lab).unwrap(), converted[i * 3..i * 3 + 3]);
                    }
                }
                for to in [PixelMode::One, PixelMode::La, PixelMode::Pa] {
                    let converted = convert_pixels_generic(&rgb, PixelMode::Rgb, to);
                    let channels = to.channels();
                    assert_eq!(converted.len(), count * channels);
                    for (pixel, out) in rgb.as_chunks::<3>().0.iter().zip(converted.chunks_exact(channels)) {
                        let gray = crate::simd::pillow_luma(pixel[0], pixel[1], pixel[2]);
                        let expected = if to == PixelMode::One { vec![if gray >= 128 { 255 } else { 0 }] } else { vec![gray, 255] };
                        assert_eq!(out, expected);
                    }
                }
            });
        }
    }

    #[test]
    pub(super) fn validates_buffer_length() {
        Python::initialize();
        Python::attach(|_| {
            let error = Image::from_pixels(2, 2, PixelMode::Rgb, vec![0; 11], None).expect_err("invalid buffer should fail");
            assert!(error.to_string().contains("expected 12 bytes"));
        });
    }

    #[test]
    pub(super) fn converts_like_pillow_fixed_point_luminance() {
        let rgb = [255, 0, 0, 0, 255, 0, 0, 0, 255, 12, 34, 56];
        assert_eq!(convert_pixels(&rgb, PixelMode::Rgb, PixelMode::L), [76, 150, 29, 30]);
    }

    #[test]
    pub(super) fn expands_and_drops_channels() {
        assert_eq!(convert_pixels(&[7, 23], PixelMode::L, PixelMode::Rgba), [7, 7, 7, 255, 23, 23, 23, 255]);
        assert_eq!(convert_pixels(&[1, 2, 3, 4], PixelMode::Rgba, PixelMode::Rgb), [1, 2, 3]);
    }

    #[test]
    fn converts_print_and_luma_chroma_channels() {
        assert_eq!(convert_pixels(&[0, 255, 255, 0, 0, 0, 0, 255], PixelMode::Cmyk, PixelMode::Rgb), [255, 0, 0, 0, 0, 0]);
        assert_eq!(convert_pixels(&[0, 128, 255], PixelMode::L, PixelMode::Cmyk), [0, 0, 0, 255, 0, 0, 0, 127, 0, 0, 0, 0]);
        assert_eq!(convert_pixels(&[255, 0, 0], PixelMode::Rgb, PixelMode::YCbCr), [76, 84, 255]);
        assert_eq!(convert_pixels(&[100, 30, 240], PixelMode::YCbCr, PixelMode::Rgb), [255, 53, 0]);
        assert_eq!(convert_pixels(&[1], PixelMode::One, PixelMode::YCbCr), [255, 128, 128]);
    }

    #[test]
    pub(super) fn expands_palette_indices_with_opaque_black_fallback() {
        let palette = [10, 20, 30, 40, 50, 60];
        assert_eq!(expand_palette::<3>(&[1, 0, 5], &palette).unwrap(), [40, 50, 60, 10, 20, 30, 0, 0, 0]);
        assert_eq!(expand_palette::<4>(&[0, 1], &[1, 2, 3, 4]).unwrap(), [1, 2, 3, 4, 0, 0, 0, 255]);
        assert!(expand_palette::<3>(&[], &palette).unwrap().is_empty());
    }

    #[test]
    pub(super) fn infers_supported_array_modes() {
        assert_eq!(array_mode(&[3, 5], "|u1").unwrap(), PixelMode::L);
        assert_eq!(array_mode(&[3, 5, 2], "|u1").unwrap(), PixelMode::La);
        assert_eq!(array_mode(&[3, 5, 3], "|u1").unwrap(), PixelMode::Rgb);
        assert_eq!(array_mode(&[3, 5, 4], "|u1").unwrap(), PixelMode::Rgba);
        assert_eq!(array_mode(&[3, 5], "<f4").unwrap(), PixelMode::F);
        for typestr in ["<u2", ">u2", "<f4", "<f8"] {
            assert!(array_mode(&[3, 5, 2], typestr).is_err());
        }
    }
}
