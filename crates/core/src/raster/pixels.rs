use pyo3::exceptions::{PyMemoryError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyInt, PyList, PyTuple};

use super::{Image, PixelMode};

// Golden-ratio multiplier spreads packed pixel keys across the high hash bits.
const PIXEL_HASH_MULTIPLIER: u32 = 0x9e37_79b9;
// A direct-mapped cache reuses Python tuples for repeated pixel values.
const PIXEL_CACHE_BITS: u32 = 8;
const PIXEL_CACHE_SIZE: usize = 1 << PIXEL_CACHE_BITS;

pub(super) fn parse_pixel(value: &Bound<'_, PyAny>, channels: usize) -> PyResult<[u8; 4]> {
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

pub(super) fn pixel_sequence<'py>(
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

pub(super) fn packed_pixel_data<'py, const C: usize>(py: Python<'py>, data: &[u8], tuple: bool) -> PyResult<Bound<'py, PyAny>> {
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

pub(super) fn wide_pixel_data<'py>(py: Python<'py>, image: &Image, tuple: bool) -> PyResult<Bound<'py, PyAny>> {
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

pub(super) fn put_sample_pixel(pixels: &mut [u8], value: &Bound<'_, PyAny>, channels: usize, maximum: u16, scale: f64, offset: f64) -> PyResult<()> {
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

pub(super) fn put_pixels<const C: usize>(pixels: &mut [u8], data: &Bound<'_, PyAny>, length: usize, scale: f64, offset: f64) -> PyResult<()> {
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

pub(super) fn float_value(bytes: &[u8]) -> f32 {
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
