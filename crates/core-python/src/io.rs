use blanket_core::raster::{alloc_pixels, integer_value, lab_raw_bytes, lab_raw_bytes_mut, pack_bilevel, pack_bilevel_into, unpack_bilevel};
use blanket_core::{Image, PixelMode};
use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{PyIndexError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyMemoryView, PySlice, PyTuple};

#[pyfunction(name = "image_tobytes", signature = (source_image, rawmode=None, stride=0, orientation=1))]
pub(crate) fn tobytes(py: Python<'_>, source_image: &Image, rawmode: Option<&str>, stride: usize, orientation: i8) -> PyResult<Py<PyBytes>> {
    let pixels = source_image.raw_data()?;
    if stride == 0
        && orientation == 1
        && rawmode.is_none_or(|mode| mode == source_image.pixel_mode().as_str() || source_image.pixel_mode() == PixelMode::L && mode == "P")
    {
        if source_image.pixel_mode() == PixelMode::One {
            let width = source_image.width() as usize;
            return PyBytes::new_with(py, width.div_ceil(8) * source_image.height() as usize, |output| {
                py.detach(|| pack_bilevel_into(pixels, width, output));
                Ok(())
            })
            .map(Bound::unbind);
        }
        if source_image.pixel_mode() == PixelMode::Lab {
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
    let source = if source_image.pixel_mode() == PixelMode::Lab {
        packed = py.detach(|| lab_raw_bytes(pixels));
        packed.as_slice()
    } else if source_image.pixel_mode() == PixelMode::One {
        packed = py.detach(|| pack_bilevel(pixels, source_image.width() as usize, source_image.height() as usize));
        packed.as_slice()
    } else {
        pixels
    };
    crate::raw::encode(py, source, source_image.pixel_mode(), source_image.size(), source_image.bit_depth(), rawmode, stride, orientation)
        .map(Bound::unbind)
}
#[pyfunction(signature = (mode, size, data, bit_depth=8, rawmode=None, stride=0, orientation=1, decoder="raw"))]
pub(crate) fn frombytes(
    py: Python<'_>, mode: &str, size: (u32, u32), data: &[u8], bit_depth: u8, rawmode: Option<&str>, stride: usize, orientation: i8, decoder: &str,
) -> PyResult<Image> {
    let mode = PixelMode::parse(mode)?;
    py.detach(|| {
        let input = crate::raw::decode(data, decoder)?;
        from_raw_input(mode, size, input, bit_depth, rawmode, stride, orientation)
    })
}

#[pyfunction(signature = (mode, size, data, bit_depth=8, rawmode=None, stride=0, orientation=1))]
pub(crate) fn frombuffer_copy(
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
pub(crate) fn mapped_getpixel(
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
        return blanket_core::buffers::copy(bytes.as_bytes());
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
        return blanket_core::buffers::copy(bytes.cast::<PyBytes>()?.as_bytes());
    }
    let bytes = obj.call_method0("tobytes")?;
    blanket_core::buffers::copy(bytes.cast::<PyBytes>()?.as_bytes())
}

#[pyfunction(signature = (obj, mode = None, bit_depth = None, interface = None))]
pub(crate) fn fromarray(
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
        return if source == mode { Ok(input) } else { blanket_image_ops::convert::convert(py, &input, mode.as_str(), None) };
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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn infers_supported_array_modes() {
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
