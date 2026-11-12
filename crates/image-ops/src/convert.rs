use blanket_convert_kernels::{alpha, simd};
use blanket_core::raster::alloc_pixels;
use blanket_core::{Image, PixelMode};
use pyo3::exceptions::{PyMemoryError, PyTypeError, PyValueError};
use pyo3::prelude::*;

#[pyfunction(name = "image_convert_palette_alpha", signature = (source_image, mode, alpha=None, index=None, bit_depth=None))]
pub fn convert_palette_alpha(
    py: Python<'_>, source_image: &Image, mode: &str, alpha: Option<std::borrow::Cow<'_, [u8]>>, index: Option<u8>, bit_depth: Option<u8>,
) -> PyResult<Image> {
    if source_image.pixel_mode() != PixelMode::L || !matches!(mode, "LA" | "RGBA") {
        return Err(PyValueError::new_err("indexed alpha conversion requires LA or RGBA output"));
    }
    let indices = source_image.pixel_data()?;
    let palette = source_image.alpha_palette(alpha.as_deref(), index)?;
    let destination = PixelMode::parse(mode)?;
    let pixels = py.detach(|| {
        if destination == PixelMode::La {
            let table = convert_pixels(&palette, PixelMode::Rgba, PixelMode::La);
            expand_palette::<2>(indices, &table)
        } else {
            expand_palette::<4>(indices, &palette)
        }
    })?;
    let result = Image::from_pixels(source_image.width(), source_image.height(), destination, pixels, None)?;
    if bit_depth.is_some_and(|depth| depth != 8) { convert(py, &result, mode, bit_depth) } else { Ok(result) }
}

#[pyfunction(name = "image_convert_matrix")]
pub fn convert_matrix(py: Python<'_>, source_image: &Image, mode: &str, matrix: Vec<f32>) -> PyResult<Image> {
    if !matches!(mode, "L" | "RGB") {
        return Err(PyValueError::new_err("illegal conversion"));
    }
    if !matches!(matrix.len(), 4 | 12) {
        return Err(PyTypeError::new_err("matrix must be tuple of length 4 or 12"));
    }
    if source_image.pixel_mode() != PixelMode::Rgb || source_image.bit_depth() != 8 {
        return Err(PyValueError::new_err("image has wrong mode"));
    }
    let destination = PixelMode::parse(mode)?;
    let channels = destination.channels();
    if matrix.len() < channels * 4 {
        return Err(PyTypeError::new_err("matrix must be tuple of length 12"));
    }
    let source = source_image.pixel_data()?;
    let length = source.len() / 3 * channels;
    let mut pixels = blanket_core::buffers::reserved(length)?;
    py.detach(|| {
        blanket_kernels::parallel::chunks_mut_above(&mut pixels.spare_capacity_mut()[..length], 16 * 1024 * channels, 256 * 1024, |chunk, dst| {
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
    Image::from_pixels(source_image.width(), source_image.height(), destination, pixels, None)
}

#[pyfunction(name = "image_convert_transparent_alpha")]
pub fn convert_transparent_alpha(py: Python<'_>, source_image: &Image, mode: &str, key: Vec<u8>) -> PyResult<Image> {
    if !matches!(source_image.pixel_mode(), PixelMode::L | PixelMode::Rgb)
        || key.len() != source_image.pixel_mode().channels()
        || source_image.bit_depth() != 8
    {
        return Err(PyValueError::new_err("invalid transparency color"));
    }
    let destination = if matches!(mode, "LA" | "La") { "LA" } else { "RGBA" };
    let mut image = convert(py, source_image, destination, None)?;
    let channels = image.pixel_mode().channels();
    let source = source_image.pixel_data()?;
    let output = image.pixels_mut().unwrap();
    py.detach(|| {
        for (pixel, output) in source.chunks_exact(key.len()).zip(output.chunks_exact_mut(channels)) {
            output[channels - 1] = if pixel == key { 0 } else { 255 };
            if pixel == key && source_image.pixel_mode() == PixelMode::Rgb && destination == "LA" {
                output[0] = key[0];
            }
        }
    });
    if mode == destination { Ok(image) } else { convert(py, &image, mode, None) }
}

#[pyfunction(name = "image_convert_dithered_bilevel")]
pub fn convert_dithered_bilevel(py: Python<'_>, source_image: &Image) -> PyResult<Image> {
    if !matches!(source_image.pixel_mode(), PixelMode::L | PixelMode::Rgb) || source_image.bit_depth() != 8 {
        return Err(PyValueError::new_err("conversion not supported"));
    }
    let source = source_image.pixel_data()?;
    let channels = source_image.pixel_mode().channels();
    let width = source_image.width() as usize;
    let mut pixels = alloc_pixels(width * source_image.height() as usize)?;
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
    Image::from_pixels(source_image.width(), source_image.height(), PixelMode::One, pixels, None)
}

#[pyfunction(name = "image_convert", signature = (source_image, mode, bit_depth=None))]
pub fn convert(py: Python<'_>, source_image: &Image, mode: &str, bit_depth: Option<u8>) -> PyResult<Image> {
    if mode == "P" && source_image.palette().is_some() {
        return source_image.copy(py);
    }
    let destination = PixelMode::parse(mode)?;
    if matches!(source_image.pixel_mode(), PixelMode::PremulLa | PixelMode::PremulRgba | PixelMode::Rgbx)
        || matches!(destination, PixelMode::PremulLa | PixelMode::PremulRgba | PixelMode::Rgbx)
    {
        if matches!(destination, PixelMode::PremulLa | PixelMode::PremulRgba | PixelMode::Rgbx) && bit_depth.is_some_and(|depth| depth != 8) {
            return Err(PyValueError::new_err("this mode requires 8-bit pixels"));
        }
        if source_image.pixel_mode() == destination {
            return source_image.copy(py);
        }
        if matches!((source_image.pixel_mode(), destination), (PixelMode::Rgbx, PixelMode::Rgba) | (PixelMode::Rgba, PixelMode::Rgbx))
            && source_image.bit_depth() == 8
            && bit_depth.is_none_or(|depth| depth == 8)
        {
            let source = source_image.raw_data()?;
            let pixels = py.detach(|| simd::opaque_rgba(source));
            return Image::from_pixels(source_image.width(), source_image.height(), destination, pixels, None);
        }
        if matches!(source_image.pixel_mode(), PixelMode::PremulLa | PixelMode::PremulRgba) {
            let straight = if source_image.pixel_mode() == PixelMode::PremulLa { PixelMode::La } else { PixelMode::Rgba };
            if source_image.pixel_mode() == PixelMode::PremulLa && destination != PixelMode::La {
                return Err(PyValueError::new_err("conversion from La requires LA"));
            }
            let source = source_image.raw_data()?;
            let pixels = py.detach(|| {
                let mut pixels = vec![0; source.len()];
                if straight == PixelMode::La {
                    alpha::convert::<2>(source, &mut pixels, false);
                } else {
                    alpha::convert::<4>(source, &mut pixels, false);
                }
                pixels
            });
            let image = Image::from_pixels(source_image.width(), source_image.height(), straight, pixels, None)?;
            return if destination == straight {
                if bit_depth.is_none_or(|depth| depth == 8) { Ok(image) } else { convert(py, &image, mode, bit_depth) }
            } else {
                convert(py, &convert(py, &image, "RGB", None)?, mode, bit_depth)
            };
        }
        if matches!(destination, PixelMode::PremulLa | PixelMode::PremulRgba) {
            let straight = if destination == PixelMode::PremulLa { PixelMode::La } else { PixelMode::Rgba };
            let converted;
            let image = if source_image.pixel_mode() == straight {
                source_image
            } else if source_image.pixel_mode().channels() >= 3 || source_image.palette().is_some() {
                converted = convert(py, &convert(py, source_image, "RGB", Some(8))?, straight.as_str(), None)?;
                &converted
            } else {
                return Err(PyValueError::new_err("unsupported premultiplied conversion"));
            };
            let source = image.raw_data()?;
            let pixels = py.detach(|| {
                let mut pixels = vec![0; source.len()];
                if straight == PixelMode::La {
                    alpha::convert::<2>(source, &mut pixels, true);
                } else {
                    alpha::convert::<4>(source, &mut pixels, true);
                }
                pixels
            });
            return Image::from_pixels(source_image.width(), source_image.height(), destination, pixels, None);
        }
        if source_image.pixel_mode() == PixelMode::Rgbx {
            let source = source_image.raw_data()?;
            if bit_depth.is_none_or(|depth| depth == 8) && matches!(destination, PixelMode::L | PixelMode::Rgb) {
                let pixels = py.detach(|| simd::convert(source, PixelMode::Rgba, destination));
                return Image::from_pixels(source_image.width(), source_image.height(), destination, pixels, None);
            }
            let pixels = py.detach(|| simd::convert(source, PixelMode::Rgba, PixelMode::Rgb));
            let image = Image::from_pixels(source_image.width(), source_image.height(), PixelMode::Rgb, pixels, None)?;
            return convert(py, &image, mode, bit_depth);
        }
        let mut image = if source_image.pixel_mode() == PixelMode::La || source_image.pixel_mode() == PixelMode::Rgb && source_image.bit_depth() == 8
        {
            convert(py, source_image, "RGBA", None)?
        } else {
            convert(py, &convert(py, source_image, "RGB", Some(8))?, "RGBA", None)?
        };
        image.reinterpret_mode(destination)?;
        return Ok(image);
    }
    if matches!(source_image.pixel_mode(), PixelMode::RgbF | PixelMode::RgbaF) || matches!(destination, PixelMode::RgbF | PixelMode::RgbaF) {
        return py.detach(|| crate::float::convert(source_image, destination, bit_depth));
    }
    if source_image.pixel_mode() == PixelMode::Lab && destination != PixelMode::Lab {
        let source = source_image.pixel_data()?;
        let pixels = py.detach(|| convert_lab(source, false))?;
        let rgb = Image::from_pixels(source_image.width(), source_image.height(), PixelMode::Rgb, pixels, None)?;
        return if destination == PixelMode::Rgb && bit_depth.is_none_or(|depth| depth == 8) { Ok(rgb) } else { convert(py, &rgb, mode, bit_depth) };
    }
    if destination == PixelMode::Lab && source_image.pixel_mode() != PixelMode::Lab {
        if bit_depth.is_some_and(|depth| depth != 8) {
            return Err(PyValueError::new_err("this mode requires 8-bit pixels"));
        }
        let converted;
        let rgb = if source_image.pixel_mode() == PixelMode::Rgb && source_image.bit_depth() == 8 {
            source_image
        } else {
            converted = convert(py, source_image, "RGB", Some(8))?;
            &converted
        };
        let source = rgb.pixel_data()?;
        let pixels = py.detach(|| convert_lab(source, true))?;
        return Image::from_pixels(source_image.width(), source_image.height(), PixelMode::Lab, pixels, None);
    }
    if destination == PixelMode::F
        && source_image.bit_depth() == 8
        && matches!(source_image.pixel_mode(), PixelMode::Rgb | PixelMode::Rgba | PixelMode::Cmyk | PixelMode::YCbCr)
    {
        if bit_depth.is_some_and(|depth| depth != 32) {
            return Err(PyValueError::new_err("bit_depth does not match float mode"));
        }
        let source = source_image.pixel_data()?;
        let pixels = py.detach(|| {
            let luma = |r, g, b| ((f64::from(r) * 299.0 + f64::from(g) * 587.0 + f64::from(b) * 114.0) as f32 / 1000.0).to_le_bytes();
            if source_image.pixel_mode() == PixelMode::Rgb {
                return map_pixels::<3, 4>(source, |p| luma(p[0], p[1], p[2]));
            }
            if source_image.pixel_mode() == PixelMode::Rgba {
                return map_pixels::<4, 4>(source, |p| luma(p[0], p[1], p[2]));
            }
            let rgb = convert_pixels(source, source_image.pixel_mode(), PixelMode::Rgb);
            rgb.as_chunks::<3>()
                .0
                .iter()
                .flat_map(|p| ((f64::from(p[0]) * 299.0 + f64::from(p[1]) * 587.0 + f64::from(p[2]) * 114.0) as f32 / 1000.0).to_le_bytes())
                .collect()
        });
        return Image::from_float_bytes(source_image.width(), source_image.height(), pixels);
    }
    if source_image.pixel_mode() == PixelMode::F || destination == PixelMode::F {
        if destination == PixelMode::F {
            if bit_depth.is_some_and(|depth| depth != 32) {
                return Err(PyValueError::new_err("bit_depth does not match float mode"));
            }
            if source_image.pixel_mode() == PixelMode::F {
                return source_image.copy(py);
            }
            let converted;
            let source = if source_image.pixel_mode().is_integer() || source_image.pixel_mode() == PixelMode::L && source_image.bit_depth() > 8 {
                source_image
            } else {
                converted = convert(py, source_image, "L", Some(8))?;
                &converted
            };
            let data = source.raw_data()?;
            let pixels = py.detach(|| {
                if source.pixel_mode().is_integer() {
                    simd::integer_to_float(data, source.pixel_mode())
                } else if source.bit_depth() > 8 {
                    simd::integer_to_float(data, PixelMode::I16)
                } else {
                    simd::gray_to_float(data)
                }
            });
            return Image::from_float_bytes(source_image.width(), source_image.height(), pixels);
        }
        if destination == PixelMode::L && bit_depth == Some(16) {
            let source = source_image.raw_data()?;
            let pixels = py.detach(|| simd::float_to_u16(source, false));
            return Image::from_sample_bytes(source_image.width(), source_image.height(), PixelMode::L, pixels, 16, None);
        }
        let data = source_image.raw_data()?;
        if destination == PixelMode::I {
            let pixels = py.detach(|| simd::float_to_integer(data));
            return Image::from_integer_bytes(source_image.width(), source_image.height(), destination, pixels);
        }
        if destination.is_integer() {
            let pixels = py.detach(|| simd::float_to_u16(data, destination.is_big_endian()));
            return Image::from_integer_bytes(source_image.width(), source_image.height(), destination, pixels);
        }
        let pixels = py.detach(|| simd::float_to_l(data));
        let gray = Image::from_pixels(source_image.width(), source_image.height(), PixelMode::L, pixels, None)?;
        if destination == PixelMode::L && bit_depth.is_none_or(|depth| depth == 8) {
            return Ok(gray);
        }
        return convert(py, &gray, mode, bit_depth);
    }
    if matches!(source_image.pixel_mode(), PixelMode::Cmyk | PixelMode::YCbCr) && destination != source_image.pixel_mode() {
        let source = source_image.pixel_data()?;
        if source_image.pixel_mode() == PixelMode::YCbCr && matches!(destination, PixelMode::L | PixelMode::La) {
            let luma = py.detach(|| simd::extract_channel::<3>(source, 0));
            let gray = Image::from_pixels(source_image.width(), source_image.height(), PixelMode::L, luma, None)?;
            return if destination == PixelMode::L && bit_depth.is_none_or(|depth| depth == 8) {
                Ok(gray)
            } else {
                convert(py, &gray, mode, bit_depth)
            };
        }
        if bit_depth.is_none_or(|depth| depth == 8) && matches!(destination, PixelMode::L | PixelMode::La | PixelMode::Rgb | PixelMode::Rgba) {
            let pixels = py.detach(|| convert_pixels(source, source_image.pixel_mode(), destination));
            return Image::from_pixels(source_image.width(), source_image.height(), destination, pixels, None);
        }
        let pixels = py.detach(|| convert_pixels(source, source_image.pixel_mode(), PixelMode::Rgb));
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
            return Image::from_pixels(source_image.width(), source_image.height(), destination, pixels, None);
        }
        let rgb = Image::from_pixels(source_image.width(), source_image.height(), PixelMode::Rgb, pixels, None)?;
        if destination == PixelMode::Rgb && bit_depth.is_none_or(|depth| depth == 8) {
            return Ok(rgb);
        }
        return convert(py, &rgb, mode, bit_depth);
    }
    if matches!(destination, PixelMode::Cmyk | PixelMode::YCbCr) && source_image.bit_depth() > 8 && !source_image.pixel_mode().is_wide_scalar() {
        let intermediate = if source_image.pixel_mode() == PixelMode::L { "L" } else { "RGB" };
        return convert(py, &convert(py, source_image, intermediate, Some(8))?, mode, bit_depth);
    }
    if source_image.pixel_mode() == PixelMode::Hsv && destination != PixelMode::Hsv {
        if bit_depth.is_none_or(|depth| depth == 8) && matches!(destination, PixelMode::L | PixelMode::Rgb | PixelMode::Rgba) {
            let source = source_image.pixel_data()?;
            let pixels = py.detach(|| convert_pixels(source, PixelMode::Hsv, destination));
            return Image::from_pixels(source_image.width(), source_image.height(), destination, pixels, None);
        }
        let source = source_image.pixel_data()?;
        let pixels = py.detach(|| hsv_to_rgb_bytes(source));
        let rgb = Image::from_pixels(source_image.width(), source_image.height(), PixelMode::Rgb, pixels, None)?;
        return if destination == PixelMode::Rgb { Ok(rgb) } else { convert(py, &rgb, mode, bit_depth) };
    }
    if destination == PixelMode::Hsv && source_image.bit_depth() > 8 {
        return convert(py, &convert(py, source_image, "RGB", Some(8))?, "HSV", None);
    }
    if source_image.pixel_mode().is_integer() || destination.is_integer() {
        if let Some(depth) = bit_depth
            && depth != (destination.sample_bytes() << 3) as u8
            && destination.is_integer()
        {
            return Err(PyValueError::new_err("bit_depth does not match integer mode"));
        }
        if source_image.pixel_mode() == destination {
            return source_image.copy(py);
        }
        if destination.is_integer() {
            let converted;
            let source = if source_image.palette().is_none() && (source_image.pixel_mode().is_integer() || source_image.pixel_mode() == PixelMode::L)
            {
                source_image
            } else {
                converted = convert(py, source_image, "L", Some(8))?;
                &converted
            };
            let data = source.raw_data()?;
            if source.pixel_mode().is_integer() {
                let output = py.detach(|| simd::convert_integer(data, source.pixel_mode(), destination));
                return Image::from_integer_bytes(source_image.width(), source_image.height(), destination, output);
            }
            let output = py.detach(|| {
                if source.bit_depth() > 8 {
                    simd::convert_integer(data, PixelMode::I16, destination)
                } else {
                    simd::gray_to_integer(data, destination)
                }
            });
            return Image::from_integer_bytes(source_image.width(), source_image.height(), destination, output);
        }
        let data = source_image.raw_data()?;
        if destination == PixelMode::L && bit_depth.is_some_and(|depth| depth > 8) {
            let pixels = py.detach(|| simd::convert_integer(data, source_image.pixel_mode(), PixelMode::I16));
            let gray = Image::from_sample_bytes(source_image.width(), source_image.height(), PixelMode::L, pixels, 16, None)?;
            return convert(py, &gray, mode, bit_depth);
        }
        if matches!(destination, PixelMode::Rgb | PixelMode::Rgba) {
            let pixels = py.detach(|| simd::integer_to_color(data, source_image.pixel_mode(), destination.channels()));
            return Image::from_pixels(source_image.width(), source_image.height(), destination, pixels, None);
        }
        let clipped = py.detach(|| simd::integer_to_l(data, source_image.pixel_mode()));
        let grayscale = Image::from_pixels(source_image.width(), source_image.height(), PixelMode::L, clipped, None)?;
        if destination == PixelMode::L && bit_depth.is_none_or(|depth| depth == 8) {
            return Ok(grayscale);
        }
        return convert(py, &grayscale, mode, bit_depth);
    }
    let depth = bit_depth.unwrap_or_else(|| source_image.bit_depth());
    if !matches!(depth, 8 | 10 | 12 | 16) {
        return Err(PyValueError::new_err("bit_depth must be 8, 10, 12, or 16"));
    }
    if (source_image.pixel_mode() != PixelMode::L && source_image.pixel_mode() != PixelMode::Rgb && source_image.pixel_mode() != PixelMode::Rgba
        || destination != PixelMode::L && destination != PixelMode::Rgb && destination != PixelMode::Rgba)
        && depth != 8
    {
        return Err(PyValueError::new_err("this mode requires 8-bit pixels"));
    }
    if source_image.bit_depth() > 8 || depth > 8 {
        if source_image.palette().is_some() {
            let expanded = convert(py, source_image, mode, Some(8))?;
            return convert(py, &expanded, mode, Some(depth));
        }
        let bytes = source_image.raw_data()?;
        if source_image.pixel_mode() == destination {
            let output = py.detach(|| simd::rescale_samples(bytes, source_image.bit_depth(), depth));
            return if depth == 8 {
                Image::from_pixels(source_image.width(), source_image.height(), destination, output, None)
            } else {
                Image::from_sample_bytes(source_image.width(), source_image.height(), destination, output, depth, None)
            };
        }
        let output = py.detach(|| simd::convert_samples(bytes, source_image.pixel_mode(), destination, source_image.bit_depth(), depth));
        return if depth == 8 {
            Image::from_pixels(source_image.width(), source_image.height(), destination, output, None)
        } else {
            Image::from_sample_bytes(source_image.width(), source_image.height(), destination, output, depth, None)
        };
    }
    if let Some((palette_mode, palette)) = source_image.palette() {
        let indices = source_image.pixel_data()?;
        let converted = py.detach(|| {
            if source_image.pixel_mode() == PixelMode::Pa {
                if matches!(destination, PixelMode::L | PixelMode::Rgb | PixelMode::Rgba) {
                    return expand_pa(indices, *palette_mode, palette, destination);
                }
                let len = indices.len().checked_mul(2).ok_or_else(|| PyMemoryError::new_err("cannot allocate image"))?;
                let mut rgba = blanket_core::buffers::reserved(len)?;
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
        return Image::from_pixels(source_image.width(), source_image.height(), destination, converted, None);
    }
    let source = source_image.pixel_mode();
    let pixels = source_image.pixel_data()?;
    if source == destination {
        return source_image.copy(py);
    }
    let converted = py.detach(|| convert_pixels(pixels, source, destination));
    Image::from_pixels(source_image.width(), source_image.height(), destination, converted, None)
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
                (_, PixelMode::Cmyk) => simd::gray_to_cmyk(source),
                _ => simd::gray_to_ycbcr(source),
            };
        }
        if from == PixelMode::Rgba {
            return if to == PixelMode::Cmyk {
                simd::color_to_cmyk::<4>(source)
            } else {
                map_pixels::<4, 3>(source, |p| rgb_to_ycbcr([p[0], p[1], p[2]]))
            };
        }
        if from == PixelMode::Rgb {
            return if to == PixelMode::Cmyk { simd::color_to_cmyk::<3>(source) } else { map_pixels::<3, 3>(source, rgb_to_ycbcr) };
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
            return simd::convert_bilevel(source, to);
        }
        let gray = simd::bilevel_to_l(source);
        return if to == PixelMode::L { gray } else { convert_pixels(&gray, PixelMode::L, to) };
    }
    if from == PixelMode::La && matches!(to, PixelMode::L | PixelMode::Rgb | PixelMode::Rgba) {
        return simd::convert_la(source, to);
    }
    if from == PixelMode::Hsv {
        if to == PixelMode::L {
            return map_pixels::<3, 1>(source, |p| {
                let [r, g, b] = hsv_to_rgb(p[0], p[1], p[2]);
                [simd::pillow_luma(r, g, b)]
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
            PixelMode::One | PixelMode::L => simd::gray_to_hsv::<1>(source),
            PixelMode::La => simd::gray_to_hsv::<2>(source),
            PixelMode::Rgb => map_pixels::<3, 3>(source, |p| rgb_to_hsv(p[0], p[1], p[2])),
            PixelMode::Rgba => map_pixels::<4, 3>(source, |p| rgb_to_hsv(p[0], p[1], p[2])),
            _ => {
                let rgb = convert_pixels(source, from, PixelMode::Rgb);
                convert_pixels(&rgb, PixelMode::Rgb, PixelMode::Hsv)
            }
        };
    }
    if matches!(from, PixelMode::L | PixelMode::Rgb | PixelMode::Rgba) && matches!(to, PixelMode::L | PixelMode::Rgb | PixelMode::Rgba) {
        return simd::convert(source, from, to);
    }
    convert_pixels_generic(source, from, to)
}

fn convert_lab(source: &[u8], to_lab: bool) -> PyResult<Vec<u8>> {
    use lcms2::{CIExyY, DisallowCache, Flags, GlobalContext, Intent, PixelFormat, Profile, Transform};
    type LabTransform = Transform<u8, u8, GlobalContext, DisallowCache>;
    // NO_CACHE makes the transform Sync; the cache only memoizes repeated pixels.
    fn build(to_lab: bool) -> Result<LabTransform, String> {
        let context = GlobalContext::new();
        let srgb = Profile::new_srgb_context(&context);
        let lab = Profile::new_lab4_context(&context, CIExyY::d50()).map_err(|error| error.to_string())?;
        let (input, input_format, output, output_format) =
            if to_lab { (&srgb, PixelFormat::RGB_8, &lab, PixelFormat::Lab_8) } else { (&lab, PixelFormat::Lab_8, &srgb, PixelFormat::RGB_8) };
        Transform::new_flags_context(context, input, input_format, output, output_format, Intent::Perceptual, Flags::NO_CACHE)
            .map_err(|error| error.to_string())
    }
    static TRANSFORMS: [std::sync::OnceLock<Result<LabTransform, String>>; 2] = [const { std::sync::OnceLock::new() }; 2];
    let transform = TRANSFORMS[usize::from(to_lab)].get_or_init(|| build(to_lab)).as_ref().map_err(|error| PyValueError::new_err(error.clone()))?;
    let mut pixels = blanket_core::buffers::reserved(source.len())?;
    pixels.resize(source.len(), 0_u8);
    let chunk = blanket_kernels::parallel::CHUNK_PIXELS * 3;
    blanket_kernels::parallel::chunks_mut_above(&mut pixels, chunk, 64 << 10, |i, dst| {
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
    blanket_kernels::parallel::chunks_mut(&mut output, blanket_kernels::parallel::CHUNK_PIXELS * D, |chunk, dst| {
        let start = chunk * blanket_kernels::parallel::CHUNK_PIXELS * S;
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
            [simd::pillow_luma(r, g, b)]
        }),
        PixelMode::La => map_pixels::<S, 2>(source, |p| {
            let [r, g, b] = rgb(p);
            [simd::pillow_luma(r, g, b), 255]
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
    let chunk_pixels = blanket_kernels::parallel::CHUNK_PIXELS;
    blanket_kernels::parallel::chunks_mut(&mut result, chunk_pixels * target_channels, |chunk, dst| {
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
    let gray = simd::pillow_luma(r, g, b);
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
        let mut complete = blanket_core::buffers::reserved(256 * channels)?;
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
    let mut expanded = blanket_core::buffers::reserved::<[u8; C]>(indices.len())?;
    blanket_kernels::parallel::chunks_mut_above(
        &mut expanded.spare_capacity_mut()[..indices.len()],
        blanket_kernels::parallel::CHUNK_PIXELS,
        blanket_kernels::parallel::MIN_PARALLEL_BYTES,
        |chunk, dst| {
            let start = chunk * blanket_kernels::parallel::CHUNK_PIXELS;
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
    let mut output = blanket_core::buffers::reserved::<u8>(len)?;
    blanket_kernels::parallel::chunks_mut_above(
        &mut output.spare_capacity_mut()[..len],
        blanket_kernels::parallel::CHUNK_PIXELS * channels,
        blanket_kernels::parallel::MIN_PARALLEL_BYTES,
        |chunk, dst| {
            let start = chunk * blanket_kernels::parallel::CHUNK_PIXELS * 2;
            let source = &indices[start..start + dst.len() / channels * 2];
            for (pair, pixel) in source.as_chunks::<2>().0.iter().zip(dst.chunks_mut(channels)) {
                let rgb = table[pair[0] as usize];
                match destination {
                    PixelMode::L => {
                        pixel[0].write(simd::pillow_luma(rgb[0], rgb[1], rgb[2]));
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
    use blanket_core::raster::lab_raw_bytes;

    use super::*;
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
                let count = blanket_kernels::parallel::CHUNK_PIXELS * 5 + 7;
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
                        let gray = simd::pillow_luma(pixel[0], pixel[1], pixel[2]);
                        let expected = if to == PixelMode::One { vec![if gray >= 128 { 255 } else { 0 }] } else { vec![gray, 255] };
                        assert_eq!(out, expected);
                    }
                }
            });
        }
    }
    #[test]
    fn converts_like_pillow_fixed_point_luminance() {
        let rgb = [255, 0, 0, 0, 255, 0, 0, 0, 255, 12, 34, 56];
        assert_eq!(convert_pixels(&rgb, PixelMode::Rgb, PixelMode::L), [76, 150, 29, 30]);
    }
    #[test]
    fn expands_and_drops_channels() {
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
    fn expands_palette_indices_with_opaque_black_fallback() {
        let palette = [10, 20, 30, 40, 50, 60];
        assert_eq!(expand_palette::<3>(&[1, 0, 5], &palette).unwrap(), [40, 50, 60, 10, 20, 30, 0, 0, 0]);
        assert_eq!(expand_palette::<4>(&[0, 1], &[1, 2, 3, 4]).unwrap(), [1, 2, 3, 4, 0, 0, 0, 255]);
        assert!(expand_palette::<3>(&[], &palette).unwrap().is_empty());
    }
}
