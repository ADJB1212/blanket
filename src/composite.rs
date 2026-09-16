use pyo3::exceptions::{PyMemoryError, PyValueError};
use pyo3::prelude::*;

use crate::raster::{Image, PixelMode};

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(image_new, module)?)?;
    module.add_function(wrap_pyfunction!(image_paste, module)?)?;
    module.add_function(wrap_pyfunction!(image_alpha_composite, module)?)?;
    module.add_function(wrap_pyfunction!(image_putalpha, module)?)?;
    Ok(())
}

#[pyfunction]
fn image_new(py: Python<'_>, mode: &str, size: (u32, u32), color: Vec<u8>) -> PyResult<Image> {
    let mode = PixelMode::parse(mode)?;
    if color.len() != mode.channels() {
        return Err(PyValueError::new_err("wrong number of color channels"));
    }
    let len = (size.0 as usize)
        .checked_mul(size.1 as usize)
        .and_then(|n| n.checked_mul(color.len()))
        .ok_or_else(|| PyMemoryError::new_err("image dimensions are too large"))?;
    let mut pixels = Vec::new();
    pixels
        .try_reserve_exact(len)
        .map_err(|_| PyMemoryError::new_err("cannot allocate image"))?;
    py.detach(|| {
        pixels.resize(len, 0);
        for pixel in pixels.chunks_exact_mut(color.len()) {
            pixel.copy_from_slice(&color);
        }
    });
    Image::from_pixels(size.0, size.1, mode, pixels, None)
}

#[pyfunction]
#[pyo3(signature = (image, source, position, mask=None, fill=false))]
fn image_paste(py: Python<'_>, image: &Image, source: &Image, position: (i64, i64), mask: Option<&Image>, fill: bool) -> PyResult<Image> {
    image.pixel_data()?;
    let mut output = image.clone();
    let pixels = output.pixels.as_mut().expect("validated open image");
    let source_pixels = source.pixel_data()?;
    if image.mode != source.mode {
        return Err(PyValueError::new_err("images do not match"));
    }
    let mask_pixels = if let Some(mask) = mask {
        if mask.palette.is_some() || !matches!(mask.mode, PixelMode::L | PixelMode::Rgba) {
            return Err(PyValueError::new_err("bad transparency mask"));
        }
        if (mask.width, mask.height) != (source.width, source.height) {
            return Err(PyValueError::new_err("images do not match"));
        }
        Some((mask.pixel_data()?, mask.mode.channels()))
    } else {
        None
    };
    let c = image.mode.channels();
    py.detach(|| {
        // Clip in destination coordinates before computing source offsets.
        let left = position.0.max(0).min(i64::from(image.width));
        let top = position.1.max(0).min(i64::from(image.height));
        let right = position.0.saturating_add(i64::from(source.width)).clamp(0, i64::from(image.width));
        let bottom = position.1.saturating_add(i64::from(source.height)).clamp(0, i64::from(image.height));
        for y in top..bottom {
            for x in left..right {
                let s = ((y - position.1) as usize * source.width as usize + (x - position.0) as usize) * c;
                let d = (y as usize * image.width as usize + x as usize) * c;
                let alpha = mask_pixels.map_or(255, |(data, mc)| u32::from(data[s / c * mc + mc - 1]));
                for channel in 0..c {
                    // Pillow's L-masked color fill replaces hidden RGB when alpha is zero.
                    let alpha = if fill && c == 4 && channel < 3 && pixels[d + 3] == 0 && alpha != 0 && mask_pixels.is_some_and(|(_, mc)| mc == 1) {
                        255
                    } else {
                        alpha
                    };
                    pixels[d + channel] =
                        ((u32::from(pixels[d + channel]) * (255 - alpha) + u32::from(source_pixels[s + channel]) * alpha + 127) / 255) as u8;
                }
            }
        }
    });
    Ok(output)
}

#[pyfunction]
fn image_alpha_composite(py: Python<'_>, background: &Image, overlay: &Image) -> PyResult<Image> {
    if background.mode != PixelMode::Rgba || overlay.mode != PixelMode::Rgba || background.palette.is_some() || overlay.palette.is_some() {
        return Err(PyValueError::new_err("image has wrong mode"));
    }
    if (background.width, background.height) != (overlay.width, overlay.height) {
        return Err(PyValueError::new_err("images do not match"));
    }
    let mut pixels = background.pixel_data()?.to_vec();
    let source = overlay.pixel_data()?;
    py.detach(|| {
        for (dst, src) in pixels.as_chunks_mut::<4>().0.iter_mut().zip(source.as_chunks::<4>().0) {
            if src[3] == 0 {
                continue;
            }
            let blend = u32::from(dst[3]) * (255 - u32::from(src[3]));
            let alpha = u32::from(src[3]) * 255 + blend;
            // Pillow's fixed-point coefficients retain seven extra precision bits.
            let coefficient = u32::from(src[3]) * 255 * 255 * 128 / alpha;
            for c in 0..3 {
                let value = u32::from(src[c]) * coefficient + u32::from(dst[c]) * (255 * 128 - coefficient) + (128 << 7);
                dst[c] = (((value >> 8) + value) >> 8 >> 7) as u8;
            }
            let value = alpha + 128;
            dst[3] = (((value >> 8) + value) >> 8) as u8;
        }
    });
    Image::from_pixels(background.width, background.height, PixelMode::Rgba, pixels, None)
}

#[pyfunction]
fn image_putalpha(py: Python<'_>, image: &Image, alpha: &Image) -> PyResult<Image> {
    if !matches!(image.mode, PixelMode::Rgb | PixelMode::Rgba) || image.palette.is_some() {
        return Err(PyValueError::new_err("putalpha requires RGB or RGBA; LA mode is not supported"));
    }
    if alpha.mode != PixelMode::L || alpha.palette.is_some() || (alpha.width, alpha.height) != (image.width, image.height) {
        return Err(PyValueError::new_err("illegal image mode or size for alpha"));
    }
    let source = image.pixel_data()?;
    let alpha = alpha.pixel_data()?;
    let mut pixels = if image.mode == PixelMode::Rgba {
        source.to_vec()
    } else {
        crate::simd::convert(source, image.mode, PixelMode::Rgba)
    };
    py.detach(|| {
        for (pixel, &a) in pixels.as_chunks_mut::<4>().0.iter_mut().zip(alpha) {
            pixel[3] = a;
        }
    });
    Image::from_pixels(image.width, image.height, PixelMode::Rgba, pixels, image.format.clone())
}
