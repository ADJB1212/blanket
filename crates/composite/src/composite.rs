use pyo3::exceptions::{PyMemoryError, PyValueError};
use pyo3::prelude::*;

use blanket_core::raster::{Image, PixelMode};

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(image_new, module)?)?;
    module.add_function(wrap_pyfunction!(image_paste, module)?)?;
    module.add_function(wrap_pyfunction!(image_alpha_composite, module)?)?;
    module.add_function(wrap_pyfunction!(image_alpha_composite_inplace, module)?)?;
    module.add_function(wrap_pyfunction!(image_putalpha, module)?)?;
    Ok(())
}

#[pyfunction]
fn image_new(py: Python<'_>, mode: &str, size: (u32, u32), color: Vec<u8>) -> PyResult<Image> {
    let mode = PixelMode::parse(mode)?;
    if color.len() != mode.channels() * mode.sample_bytes() {
        return Err(PyValueError::new_err("wrong number of color channels"));
    }
    let len = (size.0 as usize)
        .checked_mul(size.1 as usize)
        .and_then(|n| n.checked_mul(color.len()))
        .ok_or_else(|| PyMemoryError::new_err("image dimensions are too large"))?;
    let mut pixels = Vec::<u8>::new();
    pixels
        .try_reserve_exact(len)
        .map_err(|_| PyMemoryError::new_err("cannot allocate image"))?;
    py.detach(|| {
        let spare = &mut pixels.spare_capacity_mut()[..len];
        match mode {
            PixelMode::One | PixelMode::L => spare.fill(std::mem::MaybeUninit::new(color[0])),
            PixelMode::I | PixelMode::F => fill_pixels::<4>(spare, &color),
            PixelMode::I16 | PixelMode::I16L | PixelMode::I16B | PixelMode::La | PixelMode::Pa => fill_pixels::<2>(spare, &color),
            PixelMode::Rgb | PixelMode::Hsv | PixelMode::YCbCr | PixelMode::Lab => fill_pixels::<3>(spare, &color),
            PixelMode::Rgba | PixelMode::Cmyk => fill_pixels::<4>(spare, &color),
        }
        // All reserved bytes above have been initialized, including empty images.
        unsafe { pixels.set_len(len) };
    });
    if mode == PixelMode::F {
        Image::from_float_bytes(size.0, size.1, pixels)
    } else if mode.is_integer() {
        Image::from_integer_bytes(size.0, size.1, mode, pixels)
    } else {
        Image::from_pixels(size.0, size.1, mode, pixels, None)
    }
}

fn fill_pixels<const C: usize>(pixels: &mut [std::mem::MaybeUninit<u8>], color: &[u8]) {
    blanket_core::parallel::chunks_mut_above(pixels, 256 * 1024 * C, 1024 * 1024, |_, pixels| fill_pixels_chunk::<C>(pixels, color));
}

fn fill_pixels_chunk<const C: usize>(pixels: &mut [std::mem::MaybeUninit<u8>], color: &[u8]) {
    let value: [_; C] = std::array::from_fn(|c| std::mem::MaybeUninit::new(color[c]));
    if C == 3 {
        // Three-byte stores don't vectorize well. A 48-byte repeating tile
        // aligns both RGB pixels and 16-byte vector stores, on every target.
        let tile: [_; 48] = std::array::from_fn(|i| value[i % C]);
        let (tiles, tail) = pixels.as_chunks_mut::<48>();
        tiles.fill(tile);
        tail.as_chunks_mut::<C>().0.fill(value);
        return;
    }
    pixels.as_chunks_mut::<C>().0.fill(value);
}

#[pyfunction]
#[pyo3(signature = (image, source, position, mask=None, fill=false))]
fn image_paste(py: Python<'_>, image: &mut Image, source: &Image, position: (i64, i64), mask: Option<&Image>, fill: bool) -> PyResult<()> {
    if image.mode.is_wide_scalar() {
        return paste_integer(image, source, position, mask);
    }
    image.pixel_data()?;
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
    let width = image.width as usize;
    let height = image.height as usize;
    let pixels = image.pixels.as_mut().expect("validated open image");
    py.detach(|| {
        // Clip in destination coordinates before computing source offsets.
        let left = position.0.clamp(0, width as i64);
        let top = position.1.clamp(0, height as i64);
        let right = position.0.saturating_add(i64::from(source.width)).clamp(0, width as i64);
        let bottom = position.1.saturating_add(i64::from(source.height)).clamp(0, height as i64);
        if right <= left || bottom <= top {
            return;
        }
        let count = (right - left) as usize;
        if mask_pixels.is_none() && count == width && count == source.width as usize {
            let s = (top - position.1) as usize * count * c;
            let d = top as usize * width * c;
            let len = (bottom - top) as usize * count * c;
            pixels[d..d + len].copy_from_slice(&source_pixels[s..s + len]);
            return;
        }
        for y in top..bottom {
            let s = (y - position.1) as usize * source.width as usize + (left - position.0) as usize;
            let d = (y as usize * width + left as usize) * c;
            let dst = &mut pixels[d..d + count * c];
            let src = &source_pixels[s * c..(s + count) * c];
            if let Some((mask, mc)) = mask_pixels {
                let mask = &mask[s * mc..(s + count) * mc];
                match (c, mc) {
                    (1, 1) => paste_masked::<1, 1>(dst, src, mask, fill),
                    (1, 4) => paste_masked::<1, 4>(dst, src, mask, fill),
                    (2, 1) => paste_masked::<2, 1>(dst, src, mask, fill),
                    (2, 4) => paste_masked::<2, 4>(dst, src, mask, fill),
                    (3, 1) => paste_masked::<3, 1>(dst, src, mask, fill),
                    (3, 4) => paste_masked::<3, 4>(dst, src, mask, fill),
                    (4, 1) => paste_masked::<4, 1>(dst, src, mask, fill && source.mode == PixelMode::Rgba),
                    (4, 4) => paste_masked::<4, 4>(dst, src, mask, fill),
                    _ => unreachable!("validated modes"),
                }
            } else {
                dst.copy_from_slice(src);
            }
        }
    });
    Ok(())
}

fn paste_integer(image: &mut Image, source: &Image, position: (i64, i64), mask: Option<&Image>) -> PyResult<()> {
    if image.mode != source.mode {
        return Err(PyValueError::new_err("images do not match"));
    }
    let source_pixels = source.raw_data()?;
    let mask_pixels = if let Some(mask) = mask {
        if mask.palette.is_some()
            || !matches!(mask.mode, PixelMode::L | PixelMode::Rgba)
            || (mask.width, mask.height) != (source.width, source.height)
        {
            return Err(PyValueError::new_err("bad transparency mask"));
        }
        Some((mask.pixel_data()?, mask.mode.channels()))
    } else {
        None
    };
    let stride = image.mode.sample_bytes();
    let width = image.width as i64;
    let height = image.height as i64;
    let left = position.0.clamp(0, width);
    let top = position.1.clamp(0, height);
    let right = position.0.saturating_add(i64::from(source.width)).clamp(0, width);
    let bottom = position.1.saturating_add(i64::from(source.height)).clamp(0, height);
    let pixels = image.pixels.as_mut().ok_or_else(|| PyValueError::new_err("operation on closed image"))?;
    for y in top..bottom {
        for x in left..right {
            let src = ((y - position.1) as usize * source.width as usize + (x - position.0) as usize) * stride;
            let dst = (y as usize * image.width as usize + x as usize) * stride;
            let alpha = mask_pixels.map_or(255, |(mask, channels)| {
                let i = src / stride * channels;
                mask[i + channels - 1]
            });
            if alpha == 255 {
                pixels[dst..dst + stride].copy_from_slice(&source_pixels[src..src + stride]);
            } else if alpha != 0 {
                if image.mode == PixelMode::F {
                    let old = f32::from_le_bytes(pixels[dst..dst + 4].try_into().unwrap());
                    let new = f32::from_le_bytes(source_pixels[src..src + 4].try_into().unwrap());
                    let mixed = (old * (255 - alpha) as f32 + new * alpha as f32) / 255.0;
                    pixels[dst..dst + 4].copy_from_slice(&mixed.to_le_bytes());
                    continue;
                }
                for (destination, &source) in pixels[dst..dst + stride].iter_mut().zip(&source_pixels[src..src + stride]) {
                    let a = u32::from(alpha);
                    *destination = ((u32::from(*destination) * (255 - a) + u32::from(source) * a + 127) / 255) as u8;
                }
            }
        }
    }
    Ok(())
}

fn paste_masked<const C: usize, const M: usize>(dst: &mut [u8], src: &[u8], mask: &[u8], fill: bool) {
    let offset = blanket_core::pixels::paste_masked::<C, M>(dst, src, mask, fill);
    for ((d, s), m) in dst[offset * C..]
        .as_chunks_mut::<C>()
        .0
        .iter_mut()
        .zip(src[offset * C..].as_chunks::<C>().0)
        .zip(mask[offset * M..].as_chunks::<M>().0)
    {
        let alpha = u32::from(m[M - 1]);
        let replace_rgb = fill && C == 4 && M == 1 && d[C - 1] == 0 && alpha != 0;
        for c in 0..C {
            let a = if replace_rgb && c < 3 { 255 } else { alpha };
            d[c] = ((u32::from(d[c]) * (255 - a) + u32::from(s[c]) * a + 127) / 255) as u8;
        }
    }
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
    py.detach(|| composite_pixels(&mut pixels, source));
    Image::from_pixels(background.width, background.height, PixelMode::Rgba, pixels, None)
}

#[pyfunction]
fn image_alpha_composite_inplace(py: Python<'_>, image: &mut Image, overlay: &Image) -> PyResult<()> {
    if image.mode != PixelMode::Rgba || overlay.mode != PixelMode::Rgba || image.palette.is_some() || overlay.palette.is_some() {
        return Err(PyValueError::new_err("image has wrong mode"));
    }
    if (image.width, image.height) != (overlay.width, overlay.height) {
        return Err(PyValueError::new_err("images do not match"));
    }
    image.pixel_data()?;
    let source = overlay.pixel_data()?;
    let pixels = image.pixels.as_mut().expect("validated open image");
    py.detach(|| composite_pixels(pixels, source));
    Ok(())
}

fn composite_pixels(pixels: &mut [u8], source: &[u8]) {
    const CHUNK: usize = 16 * 1024 * 4;
    blanket_core::parallel::chunks_mut(pixels, CHUNK, |i, pixels| {
        let source = &source[i * CHUNK..i * CHUNK + pixels.len()];
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
}

#[pyfunction]
fn image_putalpha(py: Python<'_>, image: &mut Image, alpha: &Image) -> PyResult<()> {
    if !matches!(image.mode, PixelMode::Rgb | PixelMode::Rgba) || image.palette.is_some() {
        return Err(PyValueError::new_err("putalpha requires RGB or RGBA; LA mode is not supported"));
    }
    if alpha.mode != PixelMode::L || alpha.palette.is_some() || (alpha.width, alpha.height) != (image.width, image.height) {
        return Err(PyValueError::new_err("illegal image mode or size for alpha"));
    }
    let source = image.pixel_data()?;
    let alpha = alpha.pixel_data()?;
    if image.mode == PixelMode::Rgb {
        let mut pixels = vec![0; alpha.len() * 4];
        py.detach(|| {
            let offset = blanket_core::pixels::putalpha_rgb(source, alpha, &mut pixels);
            for ((dst, src), &a) in pixels[offset * 4..]
                .as_chunks_mut::<4>()
                .0
                .iter_mut()
                .zip(source[offset * 3..].as_chunks::<3>().0)
                .zip(&alpha[offset..])
            {
                *dst = [src[0], src[1], src[2], a];
            }
        });
        image.pixels = Some(pixels);
        image.mode = PixelMode::Rgba;
        return Ok(());
    }
    let pixels = image.pixels.as_mut().expect("validated open image");
    py.detach(|| {
        let offset = blanket_core::pixels::putalpha_rgba(pixels, alpha);
        for (pixel, &a) in pixels[offset * 4..].as_chunks_mut::<4>().0.iter_mut().zip(&alpha[offset..]) {
            pixel[3] = a;
        }
    });
    Ok(())
}
