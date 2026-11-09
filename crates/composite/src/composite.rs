use std::simd::cmp::{SimdOrd, SimdPartialEq, SimdPartialOrd};
use std::simd::num::{SimdFloat, SimdUint};
use std::simd::{Select, Simd};

use blanket_core::pixels::{Bytes, load, store};
use blanket_core::raster::{Image, PixelMode};
use pyo3::exceptions::{PyMemoryError, PyValueError};
use pyo3::prelude::*;

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(image_new, module)?)?;
    module.add_function(wrap_pyfunction!(image_paste, module)?)?;
    module.add_function(wrap_pyfunction!(image_fill, module)?)?;
    module.add_function(wrap_pyfunction!(image_alpha_composite, module)?)?;
    module.add_function(wrap_pyfunction!(image_alpha_composite_inplace, module)?)?;
    module.add_function(wrap_pyfunction!(image_putalpha, module)?)?;
    module.add_function(wrap_pyfunction!(image_putalpha_constant, module)?)?;
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (mode, size, color, bit_depth=8))]
fn image_new(py: Python<'_>, mode: &str, size: (u32, u32), color: Vec<u8>, bit_depth: u8) -> PyResult<Image> {
    let mode = PixelMode::parse(mode)?;
    if bit_depth != 8 {
        if !matches!(mode, PixelMode::L | PixelMode::Rgb | PixelMode::Rgba)
            || !matches!(bit_depth, 10 | 12 | 16)
            || color.len() != mode.channels() * 2
        {
            return Err(PyValueError::new_err("invalid high-depth fill"));
        }
        let len = (size.0 as usize)
            .checked_mul(size.1 as usize)
            .and_then(|n| n.checked_mul(color.len()))
            .ok_or_else(|| PyMemoryError::new_err("image dimensions are too large"))?;
        let mut pixels = blanket_core::buffers::reserved::<u8>(len)?;
        py.detach(|| {
            let spare = &mut pixels.spare_capacity_mut()[..len];
            match mode.channels() {
                1 => fill_pixels::<2>(spare, &color),
                3 => fill_pixels::<6>(spare, &color),
                4 => fill_pixels::<8>(spare, &color),
                _ => unreachable!(),
            }
            // Each complete sample pixel was initialized by the fill kernel.
            unsafe { pixels.set_len(len) };
        });
        return Image::from_sample_bytes(size.0, size.1, mode, pixels, bit_depth, None);
    }
    if color.len() != mode.channels() * mode.sample_bytes() {
        return Err(PyValueError::new_err("wrong number of color channels"));
    }
    let len = (size.0 as usize)
        .checked_mul(size.1 as usize)
        .and_then(|n| n.checked_mul(color.len()))
        .ok_or_else(|| PyMemoryError::new_err("image dimensions are too large"))?;
    let mut pixels = blanket_core::buffers::reserved::<u8>(len)?;
    py.detach(|| {
        let spare = &mut pixels.spare_capacity_mut()[..len];
        match mode {
            PixelMode::One | PixelMode::L => spare.fill(std::mem::MaybeUninit::new(color[0])),
            PixelMode::I | PixelMode::F => fill_pixels::<4>(spare, &color),
            PixelMode::RgbF => fill_pixels::<12>(spare, &color),
            PixelMode::RgbaF => fill_pixels::<16>(spare, &color),
            PixelMode::I16 | PixelMode::I16L | PixelMode::I16B | PixelMode::I16N | PixelMode::La | PixelMode::PremulLa | PixelMode::Pa => {
                fill_pixels::<2>(spare, &color)
            }
            PixelMode::Rgb | PixelMode::Hsv | PixelMode::YCbCr | PixelMode::Lab => fill_pixels::<3>(spare, &color),
            PixelMode::Rgba | PixelMode::PremulRgba | PixelMode::Rgbx | PixelMode::Cmyk => fill_pixels::<4>(spare, &color),
        }
        // All reserved bytes above have been initialized, including empty images.
        unsafe { pixels.set_len(len) };
    });
    if mode.is_float() {
        Image::from_color_float_bytes(size.0, size.1, mode, pixels)
    } else if mode.is_integer() {
        Image::from_integer_bytes(size.0, size.1, mode, pixels)
    } else {
        Image::from_pixels(size.0, size.1, mode, pixels, None)
    }
}

fn fill_pixels<const C: usize>(pixels: &mut [std::mem::MaybeUninit<u8>], color: &[u8]) {
    blanket_core::parallel::chunks_mut_above(pixels, 64 * 1024 * C, 4 * 1024 * 1024, |_, pixels| fill_pixels_chunk::<C>(pixels, color));
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

fn paste_mask<'a>(mask: Option<&'a Image>, source: &Image) -> PyResult<Option<(&'a [u8], usize)>> {
    mask.map(|mask| {
        if mask.palette().is_some() || !matches!(mask.pixel_mode(), PixelMode::L | PixelMode::Rgba) {
            return Err(PyValueError::new_err("bad transparency mask"));
        }
        if (mask.width(), mask.height()) != (source.width(), source.height()) {
            return Err(PyValueError::new_err("images do not match"));
        }
        Ok((mask.pixel_data()?, mask.pixel_mode().channels()))
    })
    .transpose()
}

pub(crate) fn paste_bounds(position: (i64, i64), dst_width: usize, dst_height: usize, src_width: u32, src_height: u32) -> (i64, i64, i64, i64) {
    let left = position.0.clamp(0, dst_width as i64);
    let top = position.1.clamp(0, dst_height as i64);
    let right = position.0.saturating_add(i64::from(src_width)).clamp(0, dst_width as i64);
    let bottom = position.1.saturating_add(i64::from(src_height)).clamp(0, dst_height as i64);
    (left, top, right, bottom)
}

#[pyfunction]
fn image_fill(py: Python<'_>, image: &mut Image, color: &Image, position: (i64, i64), size: (u32, u32)) -> PyResult<()> {
    if image.pixel_mode() != color.pixel_mode() || image.bit_depth() != color.bit_depth() || color.size() != (1, 1) {
        return Err(PyValueError::new_err("images do not match"));
    }
    let bytes = color.raw_data()?;
    image.raw_data()?;
    let stride = image.raw_bytes_per_pixel();
    let mut pixel = [0_u8; 16];
    pixel[..stride].copy_from_slice(bytes);
    if matches!(image.pixel_mode(), PixelMode::I16 | PixelMode::I16L | PixelMode::I16B) {
        pixel[..2].fill(bytes[usize::from(image.pixel_mode().is_big_endian())]);
    }
    let tile: [u8; 48] = std::array::from_fn(|i| pixel[i % stride]);
    let (left, top, right, bottom) = paste_bounds(position, image.width() as usize, image.height() as usize, size.0, size.1);
    if left >= right || top >= bottom {
        return Ok(());
    }
    image.ensure_owned()?;
    let row_bytes = image.width() as usize * stride;
    let pixels = image.pixels_mut().unwrap();
    py.detach(|| {
        blanket_core::parallel::chunks_mut(&mut pixels[top as usize * row_bytes..bottom as usize * row_bytes], row_bytes * 32, |_, rows| {
            for row in rows.chunks_exact_mut(row_bytes) {
                let row = &mut row[left as usize * stride..right as usize * stride];
                let (tiles, tail) = row.as_chunks_mut::<48>();
                tiles.fill(tile);
                for output in tail.chunks_exact_mut(stride) {
                    output.copy_from_slice(&pixel[..stride]);
                }
            }
        });
    });
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (image, source, position, mask=None, fill=false))]
fn image_paste(py: Python<'_>, image: &mut Image, source: &Image, position: (i64, i64), mask: Option<&Image>, fill: bool) -> PyResult<()> {
    image.ensure_owned()?;
    if fill && image.pixel_mode().is_integer() {
        return paste_integer(py, image, source, position, mask, true);
    }
    if blanket_core::samples::supported(image) {
        return py.detach(|| crate::high_depth::paste(image, source, position, mask, fill));
    }
    if image.pixel_mode().is_wide_scalar() {
        return paste_integer(py, image, source, position, mask, fill);
    }
    image.pixel_data()?;
    let source_pixels = source.pixel_data()?;
    if image.pixel_mode() != source.pixel_mode() {
        return Err(PyValueError::new_err("images do not match"));
    }
    let mask_pixels = paste_mask(mask, source)?;
    let c = image.pixel_mode().channels();
    let width = image.width() as usize;
    let height = image.height() as usize;
    let pixels = image.pixels_mut().expect("validated open image");
    py.detach(|| {
        let (left, top, right, bottom) = paste_bounds(position, width, height, source.width(), source.height());
        if right <= left || bottom <= top {
            return;
        }
        let count = (right - left) as usize;
        if mask_pixels.is_none() && count == width && count == source.width() as usize {
            let s = (top - position.1) as usize * count * c;
            let d = top as usize * width * c;
            let len = (bottom - top) as usize * count * c;
            pixels[d..d + len].copy_from_slice(&source_pixels[s..s + len]);
            return;
        }
        for y in top..bottom {
            let s = (y - position.1) as usize * source.width() as usize + (left - position.0) as usize;
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
                    (4, 1) => paste_masked::<4, 1>(dst, src, mask, fill && source.pixel_mode() == PixelMode::Rgba),
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

fn paste_integer(py: Python<'_>, image: &mut Image, source: &Image, position: (i64, i64), mask: Option<&Image>, fill: bool) -> PyResult<()> {
    if image.pixel_mode() != source.pixel_mode() {
        return Err(PyValueError::new_err("images do not match"));
    }
    let source_pixels = source.raw_data()?;
    let filled;
    let source_pixels = if fill && matches!(source.pixel_mode(), PixelMode::I16 | PixelMode::I16L | PixelMode::I16B) {
        filled = source_pixels
            .as_chunks::<2>()
            .0
            .iter()
            .flat_map(|pixel| [pixel[usize::from(source.pixel_mode().is_big_endian())]; 2])
            .collect::<Vec<_>>();
        &filled
    } else {
        source_pixels
    };
    let mask_pixels = paste_mask(mask, source)?;
    let stride = image.raw_bytes_per_pixel();
    let width = image.width() as i64;
    let bounds = paste_bounds(position, image.width() as usize, image.height() as usize, source.width(), source.height());
    let floating = image.pixel_mode().is_float();
    let pixels = image.pixels_mut().ok_or_else(|| PyValueError::new_err("operation on closed image"))?;
    py.detach(|| {
        let (left, top, right, bottom) = bounds;
        if right <= left || bottom <= top {
            return;
        }
        if mask_pixels.is_none() {
            let row_bytes = (right - left) as usize * stride;
            if left == 0 && right == width && width as u32 == source.width() {
                let src = (top - position.1) as usize * row_bytes;
                let dst = top as usize * row_bytes;
                let length = (bottom - top) as usize * row_bytes;
                pixels[dst..dst + length].copy_from_slice(&source_pixels[src..src + length]);
            } else {
                for y in top..bottom {
                    let src = ((y - position.1) as usize * source.width() as usize + (left - position.0) as usize) * stride;
                    let dst = (y as usize * width as usize + left as usize) * stride;
                    pixels[dst..dst + row_bytes].copy_from_slice(&source_pixels[src..src + row_bytes]);
                }
            }
            return;
        }
        for y in top..bottom {
            let count = (right - left) as usize;
            let source_pixel = (y - position.1) as usize * source.width() as usize + (left - position.0) as usize;
            let destination = (y as usize * width as usize + left as usize) * stride;
            let (mask, mc) = mask_pixels.unwrap();
            let mask = &mask[source_pixel * mc..(source_pixel + count) * mc];
            let src = &source_pixels[source_pixel * stride..(source_pixel + count) * stride];
            let dst = &mut pixels[destination..destination + count * stride];
            if floating {
                paste_float(dst, src, mask, stride / 4, mc);
            } else {
                match (stride, mc) {
                    (2, 1) => paste_masked::<2, 1>(dst, src, mask, false),
                    (2, 4) => paste_masked::<2, 4>(dst, src, mask, false),
                    (4, 1) => paste_masked::<4, 1>(dst, src, mask, false),
                    (4, 4) => paste_masked::<4, 4>(dst, src, mask, false),
                    _ => unreachable!(),
                }
            }
        }
    });
    Ok(())
}

fn paste_float(dst: &mut [u8], src: &[u8], mask: &[u8], channels: usize, mask_channels: usize) {
    let samples = dst.len() / 4;
    let end = samples / 8 * 8;
    for i in (0..end).step_by(8) {
        let old = Simd::<f32, 8>::from_array(std::array::from_fn(|c| f32::from_le_bytes(dst[(i + c) * 4..(i + c + 1) * 4].try_into().unwrap())));
        let new = Simd::<f32, 8>::from_array(std::array::from_fn(|c| f32::from_le_bytes(src[(i + c) * 4..(i + c + 1) * 4].try_into().unwrap())));
        let a = Simd::<f32, 8>::from_array(std::array::from_fn(|c| f32::from(mask[(i + c) / channels * mask_channels + mask_channels - 1])));
        let mixed = (old * (Simd::splat(255.0) - a) + new * a) / Simd::splat(255.0);
        let result = a.simd_eq(Simd::splat(0.0)).select(old, a.simd_eq(Simd::splat(255.0)).select(new, mixed));
        for (c, v) in result.to_array().iter().enumerate() {
            dst[(i + c) * 4..(i + c + 1) * 4].copy_from_slice(&v.to_le_bytes());
        }
    }
    for i in end..samples {
        let a = mask[i / channels * mask_channels + mask_channels - 1];
        if a == 255 {
            dst[i * 4..i * 4 + 4].copy_from_slice(&src[i * 4..i * 4 + 4]);
        } else if a != 0 {
            let old = f32::from_le_bytes(dst[i * 4..i * 4 + 4].try_into().unwrap());
            let new = f32::from_le_bytes(src[i * 4..i * 4 + 4].try_into().unwrap());
            dst[i * 4..i * 4 + 4].copy_from_slice(&((old * f32::from(255 - a) + new * f32::from(a)) / 255.0).to_le_bytes());
        }
    }
}

fn paste_masked<const C: usize, const M: usize>(dst: &mut [u8], src: &[u8], mask: &[u8], fill: bool) {
    let offset = blanket_core::pixels::paste_masked::<C, M>(dst, src, mask, fill);
    for ((d, s), m) in
        dst[offset * C..].as_chunks_mut::<C>().0.iter_mut().zip(src[offset * C..].as_chunks::<C>().0).zip(mask[offset * M..].as_chunks::<M>().0)
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
pub fn image_alpha_composite(py: Python<'_>, background: &Image, overlay: &Image) -> PyResult<Image> {
    if background.pixel_mode() != PixelMode::Rgba
        || overlay.pixel_mode() != PixelMode::Rgba
        || background.palette().is_some()
        || overlay.palette().is_some()
    {
        return Err(PyValueError::new_err("image has wrong mode"));
    }
    if (background.width(), background.height()) != (overlay.width(), overlay.height()) {
        return Err(PyValueError::new_err("images do not match"));
    }
    if blanket_core::samples::supported(background) {
        return py.detach(|| crate::high_depth::composite(background, overlay));
    }
    let background_pixels = background.pixel_data()?;
    let source = overlay.pixel_data()?;
    let pixels = py.detach(|| {
        let mut pixels = background_pixels.to_vec();
        composite_pixels(&mut pixels, source);
        pixels
    });
    Image::from_pixels(background.width(), background.height(), PixelMode::Rgba, pixels, None)
}

#[pyfunction]
fn image_alpha_composite_inplace(py: Python<'_>, image: &mut Image, overlay: &Image) -> PyResult<()> {
    image.ensure_owned()?;
    if image.pixel_mode() != PixelMode::Rgba || overlay.pixel_mode() != PixelMode::Rgba || image.palette().is_some() || overlay.palette().is_some() {
        return Err(PyValueError::new_err("image has wrong mode"));
    }
    if (image.width(), image.height()) != (overlay.width(), overlay.height()) {
        return Err(PyValueError::new_err("images do not match"));
    }
    if blanket_core::samples::supported(image) {
        return py.detach(|| crate::high_depth::composite_inplace(image, overlay));
    }
    image.pixel_data()?;
    let source = overlay.pixel_data()?;
    let pixels = image.pixels_mut().expect("validated open image");
    py.detach(|| composite_pixels(pixels, source));
    Ok(())
}

fn composite_pixels(pixels: &mut [u8], source: &[u8]) {
    const CHUNK: usize = 16 * 1024 * 4;
    blanket_core::parallel::chunks_mut(pixels, CHUNK, |i, pixels| {
        let source = &source[i * CHUNK..i * CHUNK + pixels.len()];
        let done = pixels.len() / 64 * 64;
        for i in (0..done).step_by(64) {
            let src = load::<4>(&source[i..]);
            let active = src[3].simd_ne(Bytes::splat(0));
            if !active.any() {
                continue;
            }
            if src[3].simd_eq(Bytes::splat(255)).all() {
                pixels[i..i + 64].copy_from_slice(&source[i..i + 64]);
                continue;
            }
            let dst = load::<4>(&pixels[i..]);
            let source_alpha = src[3].cast::<u32>();
            let alpha = source_alpha * Simd::splat(255) + dst[3].cast::<u32>() * (Simd::splat(255) - source_alpha);
            // Inactive lanes can have zero combined alpha.
            let numerator = source_alpha * Simd::splat(255 * 255 * 128);
            let denominator = alpha.simd_max(Simd::splat(1));
            // The float quotient is within one integer; correct it exactly.
            let estimate = (numerator.cast::<f32>() / denominator.cast::<f32>()).cast::<u32>();
            let estimate = estimate - (estimate * denominator).simd_gt(numerator).select(Simd::splat(1), Simd::splat(0));
            let coefficient = estimate + (numerator - estimate * denominator).simd_ge(denominator).select(Simd::splat(1), Simd::splat(0));
            let output = std::array::from_fn::<_, 4, _>(|c| {
                let value = if c == 3 {
                    let value = alpha + Simd::splat(128);
                    ((value >> 8) + value) >> 8
                } else {
                    let value =
                        src[c].cast::<u32>() * coefficient + dst[c].cast::<u32>() * (Simd::splat(255 * 128) - coefficient) + Simd::splat(128 << 7);
                    ((value >> 8) + value) >> 15
                };
                active.select(value.cast::<u8>(), dst[c])
            });
            store(&mut pixels[i..], output);
        }
        composite_scalar(&mut pixels[done..], &source[done..]);
    });
}

fn composite_scalar(pixels: &mut [u8], source: &[u8]) {
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
}

#[pyfunction]
fn image_putalpha_constant(py: Python<'_>, image: &mut Image, alpha: u16) -> PyResult<()> {
    if !matches!(
        image.pixel_mode(),
        PixelMode::One
            | PixelMode::L
            | PixelMode::La
            | PixelMode::Pa
            | PixelMode::Rgb
            | PixelMode::Rgba
            | PixelMode::PremulLa
            | PixelMode::PremulRgba
            | PixelMode::Rgbx
    ) {
        return Err(PyValueError::new_err("image has wrong mode"));
    }
    if matches!(image.pixel_mode(), PixelMode::PremulLa | PixelMode::PremulRgba | PixelMode::Rgbx) {
        let converted = image.convert(py, if image.pixel_mode() == PixelMode::PremulLa { "LA" } else { "RGBA" }, None)?;
        image.replace_storage(converted)?;
    }
    if image.bit_depth() > 8 && !matches!(image.pixel_mode(), PixelMode::Rgb | PixelMode::Rgba) {
        return Err(PyValueError::new_err("high-depth putalpha requires RGB or RGBA"));
    }
    image.raw_data()?;
    let bytes = alpha.to_le_bytes();
    let sample_bytes = if image.bit_depth() == 8 { 1 } else { 2 };
    let alpha = &bytes[..sample_bytes];
    if matches!(image.pixel_mode(), PixelMode::La | PixelMode::Pa | PixelMode::Rgba) {
        image.ensure_owned()?;
        let stride = image.raw_bytes_per_pixel();
        let pixels = image.pixels_mut().unwrap();
        py.detach(|| match stride {
            2 => set_constant_alpha::<2, 1>(pixels, alpha),
            4 => set_constant_alpha::<4, 1>(pixels, alpha),
            8 => set_constant_alpha::<8, 2>(pixels, alpha),
            _ => unreachable!(),
        });
        return Ok(());
    }
    let source = image.raw_data()?;
    let pixels = py.detach(|| match (image.pixel_mode().channels(), sample_bytes) {
        (1, 1) => constant_alpha::<1, 2>(source, alpha, image.pixel_mode() == PixelMode::One),
        (3, 1) => constant_alpha::<3, 4>(source, alpha, false),
        (3, 2) => constant_alpha::<6, 8>(source, alpha, false),
        _ => unreachable!(),
    })?;
    let mode = if image.pixel_mode().channels() == 3 {
        PixelMode::Rgba
    } else if image.palette().is_some() {
        PixelMode::Pa
    } else {
        PixelMode::La
    };
    image.replace_storage(Image::from_raw_bytes(image.width(), image.height(), mode, pixels, image.bit_depth())?)
}

fn set_constant_alpha<const C: usize, const A: usize>(pixels: &mut [u8], alpha: &[u8]) {
    let alpha: &[u8; A] = alpha.try_into().unwrap();
    blanket_core::parallel::chunks_mut(pixels, 16 * 1024 * C, |_, dst| {
        for pixel in dst.as_chunks_mut::<C>().0 {
            pixel[C - A..].copy_from_slice(alpha);
        }
    });
}

fn constant_alpha<const S: usize, const D: usize>(source: &[u8], alpha: &[u8], bilevel: bool) -> PyResult<Vec<u8>> {
    let len = (source.len() / S).checked_mul(D).ok_or_else(|| PyMemoryError::new_err("image dimensions are too large"))?;
    let mut pixels = blanket_core::buffers::reserved::<u8>(len)?;
    blanket_core::parallel::chunks_mut_above(
        &mut pixels.spare_capacity_mut()[..len],
        16 * 1024 * D,
        blanket_core::parallel::MIN_PARALLEL_BYTES,
        |chunk, output| {
            let start = chunk * 16 * 1024 * S;
            for (src, dst) in source[start..].as_chunks::<S>().0.iter().zip(output.as_chunks_mut::<D>().0) {
                dst[..S].write_copy_of_slice(src);
                if bilevel {
                    dst[0].write(u8::from(src[0] != 0) * 255);
                }
                dst[S..].write_copy_of_slice(alpha);
            }
        },
    );
    // Every color and alpha byte in each reserved pixel was initialized above.
    unsafe { pixels.set_len(len) };
    Ok(pixels)
}

#[pyfunction]
fn image_putalpha(py: Python<'_>, image: &mut Image, alpha: &Image) -> PyResult<()> {
    image.ensure_owned()?;
    if !matches!(
        image.pixel_mode(),
        PixelMode::One
            | PixelMode::L
            | PixelMode::La
            | PixelMode::PremulLa
            | PixelMode::Pa
            | PixelMode::Rgb
            | PixelMode::Rgba
            | PixelMode::PremulRgba
            | PixelMode::Rgbx
    ) {
        return Err(PyValueError::new_err("image has wrong mode"));
    }
    if alpha.pixel_mode() != PixelMode::L || alpha.palette().is_some() || (alpha.width(), alpha.height()) != (image.width(), image.height()) {
        return Err(PyValueError::new_err("illegal image mode or size for alpha"));
    }
    if matches!(image.pixel_mode(), PixelMode::PremulLa | PixelMode::PremulRgba | PixelMode::Rgbx) {
        let mode = if image.pixel_mode() == PixelMode::PremulLa { "LA" } else { "RGBA" };
        let converted = image.convert(py, mode, None)?;
        image.replace_storage(converted)?;
    }
    if blanket_core::samples::supported(image) {
        if !matches!(image.pixel_mode(), PixelMode::Rgb | PixelMode::Rgba) {
            return Err(PyValueError::new_err("high-depth putalpha requires RGB or RGBA"));
        }
        return py.detach(|| crate::high_depth::putalpha(image, alpha));
    }
    let source = image.pixel_data()?;
    let alpha = alpha.pixel_data()?;
    if matches!(image.pixel_mode(), PixelMode::La | PixelMode::Pa) {
        let pixels = image.pixels_mut().expect("validated open image");
        py.detach(|| {
            blanket_core::parallel::chunks_mut(pixels, 32 * 1024, |chunk, pixels| {
                for (pixel, &value) in pixels.as_chunks_mut::<2>().0.iter_mut().zip(&alpha[chunk * 16 * 1024..]) {
                    pixel[1] = value;
                }
            });
        });
        return Ok(());
    }
    if matches!(image.pixel_mode(), PixelMode::One | PixelMode::L | PixelMode::La | PixelMode::Pa) {
        let channels = image.pixel_mode().channels();
        let bilevel = image.pixel_mode() == PixelMode::One;
        let mode = if image.palette().is_some() || image.pixel_mode() == PixelMode::Pa { PixelMode::Pa } else { PixelMode::La };
        let mut pixels = vec![0; alpha.len() * 2];
        py.detach(|| {
            let end = alpha.len() / 16 * 16;
            for i in (0..end).step_by(16) {
                let gray = if channels == 1 { load::<1>(&source[i..])[0] } else { load::<2>(&source[i * 2..])[0] };
                let gray = if bilevel { gray.simd_ne(Bytes::splat(0)).select(Bytes::splat(255), Bytes::splat(0)) } else { gray };
                store::<2>(&mut pixels[i * 2..], [gray, Bytes::from_slice(&alpha[i..i + 16])]);
            }
            for i in end..alpha.len() {
                let gray = source[i * channels];
                pixels[i * 2] = if bilevel { u8::from(gray != 0) * 255 } else { gray };
                pixels[i * 2 + 1] = alpha[i];
            }
        });
        image.replace_storage(Image::from_pixels(image.width(), image.height(), mode, pixels, None)?)?;
        return Ok(());
    }
    if image.pixel_mode() == PixelMode::Rgb {
        let mut pixels = vec![0; alpha.len() * 4];
        py.detach(|| {
            let offset = blanket_core::pixels::putalpha_rgb(source, alpha, &mut pixels);
            for ((dst, src), &a) in
                pixels[offset * 4..].as_chunks_mut::<4>().0.iter_mut().zip(source[offset * 3..].as_chunks::<3>().0).zip(&alpha[offset..])
            {
                *dst = [src[0], src[1], src[2], a];
            }
        });
        image.replace_storage(Image::from_pixels(image.width(), image.height(), PixelMode::Rgba, pixels, None)?)?;
        return Ok(());
    }
    let pixels = image.pixels_mut().expect("validated open image");
    py.detach(|| {
        let offset = blanket_core::pixels::putalpha_rgba(pixels, alpha);
        for (pixel, &a) in pixels[offset * 4..].as_chunks_mut::<4>().0.iter_mut().zip(&alpha[offset..]) {
            pixel[3] = a;
        }
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn float_paste_matches_scalar_with_guards() {
        let values = [-0.0_f32, 0.0, -2.0, 0.5, 1e30, f32::NAN, f32::INFINITY, f32::NEG_INFINITY];
        for channels in [1, 3, 4] {
            for mc in [1, 4] {
                for count in 0..35 {
                    for offset in 0..4 {
                        let length = count * channels * 4;
                        let mut destination = vec![123; offset + length + 8];
                        let mut source = vec![123; offset + length + 8];
                        let mask: Vec<_> = (0..count * mc + offset).map(|i| [0, 1, 127, 128, 254, 255][i % 6]).collect();
                        for i in 0..count * channels {
                            destination[offset + i * 4..offset + i * 4 + 4].copy_from_slice(&values[i % 8].to_le_bytes());
                            source[offset + i * 4..offset + i * 4 + 4].copy_from_slice(&values[(i + 3) % 8].to_le_bytes());
                        }
                        let mut expected = destination.clone();
                        for i in 0..count * channels {
                            let a = mask[offset + i / channels * mc + mc - 1];
                            let begin = offset + i * 4;
                            if a == 255 {
                                expected[begin..begin + 4].copy_from_slice(&source[begin..begin + 4]);
                            } else if a != 0 {
                                let old = f32::from_le_bytes(expected[begin..begin + 4].try_into().unwrap());
                                let new = f32::from_le_bytes(source[begin..begin + 4].try_into().unwrap());
                                expected[begin..begin + 4].copy_from_slice(&((old * f32::from(255 - a) + new * f32::from(a)) / 255.0).to_le_bytes());
                            }
                        }
                        super::paste_float(
                            &mut destination[offset..offset + length],
                            &source[offset..offset + length],
                            &mask[offset..],
                            channels,
                            mc,
                        );
                        for i in 0..count * channels {
                            let begin = offset + i * 4;
                            let a = f32::from_le_bytes(destination[begin..begin + 4].try_into().unwrap());
                            let b = f32::from_le_bytes(expected[begin..begin + 4].try_into().unwrap());
                            let mask_value = mask[offset + i / channels * mc + mc - 1];
                            assert!(a.to_bits() == b.to_bits() || !matches!(mask_value, 0 | 255) && a.is_nan() && b.is_nan());
                        }
                        assert_eq!(&destination[..offset], &expected[..offset]);
                        assert_eq!(&destination[offset + length..], &expected[offset + length..]);
                    }
                }
            }
        }
    }
    use super::*;

    #[test]
    fn alpha_composite_matches_scalar_at_every_alpha_pair() {
        for colors in [[0, 255, 127], [255, 0, 128], [75, 120, 210]] {
            let mut background = Vec::new();
            let mut overlay = Vec::new();
            for a in 0..=255 {
                for b in 0..=255 {
                    background.extend_from_slice(&[colors[0], colors[1], colors[2], a]);
                    overlay.extend_from_slice(&[colors[2], colors[0], colors[1], b]);
                }
            }
            let mut expected = background.clone();
            composite_scalar(&mut expected, &overlay);
            composite_pixels(&mut background, &overlay);
            assert_eq!(background, expected);
        }
    }

    #[test]
    fn alpha_composite_handles_unaligned_buffers_tails_and_guards() {
        for offset in 0..16 {
            for count in [0, 1, 15, 16, 17, 31, 32, 33, 16 * 1024 + 1, 65537] {
                for alpha in [None, Some(0), Some(255)] {
                    let end = offset + count * 4;
                    let mut background: Vec<u8> = (0..end + 16).map(|i| (i * 37 + i / 11) as u8).collect();
                    let mut overlay: Vec<u8> = (0..end).map(|i| (i * 53 + i / 7) as u8).collect();
                    if let Some(alpha) = alpha {
                        for pixel in overlay[offset..].as_chunks_mut::<4>().0 {
                            pixel[3] = alpha;
                        }
                    }
                    let mut expected = background.clone();
                    composite_scalar(&mut expected[offset..end], &overlay[offset..]);
                    composite_pixels(&mut background[offset..end], &overlay[offset..]);
                    assert_eq!(background, expected, "offset={offset}, count={count}, alpha={alpha:?}");
                }
            }
        }
    }
}
