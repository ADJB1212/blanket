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
    module.add_function(wrap_pyfunction!(image_alpha_composite, module)?)?;
    module.add_function(wrap_pyfunction!(image_alpha_composite_inplace, module)?)?;
    module.add_function(wrap_pyfunction!(image_putalpha, module)?)?;
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
        let mut pixels = blanket_core::raster::alloc_pixels(len)?;
        py.detach(|| {
            for pixel in pixels.chunks_exact_mut(color.len()) {
                pixel.copy_from_slice(&color);
            }
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
    let mut pixels = Vec::<u8>::new();
    pixels.try_reserve_exact(len).map_err(|_| PyMemoryError::new_err("cannot allocate image"))?;
    py.detach(|| {
        let spare = &mut pixels.spare_capacity_mut()[..len];
        match mode {
            PixelMode::One | PixelMode::L => spare.fill(std::mem::MaybeUninit::new(color[0])),
            PixelMode::I | PixelMode::F => fill_pixels::<4>(spare, &color),
            PixelMode::RgbF => fill_pixels::<12>(spare, &color),
            PixelMode::RgbaF => fill_pixels::<16>(spare, &color),
            PixelMode::I16 | PixelMode::I16L | PixelMode::I16B | PixelMode::La | PixelMode::Pa => fill_pixels::<2>(spare, &color),
            PixelMode::Rgb | PixelMode::Hsv | PixelMode::YCbCr | PixelMode::Lab => fill_pixels::<3>(spare, &color),
            PixelMode::Rgba | PixelMode::Cmyk => fill_pixels::<4>(spare, &color),
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
        if mask.palette.is_some() || !matches!(mask.mode, PixelMode::L | PixelMode::Rgba) {
            return Err(PyValueError::new_err("bad transparency mask"));
        }
        if (mask.width, mask.height) != (source.width, source.height) {
            return Err(PyValueError::new_err("images do not match"));
        }
        Ok((mask.pixel_data()?, mask.mode.channels()))
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
#[pyo3(signature = (image, source, position, mask=None, fill=false))]
fn image_paste(py: Python<'_>, image: &mut Image, source: &Image, position: (i64, i64), mask: Option<&Image>, fill: bool) -> PyResult<()> {
    if blanket_core::samples::supported(image) {
        return py.detach(|| crate::high_depth::paste(image, source, position, mask, fill));
    }
    if image.mode.is_wide_scalar() {
        return paste_integer(py, image, source, position, mask);
    }
    image.pixel_data()?;
    let source_pixels = source.pixel_data()?;
    if image.mode != source.mode {
        return Err(PyValueError::new_err("images do not match"));
    }
    let mask_pixels = paste_mask(mask, source)?;
    let c = image.mode.channels();
    let width = image.width as usize;
    let height = image.height as usize;
    let pixels = image.pixels.as_mut().expect("validated open image");
    py.detach(|| {
        let (left, top, right, bottom) = paste_bounds(position, width, height, source.width, source.height);
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

fn paste_integer(py: Python<'_>, image: &mut Image, source: &Image, position: (i64, i64), mask: Option<&Image>) -> PyResult<()> {
    if image.mode != source.mode {
        return Err(PyValueError::new_err("images do not match"));
    }
    let source_pixels = source.raw_data()?;
    let mask_pixels = paste_mask(mask, source)?;
    let stride = image.raw_bytes_per_pixel();
    let width = image.width as i64;
    let pixels = image.pixels.as_mut().ok_or_else(|| PyValueError::new_err("operation on closed image"))?;
    py.detach(|| {
        let (left, top, right, bottom) = paste_bounds(position, image.width as usize, image.height as usize, source.width, source.height);
        if right <= left || bottom <= top {
            return;
        }
        if mask_pixels.is_none() {
            let row_bytes = (right - left) as usize * stride;
            if left == 0 && right == width && image.width == source.width {
                let src = (top - position.1) as usize * row_bytes;
                let dst = top as usize * row_bytes;
                let length = (bottom - top) as usize * row_bytes;
                pixels[dst..dst + length].copy_from_slice(&source_pixels[src..src + length]);
            } else {
                for y in top..bottom {
                    let src = ((y - position.1) as usize * source.width as usize + (left - position.0) as usize) * stride;
                    let dst = (y as usize * image.width as usize + left as usize) * stride;
                    pixels[dst..dst + row_bytes].copy_from_slice(&source_pixels[src..src + row_bytes]);
                }
            }
            return;
        }
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
                    if image.mode.is_float() {
                        for c in (0..stride).step_by(4) {
                            let old = f32::from_le_bytes(pixels[dst + c..dst + c + 4].try_into().unwrap());
                            let new = f32::from_le_bytes(source_pixels[src + c..src + c + 4].try_into().unwrap());
                            let mixed = (old * f32::from(255 - alpha) + new * f32::from(alpha)) / 255.0;
                            pixels[dst + c..dst + c + 4].copy_from_slice(&mixed.to_le_bytes());
                        }
                        continue;
                    }
                    for (destination, &source) in pixels[dst..dst + stride].iter_mut().zip(&source_pixels[src..src + stride]) {
                        let a = u32::from(alpha);
                        *destination = ((u32::from(*destination) * (255 - a) + u32::from(source) * a + 127) / 255) as u8;
                    }
                }
            }
        }
    });
    Ok(())
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
    if background.mode != PixelMode::Rgba || overlay.mode != PixelMode::Rgba || background.palette.is_some() || overlay.palette.is_some() {
        return Err(PyValueError::new_err("image has wrong mode"));
    }
    if (background.width, background.height) != (overlay.width, overlay.height) {
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
    if blanket_core::samples::supported(image) {
        let result = py.detach(|| crate::high_depth::composite(image, overlay))?;
        image.pixels = result.pixels;
        return Ok(());
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
fn image_putalpha(py: Python<'_>, image: &mut Image, alpha: &Image) -> PyResult<()> {
    if !matches!(image.mode, PixelMode::Rgb | PixelMode::Rgba) || image.palette.is_some() {
        return Err(PyValueError::new_err("putalpha requires RGB or RGBA; LA mode is not supported"));
    }
    if alpha.mode != PixelMode::L || alpha.palette.is_some() || (alpha.width, alpha.height) != (image.width, image.height) {
        return Err(PyValueError::new_err("illegal image mode or size for alpha"));
    }
    if blanket_core::samples::supported(image) {
        return py.detach(|| crate::high_depth::putalpha(image, alpha));
    }
    let source = image.pixel_data()?;
    let alpha = alpha.pixel_data()?;
    if image.mode == PixelMode::Rgb {
        let mut pixels = vec![0; alpha.len() * 4];
        py.detach(|| {
            let offset = blanket_core::pixels::putalpha_rgb(source, alpha, &mut pixels);
            for ((dst, src), &a) in
                pixels[offset * 4..].as_chunks_mut::<4>().0.iter_mut().zip(source[offset * 3..].as_chunks::<3>().0).zip(&alpha[offset..])
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

#[cfg(test)]
mod tests {
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
