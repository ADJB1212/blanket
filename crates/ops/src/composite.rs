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
    if color.len() != mode.channels() {
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
            PixelMode::L => spare.fill(std::mem::MaybeUninit::new(color[0])),
            PixelMode::Rgb => fill_pixels::<3>(spare, &color),
            PixelMode::Rgba => fill_pixels::<4>(spare, &color),
        }
        // All reserved bytes above have been initialized, including empty images.
        unsafe { pixels.set_len(len) };
    });
    Image::from_pixels(size.0, size.1, mode, pixels, None)
}

fn fill_pixels<const C: usize>(pixels: &mut [std::mem::MaybeUninit<u8>], color: &[u8]) {
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
                    (3, 1) => paste_masked::<3, 1>(dst, src, mask, fill),
                    (3, 4) => paste_masked::<3, 4>(dst, src, mask, fill),
                    (4, 1) => paste_masked::<4, 1>(dst, src, mask, fill),
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

fn paste_masked<const C: usize, const M: usize>(dst: &mut [u8], src: &[u8], mask: &[u8], fill: bool) {
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64")))]
    let offset = 0;
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let offset = if std::arch::is_x86_feature_detected!("ssse3") {
        // SAFETY: SSSE3 detected; caller validates equal pixel counts.
        unsafe { blanket_core::x86_pixels::paste_masked::<C, M>(dst, src, mask, fill) }
    } else {
        0
    };
    #[cfg(target_arch = "aarch64")]
    let offset = {
        let mut offset = 0;
        use std::arch::aarch64::*;
        // Each iteration reads and writes 16 complete pixels. NEON is mandatory
        // on AArch64, and all three buffers have validated equal pixel counts.
        unsafe {
            while offset + 16 <= dst.len() / C {
                let dp = dst.as_mut_ptr().add(offset * C);
                let sp = src.as_ptr().add(offset * C);
                let mp = mask.as_ptr().add(offset * M);
                let a = if M == 1 { vld1q_u8(mp) } else { vld4q_u8(mp).3 };
                let blend = |d, s, a| {
                    let inv = vsubq_u8(vdupq_n_u8(255), a);
                    let lo = vmlal_u8(vmull_u8(vget_low_u8(d), vget_low_u8(inv)), vget_low_u8(s), vget_low_u8(a));
                    let hi = vmlal_high_u8(vmull_high_u8(d, inv), s, a);
                    let lo = vaddq_u16(lo, vdupq_n_u16(128));
                    let hi = vaddq_u16(hi, vdupq_n_u16(128));
                    vcombine_u8(
                        vshrn_n_u16::<8>(vaddq_u16(lo, vshrq_n_u16::<8>(lo))),
                        vshrn_n_u16::<8>(vaddq_u16(hi, vshrq_n_u16::<8>(hi))),
                    )
                };
                if C == 1 {
                    vst1q_u8(dp, blend(vld1q_u8(dp), vld1q_u8(sp), a));
                } else if C == 3 {
                    let d = vld3q_u8(dp);
                    let s = vld3q_u8(sp);
                    vst3q_u8(dp, uint8x16x3_t(blend(d.0, s.0, a), blend(d.1, s.1, a), blend(d.2, s.2, a)));
                } else {
                    let d = vld4q_u8(dp);
                    let s = vld4q_u8(sp);
                    let rgb_a = if fill && M == 1 {
                        vorrq_u8(a, vandq_u8(vceqq_u8(d.3, vdupq_n_u8(0)), vcgtq_u8(a, vdupq_n_u8(0))))
                    } else {
                        a
                    };
                    vst4q_u8(
                        dp,
                        uint8x16x4_t(blend(d.0, s.0, rgb_a), blend(d.1, s.1, rgb_a), blend(d.2, s.2, rgb_a), blend(d.3, s.3, a)),
                    );
                }
                offset += 16;
            }
        }
        offset
    };
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
            #[cfg(not(any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64")))]
            let offset = 0;
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            let offset = if std::arch::is_x86_feature_detected!("ssse3") {
                // SAFETY: SSSE3 detected; validated matching RGB/alpha/output sizes.
                unsafe { blanket_core::x86_pixels::putalpha_rgb(source, alpha, &mut pixels) }
            } else {
                0
            };
            #[cfg(target_arch = "aarch64")]
            let offset = {
                let mut offset = 0;
                use std::arch::aarch64::*;
                // Buffers contain the same number of complete pixels; process
                // only complete vectors, leaving the tail to the scalar loop.
                unsafe {
                    while offset + 16 <= alpha.len() {
                        let rgb = vld3q_u8(source.as_ptr().add(offset * 3));
                        let a = vld1q_u8(alpha.as_ptr().add(offset));
                        vst4q_u8(pixels.as_mut_ptr().add(offset * 4), uint8x16x4_t(rgb.0, rgb.1, rgb.2, a));
                        offset += 16;
                    }
                }
                offset
            };
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
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64")))]
        let offset = 0;
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        let offset = if std::arch::is_x86_feature_detected!("ssse3") {
            // SAFETY: SSSE3 detected; validated matching RGBA/alpha sizes.
            unsafe { blanket_core::x86_pixels::putalpha_rgba(pixels, alpha) }
        } else {
            0
        };
        #[cfg(target_arch = "aarch64")]
        let offset = {
            use std::arch::aarch64::*;
            let mut offset = 0;
            // Read and update exactly 16 complete RGBA pixels per iteration.
            // Unaligned NEON loads/stores are valid for these byte buffers.
            unsafe {
                while offset + 16 <= alpha.len() {
                    let ptr = pixels.as_mut_ptr().add(offset * 4);
                    let rgba = vld4q_u8(ptr);
                    let a = vld1q_u8(alpha.as_ptr().add(offset));
                    vst4q_u8(ptr, uint8x16x4_t(rgba.0, rgba.1, rgba.2, a));
                    offset += 16;
                }
            }
            offset
        };
        for (pixel, &a) in pixels[offset * 4..].as_chunks_mut::<4>().0.iter_mut().zip(&alpha[offset..]) {
            pixel[3] = a;
        }
    });
    Ok(())
}
