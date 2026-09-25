//! Native primitives used by the Python ImageEnhance API.

use pyo3::exceptions::{PyMemoryError, PyValueError};
use pyo3::prelude::*;
use rayon::prelude::*;

use blanket_core::parallel::{CHUNK_PIXELS, MIN_PARALLEL_BYTES, chunks_mut, chunks_mut_above, should_parallel};
use blanket_core::raster::{Image, PixelMode};

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(enhance_blend, module)?)?;
    module.add_function(wrap_pyfunction!(enhance_color, module)?)?;
    module.add_function(wrap_pyfunction!(enhance_contrast, module)?)?;
    module.add_function(wrap_pyfunction!(enhance_brightness, module)?)?;
    module.add_function(wrap_pyfunction!(enhance_sharpness, module)?)?;
    Ok(())
}

fn buffer(len: usize) -> PyResult<Vec<u8>> {
    let mut result = Vec::new();
    result
        .try_reserve_exact(len)
        .map_err(|_| PyMemoryError::new_err("cannot allocate image"))?;
    result.resize(len, 0);
    Ok(result)
}

fn output(image: &Image, pixels: Vec<u8>) -> PyResult<Image> {
    Image::from_pixels(image.width, image.height, image.mode, pixels, None)
}

fn copy(image: &Image, source: &[u8]) -> PyResult<Image> {
    let mut pixels = Vec::new();
    pixels
        .try_reserve_exact(source.len())
        .map_err(|_| PyMemoryError::new_err("cannot allocate image"))?;
    pixels.extend_from_slice(source);
    output(image, pixels)
}

/// Pillow's core blend uses a single-precision alpha and truncates toward zero.
#[pyfunction]
fn enhance_blend(py: Python<'_>, first: &Image, second: &Image, factor: f32) -> PyResult<Image> {
    let first_pixels = first.pixel_data()?;
    let second_pixels = second.pixel_data()?;
    if first.mode != second.mode || first.width != second.width || first.height != second.height {
        return Err(PyValueError::new_err("images do not match"));
    }
    if std::ptr::eq(first, second) || factor == 0.0 {
        return copy(first, first_pixels);
    }
    if factor == 1.0 {
        return copy(second, second_pixels);
    }

    let mut pixels = buffer(first_pixels.len())?;
    py.detach(|| {
        chunks_mut(&mut pixels, CHUNK_PIXELS * first.mode.channels(), |i, dst| {
            let start = i * CHUNK_PIXELS * first.mode.channels();
            blend_bytes(
                &first_pixels[start..start + dst.len()],
                &second_pixels[start..start + dst.len()],
                dst,
                factor,
            );
        });
    });
    output(first, pixels)
}

fn blend_bytes(first: &[u8], second: &[u8], output: &mut [u8], factor: f32) {
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64")))]
    let done = 0;
    #[cfg(target_arch = "aarch64")]
    let done = unsafe { blend_bytes_neon(first, second, output, factor) };
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let done = if std::arch::is_x86_feature_detected!("sse2") {
        // SAFETY: SSE2 availability verified above.
        unsafe { blend_bytes_sse2(first, second, output, factor) }
    } else {
        0
    };
    blend_bytes_scalar(&first[done..], &second[done..], &mut output[done..], factor);
}

fn blend_bytes_scalar(first: &[u8], second: &[u8], output: &mut [u8], factor: f32) {
    for ((&a, &b), dst) in first.iter().zip(second).zip(output) {
        *dst = (f32::from(a) + factor * (f32::from(b) - f32::from(a))) as u8;
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn blend_bytes_neon(first: &[u8], second: &[u8], output: &mut [u8], factor: f32) -> usize {
    use std::arch::aarch64::*;
    let len = first.len().min(second.len()).min(output.len());
    let mut i = 0;
    // SAFETY: NEON is mandatory on AArch64. All pointer arithmetic stays
    // within validated equal-length slices.
    unsafe {
        let vfactor = vdupq_n_f32(factor);
        while i + 16 <= len {
            let a = vld1q_u8(first.as_ptr().add(i));
            let b = vld1q_u8(second.as_ptr().add(i));
            let a16_lo = vmovl_u8(vget_low_u8(a));
            let a16_hi = vmovl_high_u8(a);
            let b16_lo = vmovl_u8(vget_low_u8(b));
            let b16_hi = vmovl_high_u8(b);

            let blend_lane = |a16: uint16x4_t, b16: uint16x4_t| -> uint32x4_t {
                let af = vcvtq_f32_u32(vmovl_u16(a16));
                let bf = vcvtq_f32_u32(vmovl_u16(b16));
                let r = vfmaq_f32(af, vfactor, vsubq_f32(bf, af));
                let clamped = vmaxq_f32(vminq_f32(r, vdupq_n_f32(255.0)), vdupq_n_f32(0.0));
                vcvtq_u32_f32(clamped)
            };

            let r0 = blend_lane(vget_low_u16(a16_lo), vget_low_u16(b16_lo));
            let r1 = blend_lane(vget_high_u16(a16_lo), vget_high_u16(b16_lo));
            let r2 = blend_lane(vget_low_u16(a16_hi), vget_low_u16(b16_hi));
            let r3 = blend_lane(vget_high_u16(a16_hi), vget_high_u16(b16_hi));

            let narrow_lo = vmovn_u16(vcombine_u16(vmovn_u32(r0), vmovn_u32(r1)));
            let narrow_hi = vmovn_u16(vcombine_u16(vmovn_u32(r2), vmovn_u32(r3)));
            vst1q_u8(output.as_mut_ptr().add(i), vcombine_u8(narrow_lo, narrow_hi));
            i += 16;
        }
    }
    i
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn blend_bytes_sse2(first: &[u8], second: &[u8], output: &mut [u8], factor: f32) -> usize {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let len = first.len().min(second.len()).min(output.len());
    let mut i = 0;
    unsafe {
        let vfactor = _mm_set1_ps(factor);
        let izero = _mm_setzero_si128();
        let fmax = _mm_set1_ps(255.0);
        let fzero = _mm_setzero_ps();

        let blend_lane = |a32: __m128i, b32: __m128i| -> __m128i {
            let af = _mm_cvtepi32_ps(a32);
            let bf = _mm_cvtepi32_ps(b32);
            let r = _mm_add_ps(af, _mm_mul_ps(vfactor, _mm_sub_ps(bf, af)));
            _mm_cvttps_epi32(_mm_min_ps(_mm_max_ps(r, fzero), fmax))
        };

        while i + 16 <= len {
            let a = _mm_loadu_si128(first.as_ptr().add(i).cast());
            let b = _mm_loadu_si128(second.as_ptr().add(i).cast());

            let a_lo16 = _mm_unpacklo_epi8(a, izero);
            let a_hi16 = _mm_unpackhi_epi8(a, izero);
            let b_lo16 = _mm_unpacklo_epi8(b, izero);
            let b_hi16 = _mm_unpackhi_epi8(b, izero);

            let r0 = blend_lane(_mm_unpacklo_epi16(a_lo16, izero), _mm_unpacklo_epi16(b_lo16, izero));
            let r1 = blend_lane(_mm_unpackhi_epi16(a_lo16, izero), _mm_unpackhi_epi16(b_lo16, izero));
            let r2 = blend_lane(_mm_unpacklo_epi16(a_hi16, izero), _mm_unpacklo_epi16(b_hi16, izero));
            let r3 = blend_lane(_mm_unpackhi_epi16(a_hi16, izero), _mm_unpackhi_epi16(b_hi16, izero));

            let packed = _mm_packus_epi16(_mm_packs_epi32(r0, r1), _mm_packs_epi32(r2, r3));
            _mm_storeu_si128(output.as_mut_ptr().add(i).cast(), packed);
            i += 16;
        }
    }
    i
}

#[pyfunction]
fn enhance_color(py: Python<'_>, image: &Image) -> PyResult<Image> {
    let source = image.pixel_data()?;
    if matches!(image.mode, PixelMode::One | PixelMode::L | PixelMode::La | PixelMode::Pa) {
        return copy(image, source);
    }

    let mut pixels = buffer(source.len())?;
    py.detach(|| match image.mode {
        PixelMode::One | PixelMode::L | PixelMode::La | PixelMode::Pa => unreachable!(),
        PixelMode::Rgb => color_degenerate::<3>(source, &mut pixels),
        PixelMode::Rgba => color_degenerate::<4>(source, &mut pixels),
        _ => unreachable!(),
    });
    output(image, pixels)
}

fn color_degenerate<const C: usize>(source: &[u8], output: &mut [u8]) {
    chunks_mut(output, CHUNK_PIXELS * C, |i, dst| {
        let start = i * CHUNK_PIXELS * C;
        for (source, output) in source[start..].as_chunks::<C>().0.iter().zip(dst.as_chunks_mut::<C>().0) {
            let luma = blanket_core::simd::pillow_luma(source[0], source[1], source[2]);
            output[0] = luma;
            output[1] = luma;
            output[2] = luma;
            if C == 4 {
                output[3] = source[3];
            }
        }
    });
}

#[pyfunction]
fn enhance_contrast(py: Python<'_>, image: &Image) -> PyResult<Image> {
    let source = image.pixel_data()?;
    let sum = py.detach(|| match image.mode {
        PixelMode::One | PixelMode::L => byte_sum(source),
        PixelMode::La | PixelMode::Pa => source.as_chunks::<2>().0.iter().map(|pixel| u64::from(pixel[0])).sum(),
        PixelMode::Rgb => luminance_sum::<3>(source),
        PixelMode::Rgba => luminance_sum::<4>(source),
        _ => unreachable!(),
    });
    let pixel_count = source.len() / image.mode.channels();
    let mean = if pixel_count == 0 {
        0
    } else {
        (sum as f64 / pixel_count as f64 + 0.5) as u8
    };

    let mut pixels = buffer(source.len())?;
    py.detach(|| match image.mode {
        PixelMode::One | PixelMode::L => pixels.fill(mean),
        PixelMode::La | PixelMode::Pa => {
            for (src, dst) in source.as_chunks::<2>().0.iter().zip(pixels.as_chunks_mut::<2>().0) {
                dst.copy_from_slice(&[mean, src[1]]);
            }
        }
        PixelMode::Rgb => chunks_mut(&mut pixels, CHUNK_PIXELS * 3, |_, dst| {
            for pixel in dst.as_chunks_mut::<3>().0 {
                pixel.fill(mean);
            }
        }),
        PixelMode::Rgba => chunks_mut(&mut pixels, CHUNK_PIXELS * 4, |i, dst| {
            let start = i * CHUNK_PIXELS * 4;
            for (source, output) in source[start..].as_chunks::<4>().0.iter().zip(dst.as_chunks_mut::<4>().0) {
                *output = [mean, mean, mean, source[3]];
            }
        }),
        _ => unreachable!(),
    });
    output(image, pixels)
}

fn byte_sum(source: &[u8]) -> u64 {
    if should_parallel(source.len(), 1, MIN_PARALLEL_BYTES) {
        source.par_iter().map(|&value| u64::from(value)).sum()
    } else {
        source.iter().map(|&value| u64::from(value)).sum()
    }
}

fn luminance_sum<const C: usize>(source: &[u8]) -> u64 {
    let pixels = source.as_chunks::<C>().0;
    let luminance = |pixel: &[u8; C]| u64::from(blanket_core::simd::pillow_luma(pixel[0], pixel[1], pixel[2]));
    if should_parallel(source.len(), C, MIN_PARALLEL_BYTES) {
        pixels.par_iter().map(luminance).sum()
    } else {
        pixels.iter().map(luminance).sum()
    }
}

#[pyfunction]
fn enhance_brightness(py: Python<'_>, image: &Image) -> PyResult<Image> {
    let mut pixels = buffer(
        (image.width as usize)
            .checked_mul(image.height as usize)
            .and_then(|count| count.checked_mul(image.mode.channels()))
            .ok_or_else(|| PyValueError::new_err("image dimensions are too large"))?,
    )?;
    if image.mode == PixelMode::Rgba {
        let source = image.pixel_data()?;
        py.detach(|| {
            chunks_mut(&mut pixels, CHUNK_PIXELS * 4, |i, dst| {
                let start = i * CHUNK_PIXELS * 4;
                for (source, output) in source[start..].as_chunks::<4>().0.iter().zip(dst.as_chunks_mut::<4>().0) {
                    output[3] = source[3];
                }
            });
        });
    }
    output(image, pixels)
}

#[pyfunction]
fn enhance_sharpness(py: Python<'_>, image: &Image) -> PyResult<Image> {
    let source = image.pixel_data()?;
    let mut pixels = buffer(source.len())?;
    pixels.copy_from_slice(source);
    if image.width < 3 || image.height < 3 {
        return output(image, pixels);
    }

    let width = image.width as usize;
    py.detach(|| match image.mode {
        PixelMode::One | PixelMode::L => smooth::<1>(source, &mut pixels, width),
        PixelMode::La | PixelMode::Pa => smooth::<2>(source, &mut pixels, width),
        PixelMode::Rgb => smooth::<3>(source, &mut pixels, width),
        PixelMode::Rgba => smooth::<4>(source, &mut pixels, width),
        _ => unreachable!(),
    });
    output(image, pixels)
}

fn smooth<const C: usize>(source: &[u8], output: &mut [u8], width: usize) {
    let row_bytes = width * C;
    let height = source.len() / row_bytes;
    chunks_mut_above(output, row_bytes * 32, MIN_PARALLEL_BYTES, |band, rows| {
        for (i, row) in rows.chunks_exact_mut(row_bytes).enumerate() {
            let y = band * 32 + i;
            if y == 0 || y + 1 >= height {
                continue;
            }
            for x in 1..width - 1 {
                let center = y * row_bytes + x * C;
                let channels = if C == 4 { 3 } else { C };
                for channel in 0..channels {
                    let mut sum = u32::from(source[center + channel]) * 5;
                    for dy in [0, row_bytes, row_bytes * 2] {
                        let upper_left = center - row_bytes - C + dy + channel;
                        sum += u32::from(source[upper_left]);
                        sum += u32::from(source[upper_left + C]);
                        sum += u32::from(source[upper_left + C * 2]);
                    }
                    // The center was included once by the loop and needs a
                    // total weight of five. Pillow adds 0.5 before truncation.
                    sum -= u32::from(source[center + channel]);
                    row[x * C + channel] = ((sum + 6) / 13) as u8;
                }
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blend_truncates_and_clips_like_pillow() {
        let first = [100, 200, 10];
        let second = [200, 0, 20];
        let mut output = [0; 3];
        blend_bytes(&first, &second, &mut output, 0.5);
        assert_eq!(output, [150, 100, 15]);
        blend_bytes(&first, &second, &mut output, 2.0);
        assert_eq!(output, [255, 0, 30]);
        blend_bytes(&first, &second, &mut output, f32::NAN);
        assert_eq!(output, [0, 0, 0]);
    }

    #[test]
    fn blend_simd_matches_scalar_with_tail() {
        let first: Vec<u8> = (0..35).collect();
        let second: Vec<u8> = (0..35).rev().collect();
        for &factor in &[0.0, 0.25, 0.5, 0.75, 1.0, -0.5, 2.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut expected = vec![0u8; 35];
            blend_bytes_scalar(&first, &second, &mut expected, factor);
            let mut actual = vec![0u8; 35];
            blend_bytes(&first, &second, &mut actual, factor);
            assert_eq!(actual, expected, "factor={factor}");
        }
    }

    #[test]
    fn smooth_copies_borders_and_preserves_alpha() {
        let source: Vec<u8> = (0..100).collect();
        let mut output = source.clone();
        smooth::<4>(&source, &mut output, 5);
        assert_eq!(&output[..20], &source[..20]);
        assert_eq!(&output[80..], &source[80..]);
        for (actual, expected) in output.as_chunks::<4>().0.iter().zip(source.as_chunks::<4>().0) {
            assert_eq!(actual[3], expected[3]);
        }
    }
}
