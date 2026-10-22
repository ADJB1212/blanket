//! Native primitives used by the Python `ImageEnhance` API.

use std::simd::Simd;
use std::simd::num::SimdUint;

use blanket_core::color::{LUMA_BITS, LUMA_BLUE, LUMA_GREEN, LUMA_RED, LUMA_ROUND};
use blanket_core::parallel::{CHUNK_PIXELS, MIN_PARALLEL_BYTES, chunks_mut, chunks_mut_above, should_parallel};
use blanket_core::pixels::{Bytes, load};
use blanket_core::raster::{Image, PixelMode, alloc_pixels};
use pyo3::exceptions::{PyMemoryError, PyValueError};
use pyo3::prelude::*;
use rayon::prelude::*;

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(enhance_blend, module)?)?;
    module.add_function(wrap_pyfunction!(enhance_color, module)?)?;
    module.add_function(wrap_pyfunction!(enhance_contrast, module)?)?;
    module.add_function(wrap_pyfunction!(enhance_brightness, module)?)?;
    module.add_function(wrap_pyfunction!(enhance_sharpness, module)?)?;
    Ok(())
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

    let mut pixels = alloc_pixels(first_pixels.len())?;
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
    use std::simd::Simd;
    #[cfg(not(target_arch = "aarch64"))]
    use std::simd::cmp::SimdOrd;
    #[cfg(not(target_arch = "aarch64"))]
    use std::simd::num::SimdInt;
    use std::simd::num::{SimdFloat, SimdUint};
    let done = first.len().min(second.len()).min(output.len()) / 16 * 16;
    for i in (0..done).step_by(16) {
        let a = Simd::<u8, 16>::from_slice(&first[i..i + 16]).cast::<f32>();
        let b = Simd::<u8, 16>::from_slice(&second[i..i + 16]).cast::<f32>();
        #[cfg(target_arch = "aarch64")]
        let blended = {
            // Preserve the fused rounding of the previous ARM kernel.
            use std::simd::StdFloat;
            Simd::splat(factor).mul_add(b - a, a)
        };
        #[cfg(not(target_arch = "aarch64"))]
        let blended = a + Simd::splat(factor) * (b - a);
        #[cfg(target_arch = "aarch64")]
        let blended = blended.cast::<u8>();
        #[cfg(not(target_arch = "aarch64"))]
        let blended = blended
            .cast::<i32>()
            .simd_clamp(Simd::splat(-32768), Simd::splat(32767))
            .cast::<i16>()
            .simd_clamp(Simd::splat(0), Simd::splat(255))
            .cast::<u8>();
        blended.copy_to_slice(&mut output[i..i + 16]);
    }
    blend_bytes_scalar(&first[done..], &second[done..], &mut output[done..], factor);
}

fn blend_bytes_scalar(first: &[u8], second: &[u8], output: &mut [u8], factor: f32) {
    for ((&a, &b), dst) in first.iter().zip(second).zip(output) {
        *dst = (f32::from(a) + factor * (f32::from(b) - f32::from(a))) as u8;
    }
}

#[pyfunction]
fn enhance_color(py: Python<'_>, image: &Image) -> PyResult<Image> {
    if matches!(image.mode, PixelMode::Cmyk | PixelMode::YCbCr | PixelMode::Lab) {
        return image.convert(py, "L", None)?.convert(py, image.mode.as_str(), None);
    }
    let source = image.pixel_data()?;
    if matches!(image.mode, PixelMode::One | PixelMode::L | PixelMode::La | PixelMode::Pa) {
        return copy(image, source);
    }

    let mut pixels = alloc_pixels(source.len())?;
    py.detach(|| match image.mode {
        PixelMode::One | PixelMode::L | PixelMode::La | PixelMode::Pa => unreachable!(),
        PixelMode::Rgb | PixelMode::Hsv => color_degenerate::<3>(source, &mut pixels),
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

#[inline]
fn luminance(r: Bytes, g: Bytes, b: Bytes) -> Bytes {
    ((r.cast::<u32>() * Simd::splat(LUMA_RED)
        + g.cast::<u32>() * Simd::splat(LUMA_GREEN)
        + b.cast::<u32>() * Simd::splat(LUMA_BLUE)
        + Simd::splat(LUMA_ROUND))
        >> LUMA_BITS)
        .cast()
}

#[pyfunction]
fn enhance_contrast(py: Python<'_>, image: &Image) -> PyResult<Image> {
    if matches!(image.mode, PixelMode::Cmyk | PixelMode::YCbCr | PixelMode::Lab) {
        let gray = image.convert(py, "L", None)?;
        let source = gray.pixel_data()?;
        let mean = if source.is_empty() {
            0
        } else {
            (byte_sum(source) as f64 / source.len() as f64 + 0.5) as u8
        };
        let gray = Image::from_pixels(image.width, image.height, PixelMode::L, vec![mean; source.len()], None)?;
        return gray.convert(py, image.mode.as_str(), None);
    }
    let source = image.pixel_data()?;
    let sum = py.detach(|| match image.mode {
        PixelMode::One | PixelMode::L => byte_sum(source),
        PixelMode::La | PixelMode::Pa => source.as_chunks::<2>().0.iter().map(|pixel| u64::from(pixel[0])).sum(),
        PixelMode::Rgb | PixelMode::Hsv => luminance_sum::<3>(source),
        PixelMode::Rgba => luminance_sum::<4>(source),
        _ => unreachable!(),
    });
    let pixel_count = source.len() / image.mode.channels();
    let mean = if pixel_count == 0 {
        0
    } else {
        (sum as f64 / pixel_count as f64 + 0.5) as u8
    };

    let mut pixels = alloc_pixels(source.len())?;
    py.detach(|| match image.mode {
        PixelMode::One | PixelMode::L => pixels.fill(mean),
        PixelMode::La | PixelMode::Pa => {
            for (src, dst) in source.as_chunks::<2>().0.iter().zip(pixels.as_chunks_mut::<2>().0) {
                dst.copy_from_slice(&[mean, src[1]]);
            }
        }
        PixelMode::Rgb | PixelMode::Hsv => chunks_mut(&mut pixels, CHUNK_PIXELS * 3, |_, dst| {
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
    let sum = |source: &[u8]| {
        let done = source.len() / (16 * C) * 16 * C;
        let mut sum = 0_u64;
        for i in (0..done).step_by(16 * C) {
            let channels = load::<C>(&source[i..]);
            sum += u64::from(luminance(channels[0], channels[1], channels[2]).cast::<u32>().reduce_sum());
        }
        sum + source[done..]
            .as_chunks::<C>()
            .0
            .iter()
            .map(|p| u64::from(blanket_core::simd::pillow_luma(p[0], p[1], p[2])))
            .sum::<u64>()
    };
    if should_parallel(source.len(), C, MIN_PARALLEL_BYTES) {
        source.par_chunks(CHUNK_PIXELS * C).map(sum).sum()
    } else {
        sum(source)
    }
}

#[pyfunction]
fn enhance_brightness(py: Python<'_>, image: &Image) -> PyResult<Image> {
    let mut pixels = alloc_pixels(
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
    let mut pixels = alloc_pixels(source.len())?;
    pixels.copy_from_slice(source);
    if image.width < 3 || image.height < 3 {
        return output(image, pixels);
    }

    let width = image.width as usize;
    py.detach(|| match image.mode {
        PixelMode::One | PixelMode::L => smooth::<1, 1>(source, &mut pixels, width),
        PixelMode::La | PixelMode::Pa => smooth::<2, 2>(source, &mut pixels, width),
        PixelMode::Rgb | PixelMode::Hsv | PixelMode::YCbCr | PixelMode::Lab => smooth::<3, 3>(source, &mut pixels, width),
        PixelMode::Cmyk => smooth::<4, 4>(source, &mut pixels, width),
        PixelMode::Rgba => smooth::<4, 3>(source, &mut pixels, width),
        _ => unreachable!(),
    });
    output(image, pixels)
}

fn smooth<const C: usize, const B: usize>(source: &[u8], output: &mut [u8], width: usize) {
    let row_bytes = width * C;
    let height = source.len() / row_bytes;
    chunks_mut_above(output, row_bytes * 32, MIN_PARALLEL_BYTES, |band, rows| {
        for (i, row) in rows.chunks_exact_mut(row_bytes).enumerate() {
            let y = band * 32 + i;
            if y == 0 || y + 1 >= height {
                continue;
            }
            let done = C + (row_bytes - C * 2) / 16 * 16;
            for x in (C..done).step_by(16) {
                let center = y * row_bytes + x;
                let original = Bytes::from_slice(&source[center..]);
                let mut sum = original.cast::<u16>() * Simd::splat(4);
                for dy in [0, row_bytes, row_bytes * 2] {
                    let left = center - row_bytes - C + dy;
                    for dx in [0, C, C * 2] {
                        sum += Bytes::from_slice(&source[left + dx..]).cast::<u16>();
                    }
                }
                let filtered = ((sum + Simd::splat(6)) / Simd::splat(13)).cast::<u8>();
                let filtered = if B == C {
                    filtered
                } else {
                    let mask = Bytes::from_array(std::array::from_fn(|i| if (x + i) % C < B { 255 } else { 0 }));
                    (filtered & mask) | (original & !mask)
                };
                filtered.copy_to_slice(&mut row[x..x + 16]);
            }
            for (x, dst) in row.iter_mut().enumerate().take(row_bytes - C).skip(done) {
                if x % C < B {
                    let center = y * row_bytes + x;
                    let mut sum = u32::from(source[center]) * 4;
                    for dy in [0, row_bytes, row_bytes * 2] {
                        let left = center - row_bytes - C + dy;
                        for dx in [0, C, C * 2] {
                            sum += u32::from(source[left + dx]);
                        }
                    }
                    *dst = ((sum + 6) / 13) as u8;
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
    fn blend_preserves_fractional_rounding() {
        let first: Vec<u8> = (0..=255).flat_map(|a| [a; 256]).collect();
        let second: Vec<u8> = (0..=255).cycle().take(first.len()).collect();
        let mut actual = vec![0; first.len()];
        for factor in [0.1_f32, 0.35, 0.6, -0.6, 1.6] {
            blend_bytes(&first, &second, &mut actual, factor);
            for ((&a, &b), &value) in first.iter().zip(&second).zip(&actual) {
                let (a, b) = (f32::from(a), f32::from(b));
                #[cfg(target_arch = "aarch64")]
                let expected = factor.mul_add(b - a, a) as u8;
                #[cfg(not(target_arch = "aarch64"))]
                let expected = (a + factor * (b - a)) as u8;
                assert_eq!(value, expected);
            }
        }
    }

    #[test]
    fn blend_unaligned_tails_and_guards() {
        for count in [0, 1, 15, 16, 17, 31, 32, 33, 257] {
            for offset in 0..16 {
                let first: Vec<u8> = (0..count + offset).map(|i| (i * 37) as u8).collect();
                let second: Vec<u8> = (0..count + offset).map(|i| (i * 53 + 17) as u8).collect();
                for factor in [0.35_f32, -1.5, 2.5, f32::MAX, f32::NAN, f32::INFINITY] {
                    let mut output = vec![93; offset + count + 16];
                    blend_bytes(&first[offset..], &second[offset..], &mut output[offset..offset + count], factor);
                    for i in 0..count {
                        let a = f32::from(first[offset + i]);
                        let b = f32::from(second[offset + i]);
                        #[cfg(target_arch = "aarch64")]
                        let expected = if i < count / 16 * 16 {
                            factor.mul_add(b - a, a)
                        } else {
                            a + factor * (b - a)
                        };
                        #[cfg(not(target_arch = "aarch64"))]
                        let expected = a + factor * (b - a);
                        assert_eq!(output[offset + i], expected as u8);
                    }
                    assert!(output[..offset].iter().all(|&v| v == 93));
                    assert!(output[offset + count..].iter().all(|&v| v == 93));
                }
            }
        }
    }

    #[test]
    fn luminance_sum_matches_scalar_with_unaligned_buffers_and_tails() {
        fn check<const C: usize>(count: usize, offset: usize) {
            for fill in [None, Some(0), Some(255)] {
                let source: Vec<u8> = (0..offset + count * C).map(|i| fill.unwrap_or((i * 37 + i / 11) as u8)).collect();
                let source = &source[offset..];
                let sum: u64 = source
                    .as_chunks::<C>()
                    .0
                    .iter()
                    .map(|p| u64::from(blanket_core::simd::pillow_luma(p[0], p[1], p[2])))
                    .sum();
                assert_eq!(luminance_sum::<C>(source), sum);
            }
        }
        for offset in 0..16 {
            for count in [0, 1, 15, 16, 17, 31, 32, 33, 65] {
                check::<3>(count, offset);
                check::<4>(count, offset);
            }
        }
        for count in [CHUNK_PIXELS - 1, CHUNK_PIXELS, CHUNK_PIXELS + 1, MIN_PARALLEL_BYTES + 17] {
            check::<3>(count, 1);
            check::<4>(count, 1);
        }
    }

    #[test]
    fn smooth_matches_scalar_with_unaligned_buffers_tails_and_guards() {
        fn check<const C: usize, const B: usize>(width: usize, height: usize, offset: usize) {
            for fill in [None, Some(0), Some(255)] {
                let source: Vec<u8> = (0..offset + width * height * C + 16)
                    .map(|i| fill.unwrap_or((i * 37 + i / 11) as u8))
                    .collect();
                let mut expected = source.clone();
                for y in 1..height - 1 {
                    for x in 1..width - 1 {
                        for c in 0..B {
                            let mut sum = 0_u32;
                            for dy in 0..3 {
                                for dx in 0..3 {
                                    let weight = if dx == 1 && dy == 1 { 5 } else { 1 };
                                    sum += u32::from(source[offset + ((y + dy - 1) * width + x + dx - 1) * C + c]) * weight;
                                }
                            }
                            expected[offset + (y * width + x) * C + c] = ((sum + 6) / 13) as u8;
                        }
                    }
                }
                let mut output = source.clone();
                let end = offset + width * height * C;
                smooth::<C, B>(&source[offset..end], &mut output[offset..end], width);
                assert_eq!(output, expected, "C={C}, B={B}, width={width}, height={height}, offset={offset}");
            }
        }
        for offset in 0..16 {
            for width in 3..36 {
                check::<1, 1>(width, 5, offset);
                check::<2, 2>(width, 5, offset);
                check::<3, 3>(width, 5, offset);
                check::<4, 4>(width, 5, offset);
                check::<4, 3>(width, 5, offset);
            }
        }
        check::<1, 1>(521, 509, 1);
        check::<2, 2>(521, 509, 1);
        check::<3, 3>(521, 509, 1);
        check::<4, 4>(521, 509, 1);
        check::<4, 3>(521, 509, 1);
    }

    #[test]
    fn smooth_copies_borders_and_preserves_alpha() {
        let source: Vec<u8> = (0..100).collect();
        let mut output = source.clone();
        smooth::<4, 3>(&source, &mut output, 5);
        assert_eq!(&output[..20], &source[..20]);
        assert_eq!(&output[80..], &source[80..]);
        for (actual, expected) in output.as_chunks::<4>().0.iter().zip(source.as_chunks::<4>().0) {
            assert_eq!(actual[3], expected[3]);
        }
    }
}
