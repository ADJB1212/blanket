//! Native primitives used by the Python `ImageEnhance` API.

use blanket_core::parallel::{CHUNK_PIXELS, MIN_PARALLEL_BYTES, chunks_mut, chunks_mut_above, should_parallel};
use blanket_core::raster::{Image, PixelMode};
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
    use std::simd::Simd;
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
        blended.cast::<u8>().copy_to_slice(&mut output[i..i + 16]);
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

    let mut pixels = buffer(source.len())?;
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

    let mut pixels = buffer(source.len())?;
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
            for x in 1..width - 1 {
                let center = y * row_bytes + x * C;
                for channel in 0..B {
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
    fn blend_preserves_fractional_rounding() {
        let first: Vec<u8> = (0..=255).flat_map(|a| [a; 256]).collect();
        let second: Vec<u8> = (0..=255).cycle().take(first.len()).collect();
        let mut actual = vec![0; first.len()];
        for factor in [0.1_f32, 0.6, -0.6, 1.6] {
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
