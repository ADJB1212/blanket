#![feature(portable_simd)]

use blanket_core::parallel::{CHUNK_PIXELS, chunks_mut};
use blanket_core::raster::{Image, PixelMode, alloc_pixels};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

mod high_depth;
mod simd;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlurKind {
    Box,
    Gaussian,
}

fn output(image: &Image, pixels: Vec<u8>) -> PyResult<Image> {
    let mut result = Image::from_pixels(image.width(), image.height(), image.pixel_mode(), pixels, None)?;
    result.copy_palette_from(image);
    Ok(result)
}

fn gaussian_radius(radius: f32) -> f32 {
    let variance = radius * radius / 3.0;
    let length = (12.0 * f64::from(variance) + 1.0).sqrt() as f32;
    let integer = ((f64::from(length) - 1.0) / 2.0).floor() as f32;
    let mut fraction = (2.0 * integer + 1.0) * (integer * (integer + 1.0) - 3.0 * variance);
    fraction /= 6.0 * (variance - (integer + 1.0) * (integer + 1.0));
    integer + fraction
}

// Fractional box-filter weights use 24 bits and round at half a unit.
pub(crate) const BOX_WEIGHT_BITS: u32 = 24;
pub(crate) const BOX_WEIGHT_SCALE: u32 = 1 << BOX_WEIGHT_BITS;
pub(crate) const BOX_WEIGHT_ROUND: u64 = 1 << (BOX_WEIGHT_BITS - 1);

#[inline]
pub(crate) fn box_value(sum: u32, ends: u32, weight: u32, far_weight: u32) -> u8 {
    // Only bits 24..32 reach the output; wrapping u32 retains them.
    (sum.wrapping_mul(weight).wrapping_add(ends.wrapping_mul(far_weight)).wrapping_add(BOX_WEIGHT_ROUND as u32) >> BOX_WEIGHT_BITS) as u8
}

fn box_horizontal<const C: usize>(source: &[u8], pixels: &mut [u8], width: usize, radius: f32) {
    let integer = radius as usize;
    let weight = (BOX_WEIGHT_SCALE as f32 / (radius * 2.0 + 1.0)) as u32;
    let far_weight = BOX_WEIGHT_SCALE.wrapping_sub((2 * integer as u32 + 1).wrapping_mul(weight)) / 2;
    let stride = width * C;
    if integer <= 2 {
        chunks_mut(pixels, stride * 16, |band, rows| {
            for (i, row) in rows.chunks_exact_mut(stride).enumerate() {
                let input = &source[(band * 16 + i) * stride..][..stride];
                match integer {
                    0 => crate::simd::box_row::<C, 0>(input, row, weight, far_weight),
                    1 => crate::simd::box_row::<C, 1>(input, row, weight, far_weight),
                    _ => crate::simd::box_row::<C, 2>(input, row, weight, far_weight),
                }
            }
        });
        return;
    }
    let source = source.as_chunks::<C>().0;
    chunks_mut(pixels, stride * 16, |band, rows| {
        for (i, row) in rows.chunks_exact_mut(stride).enumerate() {
            let input = &source[(band * 16 + i) * width..][..width];
            let row = row.as_chunks_mut::<C>().0;
            let mut sums = input[0].map(|value| u32::from(value).wrapping_mul(integer as u32 + 1));
            for pixel in &input[1..=integer.min(width - 1)] {
                for channel in 0..C {
                    sums[channel] = sums[channel].wrapping_add(u32::from(pixel[channel]));
                }
            }
            for channel in 0..C {
                let repeat = u32::from(input[width - 1][channel]).wrapping_mul(integer.saturating_sub(width - 1) as u32);
                sums[channel] = sums[channel].wrapping_add(repeat);
            }
            for (x, pixel) in row.iter_mut().enumerate() {
                let left = x.saturating_sub(integer + 1);
                let right = (x + integer + 1).min(width - 1);
                let leaving = x.saturating_sub(integer);
                for channel in 0..C {
                    let ends = u32::from(input[left][channel]) + u32::from(input[right][channel]);
                    pixel[channel] = box_value(sums[channel], ends, weight, far_weight);
                    sums[channel] = sums[channel].wrapping_sub(u32::from(input[leaving][channel])).wrapping_add(u32::from(input[right][channel]));
                }
            }
        }
    });
}

fn box_vertical(source: &[u8], pixels: &mut [u8], stride: usize, radius: f32) {
    let height = source.len() / stride;
    let integer = radius as usize;
    let weight = (BOX_WEIGHT_SCALE as f32 / (radius * 2.0 + 1.0)) as u32;
    let far_weight = BOX_WEIGHT_SCALE.wrapping_sub((2 * integer as u32 + 1).wrapping_mul(weight)) / 2;
    if integer <= 2 {
        chunks_mut(pixels, stride * 16, |band, rows| {
            for (i, row) in rows.chunks_exact_mut(stride).enumerate() {
                let y = band * 16 + i;
                match integer {
                    0 => crate::simd::box_column::<0>(source, row, y, weight, far_weight),
                    1 => crate::simd::box_column::<1>(source, row, y, weight, far_weight),
                    _ => crate::simd::box_column::<2>(source, row, y, weight, far_weight),
                }
            }
        });
        return;
    }
    // Each band maintains one sum per byte column, allowing contiguous loads
    // and vectorization across columns without transposing the image.
    // Grow bands with the radius so rebuilding their initial sums stays
    // linear in image size, even when the radius exceeds the image height.
    let band_rows = 64.max(integer.min(height));
    chunks_mut(pixels, stride * band_rows, |band, rows| {
        let first = band * band_rows;
        let top = first.saturating_sub(integer);
        let bottom = (first + integer).min(height - 1);
        let mut sums = vec![0_u32; stride];
        for row in source[top * stride..(bottom + 1) * stride].chunks_exact(stride) {
            for (sum, &value) in sums.iter_mut().zip(row) {
                *sum = sum.wrapping_add(u32::from(value));
            }
        }
        let upper_repeat = integer.saturating_sub(first) as u32;
        let lower_repeat = (first + integer).saturating_sub(height - 1) as u32;
        for ((sum, &upper), &lower) in sums.iter_mut().zip(&source[..stride]).zip(&source[(height - 1) * stride..]) {
            *sum = sum.wrapping_add(u32::from(upper).wrapping_mul(upper_repeat)).wrapping_add(u32::from(lower).wrapping_mul(lower_repeat));
        }
        for (i, row) in rows.chunks_exact_mut(stride).enumerate() {
            let y = first + i;
            let left = y.saturating_sub(integer + 1) * stride;
            let right = (y + integer + 1).min(height - 1) * stride;
            let leaving = y.saturating_sub(integer) * stride;
            let upper = &source[left..left + stride];
            let lower = &source[right..right + stride];
            let leaving = &source[leaving..leaving + stride];
            for ((((dst, sum), &upper), &lower), &leaving) in row.iter_mut().zip(&mut sums).zip(upper).zip(lower).zip(leaving) {
                let ends = u32::from(upper) + u32::from(lower);
                *dst = box_value(*sum, ends, weight, far_weight);
                *sum = sum.wrapping_sub(u32::from(leaving)).wrapping_add(u32::from(lower));
            }
        }
    });
}

pub(crate) fn blur_radii(radii: (f32, f32), kind: BlurKind) -> PyResult<(f32, f32)> {
    if <[f32; 2]>::from(radii).iter().any(|value| !value.is_finite() || value.abs() >= i32::MAX as f32) {
        return Err(PyValueError::new_err("radius must be finite and less than 2**31"));
    }
    let radii = if kind == BlurKind::Gaussian { (gaussian_radius(radii.0), gaussian_radius(radii.1)) } else { radii };
    if <[f32; 2]>::from(radii).iter().any(|r| !r.is_finite() || *r < 0.0 || *r >= i32::MAX as f32) {
        return Err(PyValueError::new_err("radius must be finite and >= 0"));
    }
    Ok(radii)
}

fn blurred<const C: usize>(source: &[u8], width: usize, height: usize, radii: (f32, f32), kind: BlurKind) -> PyResult<Vec<u8>> {
    debug_assert_eq!(source.len(), width * height * C);
    let radii = blur_radii(radii, kind)?;
    let mut pixels = alloc_pixels(source.len())?;
    if pixels.is_empty() {
        return Ok(pixels);
    }
    let passes = if kind == BlurKind::Gaussian { 3 } else { 1 };
    let horizontal_passes = if radii.0 > 0.0 { passes } else { 0 };
    let vertical_passes = if radii.1 > 0.0 { passes } else { 0 };
    if horizontal_passes + vertical_passes == 0 {
        pixels.copy_from_slice(source);
        return Ok(pixels);
    }
    if radii.0 > 0.0 {
        box_horizontal::<C>(source, &mut pixels, width, radii.0);
    } else {
        box_vertical(source, &mut pixels, width * C, radii.1);
    }
    if horizontal_passes + vertical_passes == 1 {
        return Ok(pixels);
    }
    let mut scratch = alloc_pixels(source.len())?;
    for _ in 1..horizontal_passes {
        box_horizontal::<C>(&pixels, &mut scratch, width, radii.0);
        std::mem::swap(&mut pixels, &mut scratch);
    }
    for _ in usize::from(horizontal_passes == 0)..vertical_passes {
        box_vertical(&pixels, &mut scratch, width * C, radii.1);
        std::mem::swap(&mut pixels, &mut scratch);
    }
    Ok(pixels)
}

fn blur(image: &Image, source: &[u8], radii: (f32, f32), kind: BlurKind) -> PyResult<Vec<u8>> {
    let (width, height) = (image.width() as usize, image.height() as usize);
    match image.pixel_mode() {
        PixelMode::One | PixelMode::L => blurred::<1>(source, width, height, radii, kind),
        PixelMode::La | PixelMode::PremulLa | PixelMode::Pa => blurred::<2>(source, width, height, radii, kind),
        PixelMode::Rgb | PixelMode::Hsv | PixelMode::YCbCr | PixelMode::Lab => blurred::<3>(source, width, height, radii, kind),
        PixelMode::Rgba | PixelMode::PremulRgba | PixelMode::Rgbx | PixelMode::Cmyk => blurred::<4>(source, width, height, radii, kind),
        _ => unreachable!(),
    }
}

#[pyfunction(name = "filter_blur")]
pub fn filter_blur_python(py: Python<'_>, image: &Image, radii: (f32, f32), gaussian: bool) -> PyResult<Image> {
    filter_blur(py, image, radii, if gaussian { BlurKind::Gaussian } else { BlurKind::Box })
}

pub fn filter_blur(py: Python<'_>, image: &Image, radii: (f32, f32), kind: BlurKind) -> PyResult<Image> {
    if blanket_core::samples::supported(image) {
        if radii == (0.0, 0.0) {
            return image.copy(py);
        }
        return py.detach(|| crate::high_depth::blur(image, radii, kind));
    }
    let source = image.pixel_data()?;
    if radii == (0.0, 0.0) {
        return image.copy(py);
    }
    let pixels = py.detach(|| blur(image, source, radii, kind))?;
    output(image, pixels)
}

#[pyfunction]
pub fn filter_unsharp(py: Python<'_>, image: &Image, radius: f32, percent: i32, threshold: i32) -> PyResult<Image> {
    if blanket_core::samples::supported(image) {
        if percent == 0 || threshold >= i32::from(blanket_core::samples::maximum(image)) || radius == 0.0 {
            blur_radii((radius, radius), BlurKind::Gaussian)?;
            return image.copy(py);
        }
        return py.detach(|| crate::high_depth::unsharp(image, radius, percent, threshold));
    }
    let source = image.pixel_data()?;
    if percent == 0 || threshold >= 255 || radius == 0.0 {
        if !source.is_empty() {
            blur_radii((radius, radius), BlurKind::Gaussian)?;
        }
        return image.copy(py);
    }
    let pixels = py.detach(|| -> PyResult<Vec<u8>> {
        let mut pixels = blur(image, source, (radius, radius), BlurKind::Gaussian)?;
        chunks_mut(&mut pixels, CHUNK_PIXELS, |chunk, dst| sharpen(&source[chunk * CHUNK_PIXELS..], dst, percent, threshold));
        Ok(pixels)
    })?;
    output(image, pixels)
}

fn sharpen(source: &[u8], values: &mut [u8], percent: i32, threshold: i32) {
    use std::simd::prelude::*;
    // Beyond this magnitude every nonzero difference saturates, so i32 lanes stay exact.
    let percent = percent.clamp(-25_600, 25_600);
    let end = values.len() / 16 * 16;
    for i in (0..end).step_by(16) {
        let original = Simd::<u8, 16>::from_slice(&source[i..]).cast::<i32>();
        let difference = original - Simd::<u8, 16>::from_slice(&values[i..]).cast::<i32>();
        let value = (original + difference * Simd::splat(percent) / Simd::splat(100)).simd_clamp(Simd::splat(0), Simd::splat(255));
        difference.abs().simd_gt(Simd::splat(threshold)).select(value, original).cast::<u8>().copy_to_slice(&mut values[i..i + 16]);
    }
    for (&original, value) in source[end..].iter().zip(&mut values[end..]) {
        let difference = i32::from(original) - i32::from(*value);
        *value = if difference.abs() > threshold { (i32::from(original) + difference * percent / 100).clamp(0, 255) as u8 } else { original };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn box_blur_replicates_edges_and_rounds() {
        let mut result = [0; 3];
        box_horizontal::<1>(&[0, 90, 0], &mut result, 3, 1.0);
        assert_eq!(result, [30; 3]);
        box_horizontal::<1>(&[77; 3], &mut result, 3, 100.5);
        assert_eq!(result, [77; 3]);
        let mut rgb = [0; 9];
        box_horizontal::<3>(&[0, 90, 3, 90, 0, 3, 0, 90, 3], &mut rgb, 3, 1.0);
        assert_eq!(rgb, [30, 60, 3, 30, 60, 3, 30, 60, 3]);
    }

    #[test]
    fn sharpen_matches_wide_reference_for_tails_and_extreme_percents() {
        for count in [0, 1, 15, 16, 17, 31, 33, 65] {
            let source: Vec<u8> = (0..=count).map(|i| (i * 719 + 257) as u8).collect();
            for percent in [i32::MIN, -25_601, -150, 0, 1, 150, 25_601, i32::MAX] {
                for threshold in [-1, 0, 17, i32::MAX] {
                    let mut output = vec![173; count + 2];
                    for (i, v) in output[1..=count].iter_mut().enumerate() {
                        *v = source[count - i];
                    }
                    let original = output.clone();
                    sharpen(&source[1..], &mut output[1..=count], percent, threshold);
                    for i in 0..count {
                        let a = i64::from(source[1 + i]);
                        let difference = a - i64::from(original[1 + i]);
                        let expected = if difference.abs() > i64::from(threshold) {
                            (a + difference * i64::from(percent) / 100).clamp(0, 255) as u8
                        } else {
                            a as u8
                        };
                        assert_eq!(output[1 + i], expected);
                    }
                    assert_eq!((output[0], output[count + 1]), (173, 173));
                }
            }
        }
    }

    #[test]
    fn vertical_blur_matches_horizontal_pass() {
        // A 1x3 column blurred vertically equals the 3x1 row blurred horizontally.
        let column = blurred::<1>(&[0, 90, 0], 1, 3, (0.0, 1.0), BlurKind::Box).unwrap();
        let row = blurred::<1>(&[0, 90, 0], 3, 1, (1.0, 0.0), BlurKind::Box).unwrap();
        assert_eq!(column, row);
        assert_eq!(column, [30; 3]);
    }

    fn reference_box_line(input: &[u8], radius: f32) -> Vec<u8> {
        let integer = radius as usize;
        let weight = (BOX_WEIGHT_SCALE as f32 / (radius * 2.0 + 1.0)) as u64;
        let far_weight = u64::from(BOX_WEIGHT_SCALE.wrapping_sub((2 * integer as u32 + 1).wrapping_mul(weight as u32)) / 2);
        (0..input.len())
            .map(|x| {
                let start = x.saturating_sub(integer);
                let end = (x + integer).min(input.len() - 1);
                let sum = input[start..=end].iter().map(|&v| u64::from(v)).sum::<u64>()
                    + u64::from(input[0]) * integer.saturating_sub(x) as u64
                    + u64::from(input[input.len() - 1]) * (x + integer).saturating_sub(input.len() - 1) as u64;
                let ends = u64::from(input[x.saturating_sub(integer + 1)]) + u64::from(input[(x + integer + 1).min(input.len() - 1)]);
                ((sum * weight + ends * far_weight + BOX_WEIGHT_ROUND) >> BOX_WEIGHT_BITS) as u8
            })
            .collect()
    }

    fn check_box_passes<const C: usize>() {
        for width in [1, 2, 7, 15, 16, 17, 31, 33, 65] {
            for height in [1, 3, 17, 65] {
                let len = width * height * C;
                for white in [false, true] {
                    let storage: Vec<u8> = (0..len + 3).map(|i| if white { 255 } else { (i * 71 + i / 7 * 113) as u8 }).collect();
                    let source = &storage[3..];
                    for radius in [0.0, 0.1, 0.5, 1.0, 1.25, 2.9, 16.0, 100.5, 8_500_000.0, 2_000_000_000.0] {
                        let mut horizontal = vec![93; len + 5];
                        let mut vertical = horizontal.clone();
                        box_horizontal::<C>(source, &mut horizontal[1..len + 1], width, radius);
                        box_vertical(source, &mut vertical[1..len + 1], width * C, radius);
                        for result in [&horizontal, &vertical] {
                            assert_eq!(result[0], 93);
                            assert_eq!(&result[len + 1..], &[93; 4]);
                        }
                        for y in 0..height {
                            for c in 0..C {
                                let input: Vec<u8> = (0..width).map(|x| source[(y * width + x) * C + c]).collect();
                                for (x, expected) in reference_box_line(&input, radius).into_iter().enumerate() {
                                    assert_eq!(horizontal[1 + (y * width + x) * C + c], expected);
                                }
                            }
                        }
                        for x in 0..width * C {
                            let input: Vec<u8> = (0..height).map(|y| source[y * width * C + x]).collect();
                            for (y, expected) in reference_box_line(&input, radius).into_iter().enumerate() {
                                assert_eq!(vertical[1 + y * width * C + x], expected);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn box_passes_match_wide_reference_with_large_radii_and_guards() {
        check_box_passes::<1>();
        check_box_passes::<2>();
        check_box_passes::<3>();
        check_box_passes::<4>();
    }

    #[test]
    fn small_vertical_blur_preserves_parallel_bands_and_guards() {
        for channels in 1..=4 {
            let (width, height) = (257, 263);
            let stride = width * channels;
            let len = stride * height;
            let storage: Vec<_> = (0..len + 1).map(|i| (i * 71 + i / 7 * 113) as u8).collect();
            let source = &storage[1..];
            for radius in [0.25, 1.5, 2.75] {
                let mut output = vec![173; len + 2];
                box_vertical(source, &mut output[1..len + 1], stride, radius);
                for x in 0..stride {
                    let column: Vec<_> = (0..height).map(|y| source[y * stride + x]).collect();
                    for (y, expected) in reference_box_line(&column, radius).into_iter().enumerate() {
                        assert_eq!(output[1 + y * stride + x], expected);
                    }
                }
                assert_eq!((output[0], output[len + 1]), (173, 173));
            }
        }
    }
}
