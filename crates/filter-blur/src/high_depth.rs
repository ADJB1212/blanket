use std::simd::cmp::{SimdOrd, SimdPartialOrd};
use std::simd::num::{SimdFloat, SimdInt, SimdUint};
use std::simd::{Select, Simd};

use blanket_core::pixels::Samples;
use blanket_core::{Image, samples};
use pyo3::prelude::*;

use super::BlurKind;

fn box_pass(source: &[u16], output: &mut [u16], width: usize, height: usize, channels: usize, radius: f32, vertical: bool) {
    if output.is_empty() {
        return;
    }
    if vertical {
        box_vertical(source, output, width * channels, radius);
        return;
    }
    if radius < 3.0 {
        match channels {
            1 => box_horizontal_small::<1>(source, output, width, radius),
            3 => box_horizontal_small::<3>(source, output, width, radius),
            4 => box_horizontal_small::<4>(source, output, width, radius),
            _ => unreachable!(),
        }
        return;
    }
    let integer = radius as usize;
    let weight = (super::BOX_WEIGHT_SCALE as f32 / (radius * 2.0 + 1.0)) as u64;
    let far_weight = u64::from(super::BOX_WEIGHT_SCALE.wrapping_sub((2 * integer as u32 + 1).wrapping_mul(weight as u32)) / 2);
    let (length, lines, step) = if vertical { (height, width * channels, width * channels) } else { (width, height * channels, channels) };
    let end = lines / 16 * 16;
    for line in (0..end).step_by(16) {
        let bases = Simd::<usize, 16>::from_array(std::array::from_fn(|lane| {
            if vertical { line + lane } else { (line + lane) / channels * width * channels + (line + lane) % channels }
        }));
        let at = |i: usize| Samples::gather_or_default(source, bases + Simd::splat(i * step)).cast::<u64>();
        let mut sum = at(0) * Simd::splat(integer as u64 + 1) + at(length - 1) * Simd::splat(integer.saturating_sub(length - 1) as u64);
        for i in 1..=integer.min(length - 1) {
            sum += at(i);
        }
        for i in 0..length {
            let left = i.saturating_sub(integer + 1);
            let right = (i + integer + 1).min(length - 1);
            let value = (sum * Simd::splat(weight) + (at(left) + at(right)) * Simd::splat(far_weight) + Simd::splat(super::BOX_WEIGHT_ROUND))
                >> u64::from(super::BOX_WEIGHT_BITS);
            value.cast::<u16>().scatter(output, bases + Simd::splat(i * step));
            sum = sum - at(i.saturating_sub(integer)) + at(right);
        }
    }
    for line in end..lines {
        let start = if vertical { line } else { line / channels * width * channels + line % channels };
        let at = |i: usize| u64::from(source[start + i * step]);
        let mut sum = at(0) * (integer as u64 + 1) + at(length - 1) * integer.saturating_sub(length - 1) as u64;
        for i in 1..=integer.min(length - 1) {
            sum += at(i);
        }
        for i in 0..length {
            let left = i.saturating_sub(integer + 1);
            let right = (i + integer + 1).min(length - 1);
            output[start + i * step] =
                ((sum * weight + (at(left) + at(right)) * far_weight + super::BOX_WEIGHT_ROUND) >> super::BOX_WEIGHT_BITS) as u16;
            sum = sum - at(i.saturating_sub(integer)) + at(right);
        }
    }
}

fn box_horizontal_small<const C: usize>(source: &[u16], output: &mut [u16], width: usize, radius: f32) {
    let integer = radius as usize;
    let weight = (super::BOX_WEIGHT_SCALE as f32 / (radius * 2.0 + 1.0)) as u32;
    let far_weight = super::BOX_WEIGHT_SCALE.wrapping_sub((2 * integer as u32 + 1).wrapping_mul(weight)) / 2;
    let stride = width * C;
    blanket_core::parallel::chunks_mut_above(output, stride * 16, 256 * 1024, |band, rows| {
        for (i, row) in rows.chunks_exact_mut(stride).enumerate() {
            let input = &source[(band * 16 + i) * stride..][..stride];
            match integer {
                0 => box_sample_row::<C, 0>(input, row, weight, far_weight),
                1 => box_sample_row::<C, 1>(input, row, weight, far_weight),
                _ => box_sample_row::<C, 2>(input, row, weight, far_weight),
            }
        }
    });
}

fn box_sample_row<const C: usize, const R: usize>(source: &[u16], output: &mut [u16], weight: u32, far_weight: u32) {
    let width = source.len() / C;
    let first = ((R + 1) * C).min(source.len());
    let end = first + width.saturating_sub(2 * (R + 1)) * C / 16 * 16;
    let scale = 1.0 / f64::from(super::BOX_WEIGHT_SCALE);
    for x in (first..end).step_by(16) {
        let mut sum = Simd::<f64, 16>::splat(0.0);
        for tap in 0..2 * R + 1 {
            sum += Samples::from_slice(&source[x + tap * C - R * C..]).cast::<f64>();
        }
        let left = Samples::from_slice(&source[x - (R + 1) * C..]).cast::<f64>();
        let right = Samples::from_slice(&source[x + (R + 1) * C..]).cast::<f64>();
        let value =
            sum * Simd::splat(f64::from(weight)) + (left + right) * Simd::splat(f64::from(far_weight)) + Simd::splat(super::BOX_WEIGHT_ROUND as f64);
        (value * Simd::splat(scale)).cast::<u64>().cast::<u16>().copy_to_slice(&mut output[x..x + 16]);
    }
    for x in (0..first).chain(end..source.len()) {
        let sample = |tap: usize| u64::from(source[(x / C + tap).saturating_sub(R + 1).min(width - 1) * C + x % C]);
        let sum: u64 = (1..=2 * R + 1).map(sample).sum();
        output[x] = ((sum * u64::from(weight) + (sample(0) + sample(2 * R + 2)) * u64::from(far_weight) + super::BOX_WEIGHT_ROUND)
            >> super::BOX_WEIGHT_BITS) as u16;
    }
}

fn box_vertical(source: &[u16], output: &mut [u16], stride: usize, radius: f32) {
    let height = source.len() / stride;
    let integer = radius as usize;
    let weight = (super::BOX_WEIGHT_SCALE as f32 / (radius * 2.0 + 1.0)) as u32;
    let far_weight = super::BOX_WEIGHT_SCALE.wrapping_sub((2 * integer as u32 + 1).wrapping_mul(weight)) / 2;
    let scale = 1.0 / f64::from(super::BOX_WEIGHT_SCALE);
    if integer <= 2 {
        blanket_core::parallel::chunks_mut_above(output, stride * 16, 256 * 1024, |band, rows| {
            for (i, row) in rows.chunks_exact_mut(stride).enumerate() {
                let y = band * 16 + i;
                match integer {
                    0 => box_sample_column::<0>(source, row, y, weight, far_weight),
                    1 => box_sample_column::<1>(source, row, y, weight, far_weight),
                    _ => box_sample_column::<2>(source, row, y, weight, far_weight),
                }
            }
        });
        return;
    }
    let band_rows = 64.max(integer.min(height));
    blanket_core::parallel::chunks_mut_above(output, stride * band_rows, 256 * 1024, |band, rows| {
        let first = band * band_rows;
        let top = first.saturating_sub(integer);
        let bottom = (first + integer).min(height - 1);
        let mut sums = vec![0.0; stride];
        for row in source[top * stride..(bottom + 1) * stride].chunks_exact(stride) {
            for (sum, &value) in sums.iter_mut().zip(row) {
                *sum += f64::from(value);
            }
        }
        for ((sum, &upper), &lower) in sums.iter_mut().zip(&source[..stride]).zip(&source[(height - 1) * stride..]) {
            *sum += f64::from(upper) * integer.saturating_sub(first) as f64 + f64::from(lower) * (first + integer).saturating_sub(height - 1) as f64;
        }
        // Integer sums and weighted products fit exactly in f64, including large radii.
        for (i, row) in rows.chunks_exact_mut(stride).enumerate() {
            let y = first + i;
            let upper = &source[y.saturating_sub(integer + 1) * stride..][..stride];
            let lower = &source[(y + integer + 1).min(height - 1) * stride..][..stride];
            let leaving = &source[y.saturating_sub(integer) * stride..][..stride];
            for ((((dst, sum), &upper), &lower), &leaving) in row.iter_mut().zip(&mut sums).zip(upper).zip(lower).zip(leaving) {
                let ends = f64::from(upper) + f64::from(lower);
                *dst = ((*sum * f64::from(weight) + ends * f64::from(far_weight) + super::BOX_WEIGHT_ROUND as f64) * scale) as u64 as u16;
                *sum = *sum - f64::from(leaving) + f64::from(lower);
            }
        }
    });
}

fn box_sample_column<const R: usize>(source: &[u16], output: &mut [u16], y: usize, weight: u32, far_weight: u32) {
    let stride = output.len();
    let height = source.len() / stride;
    let row = |tap: usize| {
        let start = (y + tap).saturating_sub(R + 1).min(height - 1) * stride;
        &source[start..start + stride]
    };
    let upper = row(0);
    let lower = row(2 * R + 2);
    let scale = 1.0 / f64::from(super::BOX_WEIGHT_SCALE);
    let end = stride / 16 * 16;
    for x in (0..end).step_by(16) {
        let mut sum = Simd::<u32, 16>::splat(0);
        for tap in 1..=2 * R + 1 {
            sum += Samples::from_slice(&row(tap)[x..]).cast::<u32>();
        }
        let ends = Samples::from_slice(&upper[x..]).cast::<u32>() + Samples::from_slice(&lower[x..]).cast::<u32>();
        // These integer products fit exactly in f64.
        let value = sum.cast::<f64>() * Simd::splat(f64::from(weight))
            + ends.cast::<f64>() * Simd::splat(f64::from(far_weight))
            + Simd::splat(super::BOX_WEIGHT_ROUND as f64);
        (value * Simd::splat(scale)).cast::<u64>().cast::<u16>().copy_to_slice(&mut output[x..x + 16]);
    }
    for x in end..stride {
        let sum: u64 = (1..=2 * R + 1).map(|tap| u64::from(row(tap)[x])).sum();
        let ends = u64::from(upper[x]) + u64::from(lower[x]);
        output[x] = ((sum * u64::from(weight) + ends * u64::from(far_weight) + super::BOX_WEIGHT_ROUND) >> super::BOX_WEIGHT_BITS) as u16;
    }
}

fn blur_into(image: &Image, source: &[u16], values: &mut [u16], radii: (f32, f32), kind: BlurKind) {
    if source.is_empty() || radii == (0.0, 0.0) {
        values.copy_from_slice(source);
        return;
    }
    let passes = if kind == BlurKind::Gaussian { 3 } else { 1 };
    let horizontal = if radii.0 > 0.0 { passes } else { 0 };
    let total = horizontal + if radii.1 > 0.0 { passes } else { 0 };
    let mut scratch = if total > 1 { vec![0; source.len()] } else { Vec::new() };
    let pass = |src: &[u16], dst: &mut [u16], index| {
        let vertical = index >= horizontal;
        let radius = if vertical { radii.1 } else { radii.0 };
        box_pass(src, dst, image.width() as usize, image.height() as usize, image.pixel_mode().channels(), radius, vertical);
    };
    for index in 0..total {
        // Choose the first destination so the final pass lands in the output.
        if index == 0 {
            if total % 2 == 1 { pass(source, values, index) } else { pass(source, &mut scratch, index) }
        } else if (total - index) % 2 == 1 {
            pass(&scratch, values, index);
        } else {
            pass(values, &mut scratch, index);
        }
    }
}

pub(crate) fn blur(image: &Image, radii: (f32, f32), kind: BlurKind) -> PyResult<Image> {
    let bytes = image.raw_data()?;
    let radii = super::blur_radii(radii, kind)?;
    if radii == (0.0, 0.0) {
        return samples::output_bytes(image, blanket_core::buffers::copy(bytes)?);
    }
    let source = samples::view(image)?;
    samples::with_output(image, None, |values| blur_into(image, &source, values, radii, kind))
}

pub(crate) fn unsharp(image: &Image, radius: f32, percent: i32, threshold: i32) -> PyResult<Image> {
    let source = samples::view(image)?;
    let radii = super::blur_radii((radius, radius), BlurKind::Gaussian)?;
    samples::with_output(image, None, |values| {
        blur_into(image, &source, values, radii, BlurKind::Gaussian);
        sharpen(&source, values, percent, threshold, samples::maximum(image));
    })
}

fn sharpen(source: &[u16], values: &mut [u16], percent: i32, threshold: i32, maximum: u16) {
    let end = values.len() / 16 * 16;
    for i in (0..end).step_by(16) {
        let original = Samples::from_slice(&source[i..]).cast::<i64>();
        let difference = original - Samples::from_slice(&values[i..]).cast::<i64>();
        let value =
            (original + difference * Simd::splat(i64::from(percent)) / Simd::splat(100)).simd_clamp(Simd::splat(0), Simd::splat(i64::from(maximum)));
        difference.abs().simd_gt(Simd::splat(i64::from(threshold))).select(value, original).cast::<u16>().copy_to_slice(&mut values[i..i + 16]);
    }
    for (&original, value) in source[end..].iter().zip(&mut values[end..]) {
        let difference = i64::from(original) - i64::from(*value);
        *value = if difference.abs() > i64::from(threshold) {
            (i64::from(original) + difference * i64::from(percent) / 100).clamp(0, i64::from(maximum)) as u16
        } else {
            original
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn box_blur_matches_scalar_for_unaligned_slices_radii_and_guards() {
        for (width, height) in [(1, 1), (7, 3), (15, 17), (16, 16), (17, 19), (33, 5), (19, 131)] {
            for channels in [1, 3, 4] {
                let count = width * height * channels;
                let backing: Vec<_> = (0..count + 1).map(|i| (i * 719 + 257) as u16).collect();
                let source = &backing[1..];
                for radius in [0.0_f32, 0.25, 1.0, 1.5, 2.75, 3.0, 17.25, 100.5, 8_500_000.0, 2_000_000_000.0] {
                    for vertical in [false, true] {
                        let integer = radius as usize;
                        let weight = (crate::BOX_WEIGHT_SCALE as f32 / (radius * 2.0 + 1.0)) as u64;
                        let far = u64::from(crate::BOX_WEIGHT_SCALE.wrapping_sub((integer as u32 * 2 + 1).wrapping_mul(weight as u32)) / 2);
                        let mut output = vec![173; count + 2];
                        box_pass(source, &mut output[1..count + 1], width, height, channels, radius, vertical);
                        for y in 0..height {
                            for x in 0..width {
                                for c in 0..channels {
                                    let at = |delta: isize| {
                                        let xx = if vertical { x } else { x.saturating_add_signed(delta).min(width - 1) };
                                        let yy = if vertical { y.saturating_add_signed(delta).min(height - 1) } else { y };
                                        u64::from(source[(yy * width + xx) * channels + c])
                                    };
                                    let (position, length) = if vertical { (y, height) } else { (x, width) };
                                    let low = integer.min(position);
                                    let high = integer.min(length - 1 - position);
                                    let sum = (-(low as isize)..=high as isize).map(at).sum::<u64>()
                                        + at(-(position as isize)) * integer.saturating_sub(position) as u64
                                        + at((length - 1 - position) as isize) * integer.saturating_sub(length - 1 - position) as u64;
                                    let ends = at(-(integer as isize) - 1) + at(integer as isize + 1);
                                    let expected = ((sum * weight + ends * far + (1 << 23)) >> 24) as u16;
                                    assert_eq!(output[1 + (y * width + x) * channels + c], expected);
                                }
                            }
                        }
                        assert_eq!((output[0], output[count + 1]), (173, 173));
                    }
                }
            }
        }
    }

    #[test]
    fn box_blur_parallel_bands_preserve_samples_and_guards() {
        let (width, height, channels) = (257, 263, 4);
        let count = width * height * channels;
        let backing: Vec<_> = (0..count + 1).map(|i| (i * 719 + 257) as u16).collect();
        let source = &backing[1..];
        for radius in [0.25, 1.5, 2.75] {
            let integer = radius as usize;
            let weight = (crate::BOX_WEIGHT_SCALE as f32 / (radius * 2.0 + 1.0)) as u64;
            let far = u64::from(crate::BOX_WEIGHT_SCALE.wrapping_sub((2 * integer as u32 + 1) * weight as u32) / 2);
            for vertical in [false, true] {
                let mut output = vec![173; count + 2];
                box_pass(source, &mut output[1..count + 1], width, height, channels, radius, vertical);
                for y in 0..height {
                    for x in 0..width {
                        for c in 0..channels {
                            let sample = |delta: isize| {
                                let xx = if vertical { x } else { x.saturating_add_signed(delta).min(width - 1) };
                                let yy = if vertical { y.saturating_add_signed(delta).min(height - 1) } else { y };
                                u64::from(source[(yy * width + xx) * channels + c])
                            };
                            let sum: u64 = (-(integer as isize)..=integer as isize).map(sample).sum();
                            let ends = sample(-(integer as isize) - 1) + sample(integer as isize + 1);
                            let expected = ((sum * weight + ends * far + (1 << 23)) >> 24) as u16;
                            assert_eq!(output[1 + (y * width + x) * channels + c], expected);
                        }
                    }
                }
                assert_eq!((output[0], output[count + 1]), (173, 173));
            }
        }
    }

    #[test]
    fn unsharp_preserves_integer_overflow_limits_thresholds_tails_and_guards() {
        for count in [0, 1, 15, 16, 17, 31, 32, 33, 65] {
            for maximum in [1023, 4095, 65535] {
                let source: Vec<_> = (0..count + 1).map(|i| ((i * 719 + 257) % (usize::from(maximum) + 1)) as u16).collect();
                for percent in [i32::MIN, -150, 0, 150, i32::MAX] {
                    for threshold in [-1, 0, 17, i32::MAX] {
                        let mut output = vec![173; count + 2];
                        for (i, v) in output[1..count + 1].iter_mut().enumerate() {
                            *v = source[count - i];
                        }
                        let original = output.clone();
                        sharpen(&source[1..], &mut output[1..count + 1], percent, threshold, maximum);
                        for i in 0..count {
                            let a = i64::from(source[1 + i]);
                            let difference = a - i64::from(original[1 + i]);
                            let expected = if difference.abs() > i64::from(threshold) {
                                (a + difference * i64::from(percent) / 100).clamp(0, i64::from(maximum)) as u16
                            } else {
                                a as u16
                            };
                            assert_eq!(output[1 + i], expected);
                        }
                        assert_eq!((output[0], output[count + 1]), (173, 173));
                    }
                }
            }
        }
    }
}
