use std::collections::BTreeMap;
use std::simd::cmp::{SimdOrd, SimdPartialOrd};
use std::simd::num::{SimdFloat, SimdInt, SimdUint};
use std::simd::{Select, Simd};

use blanket_core::pixels::Samples;
use blanket_core::{Image, PixelMode, samples};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub(crate) fn kernel(image: &Image, size: usize, scale: f32, offset: f32, weights: &[f32]) -> PyResult<Image> {
    let source = samples::view(image)?;
    let mut values = source.to_vec();
    let (width, height, channels) = (image.width as usize, image.height as usize, image.mode.channels());
    let radius = size / 2;
    if width >= size && height >= size {
        let weights: Vec<f32> = weights.iter().map(|weight| weight / scale).collect();
        for y in radius..height - radius {
            let first = (y * width + radius) * channels;
            let last = (y * width + width - radius) * channels;
            let end = first + (last - first) / 16 * 16;
            for i in (first..end).step_by(16) {
                let mut sum = Simd::splat(offset + 0.5);
                for ky in 0..size {
                    let start = i + radius * width * channels - ky * width * channels - radius * channels;
                    let at = |kx| Samples::from_slice(&source[start + kx * channels..]).cast::<f32>();
                    let mut row = super::filter::kernel_add_simd(at(0), Simd::splat(weights[ky * size]), at(1) * Simd::splat(weights[ky * size + 1]));
                    for kx in 2..size {
                        row = super::filter::kernel_add_simd(at(kx), Simd::splat(weights[ky * size + kx]), row);
                    }
                    sum += row;
                }
                sum.cast::<u16>().simd_min(Samples::splat(samples::maximum(image))).copy_to_slice(&mut values[i..i + 16]);
            }
            for i in end..last {
                let (x, c) = ((i / channels) % width, i % channels);
                let mut sum = offset + 0.5;
                for ky in 0..size {
                    let start = ((y + radius - ky) * width + x - radius) * channels + c;
                    let mut row = f32::from(source[start + channels]) * weights[ky * size + 1];
                    row = super::filter::kernel_add(f32::from(source[start]), weights[ky * size], row);
                    for kx in 2..size {
                        row = super::filter::kernel_add(f32::from(source[start + kx * channels]), weights[ky * size + kx], row);
                    }
                    sum += row;
                }
                values[(y * width + x) * channels + c] = sum.clamp(0.0, f32::from(samples::maximum(image))) as u16;
            }
        }
    }
    samples::output(image, &values)
}

pub(crate) fn neighborhood(image: &Image, radius: usize, rank: Option<u64>) -> PyResult<Image> {
    let source = samples::view(image)?;
    let mut values = source.to_vec();
    let (width, height, channels) = (image.width as usize, image.height as usize, image.mode.channels());
    if let Some(rank) = rank
        && (radius == 1 || radius == 2 && (rank == 0 || rank == 24))
    {
        small_rank(&source, &mut values, width, height, channels, radius, rank as usize);
        return samples::output(image, &values);
    }
    sliding_neighborhood(&source, &mut values, width, height, channels, radius, rank);
    samples::output(image, &values)
}

fn sliding_neighborhood(source: &[u16], values: &mut [u16], width: usize, height: usize, channels: usize, radius: usize, rank: Option<u64>) {
    if width == 0 || height == 0 {
        return;
    }
    let mut histogram = BTreeMap::<u16, u64>::new();
    for y in 0..height {
        for c in 0..channels {
            histogram.clear();
            let column = |histogram: &mut BTreeMap<u16, u64>, xx: usize, wx: u64, add: bool| {
                for yy in y.saturating_sub(radius)..=(y + radius).min(height - 1) {
                    let wy = if rank.is_none() {
                        1
                    } else {
                        1 + if yy == 0 { radius.saturating_sub(y) } else { 0 }
                            + if yy + 1 == height { (y + radius).saturating_sub(height - 1) } else { 0 }
                    };
                    let value = source[(yy * width + xx) * channels + c];
                    let weight = wx * wy as u64;
                    if add {
                        *histogram.entry(value).or_default() += weight;
                    } else {
                        let count = histogram.get_mut(&value).unwrap();
                        *count -= weight;
                        if *count == 0 {
                            histogram.remove(&value);
                        }
                    }
                }
            };
            for xx in 0..=radius.min(width - 1) {
                let weight = if rank.is_some() {
                    1 + if xx == 0 { radius } else { 0 } + if xx + 1 == width { radius.saturating_sub(width - 1) } else { 0 }
                } else {
                    1
                };
                column(&mut histogram, xx, weight as u64, true);
            }
            for x in 0..width {
                let dst = &mut values[(y * width + x) * channels + c];
                if let Some(mut rank) = rank {
                    for (&value, &count) in &histogram {
                        if rank < count {
                            *dst = value;
                            break;
                        }
                        rank -= count;
                    }
                } else {
                    let mut best = 2;
                    for (&value, &count) in &histogram {
                        if count > best {
                            best = count;
                            *dst = value;
                        }
                    }
                }
                if x + 1 < width {
                    if rank.is_some() || x >= radius {
                        column(&mut histogram, x.saturating_sub(radius), 1, false);
                    }
                    if rank.is_some() || x + radius + 1 < width {
                        column(&mut histogram, (x + radius + 1).min(width - 1), 1, true);
                    }
                }
            }
        }
    }
}

fn small_rank(source: &[u16], output: &mut [u16], width: usize, height: usize, channels: usize, radius: usize, rank: usize) {
    let stride = width * channels;
    let taps = (radius * 2 + 1).pow(2);
    for y in 0..height {
        let end = stride / 16 * 16;
        for x in (0..end).step_by(16) {
            let mut window = [Samples::splat(0); 25];
            let mut n = 0;
            for ky in 0..=radius * 2 {
                let yy = (y + ky).saturating_sub(radius).min(height - 1);
                for kx in 0..=radius * 2 {
                    let indices = Simd::from_array(std::array::from_fn(|lane| {
                        let position = x + lane;
                        let xx = (position / channels + kx).saturating_sub(radius).min(width - 1);
                        (yy * width + xx) * channels + position % channels
                    }));
                    window[n] = if x >= radius * channels && x + 16 <= width.saturating_sub(radius) * channels {
                        Samples::from_slice(&source[yy * stride + x + kx * channels - radius * channels..])
                    } else {
                        Samples::gather_or_default(source, indices)
                    };
                    n += 1;
                }
            }
            let value = if rank == 0 {
                window[..taps].iter().copied().reduce(Samples::simd_min).unwrap()
            } else if rank + 1 == taps {
                window[..taps].iter().copied().reduce(Samples::simd_max).unwrap()
            } else {
                for i in 1..taps {
                    for j in (1..=i).rev() {
                        let a = window[j - 1];
                        let b = window[j];
                        window[j - 1] = a.simd_min(b);
                        window[j] = a.simd_max(b);
                    }
                }
                window[rank]
            };
            value.copy_to_slice(&mut output[y * stride + x..y * stride + x + 16]);
        }
        for x in end..stride {
            let mut window = [0; 25];
            let mut n = 0;
            for ky in 0..=radius * 2 {
                for kx in 0..=radius * 2 {
                    let yy = (y + ky).saturating_sub(radius).min(height - 1);
                    let xx = (x / channels + kx).saturating_sub(radius).min(width - 1);
                    window[n] = source[(yy * width + xx) * channels + x % channels];
                    n += 1;
                }
            }
            window[..taps].select_nth_unstable(rank);
            output[y * stride + x] = window[rank];
        }
    }
}

fn box_pass(source: &[u16], output: &mut [u16], width: usize, height: usize, channels: usize, radius: f32, vertical: bool) {
    let integer = radius as usize;
    let weight = (super::filter::BOX_WEIGHT_SCALE as f32 / (radius * 2.0 + 1.0)) as u64;
    let far_weight = u64::from(super::filter::BOX_WEIGHT_SCALE.wrapping_sub((2 * integer as u32 + 1).wrapping_mul(weight as u32)) / 2);
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
            let value = (sum * Simd::splat(weight) + (at(left) + at(right)) * Simd::splat(far_weight) + Simd::splat(super::filter::BOX_WEIGHT_ROUND))
                >> u64::from(super::filter::BOX_WEIGHT_BITS);
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
                ((sum * weight + (at(left) + at(right)) * far_weight + super::filter::BOX_WEIGHT_ROUND) >> super::filter::BOX_WEIGHT_BITS) as u16;
            sum = sum - at(i.saturating_sub(integer)) + at(right);
        }
    }
}

fn blurred(image: &Image, mut values: Vec<u16>, radii: (f32, f32), gaussian: bool) -> PyResult<Vec<u16>> {
    let radii = super::filter::blur_radii(radii, gaussian)?;
    if values.is_empty() || radii == (0.0, 0.0) {
        return Ok(values);
    }
    let mut scratch = vec![0; values.len()];
    for (radius, vertical) in [(radii.0, false), (radii.1, true)] {
        if radius == 0.0 {
            continue;
        }
        for _ in 0..if gaussian { 3 } else { 1 } {
            box_pass(&values, &mut scratch, image.width as usize, image.height as usize, image.mode.channels(), radius, vertical);
            std::mem::swap(&mut values, &mut scratch);
        }
    }
    Ok(values)
}

pub(crate) fn blur(image: &Image, radii: (f32, f32), gaussian: bool) -> PyResult<Image> {
    let source = samples::read(image)?;
    samples::output(image, &blurred(image, source, radii, gaussian)?)
}

pub(crate) fn unsharp(image: &Image, radius: f32, percent: i32, threshold: i32) -> PyResult<Image> {
    let source = samples::view(image)?;
    let mut values = blurred(image, source.to_vec(), (radius, radius), true)?;
    sharpen(&source, &mut values, percent, threshold, samples::maximum(image));
    samples::output(image, &values)
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

pub(crate) fn lut(image: &Image, mode: &str, channels: usize, size: (usize, usize, usize), table: &[f32]) -> PyResult<Image> {
    let target = PixelMode::parse(mode)?;
    let dimensions = <[usize; 3]>::from(size);
    let input_channels = image.mode.channels();
    let output_channels = target.channels();
    if !matches!(image.mode, PixelMode::Rgb | PixelMode::Rgba)
        || !matches!(target, PixelMode::Rgb | PixelMode::Rgba)
        || !(3..=4).contains(&channels)
        || output_channels < channels
        || output_channels > channels && output_channels > input_channels
    {
        return Err(PyValueError::new_err("image has wrong mode"));
    }
    if dimensions.iter().any(|s| !(2..=65).contains(s)) {
        return Err(PyValueError::new_err("Table size in any dimension should be from 2 to 65"));
    }
    if table.len() != channels * size.0 * size.1 * size.2 {
        return Err(PyValueError::new_err("table has wrong number of elements"));
    }
    let source = samples::view(image)?;
    let maximum = f64::from(samples::maximum(image));
    let strides = [channels, channels * size.0, channels * size.0 * size.1];
    let mut values = vec![0; source.len() / input_channels * output_channels];
    let end = source.len() / input_channels / 16 * 16;
    for i in (0..end).step_by(16) {
        use std::simd::StdFloat;
        let input = if input_channels == 4 {
            blanket_core::pixels::load_samples::<4>(&source[i * 4..])
        } else {
            let rgb = blanket_core::pixels::load_samples::<3>(&source[i * 3..]);
            [rgb[0], rgb[1], rgb[2], Samples::splat(0)]
        };
        let positions: [Simd<f64, 16>; 3] =
            std::array::from_fn(|c| input[c].cast::<f64>() / Simd::splat(maximum) * Simd::splat((dimensions[c] - 1) as f64));
        let lower: [Simd<usize, 16>; 3] = std::array::from_fn(|c| positions[c].cast::<usize>().simd_min(Simd::splat(dimensions[c] - 2)));
        let fractions: [Simd<f64, 16>; 3] = std::array::from_fn(|c| positions[c] - lower[c].cast::<f64>());
        let mut result = [Samples::splat(0); 4];
        for c in 0..output_channels {
            if c >= channels {
                result[c] = input[c];
                continue;
            }
            let mut value = Simd::splat(0.0);
            for corner in 0..8 {
                let mut index = Simd::splat(c);
                let mut weight = Simd::splat(1.0);
                for axis in 0..3 {
                    let upper = (corner >> axis) & 1;
                    index += (lower[axis] + Simd::splat(upper)) * Simd::splat(strides[axis]);
                    weight *= if upper == 1 { fractions[axis] } else { Simd::splat(1.0) - fractions[axis] };
                }
                value += Simd::<f32, 16>::gather_or_default(table, index).cast::<f64>() * weight;
            }
            result[c] = (value * Simd::splat(maximum)).round().cast::<u16>().simd_min(Samples::splat(maximum as u16));
        }
        if output_channels == 4 {
            blanket_core::pixels::store_samples(&mut values[i * 4..], result);
        } else {
            blanket_core::pixels::store_samples(&mut values[i * 3..], [result[0], result[1], result[2]]);
        }
    }
    for (pixel, dst) in
        source[end * input_channels..].chunks_exact(input_channels).zip(values[end * output_channels..].chunks_exact_mut(output_channels))
    {
        let positions: [f64; 3] = std::array::from_fn(|c| f64::from(pixel[c]) / maximum * (dimensions[c] - 1) as f64);
        let lower: [usize; 3] = std::array::from_fn(|c| (positions[c] as usize).min(dimensions[c] - 2));
        let fractions: [f64; 3] = std::array::from_fn(|c| positions[c] - lower[c] as f64);
        for (c, dst) in dst.iter_mut().enumerate() {
            if c >= channels {
                *dst = pixel[c];
                continue;
            }
            let mut value = 0.0;
            for corner in 0..8 {
                let mut index = c;
                let mut weight = 1.0;
                for axis in 0..3 {
                    let upper = (corner >> axis) & 1;
                    index += (lower[axis] + upper) * strides[axis];
                    weight *= if upper == 1 { fractions[axis] } else { 1.0 - fractions[axis] };
                }
                value += f64::from(table[index]) * weight;
            }
            *dst = (value * maximum).round().clamp(0.0, maximum) as u16;
        }
    }
    Image::from_samples(image.width, image.height, target, values, image.bit_depth, None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn box_blur_matches_scalar_for_unaligned_slices_radii_and_guards() {
        for (width, height) in [(1, 1), (7, 3), (15, 17), (16, 16), (17, 19), (33, 5)] {
            for channels in [1, 3, 4] {
                let count = width * height * channels;
                let backing: Vec<_> = (0..count + 1).map(|i| (i * 719 + 257) as u16).collect();
                let source = &backing[1..];
                for radius in [0.0_f32, 0.25, 1.0, 1.5, 17.25] {
                    for vertical in [false, true] {
                        let integer = radius as usize;
                        let weight = (super::super::filter::BOX_WEIGHT_SCALE as f32 / (radius * 2.0 + 1.0)) as u64;
                        let far =
                            u64::from(super::super::filter::BOX_WEIGHT_SCALE.wrapping_sub((integer as u32 * 2 + 1).wrapping_mul(weight as u32)) / 2);
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
                                    let sum: u64 = (-(integer as isize)..=integer as isize).map(at).sum();
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
    fn sliding_neighborhood_matches_rebuilt_windows() {
        for (width, height) in [(0, 0), (1, 1), (1, 7), (7, 1), (9, 6)] {
            for channels in [1, 3, 4] {
                for repeated in [false, true] {
                    let source: Vec<u16> = (0..width * height * channels)
                        .map(|i| if repeated { ((i * 7 + i / 5) % 4) as u16 } else { (i * 719 + 257) as u16 })
                        .collect();
                    for radius in [0, 1, 2, 4, 10] {
                        let taps = (radius * 2 + 1_usize).pow(2);
                        for rank in [None, Some(0), Some((taps / 2) as u64), Some((taps - 1) as u64)] {
                            let mut output = vec![173; source.len() + 2];
                            output[1..source.len() + 1].copy_from_slice(&source);
                            sliding_neighborhood(&source, &mut output[1..source.len() + 1], width, height, channels, radius, rank);
                            for y in 0..height {
                                for x in 0..width {
                                    for c in 0..channels {
                                        let mut window = Vec::new();
                                        if rank.is_some() {
                                            for ky in 0..=radius * 2 {
                                                for kx in 0..=radius * 2 {
                                                    let yy = (y + ky).saturating_sub(radius).min(height - 1);
                                                    let xx = (x + kx).saturating_sub(radius).min(width - 1);
                                                    window.push(source[(yy * width + xx) * channels + c]);
                                                }
                                            }
                                        } else {
                                            for yy in y.saturating_sub(radius)..=(y + radius).min(height - 1) {
                                                for xx in x.saturating_sub(radius)..=(x + radius).min(width - 1) {
                                                    window.push(source[(yy * width + xx) * channels + c]);
                                                }
                                            }
                                        }
                                        window.sort_unstable();
                                        let index = (y * width + x) * channels + c;
                                        let expected = if let Some(rank) = rank {
                                            window[rank as usize]
                                        } else {
                                            let mut best = 2;
                                            let mut value = source[index];
                                            for run in window.chunk_by(|a, b| a == b) {
                                                if run.len() > best {
                                                    best = run.len();
                                                    value = run[0];
                                                }
                                            }
                                            value
                                        };
                                        assert_eq!(output[index + 1], expected, "{width}x{height}, {channels}, {radius}, {rank:?}");
                                    }
                                }
                            }
                            assert_eq!((output[0], output[source.len() + 1]), (173, 173));
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn small_ranks_match_scalar_at_edges_vector_boundaries_and_tails() {
        for width in [1, 7, 15, 16, 17, 31, 33] {
            for channels in [1, 3, 4] {
                let height = 5;
                let count = width * height * channels;
                let backing: Vec<_> = (0..count + 1).map(|i| (i * 719 + 257) as u16).collect();
                let source = &backing[1..];
                for (radius, rank) in (0..9).map(|rank| (1, rank)).chain([(2, 0), (2, 24)]) {
                    let mut output = vec![173; count + 2];
                    small_rank(source, &mut output[1..count + 1], width, height, channels, radius, rank);
                    for y in 0..height {
                        for x in 0..width {
                            for c in 0..channels {
                                let mut window = Vec::new();
                                for ky in 0..=radius * 2 {
                                    for kx in 0..=radius * 2 {
                                        let yy = (y + ky).saturating_sub(radius).min(height - 1);
                                        let xx = (x + kx).saturating_sub(radius).min(width - 1);
                                        window.push(source[(yy * width + xx) * channels + c]);
                                    }
                                }
                                window.sort_unstable();
                                assert_eq!(output[1 + (y * width + x) * channels + c], window[rank]);
                            }
                        }
                    }
                    assert_eq!((output[0], output[count + 1]), (173, 173));
                }
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
