#![feature(portable_simd)]

use std::collections::VecDeque;
use std::simd::Simd;
#[cfg(target_arch = "aarch64")]
use std::simd::StdFloat;
use std::simd::num::{SimdFloat, SimdUint};

use blanket_core::parallel::chunks_mut;
use blanket_core::raster::{Image, PixelMode, alloc_pixels};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

mod high_depth;
mod simd;
mod wide;

fn output(image: &Image, pixels: Vec<u8>) -> PyResult<Image> {
    let mut result = Image::from_pixels(image.width(), image.height(), image.pixel_mode(), pixels, None)?;
    result.copy_palette_from(image);
    Ok(result)
}

#[pyfunction]
pub fn filter_kernel(py: Python<'_>, image: &Image, size: (i32, i32), scale: f32, offset: f32, mut kernel: Vec<f32>) -> PyResult<Image> {
    if (image.palette().is_some() && image.pixel_mode() != PixelMode::Pa) || image.pixel_mode() == PixelMode::F {
        return Err(PyValueError::new_err("image has wrong mode"));
    }
    if image.pixel_mode() == PixelMode::I {
        if size.0 != size.1 || !matches!(size.0, 3 | 5) || kernel.len() != (size.0 * size.1) as usize {
            return Err(PyValueError::new_err("bad kernel size"));
        }
        return py.detach(|| crate::wide::kernel(image, size.0 as usize, scale, offset, &kernel));
    }
    if blanket_core::samples::supported(image) {
        if size.0 != size.1 || !matches!(size.0, 3 | 5) || kernel.len() != (size.0 * size.1) as usize {
            return Err(PyValueError::new_err("bad kernel size"));
        }
        return py.detach(|| crate::high_depth::kernel(image, size.0 as usize, scale, offset, &kernel));
    }
    let source = image.pixel_data()?;
    if i64::from(size.0) * i64::from(size.1) != kernel.len() as i64 {
        return Err(PyValueError::new_err("bad kernel size"));
    }
    let mut pixels = py.detach(|| source.to_vec());
    if i64::from(image.width()) < i64::from(size.0) || i64::from(image.height()) < i64::from(size.1) {
        return output(image, pixels);
    }
    if size.0 != size.1 || !matches!(size.0, 3 | 5) {
        return Err(PyValueError::new_err("bad kernel size"));
    }
    for value in &mut kernel {
        *value /= scale;
    }
    let weights = kernel;
    let width = image.width() as usize;
    py.detach(|| match (image.pixel_mode(), size.0) {
        (PixelMode::One | PixelMode::L, 3) => convolve::<1, 3>(source, &mut pixels, width, &weights, offset),
        (PixelMode::One | PixelMode::L, _) => convolve::<1, 5>(source, &mut pixels, width, &weights, offset),
        (PixelMode::La | PixelMode::PremulLa | PixelMode::Pa, 3) => convolve::<2, 3>(source, &mut pixels, width, &weights, offset),
        (PixelMode::La | PixelMode::PremulLa | PixelMode::Pa, _) => convolve::<2, 5>(source, &mut pixels, width, &weights, offset),
        (PixelMode::Rgb | PixelMode::Hsv | PixelMode::YCbCr | PixelMode::Lab, 3) => convolve::<3, 3>(source, &mut pixels, width, &weights, offset),
        (PixelMode::Rgb | PixelMode::Hsv | PixelMode::YCbCr | PixelMode::Lab, _) => convolve::<3, 5>(source, &mut pixels, width, &weights, offset),
        (PixelMode::Rgba | PixelMode::PremulRgba | PixelMode::Rgbx | PixelMode::Cmyk, 3) => {
            convolve::<4, 3>(source, &mut pixels, width, &weights, offset)
        }
        (PixelMode::Rgba | PixelMode::PremulRgba | PixelMode::Rgbx | PixelMode::Cmyk, _) => {
            convolve::<4, 5>(source, &mut pixels, width, &weights, offset)
        }
        _ => unreachable!(),
    });
    output(image, pixels)
}

// Fixed channel and kernel sizes let every tap and band unroll; the runtime
// version spent its time on strided index arithmetic and bounds checks.
fn convolve<const C: usize, const K: usize>(source: &[u8], pixels: &mut [u8], width: usize, weights: &[f32], offset: f32) {
    if pixels.is_empty() {
        return;
    }
    let taps: &[[f32; K]] = weights.as_chunks::<K>().0;
    let radius = K / 2;
    let stride = width * C;
    let height = source.len() / stride;
    chunks_mut(pixels, stride * 16, |band, rows| {
        for (i, row) in rows.chunks_exact_mut(stride).enumerate() {
            let y = band * 16 + i;
            if y < radius || y + radius >= height {
                continue;
            }
            let first = radius * C;
            let last = (width - radius) * C;
            let end = first + (last - first) / 16 * 16;
            for x in (first..end).step_by(16) {
                let mut sums = Simd::splat(offset + 0.5);
                for (ky, taps) in taps.iter().enumerate() {
                    let start = (y + radius - ky) * stride + x - first;
                    let value = |kx| Simd::<u8, 16>::from_slice(&source[start + kx * C..][..16]).cast::<f32>();
                    let mut subtotal = kernel_add_simd(value(0), Simd::splat(taps[0]), value(1) * Simd::splat(taps[1]));
                    for (kx, &weight) in taps.iter().enumerate().skip(2) {
                        subtotal = kernel_add_simd(value(kx), Simd::splat(weight), subtotal);
                    }
                    sums += subtotal;
                }
                sums.cast::<u8>().copy_to_slice(&mut row[x..x + 16]);
            }
            for (i, pixel) in row[end..last].iter_mut().enumerate() {
                let x = end + i;
                let mut sum = offset + 0.5;
                for (ky, taps) in taps.iter().enumerate() {
                    let start = (y + radius - ky) * stride + x - first;
                    // Sum each row before accumulating, matching Pillow's f32 order.
                    let mut subtotal = kernel_add(f32::from(source[start]), taps[0], f32::from(source[start + C]) * taps[1]);
                    for kx in 2..K {
                        subtotal = kernel_add(f32::from(source[start + kx * C]), taps[kx], subtotal);
                    }
                    sum += subtotal;
                }
                *pixel = sum as u8;
            }
        }
    });
}

pub(crate) fn kernel_add_simd(value: Simd<f32, 16>, weight: Simd<f32, 16>, sum: Simd<f32, 16>) -> Simd<f32, 16> {
    #[cfg(target_arch = "aarch64")]
    {
        value.mul_add(weight, sum)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        value * weight + sum
    }
}

// Pillow's AArch64 C builds contract each row's multiply/add expressions.
// Keep the same operand order; it matters at half-integer output boundaries.
pub(crate) fn kernel_add(value: f32, weight: f32, sum: f32) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        value.mul_add(weight, sum)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        value * weight + sum
    }
}

#[pyfunction]
pub fn filter_rank(py: Python<'_>, image: &Image, size: i32, rank: i32) -> PyResult<Image> {
    if image.palette().is_some() && image.pixel_mode() != PixelMode::Pa {
        return Err(PyValueError::new_err("image has wrong mode"));
    }
    if size < 1 || size % 2 == 0 {
        return Err(PyValueError::new_err("bad filter size"));
    }
    let count = i64::from(size) * i64::from(size);
    if count > i64::from(i32::MAX) / 4 {
        return Err(PyValueError::new_err("filter size too large"));
    }
    if rank < 0 || i64::from(rank) >= count {
        return Err(PyValueError::new_err("bad rank value"));
    }
    if size == 1 {
        if !blanket_core::samples::supported(image) && !matches!(image.pixel_mode(), PixelMode::I | PixelMode::F) {
            image.pixel_data()?;
        }
        return image.copy(py);
    }
    if blanket_core::samples::supported(image) {
        return py.detach(|| crate::high_depth::neighborhood(image, (size / 2) as usize, Some(rank as u64)));
    }
    if matches!(image.pixel_mode(), PixelMode::I | PixelMode::F) {
        return py.detach(|| crate::wide::rank(image, size as usize, rank as usize));
    }
    match (size, rank) {
        (3, 0) => return small_rank::<3, 0>(py, image),
        (3, 4) => return small_rank::<3, 4>(py, image),
        (3, 8) => return small_rank::<3, 8>(py, image),
        (5, 0) => return small_rank::<5, 0>(py, image),
        (5, 24) => return small_rank::<5, 24>(py, image),
        _ => (),
    }
    if size > 1 && (rank == 0 || i64::from(rank) == count - 1) {
        return extrema_filter(py, image, (size / 2) as usize, rank != 0);
    }
    neighborhood(py, image, (size / 2) as usize, Some(rank as u64))
}

#[pyfunction]
pub fn filter_mode(py: Python<'_>, image: &Image, size: i32) -> PyResult<Image> {
    if blanket_core::samples::supported(image) {
        return py.detach(|| crate::high_depth::neighborhood(image, (size / 2).max(0) as usize, None));
    }
    neighborhood(py, image, (size / 2).max(0) as usize, None)
}

fn small_rank<const K: usize, const R: usize>(py: Python<'_>, image: &Image) -> PyResult<Image> {
    let source = image.pixel_data()?;
    let mut pixels = alloc_pixels(source.len())?;
    if !pixels.is_empty() {
        let channels = image.pixel_mode().channels();
        let stride = image.width() as usize * channels;
        py.detach(|| {
            chunks_mut(&mut pixels, stride * 16, |band, rows| {
                for (i, row) in rows.chunks_exact_mut(stride).enumerate() {
                    crate::simd::rank_row::<K, R>(source, row, channels, band * 16 + i);
                }
            });
        });
    }
    output(image, pixels)
}

fn extrema_filter(py: Python<'_>, image: &Image, radius: usize, maximum: bool) -> PyResult<Image> {
    let source = image.pixel_data()?;
    let pixels = py.detach(|| -> PyResult<Vec<u8>> {
        let mut horizontal = alloc_pixels(source.len())?;
        let mut pixels = alloc_pixels(source.len())?;
        if pixels.is_empty() {
            return Ok(pixels);
        }
        let channels = image.pixel_mode().channels();
        let stride = image.width() as usize * channels;
        chunks_mut(&mut horizontal, stride * 16, |band, rows| {
            let mut queue = VecDeque::new();
            for (i, row) in rows.chunks_exact_mut(stride).enumerate() {
                let input = &source[(band * 16 + i) * stride..][..stride];
                for channel in 0..channels {
                    extrema_line(&input[channel..], &mut row[channel..], channels, 0, radius, maximum, &mut queue);
                }
            }
        });
        // Growing bands with the radius bounds repeated window initialization.
        let band_rows = 64.max(radius.min(image.height() as usize));
        chunks_mut(&mut pixels, stride * band_rows, |band, rows| {
            let mut queue = VecDeque::new();
            for column in 0..stride {
                extrema_line(&horizontal[column..], &mut rows[column..], stride, band * band_rows, radius, maximum, &mut queue);
            }
        });
        Ok(pixels)
    })?;
    output(image, pixels)
}

pub(crate) fn extrema_line<T: Copy + Ord>(
    source: &[T], output: &mut [T], step: usize, first: usize, radius: usize, maximum: bool, queue: &mut VecDeque<(usize, T)>,
) {
    queue.clear();
    let length = source.len().div_ceil(step);
    let mut next = first.saturating_sub(radius);
    for (i, dst) in output.iter_mut().step_by(step).enumerate() {
        let center = first + i;
        let left = center.saturating_sub(radius);
        while queue.front().is_some_and(|&(index, _)| index < left) {
            queue.pop_front();
        }
        let right = (center + radius).min(length - 1);
        while next <= right {
            let value = source[next * step];
            while queue.back().is_some_and(|&(_, previous)| if maximum { previous <= value } else { previous >= value }) {
                queue.pop_back();
            }
            queue.push_back((next, value));
            next += 1;
        }
        *dst = queue.front().unwrap().1;
    }
}

fn neighborhood(py: Python<'_>, image: &Image, radius: usize, rank: Option<u64>) -> PyResult<Image> {
    let source = image.pixel_data()?;
    if radius == 0 {
        return output(image, py.detach(|| blanket_core::buffers::copy(source))?);
    }
    let mut pixels = alloc_pixels(source.len())?;
    let width = image.width() as usize;
    let height = image.height() as usize;
    let channels = image.pixel_mode().channels();
    if pixels.is_empty() {
        return output(image, pixels);
    }
    py.detach(|| {
        // Column histograms remove the window-height factor for large ranks.
        // Bound all concurrently live column buffers to 8 MiB.
        let column_bytes = width * 272 * size_of::<u16>();
        let bands = (8 * 1024 * 1024 / column_bytes).min(height);
        if let Some(rank) = rank
            && radius >= 128
            && height >= 128
            && bands > 0
        {
            let band_rows = radius.max(height.div_ceil(bands));
            chunks_mut(&mut pixels, width * channels * band_rows, |band, rows| {
                rank_columns(source, rows, width, channels, band * band_rows, radius, rank);
            });
            return;
        }
        chunks_mut(&mut pixels, width * channels * 8, |band, rows| {
            for (i, row) in rows.chunks_exact_mut(width * channels).enumerate() {
                let y = band * 8 + i;
                let top = y.saturating_sub(radius);
                let bottom = (y + radius).min(height - 1);
                let vertical_weight = |yy: usize| -> u64 {
                    if rank.is_none() {
                        return 1;
                    }
                    1 + if yy == 0 { radius.saturating_sub(y) as u64 } else { 0 }
                        + if yy == height - 1 { (y + radius).saturating_sub(height - 1) as u64 } else { 0 }
                };
                for channel in 0..channels {
                    let mut histogram = [0_u64; 256];
                    let mut coarse = [0_u64; 16];
                    let mut candidates = [0_u64; 4];
                    for yy in top..=bottom {
                        let weight = vertical_weight(yy);
                        for xx in 0..=radius.min(width - 1) {
                            let horizontal = if rank.is_some() {
                                1 + if xx == 0 { radius as u64 } else { 0 }
                                    + if xx == width - 1 { radius.saturating_sub(width - 1) as u64 } else { 0 }
                            } else {
                                1
                            };
                            let value = source[(yy * width + xx) * channels + channel] as usize;
                            histogram[value] += weight * horizontal;
                            if rank.is_some() {
                                coarse[value >> 4] += weight * horizontal;
                            } else if histogram[value] == 3 {
                                candidates[value >> 6] |= 1 << (value & 63);
                            }
                        }
                    }
                    for x in 0..width {
                        let original = source[(y * width + x) * channels + channel];
                        row[x * channels + channel] = match rank {
                            Some(rank) => select_rank(&histogram, &coarse, rank),
                            None => select_mode(&histogram, &candidates, original),
                        };
                        if x + 1 == width {
                            break;
                        }
                        for yy in top..=bottom {
                            let weight = vertical_weight(yy);
                            if rank.is_some() || x >= radius {
                                let xx = x.saturating_sub(radius);
                                let value = source[(yy * width + xx) * channels + channel] as usize;
                                histogram[value] -= weight;
                                if rank.is_some() {
                                    coarse[value >> 4] -= weight;
                                } else if histogram[value] == 2 {
                                    candidates[value >> 6] &= !(1 << (value & 63));
                                }
                            }
                            if rank.is_some() || x + radius + 1 < width {
                                let xx = (x + radius + 1).min(width - 1);
                                let value = source[(yy * width + xx) * channels + channel] as usize;
                                histogram[value] += weight;
                                if rank.is_some() {
                                    coarse[value >> 4] += weight;
                                } else if histogram[value] == 3 {
                                    candidates[value >> 6] |= 1 << (value & 63);
                                }
                            }
                        }
                    }
                }
            }
        });
    });
    output(image, pixels)
}

// Locate the occupied group before scanning its sixteen values. Updating the
// group counts with the sliding window bounds rank selection to 32 bins.
fn select_rank<T: Copy + Into<u64>>(histogram: &[T], coarse: &[T], mut rank: u64) -> u8 {
    for (group, &frequency) in coarse.iter().enumerate() {
        let frequency = frequency.into();
        if rank >= frequency {
            rank -= frequency;
            continue;
        }
        let start = group << 4;
        for (offset, &frequency) in histogram[start..start + 16].iter().enumerate() {
            let frequency = frequency.into();
            if rank < frequency {
                return (start + offset) as u8;
            }
            rank -= frequency;
        }
    }
    unreachable!("rank is within the populated histogram")
}

fn rank_columns(source: &[u8], output: &mut [u8], width: usize, channels: usize, first: usize, radius: usize, rank: u64) {
    let stride = width * channels;
    let height = source.len() / stride;
    let rows = output.len() / stride;
    let mut columns = vec![[0_u16; 272]; width];
    for channel in 0..channels {
        if channel > 0 {
            columns.fill([0; 272]);
        }
        for y in first.saturating_sub(radius)..=(first + radius).min(height - 1) {
            let weight = 1
                + if y == 0 { radius.saturating_sub(first) as u16 } else { 0 }
                + if y == height - 1 { (first + radius).saturating_sub(height - 1) as u16 } else { 0 };
            for (x, column) in columns.iter_mut().enumerate() {
                let value = usize::from(source[y * stride + x * channels + channel]);
                column[value] += weight;
                column[256 + (value >> 4)] += weight;
            }
        }
        for (i, row) in output.chunks_exact_mut(stride).enumerate() {
            let y = first + i;
            let mut histogram = [0_u32; 272];
            for (x, column) in columns.iter().enumerate().take(radius.min(width - 1) + 1) {
                let weight = 1 + if x == 0 { radius as u32 } else { 0 } + if x == width - 1 { radius.saturating_sub(width - 1) as u32 } else { 0 };
                for (total, &count) in histogram.iter_mut().zip(column) {
                    *total += u32::from(count) * weight;
                }
            }
            for x in 0..width {
                row[x * channels + channel] = select_rank(&histogram[..256], &histogram[256..], rank);
                if x + 1 < width {
                    let leaving = &columns[x.saturating_sub(radius)];
                    let entering = &columns[(x + radius + 1).min(width - 1)];
                    for ((total, &leaving), &entering) in histogram.iter_mut().zip(leaving).zip(entering) {
                        *total = *total - u32::from(leaving) + u32::from(entering);
                    }
                }
            }
            if i + 1 < rows {
                let leaving = y.saturating_sub(radius) * stride;
                let entering = (y + radius + 1).min(height - 1) * stride;
                for (x, column) in columns.iter_mut().enumerate() {
                    let old = usize::from(source[leaving + x * channels + channel]);
                    let new = usize::from(source[entering + x * channels + channel]);
                    column[old] -= 1;
                    column[256 + (old >> 4)] -= 1;
                    column[new] += 1;
                    column[256 + (new >> 4)] += 1;
                }
            }
        }
    }
}

fn select_mode(histogram: &[u64; 256], candidates: &[u64; 4], original: u8) -> u8 {
    let mut count = 2;
    let mut result = original;
    // Only values occurring at least three times can replace the original.
    for (group, &mask) in candidates.iter().enumerate() {
        let mut mask = mask;
        while mask != 0 {
            let value = group * 64 + mask.trailing_zeros() as usize;
            let frequency = histogram[value];
            if frequency > count {
                count = frequency;
                result = value as u8;
            }
            mask &= mask - 1;
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn column_ranks_preserve_edges_channels_bands_and_guards() {
        for width in [1, 7, 33] {
            for channels in 1..=4 {
                let height = 137;
                let backing: Vec<_> = (0..width * height * channels + 1).map(|i| (i * 71 + i / 7) as u8).collect();
                let source = &backing[1..];
                for radius in [1_usize, 128, 1000] {
                    let count = (2 * radius + 1).pow(2) as u64;
                    for first in [0, 64, 128] {
                        let rows = 9.min(height - first);
                        for rank in [1, count / 2, count - 2] {
                            let mut output = vec![173; rows * width * channels + 2];
                            rank_columns(source, &mut output[1..1 + rows * width * channels], width, channels, first, radius, rank);
                            for y in first..first + rows {
                                for x in 0..width {
                                    for channel in 0..channels {
                                        let mut histogram = [0_u64; 256];
                                        for yy in y.saturating_sub(radius)..=(y + radius).min(height - 1) {
                                            let wy = 1
                                                + if yy == 0 { radius.saturating_sub(y) } else { 0 }
                                                + if yy == height - 1 { (y + radius).saturating_sub(height - 1) } else { 0 };
                                            for xx in x.saturating_sub(radius)..=(x + radius).min(width - 1) {
                                                let wx = 1
                                                    + if xx == 0 { radius.saturating_sub(x) } else { 0 }
                                                    + if xx == width - 1 { (x + radius).saturating_sub(width - 1) } else { 0 };
                                                histogram[usize::from(source[(yy * width + xx) * channels + channel])] += (wx * wy) as u64;
                                            }
                                        }
                                        let mut remaining = rank;
                                        let expected = histogram
                                            .iter()
                                            .position(|&frequency| {
                                                if remaining < frequency {
                                                    true
                                                } else {
                                                    remaining -= frequency;
                                                    false
                                                }
                                            })
                                            .unwrap() as u8;
                                        assert_eq!(output[1 + ((y - first) * width + x) * channels + channel], expected);
                                    }
                                }
                            }
                            assert_eq!(output[0], 173);
                            assert_eq!(output[output.len() - 1], 173);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn sliding_extrema_preserve_strides_offsets_and_guards() {
        let mut queue = VecDeque::new();
        for step in [1, 2, 3, 4] {
            let backing: Vec<u8> = (0..137 * step + 1).map(|i| (i * 71 + i / 7) as u8).collect();
            let source = &backing[1..];
            for first in [0, 1, 63, 64, 65, 136] {
                let count = 17.min(137 - first);
                for radius in [0, 1, 4, 70, 1000] {
                    for maximum in [false, true] {
                        let mut output = vec![173; count * step + 2];
                        extrema_line(source, &mut output[1..1 + count * step], step, first, radius, maximum, &mut queue);
                        for (i, &actual) in output.iter().enumerate() {
                            if i > 0 && i <= count * step && (i - 1) % step == 0 {
                                let center = first + (i - 1) / step;
                                let left = center.saturating_sub(radius);
                                let right = (center + radius).min(136);
                                let values = (left..=right).map(|index| source[index * step]);
                                let expected = if maximum { values.max() } else { values.min() }.unwrap();
                                assert_eq!(actual, expected);
                            } else {
                                assert_eq!(actual, 173);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn convolution_vectors_match_scalar_rounding_tails_and_guards() {
        fn check<const C: usize, const K: usize>() {
            for width in [K, K + 1, K + 15, K + 16, K + 17, 67] {
                let height = 35;
                for offset in [0, 1, 7, 15] {
                    let source: Vec<u8> = (0..offset + width * height * C).map(|i| (i * 37 + 113) as u8).collect();
                    let source = &source[offset..];
                    for weights in [(0..K * K).map(|i| (i as f32 - 4.5) / 7.0).collect::<Vec<_>>(), vec![f32::INFINITY; K * K], vec![f32::NAN; K * K]]
                    {
                        let mut expected = source.to_vec();
                        let radius = K / 2;
                        for y in radius..height - radius {
                            for x in radius * C..(width - radius) * C {
                                let mut sum = -13.25 + 0.5;
                                for ky in 0..K {
                                    let start = (y + radius - ky) * width * C + x - radius * C;
                                    let taps = &weights[ky * K..][..K];
                                    let mut subtotal = kernel_add(f32::from(source[start]), taps[0], f32::from(source[start + C]) * taps[1]);
                                    for kx in 2..K {
                                        subtotal = kernel_add(f32::from(source[start + kx * C]), taps[kx], subtotal);
                                    }
                                    sum += subtotal;
                                }
                                expected[y * width * C + x] = sum as u8;
                            }
                        }
                        let mut output = vec![93; offset + source.len() + 16];
                        output[offset..offset + source.len()].copy_from_slice(source);
                        convolve::<C, K>(source, &mut output[offset..offset + source.len()], width, &weights, -13.25);
                        assert_eq!(&output[offset..offset + source.len()], expected);
                        assert!(output[..offset].iter().all(|&v| v == 93));
                        assert!(output[offset + source.len()..].iter().all(|&v| v == 93));
                    }
                }
            }
        }
        check::<1, 3>();
        check::<2, 3>();
        check::<3, 3>();
        check::<4, 3>();
        check::<1, 5>();
        check::<2, 5>();
        check::<3, 5>();
        check::<4, 5>();
    }

    #[test]
    fn convolution_orientation_and_borders() {
        let input: Vec<u8> = (0..25).collect();
        let mut result = input.clone();
        let mut kernel = [0.0; 9];
        kernel[0] = 1.0;
        convolve::<1, 3>(&input, &mut result, 5, &kernel, 0.0);
        assert_eq!(result[12], input[16]);
        assert_eq!(&result[..5], &input[..5]);
        // Interleaved channels stay independent, including alpha.
        let rgba: Vec<u8> = (0..100).collect();
        let mut result = rgba.clone();
        convolve::<4, 3>(&rgba, &mut result, 5, &kernel, 0.0);
        assert_eq!(&result[48..52], &rgba[64..68]);
        assert_eq!(&result[..20], &rgba[..20]);
    }

    #[test]
    fn mode_ties_and_rare_values() {
        let mut histogram = [0; 256];
        histogram[1] = 2;
        histogram[2] = 2;
        assert_eq!(select_mode(&histogram, &[0; 4], 99), 99);
        histogram[1] = 3;
        histogram[2] = 3;
        assert_eq!(select_mode(&histogram, &[6, 0, 0, 0], 99), 1);
    }

    #[test]
    fn hierarchical_rank_matches_sorted_values() {
        let mut histogram = [0; 256];
        let mut coarse = [0; 16];
        let mut sorted = Vec::new();
        for value in 0..256 {
            let count = ((value * 37) % 7) as u64;
            histogram[value] = count;
            coarse[value >> 4] += count;
            sorted.extend(std::iter::repeat_n(value as u8, count as usize));
        }
        for (rank, expected) in sorted.into_iter().enumerate() {
            assert_eq!(select_rank(&histogram, &coarse, rank as u64), expected);
        }
    }
}
