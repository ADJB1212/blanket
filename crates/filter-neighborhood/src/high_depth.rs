use std::collections::{BTreeMap, VecDeque};
use std::simd::Simd;
use std::simd::cmp::SimdOrd;
use std::simd::num::{SimdFloat, SimdUint};

use blanket_core::pixels::Samples;
use blanket_core::{Image, samples};
use pyo3::prelude::*;

pub(crate) fn kernel(image: &Image, size: usize, scale: f32, offset: f32, weights: &[f32]) -> PyResult<Image> {
    let source = samples::view(image)?;
    samples::with_output(image, Some(&source), |values| kernel_values(image, &source, values, size, scale, offset, weights))
}

fn kernel_values(image: &Image, source: &[u16], values: &mut [u16], size: usize, scale: f32, offset: f32, weights: &[f32]) {
    let (width, height, channels) = (image.width() as usize, image.height() as usize, image.pixel_mode().channels());
    let radius = size / 2;
    if width >= size && height >= size {
        let mut normalized = [0.0; 25];
        for (dst, weight) in normalized.iter_mut().zip(weights) {
            *dst = weight / scale;
        }
        let weights = &normalized[..weights.len()];
        let stride = width * channels;
        blanket_core::parallel::chunks_mut_above(&mut values[radius * stride..(height - radius) * stride], stride * 16, 256 * 1024, |band, rows| {
            for (row_index, output) in rows.chunks_exact_mut(stride).enumerate() {
                let y = radius + band * 16 + row_index;
                let first = (y * width + radius) * channels;
                let last = (y * width + width - radius) * channels;
                let end = first + (last - first) / 16 * 16;
                for i in (first..end).step_by(16) {
                    let mut sum = Simd::splat(offset + 0.5);
                    for ky in 0..size {
                        let start = i + radius * width * channels - ky * width * channels - radius * channels;
                        let at = |kx| Samples::from_slice(&source[start + kx * channels..]).cast::<f32>();
                        let mut row = super::kernel_add_simd(at(0), Simd::splat(weights[ky * size]), at(1) * Simd::splat(weights[ky * size + 1]));
                        for kx in 2..size {
                            row = super::kernel_add_simd(at(kx), Simd::splat(weights[ky * size + kx]), row);
                        }
                        sum += row;
                    }
                    sum.cast::<u16>()
                        .simd_min(Samples::splat(samples::maximum(image)))
                        .copy_to_slice(&mut output[i - y * stride..i - y * stride + 16]);
                }
                for i in end..last {
                    let (x, c) = ((i / channels) % width, i % channels);
                    let mut sum = offset + 0.5;
                    for ky in 0..size {
                        let start = ((y + radius - ky) * width + x - radius) * channels + c;
                        let mut row = f32::from(source[start + channels]) * weights[ky * size + 1];
                        row = super::kernel_add(f32::from(source[start]), weights[ky * size], row);
                        for kx in 2..size {
                            row = super::kernel_add(f32::from(source[start + kx * channels]), weights[ky * size + kx], row);
                        }
                        sum += row;
                    }
                    output[x * channels + c] = sum.clamp(0.0, f32::from(samples::maximum(image))) as u16;
                }
            }
        });
    }
}

pub(crate) fn neighborhood(image: &Image, radius: usize, rank: Option<u64>) -> PyResult<Image> {
    if radius == 0 {
        return samples::output_bytes(image, blanket_core::buffers::copy(image.raw_data()?)?);
    }
    let source = samples::view(image)?;
    let (width, height, channels) = (image.width() as usize, image.height() as usize, image.pixel_mode().channels());
    samples::with_output(image, rank.is_none().then_some(source.as_ref()), |values| {
        if radius > 2 && rank.is_some_and(|rank| rank == 0 || rank == (radius as u64 * 2 + 1).pow(2) - 1) {
            extrema(&source, values, width, height, channels, radius, rank != Some(0));
            return;
        }
        if let Some(rank) = rank
            && (radius == 1 || radius == 2 && (rank == 0 || rank == 24))
        {
            small_rank(&source, values, width, height, channels, radius, rank as usize);
            return;
        }
        sliding_neighborhood(&source, values, (width, height, channels, samples::maximum(image)), radius, rank);
    })
}

fn extrema(source: &[u16], output: &mut [u16], width: usize, height: usize, channels: usize, radius: usize, maximum: bool) {
    if source.is_empty() {
        return;
    }
    let stride = width * channels;
    let mut horizontal = vec![0; source.len()];
    blanket_core::parallel::chunks_mut_above(&mut horizontal, stride * 16, 256 * 1024, |band, rows| {
        let mut queue = VecDeque::new();
        for (i, row) in rows.chunks_exact_mut(stride).enumerate() {
            let input = &source[(band * 16 + i) * stride..][..stride];
            for channel in 0..channels {
                super::extrema_line(&input[channel..], &mut row[channel..], channels, 0, radius, maximum, &mut queue);
            }
        }
    });
    let band_rows = 64.max(radius.min(height));
    blanket_core::parallel::chunks_mut_above(output, stride * band_rows, 256 * 1024, |band, rows| {
        let mut queue = VecDeque::new();
        for column in 0..stride {
            super::extrema_line(&horizontal[column..], &mut rows[column..], stride, band * band_rows, radius, maximum, &mut queue);
        }
    });
}

pub(super) struct RankCounts {
    pub(super) tree: Vec<u32>,
}

impl RankCounts {
    pub(super) fn update(&mut self, value: usize, weight: u64, add: bool) {
        let weight = u32::try_from(weight).expect("validated rank window weight");
        let mut index = value + 1;
        while index < self.tree.len() {
            if add {
                self.tree[index] += weight;
            } else {
                self.tree[index] -= weight;
            }
            index += index.isolate_lowest_one();
        }
    }

    pub(super) fn select(&self, mut rank: u64) -> usize {
        let mut index = 0;
        let mut step = 1 << (usize::BITS - 1 - (self.tree.len() - 1).leading_zeros());
        while step != 0 {
            let next = index + step;
            if next < self.tree.len() && u64::from(self.tree[next]) <= rank {
                rank -= u64::from(self.tree[next]);
                index = next;
            }
            step >>= 1;
        }
        index
    }
}

fn sliding_neighborhood(source: &[u16], values: &mut [u16], dimensions: (usize, usize, usize, u16), radius: usize, rank: Option<u64>) {
    let (width, height, channels, _) = dimensions;
    if width == 0 || height == 0 {
        return;
    }
    blanket_core::parallel::chunks_mut_above(values, width * channels * 16, 256 * 1024, |band, rows| {
        sliding_neighborhood_rows(source, rows, dimensions, radius, rank, band * 16);
    });
}

fn sliding_neighborhood_rows(
    source: &[u16], values: &mut [u16], dimensions: (usize, usize, usize, u16), radius: usize, rank: Option<u64>, first: usize,
) {
    let (width, height, channels, maximum) = dimensions;
    let mut histogram = BTreeMap::<u16, u64>::new();
    // Amortize the fixed-size count tree over sufficiently large windows and rows.
    let mut counts = RankCounts { tree: if rank.is_some() && radius >= 4 && width >= 64 { vec![0; usize::from(maximum) + 2] } else { Vec::new() } };
    let maximum_rank = (radius as u64 * 2 + 1).pow(2) - 1;
    for y in first..first + values.len() / (width * channels) {
        for c in 0..channels {
            histogram.clear();
            counts.tree.fill(0);
            let column = |histogram: &mut BTreeMap<u16, u64>, counts: &mut RankCounts, xx: usize, wx: u64, add: bool| {
                for yy in y.saturating_sub(radius)..=(y + radius).min(height - 1) {
                    let wy = if rank.is_none() {
                        1
                    } else {
                        1 + if yy == 0 { radius.saturating_sub(y) } else { 0 }
                            + if yy + 1 == height { (y + radius).saturating_sub(height - 1) } else { 0 }
                    };
                    let value = source[(yy * width + xx) * channels + c];
                    let weight = wx * wy as u64;
                    if !counts.tree.is_empty() {
                        counts.update(usize::from(value), weight, add);
                    } else if add {
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
                column(&mut histogram, &mut counts, xx, weight as u64, true);
            }
            for x in 0..width {
                let dst = &mut values[((y - first) * width + x) * channels + c];
                if !counts.tree.is_empty() {
                    *dst = counts.select(rank.unwrap()) as u16;
                } else if rank == Some(0) {
                    *dst = *histogram.first_key_value().unwrap().0;
                } else if rank == Some(maximum_rank) {
                    *dst = *histogram.last_key_value().unwrap().0;
                } else if let Some(mut rank) = rank {
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
                        column(&mut histogram, &mut counts, x.saturating_sub(radius), 1, false);
                    }
                    if rank.is_some() || x + radius + 1 < width {
                        column(&mut histogram, &mut counts, (x + radius + 1).min(width - 1), 1, true);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extrema_matches_windows_across_parallel_bands() {
        let (width, height, channels) = (257, 257, 4);
        let source: Vec<u16> = (0..width * height * channels).map(|i| (i * 719 + i / 7) as u16).collect();
        for radius in [3, 9, 80] {
            for maximum in [false, true] {
                let mut output = vec![173; source.len() + 2];
                extrema(&source, &mut output[1..source.len() + 1], width, height, channels, radius, maximum);
                for y in [0_usize, 1, 63, 64, 79, 80, 127, 128, 255, 256] {
                    for x in [0_usize, 1, 63, 128, 255, 256] {
                        for c in 0..channels {
                            let mut expected = if maximum { 0 } else { u16::MAX };
                            for yy in y.saturating_sub(radius)..=(y + radius).min(height - 1) {
                                for xx in x.saturating_sub(radius)..=(x + radius).min(width - 1) {
                                    let value = source[(yy * width + xx) * channels + c];
                                    expected = if maximum { expected.max(value) } else { expected.min(value) };
                                }
                            }
                            assert_eq!(output[1 + (y * width + x) * channels + c], expected);
                        }
                    }
                }
                assert_eq!((output[0], output[source.len() + 1]), (173, 173));
            }
        }
    }

    #[test]
    fn rank_counts_preserve_weighted_order_and_removals() {
        let mut counts = RankCounts { tree: vec![0; 65537] };
        for (value, weight) in [(0, 3), (1, 2), (32767, 7), (32768, 1), (65535, 9)] {
            counts.update(value, weight, true);
        }
        let expected =
            [0, 0, 0, 1, 1, 32767, 32767, 32767, 32767, 32767, 32767, 32767, 32768, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535];
        for (rank, value) in expected.into_iter().enumerate() {
            assert_eq!(counts.select(rank as u64), value);
        }
        counts.update(32767, 7, false);
        assert_eq!(counts.select(5), 32768);
        assert_eq!(counts.select(6), 65535);
    }

    #[test]
    fn rank_counts_cover_non_power_of_two_domains() {
        for bins in 1..=257 {
            let mut counts = RankCounts { tree: vec![0; bins + 1] };
            for value in 0..bins {
                counts.update(value, 3, true);
            }
            for rank in 0..bins * 3 {
                assert_eq!(counts.select(rank as u64), rank / 3);
            }
            for value in 0..bins {
                counts.update(value, 2, false);
            }
            for rank in 0..bins {
                assert_eq!(counts.select(rank as u64), rank);
            }
        }
    }

    #[test]
    fn compact_rank_counts_cover_largest_supported_window() {
        let count = 23169_u64.pow(2);
        let mut counts = RankCounts { tree: vec![0; 4] };
        assert_eq!(size_of_val(counts.tree.as_slice()), 16);
        counts.update(0, count / 2, true);
        counts.update(2, count - count / 2, true);
        assert_eq!(counts.select(count / 2 - 1), 0);
        assert_eq!(counts.select(count / 2), 2);
        assert_eq!(counts.select(count - 1), 2);
        counts.update(0, count / 2, false);
        assert_eq!(counts.select(0), 2);
    }

    #[test]
    fn parallel_ranks_match_sorted_windows() {
        let (width, height, channels) = (257, 257, 4);
        for depth in [10, 12, 16] {
            let maximum = ((1_u32 << depth) - 1) as u16;
            let source: Vec<u16> = (0..width * height * channels).map(|i| (i * 7919) as u16 & maximum).collect();
            for rank in [1, 40, 79] {
                let mut output = vec![173; source.len() + 2];
                sliding_neighborhood(&source, &mut output[1..source.len() + 1], (width, height, channels, maximum), 4, Some(rank));
                for y in [0_usize, 15, 16, 127, 128, 255, 256] {
                    for x in [0_usize, 63, 128, 256] {
                        for c in 0..channels {
                            let mut window = Vec::new();
                            for yy in 0..9 {
                                for xx in 0..9 {
                                    let row = (y + yy).saturating_sub(4).min(height - 1);
                                    let col = (x + xx).saturating_sub(4).min(width - 1);
                                    window.push(source[(row * width + col) * channels + c]);
                                }
                            }
                            window.sort_unstable();
                            assert_eq!(output[1 + (y * width + x) * channels + c], window[rank as usize]);
                        }
                    }
                }
                assert_eq!((output[0], output[source.len() + 1]), (173, 173));
            }
        }
    }

    #[test]
    fn sliding_neighborhood_matches_rebuilt_windows() {
        for (width, height) in [(0, 0), (1, 1), (1, 7), (7, 1), (9, 6), (33, 17), (65, 3)] {
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
                            sliding_neighborhood(&source, &mut output[1..source.len() + 1], (width, height, channels, u16::MAX), radius, rank);
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
}
