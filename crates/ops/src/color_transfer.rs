//! Reference-based histogram and mean/variance transfer in encoded sample space.

use std::simd::cmp::{SimdOrd, SimdPartialEq};
use std::simd::num::{SimdInt, SimdUint};
use std::simd::{Select, Simd};

use blanket_core::buffers::{copy, output};
use blanket_core::pixels::{load, load_sample_bytes, store, store_sample_bytes};
use blanket_core::{Image, PixelMode};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(ops_reference_transfer, module)?)?;
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Transfer {
    Histogram,
    Moments,
}

struct Raster<'a> {
    bytes: &'a [u8],
    channels: usize,
    colors: usize,
    sample_bytes: usize,
    maximum: usize,
    big_endian: bool,
}

impl<'a> Raster<'a> {
    fn new(image: &'a Image) -> PyResult<Self> {
        if image.palette().is_some() {
            return Err(PyValueError::new_err("reference transfer requires unsigned grayscale or RGB images"));
        }
        let colors = match image.pixel_mode() {
            PixelMode::L | PixelMode::La | PixelMode::I16 | PixelMode::I16L | PixelMode::I16B | PixelMode::I16N => 1,
            PixelMode::Rgb | PixelMode::Rgba => 3,
            _ => return Err(PyValueError::new_err("reference transfer requires unsigned grayscale or RGB images")),
        };
        if !matches!(image.bit_depth(), 8 | 10 | 12 | 16) || image.width() == 0 || image.height() == 0 {
            return Err(PyValueError::new_err("reference transfer requires nonempty 8/10/12/16-bit images"));
        }
        Ok(Self {
            bytes: image.raw_data()?,
            channels: image.pixel_mode().channels(),
            colors,
            sample_bytes: if image.bit_depth() == 8 { 1 } else { 2 },
            maximum: (1 << image.bit_depth()) - 1,
            big_endian: image.pixel_mode().is_big_endian(),
        })
    }

    fn sample(&self, pixel: &[u8], channel: usize) -> usize {
        let offset = channel * self.sample_bytes;
        if self.sample_bytes == 1 {
            usize::from(pixel[offset])
        } else {
            let bytes = [pixel[offset], pixel[offset + 1]];
            usize::from(if self.big_endian { u16::from_be_bytes(bytes) } else { u16::from_le_bytes(bytes) })
        }
    }

    fn visible(&self, pixel: &[u8]) -> bool {
        self.channels == self.colors || self.sample(pixel, self.colors) != 0
    }

    fn luminance(&self, pixel: &[u8]) -> usize {
        if self.colors == 1 {
            self.sample(pixel, 0)
        } else {
            (299 * self.sample(pixel, 0) + 587 * self.sample(pixel, 1) + 114 * self.sample(pixel, 2) + 500) / 1000
        }
    }

    fn histograms(&self, luminance: bool) -> Vec<Vec<u64>> {
        match self.channels {
            1 => self.histograms_channels::<1>(luminance),
            2 => self.histograms_channels::<2>(luminance),
            3 => self.histograms_channels::<3>(luminance),
            4 => self.histograms_channels::<4>(luminance),
            _ => unreachable!(),
        }
    }

    fn load<const C: usize>(&self, source: &[u8]) -> [Simd<u32, 16>; C] {
        if self.sample_bytes == 1 {
            load::<C>(source).map(|value| value.cast())
        } else {
            load_sample_bytes::<C>(source).map(|value| {
                let value = if self.big_endian { (value << 8) | (value >> 8) } else { value };
                value.cast()
            })
        }
    }

    fn store<const C: usize>(&self, output: &mut [u8], values: [Simd<u32, 16>; C]) {
        if self.sample_bytes == 1 {
            store::<C>(output, values.map(|value| value.cast()));
        } else {
            store_sample_bytes::<C>(
                output,
                values.map(|value| {
                    let value = value.cast::<u16>();
                    if self.big_endian { (value << 8) | (value >> 8) } else { value }
                }),
            );
        }
    }

    fn luma_simd<const C: usize>(&self, values: &[Simd<u32, 16>; C]) -> Simd<u32, 16> {
        if self.colors == 1 {
            values[0]
        } else {
            (values[0] * Simd::splat(299) + values[1] * Simd::splat(587) + values[2] * Simd::splat(114) + Simd::splat(500)) / Simd::splat(1000)
        }
    }

    fn histograms_channels<const C: usize>(&self, luminance: bool) -> Vec<Vec<u64>> {
        let mut bins = vec![vec![0; self.maximum + 1]; if luminance { 1 } else { self.colors }];
        let stride = C * self.sample_bytes;
        let end = self.bytes.len() / (16 * stride) * (16 * stride);
        for offset in (0..end).step_by(16 * stride) {
            let values = self.load::<C>(&self.bytes[offset..]);
            let visible = if C == self.colors { [true; 16] } else { values[self.colors].simd_ne(Simd::splat(0)).to_array() };
            let luma = if luminance { self.luma_simd(&values) } else { Simd::splat(0) };
            for (channel, counts) in bins.iter_mut().enumerate() {
                let samples = if luminance { luma } else { values[channel] }.to_array();
                // Repeated bins must accumulate sequentially; SIMD scatter would lose counts.
                for (value, visible) in samples.into_iter().zip(visible) {
                    if visible {
                        counts[value as usize] += 1;
                    }
                }
            }
        }
        for pixel in self.bytes[end..].chunks_exact(stride) {
            if self.visible(pixel) {
                for (channel, counts) in bins.iter_mut().enumerate() {
                    counts[if luminance { self.luminance(pixel) } else { self.sample(pixel, channel) }] += 1;
                }
            }
        }
        bins
    }

    fn apply(&self, pixels: &mut [u8], tables: &[Vec<usize>], luminance: bool) {
        match self.channels {
            1 => self.apply_channels::<1>(pixels, tables, luminance),
            2 => self.apply_channels::<2>(pixels, tables, luminance),
            3 => self.apply_channels::<3>(pixels, tables, luminance),
            4 => self.apply_channels::<4>(pixels, tables, luminance),
            _ => unreachable!(),
        }
    }

    fn apply_channels<const C: usize>(&self, pixels: &mut [u8], tables: &[Vec<usize>], luminance: bool) {
        let stride = C * self.sample_bytes;
        let end = pixels.len() / (16 * stride) * (16 * stride);
        for offset in (0..end).step_by(16 * stride) {
            let mut values = self.load::<C>(&pixels[offset..]);
            let visible = if C == self.colors { Simd::splat(1).simd_ne(Simd::splat(0)) } else { values[self.colors].simd_ne(Simd::splat(0)) };
            let delta = if luminance {
                let luma = self.luma_simd(&values);
                Simd::<usize, 16>::gather_or_default(&tables[0], luma.cast()).cast::<i32>() - luma.cast::<i32>()
            } else {
                Simd::splat(0)
            };
            for channel in 0..self.colors {
                let value = values[channel];
                let mapped = if luminance {
                    (value.cast::<i32>() + delta).simd_clamp(Simd::splat(0), Simd::splat(self.maximum as i32)).cast::<u32>()
                } else {
                    Simd::<usize, 16>::gather_or_default(&tables[channel], value.cast()).cast::<u32>()
                };
                values[channel] = visible.select(mapped, value);
            }
            self.store::<C>(&mut pixels[offset..], values);
        }
        self.apply_tail(&mut pixels[end..], tables, luminance);
    }

    fn apply_tail(&self, pixels: &mut [u8], tables: &[Vec<usize>], luminance: bool) {
        for pixel in pixels.chunks_exact_mut(self.channels * self.sample_bytes) {
            if !self.visible(pixel) {
                continue;
            }
            let old_luma = self.luminance(pixel);
            for channel in 0..self.colors {
                let value = self.sample(pixel, channel);
                let mapped = if luminance {
                    (value as isize + tables[0][old_luma] as isize - old_luma as isize).clamp(0, self.maximum as isize) as usize
                } else {
                    tables[channel][value]
                };
                let offset = channel * self.sample_bytes;
                if self.sample_bytes == 1 {
                    pixel[offset] = mapped as u8;
                } else {
                    let value = mapped as u16;
                    pixel[offset..offset + 2].copy_from_slice(&if self.big_endian { value.to_be_bytes() } else { value.to_le_bytes() });
                }
            }
        }
    }
}

fn moments(counts: &[u64], total: u64) -> (f64, f64) {
    let mean = counts.iter().enumerate().map(|(value, &count)| value as f64 * count as f64).sum::<f64>() / total as f64;
    let variance = counts.iter().enumerate().map(|(value, &count)| (value as f64 - mean).powi(2) * count as f64).sum::<f64>() / total as f64;
    (mean, variance.sqrt())
}

fn table(source: &[u64], reference: &[u64], transfer: Transfer) -> PyResult<Vec<usize>> {
    let source_total = source.iter().sum::<u64>();
    let reference_total = reference.iter().sum::<u64>();
    if source_total == 0 || reference_total == 0 {
        return Err(PyValueError::new_err("source and reference must each contain a pixel with nonzero alpha"));
    }
    let maximum = (source.len() - 1) as f64;
    let scale = maximum / (reference.len() - 1) as f64;
    Ok(match transfer {
        Transfer::Histogram => {
            let mut cumulative = 0_u64;
            let mut target = reference.iter().position(|&count| count != 0).expect("nonempty reference histogram");
            let mut target_cumulative = reference[target];
            source
                .iter()
                .map(|&count| {
                    cumulative += count;
                    while target + 1 < reference.len()
                        && u128::from(target_cumulative) * u128::from(source_total) < u128::from(cumulative) * u128::from(reference_total)
                    {
                        target += 1;
                        target_cumulative += reference[target];
                    }
                    (target as f64 * scale).round() as usize
                })
                .collect()
        }
        Transfer::Moments => {
            let (source_mean, source_std) = moments(source, source_total);
            let (reference_mean, reference_std) = moments(reference, reference_total);
            let gain = if source_std == 0.0 { 0.0 } else { reference_std * scale / source_std };
            (0..source.len())
                .map(|value| ((value as f64 - source_mean) * gain + reference_mean * scale).round().clamp(0.0, maximum) as usize)
                .collect()
        }
    })
}

#[pyfunction]
fn ops_reference_transfer(py: Python<'_>, image: &Image, reference: &Image, mode: &str, method: &str) -> PyResult<Image> {
    let luminance = match mode {
        "channels" => false,
        "luminance" => true,
        _ => return Err(PyValueError::new_err("mode must be 'channels' or 'luminance'")),
    };
    let transfer = match method {
        "histogram" => Transfer::Histogram,
        "moments" => Transfer::Moments,
        _ => return Err(PyValueError::new_err("unknown reference transfer method")),
    };
    let source = Raster::new(image)?;
    let reference = Raster::new(reference)?;
    if !luminance && source.colors != reference.colors {
        return Err(PyValueError::new_err("channel matching requires the same number of color channels"));
    }
    let pixels = py.detach(|| {
        let source_bins = source.histograms(luminance);
        let reference_bins = reference.histograms(luminance);
        let tables = source_bins.iter().zip(&reference_bins).map(|(src, dst)| table(src, dst, transfer)).collect::<PyResult<Vec<_>>>()?;
        let mut pixels = copy(source.bytes)?;
        source.apply(&mut pixels, &tables, luminance);
        Ok::<_, PyErr>(pixels)
    })?;
    output(image, image.size(), pixels)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cdf_comparison_uses_exact_integer_ratios() {
        let source = [0, 1, 2, 0, 1];
        let reference = [0, 2, 0, 2, 4];
        assert_eq!(table(&source, &reference, Transfer::Histogram).unwrap(), [1, 1, 4, 4, 4]);
        let count = u64::MAX / 4;
        assert_eq!(table(&[count, count], &[count, count], Transfer::Histogram).unwrap(), [0, 1]);
    }

    #[test]
    fn constant_moments_use_reference_mean() {
        assert_eq!(table(&[0, 2, 0], &[1, 0, 1], Transfer::Moments).unwrap(), [1, 1, 1]);
        assert_eq!(table(&[1, 0, 1], &[0, 2, 0], Transfer::Moments).unwrap(), [1, 1, 1]);
    }

    #[test]
    fn unaligned_wide_samples_and_output_guards() {
        for big_endian in [false, true] {
            for count in [1, 15, 16, 17, 31, 32, 33, 65] {
                let mut storage = vec![173];
                for i in 0..count {
                    let value = (i * 997) as u16;
                    storage.extend(if big_endian { value.to_be_bytes() } else { value.to_le_bytes() });
                }
                storage.push(173);
                let raster = Raster { bytes: &storage[1..storage.len() - 1], channels: 1, colors: 1, sample_bytes: 2, maximum: 65535, big_endian };
                let bins = raster.histograms(false);
                assert_eq!(bins[0].iter().sum::<u64>(), count as u64);
                let identity: Vec<usize> = (0..=65535).collect();
                let mut result = storage.clone();
                let end = result.len() - 1;
                raster.apply(&mut result[1..end], &[identity], false);
                assert_eq!(result, storage);
            }
        }
    }

    #[test]
    fn simd_matches_scalar_histograms_and_application() {
        for (channels, colors, sample_bytes) in [(1, 1, 1), (2, 1, 1), (3, 3, 1), (4, 3, 1), (1, 1, 2), (3, 3, 2), (4, 3, 2)] {
            for maximum in if sample_bytes == 1 { &[255][..] } else { &[1023, 4095, 65535][..] } {
                for big_endian in [false, true] {
                    for count in [0, 1, 15, 16, 17, 31, 32, 33, 65, 257] {
                        let mut storage = vec![173];
                        for i in 0..count * channels {
                            let value = if i % 7 == 0 { 0 } else { (i * 997 + 113) % (maximum + 1) } as u16;
                            if sample_bytes == 1 {
                                storage.push(value as u8);
                            } else {
                                storage.extend(if big_endian { value.to_be_bytes() } else { value.to_le_bytes() });
                            }
                        }
                        storage.push(173);
                        let end = storage.len() - 1;
                        let raster = Raster { bytes: &storage[1..end], channels, colors, sample_bytes, maximum: *maximum, big_endian };
                        for luminance in [false, true] {
                            let mut expected_bins = vec![vec![0; maximum + 1]; if luminance { 1 } else { colors }];
                            for pixel in raster.bytes.chunks_exact(channels * sample_bytes) {
                                if raster.visible(pixel) {
                                    for (c, bins) in expected_bins.iter_mut().enumerate() {
                                        bins[if luminance { raster.luminance(pixel) } else { raster.sample(pixel, c) }] += 1;
                                    }
                                }
                            }
                            assert_eq!(raster.histograms(luminance), expected_bins);
                            let tables: Vec<Vec<usize>> =
                                (0..expected_bins.len()).map(|c| (0..=*maximum).map(|v| (maximum - v + c * 97) % (maximum + 1)).collect()).collect();
                            let mut expected = storage.clone();
                            raster.apply_tail(&mut expected[1..end], &tables, luminance);
                            let mut actual = storage.clone();
                            raster.apply(&mut actual[1..end], &tables, luminance);
                            assert_eq!(actual, expected, "channels={channels} bytes={sample_bytes} count={count} luminance={luminance}");
                            assert_eq!(actual[0], 173);
                            assert_eq!(actual[end], 173);
                        }
                    }
                }
            }
        }
    }
}
