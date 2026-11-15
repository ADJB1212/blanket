//! Portable SIMD arithmetic for radial maps and coordinate interpolation.

use std::simd::num::SimdFloat;
use std::simd::{Mask, Select, Simd};

#[derive(Debug)]
pub struct Accumulator {
    values: Simd<f64, 4>,
    alpha: Option<usize>,
    maximum: f64,
}

impl Accumulator {
    pub fn new(alpha: Option<usize>, maximum: f64) -> Self {
        assert!(alpha.is_none_or(|channel| channel < 4));
        Self { values: Simd::splat(0.0), alpha, maximum }
    }

    pub fn add(&mut self, samples: [f64; 4], weight: f64) {
        let mut samples = Simd::from_array(samples);
        if let Some(alpha) = self.alpha {
            let coverage = samples[alpha] / self.maximum;
            let colors = if coverage == 0.0 { Simd::splat(0.0) } else { samples * Simd::splat(coverage) };
            let color_channels = Mask::<i64, 4>::from_array(std::array::from_fn(|i| i != alpha));
            samples = color_channels.select(colors, samples);
        }
        self.values += Simd::splat(weight) * samples;
    }

    pub fn finish(self) -> [f64; 4] {
        let mut values = self.values;
        if let Some(alpha) = self.alpha {
            let coverage = values[alpha] / self.maximum;
            let colors = if coverage > 0.0 { values / Simd::splat(coverage) } else { Simd::splat(0.0) };
            let color_channels = Mask::<i64, 4>::from_array(std::array::from_fn(|i| i != alpha));
            values = color_channels.select(colors, values);
        }
        values.to_array()
    }
}

pub fn lens_map(output: &mut [u8], width: usize, start: usize, coefficients: [f64; 3], center: (f64, f64), focal: (f64, f64)) {
    assert!(width > 0 && output.len().is_multiple_of(12));
    let end = output.len() / 48 * 48;
    for (block, destination) in output[..end].as_chunks_mut::<48>().0.iter_mut().enumerate() {
        let index = start + block * 4;
        let column = index % width;
        let row = index / width;
        let (x, y) = if column + 4 <= width {
            (Simd::splat(column as f64) + Simd::from_array([0.0, 1.0, 2.0, 3.0]), Simd::splat(row as f64))
        } else {
            (
                Simd::<f64, 4>::from_array(std::array::from_fn(|i| ((index + i) % width) as f64)),
                Simd::<f64, 4>::from_array(std::array::from_fn(|i| ((index + i) / width) as f64)),
            )
        };
        let (sx, sy) = if coefficients == [0.0; 3] {
            (x.cast::<f32>() + Simd::splat(0.5), y.cast::<f32>() + Simd::splat(0.5))
        } else {
            let dx = x + Simd::splat(0.5) - Simd::splat(center.0);
            let dy = y + Simd::splat(0.5) - Simd::splat(center.1);
            let u = dx / Simd::splat(focal.0);
            let v = dy / Simd::splat(focal.1);
            let r2 = u * u + v * v;
            let [k1, k2, k3] = coefficients.map(Simd::splat);
            let scale = Simd::splat(1.0) + r2 * (k1 + r2 * (k2 + r2 * k3));
            ((Simd::splat(center.0) + dx * scale).cast::<f32>(), (Simd::splat(center.1) + dy * scale).cast::<f32>())
        };
        for (i, pixel) in destination.as_chunks_mut::<12>().0.iter_mut().enumerate() {
            pixel[..4].copy_from_slice(&sx[i].to_le_bytes());
            pixel[4..8].copy_from_slice(&sy[i].to_le_bytes());
            pixel[8..].fill(0);
        }
    }
    for (i, pixel) in output[end..].as_chunks_mut::<12>().0.iter_mut().enumerate() {
        let index = start + end / 12 + i;
        let x = (index % width) as f64;
        let y = (index / width) as f64;
        let (sx, sy) = if coefficients == [0.0; 3] {
            (x as f32 + 0.5, y as f32 + 0.5)
        } else {
            let dx = x + 0.5 - center.0;
            let dy = y + 0.5 - center.1;
            let u = dx / focal.0;
            let v = dy / focal.1;
            let r2 = u * u + v * v;
            let [k1, k2, k3] = coefficients;
            let scale = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3));
            ((center.0 + dx * scale) as f32, (center.1 + dy * scale) as f32)
        };
        pixel[..4].copy_from_slice(&sx.to_le_bytes());
        pixel[4..8].copy_from_slice(&sy.to_le_bytes());
        pixel[8..].fill(0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_float(actual: f64, expected: f64) {
        assert!(actual.to_bits() == expected.to_bits() || actual.is_nan() && expected.is_nan(), "{actual:?} != {expected:?}");
    }

    #[test]
    fn accumulation_preserves_scalar_order_alpha_and_nonfinite_samples() {
        let inputs = [
            ([0.1, 0.3, 0.7, 1.0], 0.0448),
            ([65535.0, 4095.0, 1023.0, 127.0], -0.0625),
            ([11.25, 231.75, 0.5, 255.0], 0.5625),
            ([f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0], -0.0625),
            ([255.0, 17.0, 39.0, 127.0], 0.375),
        ];
        for maximum in [1.0, 255.0, 1023.0, 4095.0, 65535.0] {
            for alpha in [None, Some(1), Some(3)] {
                let mut actual = Accumulator::new(alpha, maximum);
                let mut expected = [0.0; 4];
                for (samples, weight) in inputs {
                    actual.add(samples, weight);
                    let coverage = alpha.map_or(1.0, |a| samples[a] / maximum);
                    for channel in 0..4 {
                        let sample = if alpha.is_some() && alpha != Some(channel) {
                            if coverage == 0.0 { 0.0 } else { samples[channel] * coverage }
                        } else {
                            samples[channel]
                        };
                        expected[channel] += weight * sample;
                    }
                    let mut partial_expected = expected;
                    if let Some(alpha) = alpha {
                        let coverage = expected[alpha] / maximum;
                        for (channel, value) in partial_expected.iter_mut().enumerate() {
                            if channel != alpha {
                                *value = if coverage > 0.0 { *value / coverage } else { 0.0 };
                            }
                        }
                    }
                    let partial = Accumulator { values: actual.values, alpha, maximum }.finish();
                    for (actual, expected) in partial.into_iter().zip(partial_expected) {
                        assert_float(actual, expected);
                    }
                }
                if let Some(alpha) = alpha {
                    let coverage = expected[alpha] / maximum;
                    for (channel, value) in expected.iter_mut().enumerate() {
                        if channel != alpha {
                            *value = if coverage > 0.0 { *value / coverage } else { 0.0 };
                        }
                    }
                }
                for (actual, expected) in actual.finish().into_iter().zip(expected) {
                    assert_float(actual, expected);
                }
            }
        }
    }

    #[test]
    fn radial_maps_match_scalar_at_unaligned_vector_boundaries_and_tails() {
        for width in [1, 3, 4, 5, 17] {
            for count in [0, 1, 3, 4, 5, 7, 8, 9, 17, 33] {
                for offset in 0..5 {
                    for (coefficients, center, focal) in [
                        ([0.0; 3], (1e300, -1e300), (1e-300, 1e-300)),
                        ([-0.2, 0.03, 0.005], (3.25, 2.75), (4.0, 3.0)),
                        ([1e300, -1e300, 1e300], (0.0, 0.0), (1e-300, 1e-300)),
                    ] {
                        let start = 31;
                        let mut output = vec![123; offset + count * 12 + 5];
                        lens_map(&mut output[offset..offset + count * 12], width, start, coefficients, center, focal);
                        for (i, pixel) in output[offset..offset + count * 12].as_chunks::<12>().0.iter().enumerate() {
                            let x = ((start + i) % width) as f64;
                            let y = ((start + i) / width) as f64;
                            let expected = if coefficients == [0.0; 3] {
                                [x as f32 + 0.5, y as f32 + 0.5]
                            } else {
                                let dx = x + 0.5 - center.0;
                                let dy = y + 0.5 - center.1;
                                let u = dx / focal.0;
                                let v = dy / focal.1;
                                let r2 = u * u + v * v;
                                let [k1, k2, k3] = coefficients;
                                let scale = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3));
                                [(center.0 + dx * scale) as f32, (center.1 + dy * scale) as f32]
                            };
                            for (value, expected) in pixel[..8].as_chunks::<4>().0.iter().zip(expected) {
                                let actual = f32::from_le_bytes(*value);
                                assert_float(f64::from(actual), f64::from(expected));
                            }
                            assert_eq!(&pixel[8..], &[0; 4]);
                        }
                        assert!(output[..offset].iter().chain(&output[offset + count * 12..]).all(|&v| v == 123));
                    }
                }
            }
        }
    }
}
