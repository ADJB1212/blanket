//! Save-time size reduction with a decoded-sample error bound.

use pyo3::exceptions::{PyOSError, PyValueError};
use pyo3::prelude::*;

use crate::codecs::{self, ImageFormat, SaveOptions};
use crate::compressor::LosslessImageCompressor;
use blanket_core::raster::{Image, PixelMode};
use blanket_ops::ops_simd;

#[pyclass(name = "_LossyImageCompressor", module = "blanket._blanket", frozen, from_py_object)]
#[derive(Clone, Copy)]
pub struct LossyImageCompressor {
    #[pyo3(get)]
    max_rmse: f64,
    #[pyo3(get)]
    effort: u8,
}

#[pymethods]
impl LossyImageCompressor {
    #[new]
    #[pyo3(signature = (*, max_rmse=2.0, effort=7))]
    fn new(max_rmse: f64, effort: u8) -> PyResult<Self> {
        if !max_rmse.is_finite() || !(0.0..=255.0).contains(&max_rmse) {
            return Err(PyValueError::new_err("max_rmse must be finite and between 0 and 255"));
        }
        if !(1..=10).contains(&effort) {
            return Err(PyValueError::new_err("effort must be between 1 and 10"));
        }
        Ok(Self { max_rmse, effort })
    }
}

impl LossyImageCompressor {
    pub fn encode(&self, image: &Image, format: ImageFormat, options: SaveOptions) -> PyResult<Vec<u8>> {
        let lossless = LosslessImageCompressor { effort: self.effort };
        let mut output = if format == ImageFormat::Jpeg && self.max_rmse > 0.0 && !options.lossless {
            codecs::encode(image, format, options)?
        } else {
            lossless.encode(image, format, options)?
        };
        if self.max_rmse == 0.0 || options.lossless {
            return Ok(output);
        }
        let png = format == ImageFormat::Png && image.bit_depth == 8;
        if !png
            && !matches!(
                format,
                ImageFormat::Jpeg | ImageFormat::Jxl | ImageFormat::Webp | ImageFormat::Avif | ImageFormat::Heif
            )
        {
            return Ok(output);
        }
        if png {
            return self.optimize_png(image, options, output);
        }
        let reference = codecs::decode(&output, format).map_err(PyOSError::new_err)?;
        let mut lower = 1;
        let mut upper = options.quality.saturating_sub(1);
        let mut winner_quality = None;
        let mut smallest = usize::MAX;
        let mut drop = 1_u8;
        let mut bracketed = false;
        for _ in 0..self.effort {
            if lower > upper {
                break;
            }
            let quality = if bracketed {
                lower + (upper - lower) / 2
            } else {
                options.quality.saturating_sub(drop).max(1)
            };
            let candidate = codecs::encode(image, format, SaveOptions { quality, ..options })?;
            let decoded = codecs::decode(&candidate, format).map_err(PyOSError::new_err)?;
            if within_error(&decoded, &reference, self.max_rmse)? {
                upper = quality - 1;
                drop = drop.saturating_mul(2);
                if candidate.len() < smallest {
                    smallest = candidate.len();
                    winner_quality = Some(quality);
                }
                if candidate.len() < output.len() {
                    output = candidate;
                }
            } else {
                lower = quality + 1;
                bracketed = true;
            }
        }
        if format == ImageFormat::Jpeg {
            let quality = winner_quality.unwrap_or(options.quality);
            let optimized = lossless.encode(image, format, SaveOptions { quality, ..options })?;
            if optimized.len() < output.len() {
                output = optimized;
            }
        }
        Ok(output)
    }

    fn optimize_png(&self, image: &Image, options: SaveOptions, mut output: Vec<u8>) -> PyResult<Vec<u8>> {
        let channels = image.mode.channels();
        let pixels = image.pixels.as_ref().expect("validated image");
        let histogram = png_rounding_histogram(pixels, channels);
        let limit = self.max_rmse * self.max_rmse * image.width as f64 * image.height as f64 * 3.0;
        let mut tried = vec![std::array::from_fn(|sample| sample as u8)];
        let mut winner = None;
        let mut smallest = usize::MAX;
        let mut trials = 0;
        'steps: for step in (2..=256).rev() {
            let uniform = std::array::from_fn(|sample| {
                if histogram[sample] == 0 {
                    sample as u8
                } else {
                    ((sample + step / 2) / step * step).min(255) as u8
                }
            });
            for color_table in [uniform, png_centroid_table(&histogram, step)] {
                if tried.contains(&color_table) || !png_rounding_within_error(&histogram, &color_table, channels, limit) {
                    continue;
                }
                tried.push(color_table);
                let mut table = [0; 4 * 256];
                for channel in 0..channels {
                    for sample in 0..256 {
                        table[channel * 256 + sample] = if channels == 4 && channel == 3 {
                            sample as u8
                        } else {
                            color_table[sample]
                        };
                    }
                }
                let mut rounded = image.clone();
                let destination = rounded.pixels.as_mut().expect("validated image");
                match channels {
                    1 => ops_simd::lut::<1>(pixels, destination, &color_table),
                    3 => ops_simd::lut::<3>(pixels, destination, &color_table),
                    4 => ops_simd::lut::<4>(pixels, destination, &table),
                    _ => unreachable!("validated mode"),
                }
                let candidate = codecs::encode(&rounded, ImageFormat::Png, options)?;
                if candidate.len() < smallest {
                    smallest = candidate.len();
                    winner = Some(rounded);
                    if candidate.len() < output.len() {
                        output = candidate;
                    }
                }
                trials += 1;
                if trials == self.effort {
                    break 'steps;
                }
            }
        }
        if let Some(winner) = winner {
            let optimized = LosslessImageCompressor { effort: self.effort }.encode(&winner, ImageFormat::Png, options)?;
            if optimized.len() < output.len() {
                output = optimized;
            }
        }
        Ok(output)
    }
}

fn png_centroid_table(histogram: &[u64; 256], step: usize) -> [u8; 256] {
    let mut table = std::array::from_fn(|sample| sample as u8);
    for start in (0..256).step_by(step) {
        let end = (start + step).min(256);
        let count: u64 = histogram[start..end].iter().sum();
        if count == 0 {
            continue;
        }
        let total: u64 = (start..end).map(|sample| sample as u64 * histogram[sample]).sum();
        let centroid = ((total + count / 2) / count) as u8;
        for sample in start..end {
            if histogram[sample] != 0 {
                table[sample] = centroid;
            }
        }
    }
    table
}

fn within_error(candidate: &Image, reference: &Image, max_rmse: f64) -> PyResult<bool> {
    if (candidate.width, candidate.height) != (reference.width, reference.height) {
        return Ok(false);
    }
    let candidate_data = candidate.raw_data()?;
    let reference_data = reference.raw_data()?;
    let count = candidate.width as usize * candidate.height as usize;
    let limit = max_rmse * max_rmse * count as f64 * 3.0;
    if candidate.bit_depth == 8 && reference.bit_depth == 8 && candidate.mode == reference.mode {
        return Ok(within_error_8bit(candidate_data, reference_data, candidate.mode.channels(), limit));
    }
    let mut error = 0.0;
    for index in 0..count {
        let actual = rgba_sample(candidate, candidate_data, index);
        let expected = rgba_sample(reference, reference_data, index);
        if actual[3] != expected[3] {
            return Ok(false);
        }
        for channel in 0..3 {
            error += (actual[channel] - expected[channel]).powi(2);
        }
        if error > limit {
            return Ok(false);
        }
    }
    Ok(true)
}

fn png_rounding_histogram(pixels: &[u8], channels: usize) -> [u64; 256] {
    let mut histogram = [0_u64; 256];
    if channels == 4 {
        for pixel in pixels.as_chunks::<4>().0 {
            for &sample in &pixel[..3] {
                histogram[usize::from(sample)] += 1;
            }
        }
    } else {
        for &sample in pixels {
            histogram[usize::from(sample)] += 1;
        }
    }
    histogram
}

fn png_rounding_within_error(histogram: &[u64; 256], table: &[u8], channels: usize, limit: f64) -> bool {
    let error: u64 = histogram
        .iter()
        .enumerate()
        .map(|(sample, &count)| {
            let difference = sample.abs_diff(usize::from(table[sample])) as u64;
            count * difference * difference
        })
        .sum();
    (error * if channels == 1 { 3 } else { 1 }) as f64 <= limit
}

fn within_error_8bit(actual: &[u8], expected: &[u8], channels: usize, limit: f64) -> bool {
    let mut error = 0_u64;
    let mut offset = 0;
    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;
        // SAFETY: NEON is mandatory and each load covers a complete 16-byte block.
        unsafe {
            let alpha_mask = vreinterpretq_u8_u32(vdupq_n_u32(0xff00_0000));
            while offset + 16 <= actual.len() {
                let mut a = vld1q_u8(actual.as_ptr().add(offset));
                let mut b = vld1q_u8(expected.as_ptr().add(offset));
                if channels == 4 {
                    let equal = vceqq_u8(a, b);
                    if vminvq_u8(vorrq_u8(equal, vmvnq_u8(alpha_mask))) != 255 {
                        return false;
                    }
                    a = vbicq_u8(a, alpha_mask);
                    b = vbicq_u8(b, alpha_mask);
                }
                let low = vreinterpretq_s16_u16(vsubl_u8(vget_low_u8(a), vget_low_u8(b)));
                let high = vreinterpretq_s16_u16(vsubl_u8(vget_high_u8(a), vget_high_u8(b)));
                for diff in [low, high] {
                    error += vaddvq_s32(vmull_s16(vget_low_s16(diff), vget_low_s16(diff))) as u64;
                    error += vaddvq_s32(vmull_s16(vget_high_s16(diff), vget_high_s16(diff))) as u64;
                }
                if (error * if channels == 1 { 3 } else { 1 }) as f64 > limit {
                    return false;
                }
                offset += 16;
            }
        }
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if std::arch::is_x86_feature_detected!("sse2") {
        // SAFETY: CPU support is checked and the kernel bounds every load.
        match unsafe { within_error_8bit_sse2(actual, expected, channels, limit) } {
            Some(sum) => {
                error = sum;
                offset = actual.len() / 16 * 16;
            }
            None => return false,
        }
    }
    for (&a, &b) in actual[offset..].iter().zip(&expected[offset..]) {
        if channels == 4 && offset % 4 == 3 {
            if a != b {
                return false;
            }
        } else {
            let diff = i32::from(a) - i32::from(b);
            error += (diff * diff) as u64;
        }
        offset += 1;
    }
    (error * if channels == 1 { 3 } else { 1 }) as f64 <= limit
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn within_error_8bit_sse2(actual: &[u8], expected: &[u8], channels: usize, limit: f64) -> Option<u64> {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let mut error = 0_u64;
    let zero = _mm_setzero_si128();
    let alpha_mask = _mm_set1_epi32(0x00ff_ffff);
    for offset in (0..actual.len() / 16 * 16).step_by(16) {
        // SAFETY: the loop only visits complete 16-byte blocks.
        let mut a = unsafe { _mm_loadu_si128(actual.as_ptr().add(offset).cast()) };
        let mut b = unsafe { _mm_loadu_si128(expected.as_ptr().add(offset).cast()) };
        if channels == 4 {
            if _mm_movemask_epi8(_mm_cmpeq_epi8(a, b)) & 0x8888 != 0x8888 {
                return None;
            }
            a = _mm_and_si128(a, alpha_mask);
            b = _mm_and_si128(b, alpha_mask);
        }
        let low = _mm_sub_epi16(_mm_unpacklo_epi8(a, zero), _mm_unpacklo_epi8(b, zero));
        let high = _mm_sub_epi16(_mm_unpackhi_epi8(a, zero), _mm_unpackhi_epi8(b, zero));
        let sums = _mm_add_epi32(_mm_madd_epi16(low, low), _mm_madd_epi16(high, high));
        let mut lanes = [0_i32; 4];
        // SAFETY: the destination has room for one 128-bit vector.
        unsafe { _mm_storeu_si128(lanes.as_mut_ptr().cast(), sums) };
        error += lanes.into_iter().map(|lane| lane as u64).sum::<u64>();
        if (error * if channels == 1 { 3 } else { 1 }) as f64 > limit {
            return None;
        }
    }
    Some(error)
}

fn rgba_sample(image: &Image, data: &[u8], index: usize) -> [f64; 4] {
    let channels = image.mode.channels();
    let sample = |channel| {
        let offset = index * channels + channel;
        if image.bit_depth == 8 {
            f64::from(data[offset])
        } else {
            let offset = offset * 2;
            f64::from(u16::from_le_bytes([data[offset], data[offset + 1]])) * 255.0 / f64::from((1_u32 << image.bit_depth) - 1)
        }
    };
    match image.mode {
        PixelMode::One | PixelMode::L => [sample(0), sample(0), sample(0), 255.0],
        PixelMode::La | PixelMode::Pa => [sample(0), sample(0), sample(0), sample(1)],
        PixelMode::Rgb => [sample(0), sample(1), sample(2), 255.0],
        PixelMode::Rgba => [sample(0), sample(1), sample(2), sample(3)],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_bound_includes_hidden_color_but_excludes_alpha() {
        let reference = Image::from_pixels(1, 1, PixelMode::Rgba, vec![0, 0, 0, 0], None).unwrap();
        let changed = Image::from_pixels(1, 1, PixelMode::Rgba, vec![3, 3, 3, 0], None).unwrap();
        assert!(within_error(&changed, &reference, 3.0).unwrap());
        assert!(!within_error(&changed, &reference, 2.99).unwrap());
        let alpha = Image::from_pixels(1, 1, PixelMode::Rgba, vec![0, 0, 0, 1], None).unwrap();
        assert!(!within_error(&alpha, &reference, 255.0).unwrap());
    }

    #[test]
    fn compares_modes_and_normalizes_high_depth_samples() {
        let gray = Image::from_pixels(1, 1, PixelMode::L, vec![255], None).unwrap();
        let rgba = Image::from_samples(1, 1, PixelMode::Rgba, vec![1023; 4], 10, None).unwrap();
        assert!(within_error(&rgba, &gray, 0.0).unwrap());
        let dark = Image::from_samples(1, 1, PixelMode::L, vec![1019], 10, None).unwrap();
        assert!(within_error(&dark, &gray, 1.0).unwrap());
        assert!(!within_error(&dark, &gray, 0.99).unwrap());
    }

    #[test]
    fn eight_bit_error_matches_scalar_at_vector_boundaries() {
        for mode in [PixelMode::L, PixelMode::Rgb, PixelMode::Rgba] {
            let channels = mode.channels();
            for count in [1, 5, 6, 15, 16, 17, 31, 32, 33, 65] {
                let reference = (0..count * channels).map(|i| (i * 47 % 256) as u8).collect::<Vec<_>>();
                let mut candidate = reference.clone();
                for (i, value) in candidate.iter_mut().enumerate() {
                    if channels != 4 || i % 4 != 3 {
                        *value = value.wrapping_add((i % 7) as u8);
                    }
                }
                let expected_error: f64 = candidate
                    .chunks_exact(channels)
                    .zip(reference.chunks_exact(channels))
                    .map(|(actual, expected)| {
                        (0..if channels == 1 { 1 } else { 3 })
                            .map(|channel| (f64::from(actual[channel]) - f64::from(expected[channel])).powi(2))
                            .sum::<f64>()
                            * if channels == 1 { 3.0 } else { 1.0 }
                    })
                    .sum();
                let actual = Image::from_pixels(count as u32, 1, mode, candidate.clone(), None).unwrap();
                let expected = Image::from_pixels(count as u32, 1, mode, reference.clone(), None).unwrap();
                let boundary = (expected_error / (count * 3) as f64).sqrt();
                assert!(within_error(&actual, &expected, boundary + 1e-9).unwrap());
                if boundary > 0.0 {
                    assert!(!within_error(&actual, &expected, boundary * 0.99).unwrap());
                }
                if channels == 4 {
                    let alpha_index = 15.min(candidate.len() - 1) / 4 * 4 + 3;
                    candidate[alpha_index] ^= 1;
                    let changed = Image::from_pixels(count as u32, 1, mode, candidate, None).unwrap();
                    assert!(!within_error(&changed, &expected, 255.0).unwrap());
                }
            }
        }
    }

    #[test]
    fn png_centroids_minimize_bin_error() {
        let mut histogram = [0; 256];
        histogram[120] = 3;
        histogram[127] = 1;
        let table = png_centroid_table(&histogram, 256);
        assert_eq!(table[120], 122);
        assert_eq!(table[127], 122);
        assert_eq!(table[0], 0);
        assert!(png_rounding_within_error(&histogram, &table, 3, 37.0));
        assert!(!png_rounding_within_error(&histogram, &table, 3, 36.0));
        for step in 2..=256 {
            let table = png_centroid_table(&histogram, step);
            assert!(png_rounding_within_error(&histogram, &table, 3, 37.0));
        }
    }

    #[test]
    fn png_histogram_matches_rounded_pixel_error() {
        for channels in [1, 3, 4] {
            for count in [1, 15, 16, 17, 65] {
                let pixels = (0..count * channels).map(|i| (i * 53 % 256) as u8).collect::<Vec<_>>();
                let histogram = png_rounding_histogram(&pixels, channels);
                for step in [2_u16, 3, 6, 12, 64] {
                    let table = (0..256)
                        .map(|sample| ((sample as u16 + step / 2) / step * step).min(255) as u8)
                        .collect::<Vec<_>>();
                    let mut error = 0_u64;
                    for (index, &sample) in pixels.iter().enumerate() {
                        if channels != 4 || index % 4 != 3 {
                            let difference = i32::from(sample) - i32::from(table[usize::from(sample)]);
                            error += (difference * difference) as u64;
                        }
                    }
                    let error = error * if channels == 1 { 3 } else { 1 };
                    assert!(png_rounding_within_error(&histogram, &table, channels, error as f64));
                    if error > 0 {
                        assert!(!png_rounding_within_error(&histogram, &table, channels, error as f64 - 0.5));
                    }
                }
            }
        }
    }
}
