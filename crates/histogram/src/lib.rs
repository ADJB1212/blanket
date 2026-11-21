//! Shared byte and wide-sample image histograms.
#![feature(portable_simd)]

use blanket_core::parallel::{CHUNK_PIXELS, MIN_PARALLEL_BYTES, should_parallel};
use blanket_core::{Image, PixelMode};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

pub fn validate_mask<'a>(mask: Option<&'a Image>, image: &Image) -> PyResult<Option<&'a [u8]>> {
    mask.map(|mask| {
        if !matches!(mask.pixel_mode(), PixelMode::One | PixelMode::L) || mask.bit_depth() != 8 {
            return Err(PyValueError::new_err("bad transparency mask"));
        }
        if (mask.width(), mask.height()) != (image.width(), image.height()) {
            return Err(PyValueError::new_err("images do not match"));
        }
        mask.pixel_data()
    })
    .transpose()
}

#[pyfunction(signature = (image, mask=None, extrema=None))]
pub fn wide_histogram(py: Python<'_>, image: &Image, mask: Option<&Image>, extrema: Option<(f64, f64)>) -> PyResult<Vec<u64>> {
    let raw = image.raw_data()?;
    let mask_data = validate_mask(mask, image)?;
    py.detach(|| match image.pixel_mode() {
        PixelMode::I => sample_histogram(raw, image.pixel_mode(), mask_data, extrema, |v| f64::from(i32::from_le_bytes(*v))),
        PixelMode::F | PixelMode::RgbF | PixelMode::RgbaF => {
            sample_histogram(raw, image.pixel_mode(), mask_data, extrema, |v| f64::from(f32::from_le_bytes(*v)))
        }
        mode if mode.is_big_endian() => sample_histogram(raw, image.pixel_mode(), mask_data, extrema, |v| f64::from(u16::from_be_bytes(*v))),
        _ => sample_histogram(raw, image.pixel_mode(), mask_data, extrema, |v| f64::from(u16::from_le_bytes(*v))),
    })
}

fn sample_histogram<const N: usize>(
    raw: &[u8], mode: PixelMode, mask: Option<&[u8]>, extrema: Option<(f64, f64)>, sample: impl Fn(&[u8; N]) -> f64,
) -> PyResult<Vec<u64>> {
    let channels = mode.channels();
    let samples = raw.as_chunks::<N>().0;
    if samples.is_empty() {
        return Ok(vec![0; channels << 8]);
    }
    let (low, high) = extrema.unwrap_or_else(|| {
        samples.iter().map(&sample).filter(|v| v.is_finite()).fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), v| (lo.min(v), hi.max(v)))
    });
    if !low.is_finite() || !high.is_finite() || high < low {
        return Err(PyValueError::new_err("invalid extrema"));
    }
    if high == low {
        return Ok(vec![0; channels << 8]);
    }
    if N == 2 && samples.len() >= 512 * 1024 {
        // Keep truncation and floating-point operation order identical to the per-sample path.
        let table: Vec<u16> = (0..=u16::MAX)
            .map(|value| {
                let slot = ((f64::from(value) - low) * 255.0 / (high - low)).trunc() as i64;
                if (0..256).contains(&slot) { slot as u16 } else { 256 }
            })
            .collect();
        let bins = match channels {
            1 => lookup_histogram::<1, N>(samples, mask, &table, &sample),
            2 => lookup_histogram::<2, N>(samples, mask, &table, &sample),
            3 => lookup_histogram::<3, N>(samples, mask, &table, &sample),
            4 => lookup_histogram::<4, N>(samples, mask, &table, &sample),
            _ => unreachable!("validated wide histogram channels"),
        };
        return Ok(bins);
    }
    let mut bins = vec![0_u64; channels << 8];
    let scale = match mode {
        PixelMode::I => 255.0 / (high as i32).wrapping_sub(low as i32) as f32,
        PixelMode::F | PixelMode::RgbF | PixelMode::RgbaF => {
            if high as f32 == low as f32 {
                return Ok(bins);
            }
            255.0 / (high as f32 - low as f32)
        }
        _ => 0.0,
    };
    for (pixel_index, pixel) in samples.chunks_exact(channels).enumerate() {
        if mask.is_some_and(|mask| mask[pixel_index] == 0) {
            continue;
        }
        for (channel, bytes) in pixel.iter().enumerate() {
            let value = sample(bytes);
            if value.is_finite() {
                let position = match mode {
                    PixelMode::I => f64::from((value as i32).wrapping_sub(low as i32) as f32 * scale),
                    PixelMode::F | PixelMode::RgbF | PixelMode::RgbaF => f64::from((value as f32 - low as f32) * scale),
                    _ => (value - low) * 255.0 / (high - low),
                };
                let slot = position.trunc() as i64;
                if (0..256).contains(&slot) {
                    bins[channel * 256 + slot as usize] += 1;
                }
            }
        }
    }
    Ok(bins)
}

fn lookup_histogram<const C: usize, const N: usize>(
    samples: &[[u8; N]], mask: Option<&[u8]>, table: &[u16], sample: impl Fn(&[u8; N]) -> f64,
) -> Vec<u64> {
    let mut bins = [[0; 257]; C];
    for (i, pixel) in samples.as_chunks::<C>().0.iter().enumerate() {
        if mask.is_none_or(|mask| mask[i] != 0) {
            for c in 0..C {
                bins[c][usize::from(table[sample(&pixel[c]) as usize])] += 1;
            }
        }
    }
    bins.into_iter().flat_map(|band| band.into_iter().take(256)).collect()
}

#[pyfunction(signature = (image, mask=None))]
pub fn ops_histogram(py: Python<'_>, image: &Image, mask: Option<&Image>) -> PyResult<Vec<u64>> {
    let source = image.pixel_data()?;
    let mask = validate_mask(mask, image)?;
    Ok(py.detach(|| match image.pixel_mode() {
        PixelMode::One | PixelMode::L => histogram::<1>(source, mask),
        PixelMode::La | PixelMode::PremulLa | PixelMode::Pa => histogram::<2>(source, mask),
        PixelMode::Rgb | PixelMode::Hsv | PixelMode::YCbCr | PixelMode::Lab => histogram::<3>(source, mask),
        PixelMode::Rgba | PixelMode::PremulRgba | PixelMode::Rgbx | PixelMode::Cmyk => histogram::<4>(source, mask),
        _ => unreachable!(),
    }))
}

fn histogram<const C: usize>(source: &[u8], mask: Option<&[u8]>) -> Vec<u64> {
    let count = |start: usize, bytes: &[u8]| {
        let mut bins = [[0_u64; 256]; C];
        let pixels = bytes.as_chunks::<C>().0;
        // Hoist the mask decision out of the per-pixel loop.
        match mask {
            None => {
                if let Some(pixel) = blanket_kernels::ops::constant_pixel::<C>(bytes) {
                    for c in 0..C {
                        bins[c][usize::from(pixel[c])] = pixels.len() as u64;
                    }
                    return bins;
                }
                if C == 1 && has_repeated_values(bytes) {
                    bins[0] = grayscale_histogram(bytes);
                    return bins;
                }
                for pixel in pixels {
                    for c in 0..C {
                        bins[c][usize::from(pixel[c])] += 1;
                    }
                }
            }
            Some(mask) => {
                for (pixel, &selected) in pixels.iter().zip(&mask[start..]) {
                    if selected != 0 {
                        for c in 0..C {
                            bins[c][usize::from(pixel[c])] += 1;
                        }
                    }
                }
            }
        }
        bins
    };
    let bins = if should_parallel(source.len(), CHUNK_PIXELS * C, MIN_PARALLEL_BYTES) {
        source.par_chunks(CHUNK_PIXELS * C).enumerate().map(|(i, bytes)| count(i * CHUNK_PIXELS, bytes)).reduce(
            || [[0; 256]; C],
            |mut total, bins| {
                for (a, b) in total.as_flattened_mut().iter_mut().zip(bins.as_flattened()) {
                    *a += b;
                }
                total
            },
        )
    } else {
        count(0, source)
    };
    bins.into_iter().flatten().collect()
}

fn has_repeated_values(source: &[u8]) -> bool {
    use std::simd::cmp::SimdPartialEq;

    use blanket_core::pixels::Bytes;

    source.len() >= 4096 && Bytes::from_slice(source).simd_eq(Bytes::from_slice(&source[1..])).to_bitmask().count_ones() >= 8
}

fn grayscale_histogram(source: &[u8]) -> [u64; 256] {
    // Independent accumulators let repeated values update without serializing every increment.
    let [mut first, mut second, mut third, mut fourth] = [[0_u64; 256]; 4];
    let (blocks, tail) = source.as_chunks::<4>();
    for &[a, b, c, d] in blocks {
        first[usize::from(a)] += 1;
        second[usize::from(b)] += 1;
        third[usize::from(c)] += 1;
        fourth[usize::from(d)] += 1;
    }
    for &value in tail {
        first[usize::from(value)] += 1;
    }
    for (i, count) in first.iter_mut().enumerate() {
        *count += second[i] + third[i] + fourth[i];
    }
    first
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wide_lookup_matches_scalar_for_masks_and_fractional_extrema() {
        let values: Vec<_> = (0..180_001_u32).flat_map(|i| [i as u16, (i * 19) as u16, (i * 247) as u16]).collect();
        let mask: Vec<_> = (0..180_001).map(|i| u8::from(i % 3 != 0)).collect();
        let (low, high) = (1000.25, 60_000.75);
        for mask in [None, Some(mask.as_slice())] {
            let mut expected = vec![0; 768];
            for (i, pixel) in values.as_chunks::<3>().0.iter().enumerate() {
                if mask.is_none_or(|mask| mask[i] != 0) {
                    for (c, &value) in pixel.iter().enumerate() {
                        let slot = ((f64::from(value) - low) * 255.0 / (high - low)).trunc() as i64;
                        if (0..256).contains(&slot) {
                            expected[c * 256 + slot as usize] += 1;
                        }
                    }
                }
            }
            let raw: Vec<_> = values.iter().flat_map(|value| value.to_le_bytes()).collect();
            assert_eq!(
                sample_histogram(&raw, PixelMode::Rgb, mask, Some((low, high)), |bytes| f64::from(u16::from_le_bytes(*bytes))).unwrap(),
                expected,
            );
        }
    }

    #[test]
    fn color_float_histogram_decodes_samples_and_applies_pixel_masks() {
        Python::initialize();
        Python::attach(|py| {
            for mode in [PixelMode::RgbF, PixelMode::RgbaF] {
                let channels = mode.channels();
                let values = [[0.0_f32, 0.25, 0.5, 0.75], [1.0, 0.75, 0.5, 0.25]];
                let pixels = values.iter().flat_map(|pixel| pixel[..channels].iter().flat_map(|v| v.to_le_bytes())).collect();
                let image = Image::from_color_float_bytes(2, 1, mode, pixels).unwrap();
                let mask = Image::from_pixels(2, 1, PixelMode::L, vec![0, 255], None).unwrap();
                for extrema in [None, Some((0.0, 1.0))] {
                    for masked in [false, true] {
                        let actual = wide_histogram(py, &image, masked.then_some(&mask), extrema).unwrap();
                        let mut expected = vec![0; channels * 256];
                        for pixel in &values[usize::from(masked)..] {
                            for (channel, &value) in pixel[..channels].iter().enumerate() {
                                expected[channel * 256 + (value * 255.0) as usize] += 1;
                            }
                        }
                        assert_eq!(actual, expected);
                    }
                }
            }
        });
    }

    #[test]
    fn histogram_matches_scalar_for_masks_repeated_bins_and_chunk_tails() {
        fn check<const C: usize>() {
            for count in [0, 1, 2, 3, 15, 16, 17, 65535, 65536, 65537, 262145] {
                for repeated in [false, true] {
                    let mut source = vec![93];
                    source.extend((0..count * C).map(|i| if repeated { 71 } else { (i * 137 + i / C) as u8 }));
                    source.push(97);
                    let pixels = &source[1..source.len() - 1];
                    let mask: Vec<_> = (0..count).map(|i| if i % 3 == 0 { 0 } else { 255 }).collect();
                    for mask in [None, Some(mask.as_slice())] {
                        let mut expected = vec![0_u64; C << 8];
                        for (i, pixel) in pixels.as_chunks::<C>().0.iter().enumerate() {
                            if mask.is_none_or(|mask| mask[i] != 0) {
                                for c in 0..C {
                                    expected[c * 256 + usize::from(pixel[c])] += 1;
                                }
                            }
                        }
                        assert_eq!(histogram::<C>(pixels, mask), expected);
                    }
                    assert_eq!(source[0], 93);
                    assert_eq!(source[source.len() - 1], 97);
                }
            }
        }
        for threads in [1, 4] {
            rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap().install(|| {
                check::<1>();
                check::<2>();
                check::<3>();
                check::<4>();
            });
        }
    }
}
