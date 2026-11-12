//! Shared byte and wide-sample image histograms.

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
    let mut bins = vec![0_u64; channels << 8];
    if samples.is_empty() {
        return Ok(bins);
    }
    let (low, high) = extrema.unwrap_or_else(|| {
        samples.iter().map(&sample).filter(|v| v.is_finite()).fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), v| (lo.min(v), hi.max(v)))
    });
    if !low.is_finite() || !high.is_finite() || high < low {
        return Err(PyValueError::new_err("invalid extrema"));
    }
    if high == low {
        return Ok(bins);
    }
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

#[cfg(test)]
mod tests {
    use super::*;

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
