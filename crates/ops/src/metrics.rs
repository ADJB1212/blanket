//! Full-reference perceptual quality scores for normalized SDR samples.

use std::simd::Simd;
use std::simd::num::SimdFloat;

use blanket_core::parallel::with_rayon_worker_limit;
use blanket_core::{Image, PixelMode};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use ssimulacra2::{ColorPrimaries, Rgb, TransferCharacteristic, compute_frame_ssimulacra2};

const WINDOW: usize = 11;
const C1: f64 = 0.01 * 0.01;
const C2: f64 = 0.03 * 0.03;
// Wang et al., "Multiscale structural similarity for image quality assessment".
const WEIGHTS: [f64; 5] = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333];

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(image_metric, module)?)?;
    Ok(())
}

fn validate_mode(image: &Image) -> PyResult<()> {
    if image.palette().is_some()
        || !matches!(
            image.pixel_mode(),
            PixelMode::L
                | PixelMode::La
                | PixelMode::Rgb
                | PixelMode::Rgba
                | PixelMode::F
                | PixelMode::RgbF
                | PixelMode::RgbaF
                | PixelMode::I16
                | PixelMode::I16L
                | PixelMode::I16B
                | PixelMode::I16N
        )
    {
        return Err(PyValueError::new_err("metrics require L, LA, RGB, RGBA, F, RGB;F, RGBA;F, or I;16 pixels; convert explicitly"));
    }
    Ok(())
}

fn normalized(image: &Image, data: &[u8]) -> PyResult<Vec<[f64; 3]>> {
    let mode = image.pixel_mode();
    let sample_bytes = if mode.is_float() { 4 } else { usize::from(image.bit_depth().div_ceil(8)) };
    let channels = mode.channels();
    let colors = if channels < 3 { 1 } else { 3 };
    let maximum = if mode.is_float() { 1.0 } else { ((1_u32 << image.bit_depth()) - 1) as f64 };
    data.chunks_exact(channels * sample_bytes)
        .map(|pixel| {
            let mut rgb = [0.0; 3];
            for (c, value) in rgb.iter_mut().enumerate().take(colors) {
                let sample = &pixel[c * sample_bytes..(c + 1) * sample_bytes];
                *value = if mode.is_float() {
                    f64::from(f32::from_le_bytes(sample.try_into().unwrap()))
                } else if sample_bytes == 1 {
                    f64::from(sample[0]) / maximum
                } else if mode.is_big_endian() {
                    f64::from(u16::from_be_bytes(sample.try_into().unwrap())) / maximum
                } else {
                    f64::from(u16::from_le_bytes(sample.try_into().unwrap())) / maximum
                };
                if !value.is_finite() || !(0.0..=1.0).contains(value) {
                    return Err(PyValueError::new_err("metric color samples must be finite and in [0, 1] after depth normalization"));
                }
            }
            if colors == 1 {
                rgb = [rgb[0]; 3];
            }
            Ok(rgb)
        })
        .collect()
}

fn gaussian() -> [f64; WINDOW] {
    let mut weights = std::array::from_fn(|i| (-((i as f64 - 5.0).powi(2)) / (2.0 * 1.5_f64.powi(2))).exp());
    let sum: f64 = weights.iter().sum();
    for value in &mut weights {
        *value /= sum;
    }
    weights
}

const LANES: usize = 4;
type Vector = Simd<f64, LANES>;

fn plane(pixels: &[[f64; 3]], channel: usize) -> Vec<f64> {
    pixels.iter().map(|pixel| pixel[channel]).collect()
}

fn statistics(a: &[f64], b: &[f64], width: usize, weights: &[f64; WINDOW]) -> (f64, f64) {
    let height = a.len() / width;
    let output_width = width - WINDOW + 1;
    let output_height = height - WINDOW + 1;
    let mut rows: [Vec<f64>; 5] = std::array::from_fn(|_| vec![0.0; output_width * height]);
    for y in 0..height {
        let (a_row, b_row) = (&a[y * width..(y + 1) * width], &b[y * width..(y + 1) * width]);
        let base = y * output_width;
        let mut x = 0;
        while x + LANES <= output_width {
            let mut moments = [Vector::splat(0.0); 5];
            for (k, &weight) in weights.iter().enumerate() {
                let weight = Vector::splat(weight);
                let av = Vector::from_slice(&a_row[x + k..]);
                let bv = Vector::from_slice(&b_row[x + k..]);
                for (sum, value) in moments.iter_mut().zip([av, bv, av * av, bv * bv, av * bv]) {
                    *sum += weight * value;
                }
            }
            for (row, sum) in rows.iter_mut().zip(moments) {
                row[base + x..base + x + LANES].copy_from_slice(&sum.to_array());
            }
            x += LANES;
        }
        for x in x..output_width {
            let mut moments = [0.0; 5];
            for (k, &weight) in weights.iter().enumerate() {
                let (av, bv) = (a_row[x + k], b_row[x + k]);
                for (sum, value) in moments.iter_mut().zip([av, bv, av * av, bv * bv, av * bv]) {
                    *sum += weight * value;
                }
            }
            for (row, sum) in rows.iter_mut().zip(moments) {
                row[base + x] = sum;
            }
        }
    }
    let mut ssim = 0.0;
    let mut cs = 0.0;
    let (c1, c2) = (Vector::splat(C1), Vector::splat(C2));
    let (zero, two) = (Vector::splat(0.0), Vector::splat(2.0));
    for y in 0..output_height {
        let mut x = 0;
        while x + LANES <= output_width {
            let mut moments = [zero; 5];
            for (k, &weight) in weights.iter().enumerate() {
                let weight = Vector::splat(weight);
                let at = (y + k) * output_width + x;
                for (sum, row) in moments.iter_mut().zip(&rows) {
                    *sum += weight * Vector::from_slice(&row[at..]);
                }
            }
            let [ma, mb, aa, bb, ab] = moments;
            let va = (aa - ma * ma).simd_max(zero);
            let vb = (bb - mb * mb).simd_max(zero);
            let covariance = ab - ma * mb;
            let contrast = (two * covariance + c2) / (va + vb + c2);
            let structure = (two * ma * mb + c1) / (ma * ma + mb * mb + c1) * contrast;
            // Lanes are summed in pixel order to match the scalar accumulation.
            for (contrast, structure) in contrast.to_array().into_iter().zip(structure.to_array()) {
                cs += contrast;
                ssim += structure;
            }
            x += LANES;
        }
        for x in x..output_width {
            let mut moments = [0.0; 5];
            for (k, &weight) in weights.iter().enumerate() {
                for (sum, row) in moments.iter_mut().zip(&rows) {
                    *sum += weight * row[(y + k) * output_width + x];
                }
            }
            let [ma, mb, aa, bb, ab] = moments;
            let va = (aa - ma * ma).max(0.0);
            let vb = (bb - mb * mb).max(0.0);
            let covariance = ab - ma * mb;
            let contrast = (2.0 * covariance + C2) / (va + vb + C2);
            cs += contrast;
            ssim += (2.0 * ma * mb + C1) / (ma * ma + mb * mb + C1) * contrast;
        }
    }
    let count = (output_width * output_height) as f64;
    ((ssim / count).clamp(-1.0, 1.0), (cs / count).clamp(-1.0, 1.0))
}

fn downsample(pixels: &[f64], width: usize) -> Vec<f64> {
    let height = pixels.len() / width;
    let mut output = Vec::with_capacity(width.div_ceil(2) * height.div_ceil(2));
    let pairs = width / 2;
    for y in (0..height).step_by(2) {
        let upper = &pixels[y * width..(y + 1) * width];
        let lower = &pixels[(y + 1).min(height - 1) * width..][..width];
        let mut x = 0;
        while x + 16 <= 2 * pairs {
            let (top_even, top_odd) = Simd::<f64, 8>::from_slice(&upper[x..]).deinterleave(Simd::<f64, 8>::from_slice(&upper[x + 8..]));
            let (low_even, low_odd) = Simd::<f64, 8>::from_slice(&lower[x..]).deinterleave(Simd::<f64, 8>::from_slice(&lower[x + 8..]));
            output.extend_from_slice(&((((top_even + top_odd) + low_even) + low_odd) * Simd::splat(0.25)).to_array());
            x += 16;
        }
        for x in (x..width).step_by(2) {
            let x1 = (x + 1).min(width - 1);
            output.push((upper[x] + upper[x1] + lower[x] + lower[x1]) * 0.25);
        }
    }
    output
}

fn structural(a: Vec<[f64; 3]>, b: Vec<[f64; 3]>, mut width: usize, algorithm: &str) -> f64 {
    let planes = |pixels: Vec<[f64; 3]>| std::array::from_fn::<_, 3, _>(|channel| plane(&pixels, channel));
    let (mut a, mut b) = (planes(a), planes(b));
    let weights = gaussian();
    let mut scores = [1.0; 3];
    let levels = if algorithm == "ssim" { 1 } else { WEIGHTS.len() };
    for (level, &weight) in WEIGHTS.iter().enumerate().take(levels) {
        for (channel, score) in scores.iter_mut().enumerate() {
            let (ssim, cs) = statistics(&a[channel], &b[channel], width, &weights);
            if levels == 1 {
                *score = ssim;
            } else {
                *score *= (if level + 1 == levels { ssim } else { cs }).max(0.0).powf(weight);
            }
        }
        if level + 1 < levels {
            a = a.map(|plane| downsample(&plane, width));
            b = b.map(|plane| downsample(&plane, width));
            width = width.div_ceil(2);
        }
    }
    scores.iter().sum::<f64>() / 3.0
}

#[pyfunction]
fn image_metric(py: Python<'_>, reference: &Image, distorted: &Image, algorithm: &str) -> PyResult<f64> {
    py.detach(|| score(reference, distorted, algorithm))
}

/// Score normalized SDR images using the same contract as ImageMetrics.
pub fn score(reference: &Image, distorted: &Image, algorithm: &str) -> PyResult<f64> {
    let minimum = match algorithm {
        "ssim" => WINDOW,
        "ms_ssim" => (WINDOW - 1) * 16 + 1,
        "ssimulacra2" => 8,
        _ => return Err(PyValueError::new_err("unknown perceptual metric")),
    };
    validate_mode(reference)?;
    validate_mode(distorted)?;
    let a = reference.raw_data()?;
    let b = distorted.raw_data()?;
    if reference.size() != distorted.size() {
        return Err(PyValueError::new_err("metric image dimensions must match"));
    }
    let width = reference.width() as usize;
    let height = reference.height() as usize;
    if width < minimum || height < minimum {
        return Err(PyValueError::new_err(format!("{algorithm} requires images at least {minimum}x{minimum}")));
    }
    let a = normalized(reference, a)?;
    let b = normalized(distorted, b)?;
    if algorithm != "ssimulacra2" {
        return Ok(structural(a, b, width, algorithm));
    }
    let rgb = |pixels: Vec<[f64; 3]>| {
        Rgb::new(pixels.into_iter().map(|v| v.map(|c| c as f32)).collect(), width, height, TransferCharacteristic::SRGB, ColorPrimaries::BT709)
            .map_err(|err| PyValueError::new_err(err.to_string()))
    };
    let (a, b) = (rgb(a)?, rgb(b)?);
    with_rayon_worker_limit(|| compute_frame_ssimulacra2(a, b))
        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?
        .map_err(|err| PyValueError::new_err(err.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar_statistics(a: &[[f64; 3]], b: &[[f64; 3]], width: usize, channel: usize, weights: &[f64; WINDOW]) -> (f64, f64) {
        let height = a.len() / width;
        let output_width = width - WINDOW + 1;
        let mut rows = vec![[0.0; 5]; output_width * height];
        for (y, row) in rows.chunks_exact_mut(output_width).enumerate() {
            for (x, moments) in row.iter_mut().enumerate() {
                for (k, &weight) in weights.iter().enumerate() {
                    let av = a[y * width + x + k][channel];
                    let bv = b[y * width + x + k][channel];
                    for (sum, value) in moments.iter_mut().zip([av, bv, av * av, bv * bv, av * bv]) {
                        *sum += weight * value;
                    }
                }
            }
        }
        let mut ssim = 0.0;
        let mut cs = 0.0;
        for y in 0..height - WINDOW + 1 {
            for x in 0..output_width {
                let mut moments = [0.0; 5];
                for (k, &weight) in weights.iter().enumerate() {
                    for (sum, value) in moments.iter_mut().zip(rows[(y + k) * output_width + x]) {
                        *sum += weight * value;
                    }
                }
                let [ma, mb, aa, bb, ab] = moments;
                let va = (aa - ma * ma).max(0.0);
                let vb = (bb - mb * mb).max(0.0);
                let covariance = ab - ma * mb;
                let contrast = (2.0 * covariance + C2) / (va + vb + C2);
                cs += contrast;
                ssim += (2.0 * ma * mb + C1) / (ma * ma + mb * mb + C1) * contrast;
            }
        }
        let count = (output_width * (height - WINDOW + 1)) as f64;
        ((ssim / count).clamp(-1.0, 1.0), (cs / count).clamp(-1.0, 1.0))
    }

    #[test]
    fn vector_statistics_match_scalar() {
        let weights = gaussian();
        for (width, height) in [(11, 11), (12, 13), (15, 11), (20, 17), (31, 12)] {
            let make = |seed: u64| -> Vec<[f64; 3]> {
                (0..width * height).map(|i| std::array::from_fn(|c| ((i as u64 * 2654435761 + c as u64 * 97 + seed) % 1000) as f64 / 999.0)).collect()
            };
            let (a, b) = (make(1), make(5));
            for channel in 0..3 {
                assert_eq!(
                    statistics(&plane(&a, channel), &plane(&b, channel), width, &weights),
                    scalar_statistics(&a, &b, width, channel, &weights)
                );
            }
        }
    }

    #[test]
    fn vector_downsample_matches_scalar() {
        for (width, height) in [(1, 1), (2, 2), (7, 5), (8, 4), (16, 3), (17, 6), (40, 9)] {
            let pixels: Vec<f64> = (0..width * height).map(|i| ((i * 7919) % 1000) as f64 / 999.0).collect();
            let mut expected = Vec::new();
            for y in (0..height).step_by(2) {
                for x in (0..width).step_by(2) {
                    let (x1, y1) = ((x + 1).min(width - 1), (y + 1).min(height - 1));
                    expected.push((pixels[y * width + x] + pixels[y * width + x1] + pixels[y1 * width + x] + pixels[y1 * width + x1]) * 0.25);
                }
            }
            assert_eq!(downsample(&pixels, width), expected, "{width}x{height}");
        }
    }
}
