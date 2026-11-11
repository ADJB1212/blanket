//! Full-reference perceptual quality scores for normalized SDR samples.

use blanket_core::{Image, PixelMode};
use pyo3::exceptions::PyValueError;
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

fn statistics(a: &[[f64; 3]], b: &[[f64; 3]], width: usize, channel: usize, weights: &[f64; WINDOW]) -> (f64, f64) {
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

fn downsample(pixels: &[[f64; 3]], width: usize) -> Vec<[f64; 3]> {
    let height = pixels.len() / width;
    let mut output = Vec::with_capacity(width.div_ceil(2) * height.div_ceil(2));
    for y in (0..height).step_by(2) {
        for x in (0..width).step_by(2) {
            let x1 = (x + 1).min(width - 1);
            let y1 = (y + 1).min(height - 1);
            output.push(std::array::from_fn(|c| {
                (pixels[y * width + x][c] + pixels[y * width + x1][c] + pixels[y1 * width + x][c] + pixels[y1 * width + x1][c]) * 0.25
            }));
        }
    }
    output
}

fn structural(mut a: Vec<[f64; 3]>, mut b: Vec<[f64; 3]>, mut width: usize, algorithm: &str) -> f64 {
    let weights = gaussian();
    let mut scores = [1.0; 3];
    let levels = if algorithm == "ssim" { 1 } else { WEIGHTS.len() };
    for (level, &weight) in WEIGHTS.iter().enumerate().take(levels) {
        for (channel, score) in scores.iter_mut().enumerate() {
            let (ssim, cs) = statistics(&a, &b, width, channel, &weights);
            if levels == 1 {
                *score = ssim;
            } else {
                *score *= (if level + 1 == levels { ssim } else { cs }).max(0.0).powf(weight);
            }
        }
        if level + 1 < levels {
            a = downsample(&a, width);
            b = downsample(&b, width);
            width = width.div_ceil(2);
        }
    }
    scores.iter().sum::<f64>() / 3.0
}

#[pyfunction]
fn image_metric(py: Python<'_>, reference: &Image, distorted: &Image, algorithm: &str) -> PyResult<f64> {
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
    py.detach(|| {
        let a = normalized(reference, a)?;
        let b = normalized(distorted, b)?;
        if algorithm != "ssimulacra2" {
            return Ok(structural(a, b, width, algorithm));
        }
        let rgb = |pixels: Vec<[f64; 3]>| {
            Rgb::new(pixels.into_iter().map(|v| v.map(|c| c as f32)).collect(), width, height, TransferCharacteristic::SRGB, ColorPrimaries::BT709)
                .map_err(|err| PyValueError::new_err(err.to_string()))
        };
        compute_frame_ssimulacra2(rgb(a)?, rgb(b)?).map_err(|err| PyValueError::new_err(err.to_string()))
    })
}
