//! Save-time size reduction with a decoded-sample error bound.

use pyo3::exceptions::{PyOSError, PyValueError};
use pyo3::prelude::*;

use crate::codecs::{self, ImageFormat, SaveOptions};
use crate::compressor::LosslessImageCompressor;
use crate::raster::{Image, PixelMode};

#[pyclass(name = "_LossyImageCompressor", module = "blanket._blanket", frozen, from_py_object)]
#[derive(Clone, Copy)]
pub(crate) struct LossyImageCompressor {
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
    pub(crate) fn encode(&self, image: &Image, format: ImageFormat, options: SaveOptions) -> PyResult<Vec<u8>> {
        let lossless = LosslessImageCompressor { effort: self.effort };
        let mut output = lossless.encode(image, format, options)?;
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
        let reference = codecs::decode(&output, format).map_err(PyOSError::new_err)?;
        const STEPS: [u16; 10] = [2, 3, 4, 6, 8, 12, 16, 24, 32, 64];
        const QUALITY_DROPS: [u8; 10] = [1, 2, 4, 8, 12, 20, 30, 45, 65, 99];
        let mut previous_quality = options.quality;
        for trial in 0..usize::from(self.effort) {
            let candidate = if png {
                let step = STEPS[trial];
                let mut rounded = image.clone();
                let channels = image.mode.channels();
                for (index, sample) in rounded.pixels.as_mut().expect("validated image").iter_mut().enumerate() {
                    if channels != 4 || index % 4 != 3 {
                        *sample = ((u16::from(*sample) + step / 2) / step * step).min(255) as u8;
                    }
                }
                if !within_error(&rounded, &reference, self.max_rmse)? {
                    continue;
                }
                lossless.encode(&rounded, format, options)?
            } else {
                let quality = options.quality.saturating_sub(QUALITY_DROPS[trial]).max(1);
                if quality == previous_quality {
                    continue;
                }
                previous_quality = quality;
                codecs::encode(image, format, SaveOptions { quality, ..options })?
            };
            if candidate.len() < output.len() {
                let decoded = codecs::decode(&candidate, format).map_err(PyOSError::new_err)?;
                if within_error(&decoded, &reference, self.max_rmse)? {
                    output = candidate;
                }
            }
        }
        Ok(output)
    }
}

fn within_error(candidate: &Image, reference: &Image, max_rmse: f64) -> PyResult<bool> {
    if (candidate.width, candidate.height) != (reference.width, reference.height) {
        return Ok(false);
    }
    let candidate_data = candidate.raw_data()?;
    let reference_data = reference.raw_data()?;
    let count = candidate.width as usize * candidate.height as usize;
    let limit = max_rmse * max_rmse * count as f64 * 3.0;
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
        PixelMode::L => [sample(0), sample(0), sample(0), 255.0],
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
}
