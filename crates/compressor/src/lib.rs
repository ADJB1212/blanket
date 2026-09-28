pub mod compressor;
pub mod compressor_simd;
pub mod lossy_compressor;

pub use compressor::Compressor;

use blanket_codecs::codecs::{self, ImageFormat, SaveOptions};
use blanket_core::{Image, PixelMode};
use pyo3::exceptions::PyOSError;
use pyo3::prelude::*;

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    compressor::register(module)
}

#[pyfunction(name = "_encode")]
#[pyo3(signature = (image, format, quality, compress_level, lossless, effort, compressor=None))]
#[allow(clippy::too_many_arguments)]
pub fn _encode(
    py: Python<'_>, image: &Image, format: &str, quality: u8, compress_level: u8, lossless: bool, effort: u8, compressor: Option<Compressor>,
) -> PyResult<Py<pyo3::types::PyBytes>> {
    let format = ImageFormat::parse(format)?;
    let options = SaveOptions {
        quality,
        compress_level,
        lossless,
        effort,
    };
    if image.mode == PixelMode::One && format == ImageFormat::Png {
        let encoded = py.detach(|| codecs::encode_one_png(image, compress_level))?;
        return Ok(pyo3::types::PyBytes::new(py, &encoded).unbind());
    }
    if matches!(image.mode, PixelMode::One | PixelMode::La | PixelMode::Pa) {
        let target = if image.mode == PixelMode::One {
            "L"
        } else if format == ImageFormat::Png && image.mode == PixelMode::La && compressor.is_none() {
            "LA"
        } else {
            "RGBA"
        };
        if target != "LA" {
            let expanded = image.convert(py, target, None)?;
            let encoded = py.detach(|| match compressor {
                Some(compressor) => compressor.encode(&expanded, format, options),
                None => codecs::encode(&expanded, format, options),
            })?;
            return Ok(pyo3::types::PyBytes::new(py, &encoded).unbind());
        }
    }
    if let Some((palette_mode, _)) = image.palette {
        if format == ImageFormat::Png {
            let encoded = py.detach(|| codecs::encode_palette_png(image, compress_level))?;
            return Ok(pyo3::types::PyBytes::new(py, &encoded).unbind());
        }
        if format == ImageFormat::Jpeg {
            return Err(PyOSError::new_err("cannot write mode P as JPEG"));
        }
        let mode = palette_mode.as_str();
        let expanded = image.convert(py, mode, None)?;
        let encoded = py.detach(|| match compressor {
            Some(compressor) => compressor.encode(&expanded, format, options),
            None => codecs::encode(&expanded, format, options),
        })?;
        return Ok(pyo3::types::PyBytes::new(py, &encoded).unbind());
    }
    let encoded = py.detach(|| match compressor {
        Some(compressor) => compressor.encode(image, format, options),
        None => codecs::encode(image, format, options),
    })?;
    Ok(pyo3::types::PyBytes::new(py, &encoded).unbind())
}
