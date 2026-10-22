use std::path::PathBuf;

use blanket_codecs::codecs::{self, ImageFormat, SaveOptions};
use blanket_compressor::Compressor;
use blanket_core::{Image, PixelMode};
use pyo3::exceptions::{PyOSError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

#[pyfunction(name = "_encode")]
#[pyo3(signature = (image, format, quality, compress_level, lossless, effort, compressor=None, tiff_compression=0, output=None))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn _encode(
    py: Python<'_>, image: &Image, format: &str, quality: u8, compress_level: u8, lossless: bool, effort: u8, compressor: Option<Compressor>,
    tiff_compression: u8, output: Option<Bound<'_, PyAny>>,
) -> PyResult<Option<Py<PyBytes>>> {
    let format = ImageFormat::parse(format)?;
    let options = SaveOptions {
        quality,
        compress_level,
        lossless,
        effort,
        tiff_compression,
    };
    let encoded = encode_image(py, image, format, options, compressor)?;
    if let Some(output) = output {
        let path = if output.is_instance_of::<PyBytes>() {
            py.import("os")?.call_method1("fsdecode", (&output,))?.extract::<PathBuf>()?
        } else {
            output.extract::<PathBuf>()?
        };
        if path.as_os_str().as_encoded_bytes().contains(&0) {
            return Err(PyValueError::new_err("embedded null byte"));
        }
        py.detach(move || std::fs::write(path, encoded).map_err(|error| (error.raw_os_error(), error.to_string())))
            .map_err(|(errno, message)| match errno {
                Some(errno) => PyOSError::new_err((errno, message, output.unbind())),
                None => PyOSError::new_err(message),
            })?;
        return Ok(None);
    }
    Ok(Some(PyBytes::new(py, &encoded).unbind()))
}

fn encode_image(py: Python<'_>, image: &Image, format: ImageFormat, options: SaveOptions, compressor: Option<Compressor>) -> PyResult<Vec<u8>> {
    let compress_level = options.compress_level;
    if image.mode == PixelMode::One && format == ImageFormat::Png {
        return py.detach(|| codecs::encode_one_png(image, compress_level));
    }
    if matches!(image.mode, PixelMode::One | PixelMode::La | PixelMode::Pa)
        && !matches!(
            (image.mode, format),
            (
                PixelMode::One,
                ImageFormat::Ppm
                    | ImageFormat::Pam
                    | ImageFormat::Jpeg2000
                    | ImageFormat::Pcx
                    | ImageFormat::Xbm
                    | ImageFormat::Msp
                    | ImageFormat::Palm
            ) | (PixelMode::La, ImageFormat::Pam | ImageFormat::Tga | ImageFormat::Jpeg2000)
        )
    {
        let target = if image.mode == PixelMode::One {
            "L"
        } else if format == ImageFormat::Png && image.mode == PixelMode::La && compressor.is_none() {
            "LA"
        } else {
            "RGBA"
        };
        if target != "LA" {
            let expanded = image.convert(py, target, None)?;
            return py.detach(|| match compressor {
                Some(compressor) => compressor.encode(&expanded, format, options),
                None => codecs::encode(&expanded, format, options),
            });
        }
    }
    if let Some((palette_mode, _)) = image.palette {
        if matches!(format, ImageFormat::Pcx | ImageFormat::Palm) && palette_mode == PixelMode::Rgb {
            return py.detach(|| codecs::encode(image, format, options));
        }
        if format == ImageFormat::Png {
            return py.detach(|| codecs::encode_palette_png(image, compress_level));
        }
        if format == ImageFormat::Jpeg {
            return Err(PyOSError::new_err("cannot write mode P as JPEG"));
        }
        let mode = palette_mode.as_str();
        let expanded = image.convert(py, mode, None)?;
        return py.detach(|| match compressor {
            Some(compressor) => compressor.encode(&expanded, format, options),
            None => codecs::encode(&expanded, format, options),
        });
    }
    py.detach(|| match compressor {
        Some(compressor) => compressor.encode(image, format, options),
        None => codecs::encode(image, format, options),
    })
}
