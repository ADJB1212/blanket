use std::io::{BufWriter, Write};

use blanket_codecs::codecs::{self, ImageFormat, SaveOptions, output_error, output_path, write_path};
use blanket_compressor::Compressor;
use blanket_core::{Image, PixelMode};
use pyo3::exceptions::{PyOSError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

#[pyfunction(name = "_encode")]
#[pyo3(signature = (image, format, quality, compress_level, lossless, effort, compressor=None, tiff_compression=0, output=None, icc_profile=None, metadata_exif=None, metadata_xmp=None, attach_metadata=false, metadata_chunks=Vec::new(), jpeg_subsampling=-1, jpeg_optimize=false, jpeg_progressive=false, jpeg_keep_rgb=false, jpeg_qtables=Vec::new(), jpeg_smooth=0, jpeg_qtable_quality=-1, jpeg_restart_marker_blocks=0, jpeg_restart_marker_rows=0, jpeg_streamtype=0, target_size=None, pdf_dpi=72.0))]
pub(crate) fn _encode(
    py: Python<'_>, image: &Image, format: &str, quality: u8, compress_level: u8, lossless: bool, effort: u8, compressor: Option<Compressor>,
    tiff_compression: u8, output: Option<Bound<'_, PyAny>>, icc_profile: Option<&[u8]>, metadata_exif: Option<&[u8]>, metadata_xmp: Option<&[u8]>,
    attach_metadata: bool, metadata_chunks: Vec<blanket_codecs::metadata::PngChunk>, jpeg_subsampling: i8, jpeg_optimize: bool,
    jpeg_progressive: bool, jpeg_keep_rgb: bool, jpeg_qtables: Vec<[u16; 64]>, jpeg_smooth: i32, jpeg_qtable_quality: i16,
    jpeg_restart_marker_blocks: u32, jpeg_restart_marker_rows: u32, jpeg_streamtype: i32, target_size: Option<u64>, pdf_dpi: f64,
) -> PyResult<Option<Py<PyBytes>>> {
    if target_size == Some(0) {
        return Err(PyValueError::new_err("target_size must be positive"));
    }
    let format = ImageFormat::parse(format)?;
    let options = SaveOptions {
        quality,
        compress_level,
        lossless,
        effort,
        tiff_compression,
        jpeg_subsampling,
        jpeg_optimize,
        jpeg_progressive,
        jpeg_keep_rgb,
        jpeg_smooth,
        jpeg_qtables: &jpeg_qtables,
        jpeg_qtable_quality,
        jpeg_restart_marker_blocks,
        jpeg_restart_marker_rows,
        jpeg_streamtype,
        pdf_dpi,
        icc_profile: icc_profile.unwrap_or_default(),
        exif: metadata_exif.unwrap_or_default(),
        xmp: metadata_xmp.unwrap_or_default(),
    };
    if let Some(output) = output.as_ref()
        && format == ImageFormat::Png
        && compressor.is_none()
        && target_size.is_none()
        && !attach_metadata
        && image.bit_depth() == 8
        && image.palette().is_none()
        && matches!(image.pixel_mode(), PixelMode::L | PixelMode::La | PixelMode::Rgb | PixelMode::Rgba)
        && image.width() > 0
        && image.height() > 0
        && u64::from(image.width()) * u64::from(image.height()) * image.pixel_mode().channels() as u64 >= 256 * 1024
    {
        let pixels = image.pixel_data()?;
        let path = output_path(py, output)?;
        let file = py
            .detach(|| std::fs::File::create(path).map_err(|error| (error.raw_os_error(), error.to_string())))
            .map_err(|error| output_error(error, output))?;
        let mut writer = BufWriter::with_capacity(64 * 1024, file);
        py.detach(|| blanket_codecs::encode_png_to(image, pixels, compress_level, &mut writer))
            .map_err(|error| blanket_codecs::encode_error(error, format.as_str()))?;
        py.detach(|| writer.flush().map_err(|error| (error.raw_os_error(), error.to_string()))).map_err(|error| output_error(error, output))?;
        return Ok(None);
    }
    let encoded = encode_image(py, image, format, options, compressor).map_err(|error| blanket_codecs::encode_error(error, format.as_str()))?;
    let encoded = if attach_metadata && !(format == ImageFormat::Jpeg && jpeg_streamtype == 1) {
        py.detach(|| {
            let (xmp, icc) =
                if format == ImageFormat::Jpeg && jpeg_streamtype == 2 { (&[][..], &[][..]) } else { (options.xmp, options.icc_profile) };
            blanket_codecs::metadata::encode_metadata(encoded, format.as_str(), options.exif, xmp, icc, &metadata_chunks)
        })?
    } else {
        encoded
    };
    if let Some(target_size) = target_size
        && encoded.len() as u64 > target_size
    {
        return Err(PyValueError::new_err(format!(
            "target_size of {target_size} bytes cannot be reached by this search; smallest qualifying candidate is {} bytes",
            encoded.len()
        )));
    }
    if let Some(output) = output {
        write_path(py, &output, encoded)?;
        return Ok(None);
    }
    Ok(Some(PyBytes::new(py, &encoded).unbind()))
}

fn encode_image(py: Python<'_>, image: &Image, format: ImageFormat, options: SaveOptions, compressor: Option<Compressor>) -> PyResult<Vec<u8>> {
    if format == ImageFormat::Jpeg && matches!(options.jpeg_streamtype, 1 | 2) && compressor.is_some() {
        return Err(PyValueError::new_err("compression optimization requires a complete JPEG stream"));
    }
    let compress_level = options.compress_level;
    if image.pixel_mode() == PixelMode::One && format == ImageFormat::Png {
        return py.detach(|| codecs::encode_one_png(image, compress_level));
    }
    if matches!(image.pixel_mode(), PixelMode::One | PixelMode::La | PixelMode::Pa)
        && !matches!(
            (image.pixel_mode(), format),
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
        let target = if image.pixel_mode() == PixelMode::One {
            "L"
        } else if format == ImageFormat::Png && image.pixel_mode() == PixelMode::La && compressor.is_none() {
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
    if let Some((palette_mode, _)) = image.palette() {
        if format == ImageFormat::Gif && compressor.is_none() {
            return py.detach(|| codecs::encode(image, format, options));
        }
        if matches!(format, ImageFormat::Pcx | ImageFormat::Palm) && *palette_mode == PixelMode::Rgb {
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
