use std::io::Cursor;

use gamut_core::{DecodeImage, Dimensions, EncodeImage, Gray8, ImageBuf, ImageRef, Rgb8, Rgba8};
use gamut_jxl::{Distance, Effort, JxlDecoder, JxlEncoder};
use image::codecs::jpeg::JpegEncoder;
use image::codecs::png::{CompressionType, FilterType, PngEncoder};
use image::{ColorType, ExtendedColorType, ImageDecoder, ImageEncoder, ImageFormat as RustFormat};
use pyo3::exceptions::{PyOSError, PyValueError};
use pyo3::prelude::*;

use crate::UnidentifiedImageError;
use crate::raster::{Image, PixelMode};

const PNG_SIGNATURE: &[u8] = b"\x89PNG\r\n\x1a\n";
const JXL_CONTAINER_SIGNATURE: &[u8] = b"\0\0\0\x0cJXL \r\n\x87\n";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ImageFormat {
    Png,
    Jpeg,
    Jxl,
}

impl ImageFormat {
    pub(crate) fn parse(value: &str) -> PyResult<Self> {
        match value.to_ascii_uppercase().as_str() {
            "PNG" => Ok(Self::Png),
            "JPEG" | "JPG" => Ok(Self::Jpeg),
            "JXL" | "JPEGXL" | "JPEG XL" => Ok(Self::Jxl),
            _ => Err(PyValueError::new_err(format!(
                "unsupported image format {value:?}; expected PNG, JPEG, or JXL"
            ))),
        }
    }

    const fn as_str(self) -> &'static str {
        match self {
            Self::Png => "PNG",
            Self::Jpeg => "JPEG",
            Self::Jxl => "JXL",
        }
    }

    fn detect(data: &[u8]) -> Option<Self> {
        if data.starts_with(PNG_SIGNATURE) {
            Some(Self::Png)
        } else if data.starts_with(&[0xff, 0xd8, 0xff]) {
            Some(Self::Jpeg)
        } else if data.starts_with(&[0xff, 0x0a]) || data.starts_with(JXL_CONTAINER_SIGNATURE) {
            Some(Self::Jxl)
        } else {
            None
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct SaveOptions {
    pub(crate) quality: u8,
    pub(crate) compress_level: u8,
    pub(crate) lossless: bool,
    pub(crate) effort: u8,
}

#[pyfunction]
pub(crate) fn open_bytes(
    py: Python<'_>,
    data: &[u8],
    formats: Option<Vec<String>>,
) -> PyResult<Image> {
    let format = ImageFormat::detect(data)
        .ok_or_else(|| UnidentifiedImageError::new_err("cannot identify image file"))?;
    if let Some(formats) = formats {
        let allowed = formats
            .iter()
            .filter_map(|name| ImageFormat::parse(name).ok())
            .any(|candidate| candidate == format);
        if !allowed {
            return Err(UnidentifiedImageError::new_err(format!(
                "image format {} is not in formats",
                format.as_str()
            )));
        }
    }

    py.detach(|| decode(data, format))
        .map_err(UnidentifiedImageError::new_err)
}

fn decode(data: &[u8], format: ImageFormat) -> Result<Image, String> {
    match format {
        ImageFormat::Png => decode_rust_image(data, RustFormat::Png, "PNG"),
        ImageFormat::Jpeg => decode_rust_image(data, RustFormat::Jpeg, "JPEG"),
        ImageFormat::Jxl => decode_jxl(data),
    }
}

fn decode_rust_image(data: &[u8], format: RustFormat, format_name: &str) -> Result<Image, String> {
    let decoder = image::ImageReader::with_format(Cursor::new(data), format)
        .into_decoder()
        .map_err(|error| error.to_string())?;
    let color = decoder.color_type();
    let image = image::DynamicImage::from_decoder(decoder).map_err(|error| error.to_string())?;
    let (width, height) = (image.width(), image.height());
    let (mode, pixels) = match color {
        ColorType::L8 | ColorType::L16 => (PixelMode::L, image.into_luma8().into_raw()),
        ColorType::Rgb8 | ColorType::Rgb16 | ColorType::Rgb32F => {
            (PixelMode::Rgb, image.into_rgb8().into_raw())
        }
        _ => (PixelMode::Rgba, image.into_rgba8().into_raw()),
    };
    Image::from_pixels(width, height, mode, pixels, Some(format_name.to_owned()))
        .map_err(|error| error.to_string())
}

fn decode_jxl(data: &[u8]) -> Result<Image, String> {
    let decoder = JxlDecoder::new();
    let info = decoder.info(data).map_err(|error| error.to_string())?;
    let (mode, pixels) = if info.color_channels == 1 && !info.has_alpha {
        let image: ImageBuf<Gray8> = decoder.decode_image(data).map_err(|e| e.to_string())?;
        (PixelMode::L, image.as_samples().to_vec())
    } else if info.color_channels == 3 && !info.has_alpha {
        let image: ImageBuf<Rgb8> = decoder.decode_image(data).map_err(|e| e.to_string())?;
        (PixelMode::Rgb, image.as_samples().to_vec())
    } else {
        let image: ImageBuf<Rgba8> = decoder.decode_image(data).map_err(|e| e.to_string())?;
        (PixelMode::Rgba, image.as_samples().to_vec())
    };
    Image::from_pixels(
        info.dimensions.width,
        info.dimensions.height,
        mode,
        pixels,
        Some("JXL".to_owned()),
    )
    .map_err(|error| error.to_string())
}

pub(crate) fn encode(
    image: &Image,
    format: ImageFormat,
    options: SaveOptions,
) -> PyResult<Vec<u8>> {
    let pixels = image.pixel_data()?;
    match format {
        ImageFormat::Png => encode_png(image, pixels, options.compress_level),
        ImageFormat::Jpeg => encode_jpeg(image, pixels, options.quality),
        ImageFormat::Jxl => encode_jxl(image, pixels, options),
    }
}

fn color_type(mode: PixelMode) -> ExtendedColorType {
    match mode {
        PixelMode::L => ExtendedColorType::L8,
        PixelMode::Rgb => ExtendedColorType::Rgb8,
        PixelMode::Rgba => ExtendedColorType::Rgba8,
    }
}

fn encode_png(image: &Image, pixels: &[u8], compress_level: u8) -> PyResult<Vec<u8>> {
    let mut output = Vec::new();
    PngEncoder::new_with_quality(
        &mut output,
        CompressionType::Level(compress_level),
        FilterType::Adaptive,
    )
    .write_image(pixels, image.width, image.height, color_type(image.mode))
    .map_err(codec_error)?;
    Ok(output)
}

fn encode_jpeg(image: &Image, pixels: &[u8], quality: u8) -> PyResult<Vec<u8>> {
    if image.mode == PixelMode::Rgba {
        return Err(PyOSError::new_err("cannot write mode RGBA as JPEG"));
    }
    let mut output = Vec::new();
    JpegEncoder::new_with_quality(&mut output, quality)
        .write_image(pixels, image.width, image.height, color_type(image.mode))
        .map_err(codec_error)?;
    Ok(output)
}

fn encode_jxl(image: &Image, pixels: &[u8], options: SaveOptions) -> PyResult<Vec<u8>> {
    let dimensions = Dimensions::new(image.width, image.height).map_err(codec_error)?;
    let effort = Effort::from_level(options.effort)
        .ok_or_else(|| PyValueError::new_err("effort must be between 1 and 10"))?;
    let encoder = if options.lossless {
        JxlEncoder::lossless()
    } else {
        let distance = quality_to_distance(options.quality);
        JxlEncoder::lossy(Distance::new(distance).map_err(codec_error)?)
    }
    .with_effort(effort);

    match image.mode {
        PixelMode::L => encoder
            .encode_to_vec(ImageRef::<Gray8>::new(pixels, dimensions).map_err(codec_error)?)
            .map_err(codec_error),
        PixelMode::Rgb => encoder
            .encode_to_vec(ImageRef::<Rgb8>::new(pixels, dimensions).map_err(codec_error)?)
            .map_err(codec_error),
        PixelMode::Rgba => encoder
            .encode_to_vec(ImageRef::<Rgba8>::new(pixels, dimensions).map_err(codec_error)?)
            .map_err(codec_error),
    }
}

fn quality_to_distance(quality: u8) -> f32 {
    let quality = f32::from(quality);
    if quality >= 30.0 {
        0.1 + (100.0 - quality) * 0.09
    } else {
        (6.24 + 2.5_f32.powf((30.0 - quality) / 5.0) / 6.25).min(25.0)
    }
}

fn codec_error(error: impl std::fmt::Display) -> PyErr {
    PyOSError::new_err(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_supported_signatures() {
        assert_eq!(ImageFormat::detect(PNG_SIGNATURE), Some(ImageFormat::Png));
        assert_eq!(
            ImageFormat::detect(&[0xff, 0xd8, 0xff]),
            Some(ImageFormat::Jpeg)
        );
        assert_eq!(ImageFormat::detect(&[0xff, 0x0a]), Some(ImageFormat::Jxl));
        assert_eq!(ImageFormat::detect(b"not an image"), None);
    }

    #[test]
    fn quality_mapping_matches_libjxl_convention() {
        assert!((quality_to_distance(100) - 0.1).abs() < f32::EPSILON);
        assert!((quality_to_distance(90) - 1.0).abs() < f32::EPSILON);
        assert!(quality_to_distance(1) <= 25.0);
    }
}
