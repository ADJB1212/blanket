use std::io::Cursor;

use image::codecs::png::{CompressionType, FilterType, PngEncoder};
use image::{ColorType, ExtendedColorType, ImageDecoder, ImageEncoder, ImageFormat as RustFormat};
use jpegxl_rs::encode::{ColorEncoding, EncoderFrame, EncoderResult, EncoderSpeed};
use jpegxl_rs::parallel::resizable_runner::ResizableRunner;
use jpegxl_rs::parallel::threads_runner::ThreadsRunner;
use jpegxl_rs::{decoder_builder, encoder_builder};
use pyo3::exceptions::{PyOSError, PyValueError};
use pyo3::prelude::*;
use turbojpeg::{Colorspace, Compressor, Decompressor, PixelFormat, Subsamp};

use crate::UnidentifiedImageError;
use crate::raster::{Image, PixelMode};

const PNG_SIGNATURE: &[u8] = b"\x89PNG\r\n\x1a\n";

pub(crate) fn encode_palette_png(image: &Image, compress_level: u8) -> PyResult<Vec<u8>> {
    let pixels = image.pixel_data()?;
    let (mode, palette) = image.palette.as_ref().expect("palette checked by caller");
    let mut encoded = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut encoded, image.width, image.height);
        encoder.set_color(png::ColorType::Indexed);
        encoder.set_depth(png::BitDepth::Eight);
        encoder.set_compression(match compress_level {
            0 => png::Compression::NoCompression,
            1..=3 => png::Compression::Fast,
            4..=6 => png::Compression::Balanced,
            _ => png::Compression::High,
        });
        let mut rgb = Vec::new();
        let mut alpha = Vec::new();
        for color in palette.chunks_exact(mode.channels()) {
            rgb.extend_from_slice(&color[..3]);
            if *mode == PixelMode::Rgba {
                alpha.push(color[3]);
            }
        }
        encoder.set_palette(rgb);
        if !alpha.is_empty() {
            encoder.set_trns(alpha);
        }
        let mut writer = encoder.write_header().map_err(|e| PyOSError::new_err(e.to_string()))?;
        writer.write_image_data(pixels).map_err(|e| PyOSError::new_err(e.to_string()))?;
    }
    Ok(encoded)
}
const JXL_CONTAINER_SIGNATURE: &[u8] = b"\0\0\0\x0cJXL \r\n\x87\n";
const MAX_IMAGE_PIXELS: usize = 178_956_970;

thread_local! {
    // A runner is used by only its owning calling thread. Reuse its workers
    // across decodes; each image still gets a fresh decoder and metadata state.
    static JXL_DECODE_RUNNER: Option<ResizableRunner<'static>> = ResizableRunner::new(None);
}

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
pub(crate) fn open_bytes(py: Python<'_>, data: &[u8], formats: Option<Vec<String>>) -> PyResult<Image> {
    let format = ImageFormat::detect(data).ok_or_else(|| UnidentifiedImageError::new_err("cannot identify image file"))?;
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

    py.detach(|| decode(data, format)).map_err(UnidentifiedImageError::new_err)
}

fn decode(data: &[u8], format: ImageFormat) -> Result<Image, String> {
    match format {
        ImageFormat::Png => decode_rust_image(data, RustFormat::Png, "PNG"),
        ImageFormat::Jpeg => decode_jpeg(data),
        ImageFormat::Jxl => decode_jxl(data),
    }
}

fn decode_rust_image(data: &[u8], format: RustFormat, format_name: &str) -> Result<Image, String> {
    let decoder = image::ImageReader::with_format(Cursor::new(data), format)
        .into_decoder()
        .map_err(|error| error.to_string())?;
    let dimensions = decoder.dimensions();
    validate_dimensions(dimensions.0, dimensions.1)?;
    let color = decoder.color_type();
    let image = image::DynamicImage::from_decoder(decoder).map_err(|error| error.to_string())?;
    let (width, height) = (image.width(), image.height());
    let (mode, pixels) = match color {
        ColorType::L8 | ColorType::L16 => (PixelMode::L, image.into_luma8().into_raw()),
        ColorType::Rgb8 | ColorType::Rgb16 | ColorType::Rgb32F => (PixelMode::Rgb, image.into_rgb8().into_raw()),
        _ => (PixelMode::Rgba, image.into_rgba8().into_raw()),
    };
    Image::from_pixels(width, height, mode, pixels, Some(format_name.to_owned())).map_err(|error| error.to_string())
}

fn decode_jpeg(data: &[u8]) -> Result<Image, String> {
    let mut decoder = Decompressor::new().map_err(|error| error.to_string())?;
    let header = decoder.read_header(data).map_err(|error| error.to_string())?;
    let width = u32::try_from(header.width).map_err(|error| error.to_string())?;
    let height = u32::try_from(header.height).map_err(|error| error.to_string())?;
    validate_dimensions(width, height)?;

    let (mode, format) = match header.colorspace {
        Colorspace::Gray => (PixelMode::L, PixelFormat::GRAY),
        Colorspace::CMYK | Colorspace::YCCK => (PixelMode::Rgb, PixelFormat::CMYK),
        Colorspace::RGB | Colorspace::YCbCr => (PixelMode::Rgb, PixelFormat::RGB),
    };
    let pitch = header
        .width
        .checked_mul(format.size())
        .ok_or_else(|| "image dimensions overflow addressable memory".to_owned())?;
    let length = header
        .height
        .checked_mul(pitch)
        .ok_or_else(|| "image dimensions overflow addressable memory".to_owned())?;
    let mut pixels = vec![0; length];
    decoder
        .decompress(
            data,
            turbojpeg::Image {
                pixels: pixels.as_mut_slice(),
                width: header.width,
                pitch,
                height: header.height,
                format,
            },
        )
        .map_err(|error| error.to_string())?;
    let pixels = if format == PixelFormat::CMYK { cmyk_to_rgb(&pixels) } else { pixels };
    Image::from_pixels(width, height, mode, pixels, Some("JPEG".to_owned())).map_err(|error| error.to_string())
}

fn cmyk_to_rgb(cmyk: &[u8]) -> Vec<u8> {
    let mut rgb = Vec::with_capacity(cmyk.len() / 4 * 3);
    // JPEG stores CMYK samples inverted, so combining a color channel with K
    // is a multiplication rather than the usual subtractive CMYK formula.
    for pixel in cmyk.as_chunks::<4>().0 {
        rgb.push(multiply_u8(pixel[0], pixel[3]));
        rgb.push(multiply_u8(pixel[1], pixel[3]));
        rgb.push(multiply_u8(pixel[2], pixel[3]));
    }
    rgb
}

fn multiply_u8(left: u8, right: u8) -> u8 {
    let product = u16::from(left) * u16::from(right) + 128;
    ((product + (product >> 8)) >> 8) as u8
}

fn decode_jxl(data: &[u8]) -> Result<Image, String> {
    // Size the pool after reading basic info instead of starting one worker
    // per CPU even for small images with only a few independently coded groups.
    let (info, pixels) = JXL_DECODE_RUNNER.with(|runner| {
        let runner = runner.as_ref().ok_or_else(|| "cannot allocate JPEG XL thread pool".to_owned())?;
        let decoder = decoder_builder().parallel_runner(runner).build().map_err(|error| error.to_string())?;
        decoder.decode_with::<u8>(data).map_err(|error| error.to_string())
    })?;
    validate_dimensions(info.width, info.height)?;

    let (mode, pixels) = match (info.num_color_channels, info.has_alpha_channel) {
        (1, false) => (PixelMode::L, pixels),
        (3, false) => (PixelMode::Rgb, pixels),
        (1, true) => (PixelMode::Rgba, luma_alpha_to_rgba(&pixels)),
        (3, true) => (PixelMode::Rgba, pixels),
        (channels, has_alpha) => {
            return Err(format!(
                "unsupported JPEG XL channel layout: {channels} color channels, alpha={has_alpha}"
            ));
        }
    };
    Image::from_pixels(info.width, info.height, mode, pixels, Some("JXL".to_owned())).map_err(|error| error.to_string())
}

fn luma_alpha_to_rgba(luma_alpha: &[u8]) -> Vec<u8> {
    let mut rgba = Vec::with_capacity(luma_alpha.len().saturating_mul(2));
    for &[luma, alpha] in luma_alpha.as_chunks::<2>().0 {
        rgba.extend_from_slice(&[luma, luma, luma, alpha]);
    }
    rgba
}

pub(crate) fn encode(image: &Image, format: ImageFormat, options: SaveOptions) -> PyResult<Vec<u8>> {
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
    PngEncoder::new_with_quality(&mut output, CompressionType::Level(compress_level), FilterType::Adaptive)
        .write_image(pixels, image.width, image.height, color_type(image.mode))
        .map_err(codec_error)?;
    Ok(output)
}

fn encode_jpeg(image: &Image, pixels: &[u8], quality: u8) -> PyResult<Vec<u8>> {
    if image.mode == PixelMode::Rgba {
        return Err(PyOSError::new_err("cannot write mode RGBA as JPEG"));
    }

    let (format, subsampling) = match image.mode {
        PixelMode::L => (PixelFormat::GRAY, Subsamp::Gray),
        // Match the default used by Pillow/libjpeg for RGB JPEG output.
        PixelMode::Rgb => (PixelFormat::RGB, Subsamp::Sub2x2),
        PixelMode::Rgba => unreachable!("RGBA is rejected above"),
    };
    let mut encoder = Compressor::new().map_err(codec_error)?;
    encoder.set_quality(i32::from(quality)).map_err(codec_error)?;
    encoder.set_subsamp(subsampling).map_err(codec_error)?;
    encoder
        .compress_to_vec(turbojpeg::Image {
            pixels,
            width: image.width as usize,
            pitch: image.width as usize * format.size(),
            height: image.height as usize,
            format,
        })
        .map_err(codec_error)
}

fn encode_jxl(image: &Image, pixels: &[u8], options: SaveOptions) -> PyResult<Vec<u8>> {
    let (color_encoding, has_alpha) = match image.mode {
        PixelMode::L => (ColorEncoding::SrgbLuma, false),
        PixelMode::Rgb => (ColorEncoding::Srgb, false),
        PixelMode::Rgba => (ColorEncoding::Srgb, true),
    };
    let runner = ThreadsRunner::default();
    let mut encoder = encoder_builder()
        .parallel_runner(&runner)
        .has_alpha(has_alpha)
        .lossless(options.lossless)
        .speed(jxl_encoder_speed(options.effort)?)
        .decoding_speed(0)
        .use_container(false)
        .jpeg_quality(f32::from(options.quality))
        .uses_original_profile(options.lossless || options.quality == 100)
        .color_encoding(color_encoding)
        .build()
        .map_err(codec_error)?;
    let frame = EncoderFrame::new(pixels).num_channels(image.mode.channels() as u32);
    let encoded: EncoderResult<u8> = encoder.encode_frame(&frame, image.width, image.height).map_err(codec_error)?;
    Ok(encoded.data)
}

fn jxl_encoder_speed(effort: u8) -> PyResult<EncoderSpeed> {
    match effort {
        1 => Ok(EncoderSpeed::Lightning),
        2 => Ok(EncoderSpeed::Thunder),
        3 => Ok(EncoderSpeed::Falcon),
        4 => Ok(EncoderSpeed::Cheetah),
        5 => Ok(EncoderSpeed::Hare),
        6 => Ok(EncoderSpeed::Wombat),
        7 => Ok(EncoderSpeed::Squirrel),
        8 => Ok(EncoderSpeed::Kitten),
        9 => Ok(EncoderSpeed::Tortoise),
        10 => Ok(EncoderSpeed::Glacier),
        _ => Err(PyValueError::new_err("effort must be between 1 and 10")),
    }
}

fn validate_dimensions(width: u32, height: u32) -> Result<(), String> {
    let pixels = (width as usize)
        .checked_mul(height as usize)
        .ok_or_else(|| "image dimensions overflow addressable memory".to_owned())?;
    if pixels > MAX_IMAGE_PIXELS {
        return Err(format!("image size ({pixels} pixels) exceeds Blanket limit of {MAX_IMAGE_PIXELS} pixels"));
    }
    Ok(())
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
        assert_eq!(ImageFormat::detect(&[0xff, 0xd8, 0xff]), Some(ImageFormat::Jpeg));
        assert_eq!(ImageFormat::detect(&[0xff, 0x0a]), Some(ImageFormat::Jxl));
        assert_eq!(ImageFormat::detect(b"not an image"), None);
    }

    #[test]
    fn expands_luma_alpha_pixels_to_rgba() {
        assert_eq!(
            luma_alpha_to_rgba(&[0x10, 0x20, 0x30, 0x40]),
            [0x10, 0x10, 0x10, 0x20, 0x30, 0x30, 0x30, 0x40]
        );
    }

    #[test]
    fn converts_cmyk_pixels_to_rgb() {
        assert_eq!(
            cmyk_to_rgb(&[255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 0, 255]),
            [255, 0, 0, 0, 255, 0, 0, 0, 0]
        );
    }
}
