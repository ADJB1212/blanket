use std::io::Cursor;
use std::sync::LazyLock;

use image::codecs::png::{CompressionType, FilterType, PngEncoder};
use image::{ColorType, ExtendedColorType, ImageDecoder, ImageEncoder, ImageFormat as RustFormat};
use jpegxl_rs::encode::{ColorEncoding, EncoderFrame, EncoderResult, EncoderSpeed};
use jpegxl_rs::parallel::resizable_runner::ResizableRunner;
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

// libheif owns a process-wide plugin registry. Keep its initialization guard
// alive so each image does not tear down and recreate the codec plugins.
// Contexts, encoders, and pixel planes remain local to each operation.
static HEIF_LIBRARY: LazyLock<libheif_rs::LibHeif> = LazyLock::new(libheif_rs::LibHeif::new);

thread_local! {
    // A runner is used by only its owning calling thread. Reuse its workers
    // across decodes; each image still gets a fresh decoder and metadata state.
    static JXL_DECODE_RUNNER: Option<ResizableRunner<'static>> = ResizableRunner::new(None);
    // Reuse workers, but let libjxl size the pool for each frame. Tiny images
    // should not pay the synchronization cost of a full-machine thread pool.
    static JXL_ENCODE_RUNNER: Option<ResizableRunner<'static>> = ResizableRunner::new(None);
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ImageFormat {
    Bmp,
    Gif,
    Ico,
    Png,
    Jpeg,
    Jxl,
    Tiff,
    Webp,
    Dng,
    Heif,
    Avif,
    Pdf,
}

impl ImageFormat {
    pub(crate) fn parse(value: &str) -> PyResult<Self> {
        match value.to_ascii_uppercase().as_str() {
            "BMP" => Ok(Self::Bmp),
            "GIF" => Ok(Self::Gif),
            "ICO" => Ok(Self::Ico),
            "PDF" => Ok(Self::Pdf),
            "AVIF" => Ok(Self::Avif),
            "TIFF" | "TIF" => Ok(Self::Tiff),
            "WEBP" => Ok(Self::Webp),
            "DNG" => Ok(Self::Dng),
            "HEIF" | "HEIC" => Ok(Self::Heif),
            "PNG" => Ok(Self::Png),
            "JPEG" | "JPG" => Ok(Self::Jpeg),
            "JXL" | "JPEGXL" | "JPEG XL" => Ok(Self::Jxl),
            _ => Err(PyValueError::new_err(format!(
                "unsupported image format {value:?}; expected PNG, JPEG, JXL, TIFF, WEBP, DNG, HEIF, AVIF, PDF, BMP, GIF, or ICO"
            ))),
        }
    }

    const fn as_str(self) -> &'static str {
        match self {
            Self::Bmp => "BMP",
            Self::Gif => "GIF",
            Self::Ico => "ICO",
            Self::Pdf => "PDF",
            Self::Avif => "AVIF",
            Self::Tiff => "TIFF",
            Self::Webp => "WEBP",
            Self::Dng => "DNG",
            Self::Heif => "HEIF",
            Self::Png => "PNG",
            Self::Jpeg => "JPEG",
            Self::Jxl => "JXL",
        }
    }

    fn detect(data: &[u8]) -> Option<Self> {
        if data.starts_with(b"BM") {
            Some(Self::Bmp)
        } else if data.starts_with(b"GIF87a") || data.starts_with(b"GIF89a") {
            Some(Self::Gif)
        } else if data.starts_with(b"\0\0\x01\0") {
            Some(Self::Ico)
        } else if data.starts_with(PNG_SIGNATURE) {
            Some(Self::Png)
        } else if data.starts_with(&[0xff, 0xd8, 0xff]) {
            Some(Self::Jpeg)
        } else if data.starts_with(&[0xff, 0x0a]) || data.starts_with(JXL_CONTAINER_SIGNATURE) {
            Some(Self::Jxl)
        } else if data.starts_with(b"II*\0") || data.starts_with(b"MM\0*") || data.starts_with(b"II+\0") || data.starts_with(b"MM\0+") {
            Some(if is_dng(data) { Self::Dng } else { Self::Tiff })
        } else if data.starts_with(b"RIFF") && data.get(8..12) == Some(b"WEBP") {
            Some(Self::Webp)
        } else if is_avif(data) {
            Some(Self::Avif)
        } else if is_heif(data) {
            Some(Self::Heif)
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

pub(crate) fn decode(data: &[u8], format: ImageFormat) -> Result<Image, String> {
    match format {
        ImageFormat::Bmp => decode_rust_image(data, RustFormat::Bmp, "BMP"),
        ImageFormat::Gif => decode_rust_image(data, RustFormat::Gif, "GIF"),
        ImageFormat::Ico => decode_ico(data),
        ImageFormat::Pdf => Err("PDF is a write-only format".into()),
        ImageFormat::Avif => decode_rust_image(data, RustFormat::Avif, "AVIF"),
        ImageFormat::Tiff => decode_rust_image(data, RustFormat::Tiff, "TIFF"),
        ImageFormat::Webp => decode_webp(data),
        ImageFormat::Dng => decode_dng(data),
        ImageFormat::Heif => decode_heif(data),
        ImageFormat::Png => decode_rust_image(data, RustFormat::Png, "PNG"),
        ImageFormat::Jpeg => decode_jpeg(data),
        ImageFormat::Jxl => decode_jxl(data),
    }
}

fn decode_ico(data: &[u8]) -> Result<Image, String> {
    let count = data.get(4..6).ok_or("truncated ICO header")?;
    let count = usize::from(u16::from_le_bytes([count[0], count[1]]));
    let directory = data.get(6..6 + count * 16).ok_or("truncated ICO directory")?;
    let dimension = |v: u8| if v == 0 { 256_u32 } else { u32::from(v) };
    let entry = directory
        .as_chunks::<16>()
        .0
        .iter()
        .max_by_key(|entry| (dimension(entry[0]) * dimension(entry[1]), u16::from_le_bytes([entry[6], entry[7]])))
        .ok_or("empty ICO directory")?;
    let length = u32::from_le_bytes(entry[8..12].try_into().unwrap()) as usize;
    let offset = u32::from_le_bytes(entry[12..16].try_into().unwrap()) as usize;
    let end = offset.checked_add(length).ok_or("ICO payload overflow")?;
    let payload = data.get(offset..end).ok_or("truncated ICO payload")?;
    if payload.starts_with(PNG_SIGNATURE) {
        let image = decode_rust_image(payload, RustFormat::Png, "ICO")?;
        if image.width != dimension(entry[0]) || image.height != dimension(entry[1]) {
            return Err("ICO directory and PNG dimensions differ".into());
        }
        return Ok(image);
    }
    // Keep the bitmap decoder's AND-mask handling, selecting the same entry
    // as the PNG path regardless of the original directory ordering.
    let mut selected = b"\0\0\x01\0\x01\0".to_vec();
    selected.extend_from_slice(entry);
    selected[18..22].copy_from_slice(&22_u32.to_le_bytes());
    selected.extend_from_slice(payload);
    decode_rust_image(&selected, RustFormat::Ico, "ICO")
}

// DNGVersion (50706) belongs to the first classic TIFF IFD. Read only the
// bounded directory, never scan compressed image payloads for tag bytes.
fn is_dng(data: &[u8]) -> bool {
    if !data.starts_with(b"II*\0") && !data.starts_with(b"MM\0*") {
        return false;
    }
    let le = data.starts_with(b"II");
    let u16_at = |offset: usize| -> Option<u16> {
        let bytes = data.get(offset..offset.checked_add(2)?)?.try_into().ok()?;
        Some(if le { u16::from_le_bytes(bytes) } else { u16::from_be_bytes(bytes) })
    };
    let Some(bytes) = data.get(4..8).and_then(|v| v.try_into().ok()) else {
        return false;
    };
    let offset = if le { u32::from_le_bytes(bytes) } else { u32::from_be_bytes(bytes) } as usize;
    let Some(count) = u16_at(offset) else { return false };
    (0..usize::from(count)).any(|i| offset.checked_add(2 + i * 12).and_then(u16_at) == Some(50706))
}

fn decode_dng(data: &[u8]) -> Result<Image, String> {
    // Rawler's camera-specific decoders can panic on malformed raw data.
    std::panic::catch_unwind(|| {
        use rawler::formats::tiff::{GenericTiffReader, reader::TiffReader};
        // Check full-resolution IFDs as well as previews before raw allocation.
        let tiff = GenericTiffReader::new_with_buffer(data, 0, 0, None).map_err(|e| e.to_string())?;
        for ifd in tiff.find_ifds_with_tag(256_u16) {
            let width = ifd.get_entry(256_u16).ok_or("missing DNG width")?.force_u32(0);
            let height = ifd.get_entry(257_u16).ok_or("missing DNG height")?.force_u32(0);
            validate_dimensions(width, height)?;
        }
        let source = rawler::rawsource::RawSource::new_from_slice(data);
        let raw = rawler::decode(&source, &rawler::decoders::RawDecodeParams::default()).map_err(|e| e.to_string())?;
        validate_dimensions(
            u32::try_from(raw.width).map_err(|e| e.to_string())?,
            u32::try_from(raw.height).map_err(|e| e.to_string())?,
        )?;
        let developed = rawler::imgop::develop::RawDevelop::default()
            .develop_intermediate(&raw)
            .map_err(|e| e.to_string())?
            .to_dynamic_image()
            .ok_or("invalid developed DNG image")?
            .to_rgb8();
        Image::from_pixels(
            developed.width(),
            developed.height(),
            PixelMode::Rgb,
            developed.into_raw(),
            Some("DNG".into()),
        )
        .map_err(|e| e.to_string())
    })
    .unwrap_or_else(|_| Err("invalid DNG image".into()))
}

fn decode_webp(data: &[u8]) -> Result<Image, String> {
    let features = webpx::ImageInfo::from_webp(data).map_err(|error| error.to_string())?;
    validate_dimensions(features.width, features.height)?;
    // Preserve the existing first-frame behavior for animated containers.
    if features.has_animation {
        return decode_rust_image(data, RustFormat::WebP, "WEBP");
    }
    let (pixels, width, height) = if features.has_alpha {
        webpx::decode_rgba(data)
    } else {
        webpx::decode_rgb(data)
    }
    .map_err(|error| error.to_string())?;
    Image::from_pixels(
        width,
        height,
        if features.has_alpha { PixelMode::Rgba } else { PixelMode::Rgb },
        pixels,
        Some("WEBP".into()),
    )
    .map_err(|error| error.to_string())
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
    if matches!(color, ColorType::L16 | ColorType::La16 | ColorType::Rgb16 | ColorType::Rgba16) {
        let (mode, mut samples) = match color {
            ColorType::L16 => (PixelMode::L, image.into_luma16().into_raw()),
            ColorType::Rgb16 => (PixelMode::Rgb, image.into_rgb16().into_raw()),
            _ => (PixelMode::Rgba, image.into_rgba16().into_raw()),
        };
        let mut depth = 16;
        if format == RustFormat::Png {
            let reader = png::Decoder::new(Cursor::new(data)).read_info().map_err(|e| e.to_string())?;
            if let Some(bits) = reader.info().sbit.as_deref()
                && bits.len() == mode.channels()
                && matches!(bits[0], 10 | 12)
                && bits.iter().all(|&b| b == bits[0])
            {
                depth = bits[0];
                for sample in &mut samples {
                    *sample >>= 16 - depth;
                }
            }
        }
        return Image::from_samples(width, height, mode, samples, depth, Some(format_name.into())).map_err(|e| e.to_string());
    }
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
        decoder.decode(data).map_err(|error| error.to_string())
    })?;
    validate_dimensions(info.width, info.height)?;

    let pixels = match pixels {
        jpegxl_rs::decode::Pixels::Uint8(pixels) => pixels,
        jpegxl_rs::decode::Pixels::Uint16(pixels) => {
            let (mode, samples) = match (info.num_color_channels, info.has_alpha_channel) {
                (1, false) => (PixelMode::L, pixels),
                (3, false) => (PixelMode::Rgb, pixels),
                (3, true) => (PixelMode::Rgba, pixels),
                (1, true) => (
                    PixelMode::Rgba,
                    pixels.as_chunks::<2>().0.iter().flat_map(|v| [v[0], v[0], v[0], v[1]]).collect(),
                ),
                _ => return Err("unsupported JPEG XL channel layout".into()),
            };
            return Image::from_samples(info.width, info.height, mode, samples, 16, Some("JXL".into())).map_err(|e| e.to_string());
        }
        // Retain the existing display conversion for floating-point inputs.
        _ => JXL_DECODE_RUNNER.with(|runner| {
            let runner = runner.as_ref().ok_or_else(|| "cannot allocate JPEG XL thread pool".to_owned())?;
            let decoder = decoder_builder().parallel_runner(runner).build().map_err(|e| e.to_string())?;
            decoder.decode_with::<u8>(data).map(|(_, pixels)| pixels).map_err(|e| e.to_string())
        })?,
    };

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
    if format == ImageFormat::Heif {
        return encode_heif(image, options);
    }
    if image.bit_depth > 8 {
        return encode_wide(image, format, options);
    }
    let pixels = image.pixel_data()?;
    match format {
        ImageFormat::Bmp | ImageFormat::Gif | ImageFormat::Ico => {
            if image.width == 0 || image.height == 0 {
                return Err(PyValueError::new_err("cannot encode an empty image"));
            }
            if format == ImageFormat::Ico && (image.width > 256 || image.height > 256) {
                return Err(PyValueError::new_err("ICO dimensions must be between 1 and 256"));
            }
            if format == ImageFormat::Gif && (image.width > 65535 || image.height > 65535) {
                return Err(PyValueError::new_err("GIF dimensions must be between 1 and 65535"));
            }
            let mut output = Vec::new();
            match format {
                ImageFormat::Bmp => image::codecs::bmp::BmpEncoder::new(&mut output)
                    .write_image(pixels, image.width, image.height, color_type(image.mode))
                    .map_err(codec_error)?,
                ImageFormat::Ico => image::codecs::ico::IcoEncoder::new(&mut output)
                    .write_image(pixels, image.width, image.height, color_type(image.mode))
                    .map_err(codec_error)?,
                ImageFormat::Gif => {
                    let rgb;
                    let (pixels, color) = if image.mode == PixelMode::L {
                        rgb = pixels.iter().flat_map(|v| [*v; 3]).collect::<Vec<_>>();
                        (rgb.as_slice(), ExtendedColorType::Rgb8)
                    } else {
                        (pixels, color_type(image.mode))
                    };
                    image::codecs::gif::GifEncoder::new(&mut output)
                        .encode(pixels, image.width, image.height, color)
                        .map_err(codec_error)?;
                }
                _ => unreachable!(),
            }
            Ok(output)
        }
        ImageFormat::Pdf => encode_pdf(image, pixels),
        ImageFormat::Avif => {
            let rgb;
            let (pixels, color) = if image.mode == PixelMode::L {
                rgb = pixels.iter().flat_map(|v| [*v; 3]).collect::<Vec<_>>();
                (rgb.as_slice(), ExtendedColorType::Rgb8)
            } else {
                (pixels, color_type(image.mode))
            };
            let mut output = Vec::new();
            image::codecs::avif::AvifEncoder::new_with_speed_quality(&mut output, 11 - options.effort, options.quality)
                .write_image(pixels, image.width, image.height, color)
                .map_err(codec_error)?;
            Ok(output)
        }
        ImageFormat::Dng => Err(PyValueError::new_err("DNG is a read-only format")),
        ImageFormat::Heif => unreachable!(),
        ImageFormat::Tiff => {
            let mut output = Cursor::new(Vec::new());
            image::codecs::tiff::TiffEncoder::new(&mut output)
                .write_image(pixels, image.width, image.height, color_type(image.mode))
                .map_err(codec_error)?;
            Ok(output.into_inner())
        }
        ImageFormat::Webp => {
            let rgb;
            let encoder = match image.mode {
                PixelMode::L => {
                    rgb = pixels.iter().flat_map(|v| [*v; 3]).collect::<Vec<_>>();
                    webpx::Encoder::new_rgb(&rgb, image.width, image.height)
                }
                PixelMode::Rgb => webpx::Encoder::new_rgb(pixels, image.width, image.height),
                PixelMode::Rgba => webpx::Encoder::new_rgba(pixels, image.width, image.height),
            };
            encoder
                .lossless(options.lossless)
                .quality(if options.lossless { 75.0 } else { f32::from(options.quality) })
                .encode(webpx::Unstoppable)
                .map_err(codec_error)
        }
        ImageFormat::Png => encode_png(image, pixels, options.compress_level),
        ImageFormat::Jpeg => encode_jpeg(image, pixels, options.quality),
        ImageFormat::Jxl => encode_jxl(image, pixels, options),
    }
}

// PDF 1.4 image XObjects with an optional grayscale soft mask. Raw streams
// retain exact samples without adding a runtime dependency or a lossy codec.
fn encode_pdf(image: &Image, pixels: &[u8]) -> PyResult<Vec<u8>> {
    if image.width == 0 || image.height == 0 {
        return Err(PyValueError::new_err("cannot encode empty PDF image"));
    }
    let (width, height) = (image.width, image.height);
    let mut output = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n".to_vec();
    let mut offsets = Vec::new();
    let mut object = |dictionary: &str, stream: Option<&[u8]>| {
        offsets.push(output.len());
        output.extend_from_slice(format!("{} 0 obj\n<< {dictionary}", offsets.len()).as_bytes());
        if let Some(data) = stream {
            output.extend_from_slice(format!(" /Length {} >>\nstream\n", data.len()).as_bytes());
            output.extend_from_slice(data);
            output.extend_from_slice(b"\nendstream\nendobj\n");
        } else {
            output.extend_from_slice(b" >>\nendobj\n");
        }
    };
    object("/Type /Catalog /Pages 2 0 R", None);
    object("/Type /Pages /Kids [3 0 R] /Count 1", None);
    object(
        &format!("/Type /Page /Parent 2 0 R /MediaBox [0 0 {width} {height}] /Resources << /XObject << /Im0 5 0 R >> >> /Contents 4 0 R"),
        None,
    );
    let content = format!("q\n{width} 0 0 {height} 0 0 cm\n/Im0 Do\nQ\n");
    object("", Some(content.as_bytes()));
    let rgb;
    let color = if image.mode == PixelMode::L { "DeviceGray" } else { "DeviceRGB" };
    let (samples, mask) = if image.mode == PixelMode::Rgba {
        rgb = pixels
            .as_chunks::<4>()
            .0
            .iter()
            .flat_map(|pixel| pixel[..3].iter().copied())
            .collect::<Vec<_>>();
        (rgb.as_slice(), " /SMask 6 0 R")
    } else {
        (pixels, "")
    };
    object(
        &format!("/Type /XObject /Subtype /Image /Width {width} /Height {height} /ColorSpace /{color} /BitsPerComponent 8{mask}"),
        Some(samples),
    );
    if image.mode == PixelMode::Rgba {
        let alpha: Vec<u8> = pixels.as_chunks::<4>().0.iter().map(|pixel| pixel[3]).collect();
        object(
            &format!("/Type /XObject /Subtype /Image /Width {width} /Height {height} /ColorSpace /DeviceGray /BitsPerComponent 8"),
            Some(&alpha),
        );
    }
    let xref = output.len();
    let size = offsets.len() + 1;
    output.extend_from_slice(format!("xref\n0 {size}\n0000000000 65535 f \n").as_bytes());
    for offset in offsets {
        output.extend_from_slice(format!("{offset:010} 00000 n \n").as_bytes());
    }
    output.extend_from_slice(format!("trailer\n<< /Size {size} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n").as_bytes());
    Ok(output)
}

fn color_type(mode: PixelMode) -> ExtendedColorType {
    match mode {
        PixelMode::L => ExtendedColorType::L8,
        PixelMode::Rgb => ExtendedColorType::Rgb8,
        PixelMode::Rgba => ExtendedColorType::Rgba8,
    }
}

pub(crate) fn encode_png(image: &Image, pixels: &[u8], compress_level: u8) -> PyResult<Vec<u8>> {
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
    encode_jxl_samples(image, JxlSamples::Byte(pixels), options, false)
}

enum JxlSamples<'a> {
    Byte(&'a [u8]),
    Wide(&'a [u16]),
}

fn encode_jxl_samples(image: &Image, pixels: JxlSamples<'_>, options: SaveOptions, single_threaded: bool) -> PyResult<Vec<u8>> {
    let (color_encoding, has_alpha) = match image.mode {
        PixelMode::L => (ColorEncoding::SrgbLuma, false),
        PixelMode::Rgb => (ColorEncoding::Srgb, false),
        PixelMode::Rgba => (ColorEncoding::Srgb, true),
    };
    let speed = jxl_encoder_speed(options.effort)?;
    JXL_ENCODE_RUNNER.with(|runner| {
        let runner = runner.as_ref().ok_or_else(|| PyOSError::new_err("cannot allocate JPEG XL thread pool"))?;
        let builder = encoder_builder()
            .has_alpha(has_alpha)
            .lossless(options.lossless)
            .speed(speed)
            .decoding_speed(0)
            .use_container(false)
            .jpeg_quality(f32::from(options.quality))
            .uses_original_profile(options.lossless || options.quality == 100)
            .color_encoding(color_encoding);
        let mut encoder = if single_threaded {
            builder.build()
        } else {
            builder.parallel_runner(runner).build()
        }
        .map_err(codec_error)?;
        match pixels {
            JxlSamples::Byte(pixels) => {
                let frame = EncoderFrame::new(pixels).num_channels(image.mode.channels() as u32);
                let encoded: EncoderResult<u8> = encoder.encode_frame(&frame, image.width, image.height).map_err(codec_error)?;
                Ok(encoded.data)
            }
            JxlSamples::Wide(pixels) => {
                let frame = EncoderFrame::new(pixels).num_channels(image.mode.channels() as u32);
                let encoded: EncoderResult<u16> = encoder.encode_frame(&frame, image.width, image.height).map_err(codec_error)?;
                Ok(encoded.data)
            }
        }
    })
}

/// Normalize wide samples once per effort search, rather than once per encode.
/// Eight-bit input is borrowed directly from the immutable source image.
pub(crate) struct PreparedJxl<'a> {
    image: &'a Image,
    wide: Option<Vec<u16>>,
}

impl<'a> PreparedJxl<'a> {
    pub(crate) fn new(image: &'a Image) -> PyResult<Self> {
        let raw = image.raw_data()?;
        let wide = (image.bit_depth > 8).then(|| crate::compressor_simd::normalize_u16(raw, image.bit_depth));
        Ok(Self { image, wide })
    }

    pub(crate) fn encode(&self, options: SaveOptions) -> PyResult<Vec<u8>> {
        self.encode_with_threads(options, false)
    }

    pub(crate) fn encode_with_threads(&self, options: SaveOptions, single_threaded: bool) -> PyResult<Vec<u8>> {
        match &self.wide {
            Some(samples) => encode_jxl_samples(self.image, JxlSamples::Wide(samples), options, single_threaded),
            None => encode_jxl_samples(self.image, JxlSamples::Byte(self.image.raw_data()?), options, single_threaded),
        }
    }
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

fn is_avif(data: &[u8]) -> bool {
    has_brand(data, &[b"avif", b"avis"])
}

fn has_brand(data: &[u8], accepted: &[&[u8; 4]]) -> bool {
    if data.get(4..8) != Some(b"ftyp") || data.len() < 16 {
        return false;
    }
    let size = u32::from_be_bytes(data[..4].try_into().unwrap()) as usize;
    if size < 16 || size > data.len() || !size.is_multiple_of(4) {
        return false;
    }
    std::iter::once(&data[8..12])
        .chain(data[16..size].as_chunks::<4>().0.iter().map(|v| v.as_slice()))
        .any(|brand| accepted.iter().any(|candidate| brand == candidate.as_slice()))
}

fn is_heif(data: &[u8]) -> bool {
    if data.get(4..8) != Some(b"ftyp") || data.len() < 16 {
        return false;
    }
    let size = u32::from_be_bytes(data[..4].try_into().unwrap()) as usize;
    if size < 16 || size > data.len() || !size.is_multiple_of(4) {
        return false;
    }
    let brands = std::iter::once(&data[8..12]).chain(data[16..size].as_chunks::<4>().0.iter().map(|v| v.as_slice()));
    brands.into_iter().any(|brand| {
        matches!(
            brand,
            b"heic" | b"heix" | b"hevc" | b"hevx" | b"heim" | b"heis" | b"hevm" | b"hevs" | b"mif1" | b"msf1"
        )
    })
}

fn decode_heif(data: &[u8]) -> Result<Image, String> {
    use libheif_rs::{ColorSpace, DecodingOptions, HeifContext, RgbChroma};
    let lib = &*HEIF_LIBRARY;
    let context = HeifContext::read_from_bytes(data).map_err(|e| e.to_string())?;
    let handle = context.primary_image_handle().map_err(|e| e.to_string())?;
    validate_dimensions(handle.width(), handle.height())?;
    let depth = handle.luma_bits_per_pixel();
    if !matches!(depth, 8 | 10 | 12 | 16) {
        return Err(format!("unsupported HEIF bit depth: {depth}"));
    }
    let alpha = handle.has_alpha_channel();
    let chroma = match (depth > 8, alpha) {
        (false, false) => RgbChroma::Rgb,
        (false, true) => RgbChroma::Rgba,
        (true, false) => RgbChroma::HdrRgbLe,
        (true, true) => RgbChroma::HdrRgbaLe,
    };
    let mut options = DecodingOptions::new().ok_or("cannot allocate HEIF decoding options")?;
    // The context's tile-thread limit does not enable HEVC codec workers.
    // Small images do not have enough CTU rows to amortize starting workers.
    let threads = if u64::from(handle.width()) * u64::from(handle.height()) >= 256 * 1024 {
        std::thread::available_parallelism().map_or(1, |count| count.get().min(8))
    } else {
        1
    };
    options.set_num_codec_threads(threads as u32);
    let decoded = lib.decode(&handle, ColorSpace::Rgb(chroma), Some(options)).map_err(|e| e.to_string())?;
    validate_dimensions(decoded.width(), decoded.height())?;
    let mode = if alpha { PixelMode::Rgba } else { PixelMode::Rgb };
    let plane = decoded.planes().interleaved.ok_or("missing HEIF pixel plane")?;
    let row = decoded.width() as usize * mode.channels() * if depth > 8 { 2 } else { 1 };
    if row > plane.stride {
        return Err("invalid HEIF row stride".into());
    }
    let rows = plane.data.chunks_exact(plane.stride).take(decoded.height() as usize);
    if depth == 8 {
        let mut pixels = Vec::with_capacity(row * decoded.height() as usize);
        if row == plane.stride {
            pixels.extend_from_slice(&plane.data[..row * decoded.height() as usize]);
        } else {
            for source in rows {
                pixels.extend_from_slice(&source[..row]);
            }
        }
        Image::from_pixels(decoded.width(), decoded.height(), mode, pixels, Some("HEIF".into()))
    } else {
        Image::from_samples(
            decoded.width(),
            decoded.height(),
            mode,
            rows.flat_map(|source| source[..row].as_chunks::<2>().0.iter().map(|v| u16::from_le_bytes(*v)))
                .collect(),
            depth,
            Some("HEIF".into()),
        )
    }
    .map_err(|e| e.to_string())
}

fn encode_heif(image: &Image, options: SaveOptions) -> PyResult<Vec<u8>> {
    encode_heif_with_preset(image, options, "medium")
}

pub(crate) fn encode_heif_with_preset(image: &Image, options: SaveOptions, preset: &str) -> PyResult<Vec<u8>> {
    PreparedHeif::new(image)?.encode(options, preset)
}

/// Own pixel planes and encoder parameter metadata once for a preset search.
/// Every encoding resets its quality/preset and gets a fresh container.
pub(crate) struct PreparedHeif {
    native: libheif_rs::Image,
    encoder: libheif_rs::Encoder<'static>,
    presets: bool,
}

impl PreparedHeif {
    pub(crate) fn new(image: &Image) -> PyResult<Self> {
        use libheif_rs::{Channel, ColorSpace, RgbChroma};

        let pixels = image.raw_data()?;
        if image.width == 0 || image.height == 0 {
            return Err(PyValueError::new_err("cannot encode empty HEIF image"));
        }
        if !matches!(image.bit_depth, 8 | 10 | 12) {
            return Err(PyValueError::new_err("HEIF encoding supports 8, 10, or 12 bits per channel"));
        }
        let _library = &*HEIF_LIBRARY;
        let chroma = match (image.bit_depth > 8, image.mode == PixelMode::Rgba) {
            (false, false) => RgbChroma::Rgb,
            (false, true) => RgbChroma::Rgba,
            (true, false) => RgbChroma::HdrRgbLe,
            (true, true) => RgbChroma::HdrRgbaLe,
        };
        let grayscale = image.mode == PixelMode::L;
        let color_space = if grayscale { ColorSpace::Monochrome } else { ColorSpace::Rgb(chroma) };
        let channel = if grayscale { Channel::Y } else { Channel::Interleaved };
        let mut native = libheif_rs::Image::new(image.width, image.height, color_space).map_err(codec_error)?;
        native
            .create_plane(channel, image.width, image.height, image.bit_depth)
            .map_err(codec_error)?;
        let row = image.width as usize * image.mode.channels() * if image.bit_depth > 8 { 2 } else { 1 };
        let planes = native.planes_mut();
        let plane = if grayscale { planes.y } else { planes.interleaved }.ok_or_else(|| PyOSError::new_err("missing HEIF pixel plane"))?;
        // Monochrome HEVC avoids RGB expansion, color conversion, and encoding
        // two constant chroma planes. Wide monochrome planes use native endian.
        if grayscale && image.bit_depth > 8 && cfg!(target_endian = "big") {
            for (source, target) in pixels.chunks_exact(row).zip(plane.data.chunks_exact_mut(plane.stride)) {
                for (sample, dest) in source.as_chunks::<2>().0.iter().zip(target[..row].as_chunks_mut::<2>().0) {
                    *dest = u16::from_le_bytes(*sample).to_ne_bytes();
                }
            }
        } else if row == plane.stride {
            plane.data[..pixels.len()].copy_from_slice(pixels);
        } else {
            for (source, target) in pixels.chunks_exact(row).zip(plane.data.chunks_exact_mut(plane.stride)) {
                target[..row].copy_from_slice(source);
            }
        }
        let encoder = HEIF_LIBRARY
            .encoder_for_format(libheif_rs::CompressionFormat::Hevc)
            .map_err(codec_error)?;
        let presets = encoder.name().starts_with("x265 ");
        Ok(Self { native, encoder, presets })
    }

    pub(crate) fn supports_presets(&self) -> bool {
        self.presets
    }

    pub(crate) fn encode(&mut self, options: SaveOptions, preset: &str) -> PyResult<Vec<u8>> {
        use libheif_rs::{EncoderParameterValue, EncoderQuality, HeifContext};

        // Other HEVC plugins keep their defaults rather than receiving an
        // x265-specific parameter.
        if self.presets {
            self.encoder
                .set_parameter_value("preset", EncoderParameterValue::String(preset.into()))
                .map_err(codec_error)?;
        }
        self.encoder
            .set_quality(if options.lossless {
                EncoderQuality::LossLess
            } else {
                EncoderQuality::Lossy(options.quality)
            })
            .map_err(codec_error)?;
        let mut context = HeifContext::new().map_err(codec_error)?;
        context.encode_image(&self.native, &mut self.encoder, None).map_err(codec_error)?;
        context.write_to_bytes().map_err(codec_error)
    }
}

fn encode_wide(image: &Image, format: ImageFormat, options: SaveOptions) -> PyResult<Vec<u8>> {
    let pixels = image.raw_data()?;
    if !matches!(format, ImageFormat::Png | ImageFormat::Tiff | ImageFormat::Jxl) {
        return Err(PyValueError::new_err(
            "high-bit-depth saving supports HEIF, PNG, TIFF, or JXL; convert to bit_depth=8 explicitly for this format",
        ));
    }
    let samples = crate::compressor_simd::normalize_u16(pixels, image.bit_depth);
    if format == ImageFormat::Jxl {
        return encode_jxl_samples(image, JxlSamples::Wide(&samples), options, false);
    }
    if format == ImageFormat::Tiff {
        let data: Vec<u8> = samples.into_iter().flat_map(u16::to_ne_bytes).collect();
        let color = match image.mode {
            PixelMode::L => ExtendedColorType::L16,
            PixelMode::Rgb => ExtendedColorType::Rgb16,
            PixelMode::Rgba => ExtendedColorType::Rgba16,
        };
        let mut output = Cursor::new(Vec::new());
        image::codecs::tiff::TiffEncoder::new(&mut output)
            .write_image(&data, image.width, image.height, color)
            .map_err(codec_error)?;
        return Ok(output.into_inner());
    }
    let mut output = Vec::new();
    {
        let mut info = png::Info::with_size(image.width, image.height);
        info.bit_depth = png::BitDepth::Sixteen;
        info.color_type = match image.mode {
            PixelMode::L => png::ColorType::Grayscale,
            PixelMode::Rgb => png::ColorType::Rgb,
            PixelMode::Rgba => png::ColorType::Rgba,
        };
        let mut encoder = png::Encoder::with_info(&mut output, info).map_err(codec_error)?;
        encoder.set_deflate_compression(if options.compress_level == 0 {
            png::DeflateCompression::NoCompression
        } else {
            png::DeflateCompression::Level(options.compress_level)
        });
        let mut writer = encoder.write_header().map_err(codec_error)?;
        writer
            .write_chunk(png::chunk::sBIT, &vec![image.bit_depth; image.mode.channels()])
            .map_err(codec_error)?;
        writer
            .write_image_data(&samples.into_iter().flat_map(u16::to_be_bytes).collect::<Vec<_>>())
            .map_err(codec_error)?;
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reused_heif_planes_match_fresh_planes_across_presets_and_quality() {
        for mode in [PixelMode::L, PixelMode::Rgb, PixelMode::Rgba] {
            for depth in [8, 10, 12] {
                let count = 17 * 19 * mode.channels();
                let samples: Vec<u16> = (0..count).map(|i| ((i * 37) % (1 << depth)) as u16).collect();
                let image = if depth == 8 {
                    Image::from_pixels(17, 19, mode, samples.iter().map(|&v| v as u8).collect(), None).unwrap()
                } else {
                    Image::from_samples(17, 19, mode, samples, depth, None).unwrap()
                };
                let mut prepared = PreparedHeif::new(&image).unwrap();
                for (preset, lossless) in [("fast", false), ("slow", true), ("fast", false)] {
                    let options = SaveOptions {
                        quality: 71,
                        compress_level: 6,
                        lossless,
                        effort: 7,
                    };
                    let reused = prepared.encode(options, preset).unwrap();
                    let fresh = PreparedHeif::new(&image).unwrap().encode(options, preset).unwrap();
                    assert_eq!(reused, fresh, "mode={mode:?}, depth={depth}, preset={preset}");
                }
            }
        }
    }

    #[test]
    fn detects_avif_brands_before_generic_heif() {
        for data in [
            b"\0\0\0\x14ftypavif\0\0\0\0mif1".as_slice(),
            b"\0\0\0\x14ftypmif1\0\0\0\0avif",
            b"\0\0\0\x10ftypavis\0\0\0\0",
        ] {
            assert_eq!(ImageFormat::detect(data), Some(ImageFormat::Avif));
        }
        assert_eq!(ImageFormat::detect(b"\0\0\0\x20ftypavif\0\0\0\0"), None);
        assert_eq!(ImageFormat::detect(b"\0\0\0\x10ftypxxxxavif"), None);
        assert_eq!(ImageFormat::detect(b"\0\0\0\x10ftypxxxx\0\0\0\0avif"), None);
    }

    #[test]
    fn detects_supported_signatures() {
        assert_eq!(ImageFormat::detect(PNG_SIGNATURE), Some(ImageFormat::Png));
        assert_eq!(ImageFormat::detect(&[0xff, 0xd8, 0xff]), Some(ImageFormat::Jpeg));
        assert_eq!(ImageFormat::detect(&[0xff, 0x0a]), Some(ImageFormat::Jxl));
        assert_eq!(ImageFormat::detect(b"not an image"), None);
        assert_eq!(ImageFormat::detect(b"\0\0\0\x14ftypxxxx\0\0\0\0heix"), Some(ImageFormat::Heif));
        assert_eq!(ImageFormat::detect(b"\0\0\0\x10ftypheic\0\0\0\0"), Some(ImageFormat::Heif));
        assert_eq!(ImageFormat::detect(b"\0\0\0\x20ftypheic\0\0\0\0"), None);
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
