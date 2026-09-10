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
    // Encoding likewise reuses one worker pool instead of spawning a thread
    // per CPU for every save.
    static JXL_ENCODE_RUNNER: Option<ThreadsRunner<'static>> = ThreadsRunner::new(None, None);
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ImageFormat {
    Png,
    Jpeg,
    Jxl,
    Tiff,
    Webp,
    Dng,
    Heif,
}

impl ImageFormat {
    pub(crate) fn parse(value: &str) -> PyResult<Self> {
        match value.to_ascii_uppercase().as_str() {
            "TIFF" | "TIF" => Ok(Self::Tiff),
            "WEBP" => Ok(Self::Webp),
            "DNG" => Ok(Self::Dng),
            "HEIF" | "HEIC" => Ok(Self::Heif),
            "PNG" => Ok(Self::Png),
            "JPEG" | "JPG" => Ok(Self::Jpeg),
            "JXL" | "JPEGXL" | "JPEG XL" => Ok(Self::Jxl),
            _ => Err(PyValueError::new_err(format!(
                "unsupported image format {value:?}; expected PNG, JPEG, JXL, TIFF, WEBP, DNG, or HEIF"
            ))),
        }
    }

    const fn as_str(self) -> &'static str {
        match self {
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
        if data.starts_with(PNG_SIGNATURE) {
            Some(Self::Png)
        } else if data.starts_with(&[0xff, 0xd8, 0xff]) {
            Some(Self::Jpeg)
        } else if data.starts_with(&[0xff, 0x0a]) || data.starts_with(JXL_CONTAINER_SIGNATURE) {
            Some(Self::Jxl)
        } else if data.starts_with(b"II*\0") || data.starts_with(b"MM\0*") || data.starts_with(b"II+\0") || data.starts_with(b"MM\0+") {
            Some(if is_dng(data) { Self::Dng } else { Self::Tiff })
        } else if data.starts_with(b"RIFF") && data.get(8..12) == Some(b"WEBP") {
            Some(Self::Webp)
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

fn decode(data: &[u8], format: ImageFormat) -> Result<Image, String> {
    match format {
        ImageFormat::Tiff => decode_rust_image(data, RustFormat::Tiff, "TIFF"),
        ImageFormat::Webp => decode_webp(data),
        ImageFormat::Dng => decode_dng(data),
        ImageFormat::Heif => decode_heif(data),
        ImageFormat::Png => decode_rust_image(data, RustFormat::Png, "PNG"),
        ImageFormat::Jpeg => decode_jpeg(data),
        ImageFormat::Jxl => decode_jxl(data),
    }
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
    let speed = jxl_encoder_speed(options.effort)?;
    JXL_ENCODE_RUNNER.with(|runner| {
        let runner = runner.as_ref().ok_or_else(|| PyOSError::new_err("cannot allocate JPEG XL thread pool"))?;
        let mut encoder = encoder_builder()
            .parallel_runner(runner)
            .has_alpha(has_alpha)
            .lossless(options.lossless)
            .speed(speed)
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
    })
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
    use libheif_rs::{ColorSpace, HeifContext, LibHeif, RgbChroma};
    let lib = LibHeif::new();
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
    let decoded = lib.decode(&handle, ColorSpace::Rgb(chroma), None).map_err(|e| e.to_string())?;
    validate_dimensions(decoded.width(), decoded.height())?;
    let mode = if alpha { PixelMode::Rgba } else { PixelMode::Rgb };
    let plane = decoded.planes().interleaved.ok_or("missing HEIF pixel plane")?;
    let row = decoded.width() as usize * mode.channels() * if depth > 8 { 2 } else { 1 };
    if row > plane.stride {
        return Err("invalid HEIF row stride".into());
    }
    let pixels: Vec<u8> = plane
        .data
        .chunks_exact(plane.stride)
        .take(decoded.height() as usize)
        .flat_map(|v| v[..row].iter().copied())
        .collect();
    if depth == 8 {
        Image::from_pixels(decoded.width(), decoded.height(), mode, pixels, Some("HEIF".into()))
    } else {
        Image::from_samples(
            decoded.width(),
            decoded.height(),
            mode,
            pixels.as_chunks::<2>().0.iter().map(|v| u16::from_le_bytes(*v)).collect(),
            depth,
            Some("HEIF".into()),
        )
    }
    .map_err(|e| e.to_string())
}

fn encode_heif(image: &Image, options: SaveOptions) -> PyResult<Vec<u8>> {
    use libheif_rs::{Channel, ColorSpace, CompressionFormat, EncoderQuality, HeifContext, LibHeif, RgbChroma};
    let pixels = image.raw_data()?;
    if image.width == 0 || image.height == 0 {
        return Err(PyValueError::new_err("cannot encode empty HEIF image"));
    }
    if !matches!(image.bit_depth, 8 | 10 | 12) {
        return Err(PyValueError::new_err("HEIF encoding supports 8, 10, or 12 bits per channel"));
    }
    let lib = LibHeif::new();
    let mut encoder = lib.encoder_for_format(CompressionFormat::Hevc).map_err(codec_error)?;
    encoder
        .set_quality(if options.lossless {
            EncoderQuality::LossLess
        } else {
            EncoderQuality::Lossy(options.quality)
        })
        .map_err(codec_error)?;
    let chroma = match (image.bit_depth > 8, image.mode == PixelMode::Rgba) {
        (false, false) => RgbChroma::Rgb,
        (false, true) => RgbChroma::Rgba,
        (true, false) => RgbChroma::HdrRgbLe,
        (true, true) => RgbChroma::HdrRgbaLe,
    };
    let mut native = libheif_rs::Image::new(image.width, image.height, ColorSpace::Rgb(chroma)).map_err(codec_error)?;
    native
        .create_plane(Channel::Interleaved, image.width, image.height, image.bit_depth)
        .map_err(codec_error)?;
    let expanded;
    let bytes = if image.mode == PixelMode::L {
        let size = if image.bit_depth > 8 { 2 } else { 1 };
        expanded = pixels
            .chunks_exact(size)
            .flat_map(|v| v.iter().copied().cycle().take(size * 3))
            .collect::<Vec<_>>();
        &expanded
    } else {
        pixels
    };
    let channels = if image.mode == PixelMode::Rgba { 4 } else { 3 };
    let row = image.width as usize * channels * if image.bit_depth > 8 { 2 } else { 1 };
    let plane = native
        .planes_mut()
        .interleaved
        .ok_or_else(|| PyOSError::new_err("missing HEIF pixel plane"))?;
    for (source, target) in bytes.chunks_exact(row).zip(plane.data.chunks_exact_mut(plane.stride)) {
        target[..row].copy_from_slice(source);
    }
    let mut context = HeifContext::new().map_err(codec_error)?;
    context.encode_image(&native, &mut encoder, None).map_err(codec_error)?;
    context.write_to_bytes().map_err(codec_error)
}

fn encode_wide(image: &Image, format: ImageFormat, options: SaveOptions) -> PyResult<Vec<u8>> {
    let pixels = image.raw_data()?;
    if !matches!(format, ImageFormat::Png | ImageFormat::Tiff | ImageFormat::Jxl) {
        return Err(PyValueError::new_err(
            "high-bit-depth saving supports HEIF, PNG, TIFF, or JXL; convert to bit_depth=8 explicitly for this format",
        ));
    }
    let maximum = (1_u32 << image.bit_depth) - 1;
    let samples: Vec<u16> = pixels
        .as_chunks::<2>()
        .0
        .iter()
        .map(|v| ((u32::from(u16::from_le_bytes([v[0], v[1]])) * 65535 + maximum / 2) / maximum) as u16)
        .collect();
    if format == ImageFormat::Jxl {
        return JXL_ENCODE_RUNNER.with(|runner| {
            let runner = runner.as_ref().ok_or_else(|| PyOSError::new_err("cannot allocate JPEG XL thread pool"))?;
            let mut encoder = encoder_builder()
                .parallel_runner(runner)
                .has_alpha(image.mode == PixelMode::Rgba)
                .lossless(options.lossless)
                .speed(jxl_encoder_speed(options.effort)?)
                .decoding_speed(0)
                .use_container(false)
                .jpeg_quality(f32::from(options.quality))
                .uses_original_profile(options.lossless || options.quality == 100)
                .color_encoding(if image.mode == PixelMode::L {
                    ColorEncoding::SrgbLuma
                } else {
                    ColorEncoding::Srgb
                })
                .build()
                .map_err(codec_error)?;
            let frame = EncoderFrame::new(&samples).num_channels(image.mode.channels() as u32);
            let encoded: EncoderResult<u16> = encoder.encode_frame(&frame, image.width, image.height).map_err(codec_error)?;
            Ok(encoded.data)
        });
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
