use blanket_codec_common::{SaveOptions, codec_error, validate_dimensions};
use blanket_codec_image::{color_type, decode_with_metadata};
use blanket_core::{Image, PixelMode};
use image::codecs::png::{CompressionType, FilterType, PngEncoder};
use image::{ImageEncoder, ImageFormat as RustFormat};
use pyo3::exceptions::{PyOSError, PyValueError};
use pyo3::prelude::*;
use std::io::Cursor;

pub fn encode_palette_png(image: &Image, compress_level: u8) -> PyResult<Vec<u8>> {
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

/// Decode packed rows of non-interlaced, opaque 1-bit grayscale PNGs.
fn decode_bilevel_png(data: &[u8]) -> Result<Option<Image>, String> {
    // IHDR bit depth, color type, and interlace method.
    if data.get(24..29).is_none_or(|header| header[0] != 1 || header[1] != 0 || header[4] != 0) {
        return Ok(None);
    }
    let mut decoder = png::Decoder::new(Cursor::new(data));
    decoder.set_transformations(png::Transformations::IDENTITY);
    let mut reader = decoder.read_info().map_err(|error| error.to_string())?;
    if reader.info().trns.is_some() {
        return Ok(None);
    }
    let (width, height) = reader.info().size();
    validate_dimensions(width, height)?;
    let mut packed = vec![0; reader.output_buffer_size().ok_or("image dimensions overflow addressable memory")?];
    reader.next_frame(&mut packed).map_err(|error| error.to_string())?;
    let pixels = blanket_core::raster::unpack_bilevel(&packed, width as usize, height as usize);
    Image::from_pixels(width, height, PixelMode::One, pixels, Some("PNG".into()))
        .map(Some)
        .map_err(|error| error.to_string())
}

pub fn encode_png(image: &Image, pixels: &[u8], compress_level: u8) -> PyResult<Vec<u8>> {
    let mut output = Vec::new();
    PngEncoder::new_with_quality(&mut output, CompressionType::Level(compress_level), FilterType::Adaptive)
        .write_image(pixels, image.width, image.height, color_type(image.mode))
        .map_err(codec_error)?;
    Ok(output)
}

pub fn encode_one_png(image: &Image, compress_level: u8) -> PyResult<Vec<u8>> {
    let width = image.width as usize;
    let packed = blanket_core::raster::pack_bilevel(image.pixel_data()?, width, image.height as usize);
    let mut output = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut output, image.width, image.height);
        encoder.set_color(png::ColorType::Grayscale);
        encoder.set_depth(png::BitDepth::One);
        encoder.set_compression(match compress_level {
            0 => png::Compression::NoCompression,
            1..=3 => png::Compression::Fast,
            4..=6 => png::Compression::Balanced,
            _ => png::Compression::High,
        });
        encoder.set_filter(png::Filter::Sub);
        encoder
            .write_header()
            .map_err(codec_error)?
            .write_image_data(&packed)
            .map_err(codec_error)?;
    }
    Ok(output)
}

pub fn decode_png(data: &[u8]) -> Result<Image, String> {
    if let Some(image) = decode_bilevel_png(data)? {
        return Ok(image);
    }
    let reader = png::Decoder::new(Cursor::new(data)).read_info().map_err(|e| e.to_string())?;
    let info = reader.info();
    decode_with_metadata(
        data,
        RustFormat::Png,
        "PNG",
        info.sbit.as_deref(),
        info.color_type == png::ColorType::Grayscale && info.bit_depth == png::BitDepth::One,
    )
}

pub fn encode_wide_png(image: &Image, samples: Vec<u16>, options: SaveOptions) -> PyResult<Vec<u8>> {
    let mut output = Vec::new();
    {
        let mut info = png::Info::with_size(image.width, image.height);
        info.bit_depth = png::BitDepth::Sixteen;
        info.color_type = match image.mode {
            PixelMode::L => png::ColorType::Grayscale,
            PixelMode::Rgb => png::ColorType::Rgb,
            PixelMode::Rgba => png::ColorType::Rgba,
            _ => return Err(PyValueError::new_err("unsupported high-bit-depth mode")),
        };
        let mut encoder = png::Encoder::with_info(&mut output, info).map_err(codec_error)?;
        encoder.set_filter(png::Filter::Sub);
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
