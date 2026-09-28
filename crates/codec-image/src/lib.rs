use blanket_codec_common::validate_dimensions;
use blanket_core::{Image, PixelMode};
use image::{ColorType, ExtendedColorType, ImageDecoder, ImageFormat as RustFormat};
use std::io::Cursor;

pub fn color_type(mode: PixelMode) -> ExtendedColorType {
    match mode {
        PixelMode::One | PixelMode::L => ExtendedColorType::L8,
        PixelMode::La => ExtendedColorType::La8,
        PixelMode::Rgb => ExtendedColorType::Rgb8,
        PixelMode::Rgba => ExtendedColorType::Rgba8,
        PixelMode::Pa => unreachable!("PA must be expanded before encoding"),
        _ => unreachable!("integer modes must be converted before encoding"),
    }
}

pub fn decode_rust_image(data: &[u8], format: RustFormat, format_name: &str) -> Result<Image, String> {
    decode_with_metadata(data, format, format_name, None, false)
}

pub fn decode_with_metadata(
    data: &[u8], format: RustFormat, format_name: &str, significant_bits: Option<&[u8]>, bilevel: bool,
) -> Result<Image, String> {
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
        let depth = significant_bits
            .filter(|bits| bits.len() == mode.channels() && matches!(bits[0], 10 | 12) && bits.iter().all(|&b| b == bits[0]))
            .map_or(16, |bits| bits[0]);
        if depth < 16 {
            for sample in &mut samples {
                *sample >>= 16 - depth;
            }
        }
        return Image::from_samples(width, height, mode, samples, depth, Some(format_name.into())).map_err(|e| e.to_string());
    }
    let (mode, pixels) = match color {
        ColorType::L8 | ColorType::L16 => {
            let mode = if bilevel { PixelMode::One } else { PixelMode::L };
            (mode, image.into_luma8().into_raw())
        }
        ColorType::La8 => (PixelMode::La, image.into_luma_alpha8().into_raw()),
        ColorType::Rgb8 | ColorType::Rgb16 | ColorType::Rgb32F => (PixelMode::Rgb, image.into_rgb8().into_raw()),
        _ => (PixelMode::Rgba, image.into_rgba8().into_raw()),
    };
    Image::from_pixels(width, height, mode, pixels, Some(format_name.to_owned())).map_err(|error| error.to_string())
}
