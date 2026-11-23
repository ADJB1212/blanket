#![feature(portable_simd)]

use std::io::Cursor;
use std::simd::{Simd, ToBytes};

use blanket_codec_common::{DecodeError, validate_dimensions};
use blanket_core::{Image, PixelMode};
use image::{ColorType, ExtendedColorType, ImageDecoder, ImageFormat as RustFormat};

/// Shifts native-endian 16-bit samples right and stores them little-endian.
fn native_to_le_shifted(pixels: &mut [u8], shift: u8) {
    let (blocks, tail) = pixels.as_chunks_mut::<32>();
    let shift = Simd::<u16, 16>::splat(u16::from(shift));
    for block in blocks {
        *block = (Simd::<u16, 16>::from_ne_bytes(Simd::from_array(*block)) >> shift).to_le_bytes().to_array();
    }
    for bytes in tail.as_chunks_mut::<2>().0 {
        *bytes = (u16::from_ne_bytes(*bytes) >> shift[0]).to_le_bytes();
    }
}

pub fn inspect_rust_image(data: &[u8], format: RustFormat) -> Result<blanket_codec_common::ImageHeader, String> {
    let decoder = image::ImageReader::with_format(Cursor::new(data), format).into_decoder().map_err(|e| e.to_string())?;
    let (width, height) = decoder.dimensions();
    let (mode, depth) = match decoder.color_type() {
        ColorType::L8 => ("L", 8),
        ColorType::La8 => ("LA", 8),
        ColorType::Rgb8 => ("RGB", 8),
        ColorType::Rgba8 => ("RGBA", 8),
        ColorType::L16 => ("L", 16),
        ColorType::La16 => ("LA", 16),
        ColorType::Rgb16 => ("RGB", 16),
        ColorType::Rgba16 => ("RGBA", 16),
        ColorType::Rgb32F => ("RGB", 32),
        ColorType::Rgba32F => ("RGBA", 32),
        _ => return Err("unsupported header color type".into()),
    };
    blanket_codec_common::ImageHeader::new(width, height, mode, depth)
}

pub fn color_type(mode: PixelMode) -> Result<ExtendedColorType, image::ImageError> {
    Ok(match mode {
        PixelMode::One | PixelMode::L => ExtendedColorType::L8,
        PixelMode::La => ExtendedColorType::La8,
        PixelMode::Rgb => ExtendedColorType::Rgb8,
        PixelMode::Rgba => ExtendedColorType::Rgba8,
        _ => {
            use image::error::{ImageFormatHint, UnsupportedError, UnsupportedErrorKind};
            let kind = UnsupportedErrorKind::GenericFeature(format!("cannot write mode {}", mode.as_str()));
            return Err(image::ImageError::Unsupported(UnsupportedError::from_format_and_kind(ImageFormatHint::Unknown, kind)));
        }
    })
}

pub fn decode_rust_image(data: &[u8], format: RustFormat, format_name: &str) -> Result<Image, DecodeError> {
    decode_with_metadata(data, format, format_name, None, LumaMode::Grayscale)
}

/// Interpretation of decoded single-channel pixels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LumaMode {
    Grayscale,
    Bilevel,
}

pub fn decode_with_metadata(
    data: &[u8], format: RustFormat, format_name: &str, significant_bits: Option<&[u8]>, luma: LumaMode,
) -> Result<Image, DecodeError> {
    let decoder = image::ImageReader::with_format(Cursor::new(data), format).into_decoder().map_err(DecodeError::with_diagnostic_source)?;
    let dimensions = decoder.dimensions();
    validate_dimensions(dimensions.0, dimensions.1)?;
    let color = decoder.color_type();
    if matches!(color, ColorType::L16 | ColorType::Rgb16 | ColorType::Rgba16) {
        let mode = match color {
            ColorType::L16 => PixelMode::L,
            ColorType::Rgb16 => PixelMode::Rgb,
            _ => PixelMode::Rgba,
        };
        let depth = significant_bits
            .filter(|bits| bits.len() == mode.channels() && matches!(bits[0], 10 | 12) && bits.iter().all(|&b| b == bits[0]))
            .map_or(16, |bits| bits[0]);
        let length = usize::try_from(decoder.total_bytes()).map_err(DecodeError::with_source)?;
        let mut pixels = Vec::new();
        pixels.try_reserve_exact(length).map_err(DecodeError::with_source)?;
        pixels.resize(length, 0);
        decoder.read_image(&mut pixels).map_err(DecodeError::with_diagnostic_source)?;
        native_to_le_shifted(&mut pixels, 16 - depth);
        return Image::from_sample_bytes(dimensions.0, dimensions.1, mode, pixels, depth, Some(format_name.into())).map_err(DecodeError::with_source);
    }
    let image = image::DynamicImage::from_decoder(decoder).map_err(DecodeError::with_diagnostic_source)?;
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
        return Image::from_samples(width, height, mode, samples, depth, Some(format_name.into())).map_err(DecodeError::with_source);
    }
    let (mode, pixels) = match color {
        ColorType::L8 | ColorType::L16 => {
            let mode = match luma {
                LumaMode::Grayscale => PixelMode::L,
                LumaMode::Bilevel => PixelMode::One,
            };
            (mode, image.into_luma8().into_raw())
        }
        ColorType::La8 => (PixelMode::La, image.into_luma_alpha8().into_raw()),
        ColorType::Rgb8 | ColorType::Rgb16 | ColorType::Rgb32F => (PixelMode::Rgb, image.into_rgb8().into_raw()),
        _ => (PixelMode::Rgba, image.into_rgba8().into_raw()),
    };
    Image::from_pixels(width, height, mode, pixels, Some(format_name.to_owned())).map_err(DecodeError::with_source)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encoding_color_type_rejects_unsupported_modes() {
        for mode in [PixelMode::Hsv, PixelMode::Pa, PixelMode::I16] {
            let error = color_type(mode).unwrap_err();
            assert!(matches!(error, image::ImageError::Unsupported(_)));
            assert!(error.to_string().contains(mode.as_str()));
        }
        assert_eq!(color_type(PixelMode::Rgba).unwrap(), ExtendedColorType::Rgba8);
    }

    #[test]
    fn shifted_samples_match_scalar_across_blocks_and_tails() {
        for samples in [0, 1, 15, 16, 17, 33] {
            for shift in [0, 4, 6] {
                let source: Vec<u8> = (0..samples * 2).map(|i| (i * 91 + 17) as u8).collect();
                let mut actual = source.clone();
                native_to_le_shifted(&mut actual, shift);
                let mut expected = source;
                for bytes in expected.as_chunks_mut::<2>().0 {
                    *bytes = (u16::from_ne_bytes(*bytes) >> shift).to_le_bytes();
                }
                assert_eq!(actual, expected);
            }
        }
    }
}
