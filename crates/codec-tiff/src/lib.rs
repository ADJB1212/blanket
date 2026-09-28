use blanket_codec_common::{codec_error, validate_dimensions};
use blanket_codec_image::{color_type, decode_rust_image};
use blanket_core::{Image, PixelMode};
use image::{ExtendedColorType, ImageEncoder, ImageFormat as RustFormat};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::io::Cursor;

fn decode_special_tiff(data: &[u8]) -> Result<Option<Image>, String> {
    let Ok(mut decoder) = tiff::decoder::Decoder::new(Cursor::new(data)) else {
        return Ok(None);
    };
    let photometric = decoder.get_tag_u32(tiff::tags::Tag::PhotometricInterpretation).ok();
    if matches!(photometric, Some(8 | 9)) {
        let signed = photometric == Some(8);
        // Decode LAB as raw three-channel samples; the TIFF crate cannot read LAB buffers.
        let normalized = lab_tiff_as_rgb(data)?;
        decoder = tiff::decoder::Decoder::new(Cursor::new(normalized.as_slice())).map_err(|error| error.to_string())?;
        if decoder.colortype().map_err(|error| error.to_string())? != tiff::ColorType::RGB(8) {
            return Err("only 8-bit LAB TIFF images are supported".into());
        }
        let (width, height) = decoder.dimensions().map_err(|error| error.to_string())?;
        validate_dimensions(width, height)?;
        let tiff::decoder::DecodingResult::U8(pixels) = decoder.read_image().map_err(|error| error.to_string())? else {
            return Err("invalid LAB TIFF samples".into());
        };
        let pixels = if signed { blanket_core::raster::lab_raw_bytes(&pixels) } else { pixels };
        return Image::from_pixels(width, height, PixelMode::Lab, pixels, Some("TIFF".into()))
            .map(Some)
            .map_err(|error| error.to_string());
    }
    if decoder.colortype().ok() == Some(tiff::ColorType::CMYK(8)) {
        let (width, height) = decoder.dimensions().map_err(|error| error.to_string())?;
        validate_dimensions(width, height)?;
        let tiff::decoder::DecodingResult::U8(pixels) = decoder.read_image().map_err(|error| error.to_string())? else {
            return Err("invalid CMYK TIFF samples".into());
        };
        return Image::from_pixels(width, height, PixelMode::Cmyk, pixels, Some("TIFF".into()))
            .map(Some)
            .map_err(|error| error.to_string());
    }
    if decoder.colortype().ok() != Some(tiff::ColorType::Gray(32)) {
        return Ok(None);
    }
    let (width, height) = decoder.dimensions().map_err(|error| error.to_string())?;
    validate_dimensions(width, height)?;
    let tiff::decoder::DecodingResult::F32(values) = decoder.read_image().map_err(|error| error.to_string())? else {
        return Ok(None);
    };
    let mut image =
        Image::from_float_bytes(width, height, values.into_iter().flat_map(f32::to_le_bytes).collect()).map_err(|error| error.to_string())?;
    image.format = Some("TIFF".to_owned());
    Ok(Some(image))
}

fn lab_tiff_as_rgb(data: &[u8]) -> Result<Vec<u8>, String> {
    let little = data.starts_with(b"II");
    let read = |offset: usize, count: usize| -> Result<usize, String> {
        let bytes = data
            .get(offset..offset.checked_add(count).ok_or("invalid TIFF offset")?)
            .ok_or("truncated TIFF directory")?;
        let value = if little {
            bytes.iter().rev().fold(0_u64, |value, &byte| (value << 8) | u64::from(byte))
        } else {
            bytes.iter().fold(0_u64, |value, &byte| (value << 8) | u64::from(byte))
        };
        usize::try_from(value).map_err(|_| "invalid TIFF offset".into())
    };
    let big = read(2, 2)? == 43;
    let (offset, count_bytes, entry_bytes, value_offset) = if big { (read(8, 8)?, 8, 20, 12) } else { (read(4, 4)?, 2, 12, 8) };
    let count = read(offset, count_bytes)?;
    let start = offset.checked_add(count_bytes).ok_or("invalid TIFF offset")?;
    if count > data.len().saturating_sub(start) / entry_bytes {
        return Err("truncated TIFF directory".into());
    }
    for i in 0..count {
        let entry = start + i * entry_bytes;
        if read(entry, 2)? == 262 {
            if read(entry + 2, 2)? != 3 || read(entry + 4, if big { 8 } else { 4 })? != 1 {
                return Err("invalid TIFF photometric tag".into());
            }
            let mut normalized = data.to_vec();
            normalized[entry + value_offset..entry + value_offset + 2].copy_from_slice(&if little {
                2_u16.to_le_bytes()
            } else {
                2_u16.to_be_bytes()
            });
            return Ok(normalized);
        }
    }
    Err("missing TIFF photometric tag".into())
}

// DNGVersion (50706) belongs to the first classic TIFF IFD. Read only the
// bounded directory, never scan compressed image payloads for tag bytes.
// Reject DNG containers instead of opening their previews as ordinary TIFFs.
pub fn is_dng(data: &[u8]) -> bool {
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

/// Uncompressed little-endian TIFF with one strip preceding the directory.
pub fn encode_raw_tiff(image: &Image, pixels: &[u8]) -> PyResult<Vec<u8>> {
    if image.width == 0 || image.height == 0 {
        return Err(PyValueError::new_err("cannot encode an empty image"));
    }
    let (samples, bits, photometric) = match image.mode {
        PixelMode::L => (1_u16, 8_u16, 1_u16),
        PixelMode::F => (1, 32, 1),
        PixelMode::Rgb => (3, 8, 2),
        PixelMode::Rgba => (4, 8, 2),
        PixelMode::Cmyk => (4, 8, 5),
        _ => unreachable!("caller validates raw TIFF modes"),
    };
    let too_large = || PyValueError::new_err("image is too large for TIFF");
    let length = u32::try_from(pixels.len()).map_err(|_| too_large())?;
    let bits_offset = 8 + pixels.len().next_multiple_of(2);
    let ifd_offset = bits_offset + if samples > 2 { usize::from(samples) * 2 } else { 0 };
    let mut entries: Vec<(u16, u16, u32, u32)> = vec![
        (256, 4, 1, image.width),
        (257, 4, 1, image.height),
        (
            258,
            3,
            u32::from(samples),
            if samples > 2 {
                u32::try_from(bits_offset).map_err(|_| too_large())?
            } else {
                u32::from(bits)
            },
        ),
        (259, 3, 1, 1),
        (262, 3, 1, u32::from(photometric)),
        (273, 4, 1, 8),
        (277, 3, 1, u32::from(samples)),
        (278, 4, 1, image.height),
        (279, 4, 1, length),
        (284, 3, 1, 1),
    ];
    if image.mode == PixelMode::Rgba {
        entries.push((338, 3, 1, 2));
    }
    if image.mode == PixelMode::F {
        entries.push((339, 3, 1, 3));
    }
    let size = ifd_offset + 2 + entries.len() * 12 + 4;
    u32::try_from(size).map_err(|_| too_large())?;
    let mut output = Vec::with_capacity(size);
    output.extend_from_slice(b"II*\0");
    output.extend_from_slice(&(ifd_offset as u32).to_le_bytes());
    output.extend_from_slice(pixels);
    output.resize(bits_offset, 0);
    if samples > 2 {
        for _ in 0..samples {
            output.extend_from_slice(&bits.to_le_bytes());
        }
    }
    output.extend_from_slice(&(entries.len() as u16).to_le_bytes());
    for (tag, kind, count, value) in entries {
        output.extend_from_slice(&tag.to_le_bytes());
        output.extend_from_slice(&kind.to_le_bytes());
        output.extend_from_slice(&count.to_le_bytes());
        if kind == 3 && count == 1 {
            output.extend_from_slice(&(value as u16).to_le_bytes());
            output.extend_from_slice(&[0, 0]);
        } else {
            output.extend_from_slice(&value.to_le_bytes());
        }
    }
    output.extend_from_slice(&[0; 4]);
    Ok(output)
}

pub fn decode_tiff(data: &[u8]) -> Result<Image, String> {
    decode_special_tiff(data)?.map_or_else(|| decode_rust_image(data, RustFormat::Tiff, "TIFF"), Ok)
}
pub fn encode_lab_tiff(image: &Image) -> PyResult<Vec<u8>> {
    if image.width == 0 || image.height == 0 {
        return Err(PyValueError::new_err("cannot encode an empty image"));
    }
    let mut output = Cursor::new(Vec::new());
    {
        let mut encoder = tiff::encoder::TiffEncoder::new(&mut output).map_err(codec_error)?;
        let mut frame = encoder
            .new_image::<tiff::encoder::colortype::RGB8>(image.width, image.height)
            .map_err(codec_error)?;
        frame
            .encoder()
            .write_tag(tiff::tags::Tag::PhotometricInterpretation, 8_u16)
            .map_err(codec_error)?;
        frame
            .write_data(&blanket_core::raster::lab_raw_bytes(image.pixel_data()?))
            .map_err(codec_error)?;
    }
    return Ok(output.into_inner());
}

pub fn encode_float_tiff(image: &Image) -> PyResult<Vec<u8>> {
    if image.width == 0 || image.height == 0 {
        return Err(PyValueError::new_err("cannot encode an empty image"));
    }
    if cfg!(target_endian = "little") {
        return encode_raw_tiff(image, image.raw_data()?);
    }
    let samples: Vec<f32> = image
        .raw_data()?
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| f32::from_le_bytes(*bytes))
        .collect();
    let mut output = Cursor::new(Vec::new());
    tiff::encoder::TiffEncoder::new(&mut output)
        .map_err(codec_error)?
        .write_image::<tiff::encoder::colortype::Gray32Float>(image.width, image.height, &samples)
        .map_err(codec_error)?;
    return Ok(output.into_inner());
}

pub fn encode_tiff(image: &Image, pixels: &[u8]) -> PyResult<Vec<u8>> {
    if matches!(image.mode, PixelMode::L | PixelMode::Rgb | PixelMode::Rgba) {
        return encode_raw_tiff(image, pixels);
    }
    let mut output = Cursor::new(Vec::with_capacity(pixels.len().saturating_add(1024)));
    image::codecs::tiff::TiffEncoder::new(&mut output)
        .write_image(pixels, image.width, image.height, color_type(image.mode))
        .map_err(codec_error)?;
    Ok(output.into_inner())
}

pub fn encode_wide_tiff(image: &Image, samples: Vec<u16>) -> PyResult<Vec<u8>> {
    let data: Vec<u8> = samples.into_iter().flat_map(u16::to_ne_bytes).collect();
    let color = match image.mode {
        PixelMode::L => ExtendedColorType::L16,
        PixelMode::Rgb => ExtendedColorType::Rgb16,
        PixelMode::Rgba => ExtendedColorType::Rgba16,
        _ => return Err(PyValueError::new_err("unsupported high-bit-depth mode")),
    };
    let mut output = Cursor::new(Vec::new());
    image::codecs::tiff::TiffEncoder::new(&mut output)
        .write_image(&data, image.width, image.height, color)
        .map_err(codec_error)?;
    return Ok(output.into_inner());
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn lab_tiff_directory_bounds_and_byte_order() {
        for little in [false, true] {
            for big in [false, true] {
                let mut data = if little { b"II".to_vec() } else { b"MM".to_vec() };
                let push = |data: &mut Vec<u8>, value: u64, count: usize| {
                    if little {
                        data.extend_from_slice(&value.to_le_bytes()[..count]);
                    } else {
                        data.extend_from_slice(&value.to_be_bytes()[8 - count..]);
                    }
                };
                push(&mut data, if big { 43 } else { 42 }, 2);
                if big {
                    push(&mut data, 8, 2);
                    push(&mut data, 0, 2);
                }
                push(&mut data, if big { 16 } else { 8 }, if big { 8 } else { 4 });
                push(&mut data, 1, if big { 8 } else { 2 });
                push(&mut data, 262, 2);
                push(&mut data, 3, 2);
                push(&mut data, 1, if big { 8 } else { 4 });
                let value_offset = data.len();
                push(&mut data, 8, 2);
                push(&mut data, 0, if big { 6 } else { 2 });
                let mut expected = data.clone();
                expected[value_offset..value_offset + 2].copy_from_slice(&if little { 2_u16.to_le_bytes() } else { 2_u16.to_be_bytes() });
                assert_eq!(lab_tiff_as_rgb(&data).unwrap(), expected);
                for length in 0..data.len() {
                    assert!(lab_tiff_as_rgb(&data[..length]).is_err());
                }
            }
        }
    }
}
