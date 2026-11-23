#![feature(portable_simd)]

use blanket_codec_common::{DecodeError, codec_error};
use blanket_codec_image::{color_type, decode_rust_image};
use blanket_core::{Image, PixelMode};
use image::{ImageEncoder, ImageFormat as RustFormat};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

const PNG_SIGNATURE: &[u8] = b"\x89PNG\r\n\x1a\n";

pub fn inspect_bmp(data: &[u8]) -> Result<blanket_codec_common::ImageHeader, String> {
    blanket_codec_image::inspect_rust_image(data, RustFormat::Bmp)
}

pub fn inspect_ico(data: &[u8]) -> Result<blanket_codec_common::ImageHeader, String> {
    let (entry, payload) = ico_entry(data)?;
    if payload.starts_with(PNG_SIGNATURE) {
        let header = blanket_codec_png::inspect(payload)?;
        let dimension = |v: u8| if v == 0 { 256 } else { u32::from(v) };
        if (header.width, header.height) != (dimension(entry[0]), dimension(entry[1])) {
            return Err("ICO directory and PNG dimensions differ".into());
        }
        Ok(header)
    } else {
        let dib = payload.get(..40).ok_or("truncated ICO bitmap header")?;
        let width = if entry[0] == 0 { 256 } else { u32::from(entry[0]) };
        let height = if entry[1] == 0 { 256 } else { u32::from(entry[1]) };
        let size = u32::from_le_bytes(dib[..4].try_into().unwrap()) as usize;
        if size < 40
            || size > payload.len()
            || i32::from_le_bytes(dib[4..8].try_into().unwrap()) != width as i32
            || i32::from_le_bytes(dib[8..12].try_into().unwrap()) != (height * 2) as i32
            || u16::from_le_bytes([dib[12], dib[13]]) != 1
            || !matches!(u16::from_le_bytes([dib[14], dib[15]]), 1 | 4 | 8 | 16 | 24 | 32)
        {
            return Err("invalid ICO bitmap header".into());
        }
        blanket_codec_common::ImageHeader::new(width, height, "RGBA", 8)
    }
}

fn ico_entry(data: &[u8]) -> Result<(&[u8; 16], &[u8]), String> {
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
    Ok((entry, payload))
}

pub fn decode_ico(data: &[u8]) -> Result<Image, DecodeError> {
    let (entry, payload) = ico_entry(data)?;
    let dimension = |v: u8| if v == 0 { 256_u32 } else { u32::from(v) };
    if payload.starts_with(PNG_SIGNATURE) {
        let mut image = blanket_codec_png::decode_png(payload)?;
        image.format = Some("ICO".into());
        if image.width() != dimension(entry[0]) || image.height() != dimension(entry[1]) {
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

pub fn decode_bmp(data: &[u8]) -> Result<Image, DecodeError> {
    if let Some(image) = decode_direct_bmp(data) {
        return image;
    }
    decode_rust_image(data, RustFormat::Bmp, "BMP")
}

/// Uncompressed 24/32-bit layouts with byte-aligned BGR(A) samples. Other
/// layouts, and any file the generic decoder would reject, return `None`.
fn decode_direct_bmp(data: &[u8]) -> Option<Result<Image, DecodeError>> {
    let u16_at = |offset: usize| Some(u16::from_le_bytes(data.get(offset..offset + 2)?.try_into().ok()?));
    let u32_at = |offset: usize| Some(u32::from_le_bytes(data.get(offset..offset + 4)?.try_into().ok()?));
    if data.get(..2)? != b"BM" {
        return None;
    }
    let offset = u32_at(10)? as usize;
    let header = u32_at(14)?;
    if !matches!(header, 40 | 52 | 56 | 108 | 124) {
        return None;
    }
    let width = i32::from_le_bytes(data.get(18..22)?.try_into().ok()?);
    let height = i32::from_le_bytes(data.get(22..26)?.try_into().ok()?);
    if !(1..=0xffff).contains(&width) || height == 0 || height.unsigned_abs() > 0xffff || u16_at(26)? != 1 {
        return None;
    }
    let (bits, compression) = (u16_at(28)?, u32_at(30)?);
    let alpha = match (bits, compression) {
        (24 | 32, 0) => false,
        (32, 3) => {
            if (u32_at(54)?, u32_at(58)?, u32_at(62)?) != (0x00ff_0000, 0x0000_ff00, 0x0000_00ff) {
                return None;
            }
            match if header >= 56 { u32_at(66)? } else { 0 } {
                0 => false,
                0xff00_0000 => true,
                _ => return None,
            }
        }
        _ => return None,
    };
    let (width, height, top_down) = (width as usize, height.unsigned_abs() as usize, height < 0);
    let source_channels = usize::from(bits / 8);
    let stride = (width * source_channels).div_ceil(4) * 4;
    let rows = data.get(offset..offset.checked_add(stride.checked_mul(height)?)?)?;
    if let Err(error) = blanket_codec_common::validate_dimensions(width as u32, height as u32) {
        return Some(Err(error.into()));
    }
    let (mode, channels) = if alpha { (PixelMode::Rgba, 4) } else { (PixelMode::Rgb, 3) };
    let row_bytes = width * channels;
    let mut pixels = match blanket_core::buffers::reserved(row_bytes * height) {
        Ok(pixels) => pixels,
        Err(error) => return Some(Err(DecodeError::with_source(error))),
    };
    pixels.resize(row_bytes * height, 0);
    blanket_core::parallel::chunks_mut_above(&mut pixels, row_bytes, 2 << 20, |y, row| {
        let source = &rows[if top_down { y } else { height - 1 - y } * stride..][..width * source_channels];
        match (source_channels, channels) {
            (3, _) => bgr_to_rgb::<3, 3>(source, row),
            (4, 3) => bgr_to_rgb::<4, 3>(source, row),
            _ => bgr_to_rgb::<4, 4>(source, row),
        }
    });
    Some(Image::from_pixels(width as u32, height as u32, mode, pixels, Some("BMP".into())).map_err(DecodeError::with_source))
}

fn bgr_to_rgb<const I: usize, const O: usize>(source: &[u8], output: &mut [u8]) {
    use blanket_core::pixels::{Bytes, load, store};
    let end = output.len() / O / 16 * 16;
    for x in (0..end).step_by(16) {
        let p = load::<I>(&source[x * I..]);
        let mut channels = [Bytes::splat(0); O];
        channels[..3].copy_from_slice(&[p[2], p[1], p[0]]);
        if O == 4 {
            channels[3] = p[3];
        }
        store::<O>(&mut output[x * O..], channels);
    }
    for (src, dst) in source[end * I..].as_chunks::<I>().0.iter().zip(output[end * O..].as_chunks_mut::<O>().0) {
        dst[..3].copy_from_slice(&[src[2], src[1], src[0]]);
        if O == 4 {
            dst[3] = src[3];
        }
    }
}

fn validate_encode_input(image: &Image, pixels: &[u8]) -> PyResult<()> {
    if image.width() == 0 || image.height() == 0 {
        return Err(PyValueError::new_err("cannot encode an empty image"));
    }
    if !matches!(image.pixel_mode(), PixelMode::One | PixelMode::L | PixelMode::La | PixelMode::Rgb | PixelMode::Rgba) {
        return Err(PyValueError::new_err("unsupported mode for bitmap encoding"));
    }
    if pixels.len() != image.pixel_data()?.len() {
        return Err(PyValueError::new_err("invalid bitmap input buffer"));
    }
    Ok(())
}

pub fn encode_bmp(image: &Image, pixels: &[u8]) -> PyResult<Vec<u8>> {
    validate_encode_input(image, pixels)?;
    if matches!(image.pixel_mode(), PixelMode::Rgb | PixelMode::Rgba)
        && let Some(output) = encode_direct_bmp(image, pixels)
    {
        return Ok(output);
    }
    let mut output = Vec::new();
    image::codecs::bmp::BmpEncoder::new(&mut output)
        .write_image(pixels, image.width(), image.height(), color_type(image.pixel_mode()).map_err(codec_error)?)
        .map_err(codec_error)?;
    Ok(output)
}
/// Byte-identical to the generic encoder: 24-bit `BI_RGB` with an info header,
/// or 32-bit BGRA bitfields with a V4 sRGB header; rows are bottom-up.
fn encode_direct_bmp(image: &Image, pixels: &[u8]) -> Option<Vec<u8>> {
    let alpha = image.pixel_mode() == PixelMode::Rgba;
    let (channels, dib) = if alpha { (4, 108_u32) } else { (3, 40) };
    let (width, height) = (image.width() as usize, image.height() as usize);
    let stride = (width * channels).div_ceil(4) * 4;
    let image_size = u32::try_from(stride.checked_mul(height)?).ok()?;
    let offset = 14 + dib;
    let file_size = offset.checked_add(image_size)?;
    let mut output = Vec::new();
    output.try_reserve_exact(file_size as usize).ok()?;
    output.extend_from_slice(b"BM");
    for value in [
        file_size,
        0,
        offset,
        dib,
        image.width(),
        image.height(),
        1 | ((channels as u32 * 8) << 16),
        if alpha { 3 } else { 0 },
        image_size,
        0,
        0,
        0,
        0,
    ] {
        output.extend_from_slice(&value.to_le_bytes());
    }
    if alpha {
        for value in [0x00ff_0000_u32, 0x0000_ff00, 0x0000_00ff, 0xff00_0000, 0x7352_4742] {
            output.extend_from_slice(&value.to_le_bytes());
        }
        output.extend_from_slice(&[0; 48]);
    }
    let header = output.len();
    output.resize(file_size as usize, 0);
    blanket_core::parallel::chunks_mut_above(&mut output[header..], stride, 2 << 20, |row, dst| {
        let src = &pixels[(height - 1 - row) * width * channels..][..width * channels];
        let dst = &mut dst[..width * channels];
        if alpha { bgr_to_rgb::<4, 4>(src, dst) } else { bgr_to_rgb::<3, 3>(src, dst) }
    });
    Some(output)
}

pub fn encode_ico(image: &Image, pixels: &[u8]) -> PyResult<Vec<u8>> {
    validate_encode_input(image, pixels)?;
    if image.width() > 256 || image.height() > 256 {
        return Err(PyValueError::new_err("ICO dimensions must be between 1 and 256"));
    }
    let mut output = Vec::new();
    image::codecs::ico::IcoEncoder::new(&mut output)
        .write_image(pixels, image.width(), image.height(), color_type(image.pixel_mode()).map_err(codec_error)?)
        .map_err(codec_error)?;
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encoders_reject_unsupported_modes_and_depths_without_panicking() {
        Python::initialize();
        let mut images = Vec::new();
        for mode in [PixelMode::F, PixelMode::I, PixelMode::I16, PixelMode::RgbF, PixelMode::RgbaF, PixelMode::Pa, PixelMode::Cmyk, PixelMode::Hsv] {
            images.push(Image::from_raw_bytes(1, 1, mode, vec![0; mode.channels() * mode.sample_bytes()], (mode.sample_bytes() * 8) as u8).unwrap());
        }
        for mode in [PixelMode::L, PixelMode::Rgb, PixelMode::Rgba] {
            images.push(Image::from_samples(1, 1, mode, vec![0; mode.channels()], 16, None).unwrap());
        }
        for image in images {
            for encode in [encode_bmp, encode_ico] {
                let error = encode(&image, image.raw_data().unwrap()).unwrap_err();
                Python::attach(|py| assert!(error.is_instance_of::<PyValueError>(py)));
            }
        }
    }

    fn generic_encode(image: &Image, pixels: &[u8]) -> Vec<u8> {
        let mut output = Vec::new();
        image::codecs::bmp::BmpEncoder::new(&mut output)
            .write_image(pixels, image.width(), image.height(), color_type(image.pixel_mode()).unwrap())
            .unwrap();
        output
    }

    #[test]
    fn direct_bmp_matches_generic_codec() {
        Python::initialize();
        for (mode, channels) in [(PixelMode::Rgb, 3), (PixelMode::Rgba, 4)] {
            for (width, height) in [(1_usize, 1_usize), (5, 3), (15, 2), (16, 4), (17, 3), (33, 5), (64, 2)] {
                let pixels: Vec<u8> = (0..width * height * channels).map(|i| (i * 37 + i / 7) as u8).collect();
                let image = Image::from_pixels(width as u32, height as u32, mode, pixels.clone(), None).unwrap();
                let encoded = encode_bmp(&image, &pixels).unwrap();
                assert_eq!(encoded, generic_encode(&image, &pixels), "{mode:?} {width}x{height}");
                let direct = decode_direct_bmp(&encoded).unwrap().unwrap();
                let generic = decode_rust_image(&encoded, RustFormat::Bmp, "BMP").unwrap();
                assert_eq!(direct.pixel_mode(), generic.pixel_mode());
                assert_eq!(direct.pixel_data().unwrap(), generic.pixel_data().unwrap());
                assert_eq!(direct.pixel_data().unwrap(), pixels.as_slice());

                // Top-down rows and 32-bit BI_RGB, which drops the fourth byte.
                let mut flipped = encoded.clone();
                flipped[22..26].copy_from_slice(&(-(height as i32)).to_le_bytes());
                let direct = decode_direct_bmp(&flipped).unwrap().unwrap();
                assert_eq!(direct.pixel_data().unwrap(), decode_rust_image(&flipped, RustFormat::Bmp, "BMP").unwrap().pixel_data().unwrap());
                if channels == 4 {
                    let mut rgb32 = encoded.clone();
                    rgb32[30..34].copy_from_slice(&0_u32.to_le_bytes());
                    let direct = decode_direct_bmp(&rgb32).unwrap().unwrap();
                    let generic = decode_rust_image(&rgb32, RustFormat::Bmp, "BMP").unwrap();
                    assert_eq!((direct.pixel_mode(), direct.pixel_data().unwrap()), (generic.pixel_mode(), generic.pixel_data().unwrap()));
                }
                assert!(decode_direct_bmp(&encoded[..encoded.len() - 1]).is_none());
            }
        }
    }

    #[test]
    fn encoders_reject_mismatched_pixel_buffers() {
        Python::initialize();
        let image = Image::from_pixels(2, 1, PixelMode::Rgb, vec![0; 6], None).unwrap();
        for encode in [encode_bmp, encode_ico] {
            for length in [0, 5, 7] {
                let error = encode(&image, &vec![0; length]).unwrap_err();
                Python::attach(|py| assert!(error.is_instance_of::<PyValueError>(py)));
            }
        }
    }
}
