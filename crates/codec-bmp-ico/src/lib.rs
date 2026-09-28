use blanket_codec_common::codec_error;
use blanket_codec_image::{color_type, decode_rust_image};
use blanket_core::Image;
use image::{ImageEncoder, ImageFormat as RustFormat};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

const PNG_SIGNATURE: &[u8] = b"\x89PNG\r\n\x1a\n";

pub fn decode_ico(data: &[u8]) -> Result<Image, String> {
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
        let mut image = blanket_codec_png::decode_png(payload)?;
        image.format = Some("ICO".into());
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

pub fn decode_bmp(data: &[u8]) -> Result<Image, String> {
    decode_rust_image(data, RustFormat::Bmp, "BMP")
}

pub fn encode_bmp(image: &Image, pixels: &[u8]) -> PyResult<Vec<u8>> {
    if image.width == 0 || image.height == 0 {
        return Err(PyValueError::new_err("cannot encode an empty image"));
    }
    let mut output = Vec::new();
    image::codecs::bmp::BmpEncoder::new(&mut output)
        .write_image(pixels, image.width, image.height, color_type(image.mode))
        .map_err(codec_error)?;
    Ok(output)
}
pub fn encode_ico(image: &Image, pixels: &[u8]) -> PyResult<Vec<u8>> {
    if image.width == 0 || image.height == 0 {
        return Err(PyValueError::new_err("cannot encode an empty image"));
    }
    if image.width > 256 || image.height > 256 {
        return Err(PyValueError::new_err("ICO dimensions must be between 1 and 256"));
    }
    let mut output = Vec::new();
    image::codecs::ico::IcoEncoder::new(&mut output)
        .write_image(pixels, image.width, image.height, color_type(image.mode))
        .map_err(codec_error)?;
    Ok(output)
}
