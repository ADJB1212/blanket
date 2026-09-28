use pyo3::exceptions::PyOSError;
use pyo3::prelude::*;

const MAX_IMAGE_PIXELS: usize = 178_956_970;

#[derive(Clone, Copy)]
pub struct SaveOptions {
    pub quality: u8,
    pub compress_level: u8,
    pub lossless: bool,
    pub effort: u8,
}

pub fn validate_dimensions(width: u32, height: u32) -> Result<(), String> {
    let pixels = (width as usize)
        .checked_mul(height as usize)
        .ok_or_else(|| "image dimensions overflow addressable memory".to_owned())?;
    if pixels > MAX_IMAGE_PIXELS {
        return Err(format!("image size ({pixels} pixels) exceeds Blanket limit of {MAX_IMAGE_PIXELS} pixels"));
    }
    Ok(())
}

pub fn codec_error(error: impl std::fmt::Display) -> PyErr {
    PyOSError::new_err(error.to_string())
}
