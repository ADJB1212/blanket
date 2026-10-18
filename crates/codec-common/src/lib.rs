#![feature(default_field_values)]

use pyo3::exceptions::PyOSError;
use pyo3::prelude::*;

const MAX_IMAGE_PIXELS: usize = 178_956_970;

#[derive(Clone, Copy, Default)]
pub struct SaveOptions {
    pub quality: u8 = 90,
    pub compress_level: u8 = 6,
    pub lossless: bool = false,
    pub effort: u8 = 7,
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

/// Borrow native-endian sample bytes without copying the sample buffer.
pub fn u16_bytes(samples: &[u16]) -> &[u8] {
    // SAFETY: u16 has no padding, every byte is initialized, and the returned
    // slice shares the immutable borrow and lifetime of the source samples.
    unsafe { std::slice::from_raw_parts(samples.as_ptr().cast::<u8>(), std::mem::size_of_val(samples)) }
}
