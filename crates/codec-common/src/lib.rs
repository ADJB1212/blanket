#![feature(default_field_values)]

use blanket_core::Image;
use pyo3::exceptions::PyOSError;
use pyo3::prelude::*;

pub const MAX_IMAGE_PIXELS: usize = 178_956_970;

pub struct Frame {
    pub image: Image,
    pub duration: Option<f64>,
    pub disposal: u8,
    pub blend: u8,
}

#[derive(Default)]
pub struct Sequence {
    pub frames: Vec<Frame>,
    pub loop_count: Option<u32>,
    pub default_image: bool,
}

impl Sequence {
    pub fn single(image: Image) -> Self {
        Self {
            frames: vec![Frame {
                image,
                duration: None,
                disposal: 0,
                blend: 0,
            }],
            ..Self::default()
        }
    }
}

pub fn validate_frame_budget(total: &mut usize, width: u32, height: u32) -> Result<(), String> {
    validate_dimensions(width, height)?;
    *total = total
        .checked_add((width as usize) * (height as usize))
        .ok_or("frame pixel count overflow")?;
    if *total > MAX_IMAGE_PIXELS {
        return Err(format!("sequence exceeds Blanket limit of {MAX_IMAGE_PIXELS} decoded pixels"));
    }
    Ok(())
}

#[derive(Clone, Copy, Default)]
pub struct SaveOptions {
    pub quality: u8 = 90,
    pub compress_level: u8 = 6,
    pub lossless: bool = false,
    pub effort: u8 = 7,
    pub tiff_compression: u8 = 0,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_budget_checks_cumulative_pixels_and_overflow() {
        let mut total = MAX_IMAGE_PIXELS - 2;
        assert!(validate_frame_budget(&mut total, 2, 1).is_ok());
        assert!(validate_frame_budget(&mut total, 1, 1).is_err());
        let mut overflowing = usize::MAX;
        assert!(validate_frame_budget(&mut overflowing, 1, 1).is_err());
        assert!(validate_frame_budget(&mut 0, u32::MAX, u32::MAX).is_err());
    }
}
