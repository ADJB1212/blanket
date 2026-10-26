#![feature(default_field_values)]

use blanket_core::Image;
use pyo3::exceptions::PyOSError;
use pyo3::prelude::*;
use zerocopy::IntoBytes;

pub const MAX_IMAGE_PIXELS: usize = 178_956_970;

#[expect(unevaluated_default_field_value)]
pub struct Frame<I = Image> {
    pub image: I,
    pub duration: Option<f64> = None,
    pub disposal: u8 = 0,
    pub blend: u8 = 0,
}

#[expect(unevaluated_default_field_value)]
pub struct Sequence<I = Image> {
    pub frames: Vec<Frame<I>> = Vec::new(),
    pub loop_count: Option<u32> = None,
    pub default_image: bool = false,
}

impl<I> Default for Sequence<I> {
    fn default() -> Self {
        Self { .. }
    }
}

impl Sequence {
    pub fn single(image: Image) -> Self {
        Self { frames: vec![Frame { image, .. }], .. }
    }
}

pub fn validate_frame_budget(total: &mut usize, width: u32, height: u32) -> Result<(), String> {
    validate_dimensions(width, height)?;
    *total = total.checked_add((width as usize) * (height as usize)).ok_or("frame pixel count overflow")?;
    if *total > MAX_IMAGE_PIXELS {
        return Err(format!("sequence exceeds Blanket limit of {MAX_IMAGE_PIXELS} decoded pixels"));
    }
    Ok(())
}

#[derive(Clone, Copy, Default)]
pub struct SaveOptions<'a> {
    pub quality: u8 = 90,
    pub compress_level: u8 = 6,
    pub lossless: bool = false,
    pub effort: u8 = 7,
    pub tiff_compression: u8 = 0,
    pub icc_profile: &'a [u8] = &[],
    pub exif: &'a [u8] = &[],
    pub xmp: &'a [u8] = &[],
}

pub fn validate_dimensions(width: u32, height: u32) -> Result<(), String> {
    let pixels = (width as usize).checked_mul(height as usize).ok_or_else(|| "image dimensions overflow addressable memory".to_owned())?;
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
    samples.as_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_and_sequence_defaults_do_not_require_default_images() {
        struct Raster;

        let frame = Frame { image: Raster, .. };
        assert_eq!(frame.duration, None);
        assert_eq!(frame.disposal, 0);
        assert_eq!(frame.blend, 0);

        for sequence in [Sequence::<Raster> { .. }, Sequence::<Raster>::default()] {
            assert!(sequence.frames.is_empty());
            assert_eq!(sequence.loop_count, None);
            assert!(!sequence.default_image);
        }
        let sequence = Sequence { frames: vec![frame], loop_count: Some(3), default_image: true };
        assert_eq!(sequence.frames.len(), 1);
        assert_eq!(sequence.loop_count, Some(3));
        assert!(sequence.default_image);
    }

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
