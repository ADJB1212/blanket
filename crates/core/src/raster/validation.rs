use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::PixelMode;

pub(super) fn validate_raw_data(width: u32, height: u32, mode: PixelMode, bit_depth: u8, pixels: &[u8]) -> PyResult<()> {
    let sample_bytes = if mode.is_wide_scalar() {
        if bit_depth as usize != mode.sample_bytes() * 8 {
            return Err(PyValueError::new_err("invalid sample bit depth"));
        }
        mode.sample_bytes()
    } else if bit_depth == 8 {
        1
    } else if matches!(mode, PixelMode::L | PixelMode::Rgb | PixelMode::Rgba) && matches!(bit_depth, 10 | 12 | 16) {
        2
    } else {
        return Err(PyValueError::new_err("invalid sample bit depth"));
    };
    let expected =
        expected_len(width, height, mode)?.checked_mul(sample_bytes).ok_or_else(|| PyValueError::new_err("image dimensions are too large"))?;
    if pixels.len() != expected || matches!(bit_depth, 10 | 12) && !crate::pixels::samples_in_range(pixels, bit_depth) {
        return Err(PyValueError::new_err("invalid image data"));
    }
    Ok(())
}

pub(super) fn expected_len(width: u32, height: u32, mode: PixelMode) -> PyResult<usize> {
    let pixels = (width as usize).checked_mul(height as usize).ok_or_else(|| PyValueError::new_err("image dimensions are too large"))?;
    pixels.checked_mul(mode.channels()).ok_or_else(|| PyValueError::new_err("image dimensions are too large"))
}
