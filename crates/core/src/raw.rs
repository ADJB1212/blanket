use std::borrow::Cow;

use pyo3::exceptions::{PyMemoryError, PyValueError};
use pyo3::prelude::*;

use crate::PixelMode;

struct Layout {
    image_row: usize,
    raw_row: usize,
    stride: usize,
    bgr_channels: Option<usize>,
}

impl Layout {
    fn new(mode: PixelMode, width: usize, depth: u8, rawmode: Option<&str>, stride: usize, orientation: i8) -> PyResult<Self> {
        if !matches!(orientation, -1 | 1) {
            return Err(PyValueError::new_err("orientation must be 1 or -1"));
        }
        let rawmode = rawmode.unwrap_or(mode.as_str());
        let bgr_channels = match (mode, rawmode, depth) {
            (PixelMode::Rgb, "BGR", 8) => Some(3),
            (PixelMode::Rgb, "BGRX", 8) | (PixelMode::Rgba, "BGRA", 8) => Some(4),
            _ if rawmode == mode.as_str() || mode == PixelMode::L && rawmode == "P" => None,
            _ => return Err(PyValueError::new_err(format!("unsupported raw mode {rawmode} for {} at {depth} bits", mode.as_str()))),
        };
        let sample_bytes = if mode.is_float() {
            4
        } else if mode.is_integer() {
            mode.sample_bytes()
        } else if depth > 8 {
            2
        } else {
            1
        };
        let image_row = if mode == PixelMode::One {
            width.div_ceil(8)
        } else {
            width.checked_mul(mode.channels() * sample_bytes).ok_or_else(|| PyValueError::new_err("raw row size overflow"))?
        };
        let raw_row = match bgr_channels {
            Some(channels) => width.checked_mul(channels).ok_or_else(|| PyValueError::new_err("raw row size overflow"))?,
            None => image_row,
        };
        let stride = if stride == 0 { raw_row } else { stride };
        if stride < raw_row {
            return Err(PyValueError::new_err("stride must be at least the packed row size"));
        }
        Ok(Self { image_row, raw_row, stride, bgr_channels })
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn transform<'a>(
    data: &'a [u8], mode: PixelMode, size: (u32, u32), depth: u8, rawmode: Option<&str>, stride: usize, orientation: i8, encode: bool,
) -> PyResult<Cow<'a, [u8]>> {
    let layout = Layout::new(mode, size.0 as usize, depth, rawmode, stride, orientation)?;
    if layout.bgr_channels.is_none() && layout.stride == layout.image_row && orientation == 1 {
        return Ok(Cow::Borrowed(data));
    }
    let height = size.1 as usize;
    let (input_row, input_stride, output_stride) =
        if encode { (layout.image_row, layout.image_row, layout.stride) } else { (layout.raw_row, layout.stride, layout.image_row) };
    let required = if height == 0 { Some(0) } else { (height - 1).checked_mul(input_stride).and_then(|v| v.checked_add(input_row)) }
        .ok_or_else(|| PyValueError::new_err("raw buffer size overflow"))?;
    if data.len() < required {
        return Err(PyValueError::new_err("not enough image data"));
    }
    let length = output_stride.checked_mul(height).ok_or_else(|| PyValueError::new_err("raw buffer size overflow"))?;
    let mut output = Vec::new();
    output.try_reserve_exact(length).map_err(|error| PyMemoryError::new_err(error.to_string()))?;
    output.resize(length, 0);
    for row in 0..height {
        let source_row = if orientation == -1 { height - row - 1 } else { row };
        let source = &data[source_row * input_stride..source_row * input_stride + input_row];
        let target_row = if encode { layout.raw_row } else { layout.image_row };
        let target = &mut output[row * output_stride..row * output_stride + target_row];
        if let Some(channels) = layout.bgr_channels {
            let (input_channels, output_channels) = if encode { (mode.channels(), channels) } else { (channels, mode.channels()) };
            for (src, dst) in source.chunks_exact(input_channels).zip(target.chunks_exact_mut(output_channels)) {
                dst[..3].copy_from_slice(&[src[2], src[1], src[0]]);
                if output_channels == 4 {
                    dst[3] = if input_channels == 4 { src[3] } else { 0 };
                }
            }
        } else {
            target.copy_from_slice(source);
        }
    }
    Ok(Cow::Owned(output))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn padded_unaligned_bgr_rows_ignore_guards_and_final_padding() {
        let data = [99, 1, 2, 3, 4, 5, 6, 88, 88, 7, 8, 9, 10, 11, 12, 99];
        let pixels = transform(&data[1..15], PixelMode::Rgb, (2, 2), 8, Some("BGR"), 8, -1, false).unwrap();
        assert_eq!(pixels.as_ref(), &[9, 8, 7, 12, 11, 10, 3, 2, 1, 6, 5, 4]);
        let encoded = transform(&pixels, PixelMode::Rgb, (2, 2), 8, Some("BGR"), 8, -1, true).unwrap();
        assert_eq!(encoded.as_ref(), &[1, 2, 3, 4, 5, 6, 0, 0, 7, 8, 9, 10, 11, 12, 0, 0]);
        assert_eq!(data[0], 99);
        assert_eq!(data[15], 99);
    }

    #[test]
    fn raw_layout_rejects_truncation_and_overflow_before_allocation() {
        assert!(transform(&[0; 13], PixelMode::Rgb, (2, 2), 8, Some("BGR"), 8, -1, false).is_err());
        assert!(transform(&[0; 12], PixelMode::Rgb, (2, 2), 8, None, usize::MAX, 1, true).is_err());
        assert!(transform(&[0; 12], PixelMode::Rgb, (2, 2), 8, None, 5, 1, false).is_err());
        assert!(transform(&[0; 12], PixelMode::Rgb, (2, 2), 8, None, 6, 0, false).is_err());
    }
}
