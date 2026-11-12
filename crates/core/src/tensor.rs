//! Fused float32 tensor snapshots in contiguous HWC or CHW order.

use std::simd::{Simd, ToBytes};

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyByteArray;

use crate::{Image, PixelMode};

#[pyfunction]
pub fn tensor_export<'py>(
    py: Python<'py>, image: &Image, layout: &str, normalize: bool, mean: Option<Vec<f32>>, std: Option<Vec<f32>>,
) -> PyResult<Bound<'py, PyByteArray>> {
    let channels = image.mode.channels();
    let planar = match layout {
        "CHW" => true,
        "HWC" => false,
        "HW" if channels == 1 => false,
        _ => return Err(PyValueError::new_err("layout must be HWC, CHW, or HW for a single-band image")),
    };
    let mean = parameters(mean, channels, 0.0, "mean")?;
    let std = parameters(std, channels, 1.0, "std")?;
    if std.iter().any(|&value| value <= 0.0) {
        return Err(PyValueError::new_err("std values must be positive"));
    }
    let source = image.raw_data()?;
    let pixels = image.width as usize * image.height as usize;
    let len = pixels.checked_mul(channels).and_then(|n| n.checked_mul(4)).ok_or_else(|| PyValueError::new_err("tensor dimensions are too large"))?;
    let divisor = if normalize && !image.mode.is_float() && image.mode != PixelMode::I { ((1_u32 << image.bit_depth) - 1) as f32 } else { 1.0 };
    let bytes = image.raw_bytes_per_pixel() / channels;
    let read = |index: usize| -> f32 {
        let start = index * bytes;
        match bytes {
            4 => {
                let value = source[start..start + 4].try_into().unwrap();
                if image.mode.is_float() { f32::from_le_bytes(value) } else { i32::from_le_bytes(value) as f32 }
            }
            2 => {
                let value = [source[start], source[start + 1]];
                if image.mode.is_big_endian() { u16::from_be_bytes(value) as f32 } else { u16::from_le_bytes(value) as f32 }
            }
            _ => {
                let value = source[start];
                if image.mode == PixelMode::One {
                    if value == 0 { 0.0 } else { 255.0 }
                } else if image.mode == PixelMode::Lab && !index.is_multiple_of(channels) {
                    (value ^ 128) as f32
                } else {
                    value as f32
                }
            }
        }
    };
    PyByteArray::new_with(py, len, |output| {
        py.detach(|| {
            const CHUNK: usize = 16384;
            crate::parallel::chunks_mut(output, CHUNK * 4, |chunk, output| {
                let start = chunk * CHUNK;
                let location = |index: usize| {
                    if planar { (index % pixels * channels + index / pixels, index / pixels) } else { (index, index % channels) }
                };
                let end = output.len() / 64 * 64;
                for (block, dst) in output[..end].as_chunks_mut::<64>().0.iter_mut().enumerate() {
                    let locations: [_; 16] = std::array::from_fn(|lane| location(start + block * 16 + lane));
                    let values = Simd::from_array(locations.map(|(index, _)| read(index)));
                    let means = Simd::from_array(locations.map(|(_, channel)| mean[channel]));
                    let stds = Simd::from_array(locations.map(|(_, channel)| std[channel]));
                    *dst = ((values / Simd::splat(divisor) - means) / stds).to_ne_bytes().to_array();
                }
                for (offset, dst) in output[end..].as_chunks_mut::<4>().0.iter_mut().enumerate() {
                    let (index, channel) = location(start + end / 4 + offset);
                    *dst = ((read(index) / divisor - mean[channel]) / std[channel]).to_ne_bytes();
                }
            });
        });
        Ok(())
    })
}

fn parameters(values: Option<Vec<f32>>, channels: usize, default: f32, name: &str) -> PyResult<Vec<f32>> {
    let values = values.unwrap_or_else(|| vec![default; channels]);
    if values.len() != channels || values.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err(format!("{name} must contain {channels} finite float32 values")));
    }
    Ok(values)
}
