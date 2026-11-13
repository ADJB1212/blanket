//! Fused float32 tensor snapshots in contiguous HWC or CHW order.

use std::simd::num::SimdUint;
use std::simd::{Simd, ToBytes};

use blanket_core::{Image, PixelMode};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyByteArray;

#[pyfunction]
pub(crate) fn tensor_export<'py>(
    py: Python<'py>, image: &Image, layout: &str, normalize: bool, mean: Option<Vec<f32>>, std: Option<Vec<f32>>,
) -> PyResult<Bound<'py, PyByteArray>> {
    let channels = image.pixel_mode().channels();
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
    let pixels = image.width() as usize * image.height() as usize;
    let len = pixels.checked_mul(channels).and_then(|n| n.checked_mul(4)).ok_or_else(|| PyValueError::new_err("tensor dimensions are too large"))?;
    let divisor = if normalize && !image.pixel_mode().is_float() && image.pixel_mode() != PixelMode::I {
        ((1_u32 << image.bit_depth()) - 1) as f32
    } else {
        1.0
    };
    let bytes = image.raw_bytes_per_pixel() / channels;
    let interleaved_means: [Simd<f32, 16>; 4] =
        std::array::from_fn(|offset| Simd::from_array(std::array::from_fn(|lane| mean[(offset + lane) % channels])));
    let interleaved_stds: [Simd<f32, 16>; 4] =
        std::array::from_fn(|offset| Simd::from_array(std::array::from_fn(|lane| std[(offset + lane) % channels])));
    let read = |index: usize| -> f32 {
        let start = index * bytes;
        match bytes {
            4 => {
                let value = source[start..start + 4].try_into().unwrap();
                if image.pixel_mode().is_float() { f32::from_le_bytes(value) } else { i32::from_le_bytes(value) as f32 }
            }
            2 => {
                let value = [source[start], source[start + 1]];
                if image.pixel_mode().is_big_endian() { u16::from_be_bytes(value) as f32 } else { u16::from_le_bytes(value) as f32 }
            }
            _ => {
                let value = source[start];
                if image.pixel_mode() == PixelMode::One {
                    if value == 0 { 0.0 } else { 255.0 }
                } else if image.pixel_mode() == PixelMode::Lab && !index.is_multiple_of(channels) {
                    (value ^ 128) as f32
                } else {
                    value as f32
                }
            }
        }
    };
    PyByteArray::new_with(py, len, |output| {
        py.detach(|| {
            if planar && bytes == 1 && !matches!(image.pixel_mode(), PixelMode::One | PixelMode::Lab) {
                match channels {
                    1 => planar_bytes::<1>(source, output, divisor, &mean, &std),
                    2 => planar_bytes::<2>(source, output, divisor, &mean, &std),
                    3 => planar_bytes::<3>(source, output, divisor, &mean, &std),
                    4 => planar_bytes::<4>(source, output, divisor, &mean, &std),
                    _ => unreachable!("validated image channel count"),
                }
                return;
            }
            const CHUNK: usize = 16384;
            blanket_kernels::parallel::chunks_mut(output, CHUNK * 4, |chunk, output| {
                let start = chunk * CHUNK;
                let location = |index: usize| {
                    if planar { (index % pixels * channels + index / pixels, index / pixels) } else { (index, index % channels) }
                };
                let end = output.len() / 64 * 64;
                for (block, dst) in output[..end].as_chunks_mut::<64>().0.iter_mut().enumerate() {
                    let index = start + block * 16;
                    let (values, means, stds) = if !planar {
                        let values = if bytes == 1 && !matches!(image.pixel_mode(), PixelMode::One | PixelMode::Lab) {
                            Simd::<u8, 16>::from_slice(&source[index..]).cast::<f32>()
                        } else {
                            Simd::from_array(std::array::from_fn(|lane| read(index + lane)))
                        };
                        (values, interleaved_means[index % channels], interleaved_stds[index % channels])
                    } else {
                        let locations: [_; 16] = std::array::from_fn(|lane| location(index + lane));
                        (
                            Simd::from_array(locations.map(|(index, _)| read(index))),
                            Simd::from_array(locations.map(|(_, channel)| mean[channel])),
                            Simd::from_array(locations.map(|(_, channel)| std[channel])),
                        )
                    };
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

fn planar_bytes<const C: usize>(source: &[u8], output: &mut [u8], divisor: f32, mean: &[f32], std: &[f32]) {
    let pixels = source.len() / C;
    if pixels == 0 {
        return;
    }
    const CHUNK: usize = 16384;
    for (channel, plane) in output.chunks_exact_mut(pixels * 4).enumerate() {
        let mean = Simd::<f32, 16>::splat(mean[channel]);
        let std = Simd::<f32, 16>::splat(std[channel]);
        blanket_kernels::parallel::chunks_mut(plane, CHUNK * 4, |chunk, output| {
            let start = chunk * CHUNK;
            let end = output.len() / 64 * 64;
            for (block, dst) in output[..end].as_chunks_mut::<64>().0.iter_mut().enumerate() {
                let values = blanket_core::pixels::load::<C>(&source[(start + block * 16) * C..])[channel].cast::<f32>();
                *dst = ((values / Simd::splat(divisor) - mean) / std).to_ne_bytes().to_array();
            }
            for (offset, dst) in output[end..].as_chunks_mut::<4>().0.iter_mut().enumerate() {
                let value = f32::from(source[(start + end / 4 + offset) * C + channel]);
                *dst = ((value / divisor - mean[0]) / std[0]).to_ne_bytes();
            }
        });
    }
}

fn parameters(values: Option<Vec<f32>>, channels: usize, default: f32, name: &str) -> PyResult<Vec<f32>> {
    let values = values.unwrap_or_else(|| vec![default; channels]);
    if values.len() != channels || values.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err(format!("{name} must contain {channels} finite float32 values")));
    }
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn planar_byte_vectors_preserve_normalization_tails_and_guards() {
        fn check<const C: usize>() {
            let mean: [f32; C] = std::array::from_fn(|c| c as f32 / 7.0 - 0.5);
            let std: [f32; C] = std::array::from_fn(|c| c as f32 / 3.0 + 0.25);
            for count in [0, 1, 15, 16, 17, 33, 16385] {
                for offset in 0..4 {
                    let source: Vec<u8> = (0..offset + count * C).map(|i| (i * 37 + i / 7) as u8).collect();
                    for divisor in [1.0, 255.0] {
                        let mut output = vec![93; offset + count * C * 4 + 3];
                        planar_bytes::<C>(&source[offset..], &mut output[offset..offset + count * C * 4], divisor, &mean, &std);
                        for c in 0..C {
                            for pixel in 0..count {
                                let expected = ((f32::from(source[offset + pixel * C + c]) / divisor - mean[c]) / std[c]).to_ne_bytes();
                                let start = offset + (c * count + pixel) * 4;
                                assert_eq!(&output[start..start + 4], &expected);
                            }
                        }
                        assert!(output[..offset].iter().chain(&output[offset + count * C * 4..]).all(|&v| v == 93));
                    }
                }
            }
        }
        check::<1>();
        check::<2>();
        check::<3>();
        check::<4>();
    }
}
