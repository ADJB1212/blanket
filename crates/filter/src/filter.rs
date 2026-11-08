//! Pillow-compatible filter registration, color LUTs, and band merging.

use blanket_core::parallel::{CHUNK_PIXELS, chunks_mut};
use blanket_core::raster::{Image, PixelMode, alloc_pixels};
pub use blanket_filter_blur::{filter_blur, filter_unsharp};
pub use blanket_filter_neighborhood::filter_rank;
use blanket_filter_neighborhood::{filter_kernel, filter_mode};
use pyo3::exceptions::{PyMemoryError, PyValueError};
use pyo3::prelude::*;

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(filter_kernel, module)?)?;
    module.add_function(wrap_pyfunction!(filter_rank, module)?)?;
    module.add_function(wrap_pyfunction!(filter_mode, module)?)?;
    module.add_function(wrap_pyfunction!(filter_blur, module)?)?;
    module.add_function(wrap_pyfunction!(filter_unsharp, module)?)?;
    module.add_function(wrap_pyfunction!(filter_lut, module)?)?;
    module.add_function(wrap_pyfunction!(filter_merge, module)?)?;
    Ok(())
}

// Pillow stores LUT samples as signed 16-bit values with six fractional bits.
const LUT_SAMPLE_BITS: u32 = 6;
const LUT_SAMPLE_SCALE: f32 = (255 << LUT_SAMPLE_BITS) as f32;
// Positions retain 18 fractional bits; interpolation uses the upper 15.
const LUT_POSITION_BITS: u32 = 18;
const LUT_BLEND_BITS: u32 = 15;
const LUT_BLEND_SCALE: i32 = 1 << LUT_BLEND_BITS;

#[pyfunction]
fn filter_lut(py: Python<'_>, image: &Image, mode: &str, channels: usize, size: (usize, usize, usize), table: Vec<f32>) -> PyResult<Image> {
    if blanket_core::samples::supported(image) {
        return py.detach(|| crate::lut_samples::lut(image, mode, channels, size, &table));
    }
    let source = image.pixel_data()?;
    let target = PixelMode::parse(mode)?;
    let input_channels = image.mode.channels();
    let output_channels = target.channels();
    if !(3..=4).contains(&channels)
        || input_channels < 3
        || output_channels < channels
        || (output_channels > channels && output_channels > input_channels)
    {
        return Err(PyValueError::new_err("image has wrong mode"));
    }
    let dimensions = <[_; 3]>::from(size);
    if dimensions.iter().any(|s| !(2..=65).contains(s)) {
        return Err(PyValueError::new_err("Table size in any dimension should be from 2 to 65"));
    }
    if table.len() != channels * size.0 * size.1 * size.2 {
        return Err(PyValueError::new_err("table has wrong number of elements"));
    }
    let prepared: Vec<i32> =
        py.detach(|| table.into_iter().map(|v| ((v * LUT_SAMPLE_SCALE).round() as i32).clamp(i32::from(i16::MIN), i32::from(i16::MAX))).collect());
    let scales = dimensions.map(|s| ((s - 1) as f64 / 255.0 * f64::from(1 << LUT_POSITION_BITS)) as usize);
    let strides = [channels, channels * size.0, channels * size.0 * size.1];
    let positions: [[(usize, i32); 256]; 3] = std::array::from_fn(|c| {
        std::array::from_fn(|value| {
            let position = value * scales[c];
            (
                (position >> LUT_POSITION_BITS) * strides[c],
                ((position & ((1 << LUT_POSITION_BITS) - 1)) >> (LUT_POSITION_BITS - LUT_BLEND_BITS)) as i32,
            )
        })
    });
    let len = (image.width as usize)
        .checked_mul(image.height as usize)
        .and_then(|n| n.checked_mul(output_channels))
        .ok_or_else(|| PyMemoryError::new_err("cannot allocate image"))?;
    let mut pixels = alloc_pixels(len)?;
    py.detach(|| {
        chunks_mut(&mut pixels, CHUNK_PIXELS * output_channels, |chunk, dst| {
            for (i, pixel) in dst.chunks_exact_mut(output_channels).enumerate() {
                let input = &source[(chunk * CHUNK_PIXELS + i) * input_channels..];
                let coordinates = std::array::from_fn::<_, 3, _>(|c| positions[c][usize::from(input[c])]);
                let shifts = coordinates.map(|(_, shift)| shift);
                let index = coordinates.iter().map(|&(offset, _)| offset).sum::<usize>();
                let blend = |a: i32, b: i32, shift: i32| (a * (LUT_BLEND_SCALE - shift) + b * shift) >> LUT_BLEND_BITS;
                for (channel, value) in pixel.iter_mut().enumerate().take(channels) {
                    let mut planes = [0; 2];
                    for (z, plane) in planes.iter_mut().enumerate() {
                        let base = index + z * size.0 * size.1 * channels + channel;
                        let lower = blend(prepared[base], prepared[base + channels], shifts[0]);
                        let upper = blend(prepared[base + size.0 * channels], prepared[base + (size.0 + 1) * channels], shifts[0]);
                        *plane = blend(lower, upper, shifts[1]);
                    }
                    *value = ((blend(planes[0], planes[1], shifts[2]) + (1 << (LUT_SAMPLE_BITS - 1))) >> LUT_SAMPLE_BITS).clamp(0, 255) as u8;
                }
                if output_channels > channels {
                    pixel[3] = input[3];
                }
            }
        });
    });
    Image::from_pixels(image.width, image.height, target, pixels, None)
}

#[pyfunction]
fn filter_merge(py: Python<'_>, mode: &str, bands: Vec<PyRef<'_, Image>>) -> PyResult<Image> {
    let mode = PixelMode::parse(mode)?;
    if bands.len() != mode.channels() {
        return Err(PyValueError::new_err("wrong number of bands"));
    }
    let (width, height) = (bands[0].width, bands[0].height);
    if bands.iter().any(|b| b.mode != PixelMode::L || b.width != width || b.height != height || b.bit_depth != bands[0].bit_depth) {
        return Err(PyValueError::new_err("images do not match"));
    }
    if blanket_core::samples::supported(&bands[0]) {
        if mode == PixelMode::L {
            let source = bands[0].raw_data()?;
            return Image::from_sample_bytes(width, height, mode, py.detach(|| source.to_vec()), bands[0].bit_depth, None);
        }
        let sources = bands.iter().map(|b| blanket_core::samples::view(b)).collect::<PyResult<Vec<_>>>()?;
        let pixels = py.detach(|| match mode {
            PixelMode::Rgb => Ok(merge_samples::<3>(&sources)),
            PixelMode::Rgba => Ok(merge_samples::<4>(&sources)),
            _ => Err(PyValueError::new_err("this mode requires 8-bit pixels")),
        })?;
        return Image::from_sample_bytes(width, height, mode, pixels, bands[0].bit_depth, None);
    }
    let sources = bands.iter().map(|b| b.pixel_data()).collect::<PyResult<Vec<_>>>()?;
    if matches!(mode, PixelMode::One | PixelMode::L) {
        // A single band only needs a copy, not a zeroed allocation followed by
        // another full-buffer write. This matters once the band exceeds cache.
        return Image::from_pixels(width, height, mode, py.detach(|| sources[0].to_vec()), None);
    }
    let mut pixels = alloc_pixels(sources[0].len().checked_mul(bands.len()).ok_or_else(|| PyMemoryError::new_err("cannot allocate image"))?)?;
    py.detach(|| match mode {
        PixelMode::One | PixelMode::L => pixels.copy_from_slice(sources[0]),
        PixelMode::La | PixelMode::PremulLa | PixelMode::Pa => merge::<2>(&sources, &mut pixels),
        PixelMode::Rgb | PixelMode::Hsv | PixelMode::YCbCr | PixelMode::Lab => merge::<3>(&sources, &mut pixels),
        PixelMode::Rgba | PixelMode::PremulRgba | PixelMode::Rgbx | PixelMode::Cmyk => merge::<4>(&sources, &mut pixels),
        _ => unreachable!(),
    });
    Image::from_pixels(width, height, mode, pixels, None)
}

fn merge_samples<const C: usize>(sources: &[std::borrow::Cow<'_, [u16]>]) -> Vec<u8> {
    let mut values = vec![0; sources[0].len() * C * 2];
    const CHUNK: usize = 128 * 1024;
    blanket_core::parallel::chunks_mut_above(&mut values, CHUNK * C * 2, 4 * 1024 * 1024, |chunk, dst| {
        let start = chunk * CHUNK;
        let count = dst.len() / (C * 2);
        let end = count / 16 * 16;
        for i in (0..end).step_by(16) {
            let samples = std::array::from_fn(|c| blanket_core::pixels::Samples::from_slice(&sources[c][start + i..]));
            blanket_core::pixels::store_sample_bytes::<C>(&mut dst[i * C * 2..], samples);
        }
        for (i, pixel) in dst[end * C * 2..].chunks_exact_mut(C * 2).enumerate() {
            for (c, sample) in pixel.as_chunks_mut::<2>().0.iter_mut().enumerate() {
                *sample = sources[c][start + end + i].to_le_bytes();
            }
        }
    });
    values
}

fn merge<const C: usize>(sources: &[&[u8]], pixels: &mut [u8]) {
    const CHUNK: usize = 256 * 1024;
    blanket_core::parallel::chunks_mut_above(pixels, CHUNK * C, 4 * 1024 * 1024, |chunk, dst| {
        let start = chunk * CHUNK;
        let dst = dst.as_chunks_mut::<C>().0;
        let bands: [&[u8]; C] = std::array::from_fn(|c| &sources[c][start..start + dst.len()]);
        let offset = blanket_core::pixels::merge(bands, dst);
        for (i, pixel) in dst.iter_mut().enumerate().skip(offset) {
            *pixel = std::array::from_fn(|c| bands[c][i]);
        }
    });
}

#[cfg(test)]
mod tests {
    #[test]
    fn sample_merge_preserves_parallel_boundaries_and_scalar_tails() {
        fn check<const C: usize>() {
            for count in [0, 1, 15, 16, 17, 131073, 1048593] {
                let sources: Vec<std::borrow::Cow<'_, [u16]>> =
                    (0..C).map(|c| std::borrow::Cow::Owned((0..count).map(|i| (i * 719 + c * 313) as u16).collect())).collect();
                let actual = super::merge_samples::<C>(&sources);
                assert_eq!(actual.len(), count * C * 2);
                let actual: Vec<u16> = actual.as_chunks::<2>().0.iter().map(|&bytes| u16::from_le_bytes(bytes)).collect();
                for (i, pixel) in actual.as_chunks::<C>().0.iter().enumerate() {
                    assert_eq!(*pixel, std::array::from_fn(|c| sources[c][i]));
                }
            }
        }
        check::<3>();
        check::<4>();
    }
}
