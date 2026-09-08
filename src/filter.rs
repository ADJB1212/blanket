//! Pillow-compatible filtering with independent output partitions.

use pyo3::exceptions::{PyMemoryError, PyValueError};
use pyo3::prelude::*;

use crate::parallel::{CHUNK_PIXELS, chunks_mut};
use crate::raster::{Image, PixelMode};

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(filter_kernel, module)?)?;
    module.add_function(wrap_pyfunction!(filter_rank, module)?)?;
    module.add_function(wrap_pyfunction!(filter_mode, module)?)?;
    module.add_function(wrap_pyfunction!(filter_blur, module)?)?;
    module.add_function(wrap_pyfunction!(filter_unsharp, module)?)?;
    module.add_function(wrap_pyfunction!(filter_lut, module)?)?;
    module.add_function(wrap_pyfunction!(filter_merge, module)?)?;
    Ok(())
}

fn buffer(len: usize) -> PyResult<Vec<u8>> {
    let mut pixels = Vec::new();
    pixels
        .try_reserve_exact(len)
        .map_err(|_| PyMemoryError::new_err("cannot allocate image"))?;
    pixels.resize(len, 0);
    Ok(pixels)
}

fn output(image: &Image, pixels: Vec<u8>) -> PyResult<Image> {
    Image::from_pixels(image.width, image.height, image.mode, pixels, None)
}

#[pyfunction]
fn filter_kernel(py: Python<'_>, image: &Image, size: (i32, i32), scale: f32, offset: f32, kernel: Vec<f32>) -> PyResult<Image> {
    let source = image.pixel_data()?;
    if i64::from(size.0) * i64::from(size.1) != kernel.len() as i64 {
        return Err(PyValueError::new_err("bad kernel size"));
    }
    let mut pixels = buffer(source.len())?;
    pixels.copy_from_slice(source);
    if i64::from(image.width) < i64::from(size.0) || i64::from(image.height) < i64::from(size.1) {
        return output(image, pixels);
    }
    if size.0 != size.1 || !matches!(size.0, 3 | 5) {
        return Err(PyValueError::new_err("bad kernel size"));
    }
    let weights: Vec<f32> = kernel.into_iter().map(|value| value / scale).collect();
    let width = image.width as usize;
    py.detach(|| match (image.mode, size.0) {
        (PixelMode::L, 3) => convolve::<1, 3>(source, &mut pixels, width, &weights, offset),
        (PixelMode::L, _) => convolve::<1, 5>(source, &mut pixels, width, &weights, offset),
        (PixelMode::Rgb, 3) => convolve::<3, 3>(source, &mut pixels, width, &weights, offset),
        (PixelMode::Rgb, _) => convolve::<3, 5>(source, &mut pixels, width, &weights, offset),
        (PixelMode::Rgba, 3) => convolve::<4, 3>(source, &mut pixels, width, &weights, offset),
        (PixelMode::Rgba, _) => convolve::<4, 5>(source, &mut pixels, width, &weights, offset),
    });
    output(image, pixels)
}

// Fixed channel and kernel sizes let every tap and band unroll; the runtime
// version spent its time on strided index arithmetic and bounds checks.
fn convolve<const C: usize, const K: usize>(source: &[u8], pixels: &mut [u8], width: usize, weights: &[f32], offset: f32) {
    if pixels.is_empty() {
        return;
    }
    let taps: &[[f32; K]] = weights.as_chunks::<K>().0;
    let radius = K / 2;
    let stride = width * C;
    let height = source.len() / stride;
    let source = source.as_chunks::<C>().0;
    chunks_mut(pixels, stride * 16, |band, rows| {
        for (i, row) in rows.chunks_exact_mut(stride).enumerate() {
            let y = band * 16 + i;
            if y < radius || y + radius >= height {
                continue;
            }
            let row = row.as_chunks_mut::<C>().0;
            for x in radius..width - radius {
                let mut sums = [offset + 0.5; C];
                for (ky, taps) in taps.iter().enumerate() {
                    let window: &[[u8; C]; K] = source[(y + radius - ky) * width + x - radius..][..K].try_into().unwrap();
                    // Sum each row before accumulating, matching Pillow's f32 order.
                    for c in 0..C {
                        let mut subtotal = kernel_add(f32::from(window[0][c]), taps[0], f32::from(window[1][c]) * taps[1]);
                        for kx in 2..K {
                            subtotal = kernel_add(f32::from(window[kx][c]), taps[kx], subtotal);
                        }
                        sums[c] += subtotal;
                    }
                }
                for c in 0..C {
                    row[x][c] = sums[c] as u8;
                }
            }
        }
    });
}

// Pillow's AArch64 C builds contract each row's multiply/add expressions.
// Keep the same operand order; it matters at half-integer output boundaries.
fn kernel_add(value: f32, weight: f32, sum: f32) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        value.mul_add(weight, sum)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        value * weight + sum
    }
}

#[pyfunction]
fn filter_rank(py: Python<'_>, image: &Image, size: i32, rank: i32) -> PyResult<Image> {
    if size < 1 || size % 2 == 0 {
        return Err(PyValueError::new_err("bad filter size"));
    }
    let count = i64::from(size) * i64::from(size);
    if count > i64::from(i32::MAX) / 4 {
        return Err(PyValueError::new_err("filter size too large"));
    }
    if rank < 0 || i64::from(rank) >= count {
        return Err(PyValueError::new_err("bad rank value"));
    }
    neighborhood(py, image, (size / 2) as usize, Some(rank as u64))
}

#[pyfunction]
fn filter_mode(py: Python<'_>, image: &Image, size: i32) -> PyResult<Image> {
    neighborhood(py, image, (size / 2).max(0) as usize, None)
}

fn neighborhood(py: Python<'_>, image: &Image, radius: usize, rank: Option<u64>) -> PyResult<Image> {
    let source = image.pixel_data()?;
    let mut pixels = buffer(source.len())?;
    let width = image.width as usize;
    let height = image.height as usize;
    let channels = image.mode.channels();
    if pixels.is_empty() {
        return output(image, pixels);
    }
    py.detach(|| {
        chunks_mut(&mut pixels, width * channels * 8, |band, rows| {
            for (i, row) in rows.chunks_exact_mut(width * channels).enumerate() {
                let y = band * 8 + i;
                let top = y.saturating_sub(radius);
                let bottom = (y + radius).min(height - 1);
                let vertical_weight = |yy: usize| -> u64 {
                    if rank.is_none() {
                        return 1;
                    }
                    1 + if yy == 0 { radius.saturating_sub(y) as u64 } else { 0 }
                        + if yy == height - 1 {
                            (y + radius).saturating_sub(height - 1) as u64
                        } else {
                            0
                        }
                };
                for channel in 0..channels {
                    let mut histogram = [0_u64; 256];
                    for yy in top..=bottom {
                        let weight = vertical_weight(yy);
                        for xx in 0..=radius.min(width - 1) {
                            let horizontal = if rank.is_some() {
                                1 + if xx == 0 { radius as u64 } else { 0 }
                                    + if xx == width - 1 { radius.saturating_sub(width - 1) as u64 } else { 0 }
                            } else {
                                1
                            };
                            histogram[source[(yy * width + xx) * channels + channel] as usize] += weight * horizontal;
                        }
                    }
                    for x in 0..width {
                        let original = source[(y * width + x) * channels + channel];
                        row[x * channels + channel] = select_value(&histogram, rank, original);
                        if x + 1 == width {
                            break;
                        }
                        for yy in top..=bottom {
                            let weight = vertical_weight(yy);
                            if rank.is_some() || x >= radius {
                                let xx = x.saturating_sub(radius);
                                histogram[source[(yy * width + xx) * channels + channel] as usize] -= weight;
                            }
                            if rank.is_some() || x + radius + 1 < width {
                                let xx = (x + radius + 1).min(width - 1);
                                histogram[source[(yy * width + xx) * channels + channel] as usize] += weight;
                            }
                        }
                    }
                }
            }
        });
    });
    output(image, pixels)
}

fn select_value(histogram: &[u64; 256], rank: Option<u64>, original: u8) -> u8 {
    let mut count = 0;
    let mut result = original;
    for (value, &frequency) in histogram.iter().enumerate() {
        if let Some(rank) = rank {
            count += frequency;
            if count > rank {
                return value as u8;
            }
        } else if frequency > count {
            count = frequency;
            if count > 2 {
                result = value as u8;
            }
        }
    }
    result
}

fn gaussian_radius(radius: f32) -> f32 {
    let variance = radius * radius / 3.0;
    let length = (12.0 * f64::from(variance) + 1.0).sqrt() as f32;
    let integer = ((f64::from(length) - 1.0) / 2.0).floor() as f32;
    let mut fraction = (2.0 * integer + 1.0) * (integer * (integer + 1.0) - 3.0 * variance);
    fraction /= 6.0 * (variance - (integer + 1.0) * (integer + 1.0));
    integer + fraction
}

fn box_horizontal<const C: usize>(source: &[u8], pixels: &mut [u8], width: usize, radius: f32) {
    let integer = radius as usize;
    let weight = ((1_u32 << 24) as f32 / (radius * 2.0 + 1.0)) as u64;
    let far_weight = u64::from((1_u32 << 24).wrapping_sub((2 * integer as u32 + 1).wrapping_mul(weight as u32)) / 2);
    let stride = width * C;
    let source = source.as_chunks::<C>().0;
    chunks_mut(pixels, stride * 16, |band, rows| {
        for (i, row) in rows.chunks_exact_mut(stride).enumerate() {
            let input = &source[(band * 16 + i) * width..][..width];
            let row = row.as_chunks_mut::<C>().0;
            for channel in 0..C {
                let mut sum = u64::from(input[0][channel]) * (integer as u64 + 1);
                for pixel in &input[1..=integer.min(width - 1)] {
                    sum += u64::from(pixel[channel]);
                }
                sum += u64::from(input[width - 1][channel]) * integer.saturating_sub(width - 1) as u64;
                for x in 0..width {
                    let left = x.saturating_sub(integer + 1);
                    let right = (x + integer + 1).min(width - 1);
                    let ends = u64::from(input[left][channel]) + u64::from(input[right][channel]);
                    row[x][channel] = ((sum * weight + ends * far_weight + (1 << 23)) >> 24) as u8;
                    if x + 1 < width {
                        sum -= u64::from(input[x.saturating_sub(integer)][channel]);
                        sum += u64::from(input[right][channel]);
                    }
                }
            }
        }
    });
}

fn blurred<const C: usize>(source: &[u8], width: usize, height: usize, radii: (f32, f32), gaussian: bool) -> PyResult<Vec<u8>> {
    let mut pixels = buffer(source.len())?;
    pixels.copy_from_slice(source);
    if pixels.is_empty() {
        return Ok(pixels);
    }
    let radii = if gaussian {
        (gaussian_radius(radii.0), gaussian_radius(radii.1))
    } else {
        radii
    };
    if [radii.0, radii.1].iter().any(|r| !r.is_finite() || *r < 0.0 || *r >= i32::MAX as f32) {
        return Err(PyValueError::new_err("radius must be finite and >= 0"));
    }
    let mut scratch = buffer(source.len())?;
    let passes = if gaussian { 3 } else { 1 };
    if radii.0 > 0.0 {
        for _ in 0..passes {
            box_horizontal::<C>(&pixels, &mut scratch, width, radii.0);
            std::mem::swap(&mut pixels, &mut scratch);
        }
    }
    if radii.1 > 0.0 {
        // The cache-tiled rotation kernel turns the vertical pass horizontal.
        crate::ops::transpose::<C>(&pixels, &mut scratch, width, height, 5);
        std::mem::swap(&mut pixels, &mut scratch);
        for _ in 0..passes {
            box_horizontal::<C>(&pixels, &mut scratch, height, radii.1);
            std::mem::swap(&mut pixels, &mut scratch);
        }
        crate::ops::transpose::<C>(&pixels, &mut scratch, height, width, 5);
        std::mem::swap(&mut pixels, &mut scratch);
    }
    Ok(pixels)
}

fn blur(image: &Image, source: &[u8], radii: (f32, f32), gaussian: bool) -> PyResult<Vec<u8>> {
    let (width, height) = (image.width as usize, image.height as usize);
    match image.mode {
        PixelMode::L => blurred::<1>(source, width, height, radii, gaussian),
        PixelMode::Rgb => blurred::<3>(source, width, height, radii, gaussian),
        PixelMode::Rgba => blurred::<4>(source, width, height, radii, gaussian),
    }
}

#[pyfunction]
fn filter_blur(py: Python<'_>, image: &Image, radii: (f32, f32), gaussian: bool) -> PyResult<Image> {
    let source = image.pixel_data()?;
    let pixels = py.detach(|| blur(image, source, radii, gaussian))?;
    output(image, pixels)
}

#[pyfunction]
fn filter_unsharp(py: Python<'_>, image: &Image, radius: f32, percent: i32, threshold: i32) -> PyResult<Image> {
    let source = image.pixel_data()?;
    let pixels = py.detach(|| -> PyResult<Vec<u8>> {
        let mut pixels = blur(image, source, (radius, radius), true)?;
        chunks_mut(&mut pixels, CHUNK_PIXELS, |chunk, dst| {
            for (&original, blurred) in source[chunk * CHUNK_PIXELS..].iter().zip(dst) {
                let difference = i32::from(original) - i32::from(*blurred);
                *blurred = if difference.abs() > threshold {
                    (i64::from(original) + i64::from(difference) * i64::from(percent) / 100).clamp(0, 255) as u8
                } else {
                    original
                };
            }
        });
        Ok(pixels)
    })?;
    output(image, pixels)
}

#[pyfunction]
fn filter_lut(py: Python<'_>, image: &Image, mode: &str, channels: usize, size: (usize, usize, usize), table: Vec<f32>) -> PyResult<Image> {
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
    let dimensions = [size.0, size.1, size.2];
    if dimensions.iter().any(|s| !(2..=65).contains(s)) {
        return Err(PyValueError::new_err("Table size in any dimension should be from 2 to 65"));
    }
    if table.len() != channels * size.0 * size.1 * size.2 {
        return Err(PyValueError::new_err("table has wrong number of elements"));
    }
    let prepared: Vec<i32> = table.into_iter().map(|v| ((v * 16320.0).round() as i32).clamp(-32768, 32767)).collect();
    let scales = dimensions.map(|s| ((s - 1) as f64 / 255.0 * (1 << 18) as f64) as usize);
    let len = (image.width as usize)
        .checked_mul(image.height as usize)
        .and_then(|n| n.checked_mul(output_channels))
        .ok_or_else(|| PyMemoryError::new_err("cannot allocate image"))?;
    let mut pixels = buffer(len)?;
    py.detach(|| {
        chunks_mut(&mut pixels, CHUNK_PIXELS * output_channels, |chunk, dst| {
            for (i, pixel) in dst.chunks_exact_mut(output_channels).enumerate() {
                let input = &source[(chunk * CHUNK_PIXELS + i) * input_channels..];
                let positions = [
                    usize::from(input[0]) * scales[0],
                    usize::from(input[1]) * scales[1],
                    usize::from(input[2]) * scales[2],
                ];
                let shifts = positions.map(|p| ((p & ((1 << 18) - 1)) >> 3) as i32);
                let index = channels * ((positions[0] >> 18) + (positions[1] >> 18) * size.0 + (positions[2] >> 18) * size.0 * size.1);
                let blend = |a: i32, b: i32, shift: i32| (a * (32768 - shift) + b * shift) >> 15;
                for (channel, value) in pixel.iter_mut().enumerate().take(channels) {
                    let mut planes = [0; 2];
                    for (z, plane) in planes.iter_mut().enumerate() {
                        let base = index + z * size.0 * size.1 * channels + channel;
                        let lower = blend(prepared[base], prepared[base + channels], shifts[0]);
                        let upper = blend(prepared[base + size.0 * channels], prepared[base + (size.0 + 1) * channels], shifts[0]);
                        *plane = blend(lower, upper, shifts[1]);
                    }
                    *value = ((blend(planes[0], planes[1], shifts[2]) + 32) >> 6).clamp(0, 255) as u8;
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
    if bands.iter().any(|b| b.mode != PixelMode::L || b.width != width || b.height != height) {
        return Err(PyValueError::new_err("images do not match"));
    }
    let sources = bands.iter().map(|b| b.pixel_data()).collect::<PyResult<Vec<_>>>()?;
    let mut pixels = buffer(
        sources[0]
            .len()
            .checked_mul(bands.len())
            .ok_or_else(|| PyMemoryError::new_err("cannot allocate image"))?,
    )?;
    py.detach(|| match mode {
        PixelMode::L => pixels.copy_from_slice(sources[0]),
        PixelMode::Rgb => merge::<3>(&sources, &mut pixels),
        PixelMode::Rgba => merge::<4>(&sources, &mut pixels),
    });
    Image::from_pixels(width, height, mode, pixels, None)
}

fn merge<const C: usize>(sources: &[&[u8]], pixels: &mut [u8]) {
    chunks_mut(pixels, CHUNK_PIXELS * C, |chunk, dst| {
        let start = chunk * CHUNK_PIXELS;
        let dst = dst.as_chunks_mut::<C>().0;
        let bands: [&[u8]; C] = std::array::from_fn(|c| &sources[c][start..start + dst.len()]);
        for (i, pixel) in dst.iter_mut().enumerate() {
            *pixel = std::array::from_fn(|c| bands[c][i]);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn convolution_orientation_and_borders() {
        let input: Vec<u8> = (0..25).collect();
        let mut result = input.clone();
        let mut kernel = [0.0; 9];
        kernel[0] = 1.0;
        convolve::<1, 3>(&input, &mut result, 5, &kernel, 0.0);
        assert_eq!(result[12], input[16]);
        assert_eq!(&result[..5], &input[..5]);
        // Interleaved channels stay independent, including alpha.
        let rgba: Vec<u8> = (0..100).collect();
        let mut result = rgba.clone();
        convolve::<4, 3>(&rgba, &mut result, 5, &kernel, 0.0);
        assert_eq!(&result[48..52], &rgba[64..68]);
        assert_eq!(&result[..20], &rgba[..20]);
    }

    #[test]
    fn box_blur_replicates_edges_and_rounds() {
        let mut result = [0; 3];
        box_horizontal::<1>(&[0, 90, 0], &mut result, 3, 1.0);
        assert_eq!(result, [30; 3]);
        box_horizontal::<1>(&[77; 3], &mut result, 3, 100.5);
        assert_eq!(result, [77; 3]);
        let mut rgb = [0; 9];
        box_horizontal::<3>(&[0, 90, 3, 90, 0, 3, 0, 90, 3], &mut rgb, 3, 1.0);
        assert_eq!(rgb, [30, 60, 3, 30, 60, 3, 30, 60, 3]);
    }

    #[test]
    fn vertical_blur_uses_transposed_passes() {
        // A 1x3 column blurred vertically equals the 3x1 row blurred horizontally.
        let column = blurred::<1>(&[0, 90, 0], 1, 3, (0.0, 1.0), false).unwrap();
        let row = blurred::<1>(&[0, 90, 0], 3, 1, (1.0, 0.0), false).unwrap();
        assert_eq!(column, row);
        assert_eq!(column, [30; 3]);
    }

    #[test]
    fn mode_ties_and_rare_values() {
        let mut histogram = [0; 256];
        histogram[1] = 2;
        histogram[2] = 2;
        assert_eq!(select_value(&histogram, None, 99), 99);
        histogram[1] = 3;
        histogram[2] = 3;
        assert_eq!(select_value(&histogram, None, 99), 1);
        assert_eq!(select_value(&histogram, Some(3), 99), 2);
    }
}
