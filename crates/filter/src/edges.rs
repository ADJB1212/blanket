//! Normalized luminance gradients and Canny edge masks.

use std::simd::Simd;

use blanket_core::buffers::{reserved, reserved_buffer};
use blanket_core::{Image, PixelMode};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(filter_gradients, module)?)?;
    module.add_function(wrap_pyfunction!(filter_canny, module)?)?;
    Ok(())
}

fn zeros<T: Default + Clone>(len: usize) -> PyResult<Vec<T>> {
    let mut values = reserved(len)?;
    values.resize(len, T::default());
    Ok(values)
}

fn luminance(image: &Image) -> PyResult<Vec<f32>> {
    let mode = image.pixel_mode();
    if image.width() == 0 || image.height() == 0 {
        return Err(PyValueError::new_err("edge detection requires a nonempty image"));
    }
    if image.palette().is_some()
        || !matches!(
            mode,
            PixelMode::L
                | PixelMode::La
                | PixelMode::Rgb
                | PixelMode::Rgba
                | PixelMode::F
                | PixelMode::RgbF
                | PixelMode::RgbaF
                | PixelMode::I16
                | PixelMode::I16L
                | PixelMode::I16B
                | PixelMode::I16N
        )
    {
        return Err(PyValueError::new_err("edge detection requires L, LA, RGB, RGBA, F, RGB;F, RGBA;F, or I;16 pixels; convert explicitly"));
    }
    let bytes = if mode.is_float() { 4 } else { usize::from(image.bit_depth().div_ceil(8)) };
    let channels = mode.channels();
    let colors = if channels < 3 { 1 } else { 3 };
    let maximum = if mode.is_float() { 1.0 } else { ((1_u32 << image.bit_depth()) - 1) as f32 };
    let source = image.raw_data()?;
    let mut output = reserved(source.len() / (bytes * channels))?;
    for pixel in source.chunks_exact(bytes * channels) {
        let mut rgb = [0.0; 3];
        for (c, value) in rgb.iter_mut().enumerate().take(colors) {
            let sample = &pixel[c * bytes..][..bytes];
            *value = if mode.is_float() {
                f32::from_le_bytes(sample.try_into().unwrap())
            } else if bytes == 1 {
                f32::from(sample[0]) / maximum
            } else if mode.is_big_endian() {
                f32::from(u16::from_be_bytes(sample.try_into().unwrap())) / maximum
            } else {
                f32::from(u16::from_le_bytes(sample.try_into().unwrap())) / maximum
            };
            if !value.is_finite() || !(0.0..=1.0).contains(value) {
                return Err(PyValueError::new_err("edge color samples must be finite and in [0, 1] after depth normalization"));
            }
        }
        output.push(if colors == 1 { rgb[0] } else { rgb[0] * 0.299 + rgb[1] * 0.587 + rgb[2] * 0.114 });
    }
    Ok(output)
}

fn gradients<const VECTOR: bool>(source: &[f32], width: usize, height: usize, weights: (f32, f32)) -> PyResult<(Vec<f32>, Vec<f32>)> {
    let (outer, middle) = weights;
    let scale = 2.0 * (2.0 * outer + middle);
    let mut gx = zeros(source.len())?;
    let mut gy = zeros(source.len())?;
    for y in 0..height {
        let above = &source[y.saturating_sub(1) * width..][..width];
        let row = &source[y * width..][..width];
        let below = &source[(y + 1).min(height - 1) * width..][..width];
        let mut x = 1;
        if VECTOR {
            while x + 8 < width {
                let load = |row: &[f32], offset| Simd::<f32, 8>::from_slice(&row[offset..]);
                let tl = load(above, x - 1);
                let tr = load(above, x + 1);
                let bl = load(below, x - 1);
                let br = load(below, x + 1);
                let dx =
                    ((tr - tl) * Simd::splat(outer) + (load(row, x + 1) - load(row, x - 1)) * Simd::splat(middle) + (br - bl) * Simd::splat(outer))
                        / Simd::splat(scale);
                let dy = ((bl - tl) * Simd::splat(outer) + (load(below, x) - load(above, x)) * Simd::splat(middle) + (br - tr) * Simd::splat(outer))
                    / Simd::splat(scale);
                dx.copy_to_slice(&mut gx[y * width + x..]);
                dy.copy_to_slice(&mut gy[y * width + x..]);
                x += 8;
            }
        }
        for x in std::iter::once(0).chain(x..width) {
            let left = x.saturating_sub(1);
            let right = (x + 1).min(width - 1);
            gx[y * width + x] =
                ((above[right] - above[left]) * outer + (row[right] - row[left]) * middle + (below[right] - below[left]) * outer) / scale;
            gy[y * width + x] =
                ((below[left] - above[left]) * outer + (below[x] - above[x]) * middle + (below[right] - above[right]) * outer) / scale;
        }
    }
    Ok((gx, gy))
}

fn float_image(image: &Image, values: &[f32]) -> PyResult<Image> {
    let mut pixels = reserved_buffer(image.size(), 4)?;
    for value in values {
        pixels.extend_from_slice(&value.to_le_bytes());
    }
    Image::from_float_bytes(image.width(), image.height(), pixels)
}

#[pyfunction]
fn filter_gradients(py: Python<'_>, image: &Image, operator: &str) -> PyResult<(Image, Image)> {
    let weights = match operator {
        "sobel" => (1.0, 2.0),
        "scharr" => (3.0, 10.0),
        _ => return Err(PyValueError::new_err("unknown gradient operator")),
    };
    py.detach(|| {
        let source = luminance(image)?;
        let (gx, gy) = gradients::<true>(&source, image.width() as usize, image.height() as usize, weights)?;
        Ok((float_image(image, &gx)?, float_image(image, &gy)?))
    })
}

fn smooth(source: Vec<f32>, width: usize, height: usize, sigma: f32) -> PyResult<Vec<f32>> {
    if sigma == 0.0 {
        return Ok(source);
    }
    let radius = (sigma * 3.0).ceil() as isize;
    let mut kernel: Vec<f32> = (-radius..=radius).map(|i| (-0.5 * (i as f32 / sigma).powi(2)).exp()).collect();
    let sum: f32 = kernel.iter().sum();
    for value in &mut kernel {
        *value /= sum;
    }
    let mut temporary = zeros::<f32>(source.len())?;
    let mut output = zeros::<f32>(source.len())?;
    for y in 0..height {
        for x in 0..width {
            for (k, &weight) in kernel.iter().enumerate() {
                let nx = (x as isize + k as isize - radius).clamp(0, width as isize - 1) as usize;
                temporary[y * width + x] += source[y * width + nx] * weight;
            }
        }
    }
    for y in 0..height {
        for x in 0..width {
            for (k, &weight) in kernel.iter().enumerate() {
                let ny = (y as isize + k as isize - radius).clamp(0, height as isize - 1) as usize;
                output[y * width + x] += temporary[ny * width + x] * weight;
            }
        }
    }
    Ok(output)
}

fn hysteresis(thinned: &[f32], width: usize, height: usize, low: f64, high: f64) -> PyResult<Vec<u8>> {
    let mut output = zeros::<u8>(thinned.len())?;
    let mut stack = reserved(thinned.len())?;
    for (i, &value) in thinned.iter().enumerate() {
        if value > 0.0 && f64::from(value) >= high {
            output[i] = 255;
            stack.push(i);
        }
    }
    while let Some(i) = stack.pop() {
        let (x, y) = (i % width, i / width);
        for ny in y.saturating_sub(1)..=(y + 1).min(height - 1) {
            for nx in x.saturating_sub(1)..=(x + 1).min(width - 1) {
                let neighbor = ny * width + nx;
                if output[neighbor] == 0 && thinned[neighbor] > 0.0 && f64::from(thinned[neighbor]) >= low {
                    output[neighbor] = 255;
                    stack.push(neighbor);
                }
            }
        }
    }
    Ok(output)
}

fn canny(source: Vec<f32>, width: usize, height: usize, low: f64, high: f64, sigma: f32) -> PyResult<Vec<u8>> {
    let source = smooth(source, width, height, sigma)?;
    let (gx, gy) = gradients::<true>(&source, width, height, (1.0, 2.0))?;
    let mut magnitude = reserved(gx.len())?;
    magnitude.extend(gx.iter().zip(&gy).map(|(&x, &y)| x.hypot(y)));
    let mut thinned = zeros(magnitude.len())?;
    for y in 1..height.saturating_sub(1) {
        for x in 1..width.saturating_sub(1) {
            let i = y * width + x;
            let angle = gy[i].atan2(gx[i]).to_degrees().rem_euclid(180.0);
            let offset = if !(22.5..157.5).contains(&angle) {
                1
            } else if angle < 67.5 {
                width + 1
            } else if angle < 112.5 {
                width
            } else {
                width - 1
            };
            if magnitude[i] > magnitude[i - offset] && magnitude[i] >= magnitude[i + offset] {
                thinned[i] = magnitude[i];
            }
        }
    }
    hysteresis(&thinned, width, height, low, high)
}

#[pyfunction]
fn filter_canny(py: Python<'_>, image: &Image, low: f64, high: f64, sigma: f64) -> PyResult<Image> {
    if !low.is_finite() || !high.is_finite() || !(0.0..=high).contains(&low) || high > 1.0 {
        return Err(PyValueError::new_err("thresholds must satisfy 0 <= low <= high <= 1 and be finite"));
    }
    if !sigma.is_finite() || !(0.0..=32.0).contains(&sigma) {
        return Err(PyValueError::new_err("sigma must be finite and in [0, 32]"));
    }
    py.detach(|| {
        let source = luminance(image)?;
        let pixels = canny(source, image.width() as usize, image.height() as usize, low, high, sigma as f32)?;
        Image::from_pixels(image.width(), image.height(), PixelMode::L, pixels, None)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_gradients_match_scalar_with_unaligned_rows_and_tails() {
        for width in [1, 2, 7, 8, 9, 10, 15, 16, 17, 31, 32, 33] {
            for height in [1, 2, 13] {
                let storage: Vec<f32> = (0..width * height + 3).map(|i| (i * 719 % 4093) as f32 / 4093.0).collect();
                let source = &storage[1..1 + width * height];
                for weights in [(1.0, 2.0), (3.0, 10.0)] {
                    assert_eq!(
                        gradients::<true>(source, width, height, weights).unwrap(),
                        gradients::<false>(source, width, height, weights).unwrap()
                    );
                }
                assert_eq!(storage[0], 0.0);
            }
        }
    }

    #[test]
    fn hysteresis_keeps_only_eight_connected_weak_edges() {
        let mut values = vec![0.0; 49];
        values[8] = 0.8;
        values[16] = 0.3;
        values[24] = 0.3;
        values[40] = 0.3;
        let result = hysteresis(&values, 7, 7, 0.2, 0.5).unwrap();
        assert_eq!(result.iter().filter(|&&v| v == 255).count(), 3);
        assert_eq!(result[40], 0);
        assert!(hysteresis(&[0.0; 9], 3, 3, 0.0, 0.0).unwrap().iter().all(|&v| v == 0));
    }
}
