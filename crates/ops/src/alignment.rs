//! Phase-only translation correlation and bounded in-plane rotation search.

use std::f64::consts::TAU;
use std::simd::cmp::SimdPartialOrd;
use std::simd::num::{SimdFloat, SimdUint};
use std::simd::{Select, Simd, StdFloat};

use blanket_core::buffers::{copy, reserved};
use blanket_core::pixels::{load, load_sample_bytes};
use blanket_core::{Image, PixelMode};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rustfft::num_complex::Complex;
use rustfft::{FftDirection, FftPlannerScalar};

const MAX_PIXELS: usize = 4_194_304;

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(ops_phase_correlate, module)?)?;
    Ok(())
}

fn zeros<T: Clone + Default>(len: usize) -> PyResult<Vec<T>> {
    let mut values = reserved(len)?;
    values.resize(len, T::default());
    Ok(values)
}

fn grayscale_channels<const C: usize>(image: &Image, bytes: &[u8], output: &mut [f64]) {
    let wide = image.bit_depth() > 8;
    let big_endian = image.pixel_mode().is_big_endian();
    let stride = C * if wide { 2 } else { 1 };
    let end = output.len() / 16 * 16;
    for i in (0..end).step_by(16) {
        let values: [Simd<f64, 16>; C] = if wide {
            load_sample_bytes::<C>(&bytes[i * stride..]).map(|value| {
                let value = if big_endian { (value << 8) | (value >> 8) } else { value };
                value.cast()
            })
        } else {
            load::<C>(&bytes[i * stride..]).map(|value| value.cast())
        };
        let luma = if C < 3 { values[0] } else { values[0] * Simd::splat(0.299) + values[1] * Simd::splat(0.587) + values[2] * Simd::splat(0.114) };
        luma.copy_to_slice(&mut output[i..i + 16]);
    }
    for (pixel, value) in bytes[end * stride..].chunks_exact(stride).zip(&mut output[end..]) {
        let sample = |c: usize| {
            if !wide {
                f64::from(pixel[c])
            } else {
                let pair = [pixel[c * 2], pixel[c * 2 + 1]];
                f64::from(if big_endian { u16::from_be_bytes(pair) } else { u16::from_le_bytes(pair) })
            }
        };
        *value = if C < 3 { sample(0) } else { sample(0) * 0.299 + sample(1) * 0.587 + sample(2) * 0.114 };
    }
}

fn grayscale(image: &Image) -> PyResult<Vec<f64>> {
    let mode = image.pixel_mode();
    if image.palette().is_some()
        || !matches!(
            mode,
            PixelMode::L
                | PixelMode::La
                | PixelMode::Rgb
                | PixelMode::Rgba
                | PixelMode::I16
                | PixelMode::I16L
                | PixelMode::I16B
                | PixelMode::I16N
                | PixelMode::F
                | PixelMode::RgbF
                | PixelMode::RgbaF
        )
    {
        return Err(PyValueError::new_err("phase correlation requires grayscale or RGB samples; convert explicitly"));
    }
    let bytes = image.raw_data()?;
    let mut values = zeros(image.width() as usize * image.height() as usize)?;
    if mode.is_float() {
        let channels = mode.channels();
        for (pixel, value) in bytes.chunks_exact(channels * 4).zip(&mut values) {
            let sample = |c: usize| f64::from(f32::from_le_bytes(pixel[c * 4..c * 4 + 4].try_into().unwrap()));
            *value = if channels == 1 { sample(0) } else { sample(0) * 0.299 + sample(1) * 0.587 + sample(2) * 0.114 };
        }
    } else {
        match mode.channels() {
            1 => grayscale_channels::<1>(image, bytes, &mut values),
            2 => grayscale_channels::<2>(image, bytes, &mut values),
            3 => grayscale_channels::<3>(image, bytes, &mut values),
            4 => grayscale_channels::<4>(image, bytes, &mut values),
            _ => unreachable!(),
        }
    }
    if values.iter().any(|v| !v.is_finite()) {
        return Err(PyValueError::new_err("phase correlation requires finite color samples"));
    }
    center(&mut values);
    Ok(values)
}

fn center(values: &mut [f64]) {
    let scale = values.iter().map(|v| v.abs()).fold(0.0, f64::max);
    if scale == 0.0 {
        return;
    }
    let mean = values.iter().map(|v| v / scale).sum::<f64>() / values.len() as f64;
    let (blocks, tail) = values.as_chunks_mut::<8>();
    for block in blocks {
        let value = Simd::from_array(*block) / Simd::splat(scale) - Simd::splat(mean);
        *block = value.to_array();
    }
    for value in tail {
        *value = *value / scale - mean;
    }
}

fn fft2(values: &mut [Complex<f64>], width: usize, direction: FftDirection, planner: &mut FftPlannerScalar<f64>) -> PyResult<()> {
    let height = values.len() / width;
    let rows = planner.plan_fft(width, direction);
    let columns = planner.plan_fft(height, direction);
    let mut scratch = zeros(rows.get_inplace_scratch_len().max(columns.get_inplace_scratch_len()))?;
    rows.process_with_scratch(values, &mut scratch);
    let mut column = zeros(height)?;
    for x in 0..width {
        for (y, value) in column.iter_mut().enumerate() {
            *value = values[y * width + x];
        }
        columns.process_with_scratch(&mut column, &mut scratch);
        for (y, &value) in column.iter().enumerate() {
            values[y * width + x] = value;
        }
    }
    Ok(())
}

fn spectrum(values: &[f64], width: usize, window: &[f64], planner: &mut FftPlannerScalar<f64>) -> PyResult<Vec<Complex<f64>>> {
    let mut result = zeros(values.len())?;
    for ((out, &value), &weight) in result.iter_mut().zip(values).zip(window) {
        *out = Complex::new(value * weight, 0.0);
    }
    fft2(&mut result, width, FftDirection::Forward, planner)?;
    Ok(result)
}

fn cross_power(reference: &[Complex<f64>], moving: &mut [Complex<f64>]) {
    let threshold = reference.iter().zip(moving.iter()).map(|(a, b)| a.norm() * b.norm()).fold(0.0, f64::max) * 1e-12;
    let end = moving.len() / 8 * 8;
    for i in (0..end).step_by(8) {
        let ar = Simd::<f64, 8>::from_array(std::array::from_fn(|c| reference[i + c].re));
        let ai = Simd::<f64, 8>::from_array(std::array::from_fn(|c| reference[i + c].im));
        let br = Simd::<f64, 8>::from_array(std::array::from_fn(|c| moving[i + c].re));
        let bi = Simd::<f64, 8>::from_array(std::array::from_fn(|c| moving[i + c].im));
        let real = ar * br + ai * bi;
        let imag = ai * br - ar * bi;
        let magnitude = (real * real + imag * imag).sqrt();
        let valid = magnitude.simd_gt(Simd::splat(threshold));
        let denominator = magnitude.simd_max(Simd::splat(f64::MIN_POSITIVE));
        let real = valid.select(real / denominator, Simd::splat(0.0)).to_array();
        let imag = valid.select(imag / denominator, Simd::splat(0.0)).to_array();
        for c in 0..8 {
            moving[i + c] = Complex::new(real[c], imag[c]);
        }
    }
    for (a, b) in reference[end..].iter().zip(&mut moving[end..]) {
        let value = *a * b.conj();
        let magnitude = value.norm();
        *b = if magnitude > threshold { value / magnitude } else { Complex::default() };
    }
}

fn displacement(index: usize, length: usize, correction: f64) -> f64 {
    let value = index as f64 + correction;
    (value + length as f64 * 0.5).rem_euclid(length as f64) - length as f64 * 0.5
}

fn parabola(left: f64, peak: f64, right: f64) -> f64 {
    let denominator = left - 2.0 * peak + right;
    if denominator < -f64::EPSILON { (0.5 * (left - right) / denominator).clamp(-0.5, 0.5) } else { 0.0 }
}

fn correlate(
    reference: &[Complex<f64>], moving: &mut [Complex<f64>], width: usize, planner: &mut FftPlannerScalar<f64>,
) -> PyResult<(f64, f64, f64)> {
    cross_power(reference, moving);
    fft2(moving, width, FftDirection::Inverse, planner)?;
    let height = moving.len() / width;
    let mut best = 0;
    for i in 1..moving.len() {
        if moving[i].re > moving[best].re {
            best = i;
        }
    }
    let peak = moving[best].re;
    let confidence = (peak / moving.len() as f64).clamp(0.0, 1.0);
    if confidence <= 1e-12 {
        return Ok((0.0, 0.0, 0.0));
    }
    let x = best % width;
    let y = best / width;
    let dx = parabola(moving[y * width + (x + width - 1) % width].re, peak, moving[y * width + (x + 1) % width].re);
    let dy = parabola(moving[((y + height - 1) % height) * width + x].re, peak, moving[((y + 1) % height) * width + x].re);
    Ok((displacement(x, width, dx), displacement(y, height, dy), confidence))
}

fn rotate(source: &[f64], width: usize, angle: f64) -> PyResult<Vec<f64>> {
    if angle == 0.0 {
        return copy(source);
    }
    let height = source.len() / width;
    let (sin, cos) = angle.to_radians().sin_cos();
    let cx = (width - 1) as f64 * 0.5;
    let cy = (height - 1) as f64 * 0.5;
    let sample = |x: isize, y: isize| {
        if x >= 0 && y >= 0 && x < width as isize && y < height as isize { source[y as usize * width + x as usize] } else { 0.0 }
    };
    let mut result = zeros(source.len())?;
    for (i, value) in result.iter_mut().enumerate() {
        let x = (i % width) as f64 - cx;
        let y = (i / width) as f64 - cy;
        let sx = cx + cos * x - sin * y;
        let sy = cy + sin * x + cos * y;
        let ix = sx.floor() as isize;
        let iy = sy.floor() as isize;
        let fx = sx - ix as f64;
        let fy = sy - iy as f64;
        *value = (sample(ix, iy) * (1.0 - fx) + sample(ix + 1, iy) * fx) * (1.0 - fy)
            + (sample(ix, iy + 1) * (1.0 - fx) + sample(ix + 1, iy + 1) * fx) * fy;
    }
    Ok(result)
}

#[pyfunction]
fn ops_phase_correlate(
    py: Python<'_>, reference: &Image, image: &Image, max_rotation: f64, rotation_step: f64, window: Option<&str>,
) -> PyResult<(f64, f64, f64, f64)> {
    let (width, height) = reference.size();
    if reference.size() != image.size() || width < 3 || height < 3 || u64::from(width) * u64::from(height) > MAX_PIXELS as u64 {
        return Err(PyValueError::new_err("phase correlation requires equal dimensions, each at least 3, and at most 4194304 pixels"));
    }
    if !max_rotation.is_finite()
        || !(0.0..=15.0).contains(&max_rotation)
        || !rotation_step.is_finite()
        || rotation_step <= 0.0
        || (max_rotation / rotation_step).ceil() > 60.0
    {
        return Err(PyValueError::new_err("max_rotation must be in [0, 15], rotation_step positive, with at most 60 steps per direction"));
    }
    if !matches!(window, None | Some("hann")) {
        return Err(PyValueError::new_err("window must be 'hann' or None"));
    }
    let width = width as usize;
    let height = height as usize;
    let reference = grayscale(reference)?;
    let image = grayscale(image)?;
    py.detach(|| {
        let mut weights = zeros(width * height)?;
        for (i, value) in weights.iter_mut().enumerate() {
            *value = if window.is_some() {
                0.25 * (1.0 - (TAU * (i % width) as f64 / (width - 1) as f64).cos()) * (1.0 - (TAU * (i / width) as f64 / (height - 1) as f64).cos())
            } else {
                1.0
            };
        }
        let mut planner = FftPlannerScalar::new();
        let reference = spectrum(&reference, width, &weights, &mut planner)?;
        let mut moving = spectrum(&image, width, &weights, &mut planner)?;
        let (dx, dy, confidence) = correlate(&reference, &mut moving, width, &mut planner)?;
        let mut best = (dx, dy, 0.0, confidence);
        let steps = (max_rotation / rotation_step).ceil() as usize;
        for step in 1..=steps {
            for sign in [-1.0, 1.0] {
                let angle = sign * (step as f64 * rotation_step).min(max_rotation);
                let rotated = rotate(&image, width, angle)?;
                let mut moving = spectrum(&rotated, width, &weights, &mut planner)?;
                let (dx, dy, confidence) = correlate(&reference, &mut moving, width, &mut planner)?;
                if confidence > best.3 {
                    best = (dx, dy, angle, confidence);
                }
            }
        }
        Ok(best)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cross_power_matches_scalar_with_unaligned_tails_and_guards() {
        for count in [1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 65] {
            let reference: Vec<_> = (0..count + 2).map(|i| Complex::new((i * 17 % 31) as f64, (i * 7 % 13) as f64)).collect();
            let moving: Vec<_> = (0..count + 2)
                .map(|i| if i % 3 == 0 { Complex::default() } else { Complex::new((i * 3 % 19) as f64, -((i * 5 % 23) as f64)) })
                .collect();
            let reference = &reference[1..count + 1];
            let mut actual = moving.clone();
            cross_power(reference, &mut actual[1..count + 1]);
            let threshold = reference.iter().zip(&moving[1..count + 1]).map(|(a, b)| a.norm() * b.norm()).fold(0.0, f64::max) * 1e-12;
            for ((a, b), actual) in reference.iter().zip(&moving[1..count + 1]).zip(&actual[1..count + 1]) {
                let value = *a * b.conj();
                let expected = if value.norm() > threshold { value / value.norm() } else { Complex::default() };
                assert!((*actual - expected).norm() < 1e-12);
            }
            assert_eq!(actual[0], moving[0]);
            assert_eq!(actual[count + 1], moving[count + 1]);
        }
    }

    #[test]
    fn fft_round_trip_on_rectangular_and_prime_sizes() {
        for (width, height) in [(3, 5), (16, 17), (31, 19), (64, 32)] {
            let original: Vec<_> = (0..width * height).map(|i| Complex::new((i * 17 % 251) as f64, 0.0)).collect();
            let mut values = original.clone();
            let mut planner = FftPlannerScalar::new();
            fft2(&mut values, width, FftDirection::Forward, &mut planner).unwrap();
            fft2(&mut values, width, FftDirection::Inverse, &mut planner).unwrap();
            for (actual, expected) in values.iter().zip(original) {
                assert!((*actual / (width * height) as f64 - expected).norm() < 1e-9);
            }
        }
    }

    #[test]
    fn center_matches_scalar_at_vector_boundaries() {
        for count in [1, 7, 8, 9, 15, 16, 17, 33, 65] {
            let mut values: Vec<_> = (0..count + 2).map(|i| (i * 17 % 23) as f64 - 11.0).collect();
            let mut expected = values.clone();
            let scale = values[1..count + 1].iter().map(|v| v.abs()).fold(0.0, f64::max);
            let mean = values[1..count + 1].iter().map(|v| v / scale).sum::<f64>() / count as f64;
            for value in &mut expected[1..count + 1] {
                *value = *value / scale - mean;
            }
            center(&mut values[1..count + 1]);
            assert_eq!(values, expected);
        }
    }

    #[test]
    fn grayscale_simd_matches_scalar_with_wide_unaligned_tails() {
        fn check<const C: usize>(mode: PixelMode, depth: u8) {
            let sample_bytes = if depth == 8 { 1 } else { 2 };
            for count in [1, 15, 16, 17, 31, 32, 33, 65] {
                let mut storage = vec![173];
                for i in 0..count * C {
                    let value = ((i * 997 + 113) % (1 << depth)) as u16;
                    if sample_bytes == 1 {
                        storage.push(value as u8);
                    } else {
                        storage.extend(if mode.is_big_endian() { value.to_be_bytes() } else { value.to_le_bytes() });
                    }
                }
                storage.push(173);
                let bytes = &storage[1..storage.len() - 1];
                let image = Image::from_raw_bytes(count as u32, 1, mode, bytes.to_vec(), depth).unwrap();
                let mut result = vec![173.0; count + 2];
                grayscale_channels::<C>(&image, bytes, &mut result[1..count + 1]);
                for (pixel, actual) in bytes.chunks_exact(C * sample_bytes).zip(&result[1..count + 1]) {
                    let sample = |c: usize| {
                        if sample_bytes == 1 {
                            f64::from(pixel[c])
                        } else {
                            let pair = [pixel[c * 2], pixel[c * 2 + 1]];
                            f64::from(if mode.is_big_endian() { u16::from_be_bytes(pair) } else { u16::from_le_bytes(pair) })
                        }
                    };
                    let expected = if C < 3 { sample(0) } else { sample(0) * 0.299 + sample(1) * 0.587 + sample(2) * 0.114 };
                    assert!((actual - expected).abs() < 1e-10);
                }
                assert_eq!(result[0], 173.0);
                assert_eq!(result[count + 1], 173.0);
            }
        }
        check::<1>(PixelMode::L, 8);
        check::<2>(PixelMode::La, 8);
        check::<3>(PixelMode::Rgb, 8);
        check::<4>(PixelMode::Rgba, 8);
        for depth in [10, 12, 16] {
            check::<1>(PixelMode::L, depth);
            check::<3>(PixelMode::Rgb, depth);
            check::<4>(PixelMode::Rgba, depth);
        }
        check::<1>(PixelMode::I16B, 16);
    }
}
