use std::simd::Simd;
use std::simd::cmp::SimdOrd;
use std::simd::num::{SimdFloat, SimdUint};

use blanket_core::pixels::Samples;
use blanket_core::{Image, PixelMode, samples};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub(crate) fn lut(image: &Image, mode: &str, channels: usize, size: (usize, usize, usize), table: &[f32]) -> PyResult<Image> {
    let target = PixelMode::parse(mode)?;
    let dimensions = <[usize; 3]>::from(size);
    let input_channels = image.pixel_mode().channels();
    let output_channels = target.channels();
    if !matches!(image.pixel_mode(), PixelMode::Rgb | PixelMode::Rgba)
        || !matches!(target, PixelMode::Rgb | PixelMode::Rgba)
        || !(3..=4).contains(&channels)
        || output_channels < channels
        || output_channels > channels && output_channels > input_channels
    {
        return Err(PyValueError::new_err("image has wrong mode"));
    }
    if dimensions.iter().any(|s| !(2..=65).contains(s)) {
        return Err(PyValueError::new_err("Table size in any dimension should be from 2 to 65"));
    }
    if table.len() != channels * size.0 * size.1 * size.2 {
        return Err(PyValueError::new_err("table has wrong number of elements"));
    }
    let source = samples::view(image)?;
    let maximum = f64::from(samples::maximum(image));
    let strides = [channels, channels * size.0, channels * size.0 * size.1];
    let mut values = vec![0; source.len() / input_channels * output_channels];
    let end = source.len() / input_channels / 16 * 16;
    for i in (0..end).step_by(16) {
        use std::simd::StdFloat;
        let input = if input_channels == 4 {
            blanket_core::pixels::load_samples::<4>(&source[i * 4..])
        } else {
            let rgb = blanket_core::pixels::load_samples::<3>(&source[i * 3..]);
            [rgb[0], rgb[1], rgb[2], Samples::splat(0)]
        };
        let positions: [Simd<f64, 16>; 3] =
            std::array::from_fn(|c| input[c].cast::<f64>() / Simd::splat(maximum) * Simd::splat((dimensions[c] - 1) as f64));
        let lower: [Simd<usize, 16>; 3] = std::array::from_fn(|c| positions[c].cast::<usize>().simd_min(Simd::splat(dimensions[c] - 2)));
        let fractions: [Simd<f64, 16>; 3] = std::array::from_fn(|c| positions[c] - lower[c].cast::<f64>());
        let mut result = [Samples::splat(0); 4];
        for c in 0..output_channels {
            if c >= channels {
                result[c] = input[c];
                continue;
            }
            let mut value = Simd::splat(0.0);
            for corner in 0..8 {
                let mut index = Simd::splat(c);
                let mut weight = Simd::splat(1.0);
                for axis in 0..3 {
                    let upper = (corner >> axis) & 1;
                    index += (lower[axis] + Simd::splat(upper)) * Simd::splat(strides[axis]);
                    weight *= if upper == 1 { fractions[axis] } else { Simd::splat(1.0) - fractions[axis] };
                }
                value += Simd::<f32, 16>::gather_or_default(table, index).cast::<f64>() * weight;
            }
            result[c] = (value * Simd::splat(maximum)).round().cast::<u16>().simd_min(Samples::splat(maximum as u16));
        }
        if output_channels == 4 {
            blanket_core::pixels::store_samples(&mut values[i * 4..], result);
        } else {
            blanket_core::pixels::store_samples(&mut values[i * 3..], [result[0], result[1], result[2]]);
        }
    }
    for (pixel, dst) in
        source[end * input_channels..].chunks_exact(input_channels).zip(values[end * output_channels..].chunks_exact_mut(output_channels))
    {
        let positions: [f64; 3] = std::array::from_fn(|c| f64::from(pixel[c]) / maximum * (dimensions[c] - 1) as f64);
        let lower: [usize; 3] = std::array::from_fn(|c| (positions[c] as usize).min(dimensions[c] - 2));
        let fractions: [f64; 3] = std::array::from_fn(|c| positions[c] - lower[c] as f64);
        for (c, dst) in dst.iter_mut().enumerate() {
            if c >= channels {
                *dst = pixel[c];
                continue;
            }
            let mut value = 0.0;
            for corner in 0..8 {
                let mut index = c;
                let mut weight = 1.0;
                for axis in 0..3 {
                    let upper = (corner >> axis) & 1;
                    index += (lower[axis] + upper) * strides[axis];
                    weight *= if upper == 1 { fractions[axis] } else { 1.0 - fractions[axis] };
                }
                value += f64::from(table[index]) * weight;
            }
            *dst = (value * maximum).round().clamp(0.0, maximum) as u16;
        }
    }
    Image::from_samples(image.width(), image.height(), target, values, image.bit_depth(), None)
}
