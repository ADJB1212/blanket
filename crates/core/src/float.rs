use std::simd::cmp::{SimdPartialEq, SimdPartialOrd};
use std::simd::num::{SimdFloat, SimdUint};
use std::simd::{Select, Simd, StdFloat, ToBytes, simd_swizzle};

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::{Image, PixelMode};

pub fn composite(background: &mut [f32], foreground: &[f32]) {
    assert_eq!(background.len(), foreground.len());
    assert!(background.len().is_multiple_of(4));
    let end = background.len() / 16 * 16;
    for i in (0..end).step_by(16) {
        let b = Simd::<f32, 16>::from_slice(&background[i..]);
        let f = Simd::<f32, 16>::from_slice(&foreground[i..]);
        let alpha = simd_swizzle!(f, [3, 3, 3, 3, 7, 7, 7, 7, 11, 11, 11, 11, 15, 15, 15, 15]);
        let result = f + b * (Simd::splat(1.0) - alpha);
        alpha.simd_eq(Simd::splat(1.0)).select(f, result).copy_to_slice(&mut background[i..i + 16]);
    }
    for (b, f) in background[end..].as_chunks_mut::<4>().0.iter_mut().zip(foreground[end..].as_chunks::<4>().0) {
        for c in 0..4 {
            b[c] = if f[3] == 1.0 { f[c] } else { f[c] + b[c] * (1.0 - f[3]) };
        }
    }
}

fn alpha(values: &mut [f32], divide: bool, premultiply: bool) {
    let end = values.len() / 16 * 16;
    let zero = Simd::<f32, 16>::splat(0.0);
    let one = Simd::splat(1.0);
    let colors = Simd::<u32, 16>::from_array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]).simd_eq(Simd::splat(0));
    for i in (0..end).step_by(16) {
        let v = Simd::<f32, 16>::from_slice(&values[i..]);
        let a = simd_swizzle!(v, [3, 3, 3, 3, 7, 7, 7, 7, 11, 11, 11, 11, 15, 15, 15, 15]);
        let mut a = a.simd_clamp(zero, one);
        if !divide {
            a = a.is_nan().select(zero, a);
        }
        let color = if premultiply { a.simd_eq(zero).select(zero, if divide { v / a } else { v * a }) } else { v };
        colors.select(color, a).copy_to_slice(&mut values[i..i + 16]);
    }
    for p in values[end..].as_chunks_mut::<4>().0 {
        let a = if !divide && p[3].is_nan() { 0.0 } else { p[3].clamp(0.0, 1.0) };
        if premultiply {
            for v in &mut p[..3] {
                *v = if a == 0.0 {
                    0.0
                } else if divide {
                    *v / a
                } else {
                    *v * a
                };
            }
        }
        p[3] = a;
    }
}

pub fn convert(image: &Image, mode: PixelMode, depth: Option<u8>) -> PyResult<Image> {
    let color = |m| matches!(m, PixelMode::Rgb | PixelMode::Rgba | PixelMode::RgbF | PixelMode::RgbaF);
    if !color(image.mode) || !color(mode) {
        return Err(PyValueError::new_err("float color conversion requires RGB or RGBA modes"));
    }
    let float = mode.is_float();
    let depth = depth.unwrap_or(if float { 32 } else { 8 });
    if (float && depth != 32) || (!float && !matches!(depth, 8 | 10 | 12 | 16)) {
        return Err(PyValueError::new_err("invalid color sample depth"));
    }
    convert_values(image, mode, depth, values(image)?)
}

fn convert_values(image: &Image, mode: PixelMode, depth: u8, mut source: Vec<f32>) -> PyResult<Image> {
    if image.mode.channels() != mode.channels() {
        let mut mapped = vec![0.0; source.len() / image.mode.channels() * mode.channels()];
        if mode.channels() == 4 {
            crate::pixels::float_color_layout::<3, 4>(&source, &mut mapped);
        } else {
            crate::pixels::float_color_layout::<4, 3>(&source, &mut mapped);
        }
        source = mapped;
    }
    color_output(image.width, image.height, mode, depth, &source)
}

fn color_output(width: u32, height: u32, mode: PixelMode, depth: u8, values: &[f32]) -> PyResult<Image> {
    if mode.is_float() {
        let bytes = values.iter().flat_map(|value| value.to_le_bytes()).collect();
        return Image::from_color_float_bytes(width, height, mode, bytes);
    }
    let bytes = quantize(values, depth);
    if depth == 8 { Image::from_pixels(width, height, mode, bytes, None) } else { Image::from_sample_bytes(width, height, mode, bytes, depth, None) }
}

pub fn quantize(source: &[f32], depth: u8) -> Vec<u8> {
    assert!(matches!(depth, 8 | 10 | 12 | 16));
    let maximum = ((1_u32 << depth) - 1) as f32;
    let stride = if depth == 8 { 1 } else { 2 };
    let mut output = vec![0; source.len() * stride];
    let end = source.len() / 16 * 16;
    for i in (0..end).step_by(16) {
        let values = Simd::<f32, 16>::from_slice(&source[i..]);
        let samples = (values.simd_clamp(Simd::splat(0.0), Simd::splat(1.0)) * Simd::splat(maximum)).round().cast::<u16>();
        if stride == 1 {
            samples.cast::<u8>().copy_to_slice(&mut output[i..i + 16]);
        } else {
            samples.to_le_bytes().copy_to_slice(&mut output[i * 2..(i + 16) * 2]);
        }
    }
    for (i, &v) in source.iter().enumerate().skip(end) {
        let sample = (v.clamp(0.0, 1.0) * maximum).round() as u16;
        if stride == 1 {
            output[i] = sample as u8;
        } else {
            output[i * 2..i * 2 + 2].copy_from_slice(&sample.to_le_bytes());
        }
    }
    output
}

pub fn values(image: &Image) -> PyResult<Vec<f32>> {
    let data = image.raw_data()?;
    Ok(if image.mode.is_float() {
        data.as_chunks::<4>().0.iter().map(|v| f32::from_le_bytes(*v)).collect()
    } else if matches!(image.mode, PixelMode::Rgb | PixelMode::Rgba) {
        let maximum = ((1_u32 << image.bit_depth) - 1) as f32;
        let bytes = if image.bit_depth == 8 { 1 } else { 2 };
        let mut result = vec![0.0; data.len() / bytes];
        let end = result.len() / 16 * 16;
        for i in (0..end).step_by(16) {
            let samples: Simd<f32, 16> = if bytes == 1 {
                crate::pixels::Bytes::from_slice(&data[i..]).cast()
            } else {
                crate::pixels::load_u16::<false>(&data[i * 2..]).cast()
            };
            (samples / Simd::splat(maximum)).copy_to_slice(&mut result[i..i + 16]);
        }
        for (i, v) in result.iter_mut().enumerate().skip(end) {
            *v = if bytes == 1 { data[i] as f32 } else { u16::from_le_bytes([data[i * 2], data[i * 2 + 1]]) as f32 } / maximum;
        }
        result
    } else {
        return Err(PyValueError::new_err("requires RGB or RGBA color"));
    })
}

pub fn transfer(v: f32, decode: bool) -> f32 {
    let a = v.abs();
    v.signum()
        * if decode {
            if a <= 0.04045 { a / 12.92 } else { ((a + 0.055) / 1.055).powf(2.4) }
        } else if a <= 0.0031308 {
            a * 12.92
        } else {
            1.055 * a.powf(1.0 / 2.4) - 0.055
        }
}

fn transfer_colors(values: &mut [f32], channels: usize, decode: bool) {
    let end = values.len() / 8 * 8;
    for i in (0..end).step_by(8) {
        let v = Simd::<f32, 8>::from_slice(&values[i..]);
        let a = v.abs();
        let (threshold, linear, base, exponent) = if decode {
            (0.04045, a / Simd::splat(12.92), (a + Simd::splat(0.055)) / Simd::splat(1.055), 2.4_f32)
        } else {
            (0.0031308, a * Simd::splat(12.92), a, 1.0_f32 / 2.4)
        };
        let colors = Simd::<u32, 8>::from_array(std::array::from_fn(|c| u32::from((i + c) % channels < 3))).simd_ne(Simd::splat(0));
        let linear_range = a.simd_le(Simd::splat(threshold));
        let result = if (linear_range | !colors).all() {
            v.signum() * linear
        } else {
            let power = (base.cast::<f64>().ln() * Simd::splat(f64::from(exponent))).exp().cast::<f32>();
            let curved = if decode { power } else { Simd::splat(1.055) * power - Simd::splat(0.055) };
            v.signum() * linear_range.select(linear, curved)
        };
        colors.select(result, v).copy_to_slice(&mut values[i..i + 8]);
    }
    for (i, v) in values.iter_mut().enumerate().skip(end) {
        if i % channels < 3 {
            *v = transfer(*v, decode);
        }
    }
}

pub fn prepare(image: &Image, srgb: bool, premultiply: bool) -> PyResult<Image> {
    let values = prepare_values(image, srgb, premultiply)?;
    Image::from_color_float_bytes(
        image.width,
        image.height,
        if image.mode.channels() == 4 { PixelMode::RgbaF } else { PixelMode::RgbF },
        values.into_iter().flat_map(f32::to_le_bytes).collect(),
    )
}

pub fn prepare_values(image: &Image, srgb: bool, premultiply: bool) -> PyResult<Vec<f32>> {
    if !matches!(image.mode, PixelMode::Rgb | PixelMode::Rgba | PixelMode::RgbF | PixelMode::RgbaF) {
        return Err(PyValueError::new_err("linear-light processing requires RGB or RGBA color"));
    }
    let mut values = values(image)?;
    let channels = image.mode.channels();
    if srgb {
        transfer_colors(&mut values, channels, true);
    }
    if channels == 4 {
        alpha(&mut values, false, premultiply);
    }
    Ok(values)
}

pub fn finish(image: &Image, original: &Image, srgb: bool, premultiply: bool) -> PyResult<Image> {
    finish_values(image, original, srgb, premultiply, values(image)?)
}

pub fn finish_values(image: &Image, original: &Image, srgb: bool, premultiply: bool, mut values: Vec<f32>) -> PyResult<Image> {
    if image.mode.channels() == 4 {
        alpha(&mut values, true, premultiply);
    }
    if srgb {
        transfer_colors(&mut values, image.mode.channels(), false);
    }
    convert_values(image, original.mode, original.bit_depth, values)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_transfer_preserves_alpha_and_signed_zero() {
        for channels in [3, 4] {
            for decode in [false, true] {
                for count in 0..35 {
                    for offset in 0..8 {
                        let mut storage = vec![123.0; offset + count * channels + 7];
                        let values = &mut storage[offset..offset + count * channels];
                        for (i, value) in values.iter_mut().enumerate() {
                            *value = if i % channels == 3 { f32::NAN } else { [-0.0, 0.0, -0.001, 0.003][i % 4] };
                        }
                        let expected: Vec<_> =
                            values.iter().enumerate().map(|(i, &v)| if i % channels < 3 { transfer(v, decode) } else { v }).collect();
                        transfer_colors(values, channels, decode);
                        equal(values, &expected);
                        assert!(storage[..offset].iter().chain(&storage[offset + count * channels..]).all(|&v| v == 123.0));
                    }
                }
            }
        }
    }

    #[test]
    fn finish_matches_materialized_float_conversion() {
        let samples = [-0.0, 0.0, -2.0, 0.0031308, 0.04045, 0.5, 1.0, 4.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY];
        for width in [0, 1, 3, 4, 5, 17, 33] {
            for mode in [PixelMode::RgbF, PixelMode::RgbaF] {
                let data: Vec<_> = (0..width as usize * mode.channels()).map(|i| samples[i % samples.len()]).collect();
                let image = color_output(width, 1, mode, 32, &data).unwrap();
                for target in [PixelMode::Rgb, PixelMode::Rgba, PixelMode::RgbF, PixelMode::RgbaF] {
                    for depth in if target.is_float() { &[32][..] } else { &[8, 10, 12, 16][..] } {
                        let original = convert(&image, target, Some(*depth)).unwrap();
                        for srgb in [false, true] {
                            for premultiply in [false, true] {
                                let mut expected = values(&image).unwrap();
                                if mode.channels() == 4 {
                                    alpha(&mut expected, true, premultiply);
                                }
                                if srgb {
                                    transfer_colors(&mut expected, mode.channels(), false);
                                }
                                let intermediate = color_output(width, 1, mode, 32, &expected).unwrap();
                                let expected = convert(&intermediate, target, Some(*depth)).unwrap();
                                let actual = finish(&image, &original, srgb, premultiply).unwrap();
                                assert_eq!(actual.raw_data().unwrap(), expected.raw_data().unwrap());
                                assert_eq!((actual.mode, actual.bit_depth), (target, *depth));
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn quantization_matches_scalar() {
        for depth in [8, 10, 12, 16] {
            let maximum = ((1_u32 << depth) - 1) as f32;
            let values = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.0, 0.0, -1.0, 1.0, 2.0, 0.5 / maximum, 127.5 / maximum, 0.5];
            for count in 0..67 {
                for offset in 0..16 {
                    let source: Vec<_> = (0..count + offset).map(|i| values[i % values.len()]).collect();
                    let source = &source[offset..];
                    let expected: Vec<_> = source
                        .iter()
                        .flat_map(|&v| {
                            let sample = (v.clamp(0.0, 1.0) * maximum).round() as u16;
                            if depth == 8 { vec![sample as u8] } else { sample.to_le_bytes().to_vec() }
                        })
                        .collect();
                    assert_eq!(quantize(source, depth), expected);
                }
            }
        }
    }

    #[test]
    fn transfer_vectors_match_scalar() {
        let samples = [-0.0, 0.0, 0.04045, 0.0031308, -0.5, 1.0, 4.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY, f32::MIN_POSITIVE];
        for channels in [3, 4] {
            for count in 0..35 {
                for offset in 0..8 {
                    for decode in [false, true] {
                        let mut data = vec![123.0; offset + count * channels + 8];
                        let region = &mut data[offset..offset + count * channels];
                        for (i, v) in region.iter_mut().enumerate() {
                            *v = samples[i % samples.len()];
                        }
                        let expected: Vec<_> =
                            region.iter().enumerate().map(|(i, &v)| if i % channels < 3 { transfer(v, decode) } else { v }).collect();
                        transfer_colors(region, channels, decode);
                        for (&a, &b) in region.iter().zip(&expected) {
                            assert!(a.to_bits() == b.to_bits() || a.is_nan() && b.is_nan() || (a - b).abs() <= b.abs() * 2e-6, "{a} != {b}");
                        }
                        assert!(data[..offset].iter().chain(&data[offset + count * channels..]).all(|&v| v == 123.0));
                    }
                }
            }
        }
    }

    fn equal(actual: &[f32], expected: &[f32]) {
        for (a, b) in actual.iter().zip(expected) {
            assert!(a.to_bits() == b.to_bits() || a.is_nan() && b.is_nan(), "{a:?} != {b:?}");
        }
    }

    #[test]
    fn alpha_and_composite_boundaries() {
        let samples = [-0.0, 0.0, -2.0, 0.5, 1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY];
        for count in 0..35 {
            for offset in 0..4 {
                let original: Vec<_> = (0..count * 4).map(|i| samples[(i * 3 + i / 4) % samples.len()]).collect();
                for divide in [false, true] {
                    for premultiply in [false, true] {
                        let mut expected = original.clone();
                        for p in expected.as_chunks_mut::<4>().0 {
                            let a = if !divide && p[3].is_nan() { 0.0 } else { p[3].clamp(0.0, 1.0) };
                            for v in &mut p[..3] {
                                if premultiply {
                                    *v = if a == 0.0 {
                                        0.0
                                    } else if divide {
                                        *v / a
                                    } else {
                                        *v * a
                                    };
                                }
                            }
                            p[3] = a;
                        }
                        let mut guarded = vec![123.0; offset + original.len() + 4];
                        guarded[offset..offset + original.len()].copy_from_slice(&original);
                        alpha(&mut guarded[offset..offset + original.len()], divide, premultiply);
                        equal(&guarded[offset..offset + original.len()], &expected);
                        assert!(guarded[..offset].iter().chain(&guarded[offset + original.len()..]).all(|&v| v == 123.0));
                    }
                }
                let mut expected = original.clone();
                for (b, f) in expected.as_chunks_mut::<4>().0.iter_mut().zip(original.as_chunks::<4>().0) {
                    for c in 0..4 {
                        b[c] = if f[3] == 1.0 { f[c] } else { f[c] + b[c] * (1.0 - f[3]) };
                    }
                }
                let mut guarded = vec![123.0; offset + original.len() + 4];
                guarded[offset..offset + original.len()].copy_from_slice(&original);
                composite(&mut guarded[offset..offset + original.len()], &original);
                equal(&guarded[offset..offset + original.len()], &expected);
                assert!(guarded[..offset].iter().chain(&guarded[offset + original.len()..]).all(|&v| v == 123.0));
            }
        }
    }

    #[test]
    fn opaque_composite_replaces_nonfinite_background() {
        for count in [0, 1, 3, 4, 5, 15, 16, 17] {
            for offset in 0..16 {
                let original = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 1.0].repeat(count);
                let foreground = [-0.0, 0.5, 0.75, 1.0].repeat(count);
                let mut guarded = vec![123.0; offset + original.len() + 4];
                let mut source = vec![173.0; offset + foreground.len() + 4];
                guarded[offset..offset + original.len()].copy_from_slice(&original);
                source[offset..offset + foreground.len()].copy_from_slice(&foreground);
                composite(&mut guarded[offset..offset + original.len()], &source[offset..offset + foreground.len()]);
                equal(&guarded[offset..offset + original.len()], &foreground);
                assert!(guarded[..offset].iter().chain(&guarded[offset + original.len()..]).all(|&v| v == 123.0));
                assert!(source[..offset].iter().chain(&source[offset + foreground.len()..]).all(|&v| v == 173.0));
            }
        }
    }

    #[test]
    fn normalized_samples_match_scalar() {
        for depth in [8, 10, 12, 16] {
            for width in 0..35 {
                let maximum = (1_u32 << depth) - 1;
                let samples: Vec<_> = (0..width * 3).map(|i| ((i * 7919) & maximum) as u16).collect();
                let image = if depth == 8 {
                    Image::from_pixels(width, 1, PixelMode::Rgb, samples.iter().map(|&v| v as u8).collect(), None)
                } else {
                    Image::from_samples(width, 1, PixelMode::Rgb, samples.clone(), depth, None)
                }
                .unwrap();
                equal(&values(&image).unwrap(), &samples.iter().map(|&v| v as f32 / maximum as f32).collect::<Vec<_>>());
            }
        }
    }
}
