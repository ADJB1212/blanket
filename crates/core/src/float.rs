use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::{Image, PixelMode};

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
    let source = values(image)?;
    let mut bytes = Vec::new();
    let maximum = ((1_u32 << depth.min(16)) - 1) as f32;
    for p in source.chunks_exact(image.mode.channels()) {
        for c in 0..mode.channels() {
            let v = p.get(c).copied().unwrap_or(1.0);
            if float {
                bytes.extend_from_slice(&v.to_le_bytes());
            } else {
                let v = (v.clamp(0.0, 1.0) * maximum).round() as u16;
                if depth == 8 {
                    bytes.push(v as u8);
                } else {
                    bytes.extend_from_slice(&v.to_le_bytes());
                }
            }
        }
    }
    if float {
        Image::from_color_float_bytes(image.width, image.height, mode, bytes)
    } else if depth == 8 {
        Image::from_pixels(image.width, image.height, mode, bytes, None)
    } else {
        Image::from_sample_bytes(image.width, image.height, mode, bytes, depth, None)
    }
}

pub fn values(image: &Image) -> PyResult<Vec<f32>> {
    let data = image.raw_data()?;
    Ok(if image.mode.is_float() {
        data.as_chunks::<4>().0.iter().map(|v| f32::from_le_bytes(*v)).collect()
    } else if matches!(image.mode, PixelMode::Rgb | PixelMode::Rgba) {
        let maximum = ((1_u32 << image.bit_depth) - 1) as f32;
        if image.bit_depth == 8 {
            data.iter().map(|&v| v as f32 / maximum).collect()
        } else {
            data.as_chunks::<2>().0.iter().map(|v| u16::from_le_bytes(*v) as f32 / maximum).collect()
        }
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

pub fn prepare(image: &Image, srgb: bool, premultiply: bool) -> PyResult<Image> {
    if !matches!(image.mode, PixelMode::Rgb | PixelMode::Rgba | PixelMode::RgbF | PixelMode::RgbaF) {
        return Err(PyValueError::new_err("linear-light processing requires RGB or RGBA color"));
    }
    let mut values = values(image)?;
    let channels = image.mode.channels();
    for p in values.chunks_exact_mut(channels) {
        let alpha = if channels == 4 { if p[3].is_nan() { 0.0 } else { p[3].clamp(0.0, 1.0) } } else { 1.0 };
        for v in &mut p[..3] {
            if srgb {
                *v = transfer(*v, true);
            }
            if premultiply {
                *v = if alpha == 0.0 { 0.0 } else { *v * alpha };
            }
        }
        if channels == 4 {
            p[3] = alpha;
        }
    }
    Image::from_color_float_bytes(
        image.width,
        image.height,
        if channels == 4 { PixelMode::RgbaF } else { PixelMode::RgbF },
        values.into_iter().flat_map(f32::to_le_bytes).collect(),
    )
}

pub fn finish(image: &Image, original: &Image, srgb: bool, premultiply: bool) -> PyResult<Image> {
    let mut values = values(image)?;
    for p in values.chunks_exact_mut(image.mode.channels()) {
        let alpha = if p.len() == 4 { p[3].clamp(0.0, 1.0) } else { 1.0 };
        for v in &mut p[..3] {
            if premultiply {
                *v = if alpha == 0.0 { 0.0 } else { *v / alpha };
            }
            if srgb {
                *v = transfer(*v, false);
            }
        }
        if p.len() == 4 {
            p[3] = alpha;
        }
    }
    let result = Image::from_color_float_bytes(image.width, image.height, image.mode, values.into_iter().flat_map(f32::to_le_bytes).collect())?;
    convert(&result, original.mode, Some(original.bit_depth))
}
