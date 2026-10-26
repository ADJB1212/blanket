//! Explicit HDR transfer decoding and SDR rendering.

use blanket_core::parallel::{CHUNK_PIXELS, chunks_mut};
use blanket_core::raster::{Image, PixelMode, alloc_pixels};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(hdr_tone_map, module)?)?;
    Ok(())
}

fn linearize(value: f64, transfer: &str, reference_white: f64) -> f64 {
    let value = if value.is_nan() { 0.0 } else { value.clamp(0.0, 1e30) };
    match transfer {
        "pq" => {
            let p = value.min(1.0).powf(32.0 / 2523.0);
            let light = ((p - 3424.0 / 4096.0).max(0.0) / (2413.0 / 128.0 - 2392.0 / 128.0 * p)).powf(16384.0 / 2610.0);
            light * 10000.0 / reference_white
        }
        "hlg" => {
            let v = value.min(1.0);
            if v <= 0.5 { v * v / 3.0 } else { (((v - 0.55991073) / 0.17883277).exp() + 0.28466892) / 12.0 }
        }
        "srgb" if value <= 0.04045 => value / 12.92,
        "srgb" => ((value + 0.055) / 1.055).powf(2.4),
        _ => value,
    }
}

fn filmic(value: f64) -> f64 {
    (value * (0.15 * value + 0.05) + 0.004) / (value * (0.15 * value + 0.5) + 0.06) - 1.0 / 15.0
}

fn map(value: f64, operator: &str) -> f64 {
    let value = value.max(0.0);
    if !value.is_finite() {
        return 1.0;
    }
    // Bound intermediate products even for extreme finite float samples.
    let value = value.min(1e20);
    match operator {
        "reinhard" => value / (1.0 + value),
        "aces" => ((value * (2.51 * value + 0.03)) / (value * (2.43 * value + 0.59) + 0.14)).clamp(0.0, 1.0),
        _ => (filmic(value) / filmic(11.2)).clamp(0.0, 1.0),
    }
}

fn srgb(value: f64) -> f64 {
    if value <= 0.0031308 { value * 12.92 } else { 1.055 * value.powf(1.0 / 2.4) - 0.055 }
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn hdr_tone_map(
    py: Python<'_>, image: &Image, operator: &str, transfer: &str, primaries: &str, exposure: f64, reference_white: f64, peak: f64, bit_depth: u8,
) -> PyResult<Image> {
    if !matches!(operator, "reinhard" | "aces" | "filmic")
        || !matches!(transfer, "linear" | "srgb" | "pq" | "hlg")
        || !matches!(primaries, "srgb" | "bt2020")
    {
        return Err(PyValueError::new_err("unsupported HDR operator, transfer, or primaries"));
    }
    if !exposure.is_finite()
        || !exposure.exp2().is_finite()
        || !reference_white.is_finite()
        || reference_white <= 0.0
        || !peak.is_finite()
        || peak <= 0.0
        || !matches!(bit_depth, 8 | 16)
    {
        return Err(PyValueError::new_err("invalid HDR exposure, luminance, or output depth"));
    }
    let mode = match image.mode {
        PixelMode::L | PixelMode::La | PixelMode::Rgb | PixelMode::Rgba => image.mode,
        PixelMode::F | PixelMode::I16 | PixelMode::I16L | PixelMode::I16B => PixelMode::L,
        _ => return Err(PyValueError::new_err("tone mapping requires L, LA, RGB, RGBA, F, or I;16 pixels")),
    };
    if mode == PixelMode::La && bit_depth == 16 {
        return Err(PyValueError::new_err("16-bit LA output is not supported"));
    }
    let source = image.raw_data()?;
    let channels = mode.channels();
    let input_bytes = image.raw_bytes_per_pixel();
    let output_bytes = channels * usize::from(bit_depth / 8);
    let mut output = alloc_pixels(source.len() / input_bytes * output_bytes)?;
    let maximum = if image.mode == PixelMode::F { 1.0 } else { ((1_u64 << image.bit_depth) - 1) as f64 };
    py.detach(|| {
        chunks_mut(&mut output, CHUNK_PIXELS * output_bytes, |chunk, dst| {
            for (i, pixel) in dst.chunks_exact_mut(output_bytes).enumerate() {
                let start = (chunk * CHUNK_PIXELS + i) * input_bytes;
                let data = &source[start..start + input_bytes];
                let mut values = [0.0; 4];
                for (c, value) in values.iter_mut().enumerate().take(channels) {
                    *value = match image.mode {
                        PixelMode::F => f64::from(f32::from_le_bytes(data.try_into().unwrap())),
                        PixelMode::I16B => f64::from(u16::from_be_bytes(data.try_into().unwrap())) / maximum,
                        _ if image.bit_depth > 8 => f64::from(u16::from_le_bytes([data[c * 2], data[c * 2 + 1]])) / maximum,
                        _ => f64::from(data[c]) / maximum,
                    };
                }
                let colors = if matches!(mode, PixelMode::La | PixelMode::Rgba) { channels - 1 } else { channels };
                for value in &mut values[..colors] {
                    *value = linearize(*value, transfer, reference_white).min(1e30);
                }
                if transfer == "hlg" {
                    // BT.2100 display OOTF, with peak-dependent system gamma.
                    let luminance = if colors == 3 && primaries == "bt2020" {
                        0.2627 * values[0] + 0.6780 * values[1] + 0.0593 * values[2]
                    } else if colors == 3 {
                        0.2126 * values[0] + 0.7152 * values[1] + 0.0722 * values[2]
                    } else {
                        values[0]
                    };
                    let gamma = 1.2 + 0.42 * (peak / 1000.0).log10();
                    let factor = luminance.powf(gamma - 1.0);
                    for value in &mut values[..colors] {
                        *value = (*value * factor * peak / reference_white).min(1e30);
                    }
                }
                if primaries == "bt2020" && colors == 3 {
                    let [r, g, b, _] = values;
                    values[..3].copy_from_slice(&[
                        1.660491 * r - 0.587641 * g - 0.072850 * b,
                        -0.124550 * r + 1.132900 * g - 0.008350 * b,
                        -0.018151 * r - 0.100579 * g + 1.118730 * b,
                    ]);
                }
                for (c, value) in values.iter().enumerate().take(channels) {
                    let value = if c < colors { srgb(map(value * exposure.exp2(), operator)) } else { *value };
                    let sample = (value.clamp(0.0, 1.0) * if bit_depth == 8 { 255.0 } else { 65535.0 }).round() as u16;
                    if bit_depth == 8 {
                        pixel[c] = sample as u8;
                    } else {
                        pixel[c * 2..c * 2 + 2].copy_from_slice(&sample.to_le_bytes());
                    }
                }
            }
        });
    });
    if bit_depth == 8 {
        Image::from_pixels(image.width, image.height, mode, output, None)
    } else {
        Image::from_sample_bytes(image.width, image.height, mode, output, bit_depth, None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transfer_reference_values() {
        assert!((linearize(1.0, "pq", 100.0) - 100.0).abs() < 1e-9);
        assert!(linearize(0.0, "pq", 100.0).abs() < 1e-9);
        assert!((linearize(0.5, "hlg", 100.0) - 1.0 / 12.0).abs() < 1e-9);
        assert!((linearize(0.04045, "srgb", 100.0) - 0.00313080495).abs() < 1e-10);
    }

    #[test]
    fn curves_are_bounded_and_monotonic() {
        for operator in ["reinhard", "aces", "filmic"] {
            assert!(map(0.0, operator).abs() < 1e-15);
            let mut last = 0.0;
            for value in [0.01, 0.1, 0.5, 1.0, 10.0, 1e100, f64::INFINITY] {
                let next = map(value, operator);
                assert!((last..=1.0).contains(&next));
                last = next;
            }
        }
        assert_eq!(map(1.0, "reinhard"), 0.5);
        assert!((map(11.2, "filmic") - 1.0).abs() < 1e-15);
    }
}
