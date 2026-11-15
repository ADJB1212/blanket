use blanket_kernels::color::*;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyInt, PyTuple};

fn clipped(value: &Bound<'_, PyAny>, maximum: u32) -> PyResult<u32> {
    if let Ok(value) = value.extract::<i128>() {
        Ok(value.clamp(0, i128::from(maximum)) as u32)
    } else if value.lt(0)? {
        Ok(0)
    } else {
        Ok(maximum)
    }
}

#[pyfunction]
#[pyo3(signature = (color, mode, bit_depth=8, rgb=None))]
pub(crate) fn color_pixel(py: Python<'_>, color: &Bound<'_, PyAny>, mode: &str, bit_depth: u8, rgb: Option<Vec<i64>>) -> PyResult<Vec<u32>> {
    if !(1..=16).contains(&bit_depth) {
        return Err(PyValueError::new_err("invalid color bit depth"));
    }
    let channels = if mode == "YCbCr" { 3 } else { mode.len() };
    let maximum = (1u32 << bit_depth) - 1;
    if let Some(rgb) = rgb {
        if matches!(mode, "La" | "RGBa") {
            return color_getcolor(py, rgb, mode)?.extract(py);
        }
        if !matches!(rgb.len(), 3 | 4) {
            return Err(PyValueError::new_err("color must contain 3 or 4 components"));
        }
        let alpha = rgb.get(3).copied().unwrap_or(255);
        let mut values = if matches!(mode, "1" | "L" | "LA") {
            let gray =
                (rgb[0] * i64::from(LUMA_RED) + rgb[1] * i64::from(LUMA_GREEN) + rgb[2] * i64::from(LUMA_BLUE) + i64::from(LUMA_ROUND)) >> LUMA_BITS;
            if mode == "LA" { vec![gray, alpha] } else { vec![gray] }
        } else {
            let mut values = rgb[..3].to_vec();
            if matches!(mode, "RGBA" | "RGBX") {
                values.push(if mode == "RGBA" { alpha } else { 255 });
            }
            values
        };
        if bit_depth != 8 {
            for value in &mut values {
                *value = (*value * i64::from(maximum) + 127) / 255;
            }
        }
        if values.len() == 3 && mode == "CMYK" {
            values.push(i64::from(maximum));
        }
        return Ok(values.into_iter().map(|value| value.clamp(0, i64::from(maximum)) as u32).collect());
    }
    if color.is_none() {
        return Ok(vec![0; channels]);
    }
    if color.is_instance_of::<PyInt>() {
        if matches!(mode, "1" | "L") {
            return Ok(vec![clipped(color, maximum)?]);
        }
        if let Ok(value) = color.extract::<i128>() {
            return Ok((0..channels)
                .map(|i| {
                    let shift = u32::try_from(usize::from(bit_depth) * i).unwrap_or(u32::MAX);
                    let shifted = value.checked_shr(shift).unwrap_or(if value < 0 { -1 } else { 0 });
                    (shifted & i128::from(maximum)) as u32
                })
                .collect());
        }
        return (0..channels)
            .map(|i| color.call_method1("__rshift__", (usize::from(bit_depth) * i,))?.call_method1("__and__", (maximum,))?.extract())
            .collect();
    }
    let tuple = color.cast::<PyTuple>().map_err(|_| PyTypeError::new_err("color must be int or tuple"))?;
    let mut values: Vec<_> = tuple.iter().collect();
    if matches!(mode, "1" | "L") {
        if values.len() != 1 {
            return Err(PyTypeError::new_err("color must be int or single-element tuple"));
        }
    } else if (values.len() == 1 && matches!(mode, "LA" | "La")) || (values.len() == 3 && matches!(mode, "RGBA" | "RGBa" | "RGBX" | "CMYK")) {
        values.push(maximum.into_pyobject(py)?.into_any());
    } else if matches!(mode, "RGB" | "YCbCr") && values.len() == 4 {
        values.truncate(3);
    } else if values.len() != channels {
        return Err(PyTypeError::new_err("color must be int, or tuple of the appropriate channel count"));
    }
    if values.iter().any(|v| !v.is_instance_of::<PyInt>()) {
        return Err(PyTypeError::new_err("color components must be integers"));
    }
    let maximum = if mode == "1" { 255 } else { maximum };
    values.iter().map(|value| clipped(value, maximum)).collect()
}

#[pyfunction]
pub(crate) fn color_getcolor(py: Python<'_>, rgb: Vec<i64>, mode: &str) -> PyResult<Py<PyAny>> {
    if !matches!(rgb.len(), 3 | 4) {
        return Err(PyValueError::new_err("color must contain 3 or 4 components"));
    }
    let alpha = rgb.get(3).copied().unwrap_or(255);
    if matches!(mode, "La" | "RGBa") {
        let premultiply = |value: i128| {
            let product = value * i128::from(alpha) + 128;
            (product + (product >> 8)) >> 8
        };
        let components = if mode == "La" {
            let gray = (i128::from(rgb[0]) * i128::from(LUMA_RED)
                + i128::from(rgb[1]) * i128::from(LUMA_GREEN)
                + i128::from(rgb[2]) * i128::from(LUMA_BLUE)
                + i128::from(LUMA_ROUND))
                >> LUMA_BITS;
            vec![premultiply(gray), i128::from(alpha)]
        } else {
            vec![premultiply(i128::from(rgb[0])), premultiply(i128::from(rgb[1])), premultiply(i128::from(rgb[2])), i128::from(alpha)]
        };
        return Ok(PyTuple::new(py, components)?.into_any().unbind());
    }
    let components = match mode {
        "1" | "L" | "I" | "F" | "LA" | "I;16" | "I;16L" | "I;16B" | "I;16N" => {
            let luminance = (i128::from(rgb[0]) * i128::from(LUMA_RED)
                + i128::from(rgb[1]) * i128::from(LUMA_GREEN)
                + i128::from(rgb[2]) * i128::from(LUMA_BLUE)
                + i128::from(LUMA_ROUND))
                >> LUMA_BITS;
            if mode == "LA" {
                return Ok((luminance, alpha).into_pyobject(py)?.into_any().unbind());
            }
            return Ok(luminance.into_pyobject(py)?.into_any().unbind());
        }
        "RGB" | "RGBX" | "P" | "CMYK" | "YCbCr" | "LAB" => rgb[..3].to_vec(),
        "RGBA" | "PA" => vec![rgb[0], rgb[1], rgb[2], alpha],
        "HSV" => {
            let [r, g, b] = [rgb[0] as f64 / 255.0, rgb[1] as f64 / 255.0, rgb[2] as f64 / 255.0];
            let maximum = r.max(g).max(b);
            let minimum = r.min(g).min(b);
            let difference = maximum - minimum;
            let (hue, saturation) = if difference == 0.0 {
                (0.0, 0.0)
            } else {
                let rc = (maximum - r) / difference;
                let gc = (maximum - g) / difference;
                let bc = (maximum - b) / difference;
                let hue = if r == maximum {
                    bc - gc
                } else if g == maximum {
                    2.0 + rc - bc
                } else {
                    4.0 + gc - rc
                };
                ((hue / 6.0).rem_euclid(1.0), difference / maximum)
            };
            vec![(hue * 255.0) as i64, (saturation * 255.0) as i64, (maximum * 255.0) as i64]
        }
        _ => {
            return Err(PyValueError::new_err(format!("unsupported image mode: {}", mode.into_pyobject(py)?.repr()?)));
        }
    };
    Ok(PyTuple::new(py, components)?.into_any().unbind())
}
