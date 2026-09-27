use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

#[pyfunction]
pub fn color_getcolor(py: Python<'_>, rgb: Vec<i64>, mode: &str) -> PyResult<Py<PyAny>> {
    if !matches!(rgb.len(), 3 | 4) {
        return Err(PyValueError::new_err("color must contain 3 or 4 components"));
    }
    let alpha = rgb.get(3).copied().unwrap_or(255);
    let components = match mode {
        "1" | "L" | "I" | "F" | "LA" => {
            let luminance = (i128::from(rgb[0]) * 19595 + i128::from(rgb[1]) * 38470 + i128::from(rgb[2]) * 7471 + 32768) >> 16;
            if mode == "LA" {
                return Ok((luminance, alpha).into_pyobject(py)?.into_any().unbind());
            }
            let value = if mode == "1" {
                if luminance >= 128 { 255 } else { 0 }
            } else {
                luminance
            };
            return Ok(value.into_pyobject(py)?.into_any().unbind());
        }
        "RGB" => rgb[..3].to_vec(),
        "RGBA" => vec![rgb[0], rgb[1], rgb[2], alpha],
        "CMYK" => {
            let channels = [rgb[0] as f64 / 255.0, rgb[1] as f64 / 255.0, rgb[2] as f64 / 255.0];
            let black = 1.0 - channels.into_iter().fold(f64::NEG_INFINITY, f64::max);
            if black == 1.0 {
                vec![0, 0, 0, 255]
            } else {
                let mut result: Vec<i64> = channels
                    .map(|component| ((1.0 - component - black) / (1.0 - black) * 255.0).round_ties_even() as i64)
                    .into();
                result.push((black * 255.0).round_ties_even() as i64);
                result
            }
        }
        _ => {
            return Err(PyValueError::new_err(format!(
                "unsupported image mode: {}",
                mode.into_pyobject(py)?.repr()?
            )));
        }
    };
    Ok(PyTuple::new(py, components)?.into_any().unbind())
}
