pub use blanket_kernels::color::*;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

#[pyfunction]
pub fn color_getcolor(py: Python<'_>, rgb: Vec<i64>, mode: &str) -> PyResult<Py<PyAny>> {
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
