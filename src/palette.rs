//! Small palette kernels. Python owns mutable palette storage and file parsing.

use pyo3::exceptions::{PyIndexError, PyOverflowError, PyValueError, PyZeroDivisionError};
use pyo3::prelude::*;
use pyo3::types::{PyByteArray, PyBytes, PyDict, PySlice, PyTuple};
use std::collections::HashSet;

#[path = "palette_simd.rs"]
mod kernels;

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(palette_colors, module)?)?;
    module.add_function(wrap_pyfunction!(palette_ramp, module)?)?;
    module.add_function(wrap_pyfunction!(palette_linear, module)?)?;
    module.add_function(wrap_pyfunction!(palette_gamma, module)?)?;
    module.add_function(wrap_pyfunction!(palette_sepia, module)?)?;
    module.add_function(wrap_pyfunction!(palette_gradient, module)?)?;
    Ok(())
}

#[pyfunction]
fn palette_colors<'py>(py: Python<'py>, palette: &Bound<'py, PyAny>, channels: usize) -> PyResult<Bound<'py, PyDict>> {
    if channels == 0 {
        return Err(PyValueError::new_err("range() arg 3 must not be zero"));
    }
    let colors = PyDict::new(py);
    if let Ok(bytes) = palette.cast::<PyBytes>() {
        index_bytes(py, &colors, bytes.as_bytes(), channels)?;
        return Ok(colors);
    }
    if let Ok(bytes) = palette.cast::<PyByteArray>() {
        index_bytes(py, &colors, &bytes.to_vec(), channels)?;
        return Ok(colors);
    }
    for offset in (0..palette.len()?).step_by(channels) {
        let entry = palette.get_item(PySlice::new(py, offset as isize, (offset + channels) as isize, 1))?;
        let values = entry.try_iter()?.collect::<PyResult<Vec<_>>>()?;
        let key = PyTuple::new(py, values)?;
        if !colors.contains(&key)? {
            colors.set_item(key, offset / channels)?;
        }
    }
    Ok(colors)
}

fn index_bytes(py: Python<'_>, colors: &Bound<'_, PyDict>, data: &[u8], channels: usize) -> PyResult<()> {
    let mut seen = HashSet::new();
    for (index, entry) in data.chunks(channels).enumerate() {
        if seen.insert(entry) {
            colors.set_item(PyTuple::new(py, entry.iter().copied())?, index)?;
        }
    }
    Ok(())
}

#[pyfunction]
fn palette_ramp(channels: usize, reverse: bool) -> Vec<u16> {
    (0..256)
        .flat_map(|value| std::iter::repeat_n(if reverse { 255 - value } else { value }, channels))
        .collect()
}

#[pyfunction]
fn palette_linear(white: i64) -> Vec<i128> {
    if let Ok(white) = u8::try_from(white) {
        return kernels::linear_lut(white).into_iter().map(i128::from).collect();
    }
    (0..256).map(|value| (i128::from(white) * value).div_euclid(255)).collect()
}

#[pyfunction]
fn palette_gamma(exp: f64) -> Vec<u16> {
    (0..256)
        .map(|value| ((f64::from(value) / 255.0).powf(exp) * 255.0 + 0.5) as u16)
        .collect()
}

#[pyfunction]
fn palette_sepia(white: [i64; 3]) -> Vec<i128> {
    if white.iter().all(|&channel| (0..=255).contains(&channel)) {
        let luts = white.map(|channel| kernels::linear_lut(channel as u8));
        return (0..256).flat_map(|i| luts.each_ref().map(|lut| i128::from(lut[i]))).collect();
    }
    (0..256)
        .flat_map(|value| white.map(|channel| (i128::from(channel) * value).div_euclid(255)))
        .collect()
}

fn linear(middle: f64, position: f64) -> f64 {
    if position <= middle {
        if middle < 1e-10 { 0.0 } else { 0.5 * position / middle }
    } else if 1.0 - middle < 1e-10 {
        1.0
    } else {
        0.5 + 0.5 * (position - middle) / (1.0 - middle)
    }
}

fn blend(kind: usize, middle: f64, position: f64) -> PyResult<f64> {
    let value = linear(middle, position);
    let result = match kind {
        1 => {
            let denominator = middle.max(1e-10).ln();
            if denominator == 0.0 {
                return Err(PyZeroDivisionError::new_err("float division by zero"));
            }
            let exponent = 0.5_f64.ln() / denominator;
            if position == 0.0 && exponent < 0.0 {
                return Err(PyZeroDivisionError::new_err("0.0 cannot be raised to a negative power"));
            }
            position.powf(exponent)
        }
        2 => ((-std::f64::consts::PI / 2.0 + std::f64::consts::PI * value).sin() + 1.0) / 2.0,
        3 => (1.0 - (value - 1.0).powi(2)).sqrt(),
        4 => 1.0 - (1.0 - value.powi(2)).sqrt(),
        _ => value,
    };
    if result.is_nan() && matches!(kind, 3 | 4) {
        return Err(PyValueError::new_err("math domain error"));
    }
    Ok(result)
}

fn render_gradient(segments: &[(Vec<f64>, usize)]) -> PyResult<Vec<u8>> {
    let mut result = Vec::with_capacity(1024);
    let mut segment = 0;
    for index in 0..256 {
        let position = f64::from(index) / 255.0;
        let (values, kind) = loop {
            let item = segments.get(segment).ok_or_else(|| PyIndexError::new_err("list index out of range"))?;
            if item.0.len() < 11 {
                return Err(PyIndexError::new_err("list index out of range"));
            }
            if item.0[2] >= position || item.0[2].is_nan() {
                break item;
            }
            segment += 1;
        };
        let width = values[2] - values[0];
        let scale = if width < 1e-10 {
            blend(*kind, 0.5, 0.5)?
        } else {
            blend(*kind, (values[1] - values[0]) / width, (position - values[0]) / width)?
        };
        for channel in 3..7 {
            let value = (255.0 * ((values[channel + 4] - values[channel]) * scale + values[channel]) + 0.5).trunc();
            if value.is_infinite() {
                return Err(PyOverflowError::new_err("cannot convert float infinity to integer"));
            }
            if value.is_nan() {
                return Err(PyValueError::new_err("cannot convert float NaN to integer"));
            }
            result.push(value.rem_euclid(256.0) as u8);
        }
    }
    Ok(result)
}

#[pyfunction]
fn palette_gradient<'py>(py: Python<'py>, segments: Vec<(Vec<f64>, usize)>) -> PyResult<Bound<'py, PyBytes>> {
    Ok(PyBytes::new(py, &render_gradient(&segments)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ramps_and_luts() {
        assert_eq!(&palette_ramp(4, true)[..8], &[255, 255, 255, 255, 254, 254, 254, 254]);
        assert_eq!(palette_linear(255), (0..256).collect::<Vec<_>>());
        assert_eq!(palette_linear(-1)[1], -1);
        assert_eq!(palette_gamma(1.0), (0..256).collect::<Vec<_>>());
        assert_eq!(&palette_sepia([255, 240, 192])[765..], &[255, 240, 192]);
    }

    #[test]
    fn gradient_endpoints_and_alpha() {
        let values = vec![0.0, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        for kind in 0..5 {
            let result = render_gradient(&[(values.clone(), kind)]).unwrap();
            assert_eq!(result.len(), 1024);
            assert_eq!(&result[..4], &[0; 4]);
            assert_eq!(&result[1020..], &[255; 4]);
        }
    }
}
