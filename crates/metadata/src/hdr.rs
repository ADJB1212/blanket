//! PNG HDR metadata in physical chromaticity and luminance units.

use pyo3::exceptions::{PyOverflowError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyTuple};

fn encode(key: &str, values: &[f64]) -> PyResult<Vec<u8>> {
    let count = match key {
        "cicp" => 4,
        "mastering_display" => 10,
        _ => 2,
    };
    if values.len() != count || values.iter().any(|v| !v.is_finite() || *v < 0.0) {
        return Err(PyValueError::new_err(format!("{key} requires {count} finite nonnegative numbers")));
    }
    if key == "cicp" {
        if values.iter().any(|v| *v > 255.0 || v.fract() != 0.0) {
            return Err(PyValueError::new_err("cicp must contain four byte-sized integers"));
        }
        if values[2] != 0.0 || values[3] != 1.0 {
            return Err(PyValueError::new_err("PNG cICP requires identity matrix coefficients and full range"));
        }
        return Ok(values.iter().map(|v| *v as u8).collect());
    }
    if key == "mastering_display" && (values[..8].iter().any(|v| *v > 1.0) || values[9] > values[8]) {
        return Err(PyValueError::new_err("invalid mastering display chromaticities or luminance range"));
    }
    let mut result = Vec::with_capacity(if count == 10 { 24 } else { 8 });
    for (index, value) in values.iter().enumerate() {
        let chromaticity = count == 10 && index < 8;
        let packed = (value * if chromaticity { 50000.0 } else { 10000.0 }).round_ties_even();
        if !packed.is_finite() {
            return Err(PyOverflowError::new_err("cannot convert float infinity to integer"));
        }
        let maximum = if chromaticity { f64::from(u16::MAX) } else { f64::from(u32::MAX) };
        if packed > maximum {
            return Err(PyValueError::new_err(format!("{key} exceeds the PNG metadata range")));
        }
        if chromaticity {
            result.extend_from_slice(&(packed as u16).to_be_bytes());
        } else {
            result.extend_from_slice(&(packed as u32).to_be_bytes());
        }
    }
    Ok(result)
}

#[pyfunction]
pub fn _encode_hdr<'py>(py: Python<'py>, key: &str, values: Vec<f64>) -> PyResult<Bound<'py, PyBytes>> {
    Ok(PyBytes::new(py, &encode(key, &values)?))
}

#[pyfunction]
pub fn _decode_hdr<'py>(py: Python<'py>, kind: &[u8], payload: &[u8]) -> PyResult<Option<Bound<'py, PyTuple>>> {
    let (key, count) = match kind {
        b"cICP" if payload.len() == 4 => {
            if payload[2] != 0 || payload[3] != 1 {
                return Ok(None);
            }
            return Ok(Some(PyTuple::new(py, payload.iter().copied())?));
        }
        b"mDCV" if payload.len() == 24 => ("mastering_display", 10),
        b"cLLI" if payload.len() == 8 => ("content_light_level", 2),
        _ => return Ok(None),
    };
    let mut values = Vec::with_capacity(count);
    let mut offset = 0;
    for index in 0..count {
        let chromaticity = count == 10 && index < 8;
        let value = if chromaticity {
            let value = u16::from_be_bytes(payload[offset..offset + 2].try_into().expect("two bytes"));
            offset += 2;
            f64::from(value) / 50000.0
        } else {
            let value = u32::from_be_bytes(payload[offset..offset + 4].try_into().expect("four bytes"));
            offset += 4;
            f64::from(value) / 10000.0
        };
        values.push(value);
    }
    if encode(key, &values).is_err() {
        return Ok(None);
    }
    Ok(Some(PyTuple::new(py, values)?))
}
