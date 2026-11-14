//! Geometry calculations for the Python ImageOps adapters.

use pyo3::exceptions::{PyOverflowError, PyValueError, PyZeroDivisionError};
use pyo3::prelude::*;

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(ops_contain_size, module)?)?;
    module.add_function(wrap_pyfunction!(ops_cover_size, module)?)?;
    module.add_function(wrap_pyfunction!(ops_scale_size, module)?)?;
    module.add_function(wrap_pyfunction!(ops_pad_offset, module)?)?;
    module.add_function(wrap_pyfunction!(ops_fit_box, module)?)?;
    Ok(())
}

fn rounded(value: f64) -> PyResult<i64> {
    if value.is_nan() {
        return Err(PyValueError::new_err("cannot convert float NaN to integer"));
    }
    if !value.is_finite() {
        return Err(PyOverflowError::new_err("cannot convert float infinity to integer"));
    }
    let value = value.round_ties_even();
    if !(-9_223_372_036_854_775_808.0..9_223_372_036_854_775_808.0).contains(&value) {
        return Err(PyOverflowError::new_err("image dimension or offset is too large"));
    }
    Ok(value as i64)
}

#[pyfunction]
fn ops_contain_size(source: (u32, u32), size: (i64, i64)) -> PyResult<(i64, i64)> {
    aspect_size(source, size, Aspect::Contain)
}

#[pyfunction]
fn ops_cover_size(source: (u32, u32), size: (i64, i64)) -> PyResult<(i64, i64)> {
    aspect_size(source, size, Aspect::Cover)
}

enum Aspect {
    Contain,
    Cover,
}

fn aspect_size(source: (u32, u32), size: (i64, i64), aspect: Aspect) -> PyResult<(i64, i64)> {
    let (width, height) = (f64::from(source.0), f64::from(source.1));
    if source.1 == 0 {
        return Ok((size.0, 0));
    }
    if size.1 == 0 {
        return Ok((0, 0));
    }
    let ratio = width / height;
    let target = size.0 as f64 / size.1 as f64;
    let (adjust_height, adjust_width) = match aspect {
        Aspect::Contain => (ratio > target, ratio < target),
        Aspect::Cover => (ratio < target, ratio > target),
    };
    if adjust_height {
        if source.0 == 0 {
            return Err(PyZeroDivisionError::new_err("division by zero"));
        }
        Ok((size.0, rounded(height / width * size.0 as f64)?))
    } else if adjust_width {
        Ok((rounded(width / height * size.1 as f64)?, size.1))
    } else {
        Ok(size)
    }
}

#[pyfunction]
fn ops_scale_size(source: (u32, u32), factor: f64) -> PyResult<(i64, i64)> {
    if factor <= 0.0 {
        return Err(PyValueError::new_err("the factor must be greater than 0"));
    }
    Ok((rounded(f64::from(source.0) * factor)?, rounded(f64::from(source.1) * factor)?))
}

#[pyfunction]
fn ops_pad_offset(source: (u32, u32), size: (i64, i64), centering: (f64, f64)) -> PyResult<(i64, i64)> {
    // Python's min/max expression maps NaN to zero here.
    let clamp = |value: f64| if value.is_nan() { 0.0 } else { value.clamp(0.0, 1.0) };
    Ok((
        rounded((i128::from(size.0) - i128::from(source.0)) as f64 * clamp(centering.0))?,
        rounded((i128::from(size.1) - i128::from(source.1)) as f64 * clamp(centering.1))?,
    ))
}

#[pyfunction]
fn ops_fit_box(source: (u32, u32), size: (i64, i64), bleed: f64, centering: (f64, f64)) -> PyResult<Option<(f64, f64, f64, f64)>> {
    if size.0.min(size.1) == 0 || source.1 == 0 {
        return Ok(None);
    }
    if size.1 == 0 {
        return Err(PyZeroDivisionError::new_err("division by zero"));
    }
    let center = |value: f64| if (0.0..=1.0).contains(&value) { value } else { 0.5 };
    let bleed = if (0.0..0.5).contains(&bleed) { bleed } else { 0.0 };
    let bx = f64::from(source.0) * bleed;
    let by = f64::from(source.1) * bleed;
    let width = f64::from(source.0) - 2.0 * bx;
    let height = f64::from(source.1) - 2.0 * by;
    let ratio = size.0 as f64 / size.1 as f64;
    let (cw, ch) = if width / height == ratio {
        (width, height)
    } else if width / height > ratio {
        (ratio * height, height)
    } else {
        (width, width / ratio)
    };
    let left = bx + (width - cw) * center(centering.0);
    let top = by + (height - ch) * center(centering.1);
    Ok(Some((left, top, left + cw, top + ch)))
}
