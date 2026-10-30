use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyList;

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(path_compact, module)?)?;
    module.add_function(wrap_pyfunction!(path_bbox, module)?)?;
    module.add_function(wrap_pyfunction!(path_transform, module)?)?;
    Ok(())
}

#[pyfunction]
fn path_compact(points: &Bound<'_, PyList>, distance: f64) -> PyResult<usize> {
    let length = points.len();
    if length == 0 {
        return Ok(0);
    }
    let mut previous = points.get_item(0)?.extract::<(f64, f64)>()?;
    let mut kept = 1;
    for index in 1..length {
        let point = points.get_item(index)?;
        let (x, y) = point.extract::<(f64, f64)>()?;
        if (x - previous.0).abs() + (y - previous.1).abs() >= distance {
            if index != kept {
                points.set_item(kept, point)?;
            }
            previous = (x, y);
            kept += 1;
        }
    }
    points.del_slice(kept, length)?;
    Ok(length - kept)
}

#[pyfunction]
fn path_bbox(points: &Bound<'_, PyList>) -> PyResult<(f64, f64, f64, f64)> {
    if points.is_empty() {
        return Ok((0.0, 0.0, 0.0, 0.0));
    }
    let (mut left, mut top) = points.get_item(0)?.extract::<(f64, f64)>()?;
    let (mut right, mut bottom) = (left, top);
    for point in points.iter().skip(1) {
        let (x, y) = point.extract::<(f64, f64)>()?;
        if x < left {
            left = x;
        }
        if x > right {
            right = x;
        }
        if y < top {
            top = y;
        }
        if y > bottom {
            bottom = y;
        }
    }
    Ok((left, top, right, bottom))
}

#[pyfunction]
fn path_transform(points: &Bound<'_, PyList>, matrix: [f64; 6], wrap: f64) -> PyResult<()> {
    let [a, b, c, d, e, f] = matrix;
    for index in 0..points.len() {
        let (x, y) = points.get_item(index)?.extract::<(f64, f64)>()?;
        let tx = a * x + b * y + c;
        let ty = d * x + e * y + f;
        let tx = if wrap != 0.0 {
            if tx.is_infinite() && !wrap.is_nan() {
                return Err(PyValueError::new_err("math domain error"));
            }
            tx % wrap
        } else {
            tx
        };
        points.set_item(index, (tx, ty))?;
    }
    Ok(())
}
