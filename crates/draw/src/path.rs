use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyByteArray, PyBytes, PyList, PyString, PyTuple};

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(path_compact, module)?)?;
    module.add_function(wrap_pyfunction!(path_bbox, module)?)?;
    module.add_function(wrap_pyfunction!(path_transform, module)?)?;
    module.add_function(wrap_pyfunction!(path_coordinates, module)?)?;
    module.add_function(wrap_pyfunction!(path_map, module)?)?;
    Ok(())
}

fn coordinate(value: &Bound<'_, PyAny>) -> PyResult<f64> {
    if value.is_instance_of::<PyString>() || value.is_instance_of::<PyBytes>() || value.is_instance_of::<PyByteArray>() {
        return Err(PyValueError::new_err("coordinate list must contain numbers"));
    }
    value.extract()
}

#[pyfunction]
fn path_coordinates<'py>(xy: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyList>> {
    let points = PyList::empty(xy.py());
    let mut pending = None;
    for value in xy.try_iter()? {
        let value = value?;
        if value.is_instance_of::<PyTuple>() || value.is_instance_of::<PyList>() {
            if value.len()? != 2 {
                return Err(PyValueError::new_err("coordinate list must contain exactly 2 coordinates"));
            }
            for item in [value.get_item(0)?, value.get_item(1)?] {
                let number = coordinate(&item)?;
                if let Some(x) = pending.take() {
                    points.append((x, number))?;
                } else {
                    pending = Some(number);
                }
            }
        } else {
            let number = coordinate(&value)?;
            if let Some(x) = pending.take() {
                points.append((x, number))?;
            } else {
                pending = Some(number);
            }
        }
    }
    if pending.is_some() {
        return Err(PyValueError::new_err("wrong number of coordinates"));
    }
    Ok(points)
}

#[pyfunction]
fn path_map(points: &Bound<'_, PyList>, function: &Bound<'_, PyAny>) -> PyResult<()> {
    for index in 0..points.len() {
        let point = points.get_item(index)?;
        let result = function.call1(point.cast::<PyTuple>()?)?;
        let pair = result.cast::<PyTuple>().map_err(|_| PyTypeError::new_err("coordinate must be a tuple"))?;
        if pair.len() != 2 {
            return Err(PyValueError::new_err("coordinate list must contain exactly 2 coordinates"));
        }
        points.set_item(index, (coordinate(&pair.get_item(0)?)?, coordinate(&pair.get_item(1)?)?))?;
    }
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
