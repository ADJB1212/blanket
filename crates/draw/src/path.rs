use pyo3::exceptions::{PyIndexError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyByteArray, PyBytes, PyFloat, PyList, PyString, PyTuple};

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<Path>()
}

fn coordinate(value: &Bound<'_, PyAny>) -> PyResult<f64> {
    if let Ok(value) = value.cast::<PyFloat>() {
        return Ok(value.value());
    }
    if value.is_instance_of::<PyString>() || value.is_instance_of::<PyBytes>() || value.is_instance_of::<PyByteArray>() {
        return Err(PyValueError::new_err("coordinate list must contain numbers"));
    }
    value.extract()
}

#[pyclass(name = "_Path")]
#[derive(Debug, Default)]
struct Path {
    points: Vec<(f64, f64)>,
    objects: Option<Py<PyList>>,
}

impl Path {
    fn position(&self, index: isize) -> PyResult<usize> {
        let index = if index < 0 { self.points.len().cast_signed() + index } else { index };
        if index < 0 || index as usize >= self.points.len() {
            return Err(PyIndexError::new_err("path index out of range"));
        }
        Ok(index as usize)
    }

    fn set(&mut self, py: Python<'_>, index: usize, point: (f64, f64)) -> PyResult<()> {
        self.points[index] = point;
        if let Some(objects) = &self.objects {
            objects.bind(py).set_item(index, point)?;
        }
        Ok(())
    }
}

#[pymethods]
impl Path {
    #[new]
    fn new(xy: &Bound<'_, PyAny>) -> PyResult<Self> {
        let mut points = Vec::with_capacity(xy.len().unwrap_or(0));
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
                        points.push((x, number));
                    } else {
                        pending = Some(number);
                    }
                }
            } else {
                let number = coordinate(&value)?;
                if let Some(x) = pending.take() {
                    points.push((x, number));
                } else {
                    pending = Some(number);
                }
            }
        }
        if pending.is_some() {
            return Err(PyValueError::new_err("wrong number of coordinates"));
        }
        Ok(Self { points, objects: None })
    }

    #[staticmethod]
    fn zeros(count: usize) -> PyResult<Self> {
        let mut points = Vec::new();
        points.try_reserve_exact(count).map_err(|_| pyo3::exceptions::PyMemoryError::new_err("cannot allocate path"))?;
        points.resize(count, (0.0, 0.0));
        Ok(Self { points, objects: None })
    }

    #[staticmethod]
    fn from_buffer(data: &[u8]) -> Self {
        let points = data
            .as_chunks::<8>()
            .0
            .iter()
            .map(|pair| (f64::from(f32::from_ne_bytes(pair[..4].try_into().unwrap())), f64::from(f32::from_ne_bytes(pair[4..].try_into().unwrap()))))
            .collect();
        Self { points, objects: None }
    }

    fn copy(&self) -> Self {
        Self { points: self.points.clone(), objects: None }
    }

    fn slice(&self, py: Python<'_>, start: usize, stop: usize) -> Self {
        Self {
            points: self.points[start..stop].to_vec(),
            objects: self.objects.as_ref().map(|objects| objects.bind(py).get_slice(start, stop).unbind()),
        }
    }

    fn __len__(&self) -> usize {
        self.points.len()
    }

    fn __getitem__<'py>(&self, py: Python<'py>, index: isize) -> PyResult<Bound<'py, PyAny>> {
        let index = self.position(index)?;
        if let Some(objects) = &self.objects {
            return objects.bind(py).get_item(index);
        }
        Ok(self.points[index].into_pyobject(py)?.into_any())
    }

    fn __setitem__(&mut self, py: Python<'_>, index: isize, point: (f64, f64)) -> PyResult<()> {
        let index = self.position(index)?;
        self.set(py, index, point)
    }

    fn compact(&mut self, py: Python<'_>, distance: f64) -> PyResult<usize> {
        let length = self.points.len();
        let Some(mut previous) = self.points.first().copied() else {
            return Ok(0);
        };
        let mut kept = 1;
        for index in 1..length {
            let point = self.points[index];
            if (point.0 - previous.0).abs() + (point.1 - previous.1).abs() >= distance {
                self.points[kept] = point;
                if index != kept
                    && let Some(objects) = &self.objects
                {
                    let objects = objects.bind(py);
                    objects.set_item(kept, objects.get_item(index)?)?;
                }
                previous = point;
                kept += 1;
            }
        }
        self.points.truncate(kept);
        if let Some(objects) = &self.objects {
            objects.bind(py).del_slice(kept, length)?;
        }
        Ok(length - kept)
    }

    fn getbbox(&self) -> (f64, f64, f64, f64) {
        let Some(&(mut left, mut top)) = self.points.first() else {
            return (0.0, 0.0, 0.0, 0.0);
        };
        let (mut right, mut bottom) = (left, top);
        for &(x, y) in self.points.iter().skip(1) {
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
        (left, top, right, bottom)
    }

    fn map(slf: Py<Self>, py: Python<'_>, function: &Bound<'_, PyAny>) -> PyResult<()> {
        let length = slf.borrow(py).points.len();
        for index in 0..length {
            let point = slf.borrow(py).points[index];
            let result = function.call1(point)?;
            let pair = result.cast::<PyTuple>().map_err(|_| PyTypeError::new_err("coordinate must be a tuple"))?;
            if pair.len() != 2 {
                return Err(PyValueError::new_err("coordinate list must contain exactly 2 coordinates"));
            }
            let point = (coordinate(&pair.get_item(0)?)?, coordinate(&pair.get_item(1)?)?);
            slf.borrow_mut(py).set(py, index, point)?;
        }
        Ok(())
    }

    fn tolist<'py>(&mut self, py: Python<'py>, flat: bool) -> PyResult<Bound<'py, PyList>> {
        if flat {
            return PyList::new(py, self.points.iter().flat_map(|&point| <[f64; 2]>::from(point)));
        }
        if self.objects.is_none() {
            self.objects = Some(PyList::new(py, &self.points)?.unbind());
        }
        Ok(self.objects.as_ref().unwrap().bind(py).get_slice(0, self.points.len()))
    }

    fn transform(&mut self, matrix: [f64; 6], wrap: f64) -> PyResult<()> {
        let [a, b, c, d, e, f] = matrix;
        self.objects = None;
        for point in &mut self.points {
            let (x, y) = *point;
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
            *point = (tx, ty);
        }
        Ok(())
    }
}
