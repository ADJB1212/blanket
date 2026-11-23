use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyFloat, PyTuple};

use super::Image;

/// Native pixel accessor; Python subclasses supply documentation and mapped variants.
#[pyclass(name = "_PixelAccess", module = "blanket._blanket", subclass, frozen)]
#[derive(Debug)]
pub struct PixelAccess {
    image: Option<Py<Image>>,
}

fn coordinates(xy: &Bound<'_, PyAny>) -> PyResult<(i64, i64)> {
    let xy = xy.cast::<PyTuple>().ok().filter(|xy| xy.len() == 2).ok_or_else(|| PyTypeError::new_err("argument must be sequence of length 2"))?;
    let value = |v: Bound<'_, PyAny>| if v.is_instance_of::<PyFloat>() { v.call_method0("__int__")?.extract() } else { v.extract() };
    Ok((value(xy.get_item(0)?)?, value(xy.get_item(1)?)?))
}

impl PixelAccess {
    fn image(&self) -> PyResult<&Py<Image>> {
        self.image.as_ref().ok_or_else(|| PyTypeError::new_err("pixel access has no native image"))
    }
}

#[pymethods]
impl PixelAccess {
    #[new]
    #[pyo3(signature = (image=None, *_args))]
    fn new(image: Option<&Bound<'_, PyAny>>, _args: &Bound<'_, PyTuple>) -> Self {
        Self { image: image.and_then(|image| image.cast::<Image>().ok()).map(|image| image.clone().unbind()) }
    }

    fn __getitem__(&self, py: Python<'_>, xy: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let xy = coordinates(xy)?;
        self.image()?.try_borrow(py)?.getpixel(py, xy)
    }

    fn __setitem__(&self, py: Python<'_>, xy: &Bound<'_, PyAny>, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let xy = coordinates(xy)?;
        let mut image = self.image()?.try_borrow_mut(py)?;
        if let Ok(single) = value.cast::<PyTuple>()
            && single.len() == 1
        {
            let first = single.get_item(0)?;
            match image.mode() {
                "1" | "L" | "I" | "I;16" | "I;16L" | "I;16B" | "I;16N" => return image.putpixel(xy, &first),
                "LA" | "La" | "PA" => return image.putpixel(xy, (first, 0).into_pyobject(py)?.as_any()),
                _ => {}
            }
        }
        image.putpixel(xy, value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PixelMode;

    #[test]
    fn pixel_access_reports_borrow_conflicts() {
        Python::initialize();
        Python::attach(|py| {
            let image = Py::new(py, Image::from_pixels(1, 1, PixelMode::L, vec![0], None).unwrap()).unwrap();
            let access = PixelAccess { image: Some(image.clone_ref(py)) };
            let xy = (0_i64, 0_i64).into_pyobject(py).unwrap();
            let value = 1_i64.into_pyobject(py).unwrap();

            let writer = image.borrow_mut(py);
            assert!(access.__getitem__(py, xy.as_any()).unwrap_err().is_instance_of::<pyo3::exceptions::PyRuntimeError>(py));
            drop(writer);

            let _reader = image.borrow(py);
            assert!(access.__setitem__(py, xy.as_any(), value.as_any()).unwrap_err().is_instance_of::<pyo3::exceptions::PyRuntimeError>(py));
        });
    }
}
