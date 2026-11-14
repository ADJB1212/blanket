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
        self.image()?.borrow(py).getpixel(py, xy)
    }

    fn __setitem__(&self, py: Python<'_>, xy: &Bound<'_, PyAny>, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let xy = coordinates(xy)?;
        let mut image = self.image()?.borrow_mut(py);
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
