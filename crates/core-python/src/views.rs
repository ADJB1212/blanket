//! Read-only exports that retain copy-on-write raster storage.

use blanket_core::{Image, PixelMode};
use pyo3::prelude::*;
use pyo3::types::PyMemoryView;

#[pyclass(frozen, module = "blanket._blanket")]
#[derive(Debug)]
struct PixelBuffer {
    image: Image,
}

#[pymethods]
impl PixelBuffer {
    /// # Safety
    /// CPython supplies a writable Py_buffer and valid buffer request flags.
    unsafe fn __getbuffer__(slf: Bound<'_, Self>, view: *mut pyo3::ffi::Py_buffer, flags: std::ffi::c_int) -> PyResult<()> {
        let owner = slf.borrow();
        let data = owner.image.raw_data()?;
        let len = isize::try_from(data.len()).map_err(|_| pyo3::exceptions::PyOverflowError::new_err("buffer is too large"))?;
        // SAFETY: FillInfo retains this immutable exporter. Its image clone keeps
        // the allocation alive; later image writes detach through copy-on-write.
        let status = unsafe { pyo3::ffi::PyBuffer_FillInfo(view, slf.as_ptr(), data.as_ptr().cast_mut().cast(), len, 1, flags) };
        if status == -1 {
            return Err(PyErr::fetch(slf.py()));
        }
        Ok(())
    }
}

#[pyfunction]
pub(crate) fn image_buffer_view(py: Python<'_>, image: &Image) -> PyResult<Py<PyMemoryView>> {
    if matches!(image.pixel_mode(), PixelMode::One | PixelMode::Lab) {
        return Err(pyo3::exceptions::PyValueError::new_err("buffer views require an unpacked mode other than LAB; convert explicitly"));
    }
    image.raw_data()?;
    let owner = Py::new(py, PixelBuffer { image: image.clone() })?;
    Ok(PyMemoryView::from(owner.bind(py).as_any())?.unbind())
}
