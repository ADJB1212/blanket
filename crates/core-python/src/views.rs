//! Read-only exports that retain copy-on-write raster storage.

use blanket_core::{Image, PixelMode};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyCapsule, PyMemoryView, PyType};

static PILLOW_CORE_TYPE: PyOnceLock<Py<PyType>> = PyOnceLock::new();

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

#[repr(C)]
struct ArrowArray {
    length: i64,
    null_count: i64,
    offset: i64,
    n_buffers: i64,
    n_children: i64,
    buffers: *const *const std::ffi::c_void,
    children: *const *const Self,
    dictionary: *const Self,
    release: Option<unsafe extern "C" fn(*mut Self)>,
    private_data: *mut std::ffi::c_void,
}

#[pyclass(frozen, module = "blanket._blanket")]
#[derive(Debug)]
struct PillowBuffer {
    _array: Py<PyAny>,
    address: usize,
    len: isize,
}

#[pymethods]
impl PillowBuffer {
    /// # Safety
    /// CPython supplies a writable Py_buffer and valid request flags.
    unsafe fn __getbuffer__(slf: Bound<'_, Self>, view: *mut pyo3::ffi::Py_buffer, flags: std::ffi::c_int) -> PyResult<()> {
        let owner = slf.borrow();
        // SAFETY: The Arrow capsule retains the allocation. FillInfo retains
        // this exporter, keeping the capsule alive for the entire buffer lease.
        let status = unsafe { pyo3::ffi::PyBuffer_FillInfo(view, slf.as_ptr(), owner.address as *mut _, owner.len, 1, flags) };
        if status == -1 {
            return Err(PyErr::fetch(slf.py()));
        }
        Ok(())
    }
}

#[pyfunction]
pub(crate) fn pillow_buffer_view(py: Python<'_>, source: &Bound<'_, PyAny>, mode: &str, size: (u32, u32)) -> PyResult<Py<PyMemoryView>> {
    let core = source.getattr("im")?;
    let core_type = PILLOW_CORE_TYPE.get_or_try_init(py, || {
        let probe = py.import("PIL._imaging")?.call_method1("new", ("L", (1, 1)))?;
        Ok::<_, PyErr>(probe.get_type().unbind())
    })?;
    if !core.get_type().is(core_type.bind(py)) {
        return Err(PyValueError::new_err("buffer sharing requires a native Pillow core"));
    }
    if core.getattr("mode")?.extract::<String>()? != mode || core.getattr("size")?.extract::<(u32, u32)>()? != size {
        return Err(PyValueError::new_err("Pillow core layout does not match the requested image"));
    }
    let bytes = match mode {
        "L" | "P" => 1,
        "I;16" | "I;16L" | "I;16B" | "I;16N" => 2,
        "I" | "F" | "RGBA" | "RGBX" | "CMYK" => 4,
        _ => return Err(PyValueError::new_err("Pillow storage requires conversion for this mode")),
    };
    let len = (size.0 as usize)
        .checked_mul(size.1 as usize)
        .and_then(|n| n.checked_mul(bytes))
        .and_then(|n| isize::try_from(n).ok())
        .ok_or_else(|| PyValueError::new_err("image dimensions are too large"))?;
    let capsule = core.call_method0("__arrow_c_array__")?;
    let pointer = capsule.cast::<PyCapsule>()?.pointer_checked(Some(c"arrow_array"))?;
    // SAFETY: The exact native Pillow core type supplies an ArrowArray under this name.
    // The capsule remains owned by PillowBuffer and is never consumed or renamed.
    let array = unsafe { &*pointer.cast::<ArrowArray>().as_ptr() };
    if array.release.is_none() || array.offset != 0 || array.null_count != 0 || array.length != i64::from(size.0) * i64::from(size.1) {
        return Err(PyValueError::new_err("unsupported Pillow Arrow layout"));
    }
    let multiband = matches!(mode, "RGBA" | "RGBX" | "CMYK");
    if array.n_children != i64::from(multiband) {
        return Err(PyValueError::new_err("unsupported Pillow Arrow channel layout"));
    }
    let samples = if array.n_children == 1 && !array.children.is_null() {
        // SAFETY: A valid Arrow fixed-size-list array owns one child pointer.
        unsafe { (*array.children).as_ref() }.ok_or_else(|| PyValueError::new_err("missing Pillow Arrow child"))?
    } else {
        array
    };
    if samples.n_buffers != 2 || samples.buffers.is_null() || samples.offset != 0 || samples.null_count != 0 {
        return Err(PyValueError::new_err("unsupported Pillow Arrow buffers"));
    }
    if samples.length != array.length * if multiband { 4 } else { 1 } {
        return Err(PyValueError::new_err("unsupported Pillow Arrow sample count"));
    }
    // SAFETY: Primitive Arrow arrays expose their data in the second buffer.
    let address = unsafe { *samples.buffers.add(1) } as usize;
    if address == 0 || len == 0 {
        return Err(PyValueError::new_err("empty Pillow Arrow buffer"));
    }
    let owner = Py::new(py, PillowBuffer { _array: capsule.unbind(), address, len })?;
    Ok(PyMemoryView::from(owner.bind(py).as_any())?.unbind())
}
