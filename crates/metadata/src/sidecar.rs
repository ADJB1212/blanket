//! Explicit sibling XMP file I/O; packet filtering uses the embedded XMP policy.

use std::io::ErrorKind;
use std::path::PathBuf;

use pyo3::prelude::*;
use pyo3::types::PyBytes;

#[pyfunction]
pub fn _read_xmp_sidecar<'py>(py: Python<'py>, path: PathBuf) -> PyResult<Option<Bound<'py, PyBytes>>> {
    let raw = py.detach(|| -> PyResult<Option<Vec<u8>>> {
        match std::fs::read(path.with_extension("xmp")) {
            Ok(raw) if raw.is_empty() => Ok(None),
            Ok(raw) => Ok(Some(raw)),
            Err(error) if error.kind() == ErrorKind::NotFound => Ok(None),
            Err(error) => Err(error.into()),
        }
    })?;
    Ok(raw.map(|raw| PyBytes::new(py, &raw)))
}

#[pyfunction]
pub fn _write_xmp_sidecar(py: Python<'_>, path: PathBuf, raw: &[u8]) -> PyResult<PathBuf> {
    py.detach(|| {
        let path = path.with_extension("xmp");
        std::fs::write(&path, raw)?;
        Ok(path)
    })
}
