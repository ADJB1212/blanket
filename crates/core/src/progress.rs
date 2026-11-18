//! Cooperative checkpoints for operations running outside the Python GIL.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PyBool;

use crate::OperationCancelledError;

#[derive(Debug, Default)]
pub struct Progress {
    callback: Option<Py<PyAny>>,
    checkpoints: AtomicUsize,
    failed: AtomicBool,
}

impl Progress {
    pub fn new(py: Python<'_>, callback: Option<Py<PyAny>>) -> PyResult<Self> {
        if callback.as_ref().is_some_and(|callback| !callback.bind(py).is_callable()) {
            return Err(PyTypeError::new_err("progress must be callable or None"));
        }
        Ok(Self { callback, ..Self::default() })
    }

    pub fn enabled(&self) -> bool {
        self.callback.is_some()
    }

    pub fn failed(&self) -> bool {
        self.failed.load(Ordering::Relaxed)
    }

    /// Report a boundary from the calling thread. Each operation invokes its
    /// checkpoints serially, outside worker pools.
    pub fn checkpoint(&self) -> PyResult<()> {
        let Some(callback) = &self.callback else { return Ok(()) };
        let completed = self.checkpoints.fetch_add(1, Ordering::Relaxed);
        let result = Python::attach(|py| {
            let result = callback.bind(py).call1((completed,))?;
            if result.is(PyBool::new(py, false)) {
                return Err(OperationCancelledError::new_err("operation cancelled"));
            }
            Ok(())
        });
        if result.is_err() {
            self.failed.store(true, Ordering::Relaxed);
        }
        result
    }
}
