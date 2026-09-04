use pyo3::prelude::*;

/// Native implementation details for the public `blanket` Python package.
#[pymodule]
fn _blanket(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
