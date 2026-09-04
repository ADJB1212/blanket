mod raster;

use pyo3::prelude::*;
use raster::{Image, frombytes};

/// Native implementation details for the public `blanket` Python package.
#[pymodule]
fn _blanket(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    module.add_class::<Image>()?;
    module.add_function(wrap_pyfunction!(frombytes, module)?)?;
    Ok(())
}
