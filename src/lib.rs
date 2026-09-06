mod codecs;
mod ops;
mod ops_simd;
mod parallel;
mod raster;
mod simd;

use codecs::open_bytes;
use pyo3::prelude::*;
use raster::{Image, fromarray, frombytes};

pyo3::create_exception!(
    _blanket,
    UnidentifiedImageError,
    pyo3::exceptions::PyOSError
);

/// Native implementation details for the public `blanket` Python package.
#[pymodule]
fn _blanket(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    module.add(
        "UnidentifiedImageError",
        module.py().get_type::<UnidentifiedImageError>(),
    )?;
    module.add_class::<Image>()?;
    module.add_function(wrap_pyfunction!(fromarray, module)?)?;
    module.add_function(wrap_pyfunction!(frombytes, module)?)?;
    module.add_function(wrap_pyfunction!(open_bytes, module)?)?;
    ops::register(module)?;
    Ok(())
}
