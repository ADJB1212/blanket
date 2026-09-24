use blanket_codecs::open_bytes;
use blanket_core::{
    Image, UnidentifiedImageError,
    raster::{fromarray, frombytes},
};
use pyo3::prelude::*;

/// Native implementation details for the public `blanket` Python package.
#[pymodule]
fn _blanket(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    module.add("UnidentifiedImageError", module.py().get_type::<UnidentifiedImageError>())?;
    module.add_class::<Image>()?;
    module.add_function(wrap_pyfunction!(fromarray, module)?)?;
    module.add_function(wrap_pyfunction!(frombytes, module)?)?;
    module.add_function(wrap_pyfunction!(open_bytes, module)?)?;
    module.add_function(wrap_pyfunction!(blanket_codecs::_encode, module)?)?;
    blanket_ops::enhance::register(module)?;
    blanket_ops::chops::register(module)?;
    blanket_ops::composite::register(module)?;
    blanket_codecs::compressor::register(module)?;
    blanket_ops::filter::register(module)?;
    blanket_ops::ops::register(module)?;
    blanket_ops::palette::register(module)?;
    blanket_ops::quantize::register(module)?;
    blanket_ops::stat::register(module)?;
    Ok(())
}
