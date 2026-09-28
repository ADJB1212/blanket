mod encode;

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
    module.add_function(wrap_pyfunction!(encode::_encode, module)?)?;
    blanket_enhance::enhance::register(module)?;
    blanket_composite::chops::register(module)?;
    blanket_composite::composite::register(module)?;
    blanket_compressor::register(module)?;
    blanket_filter::filter::register(module)?;
    blanket_ops::ops::register(module)?;
    blanket_palette::palette::register(module)?;
    blanket_palette::quantize::register(module)?;
    blanket_stat::stat::register(module)?;
    Ok(())
}
