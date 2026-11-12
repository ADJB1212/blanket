#![feature(portable_simd)]

pub mod convert;
pub mod float;
pub mod geometry;

use pyo3::prelude::*;

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(convert::convert, module)?)?;
    module.add_function(wrap_pyfunction!(convert::convert_palette_alpha, module)?)?;
    module.add_function(wrap_pyfunction!(convert::convert_matrix, module)?)?;
    module.add_function(wrap_pyfunction!(convert::convert_transparent_alpha, module)?)?;
    module.add_function(wrap_pyfunction!(convert::convert_dithered_bilevel, module)?)?;
    module.add_function(wrap_pyfunction!(geometry::ops_transpose, module)?)?;
    Ok(())
}
