#![feature(portable_simd)]

pub mod compressor;
pub mod compressor_simd;
pub mod lossy_compressor;

pub use compressor::Compressor;

use pyo3::prelude::*;

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    compressor::register(module)
}
