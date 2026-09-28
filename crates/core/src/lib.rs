#![feature(portable_simd)]

pub mod parallel;
pub mod pixels;
pub mod raster;
pub mod simd;

pyo3::create_exception!(_blanket, UnidentifiedImageError, pyo3::exceptions::PyOSError);

pub use raster::{Image, PixelMode};
