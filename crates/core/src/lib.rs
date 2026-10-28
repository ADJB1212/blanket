#![feature(default_field_values, portable_simd)]
pub mod cache;
pub mod color;
pub mod float;
pub mod parallel;
pub mod pixels;
pub mod raster;
pub mod samples;
pub mod simd;

pyo3::create_exception!(_blanket, UnidentifiedImageError, pyo3::exceptions::PyOSError);

pub use raster::{Image, PixelMode};

pub mod buffers;
pub mod geometry;
