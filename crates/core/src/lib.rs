#![feature(default_field_values, portable_simd)]

pub mod cache;
pub mod color;
pub mod parallel;
pub mod pixels;
pub mod progress;
pub mod raster;
pub mod samples;

pyo3::create_exception!(_blanket, UnidentifiedImageError, pyo3::exceptions::PyOSError);
pyo3::create_exception!(_blanket, OperationCancelledError, pyo3::exceptions::PyException);

pub use raster::{Image, PixelMode};

pub mod buffers;
