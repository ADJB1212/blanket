#![feature(portable_simd)]
pub mod color;
pub mod parallel;
pub mod pixels;
pub mod raster;
pub mod simd;

pyo3::create_exception!(_blanket, UnidentifiedImageError, pyo3::exceptions::PyOSError);

pub use raster::{Image, PixelMode};

pub fn littlecms_version() -> String {
    let version = lcms2::version();
    format!("{}.{}", version / 1_000, version % 1_000 / 10)
}
