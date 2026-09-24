pub mod parallel;
pub mod raster;
pub mod simd;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod x86_pixels;

pyo3::create_exception!(_blanket, UnidentifiedImageError, pyo3::exceptions::PyOSError);

pub use raster::{Image, PixelMode};
