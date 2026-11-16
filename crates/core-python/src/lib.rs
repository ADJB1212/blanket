#![feature(portable_simd)]
#![feature(f16)]

mod capsule;
mod color;
mod io;
mod raw;
mod tensor;
mod views;

use pyo3::prelude::*;

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(io::fromarray, module)?)?;
    module.add_function(wrap_pyfunction!(capsule::image_capsule, module)?)?;
    module.add_function(wrap_pyfunction!(views::pillow_buffer_view, module)?)?;
    module.add_function(wrap_pyfunction!(io::frombytes, module)?)?;
    module.add_function(wrap_pyfunction!(io::frombuffer_copy, module)?)?;
    module.add_function(wrap_pyfunction!(io::mapped_getpixel, module)?)?;
    module.add_function(wrap_pyfunction!(io::tobytes, module)?)?;
    module.add_function(wrap_pyfunction!(tensor::tensor_export, module)?)?;
    module.add_function(wrap_pyfunction!(views::image_buffer_view, module)?)?;
    module.add_function(wrap_pyfunction!(color::color_getcolor, module)?)?;
    module.add_function(wrap_pyfunction!(color::color_pixel, module)?)?;
    Ok(())
}
