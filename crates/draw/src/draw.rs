//! Native primitives used by the Python `ImageDraw` API.

use blanket_core::raster::{Image, PixelMode};
use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::prelude::*;

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(draw_points, module)?)?;
    module.add_function(wrap_pyfunction!(draw_rectangle, module)?)?;
    module.add_function(wrap_pyfunction!(draw_unimplemented, module)?)?;
    Ok(())
}

fn ink(image: &Image, value: &Bound<'_, PyAny>, blend: bool) -> PyResult<Vec<u8>> {
    if blend && image.mode != PixelMode::Rgb {
        return Err(PyValueError::new_err("mode mismatch"));
    }
    image.raw_data()?;
    if !image.mode.is_wide_scalar() {
        image.pixel_data()?;
    }
    let mode = if blend { PixelMode::Rgba } else { image.mode };
    let mut pixel = Image {
        width: 1,
        height: 1,
        mode,
        pixels: Some(vec![0; mode.channels() * mode.sample_bytes()]),
        bit_depth: image.bit_depth,
        format: None,
        palette: None,
    };
    pixel.putpixel((0, 0), value)?;
    Ok(pixel.pixels.unwrap())
}

fn paint(image: &mut Image, x: i64, y: i64, ink: &[u8], blend: bool) {
    if x < 0 || y < 0 || x >= i64::from(image.width) || y >= i64::from(image.height) {
        return;
    }
    let stride = image.mode.channels() * image.mode.sample_bytes();
    let offset = (y as usize * image.width as usize + x as usize) * stride;
    let pixel = &mut image.pixels.as_mut().unwrap()[offset..offset + stride];
    if blend {
        let alpha = u32::from(ink[3]);
        for (dst, src) in pixel.iter_mut().zip(ink) {
            *dst = ((u32::from(*dst) * (255 - alpha) + u32::from(*src) * alpha + 127) / 255) as u8;
        }
    } else {
        pixel.copy_from_slice(ink);
    }
}

#[pyfunction]
fn draw_points(image: &mut Image, points: Vec<(i64, i64)>, value: &Bound<'_, PyAny>, blend: bool) -> PyResult<()> {
    let mut ink = ink(image, value, blend)?;
    // Pillow points use native byte order even for explicit-endian modes.
    if (cfg!(target_endian = "little") && image.mode == PixelMode::I16B)
        || (cfg!(target_endian = "big") && matches!(image.mode, PixelMode::I16 | PixelMode::I16L))
    {
        ink.reverse();
    }
    for (x, y) in points {
        paint(image, x, y, &ink, blend);
    }
    Ok(())
}

#[pyfunction]
fn draw_rectangle(image: &mut Image, bounds: (i64, i64, i64, i64), value: &Bound<'_, PyAny>, width: i64, blend: bool) -> PyResult<()> {
    let ink = ink(image, value, blend)?;
    let (left, top, right, bottom) = bounds;
    if width == 0 {
        for y in top.max(0)..=bottom.min(i64::from(image.height) - 1) {
            hline(image, left, right, y, &ink, blend);
        }
    } else {
        for i in 0..width {
            hline(image, left, right, top.saturating_add(i), &ink, blend);
            hline(image, left, right, bottom.saturating_sub(i), &ink, blend);
            let start = top.saturating_add(width);
            let end = bottom.saturating_sub(width).saturating_add(1);
            let (first, last) = if start < end { (start, end - 1) } else { (end.saturating_add(1), start) };
            for y in first.max(0)..=last.min(i64::from(image.height) - 1) {
                paint(image, right.saturating_sub(i), y, &ink, blend);
                paint(image, left.saturating_add(i), y, &ink, blend);
            }
        }
    }
    Ok(())
}

fn hline(image: &mut Image, left: i64, right: i64, y: i64, ink: &[u8], blend: bool) {
    if y < 0 || y >= i64::from(image.height) {
        return;
    }
    for x in left.max(0)..=right.min(i64::from(image.width) - 1) {
        paint(image, x, y, ink, blend);
    }
}

#[pyfunction]
fn draw_unimplemented(name: &str) -> PyResult<()> {
    Err(PyNotImplementedError::new_err(format!("ImageDraw.{name} is not implemented")))
}
