use blanket_kernels::parallel::chunks_mut_above;
use pyo3::exceptions::{PyMemoryError, PyValueError};
use pyo3::prelude::*;

use crate::raster::Image;

pub fn buffer(size: (u32, u32), channels: usize) -> PyResult<Vec<u8>> {
    let mut result = reserved_buffer(size, channels)?;
    result.resize(size.0 as usize * size.1 as usize * channels, 0);
    Ok(result)
}

pub fn reserved_buffer(size: (u32, u32), channels: usize) -> PyResult<Vec<u8>> {
    let len = (size.0 as usize)
        .checked_mul(size.1 as usize)
        .and_then(|n| n.checked_mul(channels))
        .ok_or_else(|| PyValueError::new_err("image dimensions are too large"))?;
    let mut result = Vec::new();
    result.try_reserve_exact(len).map_err(|_| PyMemoryError::new_err("cannot allocate image"))?;
    Ok(result)
}

pub fn output(image: &Image, size: (u32, u32), pixels: Vec<u8>) -> PyResult<Image> {
    if image.mode.is_float() {
        return Image::from_color_float_bytes(size.0, size.1, image.mode, pixels);
    }
    if image.mode.is_integer() {
        return Image::from_integer_bytes(size.0, size.1, image.mode, pixels);
    }
    if image.bit_depth > 8 {
        return Image::from_sample_bytes(size.0, size.1, image.mode, pixels, image.bit_depth, None);
    }
    let mut result = Image::from_pixels(size.0, size.1, image.mode, pixels, None)?;
    result.palette.clone_from(&image.palette);
    Ok(result)
}

pub fn copy_rows<'a>(pixels: &mut Vec<u8>, row_bytes: usize, height: usize, source_row: impl Fn(usize) -> &'a [u8] + Sync) {
    assert!(pixels.is_empty());
    let len = row_bytes.checked_mul(height).expect("validated image dimensions");
    if len == 0 {
        return;
    }
    chunks_mut_above(&mut pixels.spare_capacity_mut()[..len], row_bytes * 32, 4 * 1024 * 1024, |band, rows| {
        for (i, row) in rows.chunks_exact_mut(row_bytes).enumerate() {
            let src = source_row(band * 32 + i);
            assert_eq!(src.len(), row.len());
            row.write_copy_of_slice(src);
        }
    });
    // SAFETY: every byte in the reserved image was initialized above. If a
    // worker panics, the vector remains empty and exposes no unwritten bytes.
    unsafe { pixels.set_len(len) };
}
