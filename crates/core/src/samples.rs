//! Unsigned high-depth sample access shared by processing kernels.

use std::simd::{Simd, ToBytes};

use pyo3::prelude::*;

use crate::raster::{Image, PixelMode, alloc_pixels};

pub fn supported(image: &Image) -> bool {
    matches!(image.bit_depth, 10 | 12 | 16)
        && matches!(image.mode, PixelMode::L | PixelMode::Rgb | PixelMode::Rgba | PixelMode::I16 | PixelMode::I16L | PixelMode::I16B)
}

pub fn maximum(image: &Image) -> u16 {
    ((1_u32 << image.bit_depth) - 1) as u16
}

pub fn read(image: &Image) -> PyResult<Vec<u16>> {
    let source = image.raw_data()?;
    let mut values = Vec::new();
    values.try_reserve_exact(source.len() / 2).map_err(|_| pyo3::exceptions::PyMemoryError::new_err("cannot allocate image"))?;
    let end = source.len() / 32 * 32;
    for i in (0..end).step_by(32) {
        let block = if image.mode == PixelMode::I16B {
            crate::pixels::load_u16::<true>(&source[i..])
        } else {
            crate::pixels::load_u16::<false>(&source[i..])
        };
        values.extend_from_slice(&block.to_array());
    }
    values.extend(
        source[end..]
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&bytes| if image.mode == PixelMode::I16B { u16::from_be_bytes(bytes) } else { u16::from_le_bytes(bytes) }),
    );
    Ok(values)
}

pub fn output(image: &Image, values: &[u16]) -> PyResult<Image> {
    let mut pixels = alloc_pixels(values.len() * 2)?;
    let end = values.len() / 16 * 16;
    for i in (0..end).step_by(16) {
        let value = Simd::<u16, 16>::from_slice(&values[i..]);
        let bytes = if image.mode == PixelMode::I16B { value.to_be_bytes() } else { value.to_le_bytes() };
        bytes.copy_to_slice(&mut pixels[i * 2..i * 2 + 32]);
    }
    for (dst, &value) in pixels[end * 2..].as_chunks_mut::<2>().0.iter_mut().zip(&values[end..]) {
        *dst = if image.mode == PixelMode::I16B { value.to_be_bytes() } else { value.to_le_bytes() };
    }
    if image.mode.is_wide_scalar() {
        Image::from_integer_bytes(image.width, image.height, image.mode, pixels)
    } else {
        Image::from_sample_bytes(image.width, image.height, image.mode, pixels, image.bit_depth, None)
    }
}
