use blanket_kernels::parallel::{MIN_PARALLEL_BYTES, chunks_mut_above};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::buffers::{buffer, copy_rows, output, reserved_buffer};
use crate::raster::{Image, PixelMode};

#[pyfunction]
pub fn ops_transpose(py: Python<'_>, image: &Image, orientation: u8) -> PyResult<Image> {
    let source = image.raw_data()?;
    if !(1..=8).contains(&orientation) {
        return Err(PyValueError::new_err("invalid orientation"));
    }
    if orientation == 1 {
        return image.copy(py);
    }
    let size = if orientation >= 5 { (image.height, image.width) } else { (image.width, image.height) };
    let c = image.raw_bytes_per_pixel();
    if orientation == 4 {
        let mut pixels = reserved_buffer(size, c)?;
        py.detach(|| {
            let row_bytes = image.width as usize * c;
            copy_rows(&mut pixels, row_bytes, image.height as usize, |y| {
                let start = (image.height as usize - 1 - y) * row_bytes;
                &source[start..start + row_bytes]
            });
        });
        return output(image, size, pixels);
    }
    if c == 1 && matches!(orientation, 2 | 3) {
        let mut pixels = reserved_buffer(size, c)?;
        let width = image.width as usize;
        let height = image.height as usize;
        let len = source.len();
        py.detach(|| {
            chunks_mut_above(&mut pixels.spare_capacity_mut()[..len], width.max(1) * 32, 4 << 20, |band, rows| {
                for (i, row) in rows.chunks_exact_mut(width.max(1)).enumerate() {
                    let y = band * 32 + i;
                    let sy = if orientation == 3 { height - 1 - y } else { y };
                    blanket_kernels::ops::reverse_gray(&source[sy * width..(sy + 1) * width], row);
                }
            });
            // SAFETY: every row was initialized by the reversal kernel.
            unsafe { pixels.set_len(len) };
        });
        return output(image, size, pixels);
    }
    let mut pixels = buffer(size, c)?;
    if image.mode.is_wide_scalar() {
        py.detach(|| match c {
            2 => transpose::<2>(source, &mut pixels, image.width as usize, image.height as usize, orientation),
            4 => transpose::<4>(source, &mut pixels, image.width as usize, image.height as usize, orientation),
            12 => transpose::<12>(source, &mut pixels, image.width as usize, image.height as usize, orientation),
            16 => transpose::<16>(source, &mut pixels, image.width as usize, image.height as usize, orientation),
            _ => unreachable!(),
        });
        return output(image, size, pixels);
    }
    if image.bit_depth > 8 {
        py.detach(|| match image.mode {
            PixelMode::One | PixelMode::L => transpose::<2>(source, &mut pixels, image.width as usize, image.height as usize, orientation),
            PixelMode::La | PixelMode::Pa => transpose::<4>(source, &mut pixels, image.width as usize, image.height as usize, orientation),
            PixelMode::Rgb | PixelMode::Hsv | PixelMode::YCbCr | PixelMode::Lab => {
                transpose::<6>(source, &mut pixels, image.width as usize, image.height as usize, orientation)
            }
            PixelMode::Rgba | PixelMode::Cmyk => transpose::<8>(source, &mut pixels, image.width as usize, image.height as usize, orientation),
            _ => unreachable!(),
        });
        return output(image, size, pixels);
    }
    py.detach(|| match image.mode {
        PixelMode::One | PixelMode::L => transpose::<1>(source, &mut pixels, image.width as usize, image.height as usize, orientation),
        PixelMode::La | PixelMode::Pa => transpose::<2>(source, &mut pixels, image.width as usize, image.height as usize, orientation),
        PixelMode::Rgb | PixelMode::Hsv | PixelMode::YCbCr | PixelMode::Lab => {
            transpose::<3>(source, &mut pixels, image.width as usize, image.height as usize, orientation)
        }
        PixelMode::Rgba | PixelMode::Cmyk => transpose::<4>(source, &mut pixels, image.width as usize, image.height as usize, orientation),
        _ => unreachable!(),
    });
    output(image, size, pixels)
}

pub fn transpose<const C: usize>(source: &[u8], output: &mut [u8], w: usize, h: usize, orientation: u8) {
    if C == 3 && orientation >= 5 {
        transpose_rgb(source, output, w, h, orientation);
        return;
    }
    let source = source.as_chunks::<C>().0;
    let width = if orientation >= 5 { h } else { w };
    // Bands and tiles keep 90-degree rotations local to cache and give every
    // worker exclusive ownership of complete destination rows.
    let threshold = if orientation < 5 { 2 * 1024 * 1024 } else { MIN_PARALLEL_BYTES };
    chunks_mut_above(output, width * C * 32, threshold, |band, rows| {
        if orientation < 5 {
            for (i, row) in rows.chunks_exact_mut(w * C).enumerate() {
                let y = band * 32 + i;
                let sy = if orientation == 3 || orientation == 4 { h - 1 - y } else { y };
                let src = &source[sy * w..(sy + 1) * w];
                if orientation == 1 || orientation == 4 {
                    row.copy_from_slice(src.as_flattened());
                } else if C == 3 {
                    blanket_kernels::ops::reverse_rgb(src.as_flattened(), row);
                } else if C == 1 {
                    let src = src.as_flattened();
                    let end = w / 64 * 64;
                    for x in (0..end).step_by(64) {
                        std::simd::Simd::<u8, 64>::from_slice(&src[w - x - 64..w - x]).reverse().copy_to_slice(&mut row[x..x + 64]);
                    }
                    for (dst, src) in row[end..].iter_mut().zip(src[..w - end].iter().rev()) {
                        *dst = *src;
                    }
                } else {
                    for (dst, src) in row.as_chunks_mut::<C>().0.iter_mut().zip(src.iter().rev()) {
                        *dst = *src;
                    }
                }
            }
        } else {
            for x0 in (0..width).step_by(32) {
                let count = rows.len() / (width * C);
                for x in x0..(x0 + 32).min(width) {
                    let sy = if orientation == 6 || orientation == 7 { h - 1 - x } else { x };
                    let sx = if orientation == 7 || orientation == 8 { w - band * 32 - count } else { band << 5 };
                    let src = &source[sy * w + sx..sy * w + sx + count];
                    for (i, row) in rows.chunks_exact_mut(width * C).enumerate() {
                        let i = if orientation == 7 || orientation == 8 { count - 1 - i } else { i };
                        row[x * C..(x + 1) * C].copy_from_slice(&src[i]);
                    }
                }
            }
        }
    });
}

fn transpose_rgb(source: &[u8], output: &mut [u8], w: usize, h: usize, orientation: u8) {
    let reverse_rows = orientation == 7 || orientation == 8;
    let reverse_columns = orientation == 6 || orientation == 7;
    chunks_mut_above(output, h * 3 * 32, MIN_PARALLEL_BYTES, |band, rows| {
        for (group, rows) in rows.chunks_mut(h * 3 * 4).enumerate() {
            let count = rows.len() / (h * 3);
            let y = band * 32 + group * 4;
            if count == 4 {
                let sx = if reverse_rows { w - y - 4 } else { y };
                let end = h / 4 * 4;
                for x in (0..end).step_by(4) {
                    let input = std::array::from_fn(|i| {
                        let sy = if reverse_columns { h - 1 - x - i } else { x + i };
                        &source[(sy * w + sx) * 3..][..12]
                    });
                    let columns = blanket_kernels::ops::transpose_rgb_four(input);
                    if reverse_rows {
                        for (row, column) in rows.chunks_exact_mut(h * 3).zip(columns.iter().rev()) {
                            row[x * 3..x * 3 + 12].copy_from_slice(column);
                        }
                    } else {
                        for (row, column) in rows.chunks_exact_mut(h * 3).zip(&columns) {
                            row[x * 3..x * 3 + 12].copy_from_slice(column);
                        }
                    }
                }
                for (i, row) in rows.chunks_exact_mut(h * 3).enumerate() {
                    for x in end..h {
                        let sx = if reverse_rows { w - 1 - y - i } else { y + i };
                        let sy = if reverse_columns { h - 1 - x } else { x };
                        let src = (sy * w + sx) * 3;
                        row[x * 3..x * 3 + 3].copy_from_slice(&source[src..src + 3]);
                    }
                }
            } else {
                for (i, row) in rows.chunks_exact_mut(h * 3).enumerate() {
                    for (x, dst) in row.as_chunks_mut::<3>().0.iter_mut().enumerate() {
                        let sx = if reverse_rows { w - 1 - y - i } else { y + i };
                        let sy = if reverse_columns { h - 1 - x } else { x };
                        let src = (sy * w + sx) * 3;
                        dst.copy_from_slice(&source[src..src + 3]);
                    }
                }
            }
        }
    });
}
