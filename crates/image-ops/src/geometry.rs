use blanket_core::buffers::{buffer, copy_rows, output, reserved_buffer};
use blanket_core::{Image, PixelMode};
use blanket_kernels::parallel::{MIN_PARALLEL_BYTES, chunks_mut_above};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyfunction]
pub fn ops_transpose(py: Python<'_>, image: &Image, orientation: u8) -> PyResult<Image> {
    let source = image.raw_data()?;
    if !(1..=8).contains(&orientation) {
        return Err(PyValueError::new_err("invalid orientation"));
    }
    if orientation == 1 {
        return image.copy(py);
    }
    let size = if orientation >= 5 { (image.height(), image.width()) } else { (image.width(), image.height()) };
    let c = image.raw_bytes_per_pixel();
    if orientation == 4 {
        let mut pixels = reserved_buffer(size, c)?;
        py.detach(|| {
            let row_bytes = image.width() as usize * c;
            copy_rows(&mut pixels, row_bytes, image.height() as usize, |y| {
                let start = (image.height() as usize - 1 - y) * row_bytes;
                &source[start..start + row_bytes]
            });
        });
        return output(image, size, pixels);
    }
    if matches!(c, 1 | 2 | 4) && matches!(orientation, 2 | 3) {
        let mut pixels = reserved_buffer(size, c)?;
        let width = image.width() as usize;
        let height = image.height() as usize;
        let len = source.len();
        py.detach(|| {
            chunks_mut_above(&mut pixels.spare_capacity_mut()[..len], width.max(1) * c * 32, 4 << 20, |band, rows| {
                for (i, row) in rows.chunks_exact_mut(width.max(1) * c).enumerate() {
                    let y = band * 32 + i;
                    let sy = if orientation == 3 { height - 1 - y } else { y };
                    let src = &source[sy * width * c..(sy + 1) * width * c];
                    match c {
                        1 => blanket_kernels::ops::reverse_gray(src, row),
                        2 => blanket_kernels::ops::reverse_packed::<2>(src, row),
                        4 => blanket_kernels::ops::reverse_packed::<4>(src, row),
                        _ => unreachable!(),
                    }
                }
            });
            // SAFETY: every row was initialized by the reversal kernel.
            unsafe { pixels.set_len(len) };
        });
        return output(image, size, pixels);
    }
    let mut pixels = buffer(size, c)?;
    if image.pixel_mode().is_wide_scalar() {
        py.detach(|| match c {
            2 => transpose::<2>(source, &mut pixels, image.width() as usize, image.height() as usize, orientation),
            4 => transpose::<4>(source, &mut pixels, image.width() as usize, image.height() as usize, orientation),
            12 => transpose::<12>(source, &mut pixels, image.width() as usize, image.height() as usize, orientation),
            16 => transpose::<16>(source, &mut pixels, image.width() as usize, image.height() as usize, orientation),
            _ => unreachable!(),
        });
        return output(image, size, pixels);
    }
    if image.bit_depth() > 8 {
        py.detach(|| match image.pixel_mode() {
            PixelMode::One | PixelMode::L => transpose::<2>(source, &mut pixels, image.width() as usize, image.height() as usize, orientation),
            PixelMode::La | PixelMode::PremulLa | PixelMode::Pa => {
                transpose::<4>(source, &mut pixels, image.width() as usize, image.height() as usize, orientation)
            }
            PixelMode::Rgb | PixelMode::Hsv | PixelMode::YCbCr | PixelMode::Lab => {
                transpose::<6>(source, &mut pixels, image.width() as usize, image.height() as usize, orientation)
            }
            PixelMode::Rgba | PixelMode::PremulRgba | PixelMode::Rgbx | PixelMode::Cmyk => {
                transpose::<8>(source, &mut pixels, image.width() as usize, image.height() as usize, orientation)
            }
            _ => unreachable!(),
        });
        return output(image, size, pixels);
    }
    py.detach(|| match image.pixel_mode() {
        PixelMode::One | PixelMode::L => transpose::<1>(source, &mut pixels, image.width() as usize, image.height() as usize, orientation),
        PixelMode::La | PixelMode::PremulLa | PixelMode::Pa => {
            transpose::<2>(source, &mut pixels, image.width() as usize, image.height() as usize, orientation)
        }
        PixelMode::Rgb | PixelMode::Hsv | PixelMode::YCbCr | PixelMode::Lab => {
            transpose::<3>(source, &mut pixels, image.width() as usize, image.height() as usize, orientation)
        }
        PixelMode::Rgba | PixelMode::PremulRgba | PixelMode::Rgbx | PixelMode::Cmyk => {
            transpose::<4>(source, &mut pixels, image.width() as usize, image.height() as usize, orientation)
        }
        _ => unreachable!(),
    });
    output(image, size, pixels)
}

pub fn transpose<const C: usize>(source: &[u8], output: &mut [u8], w: usize, h: usize, orientation: u8) {
    if C == 3 && orientation >= 5 {
        transpose_four::<3, 12>(source, output, w, h, orientation, blanket_kernels::ops::transpose_rgb_four);
        return;
    }
    if C == 4 && orientation >= 5 {
        transpose_four::<4, 16>(source, output, w, h, orientation, blanket_kernels::ops::transpose_rgba_four);
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
                    let reverse = orientation == 7 || orientation == 8;
                    let sx = if reverse { w - band * 32 - count } else { band * 32 };
                    let src = &source[sy * w + sx..sy * w + sx + count];
                    if reverse {
                        for (row, pixel) in rows.rchunks_exact_mut(width * C).zip(src) {
                            row[x * C..(x + 1) * C].copy_from_slice(pixel);
                        }
                    } else {
                        for (row, pixel) in rows.chunks_exact_mut(width * C).zip(src) {
                            row[x * C..(x + 1) * C].copy_from_slice(pixel);
                        }
                    }
                }
            }
        }
    });
}

fn transpose_four<const C: usize, const N: usize>(
    source: &[u8], output: &mut [u8], w: usize, h: usize, orientation: u8, kernel: impl Fn([&[u8]; 4]) -> [[u8; N]; 4] + Sync + Send,
) {
    let reverse_rows = orientation == 7 || orientation == 8;
    let reverse_columns = orientation == 6 || orientation == 7;
    chunks_mut_above(output, h * C * 32, 1024 << 10, |band, rows| {
        for x0 in (0..h).step_by(32) {
            let limit = (x0 + 32).min(h);
            for (group, rows) in rows.chunks_mut(h * C * 4).enumerate() {
                let count = rows.len() / (h * C);
                let y = band * 32 + group * 4;
                if count == 4 {
                    let sx = if reverse_rows { w - y - 4 } else { y };
                    let end = limit / 4 * 4;
                    for x in (x0..end).step_by(4) {
                        let input = std::array::from_fn(|i| {
                            let sy = if reverse_columns { h - 1 - x - i } else { x + i };
                            &source[(sy * w + sx) * C..][..N]
                        });
                        let columns = kernel(input);
                        if reverse_rows {
                            for (row, column) in rows.chunks_exact_mut(h * C).zip(columns.iter().rev()) {
                                row[x * C..x * C + N].copy_from_slice(column);
                            }
                        } else {
                            for (row, column) in rows.chunks_exact_mut(h * C).zip(&columns) {
                                row[x * C..x * C + N].copy_from_slice(column);
                            }
                        }
                    }
                    for (i, row) in rows.chunks_exact_mut(h * C).enumerate() {
                        for x in end..limit {
                            let sx = if reverse_rows { w - 1 - y - i } else { y + i };
                            let sy = if reverse_columns { h - 1 - x } else { x };
                            let src = (sy * w + sx) * C;
                            row[x * C..x * C + C].copy_from_slice(&source[src..src + C]);
                        }
                    }
                } else {
                    for (i, row) in rows.chunks_exact_mut(h * C).enumerate() {
                        for (x, dst) in row.as_chunks_mut::<C>().0.iter_mut().enumerate().take(limit).skip(x0) {
                            let sx = if reverse_rows { w - 1 - y - i } else { y + i };
                            let sy = if reverse_columns { h - 1 - x } else { x };
                            let src = (sy * w + sx) * C;
                            dst.copy_from_slice(&source[src..src + C]);
                        }
                    }
                }
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiled_transposes_match_scalar_coordinates_with_tails_and_guards() {
        fn check<const C: usize>(w: usize, h: usize, offset: usize) {
            let source: Vec<u8> = (0..offset + w * h * C).map(|i| (i * 37 + i / 7) as u8).collect();
            for orientation in 1..=8 {
                let (width, height) = if orientation >= 5 { (h, w) } else { (w, h) };
                let mut expected = vec![93; offset + w * h * C + 16];
                for y in 0..height {
                    for x in 0..width {
                        let (sx, sy) = match orientation {
                            1 => (x, y),
                            2 => (w - 1 - x, y),
                            3 => (w - 1 - x, h - 1 - y),
                            4 => (x, h - 1 - y),
                            5 => (y, x),
                            6 => (y, h - 1 - x),
                            7 => (w - 1 - y, h - 1 - x),
                            8 => (w - 1 - y, x),
                            _ => unreachable!(),
                        };
                        let start = offset + (y * width + x) * C;
                        let src = offset + (sy * w + sx) * C;
                        expected[start..start + C].copy_from_slice(&source[src..src + C]);
                    }
                }
                let mut output = vec![93; expected.len()];
                transpose::<C>(&source[offset..], &mut output[offset..offset + w * h * C], w, h, orientation);
                assert_eq!(output, expected, "C={C}, w={w}, h={h}, offset={offset}, orientation={orientation}");
            }
        }
        for offset in [0, 1, 7] {
            for w in [0, 1, 3, 4, 5, 15, 16, 17, 31, 32, 33] {
                for h in [0, 1, 3, 4, 5, 31, 32, 33] {
                    check::<3>(w, h, offset);
                    check::<4>(w, h, offset);
                }
            }
        }
        check::<3>(521, 509, 1);
        check::<4>(521, 509, 1);
    }
}
