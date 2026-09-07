//! Native primitives used by the Python ImageEnhance API.

use pyo3::exceptions::{PyMemoryError, PyValueError};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::parallel::{CHUNK_PIXELS, MIN_PARALLEL_BYTES, chunks_mut, chunks_mut_above};
use crate::raster::{Image, PixelMode};

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(enhance_blend, module)?)?;
    module.add_function(wrap_pyfunction!(enhance_color, module)?)?;
    module.add_function(wrap_pyfunction!(enhance_contrast, module)?)?;
    module.add_function(wrap_pyfunction!(enhance_brightness, module)?)?;
    module.add_function(wrap_pyfunction!(enhance_sharpness, module)?)?;
    Ok(())
}

fn buffer(len: usize) -> PyResult<Vec<u8>> {
    let mut result = Vec::new();
    result
        .try_reserve_exact(len)
        .map_err(|_| PyMemoryError::new_err("cannot allocate image"))?;
    result.resize(len, 0);
    Ok(result)
}

fn output(image: &Image, pixels: Vec<u8>) -> PyResult<Image> {
    Image::from_pixels(image.width, image.height, image.mode, pixels, None)
}

fn copy(image: &Image, source: &[u8]) -> PyResult<Image> {
    let mut pixels = buffer(source.len())?;
    pixels.copy_from_slice(source);
    output(image, pixels)
}

/// Pillow's core blend uses a single-precision alpha and truncates toward zero.
#[pyfunction]
fn enhance_blend(py: Python<'_>, first: &Image, second: &Image, factor: f32) -> PyResult<Image> {
    let first_pixels = first.pixel_data()?;
    let second_pixels = second.pixel_data()?;
    if first.mode != second.mode || first.width != second.width || first.height != second.height {
        return Err(PyValueError::new_err("images do not match"));
    }
    if std::ptr::eq(first, second) || factor == 0.0 {
        return copy(first, first_pixels);
    }
    if factor == 1.0 {
        return copy(second, second_pixels);
    }

    let mut pixels = buffer(first_pixels.len())?;
    py.detach(|| {
        chunks_mut(&mut pixels, CHUNK_PIXELS * first.mode.channels(), |i, dst| {
            let start = i * CHUNK_PIXELS * first.mode.channels();
            blend_bytes(
                &first_pixels[start..start + dst.len()],
                &second_pixels[start..start + dst.len()],
                dst,
                factor,
            );
        });
    });
    output(first, pixels)
}

fn blend_bytes(first: &[u8], second: &[u8], output: &mut [u8], factor: f32) {
    for ((&a, &b), dst) in first.iter().zip(second).zip(output) {
        *dst = (f32::from(a) + factor * (f32::from(b) - f32::from(a))) as u8;
    }
}

#[pyfunction]
fn enhance_color(py: Python<'_>, image: &Image) -> PyResult<Image> {
    let source = image.pixel_data()?;
    if image.mode == PixelMode::L {
        return copy(image, source);
    }

    let mut pixels = buffer(source.len())?;
    py.detach(|| match image.mode {
        PixelMode::L => unreachable!(),
        PixelMode::Rgb => color_degenerate::<3>(source, &mut pixels),
        PixelMode::Rgba => color_degenerate::<4>(source, &mut pixels),
    });
    output(image, pixels)
}

fn color_degenerate<const C: usize>(source: &[u8], output: &mut [u8]) {
    chunks_mut(output, CHUNK_PIXELS * C, |i, dst| {
        let start = i * CHUNK_PIXELS * C;
        for (source, output) in source[start..].as_chunks::<C>().0.iter().zip(dst.as_chunks_mut::<C>().0) {
            let luma = crate::simd::pillow_luma(source[0], source[1], source[2]);
            output[0] = luma;
            output[1] = luma;
            output[2] = luma;
            if C == 4 {
                output[3] = source[3];
            }
        }
    });
}

#[pyfunction]
fn enhance_contrast(py: Python<'_>, image: &Image) -> PyResult<Image> {
    let source = image.pixel_data()?;
    let sum = py.detach(|| match image.mode {
        PixelMode::L => byte_sum(source),
        PixelMode::Rgb => luminance_sum::<3>(source),
        PixelMode::Rgba => luminance_sum::<4>(source),
    });
    let pixel_count = source.len() / image.mode.channels();
    let mean = if pixel_count == 0 {
        0
    } else {
        (sum as f64 / pixel_count as f64 + 0.5) as u8
    };

    let mut pixels = buffer(source.len())?;
    py.detach(|| match image.mode {
        PixelMode::L => pixels.fill(mean),
        PixelMode::Rgb => chunks_mut(&mut pixels, CHUNK_PIXELS * 3, |_, dst| {
            for pixel in dst.as_chunks_mut::<3>().0 {
                pixel.fill(mean);
            }
        }),
        PixelMode::Rgba => chunks_mut(&mut pixels, CHUNK_PIXELS * 4, |i, dst| {
            let start = i * CHUNK_PIXELS * 4;
            for (source, output) in source[start..].as_chunks::<4>().0.iter().zip(dst.as_chunks_mut::<4>().0) {
                *output = [mean, mean, mean, source[3]];
            }
        }),
    });
    output(image, pixels)
}

fn byte_sum(source: &[u8]) -> u64 {
    if source.len() >= MIN_PARALLEL_BYTES {
        source.par_iter().map(|&value| u64::from(value)).sum()
    } else {
        source.iter().map(|&value| u64::from(value)).sum()
    }
}

fn luminance_sum<const C: usize>(source: &[u8]) -> u64 {
    let pixels = source.as_chunks::<C>().0;
    let luminance = |pixel: &[u8; C]| u64::from(crate::simd::pillow_luma(pixel[0], pixel[1], pixel[2]));
    if source.len() >= MIN_PARALLEL_BYTES {
        pixels.par_iter().map(luminance).sum()
    } else {
        pixels.iter().map(luminance).sum()
    }
}

#[pyfunction]
fn enhance_brightness(py: Python<'_>, image: &Image) -> PyResult<Image> {
    let mut pixels = buffer(
        (image.width as usize)
            .checked_mul(image.height as usize)
            .and_then(|count| count.checked_mul(image.mode.channels()))
            .ok_or_else(|| PyValueError::new_err("image dimensions are too large"))?,
    )?;
    if image.mode == PixelMode::Rgba {
        let source = image.pixel_data()?;
        py.detach(|| {
            chunks_mut(&mut pixels, CHUNK_PIXELS * 4, |i, dst| {
                let start = i * CHUNK_PIXELS * 4;
                for (source, output) in source[start..].as_chunks::<4>().0.iter().zip(dst.as_chunks_mut::<4>().0) {
                    output[3] = source[3];
                }
            });
        });
    }
    output(image, pixels)
}

#[pyfunction]
fn enhance_sharpness(py: Python<'_>, image: &Image) -> PyResult<Image> {
    let source = image.pixel_data()?;
    let mut pixels = buffer(source.len())?;
    pixels.copy_from_slice(source);
    if image.width < 3 || image.height < 3 {
        return output(image, pixels);
    }

    let width = image.width as usize;
    py.detach(|| match image.mode {
        PixelMode::L => smooth::<1>(source, &mut pixels, width),
        PixelMode::Rgb => smooth::<3>(source, &mut pixels, width),
        PixelMode::Rgba => smooth::<4>(source, &mut pixels, width),
    });
    output(image, pixels)
}

fn smooth<const C: usize>(source: &[u8], output: &mut [u8], width: usize) {
    let row_bytes = width * C;
    let height = source.len() / row_bytes;
    chunks_mut_above(output, row_bytes * 32, MIN_PARALLEL_BYTES, |band, rows| {
        for (i, row) in rows.chunks_exact_mut(row_bytes).enumerate() {
            let y = band * 32 + i;
            if y == 0 || y + 1 >= height {
                continue;
            }
            for x in 1..width - 1 {
                let center = y * row_bytes + x * C;
                let channels = if C == 4 { 3 } else { C };
                for channel in 0..channels {
                    let mut sum = u32::from(source[center + channel]) * 5;
                    for dy in [0, row_bytes, row_bytes * 2] {
                        let upper_left = center - row_bytes - C + dy + channel;
                        sum += u32::from(source[upper_left]);
                        sum += u32::from(source[upper_left + C]);
                        sum += u32::from(source[upper_left + C * 2]);
                    }
                    // The center was included once by the loop and needs a
                    // total weight of five. Pillow adds 0.5 before truncation.
                    sum -= u32::from(source[center + channel]);
                    row[x * C + channel] = ((sum + 6) / 13) as u8;
                }
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blend_truncates_and_clips_like_pillow() {
        let first = [100, 200, 10];
        let second = [200, 0, 20];
        let mut output = [0; 3];
        blend_bytes(&first, &second, &mut output, 0.5);
        assert_eq!(output, [150, 100, 15]);
        blend_bytes(&first, &second, &mut output, 2.0);
        assert_eq!(output, [255, 0, 30]);
        blend_bytes(&first, &second, &mut output, f32::NAN);
        assert_eq!(output, [0, 0, 0]);
    }

    #[test]
    fn smooth_copies_borders_and_preserves_alpha() {
        let source: Vec<u8> = (0..100).collect();
        let mut output = source.clone();
        smooth::<4>(&source, &mut output, 5);
        assert_eq!(&output[..20], &source[..20]);
        assert_eq!(&output[80..], &source[80..]);
        for (actual, expected) in output.as_chunks::<4>().0.iter().zip(source.as_chunks::<4>().0) {
            assert_eq!(actual[3], expected[3]);
        }
    }
}
