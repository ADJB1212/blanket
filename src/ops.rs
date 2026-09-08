//! Pixel primitives used by the Python ImageOps API.
use pyo3::exceptions::{PyMemoryError, PyValueError};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::borrow::Cow;

use crate::parallel::{CHUNK_PIXELS, MIN_PARALLEL_BYTES, chunks_mut, chunks_mut_above};
use crate::raster::{Image, PixelMode};

type BoxI = (i64, i64, i64, i64);
type BoxF = (f64, f64, f64, f64);
type Mesh = Vec<(BoxI, [f64; 8])>;

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(ops_lut, module)?)?;
    module.add_function(wrap_pyfunction!(ops_colorize, module)?)?;
    module.add_function(wrap_pyfunction!(ops_histogram, module)?)?;
    module.add_function(wrap_pyfunction!(ops_split, module)?)?;
    module.add_function(wrap_pyfunction!(ops_entropy, module)?)?;
    module.add_function(wrap_pyfunction!(ops_canvas, module)?)?;
    module.add_function(wrap_pyfunction!(ops_transpose, module)?)?;
    module.add_function(wrap_pyfunction!(ops_resize, module)?)?;
    module.add_function(wrap_pyfunction!(ops_reduce, module)?)?;
    module.add_function(wrap_pyfunction!(ops_mesh, module)?)?;
    module.add_function(wrap_pyfunction!(ops_warp, module)?)?;
    module.add_function(wrap_pyfunction!(ops_affine, module)?)?;
    Ok(())
}

fn buffer(size: (u32, u32), channels: usize) -> PyResult<Vec<u8>> {
    let mut result = reserved_buffer(size, channels)?;
    result.resize(size.0 as usize * size.1 as usize * channels, 0);
    Ok(result)
}

fn reserved_buffer(size: (u32, u32), channels: usize) -> PyResult<Vec<u8>> {
    let len = (size.0 as usize)
        .checked_mul(size.1 as usize)
        .and_then(|n| n.checked_mul(channels))
        .ok_or_else(|| PyValueError::new_err("image dimensions are too large"))?;
    let mut result = Vec::new();
    result
        .try_reserve_exact(len)
        .map_err(|_| PyMemoryError::new_err("cannot allocate image"))?;
    Ok(result)
}

fn output(image: &Image, size: (u32, u32), pixels: Vec<u8>) -> PyResult<Image> {
    let mut result = Image::from_pixels(size.0, size.1, image.mode, pixels, None)?;
    result.palette = image.palette.clone();
    Ok(result)
}

/// Copy complete rows into a reserved, empty image without first zeroing it.
fn copy_rows<'a>(pixels: &mut Vec<u8>, row_bytes: usize, height: usize, source_row: impl Fn(usize) -> &'a [u8] + Sync) {
    assert!(pixels.is_empty());
    let len = row_bytes.checked_mul(height).expect("validated image dimensions");
    if len == 0 {
        return;
    }
    if len < 4 * 1024 * 1024 {
        for y in 0..height {
            pixels.extend_from_slice(source_row(y));
        }
        return;
    }
    pixels.spare_capacity_mut()[..len]
        .par_chunks_mut(row_bytes * 32)
        .enumerate()
        .for_each(|(band, rows)| {
            for (i, row) in rows.chunks_exact_mut(row_bytes).enumerate() {
                let src = source_row(band * 32 + i);
                assert_eq!(src.len(), row.len());
                // SAFETY: source and destination are separate allocations.
                // Each worker initializes only its disjoint destination rows.
                unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), row.as_mut_ptr().cast::<u8>(), row.len()) };
            }
        });
    // SAFETY: every byte in the reserved image was initialized above. If a
    // worker panics, the vector remains empty and exposes no unwritten bytes.
    unsafe { pixels.set_len(len) };
}

#[pyfunction]
fn ops_split(py: Python<'_>, image: &Image) -> PyResult<Vec<Image>> {
    let source = image.pixel_data()?;
    let channels = image.mode.channels();
    if channels == 1 {
        return py.detach(|| Image::from_pixels(image.width, image.height, PixelMode::L, source.to_vec(), None).map(|band| vec![band]));
    }
    let mut bands = (0..channels)
        .map(|_| buffer((image.width, image.height), 1))
        .collect::<PyResult<Vec<_>>>()?;
    py.detach(|| {
        let split = |offset: usize, red: &mut [u8], green: &mut [u8], blue: &mut [u8], alpha: Option<&mut [u8]>| {
            let src = &source[offset * channels..(offset + red.len()) * channels];
            if let Some(alpha) = alpha {
                for ((((r, g), b), a), pixel) in red.iter_mut().zip(green).zip(blue).zip(alpha).zip(src.as_chunks::<4>().0) {
                    [*r, *g, *b, *a] = *pixel;
                }
            } else {
                for (((r, g), b), pixel) in red.iter_mut().zip(green).zip(blue).zip(src.as_chunks::<3>().0) {
                    [*r, *g, *b] = *pixel;
                }
            }
        };
        // Deinterleave all bands in one source pass and one Rayon dispatch.
        let [red, green, blue, rest @ ..] = bands.as_mut_slice() else {
            unreachable!()
        };
        if source.len() < 2 * 1024 * 1024 {
            split(0, red, green, blue, rest.first_mut().map(Vec::as_mut_slice));
        } else if let Some(alpha) = rest.first_mut() {
            red.par_chunks_mut(CHUNK_PIXELS)
                .zip(green.par_chunks_mut(CHUNK_PIXELS))
                .zip(blue.par_chunks_mut(CHUNK_PIXELS))
                .zip(alpha.par_chunks_mut(CHUNK_PIXELS))
                .enumerate()
                .for_each(|(i, (((r, g), b), a))| split(i * CHUNK_PIXELS, r, g, b, Some(a)));
        } else {
            red.par_chunks_mut(CHUNK_PIXELS)
                .zip(green.par_chunks_mut(CHUNK_PIXELS))
                .zip(blue.par_chunks_mut(CHUNK_PIXELS))
                .enumerate()
                .for_each(|(i, ((r, g), b))| split(i * CHUNK_PIXELS, r, g, b, None));
        }
    });
    bands
        .into_iter()
        .map(|pixels| Image::from_pixels(image.width, image.height, PixelMode::L, pixels, None))
        .collect()
}

#[pyfunction(signature = (image, mask=None))]
fn ops_entropy(py: Python<'_>, image: &Image, mask: Option<&Image>) -> PyResult<f64> {
    let bins = ops_histogram(py, image, mask)?;
    let total = bins.iter().sum::<u64>() as f64;
    if total == 0.0 {
        return Ok(f64::NAN);
    }
    Ok(-bins
        .into_iter()
        .filter(|&n| n != 0)
        .map(|n| {
            let probability = n as f64 / total;
            probability * probability.log2()
        })
        .sum::<f64>())
}

#[pyfunction]
fn ops_lut(py: Python<'_>, image: &Image, lut: Vec<u8>) -> PyResult<Image> {
    let source = image.pixel_data()?;
    let channels = image.mode.channels();
    if lut.len() != 256 && lut.len() != 256 * channels {
        return Err(PyValueError::new_err("wrong number of lut entries"));
    }
    let mut pixels = buffer((image.width, image.height), channels)?;
    py.detach(|| match image.mode {
        PixelMode::L => apply_lut::<1>(source, &mut pixels, &lut),
        PixelMode::Rgb => apply_lut::<3>(source, &mut pixels, &lut),
        PixelMode::Rgba => apply_lut::<4>(source, &mut pixels, &lut),
    });
    output(image, (image.width, image.height), pixels)
}

fn apply_lut<const C: usize>(source: &[u8], output: &mut [u8], lut: &[u8]) {
    chunks_mut(output, CHUNK_PIXELS * C, |i, dst| {
        let start = i * CHUNK_PIXELS * C;
        crate::ops_simd::lut::<C>(&source[start..start + dst.len()], dst, lut);
    });
}

#[pyfunction]
fn ops_colorize(py: Python<'_>, image: &Image, tables: Vec<u8>) -> PyResult<Image> {
    if image.mode != PixelMode::L || tables.len() != 768 {
        return Err(PyValueError::new_err("colorize requires L pixels and three 256-entry tables"));
    }
    let source = image.pixel_data()?;
    let mut output = buffer((image.width, image.height), 3)?;
    py.detach(|| {
        chunks_mut(&mut output, CHUNK_PIXELS * 3, |i, dst| {
            let start = i * CHUNK_PIXELS;
            crate::ops_simd::colorize(&source[start..start + dst.len() / 3], dst, &tables);
        })
    });
    Image::from_pixels(image.width, image.height, PixelMode::Rgb, output, None)
}

#[pyfunction(signature = (image, mask=None))]
fn ops_histogram(py: Python<'_>, image: &Image, mask: Option<&Image>) -> PyResult<Vec<u64>> {
    let source = image.pixel_data()?;
    let mask = if let Some(mask) = mask {
        if mask.mode != PixelMode::L {
            return Err(PyValueError::new_err("bad transparency mask"));
        }
        if (mask.width, mask.height) != (image.width, image.height) {
            return Err(PyValueError::new_err("images do not match"));
        }
        Some(mask.pixel_data()?)
    } else {
        None
    };
    Ok(py.detach(|| match image.mode {
        PixelMode::L => histogram::<1>(source, mask),
        PixelMode::Rgb => histogram::<3>(source, mask),
        PixelMode::Rgba => histogram::<4>(source, mask),
    }))
}

fn histogram<const C: usize>(source: &[u8], mask: Option<&[u8]>) -> Vec<u64> {
    let count = |start: usize, bytes: &[u8]| {
        let mut bins = vec![0_u64; C * 256];
        for (i, pixel) in bytes.as_chunks::<C>().0.iter().enumerate() {
            if mask.is_none_or(|m| m[start + i] != 0) {
                for c in 0..C {
                    bins[c * 256 + usize::from(pixel[c])] += 1;
                }
            }
        }
        bins
    };
    if source.len() < MIN_PARALLEL_BYTES {
        return count(0, source);
    }
    source
        .par_chunks(CHUNK_PIXELS * C)
        .enumerate()
        .map(|(i, bytes)| count(i * CHUNK_PIXELS, bytes))
        .reduce(
            || vec![0; C * 256],
            |mut total, bins| {
                for (a, b) in total.iter_mut().zip(bins) {
                    *a += b;
                }
                total
            },
        )
}

/// Place the source on a filled canvas. Negative offsets clip the source.
#[pyfunction]
fn ops_canvas(py: Python<'_>, image: &Image, size: (u32, u32), offset: (i64, i64), fill: Vec<u8>) -> PyResult<Image> {
    let source = image.pixel_data()?;
    let channels = image.mode.channels();
    if fill.len() != channels {
        return Err(PyValueError::new_err("invalid fill color"));
    }
    if offset.0 <= 0
        && offset.1 <= 0
        && offset.0.saturating_add(i64::from(image.width)) >= i64::from(size.0)
        && offset.1.saturating_add(i64::from(image.height)) >= i64::from(size.1)
    {
        // An interior crop writes every byte. Append complete source rows so
        // allocation does not first zero memory that is immediately replaced.
        let mut pixels = reserved_buffer(size, channels)?;
        py.detach(|| {
            let row_bytes = size.0 as usize * channels;
            copy_rows(&mut pixels, row_bytes, size.1 as usize, |y| {
                let start = (((y as i64 - offset.1) as usize * image.width as usize) + (-offset.0) as usize) * channels;
                &source[start..start + row_bytes]
            });
        });
        return output(image, size, pixels);
    }
    let mut pixels = reserved_buffer(size, channels)?;
    py.detach(|| {
        let left = offset.0.max(0).min(i64::from(size.0));
        let right = (offset.0.saturating_add(i64::from(image.width))).clamp(0, i64::from(size.0));
        let top = offset.1.max(0).min(i64::from(size.1));
        let bottom = (offset.1.saturating_add(i64::from(image.height))).clamp(0, i64::from(size.1));
        let row_bytes = size.0 as usize * channels;
        let border = fill.repeat(size.0 as usize);
        // Append each byte once; a padded image is mostly source data and
        // does not need a zeroed full-size allocation or worker scheduling.
        for y in 0..i64::from(size.1) {
            if y >= top && y < bottom && right > left {
                let lo = left as usize * channels;
                let hi = right as usize * channels;
                pixels.extend_from_slice(&border[..lo]);
                let src = ((y - offset.1) as usize * image.width as usize + (left - offset.0) as usize) * channels;
                pixels.extend_from_slice(&source[src..src + hi - lo]);
                pixels.extend_from_slice(&border[hi..]);
            } else {
                pixels.extend_from_slice(&border[..row_bytes]);
            }
        }
    });
    output(image, size, pixels)
}

/// EXIF orientation numbers also describe all eight rectangular symmetries.
#[pyfunction]
fn ops_transpose(py: Python<'_>, image: &Image, orientation: u8) -> PyResult<Image> {
    let source = image.pixel_data()?;
    if !(1..=8).contains(&orientation) {
        return Err(PyValueError::new_err("invalid orientation"));
    }
    if orientation == 1 {
        return py.detach(|| output(image, (image.width, image.height), source.to_vec()));
    }
    let size = if orientation >= 5 {
        (image.height, image.width)
    } else {
        (image.width, image.height)
    };
    let c = image.mode.channels();
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
    let mut pixels = buffer(size, c)?;
    py.detach(|| match image.mode {
        PixelMode::L => transpose::<1>(source, &mut pixels, image.width as usize, image.height as usize, orientation),
        PixelMode::Rgb => transpose::<3>(source, &mut pixels, image.width as usize, image.height as usize, orientation),
        PixelMode::Rgba => transpose::<4>(source, &mut pixels, image.width as usize, image.height as usize, orientation),
    });
    output(image, size, pixels)
}

fn transpose<const C: usize>(source: &[u8], output: &mut [u8], w: usize, h: usize, orientation: u8) {
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
                    crate::ops_simd::reverse_rgb(src.as_flattened(), row);
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
                    let sx = if orientation == 7 || orientation == 8 {
                        w - band * 32 - count
                    } else {
                        band * 32
                    };
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

fn kernel(x: f64, method: u8) -> f64 {
    if method == 4 {
        return if x > -0.5 && x <= 0.5 { 1.0 } else { 0.0 };
    }
    let x = x.abs();
    match method {
        2 => (1.0 - x).max(0.0),
        5 => {
            if x == 0.0 {
                1.0
            } else if x >= 1.0 {
                0.0
            } else {
                let p = x * std::f64::consts::PI;
                p.sin() / p * (f64::from(0.54_f32) + f64::from(0.46_f32) * p.cos())
            }
        }
        3 => {
            if x < 1.0 {
                ((1.5 * x - 2.5) * x) * x + 1.0
            } else if x < 2.0 {
                (((x - 5.0) * x + 8.0) * x - 4.0) * -0.5
            } else {
                0.0
            }
        }
        1 => {
            if x == 0.0 {
                1.0
            } else if x >= 3.0 {
                0.0
            } else {
                let p = x * std::f64::consts::PI;
                (p.sin() / p) * ((p / 3.0).sin() / (p / 3.0))
            }
        }
        _ => 0.0,
    }
}

// Pillow's 8-bit resampler rounds signed coefficients to 22-bit fixed point,
// and rounds/clamps pixels after each of its two separable passes.
fn weights(input: u32, output: u32, start: f64, end: f64, method: u8) -> Vec<(usize, Vec<i32>)> {
    let scale = f64::from(end as f32 - start as f32) / f64::from(output);
    let filter_scale = scale.max(1.0);
    let support = match method {
        1 => 3.0,
        3 => 2.0,
        4 => 0.5,
        _ => 1.0,
    } * filter_scale;
    (0..output)
        .map(|i| {
            let center = start + (f64::from(i) + 0.5) * scale;
            let left = ((center - support + 0.5) as i64).clamp(0, i64::from(input)) as usize;
            let right = ((center + support + 0.5) as i64).clamp(0, i64::from(input)) as usize;
            let raw: Vec<f64> = (left..right)
                .map(|x| kernel((x as f64 - center + 0.5) * (1.0 / filter_scale), method))
                .collect();
            let sum: f64 = raw.iter().sum();
            (
                left,
                raw.into_iter()
                    .map(|v| if sum == 0.0 { 0 } else { (v / sum * 4194304.0).round() as i32 })
                    .collect(),
            )
        })
        .collect()
}

fn premultiply(pixels: &mut [u8]) {
    chunks_mut(pixels, CHUNK_PIXELS * 4, |_, chunk| {
        for p in chunk.as_chunks_mut::<4>().0 {
            for c in 0..3 {
                let v = u32::from(p[c]) * u32::from(p[3]) + 128;
                p[c] = (((v >> 8) + v) >> 8) as u8;
            }
        }
    });
}

fn unpremultiply(pixels: &mut [u8]) {
    chunks_mut(pixels, CHUNK_PIXELS * 4, |_, chunk| {
        for p in chunk.as_chunks_mut::<4>().0 {
            if p[3] != 0 && p[3] != 255 {
                for c in 0..3 {
                    p[c] = (u32::from(p[c]) * 255 / u32::from(p[3])).min(255) as u8;
                }
            }
        }
    });
}

/// Integer box reduction used by resize's reducing_gap optimization.
#[pyfunction]
fn ops_reduce(py: Python<'_>, image: &Image, factor: (u32, u32), bounds: (u32, u32, u32, u32)) -> PyResult<Image> {
    let source = image.pixel_data()?;
    let (fx, fy) = factor;
    let (left, top, right, bottom) = bounds;
    if fx == 0 || fy == 0 || left > right || top > bottom || right > image.width || bottom > image.height {
        return Err(PyValueError::new_err("invalid reduction bounds or factor"));
    }
    let size = ((right - left).div_ceil(fx), (bottom - top).div_ceil(fy));
    let c = image.mode.channels();
    let mut pixels = buffer(size, c)?;
    py.detach(|| match image.mode {
        PixelMode::L => reduce_pixels::<1>(source, &mut pixels, image.width, size, factor, bounds),
        PixelMode::Rgb => reduce_pixels::<3>(source, &mut pixels, image.width, size, factor, bounds),
        PixelMode::Rgba => {
            let mut source = source.to_vec();
            premultiply(&mut source);
            reduce_pixels::<4>(&source, &mut pixels, image.width, size, factor, bounds);
            unpremultiply(&mut pixels);
        }
    });
    output(image, size, pixels)
}

fn reduce_pixels<const C: usize>(source: &[u8], output: &mut [u8], width: u32, size: (u32, u32), factor: (u32, u32), bounds: (u32, u32, u32, u32)) {
    if output.is_empty() {
        return;
    }
    let (fx, fy) = factor;
    let (left, top, right, bottom) = bounds;
    let source = source.as_chunks::<C>().0;
    let row_bytes = size.0 as usize * C;
    // Reduction reads many more bytes than it writes. Schedule by source work.
    let threshold = if cfg!(target_arch = "aarch64") && C == 1 && fx == 3 && fy == 3 {
        128 * 1024
    } else {
        MIN_PARALLEL_BYTES / (fx as usize * fy as usize).max(1)
    };
    chunks_mut_above(output, row_bytes * 8, threshold, |band, rows| {
        for (i, row) in rows.chunks_exact_mut(row_bytes).enumerate() {
            let y0 = top + (band * 8 + i) as u32 * fy;
            let y1 = y0.saturating_add(fy).min(bottom);
            if fx == 3 && fy == 3 && y1 - y0 == 3 {
                let full = ((right - left) / 3) as usize;
                let start = y0 as usize * width as usize + left as usize;
                let stride = width as usize;
                let upper = source[start..start + full * 3].as_chunks::<3>().0;
                let middle = source[start + stride..start + stride + full * 3].as_chunks::<3>().0;
                let lower = source[start + stride * 2..start + stride * 2 + full * 3].as_chunks::<3>().0;
                let done = if C == 1 {
                    crate::ops_simd::reduce_three_l(
                        upper.as_flattened().as_flattened(),
                        middle.as_flattened().as_flattened(),
                        lower.as_flattened().as_flattened(),
                        &mut row[..full],
                    )
                } else {
                    0
                };
                // Fixed contiguous groups expose the nine-sample sum to SIMD
                // without repeating row offsets and bounds checks per pixel.
                for (((dst, upper), middle), lower) in row.as_chunks_mut::<C>().0[..full].iter_mut().zip(upper).zip(middle).zip(lower).skip(done) {
                    for channel in 0..C {
                        let mut sum = 0_u32;
                        for samples in [upper, middle, lower] {
                            sum += u32::from(samples[0][channel]) + u32::from(samples[1][channel]) + u32::from(samples[2][channel]);
                        }
                        dst[channel] = (((sum + 4) * 1_864_135) >> 24) as u8;
                    }
                }
                if (right - left).is_multiple_of(3) {
                    continue;
                }
            }
            if fx == 2 && fy == 2 && y1 - y0 == 2 && (right - left).is_multiple_of(2) {
                let start = y0 as usize * width as usize + left as usize;
                let end = start + (right - left) as usize;
                let upper = source[start..end].as_chunks::<2>().0;
                let lower = source[start + width as usize..end + width as usize].as_chunks::<2>().0;
                for ((dst, upper), lower) in row.as_chunks_mut::<C>().0.iter_mut().zip(upper).zip(lower) {
                    for channel in 0..C {
                        let sum =
                            u16::from(upper[0][channel]) + u16::from(upper[1][channel]) + u16::from(lower[0][channel]) + u16::from(lower[1][channel]);
                        dst[channel] = ((sum + 2) >> 2) as u8;
                    }
                }
                continue;
            }
            for (x, dst) in row.as_chunks_mut::<C>().0.iter_mut().enumerate() {
                if fx == 3 && fy == 3 && y1 - y0 == 3 && x < ((right - left) / 3) as usize {
                    continue;
                }
                let x0 = left + x as u32 * fx;
                let x1 = x0.saturating_add(fx).min(right);
                let count = u64::from(x1 - x0) * u64::from(y1 - y0);
                // Match Pillow's 24-bit reciprocal, including its rounding.
                let multiplier = ((1_u64 << 24) as f32 / count as f32) as u64;
                let mut sums = [0_u64; C];
                for sy in y0..y1 {
                    let start = sy as usize * width as usize;
                    for pixel in &source[start + x0 as usize..start + x1 as usize] {
                        for channel in 0..C {
                            sums[channel] += u64::from(pixel[channel]);
                        }
                    }
                }
                for channel in 0..C {
                    dst[channel] = (((sums[channel] + count / 2) * multiplier) >> 24) as u8;
                }
            }
        }
    });
}

#[pyfunction]
fn ops_resize(py: Python<'_>, image: &Image, size: (u32, u32), method: u8, bounds: BoxF) -> PyResult<Image> {
    let source = image.pixel_data()?;
    if method > 5 {
        return Err(PyValueError::new_err("unknown resampling filter"));
    }
    if size.0 == 0 || size.1 == 0 {
        return Err(PyValueError::new_err("height and width must be > 0"));
    }
    // The public Pillow resize API passes its box through C floats.
    let b = [bounds.0, bounds.1, bounds.2, bounds.3].map(|x| f64::from(x as f32));
    if b.iter().any(|x| !x.is_finite())
        || b[0] < 0.0
        || b[1] < 0.0
        || b[2] > f64::from(image.width)
        || b[3] > f64::from(image.height)
        || b[2] < b[0]
        || b[3] < b[1]
    {
        return Err(PyValueError::new_err("invalid resize box"));
    }
    // Pillow resamples very tall images vertically first to limit temporary
    // storage. Pass order affects 8-bit rounding, so retain it as well.
    if u64::from(image.height) > u64::from(image.width) * 100 && size.1 < image.height && image.width > 0 {
        let transposed = ops_transpose(py, image, 5)?;
        let resized = ops_resize(py, &transposed, (size.1, size.0), method, (bounds.1, bounds.0, bounds.3, bounds.2))?;
        return ops_transpose(py, &resized, 5);
    }
    let c = image.mode.channels();
    let mut result = buffer(size, c)?;
    if image.width == 0 || image.height == 0 {
        return output(image, size, result);
    }
    py.detach(|| -> PyResult<()> {
        if method == 0 {
            let dx = f64::from(b[2] as f32 - b[0] as f32) / f64::from(size.0);
            let dy = f64::from(b[3] as f32 - b[1] as f32) / f64::from(size.1);
            let positions = |start: f64, step: f64, count: u32, limit: u32| {
                let mut pos = start + 0.5 * step;
                (0..count)
                    .map(|_| {
                        let value = (pos as usize).min(limit as usize - 1);
                        pos += step;
                        value
                    })
                    .collect::<Vec<_>>()
            };
            let xs = positions(b[0], dx, size.0, image.width);
            let ys = positions(b[1], dy, size.1, image.height);
            match image.mode {
                PixelMode::L => resize_nearest::<1>(source, &mut result, image.width, &xs, &ys),
                PixelMode::Rgb => resize_nearest::<3>(source, &mut result, image.width, &xs, &ys),
                PixelMode::Rgba => resize_nearest::<4>(source, &mut result, image.width, &xs, &ys),
            }
            return Ok(());
        }
        let mut source = Cow::Borrowed(source);
        if c == 4 {
            premultiply(source.to_mut());
        }
        match image.mode {
            PixelMode::L => resample::<1>(&source, &mut result, image, size, b, method)?,
            PixelMode::Rgb => resample::<3>(&source, &mut result, image, size, b, method)?,
            PixelMode::Rgba => resample::<4>(&source, &mut result, image, size, b, method)?,
        }
        if c == 4 {
            unpremultiply(&mut result);
        }
        Ok(())
    })?;
    output(image, size, result)
}

// Fixed-size pixels let LLVM inline the gathers instead of calling memcpy
// for every output pixel. Repeated source rows only need one gather per band.
fn resize_nearest<const C: usize>(source: &[u8], output: &mut [u8], width: u32, xs: &[usize], ys: &[usize]) {
    let row_bytes = xs.len() * C;
    let half_width = xs.len() * 2 == width as usize && xs.iter().enumerate().all(|(x, &sx)| sx == x * 2 + 1);
    let source = source.as_chunks::<C>().0;
    chunks_mut_above(output, row_bytes * 16, 1024 * 1024, |band, rows| {
        for i in 0..rows.len() / row_bytes {
            let sy = ys[band * 16 + i];
            if i > 0 && sy == ys[band * 16 + i - 1] {
                rows.copy_within((i - 1) * row_bytes..i * row_bytes, i * row_bytes);
                continue;
            }
            let src = &source[sy * width as usize..(sy + 1) * width as usize];
            let row = &mut rows[i * row_bytes..(i + 1) * row_bytes];
            let dst = row.as_chunks_mut::<C>().0;
            if half_width {
                // Contiguous pairs expose the common 2:1 gather to SIMD.
                crate::ops_simd::nearest_half::<C>(src.as_flattened(), dst.as_flattened_mut());
            } else {
                for (dst, &sx) in dst.iter_mut().zip(xs) {
                    *dst = src[sx];
                }
            }
        }
    });
}

fn resample<const C: usize>(source: &[u8], output: &mut [u8], image: &Image, size: (u32, u32), b: [f64; 4], method: u8) -> PyResult<()> {
    let horizontal = weights(image.width, size.0, b[0], b[2], method);
    let vertical = weights(image.height, size.1, b[1], b[3], method);
    let first_row = vertical.first().unwrap().0;
    let (last_start, last_weights) = vertical.last().unwrap();
    let last_row = last_start + last_weights.len();
    let row_bytes = size.0 as usize * C;
    let input_stride = image.width as usize * C;
    let horizontal_identity = size.0 == image.width && b[0] == 0.0 && b[2] == f64::from(image.width);
    let temp = if horizontal_identity {
        Cow::Borrowed(&source[first_row * input_stride..last_row * input_stride])
    } else {
        let mut temp = buffer((size.0, (last_row - first_row) as u32), C)?;
        let narrow = horizontal
            .iter()
            .all(|(_, coefficients)| coefficients.iter().map(|&w| i64::from(w).abs()).sum::<i64>() * 255 + (1 << 21) <= i64::from(i32::MAX));
        chunks_mut(&mut temp, row_bytes * 16, |band, rows| {
            for (i, dst) in rows.chunks_exact_mut(row_bytes).enumerate() {
                let y = first_row + band * 16 + i;
                let src = source[y * input_stride..(y + 1) * input_stride].as_chunks::<C>().0;
                for ((start, coefficients), dst) in horizontal.iter().zip(dst.as_chunks_mut::<C>().0) {
                    if narrow {
                        let mut sums = [1_i32 << 21; C];
                        for (pixel, &weight) in src[*start..*start + coefficients.len()].iter().zip(coefficients) {
                            for c in 0..C {
                                sums[c] += i32::from(pixel[c]) * weight;
                            }
                        }
                        for c in 0..C {
                            dst[c] = (sums[c] >> 22).clamp(0, 255) as u8;
                        }
                    } else {
                        for c in 0..C {
                            let sum = coefficients
                                .iter()
                                .enumerate()
                                .fold(1_i64 << 21, |sum, (i, &w)| sum + i64::from(src[start + i][c]) * i64::from(w));
                            dst[c] = (sum >> 22).clamp(0, 255) as u8;
                        }
                    }
                }
            }
        });
        Cow::Owned(temp)
    };
    chunks_mut(output, row_bytes * 16, |band, rows| {
        for (i, dst) in rows.chunks_exact_mut(row_bytes).enumerate() {
            let (start, coefficients) = &vertical[band * 16 + i];
            let src = &temp[(start - first_row) * row_bytes..];
            let vectorized = crate::ops_simd::vertical(src, dst, row_bytes, coefficients);
            for (x, out) in dst.iter_mut().enumerate().skip(vectorized) {
                let sum = coefficients
                    .iter()
                    .enumerate()
                    .fold(1_i64 << 21, |sum, (i, &w)| sum + i64::from(src[i * row_bytes + x]) * i64::from(w));
                *out = (sum >> 22).clamp(0, 255) as u8;
            }
        }
    });
    Ok(())
}

fn cubic(v: [f64; 4], t: f64) -> f64 {
    let a = -v[0] + v[1] - v[2] + v[3];
    let b = 2.0 * v[0] - 2.0 * v[1] + v[2] - v[3];
    let c = -v[0] + v[2];
    ((a * t + b) * t + c) * t + v[1]
}

fn interpolate(a: f64, b: f64, t: f64) -> f64 {
    // C compilers contract this expression on ARM; preserve that rounding.
    if cfg!(target_arch = "aarch64") {
        (b - a).mul_add(t, a)
    } else {
        a + (b - a) * t
    }
}

fn sample(source: &[u8], image: &Image, x: f64, y: f64, method: u8, dst: &mut [u8]) {
    let (w, h, c) = (image.width as usize, image.height as usize, image.mode.channels());
    if !x.is_finite() || !y.is_finite() || x < 0.0 || y < 0.0 || x >= w as f64 || y >= h as f64 {
        dst.fill(0);
        return;
    }
    if method == 0 {
        let start = (y as usize * w + x as usize) * c;
        dst.copy_from_slice(&source[start..start + c]);
        return;
    }
    let (x, y) = (x - 0.5, y - 0.5);
    let (ix, iy) = (x.floor() as i64, y.floor() as i64);
    let (tx, ty) = (x - x.floor(), y - y.floor());
    for (ch, out) in dst.iter_mut().enumerate() {
        let get = |dx: i64, dy: i64| {
            f64::from(source[((iy + dy).clamp(0, h as i64 - 1) as usize * w + (ix + dx).clamp(0, w as i64 - 1) as usize) * c + ch])
        };
        let value = if method == 2 {
            let top = interpolate(get(0, 0), get(1, 0), tx);
            let bottom = interpolate(get(0, 1), get(1, 1), tx);
            interpolate(top, bottom, ty)
        } else {
            cubic([-1, 0, 1, 2].map(|dy| cubic([-1, 0, 1, 2].map(|dx| get(dx, dy)), tx)), ty)
        };
        *out = value.clamp(0.0, 255.0) as u8;
    }
}

fn affine_coordinate(a: f64, b: f64, c: f64, x: f64, y: f64) -> f64 {
    // Match the contraction order used by Pillow's ARM affine mapper.
    if cfg!(target_arch = "aarch64") {
        a.mul_add(x, b * y) + c
    } else {
        a * x + b * y + c
    }
}

fn quad_coordinate(origin: f64, a: f64, b: f64, c: f64, u: f64, v: f64) -> f64 {
    if cfg!(target_arch = "aarch64") {
        (c * u).mul_add(v, b.mul_add(v, a.mul_add(u, origin)))
    } else {
        origin + a * u + b * v + c * u * v
    }
}

fn nearest_columns<const C: usize>(source: Option<&[u8]>, row: &mut [u8], columns: &[Option<usize>], fill: &[u8]) {
    let fill: [u8; C] = fill.try_into().unwrap();
    for (dst, column) in row.as_chunks_mut::<C>().0.iter_mut().zip(columns) {
        *dst = if let (Some(src), Some(offset)) = (source, column) {
            src[*offset..*offset + C].try_into().unwrap()
        } else {
            fill
        };
    }
}

fn nearest_fixed<const C: usize>(source: &[u8], row: &mut [u8], size: (u32, u32), coordinates: (i64, i64), steps: (i64, i64), fill: &[u8]) {
    let source = source.as_chunks::<C>().0;
    let fill: [u8; C] = fill.try_into().unwrap();
    let (mut x, mut y) = coordinates;
    for dst in row.as_chunks_mut::<C>().0 {
        let (sx, sy) = (x >> 16, y >> 16);
        *dst = if sx >= 0 && sy >= 0 && sx < i64::from(size.0) && sy < i64::from(size.1) {
            source[sy as usize * size.0 as usize + sx as usize]
        } else {
            fill
        };
        x += steps.0;
        y += steps.1;
    }
}

/// Reverse affine mapping, sharing the mesh interpolation and alpha kernels.
#[pyfunction]
fn ops_affine(py: Python<'_>, image: &Image, size: (u32, u32), matrix: [f64; 6], method: u8, fill: Vec<u8>) -> PyResult<Image> {
    let source = image.pixel_data()?;
    let channels = image.mode.channels();
    if ![0, 2, 3].contains(&method) || !matrix.iter().all(|v| v.is_finite()) || fill.len() != channels {
        return Err(PyValueError::new_err("invalid affine transform"));
    }
    let [a, b, c, d, e, f] = matrix;
    // Pillow uses 16.16 coordinates for nearest affine rotations when all
    // corners fit, but leaves axis-aligned transforms in floating point.
    let fixed = method == 0
        && (b != 0.0 || d != 0.0)
        && [(0.0, 0.0), (size.0 as f64, 0.0), (0.0, size.1 as f64), (size.0 as f64, size.1 as f64)]
            .iter()
            .all(|&(x, y)| (a * x + b * y + c).abs() < 32768.0 && (d * x + e * y + f).abs() < 32768.0);
    let fix = |v: f64| (v * 65536.0 + 0.5).floor() as i64;
    let coefficients = [fix(a), fix(b), fix(c + a * 0.5 + b * 0.5), fix(d), fix(e), fix(f + d * 0.5 + e * 0.5)];
    let mut pixels = buffer(size, channels)?;
    if pixels.is_empty() {
        return output(image, size, pixels);
    }
    py.detach(|| {
        let mut source = Cow::Borrowed(source);
        if channels == 4 && method != 0 {
            premultiply(source.to_mut());
        }
        // Nearest's floating path advances coordinates incrementally. Direct
        // multiplication can pick a different pixel at fractional boundaries.
        let row_origins: Vec<_> = if method == 0 && !fixed {
            let (mut x, mut y) = (c + b * 0.5 + a * 0.5, f + e * 0.5 + d * 0.5);
            (0..size.1)
                .map(|_| {
                    let origin = (x, y);
                    x += b;
                    y += e;
                    origin
                })
                .collect()
        } else {
            Vec::new()
        };
        let row_bytes = size.0 as usize * channels;
        // Axis-aligned nearest transforms reuse exactly the same horizontal
        // coordinates on every row. Preserve incremental floating rounding.
        let columns: Option<Vec<_>> = (method == 0 && b == 0.0 && d == 0.0).then(|| {
            let mut x = c + a * 0.5;
            (0..size.0)
                .map(|_| {
                    let offset = (x >= 0.0 && x < image.width as f64).then(|| x as usize * channels);
                    x += a;
                    offset
                })
                .collect()
        });
        if channels != 1
            && let Some(columns) = &columns
        {
            // Interior extents need neither per-pixel fill checks nor
            // optional byte offsets; reuse the typed resize gather.
            if columns.iter().all(Option::is_some) && row_origins.iter().all(|&(_, y)| y >= 0.0 && y < image.height as f64) {
                let xs: Vec<_> = columns.iter().map(|x| x.unwrap() / channels).collect();
                let ys: Vec<_> = row_origins.iter().map(|&(_, y)| y as usize).collect();
                if channels == 3 {
                    // The grayscale table kernel also gathers RGB bytes when
                    // each pixel coordinate is expanded into its three bands.
                    let bytes: Vec<_> = columns.iter().flat_map(|x| (0..3).map(move |c| Some(x.unwrap() + c))).collect();
                    let map = crate::ops_simd::NearestL::new(&bytes, image.width as usize * 3).unwrap();
                    if map.is_vectorized() {
                        chunks_mut_above(&mut pixels, row_bytes * 16, 512 * 1024, |band, rows| {
                            for (i, row) in rows.chunks_exact_mut(row_bytes).enumerate() {
                                let start = ys[band * 16 + i] * image.width as usize * 3;
                                map.sample(&source[start..start + image.width as usize * 3], row);
                            }
                        });
                        return;
                    }
                }
                match image.mode {
                    PixelMode::Rgb => resize_nearest::<3>(&source, &mut pixels, image.width, &xs, &ys),
                    PixelMode::Rgba => resize_nearest::<4>(&source, &mut pixels, image.width, &xs, &ys),
                    PixelMode::L => unreachable!(),
                }
                return;
            }
        }
        let nearest_l = columns
            .as_ref()
            .filter(|_| channels == 1)
            .and_then(|columns| crate::ops_simd::NearestL::new(columns, image.width as usize));
        // Table gathers are memory work: small outputs cost less than a Rayon
        // dispatch. Other transforms still benefit from finer scheduling.
        let threshold = if nearest_l.as_ref().is_some_and(|map| map.is_vectorized()) {
            MIN_PARALLEL_BYTES
        } else {
            32 * 1024
        };
        chunks_mut_above(&mut pixels, row_bytes * 16, threshold, |band, rows| {
            for (i, row) in rows.chunks_exact_mut(row_bytes).enumerate() {
                let y = (band * 16 + i) as f64;
                let mut nearest = row_origins.get(band * 16 + i).copied().unwrap_or_default();
                if let Some(columns) = &columns {
                    let sy = nearest.1;
                    let source_row = if sy >= 0.0 && sy < image.height as f64 {
                        let start = sy as usize * image.width as usize * channels;
                        Some(&source[start..start + image.width as usize * channels])
                    } else {
                        None
                    };
                    if let (Some(map), Some(src)) = (&nearest_l, source_row) {
                        map.sample(src, row);
                        continue;
                    }
                    match image.mode {
                        PixelMode::L => nearest_columns::<1>(source_row, row, columns, &fill),
                        PixelMode::Rgb => nearest_columns::<3>(source_row, row, columns, &fill),
                        PixelMode::Rgba => nearest_columns::<4>(source_row, row, columns, &fill),
                    }
                    continue;
                }
                if fixed {
                    let [a, b, c, d, e, f] = coefficients;
                    let origin = (b * y as i64 + c, e * y as i64 + f);
                    let steps = (a, d);
                    let size = (image.width, image.height);
                    match image.mode {
                        PixelMode::L => nearest_fixed::<1>(&source, row, size, origin, steps, &fill),
                        PixelMode::Rgb => nearest_fixed::<3>(&source, row, size, origin, steps, &fill),
                        PixelMode::Rgba => nearest_fixed::<4>(&source, row, size, origin, steps, &fill),
                    }
                    continue;
                }
                for (x, dst) in row.chunks_exact_mut(channels).enumerate() {
                    let (sx, sy) = if method == 0 {
                        let coordinate = nearest;
                        nearest.0 += a;
                        nearest.1 += d;
                        coordinate
                    } else {
                        (
                            affine_coordinate(a, b, c, x as f64 + 0.5, y + 0.5),
                            affine_coordinate(d, e, f, x as f64 + 0.5, y + 0.5),
                        )
                    };
                    if sx >= 0.0 && sy >= 0.0 && sx < image.width as f64 && sy < image.height as f64 {
                        sample(&source, image, sx, sy, method, dst);
                    } else {
                        dst.copy_from_slice(&fill);
                    }
                }
            }
        });
        // Fill values also belong to Pillow's intermediate premultiplied mode.
        if channels == 4 && method != 0 {
            unpremultiply(&mut pixels);
        }
    });
    output(image, size, pixels)
}

#[pyfunction]
fn ops_mesh(py: Python<'_>, image: &Image, mesh: Mesh, method: u8) -> PyResult<Image> {
    ops_warp(py, image, (image.width, image.height), mesh, (method, false), None)
}

#[pyfunction]
fn ops_warp(py: Python<'_>, image: &Image, size: (u32, u32), mesh: Mesh, filters: (u8, bool), fill: Option<Vec<u8>>) -> PyResult<Image> {
    let (method, perspective) = filters;
    let source = image.pixel_data()?;
    if ![0, 2, 3].contains(&method) {
        return Err(PyValueError::new_err("mesh transforms support NEAREST, BILINEAR and BICUBIC"));
    }
    let c = image.mode.channels();
    if fill.as_ref().is_some_and(|v| v.len() != c) || mesh.iter().any(|(_, q)| q.iter().any(|v| !v.is_finite())) {
        return Err(PyValueError::new_err("invalid warp data"));
    }
    let mut result = buffer(size, c)?;
    if result.is_empty() {
        return output(image, size, result);
    }
    py.detach(|| {
        if let Some(fill) = &fill {
            for pixel in result.chunks_exact_mut(c) {
                pixel.copy_from_slice(fill);
            }
        }
        let mut source = Cow::Borrowed(source);
        if c == 4 && method != 0 {
            premultiply(source.to_mut());
        }
        for ((left, top, right, bottom), q) in mesh {
            if right <= left || bottom <= top {
                continue;
            }
            let w = right as f64 - left as f64;
            let h = bottom as f64 - top as f64;
            let ax = (q[6] - q[0]) * (1.0 / w);
            let bx = (q[2] - q[0]) * (1.0 / h);
            let cx = (q[4] - q[2] - q[6] + q[0]) * (1.0 / w) * (1.0 / h);
            let ay = (q[7] - q[1]) * (1.0 / w);
            let by = (q[3] - q[1]) * (1.0 / h);
            let cy = (q[5] - q[3] - q[7] + q[1]) * (1.0 / w) * (1.0 / h);
            let first = top.clamp(0, i64::from(size.1)) as usize;
            let last = bottom.clamp(0, i64::from(size.1)) as usize;
            let row_bytes = size.0 as usize * c;
            // Mesh entries retain their order (later boxes overwrite earlier
            // ones); only disjoint rows within an entry execute concurrently.
            chunks_mut_above(
                &mut result[first * row_bytes..last * row_bytes],
                row_bytes * 16,
                32 * 1024,
                |band, rows| {
                    for (i, row) in rows.chunks_exact_mut(row_bytes).enumerate() {
                        let y = (first + band * 16 + i) as i64;
                        for x in left.max(0)..right.min(i64::from(size.0)) {
                            let u = (x - left.max(0)) as f64 + 0.5;
                            let v = (y - top.max(0)) as f64 + 0.5;
                            let (sx, sy) = if perspective {
                                let divisor = affine_coordinate(q[6], q[7], 1.0, u, v);
                                (
                                    affine_coordinate(q[0], q[1], q[2], u, v) / divisor,
                                    affine_coordinate(q[3], q[4], q[5], u, v) / divisor,
                                )
                            } else {
                                (quad_coordinate(q[0], ax, bx, cx, u, v), quad_coordinate(q[1], ay, by, cy, u, v))
                            };
                            let dst = x as usize * c;
                            if fill.is_none() || (sx >= 0.0 && sy >= 0.0 && sx < image.width as f64 && sy < image.height as f64) {
                                sample(&source, image, sx, sy, method, &mut row[dst..dst + c]);
                            }
                        }
                    }
                },
            );
        }
        if c == 4 && method != 0 {
            unpremultiply(&mut result);
        }
    });
    output(image, size, result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alpha_rounding_and_zero_alpha() {
        let mut pixels = [255, 127, 0, 128, 255, 127, 64, 0];
        premultiply(&mut pixels);
        assert_eq!(pixels, [128, 64, 0, 128, 0, 0, 0, 0]);
        unpremultiply(&mut pixels);
        assert_eq!(pixels, [255, 127, 0, 128, 0, 0, 0, 0]);
    }

    #[test]
    fn box_filter_has_half_open_support() {
        assert_eq!(kernel(-0.5, 4), 0.0);
        assert_eq!(kernel(0.5, 4), 1.0);
    }

    #[test]
    fn transpose_non_square_and_clip_canvas() {
        Python::initialize();
        Python::attach(|py| {
            let image = Image::from_pixels(3, 2, PixelMode::L, vec![1, 2, 3, 4, 5, 6], None).unwrap();
            let rotated = ops_transpose(py, &image, 6).unwrap();
            assert_eq!((rotated.width, rotated.height), (2, 3));
            assert_eq!(rotated.pixel_data().unwrap(), [4, 1, 5, 2, 6, 3]);
            let clipped = ops_canvas(py, &image, (3, 3), (-1, 1), vec![9]).unwrap();
            assert_eq!(clipped.pixel_data().unwrap(), [9, 9, 9, 2, 3, 9, 5, 6, 9]);
        });
    }
}
