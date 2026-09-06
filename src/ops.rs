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
    module.add_function(wrap_pyfunction!(ops_canvas, module)?)?;
    module.add_function(wrap_pyfunction!(ops_transpose, module)?)?;
    module.add_function(wrap_pyfunction!(ops_resize, module)?)?;
    module.add_function(wrap_pyfunction!(ops_mesh, module)?)?;
    Ok(())
}

fn buffer(size: (u32, u32), channels: usize) -> PyResult<Vec<u8>> {
    let len = (size.0 as usize)
        .checked_mul(size.1 as usize)
        .and_then(|n| n.checked_mul(channels))
        .ok_or_else(|| PyValueError::new_err("image dimensions are too large"))?;
    let mut result = Vec::new();
    result
        .try_reserve_exact(len)
        .map_err(|_| PyMemoryError::new_err("cannot allocate image"))?;
    result.resize(len, 0);
    Ok(result)
}

fn output(image: &Image, size: (u32, u32), pixels: Vec<u8>) -> PyResult<Image> {
    Image::from_pixels(size.0, size.1, image.mode, pixels, None)
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
        return Err(PyValueError::new_err(
            "colorize requires L pixels and three 256-entry tables",
        ));
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
fn ops_canvas(
    py: Python<'_>,
    image: &Image,
    size: (u32, u32),
    offset: (i64, i64),
    fill: Vec<u8>,
) -> PyResult<Image> {
    let source = image.pixel_data()?;
    let channels = image.mode.channels();
    if fill.len() != channels {
        return Err(PyValueError::new_err("invalid fill color"));
    }
    let mut pixels = buffer(size, channels)?;
    py.detach(|| {
        let left = offset.0.max(0).min(i64::from(size.0));
        let right = (offset.0.saturating_add(i64::from(image.width))).clamp(0, i64::from(size.0));
        let top = offset.1.max(0).min(i64::from(size.1));
        let bottom = (offset.1.saturating_add(i64::from(image.height))).clamp(0, i64::from(size.1));
        let row_bytes = size.0 as usize * channels;
        let fill_row = |row: &mut [u8]| {
            if fill.iter().all(|&v| v == 0) {
                return;
            } // buffer is already zeroed
            match channels {
                1 => row.fill(fill[0]),
                3 => row
                    .as_chunks_mut::<3>()
                    .0
                    .iter_mut()
                    .for_each(|p| p.copy_from_slice(&fill)),
                4 => row
                    .as_chunks_mut::<4>()
                    .0
                    .iter_mut()
                    .for_each(|p| p.copy_from_slice(&fill)),
                _ => unreachable!(),
            }
        };
        chunks_mut_above(
            &mut pixels,
            row_bytes * 32,
            2 * 1024 * 1024,
            |band, rows| {
                for (i, row) in rows.chunks_exact_mut(row_bytes).enumerate() {
                    let y = (band * 32 + i) as i64;
                    if y >= top && y < bottom && right > left {
                        let lo = left as usize * channels;
                        let hi = right as usize * channels;
                        fill_row(&mut row[..lo]);
                        fill_row(&mut row[hi..]);
                        let src = ((y - offset.1) as usize * image.width as usize
                            + (left - offset.0) as usize)
                            * channels;
                        row[lo..hi].copy_from_slice(&source[src..src + hi - lo]);
                    } else {
                        fill_row(row);
                    }
                }
            },
        );
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
    let size = if orientation >= 5 {
        (image.height, image.width)
    } else {
        (image.width, image.height)
    };
    let c = image.mode.channels();
    let mut pixels = buffer(size, c)?;
    py.detach(|| match image.mode {
        PixelMode::L => transpose::<1>(
            source,
            &mut pixels,
            image.width as usize,
            image.height as usize,
            orientation,
        ),
        PixelMode::Rgb => transpose::<3>(
            source,
            &mut pixels,
            image.width as usize,
            image.height as usize,
            orientation,
        ),
        PixelMode::Rgba => transpose::<4>(
            source,
            &mut pixels,
            image.width as usize,
            image.height as usize,
            orientation,
        ),
    });
    output(image, size, pixels)
}

fn transpose<const C: usize>(
    source: &[u8],
    output: &mut [u8],
    w: usize,
    h: usize,
    orientation: u8,
) {
    let source = source.as_chunks::<C>().0;
    let width = if orientation >= 5 { h } else { w };
    // Bands and tiles keep 90-degree rotations local to cache and give every
    // worker exclusive ownership of complete destination rows.
    let threshold = if orientation < 5 {
        2 * 1024 * 1024
    } else {
        MIN_PARALLEL_BYTES
    };
    chunks_mut_above(output, width * C * 32, threshold, |band, rows| {
        if orientation < 5 {
            for (i, row) in rows.chunks_exact_mut(w * C).enumerate() {
                let y = band * 32 + i;
                let sy = if orientation == 3 || orientation == 4 {
                    h - 1 - y
                } else {
                    y
                };
                let src = &source[sy * w..(sy + 1) * w];
                if orientation == 1 || orientation == 4 {
                    row.copy_from_slice(src.as_flattened());
                } else {
                    for (dst, src) in row.as_chunks_mut::<C>().0.iter_mut().zip(src.iter().rev()) {
                        *dst = *src;
                    }
                }
            }
        } else {
            for x0 in (0..width).step_by(32) {
                for (i, row) in rows.chunks_exact_mut(width * C).enumerate() {
                    let y = band * 32 + i;
                    let row = row.as_chunks_mut::<C>().0;
                    for (x, dst) in row
                        .iter_mut()
                        .enumerate()
                        .take((x0 + 32).min(width))
                        .skip(x0)
                    {
                        let sx = if orientation == 7 || orientation == 8 {
                            w - 1 - y
                        } else {
                            y
                        };
                        let sy = if orientation == 6 || orientation == 7 {
                            h - 1 - x
                        } else {
                            x
                        };
                        *dst = source[sy * w + sx];
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
                    .map(|v| {
                        if sum == 0.0 {
                            0
                        } else {
                            (v / sum * 4194304.0).round() as i32
                        }
                    })
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

#[pyfunction]
fn ops_resize(
    py: Python<'_>,
    image: &Image,
    size: (u32, u32),
    method: u8,
    bounds: BoxF,
) -> PyResult<Image> {
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
    if u64::from(image.height) > u64::from(image.width) * 100
        && size.1 < image.height
        && image.width > 0
    {
        let transposed = ops_transpose(py, image, 5)?;
        let resized = ops_resize(
            py,
            &transposed,
            (size.1, size.0),
            method,
            (bounds.1, bounds.0, bounds.3, bounds.2),
        )?;
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
            let row_bytes = size.0 as usize * c;
            chunks_mut(&mut result, row_bytes * 16, |band, rows| {
                for (i, row) in rows.chunks_exact_mut(row_bytes).enumerate() {
                    let sy = ys[band * 16 + i];
                    for (x, &sx) in xs.iter().enumerate() {
                        let src = (sy * image.width as usize + sx) * c;
                        row[x * c..(x + 1) * c].copy_from_slice(&source[src..src + c]);
                    }
                }
            });
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

fn resample<const C: usize>(
    source: &[u8],
    output: &mut [u8],
    image: &Image,
    size: (u32, u32),
    b: [f64; 4],
    method: u8,
) -> PyResult<()> {
    let horizontal = weights(image.width, size.0, b[0], b[2], method);
    let vertical = weights(image.height, size.1, b[1], b[3], method);
    let first_row = vertical.first().unwrap().0;
    let (last_start, last_weights) = vertical.last().unwrap();
    let last_row = last_start + last_weights.len();
    let row_bytes = size.0 as usize * C;
    let input_stride = image.width as usize * C;
    let horizontal_identity =
        size.0 == image.width && b[0] == 0.0 && b[2] == f64::from(image.width);
    let temp = if horizontal_identity {
        Cow::Borrowed(&source[first_row * input_stride..last_row * input_stride])
    } else {
        let mut temp = buffer((size.0, (last_row - first_row) as u32), C)?;
        let narrow = horizontal.iter().all(|(_, coefficients)| {
            coefficients
                .iter()
                .map(|&w| i64::from(w).abs())
                .sum::<i64>()
                * 255
                + (1 << 21)
                <= i64::from(i32::MAX)
        });
        chunks_mut(&mut temp, row_bytes * 16, |band, rows| {
            for (i, dst) in rows.chunks_exact_mut(row_bytes).enumerate() {
                let y = first_row + band * 16 + i;
                let src = source[y * input_stride..(y + 1) * input_stride]
                    .as_chunks::<C>()
                    .0;
                for ((start, coefficients), dst) in
                    horizontal.iter().zip(dst.as_chunks_mut::<C>().0)
                {
                    if narrow {
                        let mut sums = [1_i32 << 21; C];
                        for (pixel, &weight) in src[*start..*start + coefficients.len()]
                            .iter()
                            .zip(coefficients)
                        {
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
                                .fold(1_i64 << 21, |sum, (i, &w)| {
                                    sum + i64::from(src[start + i][c]) * i64::from(w)
                                });
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
                    .fold(1_i64 << 21, |sum, (i, &w)| {
                        sum + i64::from(src[i * row_bytes + x]) * i64::from(w)
                    });
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
    let (w, h, c) = (
        image.width as usize,
        image.height as usize,
        image.mode.channels(),
    );
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
            f64::from(
                source[((iy + dy).clamp(0, h as i64 - 1) as usize * w
                    + (ix + dx).clamp(0, w as i64 - 1) as usize)
                    * c
                    + ch],
            )
        };
        let value = if method == 2 {
            let top = interpolate(get(0, 0), get(1, 0), tx);
            let bottom = interpolate(get(0, 1), get(1, 1), tx);
            interpolate(top, bottom, ty)
        } else {
            cubic(
                [-1, 0, 1, 2].map(|dy| cubic([-1, 0, 1, 2].map(|dx| get(dx, dy)), tx)),
                ty,
            )
        };
        *out = value.clamp(0.0, 255.0) as u8;
    }
}

#[pyfunction]
fn ops_mesh(py: Python<'_>, image: &Image, mesh: Mesh, method: u8) -> PyResult<Image> {
    let source = image.pixel_data()?;
    if ![0, 2, 3].contains(&method) {
        return Err(PyValueError::new_err(
            "mesh transforms support NEAREST, BILINEAR and BICUBIC",
        ));
    }
    let size = (image.width, image.height);
    let c = image.mode.channels();
    let mut result = buffer(size, c)?;
    py.detach(|| {
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
            chunks_mut(
                &mut result[first * row_bytes..last * row_bytes],
                row_bytes * 16,
                |band, rows| {
                    for (i, row) in rows.chunks_exact_mut(row_bytes).enumerate() {
                        let y = (first + band * 16 + i) as i64;
                        for x in left.max(0)..right.min(i64::from(size.0)) {
                            let u = (x - left.max(0)) as f64 + 0.5;
                            let v = (y - top.max(0)) as f64 + 0.5;
                            let sx = q[0] + ax * u + bx * v + cx * u * v;
                            let sy = q[1] + ay * u + by * v + cy * u * v;
                            let dst = x as usize * c;
                            sample(&source, image, sx, sy, method, &mut row[dst..dst + c]);
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
            let image =
                Image::from_pixels(3, 2, PixelMode::L, vec![1, 2, 3, 4, 5, 6], None).unwrap();
            let rotated = ops_transpose(py, &image, 6).unwrap();
            assert_eq!((rotated.width, rotated.height), (2, 3));
            assert_eq!(rotated.pixel_data().unwrap(), [4, 1, 5, 2, 6, 3]);
            let clipped = ops_canvas(py, &image, (3, 3), (-1, 1), vec![9]).unwrap();
            assert_eq!(clipped.pixel_data().unwrap(), [9, 9, 9, 2, 3, 9, 5, 6, 9]);
        });
    }
}
