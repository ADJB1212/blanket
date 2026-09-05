//! Pixel primitives used by the Python ImageOps API.
use pyo3::exceptions::{PyMemoryError, PyValueError};
use pyo3::prelude::*;

use crate::raster::{Image, PixelMode};

type BoxI = (i64, i64, i64, i64);
type BoxF = (f64, f64, f64, f64);
type Mesh = Vec<(BoxI, [f64; 8])>;

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(ops_lut, module)?)?;
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
    let pixels = py.detach(|| {
        source
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                lut[usize::from(v)
                    + if lut.len() == 256 {
                        0
                    } else {
                        (i % channels) * 256
                    }]
            })
            .collect()
    });
    output(image, (image.width, image.height), pixels)
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
    Ok(py.detach(|| {
        let channels = image.mode.channels();
        let mut histogram = vec![0; channels * 256];
        for (i, pixel) in source.chunks_exact(channels).enumerate() {
            if mask.is_none_or(|m| m[i] != 0) {
                for (c, &v) in pixel.iter().enumerate() {
                    histogram[c * 256 + usize::from(v)] += 1;
                }
            }
        }
        histogram
    }))
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
        for p in pixels.chunks_exact_mut(channels) {
            p.copy_from_slice(&fill);
        }
        let left = offset.0.max(0).min(i64::from(size.0));
        let right = (offset.0.saturating_add(i64::from(image.width))).clamp(0, i64::from(size.0));
        let top = offset.1.max(0).min(i64::from(size.1));
        let bottom = (offset.1.saturating_add(i64::from(image.height))).clamp(0, i64::from(size.1));
        if right <= left {
            return;
        }
        let count = (right - left) as usize * channels;
        for y in top..bottom {
            let dst = (y as usize * size.0 as usize + left as usize) * channels;
            let src = ((y - offset.1) as usize * image.width as usize + (left - offset.0) as usize)
                * channels;
            pixels[dst..dst + count].copy_from_slice(&source[src..src + count]);
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
    let (w, h) = (image.width as usize, image.height as usize);
    let size = if orientation >= 5 {
        (image.height, image.width)
    } else {
        (image.width, image.height)
    };
    let c = image.mode.channels();
    let mut pixels = buffer(size, c)?;
    py.detach(|| {
        for y in 0..size.1 as usize {
            for x in 0..size.0 as usize {
                let (sx, sy) = match orientation {
                    2 => (w - 1 - x, y),
                    3 => (w - 1 - x, h - 1 - y),
                    4 => (x, h - 1 - y),
                    5 => (y, x),
                    6 => (y, h - 1 - x),
                    7 => (w - 1 - y, h - 1 - x),
                    8 => (w - 1 - y, x),
                    _ => (x, y),
                };
                let src = (sy * w + sx) * c;
                let dst = (y * size.0 as usize + x) * c;
                pixels[dst..dst + c].copy_from_slice(&source[src..src + c]);
            }
        }
    });
    output(image, size, pixels)
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
    for p in pixels.as_chunks_mut::<4>().0 {
        for c in 0..3 {
            let v = u32::from(p[c]) * u32::from(p[3]) + 128;
            p[c] = (((v >> 8) + v) >> 8) as u8;
        }
    }
}

fn unpremultiply(pixels: &mut [u8]) {
    for p in pixels.as_chunks_mut::<4>().0 {
        if p[3] != 0 && p[3] != 255 {
            for c in 0..3 {
                p[c] = (u32::from(p[c]) * 255 / u32::from(p[3])).min(255) as u8;
            }
        }
    }
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
            let mut fy = b[1] + 0.5 * dy;
            for y in 0..size.1 as usize {
                let sy = (fy as usize).min(image.height as usize - 1);
                let mut fx = b[0] + 0.5 * dx;
                for x in 0..size.0 as usize {
                    let sx = (fx as usize).min(image.width as usize - 1);
                    let src = (sy * image.width as usize + sx) * c;
                    let dst = (y * size.0 as usize + x) * c;
                    result[dst..dst + c].copy_from_slice(&source[src..src + c]);
                    fx += dx;
                }
                fy += dy;
            }
            return Ok(());
        }
        let mut source = source.to_vec();
        if c == 4 {
            premultiply(&mut source);
        }
        let horizontal = weights(image.width, size.0, b[0], b[2], method);
        let vertical = weights(image.height, size.1, b[1], b[3], method);
        let mut temp = buffer((size.0, image.height), c)?;
        for y in 0..image.height as usize {
            for (x, (start, coefficients)) in horizontal.iter().enumerate() {
                for channel in 0..c {
                    let mut sum = 1_i64 << 21;
                    for (i, &weight) in coefficients.iter().enumerate() {
                        sum +=
                            i64::from(source[(y * image.width as usize + start + i) * c + channel])
                                * i64::from(weight);
                    }
                    temp[(y * size.0 as usize + x) * c + channel] = (sum >> 22).clamp(0, 255) as u8;
                }
            }
        }
        for (y, (start, coefficients)) in vertical.iter().enumerate() {
            for x in 0..size.0 as usize {
                for channel in 0..c {
                    let mut sum = 1_i64 << 21;
                    for (i, &weight) in coefficients.iter().enumerate() {
                        sum += i64::from(temp[((start + i) * size.0 as usize + x) * c + channel])
                            * i64::from(weight);
                    }
                    result[(y * size.0 as usize + x) * c + channel] =
                        (sum >> 22).clamp(0, 255) as u8;
                }
            }
        }
        if c == 4 {
            unpremultiply(&mut result);
        }
        Ok(())
    })?;
    output(image, size, result)
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
        let mut source = source.to_vec();
        if c == 4 && method != 0 {
            premultiply(&mut source);
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
            for y in top.max(0)..bottom.min(i64::from(size.1)) {
                for x in left.max(0)..right.min(i64::from(size.0)) {
                    let u = (x - left.max(0)) as f64 + 0.5;
                    let v = (y - top.max(0)) as f64 + 0.5;
                    let sx = q[0] + ax * u + bx * v + cx * u * v;
                    let sy = q[1] + ay * u + by * v + cy * u * v;
                    let dst = (y as usize * size.0 as usize + x as usize) * c;
                    sample(&source, image, sx, sy, method, &mut result[dst..dst + c]);
                }
            }
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
