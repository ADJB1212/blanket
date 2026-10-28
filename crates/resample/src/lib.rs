use std::borrow::Cow;
use std::num::NonZeroUsize;
use std::sync::{Arc, LazyLock, Mutex};

use blanket_core::buffers::{buffer, copy_rows, output, reserved_buffer};
use blanket_core::geometry::ops_transpose;
use blanket_core::raster::{Image, PixelMode};
use blanket_kernels::cache::ByteLru;
use blanket_kernels::ops as ops_simd;
use blanket_kernels::parallel::{CHUNK_PIXELS, MIN_PARALLEL_BYTES, chunks_mut, chunks_mut_above};
use ops_simd::{RESAMPLE_BITS, RESAMPLE_ROUND, RESAMPLE_SCALE};
use pyo3::exceptions::{PyMemoryError, PyValueError};
use pyo3::prelude::*;
type BoxI = (i64, i64, i64, i64);
type BoxF = (f64, f64, f64, f64);
type Mesh = Vec<(BoxI, [f64; 8])>;
pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(ops_color_resize, module)?)?;
    module.add_function(wrap_pyfunction!(ops_resize, module)?)?;
    module.add_function(wrap_pyfunction!(ops_reduce, module)?)?;
    module.add_function(wrap_pyfunction!(ops_mesh, module)?)?;
    module.add_function(wrap_pyfunction!(ops_warp, module)?)?;
    module.add_function(wrap_pyfunction!(ops_affine, module)?)?;
    Ok(())
}

#[pyfunction]
fn ops_color_resize(py: Python<'_>, image: &Image, size: (u32, u32), method: u8, bounds: BoxF, srgb: bool, premultiply: bool) -> PyResult<Image> {
    let prepared = py.detach(|| blanket_core::float::prepare(image, srgb, premultiply))?;
    let resized = ops_resize(py, &prepared, size, method, bounds, None)?;
    py.detach(|| blanket_core::float::finish(&resized, image, srgb, premultiply))
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
struct ResampleWeights {
    bounds: Vec<(usize, std::ops::Range<usize>)>,
    coefficients: Vec<i32>,
}

impl ResampleWeights {
    fn get(&self, index: usize) -> (usize, &[i32]) {
        let (start, range) = &self.bounds[index];
        (*start, &self.coefficients[range.clone()])
    }

    fn iter(&self) -> impl ExactSizeIterator<Item = (usize, &[i32])> {
        self.bounds.iter().map(|(start, range)| (*start, &self.coefficients[range.clone()]))
    }
}

type WeightKey = (u32, u32, u64, u64, u8);
static WEIGHTS: LazyLock<Mutex<ByteLru<WeightKey, Arc<ResampleWeights>>>> =
    LazyLock::new(|| Mutex::new(ByteLru::new(NonZeroUsize::new(128).unwrap(), 16 * 1024 * 1024)));

fn weights(input: u32, output: u32, start: f64, end: f64, method: u8) -> Arc<ResampleWeights> {
    let key = (input, output, start.to_bits(), end.to_bits(), method);
    let cached = WEIGHTS.lock().unwrap_or_else(|error| error.into_inner()).get(&key).cloned();
    if let Some(weights) = cached {
        return weights;
    }
    let weights = Arc::new(compute_weights(input, output, start, end, method));
    let bytes = weights.bounds.capacity() * size_of::<(usize, std::ops::Range<usize>)>()
        + weights.coefficients.capacity() * size_of::<i32>()
        + size_of::<ResampleWeights>();
    WEIGHTS.lock().unwrap_or_else(|error| error.into_inner()).insert(key, Arc::clone(&weights), bytes);
    weights
}

fn compute_weights(input: u32, output: u32, start: f64, end: f64, method: u8) -> ResampleWeights {
    let scale = f64::from(end as f32 - start as f32) / f64::from(output);
    let filter_scale = scale.max(1.0);
    let support = match method {
        1 => 3.0,
        3 => 2.0,
        4 => 0.5,
        _ => 1.0,
    } * filter_scale;
    let mut raw = Vec::new();
    let mut weights = ResampleWeights { bounds: Vec::with_capacity(output as usize), coefficients: Vec::new() };
    for i in 0..output {
        let center = start + (f64::from(i) + 0.5) * scale;
        let left = ((center - support + 0.5) as i64).clamp(0, i64::from(input)) as usize;
        let right = ((center + support + 0.5) as i64).clamp(0, i64::from(input)) as usize;
        raw.clear();
        raw.extend((left..right).map(|x| kernel((x as f64 - center + 0.5) * (1.0 / filter_scale), method)));
        let sum: f64 = raw.iter().sum();
        let offset = weights.coefficients.len();
        weights.coefficients.extend(raw.iter().map(|&v| if sum == 0.0 { 0 } else { (v / sum * f64::from(RESAMPLE_SCALE)).round() as i32 }));
        weights.bounds.push((left, offset..weights.coefficients.len()));
    }
    weights
}

fn premultiply(pixels: &mut [u8]) {
    chunks_mut(pixels, CHUNK_PIXELS << 2, |_, chunk| {
        blanket_core::pixels::alpha_scale::<4, false>(chunk);
    });
}

fn unpremultiply(pixels: &mut [u8]) {
    chunks_mut(pixels, CHUNK_PIXELS << 2, |_, chunk| {
        blanket_core::pixels::alpha_scale::<4, true>(chunk);
    });
}

fn premultiply_la(pixels: &mut [u8]) {
    blanket_core::pixels::alpha_scale::<2, false>(pixels);
}

fn unpremultiply_la(pixels: &mut [u8]) {
    blanket_core::pixels::alpha_scale::<2, true>(pixels);
}

fn premultiply_image(mode: PixelMode, pixels: &mut [u8]) {
    if mode == PixelMode::Rgba {
        premultiply(pixels);
    } else if mode == PixelMode::La {
        premultiply_la(pixels);
    }
}

fn unpremultiply_image(mode: PixelMode, pixels: &mut [u8]) {
    if mode == PixelMode::Rgba {
        unpremultiply(pixels);
    } else if mode == PixelMode::La {
        unpremultiply_la(pixels);
    }
}

/// Integer box reduction used by resize's reducing_gap optimization.
#[pyfunction]
fn ops_reduce(py: Python<'_>, image: &Image, factor: (u32, u32), bounds: (u32, u32, u32, u32)) -> PyResult<Image> {
    if matches!(image.mode, PixelMode::RgbF | PixelMode::RgbaF) {
        return Err(PyValueError::new_err("use resize for float color images"));
    }
    let (fx, fy) = factor;
    let (left, top, right, bottom) = bounds;
    if fx == 0 || fy == 0 || left > right || top > bottom || right > image.width || bottom > image.height {
        return Err(PyValueError::new_err("invalid reduction bounds or factor"));
    }
    let size = ((right - left).div_ceil(fx), (bottom - top).div_ceil(fy));
    if image.mode == PixelMode::F {
        let source = image.raw_data()?;
        let mut pixels = buffer(size, 4)?;
        py.detach(|| {
            if size.0 == 0 {
                return;
            }
            blanket_core::parallel::chunks_mut(&mut pixels, size.0 as usize * 4, |y, row| {
                let y = y as u32;
                let sy = top + y * fy;
                let end_y = sy.saturating_add(fy).min(bottom);
                let mut first = 0;
                if fx == 2 && end_y - sy == 2 {
                    let start = (sy as usize * image.width as usize + left as usize) << 2;
                    let width = (right - left) as usize * 4;
                    first =
                        crate::ops_simd::reduce_float_two(&source[start..start + width], &source[start + image.width as usize * 4..][..width], row)
                            as u32;
                }
                for x in first..size.0 {
                    let mut sum = 0.0_f64;
                    let sx = left + x * fx;
                    let end_x = sx.saturating_add(fx).min(right);
                    for sy in sy..end_y {
                        let start = (sy as usize * image.width as usize + sx as usize) << 2;
                        for bytes in source[start..start + (end_x - sx) as usize * 4].as_chunks::<4>().0 {
                            sum += f64::from(f32::from_le_bytes(*bytes));
                        }
                    }
                    let count = f64::from(end_x - sx) * f64::from(end_y - sy);
                    let start = x as usize * 4;
                    row[start..start + 4].copy_from_slice(&((sum / count) as f32).to_le_bytes());
                }
            });
        });
        return output(image, size, pixels);
    }
    let source = image.pixel_data()?;
    let c = image.mode.channels();
    if factor == (1, 1) {
        return output(image, size, reduction_crop(source, image.width, bounds, c)?);
    }
    let mut pixels = buffer(size, c)?;
    py.detach(|| -> PyResult<()> {
        match image.mode {
            PixelMode::One | PixelMode::L => reduce_pixels::<1>(source, &mut pixels, image.width, size, factor, bounds),
            PixelMode::Pa => reduce_pixels::<2>(source, &mut pixels, image.width, size, factor, bounds),
            PixelMode::Rgb | PixelMode::Hsv | PixelMode::YCbCr | PixelMode::Lab => {
                reduce_pixels::<3>(source, &mut pixels, image.width, size, factor, bounds)
            }
            PixelMode::Cmyk => reduce_pixels::<4>(source, &mut pixels, image.width, size, factor, bounds),
            PixelMode::La | PixelMode::Rgba => {
                let mut source = reduction_crop(source, image.width, bounds, c)?;
                premultiply_image(image.mode, &mut source);
                if c == 2 {
                    reduce_pixels::<2>(&source, &mut pixels, right - left, size, factor, (0, 0, right - left, bottom - top));
                } else {
                    reduce_pixels::<4>(&source, &mut pixels, right - left, size, factor, (0, 0, right - left, bottom - top));
                }
                unpremultiply_image(image.mode, &mut pixels);
            }
            _ => unreachable!(),
        }
        Ok(())
    })?;
    output(image, size, pixels)
}

fn reduction_crop(source: &[u8], width: u32, bounds: (u32, u32, u32, u32), channels: usize) -> PyResult<Vec<u8>> {
    let (left, top, right, bottom) = bounds;
    let mut pixels = reserved_buffer((right - left, bottom - top), channels)?;
    let row_bytes = (right - left) as usize * channels;
    copy_rows(&mut pixels, row_bytes, (bottom - top) as usize, |y| {
        let start = ((top as usize + y) * width as usize + left as usize) * channels;
        &source[start..start + row_bytes]
    });
    Ok(pixels)
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
    let threshold = if cfg!(target_arch = "aarch64") && C == 1 && matches!((fx, fy), (3, 3) | (4, 4)) {
        128 << 10
    } else {
        MIN_PARALLEL_BYTES / (fx as usize * fy as usize).max(1)
    };
    chunks_mut_above(output, row_bytes * 8, threshold, |band, rows| {
        let mut column_sums = if fx == 1 { vec![0_u64; row_bytes] } else { Vec::new() };
        for (i, row) in rows.chunks_exact_mut(row_bytes).enumerate() {
            let y0 = top + (band * 8 + i) as u32 * fy;
            let y1 = y0.saturating_add(fy).min(bottom);
            if fx == 1 {
                let count = u64::from(y1 - y0);
                let multiplier = ((1_u64 << 24) as f32 / count as f32) as u64;
                column_sums.fill(0);
                for sy in y0..y1 {
                    let start = sy as usize * width as usize + left as usize;
                    for (sum, &sample) in column_sums.iter_mut().zip(source[start..start + size.0 as usize].as_flattened()) {
                        *sum += u64::from(sample);
                    }
                }
                for (dst, sum) in row.iter_mut().zip(&column_sums) {
                    *dst = (((sum + count / 2) * multiplier) >> 24) as u8;
                }
                continue;
            }
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
                        dst[channel] =
                            (((sum + 4) * blanket_core::pixels::BOX_3X3_RECIPROCAL) >> blanket_core::pixels::BOX_3X3_RECIPROCAL_BITS) as u8;
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
                let done = if C == 1 {
                    crate::ops_simd::reduce_two_l(upper.as_flattened().as_flattened(), lower.as_flattened().as_flattened(), row)
                } else {
                    0
                };
                for ((dst, upper), lower) in row.as_chunks_mut::<C>().0.iter_mut().zip(upper).zip(lower).skip(done) {
                    for channel in 0..C {
                        let sum =
                            u16::from(upper[0][channel]) + u16::from(upper[1][channel]) + u16::from(lower[0][channel]) + u16::from(lower[1][channel]);
                        dst[channel] = ((sum + 2) >> 2) as u8;
                    }
                }
                continue;
            }
            if fx == 4 && fy == 4 && y1 - y0 == 4 && (right - left).is_multiple_of(4) {
                let start = y0 as usize * width as usize + left as usize;
                let count = (right - left) as usize;
                let stride = width as usize;
                let rows = std::array::from_fn::<_, 4, _>(|i| source[start + i * stride..start + i * stride + count].as_chunks::<4>().0);
                for (x, dst) in row.as_chunks_mut::<C>().0.iter_mut().enumerate() {
                    for c in 0..C {
                        let mut sum = 8_u16;
                        for samples in &rows {
                            for pixel in &samples[x] {
                                sum += u16::from(pixel[c]);
                            }
                        }
                        dst[c] = (sum >> 4) as u8;
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

#[pyfunction(signature = (image, size, method, bounds, reducing_gap=None))]
pub fn ops_resize(py: Python<'_>, image: &Image, size: (u32, u32), method: u8, bounds: BoxF, reducing_gap: Option<f64>) -> PyResult<Image> {
    let source = image.raw_data()?;
    if method > 5 {
        return Err(PyValueError::new_err("unknown resampling filter"));
    }
    if size.0 == 0 || size.1 == 0 {
        return Err(PyValueError::new_err("height and width must be > 0"));
    }
    // The public Pillow resize API passes its box through C floats.
    let mut b = <[_; 4]>::from(bounds).map(|x| f64::from(x as f32));
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
    if matches!(image.mode, PixelMode::RgbF | PixelMode::RgbaF) {
        let bands = (0..image.mode.channels())
            .map(|c| {
                let band = image.getchannel(py, c)?;
                ops_resize(py, &band, size, method, bounds, reducing_gap)
            })
            .collect::<PyResult<Vec<_>>>()?;
        let mut pixels = buffer(size, image.raw_bytes_per_pixel())?;
        for (i, p) in pixels.chunks_exact_mut(image.raw_bytes_per_pixel()).enumerate() {
            for (c, band) in bands.iter().enumerate() {
                p[c * 4..c * 4 + 4].copy_from_slice(&band.raw_data()?[i * 4..i * 4 + 4]);
            }
        }
        return output(image, size, pixels);
    }
    if image.mode.is_wide_scalar() {
        return py.detach(|| resize_integer(image, size, b, method));
    }
    if image.bit_depth > 8 {
        return py.detach(|| resize_wide(image, size, b, method));
    }
    // Pillow resamples very tall images vertically first to limit temporary
    // storage. Pass order affects 8-bit rounding, so retain it as well.
    if u64::from(image.height) > u64::from(image.width) * 100 && size.1 < image.height && image.width > 0 {
        let transposed = ops_transpose(py, image, 5)?;
        let resized = ops_resize(py, &transposed, (size.1, size.0), method, (bounds.1, bounds.0, bounds.3, bounds.2), reducing_gap)?;
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
                PixelMode::One | PixelMode::L => resize_nearest::<1>(source, &mut result, image.width, &xs, &ys),
                PixelMode::La | PixelMode::Pa => resize_nearest::<2>(source, &mut result, image.width, &xs, &ys),
                PixelMode::Rgb | PixelMode::Hsv | PixelMode::YCbCr | PixelMode::Lab => {
                    resize_nearest::<3>(source, &mut result, image.width, &xs, &ys)
                }
                PixelMode::Rgba | PixelMode::Cmyk => resize_nearest::<4>(source, &mut result, image.width, &xs, &ys),
                _ => unreachable!(),
            }
            return Ok(());
        }
        let mut source = Cow::Borrowed(source);
        if matches!(image.mode, PixelMode::La | PixelMode::Rgba) {
            premultiply_image(image.mode, source.to_mut());
        }
        let mut reduced = None;
        if let Some(gap) = reducing_gap {
            let fx = (((bounds.2 - bounds.0) / f64::from(size.0) / gap) as u32).max(1);
            let fy = (((bounds.3 - bounds.1) / f64::from(size.1) / gap) as u32).max(1);
            if fx > 1 || fy > 1 {
                let support = match method {
                    1 => 3.0,
                    3 => 2.0,
                    4 => 0.5,
                    _ => 1.0,
                } - 0.5;
                let sx = support * (bounds.2 - bounds.0) / f64::from(size.0);
                let sy = support * (bounds.3 - bounds.1) / f64::from(size.1);
                let safe = (
                    (bounds.0 - sx).max(0.0) as u32,
                    (bounds.1 - sy).max(0.0) as u32,
                    (bounds.2 + sx).ceil().min(f64::from(image.width)) as u32,
                    (bounds.3 + sy).ceil().min(f64::from(image.height)) as u32,
                );
                let reduced_size = ((safe.2 - safe.0).div_ceil(fx), (safe.3 - safe.1).div_ceil(fy));
                let mut pixels = buffer(reduced_size, c)?;
                match image.mode {
                    PixelMode::La => reduce_pixels::<2>(&source, &mut pixels, image.width, reduced_size, (fx, fy), safe),
                    PixelMode::Rgba => reduce_pixels::<4>(&source, &mut pixels, image.width, reduced_size, (fx, fy), safe),
                    _ => return Err(PyValueError::new_err("resize reduction requires an alpha image")),
                }
                b = [
                    (bounds.0 - f64::from(safe.0)) / f64::from(fx),
                    (bounds.1 - f64::from(safe.1)) / f64::from(fy),
                    (bounds.2 - f64::from(safe.0)) / f64::from(fx),
                    (bounds.3 - f64::from(safe.1)) / f64::from(fy),
                ]
                .map(|x| f64::from(x as f32));
                reduced = Some(output(image, reduced_size, pixels)?);
            }
        }
        let image = reduced.as_ref().unwrap_or(image);
        if reduced.is_some() {
            source = Cow::Borrowed(image.pixel_data()?);
        }
        match image.mode {
            PixelMode::One | PixelMode::L => resample::<1>(&source, &mut result, image, size, b, method)?,
            PixelMode::La | PixelMode::Pa => resample::<2>(&source, &mut result, image, size, b, method)?,
            PixelMode::Rgb | PixelMode::Hsv | PixelMode::YCbCr | PixelMode::Lab => resample::<3>(&source, &mut result, image, size, b, method)?,
            PixelMode::Rgba | PixelMode::Cmyk => resample::<4>(&source, &mut result, image, size, b, method)?,
            _ => unreachable!(),
        }
        unpremultiply_image(image.mode, &mut result);
        Ok(())
    })?;
    output(image, size, result)
}

fn nearest_positions(bounds: [f64; 4], size: (u32, u32), src_width: u32, src_height: u32) -> (Vec<usize>, Vec<usize>) {
    let xs = (0..size.0)
        .map(|x| ((bounds[0] + (f64::from(x) + 0.5) * (bounds[2] - bounds[0]) / f64::from(size.0)) as usize).min(src_width as usize - 1))
        .collect();
    let ys = (0..size.1)
        .map(|y| ((bounds[1] + (f64::from(y) + 0.5) * (bounds[3] - bounds[1]) / f64::from(size.1)) as usize).min(src_height as usize - 1))
        .collect();
    (xs, ys)
}

fn resize_integer(image: &Image, size: (u32, u32), bounds: [f64; 4], method: u8) -> PyResult<Image> {
    let source = image.raw_data()?;
    let stride = image.mode.sample_bytes();
    let mut pixels = buffer(size, stride)?;
    if image.width == 0 || image.height == 0 {
        return output(image, size, pixels);
    }
    let value = |index: usize| -> f64 {
        let start = index * stride;
        match image.mode {
            PixelMode::I => f64::from(i32::from_le_bytes(source[start..start + 4].try_into().unwrap())),
            PixelMode::F => f64::from(f32::from_le_bytes(source[start..start + 4].try_into().unwrap())),
            // Pillow resamples I;16B from its raw bytes as little-endian samples.
            _ => f64::from(u16::from_le_bytes(source[start..start + 2].try_into().unwrap())),
        }
    };
    if method == 0 {
        let (xs, ys) = nearest_positions(bounds, size, image.width, image.height);
        if stride == 4 {
            resize_nearest::<4>(source, &mut pixels, image.width, &xs, &ys);
        } else {
            resize_nearest::<2>(source, &mut pixels, image.width, &xs, &ys);
        }
    } else {
        let horizontal = weights(image.width, size.0, bounds[0], bounds[2], method);
        let vertical = weights(image.height, size.1, bounds[1], bounds[3], method);
        let first = vertical.get(0).0;
        let (last, tail) = vertical.get(size.1 as usize - 1);
        let mut rows = vec![0.0; (last + tail.len() - first) * size.0 as usize];
        let row_width = size.0 as usize;
        blanket_core::parallel::chunks_mut_above(&mut rows, row_width * 16, 256 * 1024, |band, chunk| {
            let vector_rows = if image.mode == PixelMode::F { chunk.len() / row_width / 4 * 4 } else { 0 };
            for row in (0..vector_rows).step_by(4) {
                let sy = first + band * 16 + row;
                for (x, (start, coefficients)) in horizontal.iter().enumerate() {
                    let source_rows = std::array::from_fn(|r| &source[((sy + r) * image.width as usize + start) * 4..]);
                    let values = crate::ops_simd::horizontal_float_four(source_rows, coefficients);
                    for r in 0..4 {
                        chunk[(row + r) * row_width + x] = values[r];
                    }
                }
            }
            for (row, dst) in chunk.chunks_exact_mut(row_width).enumerate().skip(vector_rows) {
                let sy = first + band * 16 + row;
                for (x, (start, coefficients)) in horizontal.iter().enumerate() {
                    dst[x] = coefficients
                        .iter()
                        .enumerate()
                        .map(|(i, &weight)| value(sy * image.width as usize + start + i) * f64::from(weight) / f64::from(RESAMPLE_SCALE))
                        .sum();
                }
            }
        });
        blanket_core::parallel::chunks_mut_above(&mut pixels, row_width * stride * 16, 256 << 10, |band, chunk| {
            for (row, dst) in chunk.chunks_exact_mut(row_width * stride).enumerate() {
                let (start, coefficients) = vertical.get(band * 16 + row);
                if image.mode == PixelMode::F {
                    crate::ops_simd::vertical_float(&rows[(start - first) * row_width..], dst, row_width, coefficients);
                    continue;
                }
                for x in 0..row_width {
                    let sample: f64 = coefficients
                        .iter()
                        .enumerate()
                        .map(|(i, &weight)| rows[(start - first + i) * row_width + x] * f64::from(weight) / f64::from(RESAMPLE_SCALE))
                        .sum();
                    let offset = x * stride;
                    match image.mode {
                        PixelMode::I => {
                            dst[offset..offset + 4].copy_from_slice(&(sample.round().clamp(i32::MIN as f64, i32::MAX as f64) as i32).to_le_bytes())
                        }
                        PixelMode::F => dst[offset..offset + 4].copy_from_slice(&(sample as f32).to_le_bytes()),
                        _ => dst[offset..offset + 2].copy_from_slice(&(sample.round().clamp(0.0, 65535.0) as u16).to_le_bytes()),
                    }
                }
            }
        });
    }
    output(image, size, pixels)
}

// Fixed-size pixels let LLVM inline the gathers instead of calling memcpy
// for every output pixel. Repeated source rows only need one gather per band.
fn resize_wide(image: &Image, size: (u32, u32), bounds: [f64; 4], method: u8) -> PyResult<Image> {
    match image.mode.channels() {
        1 => resize_wide_channels::<1>(image, size, bounds, method),
        3 => resize_wide_channels::<3>(image, size, bounds, method),
        4 => resize_wide_channels::<4>(image, size, bounds, method),
        _ => unreachable!("validated wide mode"),
    }
}

fn resize_wide_channels<const C: usize>(image: &Image, size: (u32, u32), bounds: [f64; 4], method: u8) -> PyResult<Image> {
    let channels = C;
    let bytes = image.raw_data()?;
    let mut pixels = buffer(size, channels << 1)?;
    if image.width == 0 || image.height == 0 {
        return output(image, size, pixels);
    }
    if method == 0 {
        let (xs, ys) = nearest_positions(bounds, size, image.width, image.height);
        match image.mode {
            PixelMode::One | PixelMode::L => resize_nearest::<2>(bytes, &mut pixels, image.width, &xs, &ys),
            PixelMode::La | PixelMode::Pa => resize_nearest::<4>(bytes, &mut pixels, image.width, &xs, &ys),
            PixelMode::Rgb | PixelMode::Hsv | PixelMode::YCbCr | PixelMode::Lab => resize_nearest::<6>(bytes, &mut pixels, image.width, &xs, &ys),
            PixelMode::Rgba | PixelMode::Cmyk => resize_nearest::<8>(bytes, &mut pixels, image.width, &xs, &ys),
            _ => unreachable!(),
        }
        return output(image, size, pixels);
    }
    let maximum = f64::from((1_u32 << image.bit_depth) - 1);
    let horizontal = weights(image.width, size.0, bounds[0], bounds[2], method);
    let vertical = weights(image.height, size.1, bounds[1], bounds[3], method);
    let first = vertical.get(0).0;
    let (last, tail) = vertical.get(size.1 as usize - 1);
    let rows = last + tail.len() - first;
    let stride = size.0 as usize * channels;
    let length = rows.checked_mul(stride).ok_or_else(|| PyMemoryError::new_err("image dimensions are too large"))?;
    let mut temp = Vec::new();
    temp.try_reserve_exact(length).map_err(|_| PyMemoryError::new_err("cannot allocate image"))?;
    temp.resize(length, 0.0);
    let sample = |i: usize| f64::from(u16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]));
    chunks_mut_above(&mut temp, stride * 8, MIN_PARALLEL_BYTES, |band, rows| {
        let four = if C == 1 { rows.len() / (stride * 4) * 4 } else { 0 };
        for group in 0..four / 4 {
            let y = first + band * 8 + group * 4;
            let source = std::array::from_fn::<_, 4, _>(|i| &bytes[(y + i) * image.width as usize * 2..][..image.width as usize * 2]);
            for (x, (start, coefficients)) in horizontal.iter().enumerate() {
                let values = crate::ops_simd::horizontal_wide_four(source.map(|row| &row[start * 2..]), coefficients);
                for (i, value) in values.into_iter().enumerate() {
                    rows[(group * 4 + i) * stride + x] = value;
                }
            }
        }
        for (y, row) in rows.chunks_exact_mut(stride).enumerate().skip(four) {
            let y = band * 8 + y;
            for (x, (start, coefficients)) in horizontal.iter().enumerate() {
                if C > 1 {
                    let offset = ((y + first) * image.width as usize + start) * C * 2;
                    let values = crate::ops_simd::horizontal_wide::<C>(&bytes[offset..], coefficients, maximum);
                    row[x * C..][..C].copy_from_slice(&values);
                    continue;
                }
                for channel in 0..channels {
                    let value = coefficients
                        .iter()
                        .enumerate()
                        .map(|(i, &weight)| {
                            let offset = ((y + first) * image.width as usize + start + i) * channels;
                            let value = sample(offset + channel);
                            let value = if channels == 4 && channel < 3 { value * sample(offset + 3) / maximum } else { value };
                            value * f64::from(weight) / f64::from(RESAMPLE_SCALE)
                        })
                        .sum::<f64>();
                    row[x * channels + channel] = value;
                }
            }
        }
    });
    chunks_mut(&mut pixels, stride * 2 * 8, |band, rows| {
        let mut values = vec![0.0; stride];
        for (y, row) in rows.chunks_exact_mut(stride * 2).enumerate() {
            let (start, coefficients) = vertical.get(band * 8 + y);
            let source = &temp[(start - first) * stride..];
            let done = crate::ops_simd::vertical_wide(source, &mut values, stride, coefficients, maximum);
            for (x, value) in values.iter_mut().enumerate().skip(done) {
                *value = coefficients
                    .iter()
                    .enumerate()
                    .map(|(i, &weight)| source[i * stride + x] * f64::from(weight) / f64::from(RESAMPLE_SCALE))
                    .sum::<f64>()
                    .clamp(0.0, maximum);
            }
            for x in 0..size.0 as usize {
                let values = &values[x * channels..][..channels];
                for channel in 0..channels {
                    let value = if channels == 4 && channel < 3 {
                        if values[3] > 0.0 { values[channel] * maximum / values[3] } else { 0.0 }
                    } else {
                        values[channel]
                    };
                    let offset = (x * channels + channel) * 2;
                    row[offset..offset + 2].copy_from_slice(&(value.round().clamp(0.0, maximum) as u16).to_le_bytes());
                }
            }
        }
    });
    output(image, size, pixels)
}

fn resize_nearest<const C: usize>(source: &[u8], output: &mut [u8], width: u32, xs: &[usize], ys: &[usize]) {
    let row_bytes = xs.len() * C;
    let half_width = xs.len() * 2 == width as usize && xs.iter().enumerate().all(|(x, &sx)| sx == x * 2 + 1);
    let double_rgb = C == 3 && xs.len() == width as usize * 2 && xs.iter().enumerate().all(|(x, &sx)| sx == x / 2);
    let map = (C == 3 && !half_width && !double_rgb && row_bytes >= 16 && ys.len() >= 4).then(|| {
        let columns: Vec<_> = xs.iter().flat_map(|&x| (0..C).map(move |c| Some(x * C + c))).collect();
        crate::ops_simd::NearestL::new(&columns, width as usize * C).expect("validated resize coordinates")
    });
    let source = source.as_chunks::<C>().0;
    chunks_mut_above(output, row_bytes << 4, 1024 << 10, |band, rows| {
        for i in 0..rows.len() / row_bytes {
            let sy = ys[band * 16 + i];
            if i > 0 && sy == ys[band * 16 + i - 1] {
                rows.copy_within((i - 1) * row_bytes..i * row_bytes, i * row_bytes);
                continue;
            }
            let src = &source[sy * width as usize..(sy + 1) * width as usize];
            let row = &mut rows[i * row_bytes..(i + 1) * row_bytes];
            let dst = row.as_chunks_mut::<C>().0;
            if double_rgb {
                crate::ops_simd::nearest_double_rgb(src.as_flattened(), dst.as_flattened_mut());
            } else if half_width {
                // Contiguous pairs expose the common 2:1 gather to SIMD.
                crate::ops_simd::nearest_half::<C>(src.as_flattened(), dst.as_flattened_mut());
            } else if let Some(map) = &map {
                map.sample(src.as_flattened(), dst.as_flattened_mut());
            } else {
                for (dst, &sx) in dst.iter_mut().zip(xs) {
                    *dst = src[sx];
                }
            }
        }
    });
}

fn resample<const C: usize>(source: &[u8], output: &mut [u8], image: &Image, size: (u32, u32), b: [f64; 4], method: u8) -> PyResult<()> {
    if method == 4
        && u64::from(size.0) * 2 == u64::from(image.width)
        && u64::from(size.1) * 2 == u64::from(image.height)
        && b == [0.0, 0.0, f64::from(image.width), f64::from(image.height)]
    {
        let stride = image.width as usize * C;
        let row_bytes = size.0 as usize * C;
        chunks_mut(output, row_bytes * 16, |band, rows| {
            for (i, row) in rows.chunks_exact_mut(row_bytes).enumerate() {
                let start = (band * 16 + i) * 2 * stride;
                crate::ops_simd::box_half::<C>(&source[start..start + stride], &source[start + stride..start + 2 * stride], row);
            }
        });
        return Ok(());
    }
    if size.1 == image.height && b[1] == 0.0 && b[3] == f64::from(image.height) {
        let horizontal = weights(image.width, size.0, b[0], b[2], method);
        resample_horizontal::<C>(source, output, image.width as usize, 0, &horizontal);
        return Ok(());
    }
    let vertical = weights(image.height, size.1, b[1], b[3], method);
    let first_row = vertical.get(0).0;
    let (last_start, last_weights) = vertical.get(size.1 as usize - 1);
    let last_row = last_start + last_weights.len();
    let row_bytes = size.0 as usize * C;
    let input_stride = image.width as usize * C;
    let horizontal_identity = size.0 == image.width && b[0] == 0.0 && b[2] == f64::from(image.width);
    let temp = if horizontal_identity {
        Cow::Borrowed(&source[first_row * input_stride..last_row * input_stride])
    } else {
        let horizontal = weights(image.width, size.0, b[0], b[2], method);
        let mut temp = buffer((size.0, (last_row - first_row) as u32), C)?;
        resample_horizontal::<C>(source, &mut temp, image.width as usize, first_row, &horizontal);
        Cow::Owned(temp)
    };
    chunks_mut(output, row_bytes << 4, |band, rows| {
        for (i, dst) in rows.chunks_exact_mut(row_bytes).enumerate() {
            let (start, coefficients) = vertical.get(band * 16 + i);
            let src = &temp[(start - first_row) * row_bytes..];
            let vectorized = crate::ops_simd::vertical(src, dst, row_bytes, coefficients);
            for (x, out) in dst.iter_mut().enumerate().skip(vectorized) {
                let sum = coefficients
                    .iter()
                    .enumerate()
                    .fold(i64::from(RESAMPLE_ROUND), |sum, (i, &w)| sum + i64::from(src[i * row_bytes + x]) * i64::from(w));
                *out = (sum >> RESAMPLE_BITS).clamp(0, 255) as u8;
            }
        }
    });
    Ok(())
}

fn resample_horizontal<const C: usize>(source: &[u8], output: &mut [u8], width: usize, first_row: usize, horizontal: &ResampleWeights) {
    let row_bytes = horizontal.bounds.len() * C;
    let input_stride = width * C;
    let narrow = horizontal.iter().all(|(_, coefficients)| {
        coefficients.iter().map(|&w| i64::from(w).abs()).sum::<i64>() * 255 + i64::from(RESAMPLE_ROUND) <= i64::from(i32::MAX)
    });
    if narrow && width == horizontal.bounds.len() * 2 {
        let blocks: Vec<_> = (0..horizontal.bounds.len() / 16)
            .map(|block| {
                let x = block * 16;
                let (start, coefficients) = horizontal.get(x);
                (start + 31 + coefficients.len() <= width
                    && (1..16).all(|i| {
                        let (next, weights) = horizontal.get(x + i);
                        next == start + i * 2 && weights == coefficients
                    }))
                .then_some((start, coefficients))
            })
            .collect();
        if blocks.iter().any(Option::is_some) {
            chunks_mut(output, row_bytes * 16, |band, rows| {
                for (i, dst) in rows.chunks_exact_mut(row_bytes).enumerate() {
                    let y = first_row + band * 16 + i;
                    let src = &source[y * input_stride..(y + 1) * input_stride];
                    for (block, out) in dst.chunks_mut(16 * C).enumerate() {
                        if let Some(&(start, coefficients)) = blocks.get(block).and_then(Option::as_ref) {
                            crate::ops_simd::horizontal_half::<C>(&src[start * C..], out, coefficients);
                        } else {
                            for (i, pixel) in out.as_chunks_mut::<C>().0.iter_mut().enumerate() {
                                let (start, coefficients) = horizontal.get(block * 16 + i);
                                let mut sums = [RESAMPLE_ROUND; C];
                                for (input, &weight) in src[start * C..].as_chunks::<C>().0.iter().zip(coefficients) {
                                    for c in 0..C {
                                        sums[c] += i32::from(input[c]) * weight;
                                    }
                                }
                                for c in 0..C {
                                    pixel[c] = (sums[c] >> RESAMPLE_BITS).clamp(0, 255) as u8;
                                }
                            }
                        }
                    }
                }
            });
            return;
        }
    }
    let batch_rows = narrow && (matches!(C, 2 | 3) || horizontal.iter().all(|(_, coefficients)| coefficients.len() <= 8));
    if batch_rows && output.len() >= row_bytes * 4 {
        resample_horizontal_batched::<C>(source, output, width, first_row, horizontal);
        return;
    }
    chunks_mut(output, row_bytes << 4, |band, rows| {
        for (i, dst) in rows.chunks_exact_mut(row_bytes).enumerate() {
            let y = first_row + band * 16 + i;
            let src = source[y * input_stride..(y + 1) * input_stride].as_chunks::<C>().0;
            for ((start, coefficients), dst) in horizontal.iter().zip(dst.as_chunks_mut::<C>().0) {
                if narrow {
                    let mut sums = [RESAMPLE_ROUND; C];
                    for (pixel, &weight) in src[start..start + coefficients.len()].iter().zip(coefficients) {
                        for c in 0..C {
                            sums[c] += i32::from(pixel[c]) * weight;
                        }
                    }
                    for c in 0..C {
                        dst[c] = (sums[c] >> RESAMPLE_BITS).clamp(0, 255) as u8;
                    }
                } else {
                    for c in 0..C {
                        let sum = coefficients
                            .iter()
                            .enumerate()
                            .fold(i64::from(RESAMPLE_ROUND), |sum, (i, &w)| sum + i64::from(src[start + i][c]) * i64::from(w));
                        dst[c] = (sum >> RESAMPLE_BITS).clamp(0, 255) as u8;
                    }
                }
            }
        }
    });
}

fn resample_horizontal_batched<const C: usize>(source: &[u8], output: &mut [u8], width: usize, first_row: usize, horizontal: &ResampleWeights) {
    let row_bytes = horizontal.bounds.len() * C;
    let input_stride = width * C;
    chunks_mut(output, row_bytes * 16, |band, rows| {
        let four = rows.len() / (row_bytes * 4) * 4;
        for group in 0..four / 4 {
            let y = first_row + band * 16 + group * 4;
            let src = std::array::from_fn::<_, 4, _>(|row| source[(y + row) * input_stride..(y + row + 1) * input_stride].as_chunks::<C>().0);
            for (x, (start, coefficients)) in horizontal.iter().enumerate() {
                let values = crate::ops_simd::horizontal_four::<C>(src.map(|row| &row[start..start + coefficients.len()]), coefficients);
                for (row, value) in values.iter().enumerate() {
                    let start = (group * 4 + row) * row_bytes + x * C;
                    rows[start..start + C].copy_from_slice(value);
                }
            }
        }
        for (i, dst) in rows.chunks_exact_mut(row_bytes).enumerate().skip(four) {
            let y = first_row + band * 16 + i;
            let src = source[y * input_stride..(y + 1) * input_stride].as_chunks::<C>().0;
            for ((start, coefficients), dst) in horizontal.iter().zip(dst.as_chunks_mut::<C>().0) {
                let mut sums = [RESAMPLE_ROUND; C];
                for (pixel, &weight) in src[start..start + coefficients.len()].iter().zip(coefficients) {
                    for c in 0..C {
                        sums[c] += i32::from(pixel[c]) * weight;
                    }
                }
                for c in 0..C {
                    dst[c] = (sums[c] >> RESAMPLE_BITS).clamp(0, 255) as u8;
                }
            }
        }
    });
}

fn cubic(v: [f64; 4], t: f64) -> f64 {
    let a = -v[0] + v[1] - v[2] + v[3];
    let b = 2.0 * v[0] - 2.0 * v[1] + v[2] - v[3];
    let c = -v[0] + v[2];
    ((a * t + b) * t + c) * t + v[1]
}

fn interpolate(a: f64, b: f64, t: f64) -> f64 {
    // C compilers contract this expression on ARM; preserve that rounding.
    if cfg!(target_arch = "aarch64") { (b - a).mul_add(t, a) } else { a + (b - a) * t }
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
    match c {
        2 => return crate::ops_simd::sample_color::<2>(source, dst, (w, h), (ix, iy), (tx, ty), method),
        3 => return crate::ops_simd::sample_color::<3>(source, dst, (w, h), (ix, iy), (tx, ty), method),
        4 => return crate::ops_simd::sample_color::<4>(source, dst, (w, h), (ix, iy), (tx, ty), method),
        _ => (),
    }
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

fn sample_float(source: &[u8], width: usize, height: usize, x: f64, y: f64, method: u8) -> f32 {
    if !x.is_finite() || !y.is_finite() || x < 0.0 || y < 0.0 || x >= width as f64 || y >= height as f64 {
        return 0.0;
    }
    let get = |x: i64, y: i64| {
        let x = x.clamp(0, width as i64 - 1) as usize;
        let y = y.clamp(0, height as i64 - 1) as usize;
        let start = (y * width + x) << 2;
        f64::from(f32::from_le_bytes(source[start..start + 4].try_into().unwrap()))
    };
    if method == 0 {
        return get(x as i64, y as i64) as f32;
    }
    let (x, y) = (x - 0.5, y - 0.5);
    let (ix, iy) = (x.floor() as i64, y.floor() as i64);
    let (tx, ty) = (x - x.floor(), y - y.floor());
    if method == 2 {
        let top = interpolate(get(ix, iy), get(ix + 1, iy), tx);
        let bottom = interpolate(get(ix, iy + 1), get(ix + 1, iy + 1), tx);
        interpolate(top, bottom, ty) as f32
    } else {
        cubic([-1, 0, 1, 2].map(|dy| cubic([-1, 0, 1, 2].map(|dx| get(ix + dx, iy + dy)), tx)), ty) as f32
    }
}

fn affine_coordinate(a: f64, b: f64, c: f64, x: f64, y: f64) -> f64 {
    // Match the contraction order used by Pillow's ARM affine mapper.
    if cfg!(target_arch = "aarch64") { a.mul_add(x, b * y) + c } else { a * x + b * y + c }
}

fn quad_coordinate(origin: f64, a: f64, b: f64, c: f64, u: f64, v: f64) -> f64 {
    if cfg!(target_arch = "aarch64") { (c * u).mul_add(v, b.mul_add(v, a.mul_add(u, origin))) } else { origin + a * u + b * v + c * u * v }
}

fn nearest_columns<const C: usize>(source: Option<&[u8]>, row: &mut [u8], columns: &[Option<usize>], fill: &[u8]) {
    let fill: [u8; C] = fill.try_into().unwrap();
    for (dst, column) in row.as_chunks_mut::<C>().0.iter_mut().zip(columns) {
        *dst = if let (Some(src), Some(offset)) = (source, column) { src[*offset..*offset + C].try_into().unwrap() } else { fill };
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

struct NearestWarp<'a> {
    size: (u32, u32),
    q: &'a [f64; 8],
    coefficients: [f64; 6],
    perspective: bool,
    fill: Option<&'a [u8]>,
    preserve_invalid: bool,
}

fn warp_nearest_row<const C: usize>(source: &[u8], row: &mut [u8], warp: &NearestWarp<'_>, v: f64) {
    let source = source.as_chunks::<C>().0;
    let fill = warp.fill.map(|pixel| pixel.try_into().unwrap()).unwrap_or([0; C]);
    let [ax, bx, cx, ay, by, cy] = warp.coefficients;
    let width = warp.size.0 as f64;
    let height = warp.size.1 as f64;
    for (x, dst) in row.as_chunks_mut::<C>().0.iter_mut().enumerate() {
        let u = x as f64 + 0.5;
        let (sx, sy) = if warp.perspective {
            let divisor = affine_coordinate(warp.q[6], warp.q[7], 1.0, u, v);
            (affine_coordinate(warp.q[0], warp.q[1], warp.q[2], u, v) / divisor, affine_coordinate(warp.q[3], warp.q[4], warp.q[5], u, v) / divisor)
        } else {
            (quad_coordinate(warp.q[0], ax, bx, cx, u, v), quad_coordinate(warp.q[1], ay, by, cy, u, v))
        };
        if sx >= 0.0 && sy >= 0.0 && sx < width && sy < height {
            *dst = source[sy as usize * warp.size.0 as usize + sx as usize];
        } else if !warp.preserve_invalid {
            *dst = fill;
        }
    }
}

/// Reverse affine mapping, sharing the mesh interpolation and alpha kernels.
#[pyfunction(signature = (image, size, matrix, method, fill, color_string=false))]
pub fn ops_affine(
    py: Python<'_>, image: &Image, size: (u32, u32), matrix: [f64; 6], method: u8, mut fill: Vec<u8>, color_string: bool,
) -> PyResult<Image> {
    if matches!(image.mode, PixelMode::RgbF | PixelMode::RgbaF) {
        return Err(PyValueError::new_err("affine transforms do not support float color images"));
    }
    if image.mode == PixelMode::F {
        if ![0, 2, 3].contains(&method) || !matrix.iter().all(|v| v.is_finite()) || fill.len() != 4 {
            return Err(PyValueError::new_err("invalid affine transform"));
        }
        let source = image.raw_data()?;
        let mut pixels = buffer(size, 4)?;
        let [a, b, c, d, e, f] = matrix;
        py.detach(|| {
            for y in 0..size.1 as usize {
                for x in 0..size.0 as usize {
                    let sx = affine_coordinate(a, b, c, x as f64 + 0.5, y as f64 + 0.5);
                    let sy = affine_coordinate(d, e, f, x as f64 + 0.5, y as f64 + 0.5);
                    let value = if sx >= 0.0 && sy >= 0.0 && sx < image.width as f64 && sy < image.height as f64 {
                        sample_float(source, image.width as usize, image.height as usize, sx, sy, method).to_le_bytes()
                    } else {
                        fill.as_slice().try_into().unwrap()
                    };
                    pixels[(y * size.0 as usize + x) * 4..(y * size.0 as usize + x + 1) * 4].copy_from_slice(&value);
                }
            }
        });
        return output(image, size, pixels);
    }
    let source = image.pixel_data()?;
    let channels = image.mode.channels();
    if ![0, 2, 3].contains(&method) || !matrix.iter().all(|v| v.is_finite()) || fill.len() != channels {
        return Err(PyValueError::new_err("invalid affine transform"));
    }
    if method != 0 && color_string {
        premultiply_image(image.mode, &mut fill);
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
        if method != 0 && matches!(image.mode, PixelMode::La | PixelMode::Rgba) {
            premultiply_image(image.mode, source.to_mut());
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
                if channels == 3 {
                    let columns: Vec<_> = columns.iter().flat_map(|x| (0..3).map(move |c| x.map(|x| x + c))).collect();
                    let map = crate::ops_simd::NearestL::new(&columns, image.width as usize * 3).expect("validated byte coordinates");
                    chunks_mut_above(&mut pixels, row_bytes * 16, 1024 << 10, |band, rows| {
                        for (i, row) in rows.chunks_exact_mut(row_bytes).enumerate() {
                            let start = row_origins[band * 16 + i].1 as usize * image.width as usize * 3;
                            map.sample(&source[start..start + image.width as usize * 3], row);
                        }
                    });
                    return;
                }
                let xs: Vec<_> = columns.iter().map(|x| x.unwrap() / channels).collect();
                let ys: Vec<_> = row_origins.iter().map(|&(_, y)| y as usize).collect();
                match image.mode {
                    PixelMode::Rgb | PixelMode::Hsv | PixelMode::YCbCr | PixelMode::Lab => {
                        resize_nearest::<3>(&source, &mut pixels, image.width, &xs, &ys)
                    }
                    PixelMode::Rgba | PixelMode::Cmyk => resize_nearest::<4>(&source, &mut pixels, image.width, &xs, &ys),
                    PixelMode::One | PixelMode::L => unreachable!(),
                    PixelMode::La | PixelMode::Pa => resize_nearest::<2>(&source, &mut pixels, image.width, &xs, &ys),
                    _ => unreachable!(),
                }
                return;
            }
        }
        let nearest_l = columns.as_ref().filter(|_| channels == 1).and_then(|columns| crate::ops_simd::NearestL::new(columns, image.width as usize));
        // Table gathers are memory work: small outputs cost less than a Rayon
        // dispatch. Other transforms still benefit from finer scheduling.
        let threshold = if nearest_l.as_ref().is_some_and(|map| map.is_vectorized()) { MIN_PARALLEL_BYTES } else { 32 << 10 };
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
                        PixelMode::One | PixelMode::L => nearest_columns::<1>(source_row, row, columns, &fill),
                        PixelMode::La | PixelMode::Pa => nearest_columns::<2>(source_row, row, columns, &fill),
                        PixelMode::Rgb | PixelMode::Hsv | PixelMode::YCbCr | PixelMode::Lab => nearest_columns::<3>(source_row, row, columns, &fill),
                        PixelMode::Rgba | PixelMode::Cmyk => nearest_columns::<4>(source_row, row, columns, &fill),
                        _ => unreachable!(),
                    }
                    continue;
                }
                if fixed {
                    let [a, b, c, d, e, f] = coefficients;
                    let origin = (b * y as i64 + c, e * y as i64 + f);
                    let steps = (a, d);
                    let size = (image.width, image.height);
                    match image.mode {
                        PixelMode::One | PixelMode::L => nearest_fixed::<1>(&source, row, size, origin, steps, &fill),
                        PixelMode::La | PixelMode::Pa => nearest_fixed::<2>(&source, row, size, origin, steps, &fill),
                        PixelMode::Rgb | PixelMode::Hsv | PixelMode::YCbCr | PixelMode::Lab => {
                            nearest_fixed::<3>(&source, row, size, origin, steps, &fill)
                        }
                        PixelMode::Rgba | PixelMode::Cmyk => nearest_fixed::<4>(&source, row, size, origin, steps, &fill),
                        _ => unreachable!(),
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
                        (affine_coordinate(a, b, c, x as f64 + 0.5, y + 0.5), affine_coordinate(d, e, f, x as f64 + 0.5, y + 0.5))
                    };
                    if sx >= 0.0 && sy >= 0.0 && sx < image.width as f64 && sy < image.height as f64 {
                        sample(&source, image, sx, sy, method, dst);
                    } else {
                        dst.copy_from_slice(&fill);
                    }
                }
            }
        });
        if method != 0 {
            unpremultiply_image(image.mode, &mut pixels);
        }
    });
    output(image, size, pixels)
}

#[pyfunction]
fn ops_mesh(py: Python<'_>, image: &Image, mesh: Mesh, method: u8) -> PyResult<Image> {
    ops_warp(py, image, (image.width, image.height), mesh, (method, false), None, false)
}

#[pyfunction(signature = (image, size, mesh, filters, fill, color_string=false))]
fn ops_warp(
    py: Python<'_>, image: &Image, size: (u32, u32), mesh: Mesh, filters: (u8, bool), mut fill: Option<Vec<u8>>, color_string: bool,
) -> PyResult<Image> {
    if matches!(image.mode, PixelMode::RgbF | PixelMode::RgbaF) {
        return Err(PyValueError::new_err("warps do not support float color images"));
    }
    let (method, perspective) = filters;
    if image.mode == PixelMode::F {
        if ![0, 2, 3].contains(&method) || fill.as_ref().is_some_and(|v| v.len() != 4) || mesh.iter().any(|(_, q)| q.iter().any(|v| !v.is_finite())) {
            return Err(PyValueError::new_err("invalid warp data"));
        }
        let source = image.raw_data()?;
        let mut pixels = buffer(size, 4)?;
        py.detach(|| {
            if let Some(fill) = &fill {
                for pixel in pixels.as_chunks_mut::<4>().0 {
                    pixel.copy_from_slice(fill);
                }
            }
            for ((left, top, right, bottom), q) in mesh {
                if right <= left || bottom <= top {
                    continue;
                }
                let w = (right - left) as f64;
                let h = (bottom - top) as f64;
                let ax = (q[6] - q[0]) / w;
                let bx = (q[2] - q[0]) / h;
                let cx = (q[4] - q[2] - q[6] + q[0]) / (w * h);
                let ay = (q[7] - q[1]) / w;
                let by = (q[3] - q[1]) / h;
                let cy = (q[5] - q[3] - q[7] + q[1]) / (w * h);
                for y in top.max(0)..bottom.min(i64::from(size.1)) {
                    for x in left.max(0)..right.min(i64::from(size.0)) {
                        let u = (x - left) as f64 + 0.5;
                        let v = (y - top) as f64 + 0.5;
                        let (sx, sy) = if perspective {
                            let divisor = affine_coordinate(q[6], q[7], 1.0, u, v);
                            (affine_coordinate(q[0], q[1], q[2], u, v) / divisor, affine_coordinate(q[3], q[4], q[5], u, v) / divisor)
                        } else {
                            (quad_coordinate(q[0], ax, bx, cx, u, v), quad_coordinate(q[1], ay, by, cy, u, v))
                        };
                        if fill.is_none() || (sx >= 0.0 && sy >= 0.0 && sx < image.width as f64 && sy < image.height as f64) {
                            let value = sample_float(source, image.width as usize, image.height as usize, sx, sy, method);
                            let start = (y as usize * size.0 as usize + x as usize) * 4;
                            pixels[start..start + 4].copy_from_slice(&value.to_le_bytes());
                        }
                    }
                }
            }
        });
        return output(image, size, pixels);
    }
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
    if method != 0
        && color_string
        && let Some(fill) = &mut fill
    {
        premultiply_image(image.mode, fill);
    }
    let full_nearest = method == 0
        && mesh.len() == 1
        && mesh[0].0.0 <= 0
        && mesh[0].0.1 <= 0
        && mesh[0].0.2 >= i64::from(size.0)
        && mesh[0].0.3 >= i64::from(size.1);
    py.detach(|| {
        if let Some(fill) = &fill
            && !full_nearest
        {
            for pixel in result.chunks_exact_mut(c) {
                pixel.copy_from_slice(fill);
            }
        }
        let mut source = Cow::Borrowed(source);
        if method != 0 && matches!(image.mode, PixelMode::La | PixelMode::Rgba) {
            premultiply_image(image.mode, source.to_mut());
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
            let start = left.max(0) as usize;
            let end = right.min(i64::from(size.0)) as usize;
            if start >= end {
                continue;
            }
            let warp = NearestWarp {
                size: (image.width, image.height),
                q: &q,
                coefficients: [ax, bx, cx, ay, by, cy],
                perspective,
                fill: fill.as_deref(),
                preserve_invalid: fill.is_some() && !full_nearest,
            };
            // Mesh entries retain their order (later boxes overwrite earlier
            // ones); only disjoint rows within an entry execute concurrently.
            chunks_mut_above(&mut result[first * row_bytes..last * row_bytes], row_bytes << 4, 32 << 10, |band, rows| {
                for (i, row) in rows.chunks_exact_mut(row_bytes).enumerate() {
                    let y = (first + band * 16 + i) as i64;
                    let v = (y - top.max(0)) as f64 + 0.5;
                    if method == 0 {
                        let dst = &mut row[start * c..end * c];
                        match image.mode {
                            PixelMode::One | PixelMode::L => warp_nearest_row::<1>(&source, dst, &warp, v),
                            PixelMode::La | PixelMode::Pa => warp_nearest_row::<2>(&source, dst, &warp, v),
                            PixelMode::Rgb | PixelMode::Hsv | PixelMode::YCbCr | PixelMode::Lab => warp_nearest_row::<3>(&source, dst, &warp, v),
                            PixelMode::Rgba | PixelMode::Cmyk => warp_nearest_row::<4>(&source, dst, &warp, v),
                            _ => unreachable!(),
                        }
                        continue;
                    }
                    for x in start..end {
                        let u = (x - start) as f64 + 0.5;
                        let (sx, sy) = if perspective {
                            let divisor = affine_coordinate(q[6], q[7], 1.0, u, v);
                            (affine_coordinate(q[0], q[1], q[2], u, v) / divisor, affine_coordinate(q[3], q[4], q[5], u, v) / divisor)
                        } else {
                            (quad_coordinate(q[0], ax, bx, cx, u, v), quad_coordinate(q[1], ay, by, cy, u, v))
                        };
                        let dst = x * c;
                        if fill.is_none() || (sx >= 0.0 && sy >= 0.0 && sx < image.width as f64 && sy < image.height as f64) {
                            sample(&source, image, sx, sy, method, &mut row[dst..dst + c]);
                        }
                    }
                }
            });
        }
        if method != 0 {
            unpremultiply_image(image.mode, &mut result);
        }
    });
    output(image, size, result)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn nearest_rgb_mapping_preserves_pixels_repeated_rows_tails_and_guards() {
        for (width, height, target, rows) in [(5, 7, 17, 19), (33, 7, 65, 19), (97, 7, 33, 19), (769, 7, 1153, 513)] {
            let mut source = vec![93];
            source.extend((0..width * height * 3).map(|i| (i * 137 + i / 3) as u8));
            source.push(97);
            for reverse in [false, true] {
                let xs: Vec<_> = (0..target).map(|x| if reverse { width - 1 - x * width / target } else { x * width / target }).collect();
                let ys: Vec<_> = (0..rows).map(|y| y / 3 % height).collect();
                let mut output = vec![91; target * rows * 3 + 2];
                resize_nearest::<3>(&source[1..source.len() - 1], &mut output[1..target * rows * 3 + 1], width as u32, &xs, &ys);
                for (y, row) in output[1..output.len() - 1].chunks_exact(target * 3).enumerate() {
                    for (x, pixel) in row.as_chunks::<3>().0.iter().enumerate() {
                        let start = 1 + (ys[y] * width + xs[x]) * 3;
                        assert_eq!(pixel, &source[start..start + 3]);
                    }
                }
                assert_eq!(output[0], 91);
                assert_eq!(output[output.len() - 1], 91);
            }
            assert_eq!(source[0], 93);
            assert_eq!(source[source.len() - 1], 97);
        }
    }

    #[test]
    fn cached_weights_reuse_tables_and_preserve_exact_coefficients() {
        for method in 1..=5 {
            for (input, output, start, end) in [(67, 11, 0.0, 67.0), (67, 101, 2.3, 59.2), (49, 7, 1.7, 43.8)] {
                let actual = super::weights(input, output, start, end, method);
                let cached = super::weights(input, output, start, end, method);
                assert!(std::sync::Arc::ptr_eq(&actual, &cached));
                let expected = super::compute_weights(input, output, start, end, method);
                assert_eq!(actual.bounds, expected.bounds);
                assert_eq!(actual.coefficients, expected.coefficients);
                let changed = super::weights(input, output, start + 0.125, end, method);
                assert!(!std::sync::Arc::ptr_eq(&actual, &changed));
            }
        }
    }

    #[test]
    fn reduction_crop_only_copies_requested_rows_and_columns() {
        for channels in [2, 4] {
            let source: Vec<u8> = (0..19 * 17 * channels).map(|i| (i * 37) as u8).collect();
            for bounds in [(3, 5, 8, 9), (0, 0, 19, 17), (19, 17, 19, 17)] {
                let (left, top, right, bottom) = bounds;
                let crop = reduction_crop(&source, 19, bounds, channels).unwrap();
                assert_eq!(crop.len(), (right - left) as usize * (bottom - top) as usize * channels);
                let mut expected = Vec::new();
                for y in top..bottom {
                    for x in left..right {
                        let start = (y * 19 + x) as usize * channels;
                        expected.extend_from_slice(&source[start..start + channels]);
                    }
                }
                assert_eq!(crop, expected);
            }
        }
    }

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
}
