//! Pixel primitives used by the Python ImageOps API.
use std::borrow::Cow;
use std::num::NonZeroUsize;
use std::sync::{Arc, LazyLock, Mutex};

use blanket_core::cache::ByteLru;
use blanket_core::parallel::{CHUNK_PIXELS, MIN_PARALLEL_BYTES, chunks_mut, chunks_mut_above, should_parallel};
use blanket_core::raster::{Image, PixelMode};
use pyo3::exceptions::{PyMemoryError, PyValueError};
use pyo3::prelude::*;
use rayon::prelude::*;

// Pillow resampling coefficients retain 22 fractional bits between passes.
pub(crate) const RESAMPLE_BITS: u32 = 22;
pub(crate) const RESAMPLE_SCALE: i32 = 1 << RESAMPLE_BITS;
// Half a coefficient unit rounds the accumulator before conversion to a sample.
pub(crate) const RESAMPLE_ROUND: i32 = RESAMPLE_SCALE / 2;

type BoxI = (i64, i64, i64, i64);
type BoxF = (f64, f64, f64, f64);
type Mesh = Vec<(BoxI, [f64; 8])>;

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(ops_color_resize, module)?)?;
    module.add_function(wrap_pyfunction!(ops_color_composite, module)?)?;
    module.add_function(wrap_pyfunction!(ops_lut, module)?)?;
    module.add_function(wrap_pyfunction!(ops_colorize, module)?)?;
    module.add_function(wrap_pyfunction!(ops_histogram, module)?)?;
    module.add_function(wrap_pyfunction!(ops_split, module)?)?;
    module.add_function(wrap_pyfunction!(ops_entropy, module)?)?;
    module.add_function(wrap_pyfunction!(ops_spread, module)?)?;
    module.add_function(wrap_pyfunction!(ops_remap_indices, module)?)?;
    module.add_function(wrap_pyfunction!(ops_xbm_bits, module)?)?;
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

#[pyfunction]
fn ops_color_resize(py: Python<'_>, image: &Image, size: (u32, u32), method: u8, bounds: BoxF, srgb: bool, premultiply: bool) -> PyResult<Image> {
    let prepared = py.detach(|| blanket_core::float::prepare(image, srgb, premultiply))?;
    let resized = ops_resize(py, &prepared, size, method, bounds, None)?;
    py.detach(|| blanket_core::float::finish(&resized, image, srgb, premultiply))
}

#[pyfunction]
fn ops_color_composite(py: Python<'_>, background: &Image, foreground: &Image, srgb: bool) -> PyResult<Image> {
    if background.width != foreground.width
        || background.height != foreground.height
        || background.mode != foreground.mode
        || background.bit_depth != foreground.bit_depth
        || !matches!(background.mode, PixelMode::Rgba | PixelMode::RgbaF)
    {
        return Err(PyValueError::new_err("compositing requires matching RGBA images"));
    }
    py.detach(|| {
        let back = blanket_core::float::prepare(background, srgb, true)?;
        let front = blanket_core::float::prepare(foreground, srgb, true)?;
        let mut values = blanket_core::float::values(&back)?;
        let source = blanket_core::float::values(&front)?;
        blanket_core::float::composite(&mut values, &source);
        let result = Image::from_color_float_bytes(back.width, back.height, back.mode, values.into_iter().flat_map(f32::to_le_bytes).collect())?;
        blanket_core::float::finish(&result, background, srgb, true)
    })
}

fn reserved_buffer(size: (u32, u32), channels: usize) -> PyResult<Vec<u8>> {
    let len = (size.0 as usize)
        .checked_mul(size.1 as usize)
        .and_then(|n| n.checked_mul(channels))
        .ok_or_else(|| PyValueError::new_err("image dimensions are too large"))?;
    let mut result = Vec::new();
    result.try_reserve_exact(len).map_err(|_| PyMemoryError::new_err("cannot allocate image"))?;
    Ok(result)
}

fn output(image: &Image, size: (u32, u32), pixels: Vec<u8>) -> PyResult<Image> {
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

/// Copy complete rows into a reserved, empty image without first zeroing it.
fn copy_rows<'a>(pixels: &mut Vec<u8>, row_bytes: usize, height: usize, source_row: impl Fn(usize) -> &'a [u8] + Sync) {
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

#[pyfunction]
fn ops_split(py: Python<'_>, image: &Image) -> PyResult<Vec<Image>> {
    if matches!(image.mode, PixelMode::RgbF | PixelMode::RgbaF) {
        return (0..image.mode.channels()).map(|c| image.getchannel(py, c)).collect();
    }
    if image.mode.is_wide_scalar() {
        return image.copy(py).map(|copy| vec![copy]);
    }
    if image.bit_depth > 8 {
        let source = image.raw_data()?;
        return py.detach(|| {
            (0..image.mode.channels())
                .map(|channel| {
                    let pixels = blanket_core::simd::extract_sample_channel(source, image.mode.channels(), channel);
                    Image::from_sample_bytes(image.width, image.height, PixelMode::L, pixels, image.bit_depth, None)
                })
                .collect()
        });
    }
    let source = image.pixel_data()?;
    let channels = image.mode.channels();
    if channels == 1 {
        return py.detach(|| Image::from_pixels(image.width, image.height, PixelMode::L, source.to_vec(), None).map(|band| vec![band]));
    }
    let mut bands = (0..channels).map(|_| buffer((image.width, image.height), 1)).collect::<PyResult<Vec<_>>>()?;
    py.detach(|| {
        if let [luma, alpha] = bands.as_mut_slice() {
            let split = |offset: usize, luma: &mut [u8], alpha: &mut [u8]| {
                for ((l, a), pixel) in luma.iter_mut().zip(alpha).zip(source[offset * 2..].as_chunks::<2>().0) {
                    [*l, *a] = *pixel;
                }
            };
            if should_parallel(source.len(), CHUNK_PIXELS * 2, 2 * 1024 * 1024) {
                luma.par_chunks_mut(CHUNK_PIXELS)
                    .zip(alpha.par_chunks_mut(CHUNK_PIXELS))
                    .enumerate()
                    .for_each(|(i, (l, a))| split(i * CHUNK_PIXELS, l, a));
            } else {
                split(0, luma, alpha);
            }
            return;
        }
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
        let [red, green, blue, rest @ ..] = bands.as_mut_slice() else { unreachable!() };
        if !should_parallel(source.len(), CHUNK_PIXELS * channels, 2 * 1024 * 1024) {
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
    bands.into_iter().map(|pixels| Image::from_pixels(image.width, image.height, PixelMode::L, pixels, None)).collect()
}

#[pyfunction(signature = (image, mask=None, extrema=None))]
fn ops_entropy(py: Python<'_>, image: &Image, mask: Option<&Image>, extrema: Option<(f64, f64)>) -> PyResult<f64> {
    let bins = if image.bit_depth == 8 && !image.mode.is_wide_scalar() {
        ops_histogram(py, image, mask)?
    } else {
        wide_histogram(py, image, mask, extrema)?
    };
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

fn validate_mask<'a>(mask: Option<&'a Image>, image: &Image) -> PyResult<Option<&'a [u8]>> {
    mask.map(|mask| {
        if mask.mode != PixelMode::L || mask.bit_depth != 8 {
            return Err(PyValueError::new_err("bad transparency mask"));
        }
        if (mask.width, mask.height) != (image.width, image.height) {
            return Err(PyValueError::new_err("images do not match"));
        }
        mask.pixel_data()
    })
    .transpose()
}

fn wide_histogram(py: Python<'_>, image: &Image, mask: Option<&Image>, extrema: Option<(f64, f64)>) -> PyResult<Vec<u64>> {
    let raw = image.raw_data()?;
    let mask_data = validate_mask(mask, image)?;
    let channels = image.mode.channels();
    py.detach(|| match image.mode {
        PixelMode::I => sample_histogram(raw, channels, mask_data, extrema, |v| f64::from(i32::from_le_bytes(*v))),
        PixelMode::F => sample_histogram(raw, channels, mask_data, extrema, |v| f64::from(f32::from_le_bytes(*v))),
        PixelMode::I16B => sample_histogram(raw, channels, mask_data, extrema, |v| f64::from(u16::from_be_bytes(*v))),
        _ => sample_histogram(raw, channels, mask_data, extrema, |v| f64::from(u16::from_le_bytes(*v))),
    })
}

fn sample_histogram<const N: usize>(
    raw: &[u8], channels: usize, mask: Option<&[u8]>, extrema: Option<(f64, f64)>, sample: impl Fn(&[u8; N]) -> f64,
) -> PyResult<Vec<u64>> {
    let samples = raw.as_chunks::<N>().0;
    let mut bins = vec![0_u64; channels << 8];
    if samples.is_empty() {
        return Ok(bins);
    }
    let (low, high) = extrema.unwrap_or_else(|| {
        samples.iter().map(&sample).filter(|v| v.is_finite()).fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), v| (lo.min(v), hi.max(v)))
    });
    if !low.is_finite() || !high.is_finite() || high < low {
        return Err(PyValueError::new_err("invalid extrema"));
    }
    if high == low {
        return Ok(bins);
    }
    for (pixel_index, pixel) in samples.chunks_exact(channels).enumerate() {
        if mask.is_some_and(|mask| mask[pixel_index] == 0) {
            continue;
        }
        for (channel, bytes) in pixel.iter().enumerate() {
            let value = sample(bytes);
            if value.is_finite() {
                let slot = ((value - low) * 255.0 / (high - low)).trunc() as i64;
                if (0..256).contains(&slot) {
                    bins[channel * 256 + slot as usize] += 1;
                }
            }
        }
    }
    Ok(bins)
}

#[pyfunction]
fn ops_spread(py: Python<'_>, image: &Image, distance: i64) -> PyResult<Image> {
    let source = image.raw_data()?;
    let channels = image.raw_bytes_per_pixel();
    let distance = distance.unsigned_abs().min(u64::from(image.width.max(image.height))) as i64;
    if distance == 0 || image.width == 0 || image.height == 0 {
        return output(image, (image.width, image.height), py.detach(|| source.to_vec()));
    }
    let mut state = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_nanos() as u64 | 1;
    let mut pixels = reserved_buffer((image.width, image.height), channels)?;
    py.detach(|| {
        let spare = &mut pixels.spare_capacity_mut()[..source.len()];
        match channels {
            1 => spread_pixels::<1>(source, spare, image.width as usize, distance, &mut state),
            2 => spread_pixels::<2>(source, spare, image.width as usize, distance, &mut state),
            3 => spread_pixels::<3>(source, spare, image.width as usize, distance, &mut state),
            4 => spread_pixels::<4>(source, spare, image.width as usize, distance, &mut state),
            6 => spread_pixels::<6>(source, spare, image.width as usize, distance, &mut state),
            8 => spread_pixels::<8>(source, spare, image.width as usize, distance, &mut state),
            _ => unreachable!("validated pixel layout"),
        }
    });
    // SAFETY: the spread kernel initializes every reserved pixel above.
    unsafe { pixels.set_len(source.len()) };
    output(image, (image.width, image.height), pixels)
}

fn spread_pixels<const C: usize>(source: &[u8], output: &mut [std::mem::MaybeUninit<u8>], width: usize, distance: i64, state: &mut u64) {
    let source = source.as_chunks::<C>().0;
    let height = source.len() / width;
    let mut random = || {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    };
    for (y, row) in output.chunks_exact_mut(width * C).enumerate() {
        for (x, pixel) in row.as_chunks_mut::<C>().0.iter_mut().enumerate() {
            let dx = (random() % (2 * distance as u64 + 1)) as i64 - distance;
            let dy = (random() % (2 * distance as u64 + 1)) as i64 - distance;
            let sx = (x as i64 + dx).clamp(0, width as i64 - 1) as usize;
            let sy = (y as i64 + dy).clamp(0, height as i64 - 1) as usize;
            *pixel = source[sy * width + sx].map(std::mem::MaybeUninit::new);
        }
    }
}

#[pyfunction]
fn ops_remap_indices(py: Python<'_>, image: &Image, mapping: Vec<u8>) -> PyResult<Image> {
    py.detach(|| remap_indices(image, &mapping))
}

fn remap_indices(image: &Image, mapping: &[u8]) -> PyResult<Image> {
    if image.mode != PixelMode::L || mapping.len() != 256 {
        return Err(PyValueError::new_err("invalid palette remapping"));
    }
    let source = image.pixel_data()?;
    let mut pixels = buffer((image.width, image.height), 1)?;
    apply_lut::<1>(source, &mut pixels, mapping);
    Image::from_pixels(image.width, image.height, PixelMode::L, pixels, None)
}

#[pyfunction]
fn ops_xbm_bits(image: &Image) -> PyResult<Vec<u8>> {
    if image.mode != PixelMode::One {
        return Err(PyValueError::new_err("not a bitmap"));
    }
    let source = image.pixel_data()?;
    let width = image.width as usize;
    let row_bytes = width.div_ceil(8);
    let mut result = vec![0_u8; row_bytes * image.height as usize];
    for (y, row) in source.chunks(width.max(1)).enumerate() {
        for (x, &value) in row.iter().enumerate() {
            if value != 0 {
                result[y * row_bytes + x / 8] |= 1 << (x % 8);
            }
        }
    }
    Ok(result)
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
        PixelMode::One | PixelMode::L => apply_lut::<1>(source, &mut pixels, &lut),
        PixelMode::La | PixelMode::Pa => apply_lut::<2>(source, &mut pixels, &lut),
        PixelMode::Rgb | PixelMode::Hsv | PixelMode::YCbCr | PixelMode::Lab => apply_lut::<3>(source, &mut pixels, &lut),
        PixelMode::Rgba | PixelMode::Cmyk => apply_lut::<4>(source, &mut pixels, &lut),
        _ => unreachable!(),
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
pub fn ops_histogram(py: Python<'_>, image: &Image, mask: Option<&Image>) -> PyResult<Vec<u64>> {
    let source = image.pixel_data()?;
    let mask = validate_mask(mask, image)?;
    Ok(py.detach(|| match image.mode {
        PixelMode::One | PixelMode::L => histogram::<1>(source, mask),
        PixelMode::La | PixelMode::Pa => histogram::<2>(source, mask),
        PixelMode::Rgb | PixelMode::Hsv | PixelMode::YCbCr | PixelMode::Lab => histogram::<3>(source, mask),
        PixelMode::Rgba | PixelMode::Cmyk => histogram::<4>(source, mask),
        _ => unreachable!(),
    }))
}

fn histogram<const C: usize>(source: &[u8], mask: Option<&[u8]>) -> Vec<u64> {
    let count = |start: usize, bytes: &[u8]| {
        let mut bins = vec![0_u64; C << 8];
        let pixels = bytes.as_chunks::<C>().0;
        // Hoist the mask decision out of the per-pixel loop.
        match mask {
            None => {
                if let Some(pixel) = crate::ops_simd::constant_pixel::<C>(bytes) {
                    for c in 0..C {
                        bins[c * 256 + usize::from(pixel[c])] = pixels.len() as u64;
                    }
                    return bins;
                }
                for pixel in pixels {
                    for c in 0..C {
                        bins[c * 256 + usize::from(pixel[c])] += 1;
                    }
                }
            }
            Some(mask) => {
                for (pixel, &selected) in pixels.iter().zip(&mask[start..]) {
                    if selected != 0 {
                        for c in 0..C {
                            bins[c * 256 + usize::from(pixel[c])] += 1;
                        }
                    }
                }
            }
        }
        bins
    };
    if !should_parallel(source.len(), CHUNK_PIXELS * C, MIN_PARALLEL_BYTES) {
        return count(0, source);
    }
    source.par_chunks(CHUNK_PIXELS * C).enumerate().map(|(i, bytes)| count(i * CHUNK_PIXELS, bytes)).reduce(
        || vec![0; C << 8],
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
    let source = image.raw_data()?;
    if fill.len() != image.mode.channels() * if image.mode.is_float() { 4 } else { 1 } {
        return Err(PyValueError::new_err("invalid fill color"));
    }
    let fill = if image.mode.is_float() {
        fill
    } else if image.mode.is_integer() {
        match image.mode {
            PixelMode::I => i32::from(fill[0]).to_le_bytes().to_vec(),
            PixelMode::I16B => u16::from(fill[0]).to_be_bytes().to_vec(),
            _ => u16::from(fill[0]).to_le_bytes().to_vec(),
        }
    } else if image.bit_depth > 8 {
        let maximum = (1_u32 << image.bit_depth) - 1;
        fill.into_iter().flat_map(|v| (((u32::from(v) * maximum + 127) / 255) as u16).to_le_bytes()).collect()
    } else {
        fill
    };
    let channels = image.raw_bytes_per_pixel();
    if offset.0 <= 0
        && offset.1 <= 0
        && offset.0.saturating_add(i64::from(image.width)) >= i64::from(size.0)
        && offset.1.saturating_add(i64::from(image.height)) >= i64::from(size.1)
    {
        // An interior crop writes every byte directly into reserved storage.
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
                    crate::ops_simd::reverse_gray(&source[sy * width..(sy + 1) * width], row);
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
                    crate::ops_simd::reverse_rgb(src.as_flattened(), row);
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
                    let columns = crate::ops_simd::transpose_rgb_four(input);
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
        for p in chunk.as_chunks_mut::<4>().0 {
            for c in 0..3 {
                let v = u32::from(p[c]) * u32::from(p[3]) + 128;
                p[c] = (((v >> 8) + v) >> 8) as u8;
            }
        }
    });
}

fn unpremultiply(pixels: &mut [u8]) {
    chunks_mut(pixels, CHUNK_PIXELS << 2, |_, chunk| {
        for p in chunk.as_chunks_mut::<4>().0 {
            if p[3] != 0 && p[3] != 255 {
                for c in 0..3 {
                    p[c] = (u32::from(p[c]) * 255 / u32::from(p[3])).min(255) as u8;
                }
            }
        }
    });
}

fn premultiply_la(pixels: &mut [u8]) {
    for pixel in pixels.as_chunks_mut::<2>().0 {
        let value = u32::from(pixel[0]) * u32::from(pixel[1]) + 128;
        pixel[0] = (((value >> 8) + value) >> 8) as u8;
    }
}

fn unpremultiply_la(pixels: &mut [u8]) {
    for pixel in pixels.as_chunks_mut::<2>().0 {
        if pixel[1] != 0 && pixel[1] != 255 {
            pixel[0] = (u32::from(pixel[0]) * 255 / u32::from(pixel[1])).min(255) as u8;
        }
    }
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
    #[test]
    fn histogram_matches_scalar_for_masks_repeated_bins_and_chunk_tails() {
        fn check<const C: usize>() {
            for count in [0, 1, 2, 3, 15, 16, 17, 65535, 65536, 65537, 262145] {
                for repeated in [false, true] {
                    let mut source = vec![93];
                    source.extend((0..count * C).map(|i| if repeated { 71 } else { (i * 137 + i / C) as u8 }));
                    source.push(97);
                    let pixels = &source[1..source.len() - 1];
                    let mask: Vec<_> = (0..count).map(|i| if i % 3 == 0 { 0 } else { 255 }).collect();
                    for mask in [None, Some(mask.as_slice())] {
                        let mut expected = vec![0_u64; C << 8];
                        for (i, pixel) in pixels.as_chunks::<C>().0.iter().enumerate() {
                            if mask.is_none_or(|mask| mask[i] != 0) {
                                for c in 0..C {
                                    expected[c * 256 + usize::from(pixel[c])] += 1;
                                }
                            }
                        }
                        assert_eq!(histogram::<C>(pixels, mask), expected);
                    }
                    assert_eq!(source[0], 93);
                    assert_eq!(source[source.len() - 1], 97);
                }
            }
        }
        for threads in [1, 4] {
            rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap().install(|| {
                check::<1>();
                check::<2>();
                check::<3>();
                check::<4>();
            });
        }
    }

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

    use super::*;

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

    #[test]
    fn grayscale_reversal_preserves_unaligned_tails_and_guards() {
        for width in [0, 1, 15, 16, 17, 63, 64, 65, 127, 128, 129] {
            for height in [1, 35] {
                for offset in 0..16 {
                    let source: Vec<u8> = (0..offset + width * height).map(|i| (i * 37 + 113) as u8).collect();
                    let mut spare = vec![std::mem::MaybeUninit::new(93); offset + width + 16];
                    crate::ops_simd::reverse_gray(&source[offset..offset + width], &mut spare[offset..offset + width]);
                    // SAFETY: the whole buffer was initialized before the call.
                    let actual: Vec<_> = spare.into_iter().map(|v| unsafe { v.assume_init() }).collect();
                    assert_eq!(&actual[offset..offset + width], source[offset..offset + width].iter().rev().copied().collect::<Vec<_>>());
                    assert!(actual[..offset].iter().all(|&v| v == 93));
                    assert!(actual[offset + width..].iter().all(|&v| v == 93));
                    for orientation in [2, 3] {
                        let mut output = vec![93; offset + width * height + 16];
                        transpose::<1>(&source[offset..], &mut output[offset..offset + width * height], width, height, orientation);
                        for y in 0..height {
                            let sy = if orientation == 3 { height - 1 - y } else { y };
                            for x in 0..width {
                                assert_eq!(output[offset + y * width + x], source[offset + sy * width + width - 1 - x]);
                            }
                        }
                        assert!(output[..offset].iter().all(|&v| v == 93));
                        assert!(output[offset + width * height..].iter().all(|&v| v == 93));
                    }
                }
            }
        }
    }

    #[test]
    fn rgb_transpose_preserves_parallel_bands_and_partial_tiles() {
        for threads in [1, 4] {
            rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap().install(|| {
                let (width, height) = (517, 263);
                let count = width * height * 3;
                let source: Vec<_> = (0..count + 1).map(|i| (i * 137 + i / 7) as u8).collect();
                for orientation in 5..=8 {
                    let mut output = vec![173; count + 2];
                    transpose::<3>(&source[1..], &mut output[1..count + 1], width, height, orientation);
                    for y in 0..width {
                        for x in 0..height {
                            let sx = if orientation == 7 || orientation == 8 { width - 1 - y } else { y };
                            let sy = if orientation == 6 || orientation == 7 { height - 1 - x } else { x };
                            let start = 1 + (sy * width + sx) * 3;
                            let dest = 1 + (y * height + x) * 3;
                            assert_eq!(&output[dest..dest + 3], &source[start..start + 3]);
                        }
                    }
                    assert_eq!((output[0], output[count + 1]), (173, 173));
                }
            });
        }
    }

    #[test]
    fn tiled_transpose_preserves_unaligned_pixels_edges_and_guards() {
        fn check<const C: usize>() {
            for (w, h) in [(1, 1), (1, 33), (33, 1), (17, 31), (32, 32), (33, 65), (67, 35)] {
                for orientation in 5..=8 {
                    for offset in 0..16 {
                        let source: Vec<u8> = (0..offset + w * h * C).map(|i| (i * 37 + 113) as u8).collect();
                        let mut output = vec![0xa5; offset + w * h * C + 16];
                        transpose::<C>(&source[offset..], &mut output[offset..offset + w * h * C], w, h, orientation);
                        assert!(output[..offset].iter().all(|&v| v == 0xa5));
                        assert!(output[offset + w * h * C..].iter().all(|&v| v == 0xa5));
                        for y in 0..w {
                            for x in 0..h {
                                let sy = if orientation == 6 || orientation == 7 { h - 1 - x } else { x };
                                let sx = if orientation == 7 || orientation == 8 { w - 1 - y } else { y };
                                let src = offset + (sy * w + sx) * C;
                                let dst = offset + (y * h + x) * C;
                                assert_eq!(output[dst..dst + C], source[src..src + C]);
                            }
                        }
                    }
                }
            }
        }
        check::<1>();
        check::<2>();
        check::<3>();
        check::<4>();
        check::<6>();
        check::<8>();
    }

    #[test]
    fn spread_keeps_pixel_values_and_zero_distance_copies() {
        Python::initialize();
        Python::attach(|py| {
            let image = Image::from_pixels(3, 2, PixelMode::Rgb, (0..18).collect(), None).unwrap();
            assert_eq!(ops_spread(py, &image, 0).unwrap().pixel_data().unwrap(), image.pixel_data().unwrap());
            let spread = ops_spread(py, &image, 2).unwrap();
            assert_eq!((spread.width, spread.height, spread.mode), (3, 2, PixelMode::Rgb));
            for pixel in spread.pixel_data().unwrap().as_chunks::<3>().0 {
                assert!(image.pixel_data().unwrap().as_chunks::<3>().0.contains(pixel));
            }
        });
    }

    #[test]
    fn spread_initializes_complete_pixels_without_touching_guards() {
        fn check<const C: usize>() {
            for width in [1, 3, 17] {
                for height in [1, 5] {
                    let input: Vec<u8> = (0..width * height * C + 3).map(|i| (i * 73 + i / 7) as u8).collect();
                    let source = &input[3..];
                    for distance in [1, 2, 30] {
                        let mut state = 0x1234_5678_u64;
                        let mut reference_state = state;
                        let mut random = || {
                            reference_state ^= reference_state << 13;
                            reference_state ^= reference_state >> 7;
                            reference_state ^= reference_state << 17;
                            reference_state
                        };
                        let mut expected = Vec::new();
                        for y in 0..height {
                            for x in 0..width {
                                let dx = (random() % (distance as u64 * 2 + 1)) as i64 - distance;
                                let dy = (random() % (distance as u64 * 2 + 1)) as i64 - distance;
                                let sx = (x as i64 + dx).clamp(0, width as i64 - 1) as usize;
                                let sy = (y as i64 + dy).clamp(0, height as i64 - 1) as usize;
                                let offset = (sy * width + sx) * C;
                                expected.extend_from_slice(&source[offset..offset + C]);
                            }
                        }
                        let mut actual = vec![std::mem::MaybeUninit::new(93); source.len() + 8];
                        spread_pixels::<C>(source, &mut actual[3..3 + source.len()], width, distance, &mut state);
                        // SAFETY: the whole buffer was initialized before the call.
                        let actual: Vec<_> = actual.into_iter().map(|value| unsafe { value.assume_init() }).collect();
                        assert_eq!(&actual[..3], &[93; 3]);
                        assert_eq!(&actual[3..3 + source.len()], expected);
                        assert_eq!(&actual[3 + source.len()..], &[93; 5]);
                        assert_eq!(state, reference_state);
                    }
                }
            }
        }
        check::<1>();
        check::<2>();
        check::<3>();
        check::<4>();
        check::<6>();
        check::<8>();
    }

    #[test]
    fn entropy_accepts_wide_samples_and_mask() {
        Python::initialize();
        Python::attach(|py| {
            let image = Image::from_integer_bytes(4, 1, PixelMode::I16, vec![0, 0, 0, 1, 0, 2, 0, 3]).unwrap();
            assert_eq!(ops_entropy(py, &image, None, None).unwrap(), 2.0);
            let mask = Image::from_pixels(4, 1, PixelMode::L, vec![1, 0, 1, 0], None).unwrap();
            assert_eq!(ops_entropy(py, &image, Some(&mask), Some((0.0, 512.0))).unwrap(), 1.0);
        });
    }

    #[test]
    fn remap_indices_and_pack_xbm_rows() {
        let indexed = Image::from_pixels(3, 1, PixelMode::L, vec![0, 1, 2], None).unwrap();
        let mut mapping: Vec<u8> = (0..=255).collect();
        mapping[..3].copy_from_slice(&[1, 2, 0]);
        assert_eq!(remap_indices(&indexed, &mapping).unwrap().pixel_data().unwrap(), [1, 2, 0]);
        for length in [0, 15, 16, 17, CHUNK_PIXELS * 3 + 7] {
            let source: Vec<_> = (0..length).map(|i| (i * 73) as u8).collect();
            let expected: Vec<_> = source.iter().map(|&value| mapping[value as usize]).collect();
            let indexed = Image::from_pixels(length as u32, 1, PixelMode::L, source, None).unwrap();
            assert_eq!(remap_indices(&indexed, &mapping).unwrap().pixel_data().unwrap(), expected);
        }
        let bitmap = Image::from_pixels(9, 2, PixelMode::One, vec![255, 0, 0, 0, 0, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255], None).unwrap();
        assert_eq!(ops_xbm_bits(&bitmap).unwrap(), [0x81, 0, 0, 1]);
    }
}
