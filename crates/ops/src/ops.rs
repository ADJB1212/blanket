//! Pixel primitives used by the Python ImageOps API.

use blanket_core::buffers::{buffer, copy_rows, output, reserved_buffer};
pub use blanket_core::geometry::{ops_transpose, transpose};
use blanket_core::parallel::{CHUNK_PIXELS, MIN_PARALLEL_BYTES, chunks_mut, should_parallel};
use blanket_core::raster::{Image, PixelMode};
use pyo3::exceptions::{PyIndexError, PyOSError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use rayon::prelude::*;

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(ops_color_composite, module)?)?;
    module.add_function(wrap_pyfunction!(ops_lut, module)?)?;
    module.add_function(wrap_pyfunction!(ops_colorize, module)?)?;
    module.add_function(wrap_pyfunction!(ops_histogram, module)?)?;
    module.add_function(wrap_pyfunction!(ops_equalize, module)?)?;
    module.add_function(wrap_pyfunction!(ops_autocontrast, module)?)?;
    module.add_function(wrap_pyfunction!(ops_colorize_stops, module)?)?;
    module.add_function(wrap_pyfunction!(ops_split, module)?)?;
    module.add_function(wrap_pyfunction!(ops_entropy, module)?)?;
    module.add_function(wrap_pyfunction!(ops_spread, module)?)?;
    module.add_function(wrap_pyfunction!(ops_remap_indices, module)?)?;
    module.add_function(wrap_pyfunction!(ops_xbm_bits, module)?)?;
    module.add_function(wrap_pyfunction!(ops_xbm, module)?)?;
    module.add_function(wrap_pyfunction!(ops_canvas, module)?)?;
    module.add_function(wrap_pyfunction!(ops_transpose, module)?)?;
    Ok(())
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

/// Copy complete rows into a reserved, empty image without first zeroing it.

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
fn ops_xbm<'py>(py: Python<'py>, image: &Image, name: &str) -> PyResult<Bound<'py, PyBytes>> {
    let data = py.detach(|| {
        let bits = ops_xbm_bits(image)?;
        let mut result =
            format!("#define {name}_width {}\n#define {name}_height {}\nstatic char {name}_bits[] = {{\n", image.width, image.height).into_bytes();
        result.reserve(bits.len() * 5 + 3);
        const HEX: &[u8; 16] = b"0123456789abcdef";
        for (i, value) in bits.into_iter().enumerate() {
            if i != 0 {
                result.push(b',');
            }
            result.extend_from_slice(b"0x");
            result.push(HEX[usize::from(value >> 4)]);
            result.push(HEX[usize::from(value & 15)]);
        }
        result.extend_from_slice(b"\n};");
        Ok::<_, PyErr>(result)
    })?;
    Ok(PyBytes::new(py, &data))
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

fn check_tone_mode(image: &Image) -> PyResult<()> {
    if !matches!(image.mode, PixelMode::L | PixelMode::Rgb) {
        return Err(PyOSError::new_err(format!("not supported for mode {}", image.mode.as_str())));
    }
    Ok(())
}

#[pyfunction]
fn ops_equalize(py: Python<'_>, image: &Image, mask: Option<&Image>) -> PyResult<Image> {
    let bins = ops_histogram(py, image, mask)?;
    check_tone_mode(image)?;
    let mut table = Vec::with_capacity(bins.len());
    for counts in bins.as_chunks::<256>().0 {
        let total = counts.iter().sum::<u64>();
        let last = counts.iter().rfind(|&&count| count != 0).copied().unwrap_or(0);
        let step = (total - last) / 255;
        if step == 0 {
            table.extend(0..=255);
        } else {
            let mut cumulative = step / 2;
            for &count in counts {
                table.push((cumulative / step).min(255) as u8);
                cumulative += count;
            }
        }
    }
    ops_lut(py, image, table)
}

#[pyfunction]
fn ops_autocontrast(
    py: Python<'_>, image: &Image, histogram_image: &Image, mask: Option<&Image>, cutoff: [f64; 2], ignore: Vec<i64>,
) -> PyResult<Image> {
    let mut bins = ops_histogram(py, histogram_image, mask)?;
    check_tone_mode(image)?;
    if cutoff.iter().any(|&value| !(0.0..=100.0).contains(&value)) {
        return Err(PyValueError::new_err("cutoff must be between 0 and 100"));
    }
    if ignore.iter().any(|&value| !(-256..256).contains(&value)) {
        return Err(PyIndexError::new_err("list assignment index out of range"));
    }
    let mut table = Vec::with_capacity(bins.len());
    for counts in bins.as_chunks_mut::<256>().0 {
        for &value in &ignore {
            counts[value.rem_euclid(256) as usize] = 0;
        }
        let total = counts.iter().sum::<u64>();
        for (tail, percent) in cutoff.into_iter().enumerate() {
            let product = total as f64 * percent;
            // Match Python's float floor division, including its quotient correction.
            let quotient = (product - product % 100.0) / 100.0;
            let floor = quotient.floor();
            let mut remaining = (floor + f64::from(quotient - floor > 0.5)) as u64;
            for offset in 0..256 {
                let index = if tail == 0 { offset } else { 255 - offset };
                let removed = remaining.min(counts[index]);
                counts[index] -= removed;
                remaining -= removed;
                if remaining == 0 {
                    break;
                }
            }
        }
        let low = counts.iter().position(|&count| count != 0);
        let high = counts.iter().rposition(|&count| count != 0);
        match (low, high) {
            (Some(low), Some(high)) if low < high => {
                let gain = 255.0 / (high - low) as f64;
                let offset = -(low as f64) * gain;
                table.extend((0..256).map(|value| ((value as f64 * gain + offset) as i64).clamp(0, 255) as u8));
            }
            _ => table.extend(0..=255),
        }
    }
    ops_lut(py, image, table)
}

#[pyfunction]
fn ops_colorize_stops(py: Python<'_>, image: &Image, stops: Vec<(usize, [u8; 3])>) -> PyResult<Image> {
    if !(2..=3).contains(&stops.len()) || stops.iter().any(|stop| stop.0 > 255) || stops.windows(2).any(|pair| pair[0].0 > pair[1].0) {
        return Err(PyValueError::new_err("invalid colorize stops"));
    }
    let mut table = vec![0; 768];
    for value in 0..256 {
        let right = stops.partition_point(|stop| stop.0 <= value);
        for band in 0..3 {
            table[band * 256 + value] = if right == 0 {
                stops[0].1[band]
            } else if right == stops.len() {
                stops[right - 1].1[band]
            } else {
                let (start, first) = stops[right - 1];
                let (end, last) = stops[right];
                let delta = i32::from(last[band]) - i32::from(first[band]);
                (i32::from(first[band]) + ((value - start) as i32 * delta).div_euclid((end - start) as i32)) as u8
            };
        }
    }
    ops_colorize(py, image, table)
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
#[cfg(test)]
mod tests {
    use super::*;
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
