use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{PyIndexError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyTuple};
use std::collections::HashMap;

use crate::codecs::{self, ImageFormat, SaveOptions};

fn parse_pixel(value: &Bound<'_, PyAny>, channels: usize) -> PyResult<[u8; 4]> {
    if !value.is_exact_instance_of::<PyTuple>()
        && let Ok(number) = value.extract::<i64>()
    {
        return Ok(if channels == 1 {
            [number.clamp(0, 255) as u8, 0, 0, 0]
        } else {
            std::array::from_fn(|i| ((number >> (i * 8)) & 255) as u8)
        });
    }
    let tuple = value.cast::<PyTuple>().map_err(|_| PyTypeError::new_err("color must be int or tuple"))?;
    let count = tuple.len();
    if !(count == channels || channels >= 3 && matches!(count, 3 | 4)) {
        return Err(PyTypeError::new_err("wrong number of color components"));
    }
    let mut pixel = [0, 0, 0, 255];
    for (i, component) in pixel.iter_mut().enumerate().take(channels.min(count)) {
        *component = tuple.get_borrowed_item(i)?.extract::<i64>()?.clamp(0, 255) as u8;
    }
    Ok(pixel)
}

type ColorCounts = [(u64, usize, usize); 512];

fn bounding_box_l(pixels: &[u8], width: usize) -> Option<(usize, usize, usize, usize)> {
    let top = pixels.chunks_exact(width).position(|row| row.iter().any(|&v| v != 0))?;
    let bottom = pixels.chunks_exact(width).rposition(|row| row.iter().any(|&v| v != 0)).unwrap();
    let first = &pixels[top * width..(top + 1) * width];
    let mut left = first.iter().position(|&v| v != 0).unwrap();
    let mut right = first.iter().rposition(|&v| v != 0).unwrap() + 1;
    let remaining = &pixels[(top + 1) * width..(bottom + 1) * width];
    if left > 8 || width - right > 8 {
        // Wide empty margins are cheaper to scan contiguously by row.
        for row in remaining.chunks_exact(width) {
            if let Some(x) = row[..left].iter().position(|&v| v != 0) {
                left = x;
            }
            if let Some(x) = row[right..].iter().rposition(|&v| v != 0) {
                right += x + 1;
            }
            if left == 0 && right == width {
                break;
            }
        }
        return Some((left, top, right, bottom + 1));
    }
    // Search only columns outside the first occupied row's bounds. Separate
    // edges avoid revisiting an already complete right edge on every row.
    for x in 0..left {
        if remaining.chunks_exact(width).any(|row| row[x] != 0) {
            left = x;
            break;
        }
    }
    for x in (right..width).rev() {
        if remaining.chunks_exact(width).any(|row| row[x] != 0) {
            right = x + 1;
            break;
        }
    }
    Some((left, top, right, bottom + 1))
}

fn extrema<const C: usize>(data: &[u8]) -> Vec<(u16, u16)> {
    let mut minimum = [255_u8; C];
    let mut maximum = [0_u8; C];
    for block in data.as_chunks::<C>().0.chunks(256) {
        for pixel in block {
            for c in 0..C {
                minimum[c] = minimum[c].min(pixel[c]);
                maximum[c] = maximum[c].max(pixel[c]);
            }
        }
        if minimum == [0; C] && maximum == [255; C] {
            break;
        }
    }
    minimum.into_iter().zip(maximum).map(|(a, b)| (u16::from(a), u16::from(b))).collect()
}

fn count_colors<const N: usize>(data: &[u8], maxcolors: usize) -> Option<ColorCounts> {
    const CHUNK: usize = 128 * 1024;
    if maxcolors >= 64 && crate::parallel::should_parallel(data.len() / N, CHUNK, CHUNK * 2) {
        use rayon::prelude::*;
        let tables: Option<Vec<_>> = data
            .par_chunks(CHUNK * N)
            .map(|chunk| count_colors_serial::<N>(chunk, maxcolors))
            .collect();
        let mut merged = [(0, 0, 0); 512];
        let mut used = 0;
        for (chunk, table) in tables?.into_iter().enumerate() {
            for (key, count, index) in table.into_iter().filter(|entry| entry.1 != 0) {
                let mut slot = (key.wrapping_mul(0x9e3779b97f4a7c15) >> 55) as usize;
                loop {
                    let entry = &mut merged[slot];
                    if entry.1 == 0 {
                        used += 1;
                        if used > maxcolors {
                            return None;
                        }
                        *entry = (key, count, chunk * CHUNK + index);
                        break;
                    }
                    if entry.0 == key {
                        entry.1 += count;
                        break;
                    }
                    slot = (slot + 1) & 511;
                }
            }
        }
        return Some(merged);
    }
    count_colors_serial::<N>(data, maxcolors)
}

fn count_colors_serial<const N: usize>(data: &[u8], maxcolors: usize) -> Option<ColorCounts> {
    let mut table = [(0_u64, 0_usize, 0_usize); 512];
    let mut used = 0;
    for (i, pixel) in data.as_chunks::<N>().0.iter().enumerate() {
        let mut bytes = [0; 8];
        bytes[..N].copy_from_slice(pixel);
        let key = u64::from_le_bytes(bytes);
        let mut slot = (key.wrapping_mul(0x9e3779b97f4a7c15) >> 55) as usize;
        loop {
            let entry = &mut table[slot];
            if entry.1 == 0 {
                used += 1;
                if used > maxcolors {
                    return None;
                }
                *entry = (key, 1, i);
                break;
            }
            if entry.0 == key {
                entry.1 += 1;
                break;
            }
            slot = (slot + 1) & 511;
        }
    }
    Some(table)
}

fn put_pixels<const C: usize>(pixels: &mut [u8], data: &Bound<'_, PyAny>, length: usize, scale: f64, offset: f64) -> PyResult<()> {
    let mut output = pixels.as_chunks_mut::<C>().0.iter_mut();
    let mut write = |value: &Bound<'_, PyAny>| -> PyResult<()> {
        let pixel = if C == 1 {
            let number = value.extract::<f64>()?;
            [(number * scale + offset).clamp(0.0, 255.0) as u8, 0, 0, 0]
        } else {
            parse_pixel(value, C)?
        };
        *output.next().expect("validated data length") = pixel[..C].try_into().unwrap();
        Ok(())
    };
    if let Ok(list) = data.cast_exact::<PyList>() {
        for i in 0..length {
            write(&list.get_item(i)?)?;
        }
    } else if let Ok(tuple) = data.cast_exact::<PyTuple>() {
        for value in tuple.iter_borrowed() {
            write(&value)?;
        }
    } else {
        for i in 0..length {
            write(&data.get_item(i)?)?;
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PixelMode {
    L,
    Rgb,
    Rgba,
}

impl PixelMode {
    pub(crate) fn parse(value: &str) -> PyResult<Self> {
        match value {
            "L" => Ok(Self::L),
            "RGB" => Ok(Self::Rgb),
            "RGBA" => Ok(Self::Rgba),
            _ => Err(PyValueError::new_err(format!(
                "unsupported image mode {value:?}; expected 'L', 'RGB', or 'RGBA'"
            ))),
        }
    }

    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::L => "L",
            Self::Rgb => "RGB",
            Self::Rgba => "RGBA",
        }
    }

    pub(crate) const fn channels(self) -> usize {
        match self {
            Self::L => 1,
            Self::Rgb => 3,
            Self::Rgba => 4,
        }
    }
}

#[pyclass(name = "_Image", module = "blanket._blanket", skip_from_py_object)]
#[derive(Clone)]
pub(crate) struct Image {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) mode: PixelMode,
    pub(crate) pixels: Option<Vec<u8>>,
    pub(crate) bit_depth: u8,
    pub(crate) format: Option<String>,
    pub(crate) palette: Option<(PixelMode, Vec<u8>)>,
}

impl Image {
    pub(crate) fn from_pixels(width: u32, height: u32, mode: PixelMode, pixels: Vec<u8>, format: Option<String>) -> PyResult<Self> {
        let expected = expected_len(width, height, mode)?;
        if pixels.len() != expected {
            return Err(PyValueError::new_err(format!(
                "not enough image data: expected {expected} bytes, got {}",
                pixels.len()
            )));
        }
        Ok(Self {
            width,
            height,
            mode,
            pixels: Some(pixels),
            bit_depth: 8,
            format,
            palette: None,
        })
    }

    pub(crate) fn pixel_data(&self) -> PyResult<&[u8]> {
        if self.bit_depth != 8 {
            return Err(PyValueError::new_err(
                "this operation requires 8-bit pixels; use convert(..., bit_depth=8) explicitly",
            ));
        }
        self.raw_data()
    }

    pub(crate) fn raw_data(&self) -> PyResult<&[u8]> {
        self.pixels.as_deref().ok_or_else(|| PyValueError::new_err("operation on closed image"))
    }

    pub(crate) fn from_samples(width: u32, height: u32, mode: PixelMode, samples: Vec<u16>, bit_depth: u8, format: Option<String>) -> PyResult<Self> {
        if !matches!(bit_depth, 10 | 12 | 16) {
            return Err(PyValueError::new_err("bit_depth must be 8, 10, 12, or 16"));
        }
        if samples.len() != expected_len(width, height, mode)? || samples.iter().any(|&v| u32::from(v) >= (1_u32 << bit_depth)) {
            return Err(PyValueError::new_err("invalid sample count or sample outside bit_depth range"));
        }
        Ok(Self {
            width,
            height,
            mode,
            pixels: Some(samples.into_iter().flat_map(u16::to_le_bytes).collect()),
            bit_depth,
            format,
            palette: None,
        })
    }
}

#[pymethods]
impl Image {
    fn getextrema(&self, py: Python<'_>) -> PyResult<Vec<(u16, u16)>> {
        let data = self.raw_data()?;
        let channels = self.mode.channels();
        Ok(py.detach(|| {
            if data.is_empty() {
                return Vec::new();
            }
            if self.bit_depth == 8 {
                // Once every band spans the entire domain, no later pixel
                // can change the answer. Check only once per block.
                return match self.mode {
                    PixelMode::L => extrema::<1>(data),
                    PixelMode::Rgb => extrema::<3>(data),
                    PixelMode::Rgba => extrema::<4>(data),
                };
            }
            let mut ranges = vec![(u16::MAX, 0); channels];
            for (i, bytes) in data.as_chunks::<2>().0.iter().enumerate() {
                let value = u16::from_le_bytes(*bytes);
                let range = &mut ranges[i % channels];
                range.0 = range.0.min(value);
                range.1 = range.1.max(value);
            }
            ranges
        }))
    }

    fn getchannel(&self, py: Python<'_>, channel: usize) -> PyResult<Self> {
        let data = self.raw_data()?;
        let channels = self.mode.channels();
        if channel >= channels {
            return Err(PyValueError::new_err("band index out of range"));
        }
        py.detach(|| {
            if self.bit_depth == 8 {
                let pixels = match self.mode {
                    PixelMode::L => data.to_vec(),
                    PixelMode::Rgb => data.as_chunks::<3>().0.iter().map(|pixel| pixel[channel]).collect(),
                    PixelMode::Rgba => data.as_chunks::<4>().0.iter().map(|pixel| pixel[channel]).collect(),
                };
                Self::from_pixels(self.width, self.height, PixelMode::L, pixels, None)
            } else {
                let samples = data
                    .chunks_exact(channels * 2)
                    .map(|pixel| u16::from_le_bytes([pixel[channel * 2], pixel[channel * 2 + 1]]))
                    .collect();
                Self::from_samples(self.width, self.height, PixelMode::L, samples, self.bit_depth, None)
            }
        })
    }

    fn getdata(&self, py: Python<'_>) -> PyResult<Py<PyList>> {
        let data = self.raw_data()?;
        if self.bit_depth == 8 {
            return Ok(match self.mode {
                PixelMode::L => PyList::new(py, data.iter().copied())?,
                PixelMode::Rgb => PyList::new(py, data.as_chunks::<3>().0.iter().map(|p| (p[0], p[1], p[2])))?,
                PixelMode::Rgba => PyList::new(py, data.as_chunks::<4>().0.iter().map(|p| (p[0], p[1], p[2], p[3])))?,
            }
            .unbind());
        }
        let result = PyList::empty(py);
        for y in 0..self.height {
            for x in 0..self.width {
                result.append(self.getpixel(py, (i64::from(x), i64::from(y)))?)?;
            }
        }
        Ok(result.unbind())
    }

    fn getcolors(&self, py: Python<'_>, maxcolors: usize) -> PyResult<Option<Py<PyList>>> {
        let data = self.raw_data()?;
        let stride = self.mode.channels() * if self.bit_depth == 8 { 1 } else { 2 };
        if stride == 1 {
            let counts = py.detach(|| {
                let mut counts = [0_usize; 256];
                for &value in data {
                    counts[value as usize] += 1;
                }
                counts
            });
            if counts.iter().filter(|&&n| n != 0).count() > maxcolors {
                return Ok(None);
            }
            return Ok(Some(
                PyList::new(py, counts.into_iter().enumerate().filter(|&(_, n)| n != 0).map(|(v, n)| (n, v)))?.unbind(),
            ));
        }
        if maxcolors <= 256 {
            // A bounded open-addressed table avoids hashing a byte slice and
            // allocating a map entry for each distinct small pixel value.
            let counts = py.detach(|| match stride {
                2 => count_colors::<2>(data, maxcolors),
                3 => count_colors::<3>(data, maxcolors),
                4 => count_colors::<4>(data, maxcolors),
                6 => count_colors::<6>(data, maxcolors),
                8 => count_colors::<8>(data, maxcolors),
                _ => unreachable!("validated pixel stride"),
            });
            let Some(counts) = counts else { return Ok(None) };
            let result = PyList::empty(py);
            for (_, count, i) in counts.into_iter().filter(|entry| entry.1 != 0) {
                result.append((
                    count,
                    self.getpixel(py, ((i % self.width as usize) as i64, (i / self.width as usize) as i64))?,
                ))?;
            }
            return Ok(Some(result.unbind()));
        }
        let counts = py.detach(|| {
            let mut counts = HashMap::<&[u8], (usize, usize)>::new();
            for (i, pixel) in data.chunks_exact(stride).enumerate() {
                let entry = counts.entry(pixel).or_insert((0, i));
                entry.0 += 1;
                if counts.len() > maxcolors {
                    return None;
                }
            }
            Some(counts)
        });
        let Some(counts) = counts else { return Ok(None) };
        let result = PyList::empty(py);
        for (count, i) in counts.into_values() {
            let pixel = self.getpixel(py, ((i % self.width as usize) as i64, (i / self.width as usize) as i64))?;
            result.append((count, pixel))?;
        }
        Ok(Some(result.unbind()))
    }

    fn putpixel(&mut self, xy: (i64, i64), value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.pixel_data()?;
        let x = if xy.0 < 0 { xy.0 + i64::from(self.width) } else { xy.0 };
        let y = if xy.1 < 0 { xy.1 + i64::from(self.height) } else { xy.1 };
        if x < 0 || y < 0 || x >= i64::from(self.width) || y >= i64::from(self.height) {
            return Err(PyIndexError::new_err("image index out of range"));
        }
        let channels = self.mode.channels();
        let pixel = parse_pixel(value, channels)?;
        let offset = (y as usize * self.width as usize + x as usize) * channels;
        self.pixels.as_mut().unwrap()[offset..offset + channels].copy_from_slice(&pixel[..channels]);
        Ok(())
    }

    fn putdata(&mut self, data: &Bound<'_, PyAny>, scale: f64, offset: f64) -> PyResult<()> {
        self.pixel_data()?;
        let length = data.len()?;
        if length > self.width as usize * self.height as usize {
            return Err(PyTypeError::new_err("too many data entries"));
        }
        let pixels = self.pixels.as_mut().unwrap();
        match self.mode {
            PixelMode::L => put_pixels::<1>(pixels, data, length, scale, offset),
            PixelMode::Rgb => put_pixels::<3>(pixels, data, length, scale, offset),
            PixelMode::Rgba => put_pixels::<4>(pixels, data, length, scale, offset),
        }
    }

    fn getbbox(&self, py: Python<'_>, alpha_only: bool) -> PyResult<Option<(u32, u32, u32, u32)>> {
        let pixels = self.raw_data()?;
        let channels = self.mode.channels();
        let sample_bytes = if self.bit_depth == 8 { 1 } else { 2 };
        Ok(py.detach(|| {
            if self.width == 0 || self.height == 0 {
                return None;
            }
            let stride = channels * sample_bytes;
            if stride == 1 {
                return bounding_box_l(pixels, self.width as usize)
                    .map(|(left, top, right, bottom)| (left as u32, top as u32, right as u32, bottom as u32));
            }
            let row_bytes = self.width as usize * stride;
            let occupied = |pixel: &[u8]| {
                let values = if alpha_only && self.mode == PixelMode::Rgba {
                    &pixel[3 * sample_bytes..]
                } else {
                    pixel
                };
                values.iter().any(|&v| v != 0)
            };
            let mut rows = pixels.chunks_exact(row_bytes);
            let top = rows.position(|row| row.chunks_exact(stride).any(occupied))?;
            let bottom = pixels
                .chunks_exact(row_bytes)
                .rposition(|row| row.chunks_exact(stride).any(occupied))
                .unwrap();
            let mut left = self.width as usize;
            let mut right = 0;
            for row in pixels[top * row_bytes..(bottom + 1) * row_bytes].chunks_exact(row_bytes) {
                if let Some(x) = row[..left * stride].chunks_exact(stride).position(occupied) {
                    left = x;
                }
                if let Some(x) = row[right * stride..].chunks_exact(stride).rposition(occupied) {
                    right += x + 1;
                }
                if left == 0 && right == self.width as usize {
                    break;
                }
            }
            Some((left as u32, top as u32, right as u32, bottom as u32 + 1))
        }))
    }

    #[getter]
    fn bit_depth(&self) -> u8 {
        self.bit_depth
    }

    #[getter]
    fn mode(&self) -> &'static str {
        if self.palette.is_some() { "P" } else { self.mode.as_str() }
    }

    #[getter]
    fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    #[getter]
    fn width(&self) -> u32 {
        self.width
    }

    #[getter]
    fn height(&self) -> u32 {
        self.height
    }

    #[getter]
    fn format(&self) -> Option<&str> {
        self.format.as_deref()
    }

    #[getter]
    fn info(&self, py: Python<'_>) -> Py<PyDict> {
        PyDict::new(py).unbind()
    }

    fn load(&self) -> PyResult<()> {
        self.raw_data().map(|_| ())
    }

    fn copy(&self, py: Python<'_>) -> PyResult<Self> {
        let source = self.raw_data()?;
        let mut pixels = Vec::<u8>::with_capacity(source.len());
        let row = self.width as usize * self.mode.channels() * if self.bit_depth == 8 { 1 } else { 2 };
        let chunk = if source.len() >= 4 * 1024 * 1024 { 256 * 1024 } else { row.max(4096) };
        py.detach(|| {
            crate::parallel::chunks_mut_above(&mut pixels.spare_capacity_mut()[..source.len()], chunk, 4 * 1024 * 1024, |i, dst| {
                // Disjoint output partitions initialize the entire reserved buffer.
                // Source and destination are separate allocations of equal length.
                unsafe { std::ptr::copy_nonoverlapping(source[i * chunk..].as_ptr(), dst.as_mut_ptr().cast::<u8>(), dst.len()) };
            });
        });
        // Every byte is initialized above, including any partial final partition.
        unsafe { pixels.set_len(source.len()) };
        Ok(Self {
            width: self.width,
            height: self.height,
            mode: self.mode,
            pixels: Some(pixels),
            bit_depth: self.bit_depth,
            format: None,
            palette: self.palette.clone(),
        })
    }

    fn getpixel(&self, py: Python<'_>, xy: (i64, i64)) -> PyResult<Py<PyAny>> {
        let pixels = self.raw_data()?;
        let x = if xy.0 < 0 { xy.0 + i64::from(self.width) } else { xy.0 };
        let y = if xy.1 < 0 { xy.1 + i64::from(self.height) } else { xy.1 };
        if x < 0 || y < 0 || x >= i64::from(self.width) || y >= i64::from(self.height) {
            return Err(PyIndexError::new_err("image index out of range"));
        }
        let channels = self.mode.channels();
        let offset = (y as usize * self.width as usize + x as usize) * channels;
        if self.bit_depth > 8 {
            let sample = |i: usize| u16::from_le_bytes([pixels[2 * i], pixels[2 * i + 1]]);
            return Ok(match self.mode {
                PixelMode::L => sample(offset).into_pyobject(py)?.into_any().unbind(),
                PixelMode::Rgb => (sample(offset), sample(offset + 1), sample(offset + 2))
                    .into_pyobject(py)?
                    .into_any()
                    .unbind(),
                PixelMode::Rgba => (sample(offset), sample(offset + 1), sample(offset + 2), sample(offset + 3))
                    .into_pyobject(py)?
                    .into_any()
                    .unbind(),
            });
        }
        Ok(match self.mode {
            PixelMode::L => pixels[offset].into_pyobject(py)?.into_any().unbind(),
            PixelMode::Rgb => (pixels[offset], pixels[offset + 1], pixels[offset + 2])
                .into_pyobject(py)?
                .into_any()
                .unbind(),
            PixelMode::Rgba => (pixels[offset], pixels[offset + 1], pixels[offset + 2], pixels[offset + 3])
                .into_pyobject(py)?
                .into_any()
                .unbind(),
        })
    }

    fn palette_data(&self) -> Option<(String, Vec<u8>)> {
        self.palette.as_ref().map(|(mode, data)| (mode.as_str().to_owned(), data.clone()))
    }

    fn set_palette(&mut self, mode: &str, data: Vec<u8>) -> PyResult<()> {
        self.pixel_data()?;
        let mode = PixelMode::parse(mode)?;
        if self.mode != PixelMode::L || mode == PixelMode::L || data.len() > 256 * mode.channels() || !data.len().is_multiple_of(mode.channels()) {
            return Err(PyValueError::new_err("invalid palette"));
        }
        self.palette = Some((mode, data));
        Ok(())
    }

    fn close(&mut self) {
        self.pixels = None;
    }

    #[pyo3(signature = (mode, bit_depth=None))]
    fn convert(&self, py: Python<'_>, mode: &str, bit_depth: Option<u8>) -> PyResult<Self> {
        if mode == "P" && self.palette.is_some() {
            return self.copy(py);
        }
        let destination = PixelMode::parse(mode)?;
        let depth = bit_depth.unwrap_or(self.bit_depth);
        if !matches!(depth, 8 | 10 | 12 | 16) {
            return Err(PyValueError::new_err("bit_depth must be 8, 10, 12, or 16"));
        }
        if self.bit_depth > 8 || depth > 8 {
            if self.palette.is_some() {
                let expanded = self.convert(py, mode, Some(8))?;
                return expanded.convert(py, mode, Some(depth));
            }
            let bytes = self.raw_data()?;
            let samples: Vec<u16> = if self.bit_depth == 8 {
                bytes.iter().map(|&v| u16::from(v)).collect()
            } else {
                bytes.as_chunks::<2>().0.iter().map(|v| u16::from_le_bytes(*v)).collect()
            };
            let source_max = (1_u32 << self.bit_depth) - 1;
            let target_max = (1_u32 << depth) - 1;
            let mut output = Vec::with_capacity(expected_len(self.width, self.height, destination)?);
            for pixel in samples.chunks_exact(self.mode.channels()) {
                let (r, g, b, a) = match self.mode {
                    PixelMode::L => (pixel[0], pixel[0], pixel[0], source_max as u16),
                    PixelMode::Rgb => (pixel[0], pixel[1], pixel[2], source_max as u16),
                    PixelMode::Rgba => (pixel[0], pixel[1], pixel[2], pixel[3]),
                };
                let scale = |v: u16| ((u32::from(v) * target_max + source_max / 2) / source_max) as u16;
                match destination {
                    PixelMode::L => output.push(scale(
                        ((u64::from(r) * 19595 + u64::from(g) * 38470 + u64::from(b) * 7471 + 32768) >> 16) as u16,
                    )),
                    PixelMode::Rgb => output.extend([scale(r), scale(g), scale(b)]),
                    PixelMode::Rgba => output.extend([scale(r), scale(g), scale(b), scale(a)]),
                }
            }
            return if depth == 8 {
                Self::from_pixels(self.width, self.height, destination, output.into_iter().map(|v| v as u8).collect(), None)
            } else {
                Self::from_samples(self.width, self.height, destination, output, depth, None)
            };
        }
        if let Some((palette_mode, palette)) = &self.palette {
            let indices = self.pixel_data()?;
            let converted = py.detach(|| {
                let expanded = match palette_mode {
                    PixelMode::L => expand_palette::<1>(indices, palette),
                    PixelMode::Rgb => expand_palette::<3>(indices, palette),
                    PixelMode::Rgba => expand_palette::<4>(indices, palette),
                };
                if *palette_mode == destination {
                    expanded
                } else {
                    convert_pixels(&expanded, *palette_mode, destination)
                }
            });
            return Self::from_pixels(self.width, self.height, destination, converted, None);
        }
        let source = self.mode;
        let pixels = self.pixel_data()?;
        let converted = py.detach(|| convert_pixels(pixels, source, destination));
        Self::from_pixels(self.width, self.height, destination, converted, None)
    }

    fn tobytes(&self, py: Python<'_>) -> PyResult<Py<PyBytes>> {
        Ok(PyBytes::new(py, self.raw_data()?).unbind())
    }

    #[pyo3(name = "_encode")]
    #[pyo3(signature = (format, quality, compress_level, lossless, effort, compressor=None))]
    #[allow(clippy::too_many_arguments)]
    fn encode(
        &self, py: Python<'_>, format: &str, quality: u8, compress_level: u8, lossless: bool, effort: u8,
        compressor: Option<&crate::compressor::LosslessImageCompressor>,
    ) -> PyResult<Py<PyBytes>> {
        let format = ImageFormat::parse(format)?;
        let options = SaveOptions {
            quality,
            compress_level,
            lossless,
            effort,
        };
        if let Some((palette_mode, _)) = &self.palette {
            if format == ImageFormat::Png {
                let encoded = py.detach(|| codecs::encode_palette_png(self, compress_level))?;
                return Ok(PyBytes::new(py, &encoded).unbind());
            }
            if format == ImageFormat::Jpeg {
                return Err(pyo3::exceptions::PyOSError::new_err("cannot write mode P as JPEG"));
            }
            let mode = palette_mode.as_str();
            let expanded = self.convert(py, mode, None)?;
            let encoded = py.detach(|| match compressor {
                Some(compressor) => compressor.encode(&expanded, format, options),
                None => codecs::encode(&expanded, format, options),
            })?;
            return Ok(PyBytes::new(py, &encoded).unbind());
        }
        let encoded = py.detach(|| match compressor {
            Some(compressor) => compressor.encode(self, format, options),
            None => codecs::encode(self, format, options),
        })?;
        Ok(PyBytes::new(py, &encoded).unbind())
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __exit__(&mut self, _exc_type: &Bound<'_, PyAny>, _exc_value: &Bound<'_, PyAny>, _traceback: &Bound<'_, PyAny>) {
        self.close();
    }

    fn __repr__(&self) -> String {
        format!(
            "<blanket.Image.Image image mode={} size={}x{}>",
            self.mode.as_str(),
            self.width,
            self.height
        )
    }
}

#[pyfunction(signature = (mode, size, data, bit_depth=8))]
pub(crate) fn frombytes(mode: &str, size: (u32, u32), data: &[u8], bit_depth: u8) -> PyResult<Image> {
    let mode = PixelMode::parse(mode)?;
    if bit_depth != 8 {
        if !data.len().is_multiple_of(2) {
            return Err(PyValueError::new_err("16-bit storage requires an even byte count"));
        }
        return Image::from_samples(
            size.0,
            size.1,
            mode,
            data.as_chunks::<2>().0.iter().map(|v| u16::from_le_bytes(*v)).collect(),
            bit_depth,
            None,
        );
    }
    Image::from_pixels(size.0, size.1, mode, data.to_vec(), None)
}

#[pyfunction(signature = (obj, mode = None, bit_depth = None))]
pub(crate) fn fromarray(py: Python<'_>, obj: &Bound<'_, PyAny>, mode: Option<&str>, bit_depth: Option<u8>) -> PyResult<Image> {
    let interface = obj.getattr("__array_interface__")?;
    let shape: Vec<usize> = interface.get_item("shape")?.extract()?;
    let typestr: String = interface.get_item("typestr")?.extract()?;
    let wide = matches!(typestr.as_str(), "<u2" | ">u2" | "=u2");
    let inferred_mode = array_mode(&shape, if wide { "|u1" } else { &typestr })?;
    let mode = mode.map_or(Ok(inferred_mode), PixelMode::parse)?;

    let maximum_dimensions = match mode {
        PixelMode::L => 2,
        PixelMode::Rgb => 3,
        PixelMode::Rgba => 4,
    };
    if shape.len() > maximum_dimensions {
        return Err(PyValueError::new_err(format!(
            "too many dimensions: {} > {maximum_dimensions}",
            shape.len()
        )));
    }

    let (width, height) = match shape.as_slice() {
        [] => {
            return Err(PyValueError::new_err("array must have at least one dimension"));
        }
        [height] => (1, *height),
        [height, width, ..] => (*width, *height),
    };
    let width = u32::try_from(width).map_err(|_| PyValueError::new_err("image dimensions are too large"))?;
    let height = u32::try_from(height).map_err(|_| PyValueError::new_err("image dimensions are too large"))?;

    let depth = bit_depth.unwrap_or(if wide { 16 } else { 8 });
    if wide {
        let raw: Vec<u8> = obj.call_method0("tobytes")?.extract()?;
        let samples = raw
            .as_chunks::<2>()
            .0
            .iter()
            .map(|v| {
                if typestr == ">u2" {
                    u16::from_be_bytes([v[0], v[1]])
                } else {
                    u16::from_le_bytes([v[0], v[1]])
                }
            })
            .collect();
        return Image::from_samples(width, height, mode, samples, depth, None);
    }
    if depth != 8 {
        return Err(PyValueError::new_err("high-bit-depth arrays must use uint16 samples"));
    }
    let pixels = PyBuffer::<u8>::get(obj)?.to_vec(py)?;
    Image::from_pixels(width, height, mode, pixels, None)
}

fn array_mode(shape: &[usize], typestr: &str) -> PyResult<PixelMode> {
    if typestr == "|u1" {
        match shape {
            [_] | [_, _] => return Ok(PixelMode::L),
            [_, _, 3] => return Ok(PixelMode::Rgb),
            [_, _, 4] => return Ok(PixelMode::Rgba),
            _ => {}
        }
    }

    let type_shape = match shape {
        [] | [_] | [_, _] => vec![1, 1],
        [_, _, channels, ..] => vec![1, 1, *channels],
    };
    Err(PyTypeError::new_err(format!("cannot handle this data type: {type_shape:?}, {typestr}")))
}

fn expected_len(width: u32, height: u32, mode: PixelMode) -> PyResult<usize> {
    let pixels = (width as usize)
        .checked_mul(height as usize)
        .ok_or_else(|| PyValueError::new_err("image dimensions are too large"))?;
    pixels
        .checked_mul(mode.channels())
        .ok_or_else(|| PyValueError::new_err("image dimensions are too large"))
}

fn convert_pixels(source: &[u8], from: PixelMode, to: PixelMode) -> Vec<u8> {
    if from == to {
        return source.to_vec();
    }

    crate::simd::convert(source, from, to)
}

/// Resolve palette indices through a complete 256-entry table. Missing
/// entries read as opaque black, as before.
fn expand_palette<const C: usize>(indices: &[u8], palette: &[u8]) -> Vec<u8> {
    let table: [[u8; C]; 256] = std::array::from_fn(|slot| {
        let mut color = [0; C];
        color.copy_from_slice(palette.get(slot * C..slot * C + C).unwrap_or(&[0, 0, 0, 255][..C]));
        color
    });
    let mut expanded = vec![0; indices.len() * C];
    crate::parallel::chunks_mut(&mut expanded, crate::parallel::CHUNK_PIXELS * C, |chunk, dst| {
        let start = chunk * crate::parallel::CHUNK_PIXELS;
        for (pixel, &slot) in dst.as_chunks_mut::<C>().0.iter_mut().zip(&indices[start..]) {
            *pixel = table[usize::from(slot)];
        }
    });
    expanded
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_buffer_length() {
        Python::initialize();
        Python::attach(|_| {
            let error = Image::from_pixels(2, 2, PixelMode::Rgb, vec![0; 11], None)
                .err()
                .expect("invalid buffer should fail");
            assert!(error.to_string().contains("expected 12 bytes"));
        });
    }

    #[test]
    fn converts_like_pillow_fixed_point_luminance() {
        let rgb = [255, 0, 0, 0, 255, 0, 0, 0, 255, 12, 34, 56];
        assert_eq!(convert_pixels(&rgb, PixelMode::Rgb, PixelMode::L), [76, 150, 29, 30]);
    }

    #[test]
    fn expands_and_drops_channels() {
        assert_eq!(convert_pixels(&[7, 23], PixelMode::L, PixelMode::Rgba), [7, 7, 7, 255, 23, 23, 23, 255]);
        assert_eq!(convert_pixels(&[1, 2, 3, 4], PixelMode::Rgba, PixelMode::Rgb), [1, 2, 3]);
    }

    #[test]
    fn expands_palette_indices_with_opaque_black_fallback() {
        let palette = [10, 20, 30, 40, 50, 60];
        assert_eq!(expand_palette::<3>(&[1, 0, 5], &palette), [40, 50, 60, 10, 20, 30, 0, 0, 0]);
        assert_eq!(expand_palette::<4>(&[0, 1], &[1, 2, 3, 4]), [1, 2, 3, 4, 0, 0, 0, 255]);
        assert!(expand_palette::<3>(&[], &palette).is_empty());
    }

    #[test]
    fn infers_supported_array_modes() {
        assert_eq!(array_mode(&[3, 5], "|u1").unwrap(), PixelMode::L);
        assert_eq!(array_mode(&[3, 5, 3], "|u1").unwrap(), PixelMode::Rgb);
        assert_eq!(array_mode(&[3, 5, 4], "|u1").unwrap(), PixelMode::Rgba);
        assert!(array_mode(&[3, 5], "<f4").is_err());
        assert!(array_mode(&[3, 5, 2], "|u1").is_err());
    }
}
