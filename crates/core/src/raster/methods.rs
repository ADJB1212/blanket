use std::sync::Arc;

use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{PyIndexError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyByteArray, PyBytes, PyDict, PyList, PyMemoryView, PySlice, PyTuple};
use rustc_hash::FxHashMap;

use super::pixels::{
    float_value, integer_bytes, integer_value, packed_pixel_data, parse_pixel, pixel_sequence, put_pixels, put_sample_pixel, wide_pixel_data,
};
use super::queries::{bounding_box_l, count_colors, extrema};
use super::validation::expected_len;
use super::{Image, PixelMode};

#[pymethods]
impl Image {
    pub fn getextrema_float(&self, py: Python<'_>) -> PyResult<Vec<(f32, f32)>> {
        if !matches!(self.mode, PixelMode::RgbF | PixelMode::RgbaF) {
            return Err(PyValueError::new_err("float color image required"));
        }
        let data = self.raw_data()?;
        let channels = self.mode.channels();
        Ok(py.detach(|| {
            if data.is_empty() {
                return Vec::new();
            }
            let mut ranges = vec![(f32::NAN, f32::NAN); channels];
            for pixel in data.chunks_exact(channels * 4) {
                for (range, bytes) in ranges.iter_mut().zip(pixel.as_chunks::<4>().0) {
                    let value = float_value(bytes);
                    if range.0.is_nan() && range.1.is_nan() {
                        *range = (value, value);
                    } else {
                        if value < range.0 {
                            range.0 = value;
                        }
                        if value > range.1 {
                            range.1 = value;
                        }
                    }
                }
            }
            ranges
        }))
    }

    pub fn getextrema_scalar(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        if !self.mode.is_wide_scalar() || self.mode.channels() != 1 {
            return Err(PyValueError::new_err("scalar extrema require a wide scalar mode"));
        }
        let data = self.raw_data()?;
        if data.is_empty() {
            return Ok(None);
        }
        if self.mode == PixelMode::F {
            let (lo, hi) = py.detach(|| {
                let values = data.as_chunks::<4>().0.iter().map(|bytes| float_value(bytes));
                values.fold((f32::NAN, f32::NAN), |(lo, hi), v| {
                    if lo.is_nan() && hi.is_nan() {
                        return (v, v);
                    }
                    (if v < lo { v } else { lo }, if v > hi { v } else { hi })
                })
            });
            return Ok(Some((lo, hi).into_pyobject(py)?.into_any().unbind()));
        }
        let mode = self.mode;
        let (lo, hi) = py.detach(|| {
            data.chunks_exact(mode.sample_bytes())
                .map(|bytes| integer_value(mode, bytes))
                .fold((i64::MAX, i64::MIN), |(lo, hi), v| (lo.min(v), hi.max(v)))
        });
        Ok(Some((lo, hi).into_pyobject(py)?.into_any().unbind()))
    }

    pub fn getextrema(&self, py: Python<'_>) -> PyResult<Vec<(u16, u16)>> {
        if self.mode.is_integer() {
            return Err(PyValueError::new_err("integer extrema require getdata"));
        }
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
                    PixelMode::One | PixelMode::L => extrema::<1>(data),
                    PixelMode::La | PixelMode::PremulLa | PixelMode::Pa => extrema::<2>(data),
                    PixelMode::Rgb | PixelMode::Hsv | PixelMode::YCbCr | PixelMode::Lab => extrema::<3>(data),
                    PixelMode::Rgba | PixelMode::PremulRgba | PixelMode::Rgbx | PixelMode::Cmyk => extrema::<4>(data),
                    _ => unreachable!(),
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

    pub fn getchannel(&self, py: Python<'_>, channel: usize) -> PyResult<Self> {
        if self.mode.is_float() && self.mode.channels() > 1 {
            if channel >= self.mode.channels() {
                return Err(PyValueError::new_err("band index out of range"));
            }
            let data = self.raw_data()?;
            let pixels = py.detach(|| {
                data.chunks_exact(self.raw_bytes_per_pixel())
                    .map(|p| <[u8; 4]>::try_from(&p[channel * 4..channel * 4 + 4]).unwrap())
                    .collect::<Vec<_>>()
                    .into_flattened()
            });
            return Self::from_float_bytes(self.width, self.height, pixels);
        }
        if self.mode.is_wide_scalar() {
            return if channel == 0 { self.copy(py) } else { Err(PyValueError::new_err("band index out of range")) };
        }
        let data = self.raw_data()?;
        let channels = self.mode.channels();
        if channel >= channels {
            return Err(PyValueError::new_err("band index out of range"));
        }
        if matches!(self.mode, PixelMode::One | PixelMode::L) {
            let mut band = self.copy(py)?;
            band.mode = PixelMode::L;
            band.palette = None;
            return Ok(band);
        }
        py.detach(|| {
            if self.bit_depth == 8 {
                let pixels = match self.mode {
                    PixelMode::La | PixelMode::PremulLa | PixelMode::Pa => crate::pixels::extract_two_channel(data, channel),
                    PixelMode::Rgb | PixelMode::Hsv | PixelMode::YCbCr | PixelMode::Lab => crate::pixels::extract_channel::<3>(data, channel),
                    PixelMode::Rgba | PixelMode::PremulRgba | PixelMode::Rgbx | PixelMode::Cmyk => crate::pixels::extract_channel::<4>(data, channel),
                    _ => unreachable!(),
                };
                Self::from_pixels(self.width, self.height, PixelMode::L, pixels, None)
            } else {
                let pixels = crate::pixels::extract_sample_channel(data, channels, channel);
                Self::from_sample_bytes(self.width, self.height, PixelMode::L, pixels, self.bit_depth, None)
            }
        })
    }

    pub fn getdata(&self, py: Python<'_>) -> PyResult<Py<PyList>> {
        let data = self.raw_data()?;
        if self.bit_depth == 8 && self.mode.channels() > 1 {
            let result = match self.mode.channels() {
                2 => packed_pixel_data::<2>(py, data, false)?,
                3 => packed_pixel_data::<3>(py, data, false)?,
                4 => packed_pixel_data::<4>(py, data, false)?,
                _ => unreachable!(),
            };
            return Ok(result.cast_into::<PyList>()?.unbind());
        }
        if self.mode == PixelMode::F {
            return Ok(PyList::new(py, data.as_chunks::<4>().0.iter().map(|bytes| float_value(bytes)))?.unbind());
        }
        if self.mode.is_integer() {
            return Ok(PyList::new(py, data.chunks_exact(self.mode.sample_bytes()).map(|bytes| integer_value(self.mode, bytes)))?.unbind());
        }
        if self.bit_depth == 8 {
            return Ok(PyList::new(py, data.iter().copied())?.unbind());
        }
        Ok(wide_pixel_data(py, self, false)?.cast_into::<PyList>()?.unbind())
    }

    pub fn get_flattened_data(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        let data = self.raw_data()?;
        if self.bit_depth == 8 && self.mode.channels() > 1 {
            let result = match self.mode.channels() {
                2 => packed_pixel_data::<2>(py, data, true)?,
                3 => packed_pixel_data::<3>(py, data, true)?,
                4 => packed_pixel_data::<4>(py, data, true)?,
                _ => unreachable!(),
            };
            return Ok(result.cast_into::<PyTuple>()?.unbind());
        }
        if self.mode == PixelMode::F {
            return Ok(PyTuple::new(py, data.as_chunks::<4>().0.iter().map(|bytes| float_value(bytes)))?.unbind());
        }
        if self.mode.is_integer() {
            return Ok(PyTuple::new(py, data.chunks_exact(self.mode.sample_bytes()).map(|bytes| integer_value(self.mode, bytes)))?.unbind());
        }
        if self.bit_depth == 8 {
            return Ok(PyTuple::new(py, data.iter().copied())?.unbind());
        }
        Ok(wide_pixel_data(py, self, true)?.cast_into::<PyTuple>()?.unbind())
    }

    pub fn getbanddata<'py>(&self, py: Python<'py>, channel: usize, tuple: bool) -> PyResult<Bound<'py, PyAny>> {
        let data = self.raw_data()?;
        if channel >= self.mode.channels() {
            return Err(PyValueError::new_err("band index out of range"));
        }
        let stride = self.raw_bytes_per_pixel();
        let sample_bytes = stride / self.mode.channels();
        let items = data.chunks_exact(stride).map(|pixel| {
            let sample = &pixel[channel * sample_bytes..(channel + 1) * sample_bytes];
            if self.mode.is_float() {
                Ok(float_value(sample).into_pyobject(py)?.into_any())
            } else if self.mode.is_integer() {
                Ok(integer_value(self.mode, sample).into_pyobject(py)?.into_any())
            } else if sample_bytes == 2 {
                Ok(u16::from_le_bytes([sample[0], sample[1]]).into_pyobject(py)?.into_any())
            } else {
                Ok(sample[0].into_pyobject(py)?.into_any())
            }
        });
        pixel_sequence(py, items, tuple)
    }

    pub fn getcolors(&self, py: Python<'_>, maxcolors: usize) -> PyResult<Option<Py<PyList>>> {
        let data = self.raw_data()?;
        let stride = self.mode.channels()
            * if self.mode.is_wide_scalar() {
                self.mode.sample_bytes()
            } else if self.bit_depth == 8 {
                1
            } else {
                2
            };
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
            return Ok(Some(PyList::new(py, counts.into_iter().enumerate().filter(|&(_, n)| n != 0).map(|(v, n)| (n, v)))?.unbind()));
        }
        if maxcolors <= 256 && stride <= 8 {
            // A bounded open-addressed table avoids hashing a byte slice and
            // allocating a map entry for each distinct small pixel value.
            let counts = py.detach(|| match stride {
                2 => count_colors::<2>(data, maxcolors),
                3 => count_colors::<3>(data, maxcolors),
                4 => count_colors::<4>(data, maxcolors),
                6 => count_colors::<6>(data, maxcolors),
                8 => count_colors::<8>(data, maxcolors),
                _ => return None,
            });
            let Some(counts) = counts else { return Ok(None) };
            let entries = counts.into_iter().filter(|entry| entry.1 != 0).map(|(_, count, i)| (count, i));
            return Ok(Some(self.color_count_list(py, entries)?.unbind()));
        }
        let counts = py.detach(|| {
            let mut counts = FxHashMap::<&[u8], (usize, usize)>::default();
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
        Ok(Some(self.color_count_list(py, counts.into_values())?.unbind()))
    }

    pub fn putpixel(&mut self, xy: (i64, i64), value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.ensure_owned()?;
        self.raw_data()?;
        let x = if xy.0 < 0 { xy.0 + i64::from(self.width) } else { xy.0 };
        let y = if xy.1 < 0 { xy.1 + i64::from(self.height) } else { xy.1 };
        if x < 0 || y < 0 || x >= i64::from(self.width) || y >= i64::from(self.height) {
            return Err(PyIndexError::new_err("image index out of range"));
        }
        if self.mode.is_float() && self.mode.channels() > 1 {
            let values: Vec<f32> = value.extract()?;
            if values.len() != self.mode.channels() {
                return Err(PyValueError::new_err("wrong number of color components"));
            }
            let start = (y as usize * self.width as usize + x as usize) * self.raw_bytes_per_pixel();
            let pixels = self.write_pixels(value.py())?;
            for (sample, value) in pixels[start..start + values.len() * 4].as_chunks_mut::<4>().0.iter_mut().zip(values) {
                *sample = value.to_le_bytes();
            }
            return Ok(());
        }
        if self.mode == PixelMode::F {
            let start = (y as usize * self.width as usize + x as usize) * 4;
            let sample = value.extract::<f64>()? as f32;
            self.write_pixels(value.py())?[start..start + 4].copy_from_slice(&sample.to_le_bytes());
            return Ok(());
        }
        if self.mode.is_integer() {
            let bytes = integer_bytes(self.mode, value.extract::<i64>()?);
            let stride = self.mode.sample_bytes();
            let offset = (y as usize * self.width as usize + x as usize) * stride;
            self.write_pixels(value.py())?[offset..offset + stride].copy_from_slice(&bytes[..stride]);
            return Ok(());
        }
        if crate::samples::supported(self) {
            let channels = self.mode.channels();
            let maximum = crate::samples::maximum(self);
            let start = (y as usize * self.width as usize + x as usize) * channels * 2;
            return put_sample_pixel(&mut self.write_pixels(value.py())?[start..start + channels * 2], value, channels, maximum, 1.0, 0.0);
        }
        self.pixel_data()?;
        let channels = self.mode.channels();
        let pixel = if self.mode == PixelMode::One { [value.extract::<i64>()?.clamp(0, 255) as u8, 0, 0, 0] } else { parse_pixel(value, channels)? };
        let offset = (y as usize * self.width as usize + x as usize) * channels;
        self.write_pixels(value.py())?[offset..offset + channels].copy_from_slice(&pixel[..channels]);
        Ok(())
    }

    pub fn putdata(&mut self, data: &Bound<'_, PyAny>, scale: f64, offset: f64) -> PyResult<()> {
        self.ensure_owned()?;
        self.raw_data()?;
        let length = data.len()?;
        if length > self.width as usize * self.height as usize {
            return Err(PyTypeError::new_err("too many data entries"));
        }
        if length == 0 && !self.mode.is_float() && !self.mode.is_integer() {
            return Ok(());
        }
        if self.bit_depth == 8
            && self.mode.channels() == 1
            && scale == 1.0
            && offset == 0.0
            && let Ok(bytes) = data.cast_exact::<PyBytes>()
        {
            let source = bytes.as_bytes();
            let pixels = self.pixels.as_mut().unwrap();
            if length == pixels.len() && Arc::strong_count(pixels) > 1 {
                *pixels = Arc::new(data.py().detach(|| crate::buffers::copy(source))?);
            } else {
                let target = &mut self.write_pixels(data.py())?[..length];
                data.py().detach(|| target.copy_from_slice(source));
            }
            return Ok(());
        }
        if crate::samples::supported(self) && !self.mode.is_integer() {
            let channels = self.mode.channels();
            let maximum = crate::samples::maximum(self);
            let pixels = self.write_pixels(data.py())?;
            for i in 0..length {
                let start = i * channels * 2;
                put_sample_pixel(&mut pixels[start..start + channels * 2], &data.get_item(i)?, channels, maximum, scale, offset)?;
            }
            return Ok(());
        }
        if self.mode.is_float() && self.mode.channels() > 1 {
            let channels = self.mode.channels();
            let stride = channels * 4;
            let pixels = self.write_pixels(data.py())?;
            for (i, item) in data.try_iter()?.enumerate() {
                if i >= length {
                    return Err(PyTypeError::new_err("too many data entries"));
                }
                let item = item?;
                let mut values = [0.0; 4];
                let values = &mut values[..channels];
                if let Ok(tuple) = item.cast_exact::<PyTuple>()
                    && tuple.len() == channels
                {
                    for (c, value) in values.iter_mut().enumerate() {
                        *value = tuple.get_borrowed_item(c)?.extract::<f64>()?;
                    }
                } else {
                    let samples: Vec<f64> = item.extract()?;
                    if samples.len() != channels {
                        return Err(PyValueError::new_err("wrong number of color components"));
                    }
                    values.copy_from_slice(&samples);
                }
                for (&value, dst) in values.iter().zip(pixels[i * stride..(i + 1) * stride].as_chunks_mut::<4>().0) {
                    *dst = ((value * scale + offset) as f32).to_le_bytes();
                }
            }
            return Ok(());
        }
        if self.mode == PixelMode::F {
            let pixels = self.write_pixels(data.py())?;
            for (i, item) in data.try_iter()?.enumerate() {
                if i >= length {
                    return Err(PyTypeError::new_err("too many data entries"));
                }
                let value = (item?.extract::<f64>()? * scale + offset) as f32;
                pixels[i * 4..i * 4 + 4].copy_from_slice(&value.to_le_bytes());
            }
            return Ok(());
        }
        if self.mode.is_integer() {
            let mode = self.mode;
            let stride = mode.sample_bytes();
            let pixels = self.write_pixels(data.py())?;
            for (i, item) in data.try_iter()?.enumerate() {
                if i >= length {
                    return Err(PyTypeError::new_err("too many data entries"));
                }
                let value = (item?.extract::<f64>()? * scale + offset) as i64;
                let bytes = integer_bytes(mode, value);
                pixels[i * stride..(i + 1) * stride].copy_from_slice(&bytes[..stride]);
            }
            return Ok(());
        }
        self.pixel_data()?;
        let mode = self.mode;
        let pixels = self.write_pixels(data.py())?;
        match mode {
            PixelMode::One | PixelMode::L => put_pixels::<1>(pixels, data, length, scale, offset),
            PixelMode::La | PixelMode::PremulLa | PixelMode::Pa => put_pixels::<2>(pixels, data, length, scale, offset),
            PixelMode::Rgb | PixelMode::Hsv | PixelMode::YCbCr | PixelMode::Lab => put_pixels::<3>(pixels, data, length, scale, offset),
            PixelMode::Rgba | PixelMode::PremulRgba | PixelMode::Rgbx | PixelMode::Cmyk => put_pixels::<4>(pixels, data, length, scale, offset),
            _ => unreachable!(),
        }
    }

    pub fn point_float(&self, function: &Bound<'_, PyAny>) -> PyResult<Self> {
        if self.mode != PixelMode::F {
            return Err(PyValueError::new_err("point operation requires F mode"));
        }
        let source = self.raw_data()?;
        let mut pixels = crate::buffers::reserved(source.len())?;
        for bytes in source.as_chunks::<4>().0 {
            let value = function.call1((float_value(bytes),))?.extract::<f64>()? as f32;
            pixels.extend_from_slice(&value.to_le_bytes());
        }
        Self::from_float_bytes(self.width, self.height, pixels)
    }

    pub fn getbbox(&self, py: Python<'_>, alpha_only: bool) -> PyResult<Option<(u32, u32, u32, u32)>> {
        let pixels = self.raw_data()?;
        let channels = self.mode.channels();
        let sample_bytes = if self.mode.is_wide_scalar() {
            self.mode.sample_bytes()
        } else if self.bit_depth == 8 {
            1
        } else {
            2
        };
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
                let values = if alpha_only
                    && matches!(
                        self.mode,
                        PixelMode::Rgba | PixelMode::PremulRgba | PixelMode::RgbaF | PixelMode::La | PixelMode::PremulLa | PixelMode::Pa
                    ) {
                    &pixel[(channels - 1) * sample_bytes..]
                } else {
                    pixel
                };
                if matches!(self.mode, PixelMode::RgbF | PixelMode::RgbaF) {
                    values.as_chunks::<4>().0.iter().any(|bytes| float_value(bytes) != 0.0)
                } else {
                    values.iter().any(|&v| v != 0)
                }
            };
            let mut rows = pixels.chunks_exact(row_bytes);
            let top = rows.position(|row| row.chunks_exact(stride).any(occupied))?;
            let bottom = pixels.chunks_exact(row_bytes).rposition(|row| row.chunks_exact(stride).any(occupied)).unwrap();
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
    pub fn bit_depth(&self) -> u8 {
        self.bit_depth
    }

    #[getter]
    pub fn mode(&self) -> &'static str {
        if self.mode == PixelMode::Pa {
            "PA"
        } else if self.palette.is_some() {
            "P"
        } else {
            self.mode.as_str()
        }
    }

    #[getter]
    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    #[getter]
    pub fn width(&self) -> u32 {
        self.width
    }

    #[getter]
    pub fn height(&self) -> u32 {
        self.height
    }

    #[getter]
    pub fn format(&self) -> Option<&str> {
        self.format.as_deref()
    }

    #[getter]
    pub fn info(&self, py: Python<'_>) -> Py<PyDict> {
        PyDict::new(py).unbind()
    }

    pub fn load(&self) -> PyResult<()> {
        self.raw_data().map(|_| ())
    }

    #[staticmethod]
    fn _mapped_layout(mode: &str, size: (u32, u32)) -> PyResult<Self> {
        // The Python mapping supplies pixels separately.
        let palette = (mode == "P").then(|| (PixelMode::Rgb, Vec::new()));
        let mode = if mode == "P" { PixelMode::L } else { PixelMode::parse(mode)? };
        let bit_depth = (mode.sample_bytes() * 8) as u8;
        expected_len(size.0, size.1, mode)?
            .checked_mul(mode.sample_bytes())
            .ok_or_else(|| PyValueError::new_err("image dimensions are too large"))?;
        Ok(Self { width: size.0, height: size.1, mode, pixels: None, bit_depth, palette, .. })
    }

    fn _write_mapped_rows(&self, data: &Bound<'_, PyMemoryView>, mode: &str, stride: usize, orientation: i8) -> PyResult<()> {
        let source = self.raw_data()?;
        if self.mode.as_str() != mode && !(self.mode == PixelMode::L && mode == "P") {
            return Err(PyValueError::new_err("mapped buffer mode does not match"));
        }
        let row_bytes = self.width as usize * self.raw_bytes_per_pixel();
        let height = self.height as usize;
        let required = stride.checked_mul(height).ok_or_else(|| PyValueError::new_err("image dimensions are too large"))?;
        let buffer = PyBuffer::<u8>::get(data.as_any())?;
        if stride < row_bytes
            || !matches!(orientation, -1 | 1)
            || !buffer.is_c_contiguous()
            || buffer.dimensions() != 1
            || buffer.len_bytes() != required
        {
            return Err(PyValueError::new_err("invalid mapped buffer layout"));
        }
        if source.is_empty() {
            return Ok(());
        }
        let py = data.py();
        if stride == row_bytes && orientation == 1 {
            return buffer.copy_from_slice(py, source);
        }
        for (row, source) in source.chunks_exact(row_bytes).enumerate() {
            let row = if orientation == 1 { row } else { height - 1 - row };
            let offset = row * stride;
            let view = data.get_item(PySlice::new(py, offset as isize, (offset + row_bytes) as isize, 1))?;
            PyBuffer::<u8>::get(&view)?.copy_from_slice(py, source)?;
        }
        Ok(())
    }

    fn _replace_pixels(&mut self, mut replacement: PyRefMut<'_, Self>) -> PyResult<()> {
        if self.size() != replacement.size() || self.mode != replacement.mode || self.bit_depth != replacement.bit_depth {
            return Err(PyValueError::new_err("replacement image layout does not match"));
        }
        self.raw_data()?;
        replacement.raw_data()?;
        std::mem::swap(&mut self.pixels, &mut replacement.pixels);
        std::mem::swap(&mut self.shared_pixels, &mut replacement.shared_pixels);
        std::mem::swap(&mut self.sample_validation, &mut replacement.sample_validation);
        Ok(())
    }

    fn _copy_palette(&mut self, source: &Self) {
        self.copy_palette_from(source);
    }

    fn _bilevel_array_bytes(&self, py: Python<'_>) -> PyResult<Py<PyBytes>> {
        let source = self.raw_data()?;
        PyBytes::new_with(py, source.len(), |output| {
            py.detach(|| crate::pixels::bilevel_to_l_into(source, output));
            Ok(())
        })
        .map(Bound::unbind)
    }

    fn _writable_samples<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyByteArray>> {
        Ok(PyByteArray::new(py, self.raw_data()?))
    }

    pub fn pixels_equal(&self, py: Python<'_>, other: &Self) -> PyResult<bool> {
        let first = self.raw_data()?;
        let second = other.raw_data()?;
        if self.mode != other.mode || self.size() != other.size() || self.bit_depth != other.bit_depth {
            return Ok(false);
        }
        if std::ptr::eq(first, second) {
            return Ok(true);
        }
        Ok(py.detach(|| if self.mode == PixelMode::One { first.iter().zip(second).all(|(&a, &b)| (a != 0) == (b != 0)) } else { first == second }))
    }

    pub fn copy(&self, py: Python<'_>) -> PyResult<Self> {
        let source = self.raw_data()?;
        let pixels = if let Some(pixels) = &self.pixels {
            py.detach(|| Arc::clone(pixels))
        } else {
            let mut pixels = crate::buffers::reserved::<u8>(source.len())?;
            let chunk = 256 * 1024;
            py.detach(|| {
                if source.len() < 4 << 20 {
                    pixels.extend_from_slice(source);
                    return;
                }
                crate::parallel::chunks_mut_above(&mut pixels.spare_capacity_mut()[..source.len()], chunk, 4 << 20, |i, dst| {
                    dst.write_copy_of_slice(&source[i * chunk..i * chunk + dst.len()]);
                });
            });
            // SAFETY: Every byte is initialized above, including any partial final partition.
            unsafe { pixels.set_len(source.len()) };
            Arc::new(pixels)
        };
        Ok(Self {
            width: self.width,
            height: self.height,
            mode: self.mode,
            pixels: Some(pixels),
            bit_depth: self.bit_depth,
            palette: self.palette.as_ref().map(|(mode, data)| crate::buffers::copy(data).map(|data| (*mode, data))).transpose()?,
            ..
        })
    }

    pub fn getpixel(&self, py: Python<'_>, xy: (i64, i64)) -> PyResult<Py<PyAny>> {
        let pixels = self.raw_data()?;
        let x = if xy.0 < 0 { xy.0 + i64::from(self.width) } else { xy.0 };
        let y = if xy.1 < 0 { xy.1 + i64::from(self.height) } else { xy.1 };
        if x < 0 || y < 0 || x >= i64::from(self.width) || y >= i64::from(self.height) {
            return Err(PyIndexError::new_err("image index out of range"));
        }
        let channels = self.mode.channels();
        let offset = (y as usize * self.width as usize + x as usize) * channels;
        if self.mode.is_float() && channels > 1 {
            return Ok(PyTuple::new(py, (0..channels).map(|c| float_value(&pixels[(offset + c) * 4..(offset + c + 1) * 4])))?.into_any().unbind());
        }
        if self.mode == PixelMode::F {
            return Ok(float_value(&pixels[offset * 4..offset * 4 + 4]).into_pyobject(py)?.into_any().unbind());
        }
        if self.mode.is_integer() {
            let start = offset * self.mode.sample_bytes();
            return Ok(integer_value(self.mode, &pixels[start..start + self.mode.sample_bytes()]).into_pyobject(py)?.into_any().unbind());
        }
        if self.bit_depth > 8 {
            let sample = |i: usize| u16::from_le_bytes([pixels[2 * i], pixels[2 * i + 1]]);
            return Ok(match self.mode {
                PixelMode::One | PixelMode::L => sample(offset).into_pyobject(py)?.into_any().unbind(),
                PixelMode::La | PixelMode::PremulLa | PixelMode::Pa => (sample(offset), sample(offset + 1)).into_pyobject(py)?.into_any().unbind(),
                PixelMode::Rgb | PixelMode::Hsv | PixelMode::YCbCr | PixelMode::Lab => {
                    (sample(offset), sample(offset + 1), sample(offset + 2)).into_pyobject(py)?.into_any().unbind()
                }
                PixelMode::Rgba | PixelMode::PremulRgba | PixelMode::Rgbx | PixelMode::Cmyk => {
                    (sample(offset), sample(offset + 1), sample(offset + 2), sample(offset + 3)).into_pyobject(py)?.into_any().unbind()
                }
                _ => unreachable!(),
            });
        }
        Ok(match self.mode {
            PixelMode::One | PixelMode::L => pixels[offset].into_pyobject(py)?.into_any().unbind(),
            PixelMode::La | PixelMode::PremulLa | PixelMode::Pa => (pixels[offset], pixels[offset + 1]).into_pyobject(py)?.into_any().unbind(),
            PixelMode::Rgb | PixelMode::Hsv | PixelMode::YCbCr | PixelMode::Lab => {
                (pixels[offset], pixels[offset + 1], pixels[offset + 2]).into_pyobject(py)?.into_any().unbind()
            }
            PixelMode::Rgba | PixelMode::PremulRgba | PixelMode::Rgbx | PixelMode::Cmyk => {
                (pixels[offset], pixels[offset + 1], pixels[offset + 2], pixels[offset + 3]).into_pyobject(py)?.into_any().unbind()
            }
            _ => unreachable!(),
        })
    }

    pub fn palette_data<'py>(&self, py: Python<'py>) -> Option<(&'static str, Bound<'py, PyBytes>)> {
        self.palette.as_ref().map(|(mode, data)| (mode.as_str(), PyBytes::new(py, data)))
    }

    pub fn set_palette(&mut self, mode: &str, data: std::borrow::Cow<'_, [u8]>) -> PyResult<()> {
        self.pixel_data()?;
        let mode = PixelMode::parse(mode)?;
        if !matches!(self.mode, PixelMode::L | PixelMode::Pa)
            || !matches!(mode, PixelMode::Rgb | PixelMode::Rgba)
            || data.len() > 256 * mode.channels()
            || !data.len().is_multiple_of(mode.channels())
        {
            return Err(PyValueError::new_err("invalid palette"));
        }
        if let Some((old_mode, old_data)) = &mut self.palette {
            *old_mode = mode;
            if old_data.as_slice() != data.as_ref() {
                old_data.clear();
                old_data.extend_from_slice(&data);
            }
        } else {
            self.palette = Some((mode, data.into_owned()));
        }
        Ok(())
    }

    #[pyo3(signature = (alpha=None, index=None))]
    pub fn set_palette_alpha(&mut self, alpha: Option<std::borrow::Cow<'_, [u8]>>, index: Option<u8>) -> PyResult<()> {
        self.pixel_data()?;
        let output = self.alpha_palette(alpha.as_deref(), index)?;
        self.palette = Some((PixelMode::Rgba, output));
        Ok(())
    }

    pub fn close(&mut self) {
        self.pixels = None;
        self.shared_pixels = None;
    }

    pub fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    pub fn __exit__(&mut self, _exc_type: &Bound<'_, PyAny>, _exc_value: &Bound<'_, PyAny>, _traceback: &Bound<'_, PyAny>) {
        self.close();
    }

    pub fn __repr__(&self) -> String {
        format!("<blanket.Image.Image image mode={} size={}x{}>", self.mode.as_str(), self.width, self.height)
    }
}
