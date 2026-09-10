use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{PyIndexError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

use crate::codecs::{self, ImageFormat, SaveOptions};

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

    fn copy(&self) -> PyResult<Self> {
        self.raw_data()?;
        let mut result = self.clone();
        result.format = None;
        Ok(result)
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
            return self.copy();
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
    fn encode(&self, py: Python<'_>, format: &str, quality: u8, compress_level: u8, lossless: bool, effort: u8) -> PyResult<Py<PyBytes>> {
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
            let encoded = py.detach(|| codecs::encode(&expanded, format, options))?;
            return Ok(PyBytes::new(py, &encoded).unbind());
        }
        let encoded = py.detach(|| codecs::encode(self, format, options))?;
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
