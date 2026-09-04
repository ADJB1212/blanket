use pyo3::exceptions::PyValueError;
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
    pub(crate) format: Option<String>,
}

impl Image {
    pub(crate) fn from_pixels(
        width: u32,
        height: u32,
        mode: PixelMode,
        pixels: Vec<u8>,
        format: Option<String>,
    ) -> PyResult<Self> {
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
            format,
        })
    }

    pub(crate) fn pixel_data(&self) -> PyResult<&[u8]> {
        self.pixels
            .as_deref()
            .ok_or_else(|| PyValueError::new_err("operation on closed image"))
    }
}

#[pymethods]
impl Image {
    #[getter]
    fn mode(&self) -> &'static str {
        self.mode.as_str()
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
        self.pixel_data().map(|_| ())
    }

    fn close(&mut self) {
        self.pixels = None;
    }

    fn convert(&self, py: Python<'_>, mode: &str) -> PyResult<Self> {
        let destination = PixelMode::parse(mode)?;
        let source = self.mode;
        let pixels = self.pixel_data()?;
        let converted = py.detach(|| convert_pixels(pixels, source, destination));
        Self::from_pixels(self.width, self.height, destination, converted, None)
    }

    fn tobytes(&self, py: Python<'_>) -> PyResult<Py<PyBytes>> {
        Ok(PyBytes::new(py, self.pixel_data()?).unbind())
    }

    #[pyo3(name = "_encode")]
    fn encode(
        &self,
        py: Python<'_>,
        format: &str,
        quality: u8,
        compress_level: u8,
        lossless: bool,
        effort: u8,
    ) -> PyResult<Py<PyBytes>> {
        let format = ImageFormat::parse(format)?;
        let options = SaveOptions {
            quality,
            compress_level,
            lossless,
            effort,
        };
        let encoded = py.detach(|| codecs::encode(self, format, options))?;
        Ok(PyBytes::new(py, &encoded).unbind())
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __exit__(
        &mut self,
        _exc_type: &Bound<'_, PyAny>,
        _exc_value: &Bound<'_, PyAny>,
        _traceback: &Bound<'_, PyAny>,
    ) {
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

#[pyfunction]
pub(crate) fn frombytes(mode: &str, size: (u32, u32), data: &[u8]) -> PyResult<Image> {
    let mode = PixelMode::parse(mode)?;
    Image::from_pixels(size.0, size.1, mode, data.to_vec(), None)
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
        assert_eq!(
            convert_pixels(&rgb, PixelMode::Rgb, PixelMode::L),
            [76, 150, 29, 30]
        );
    }

    #[test]
    fn expands_and_drops_channels() {
        assert_eq!(
            convert_pixels(&[7, 23], PixelMode::L, PixelMode::Rgba),
            [7, 7, 7, 255, 23, 23, 23, 255]
        );
        assert_eq!(
            convert_pixels(&[1, 2, 3, 4], PixelMode::Rgba, PixelMode::Rgb),
            [1, 2, 3]
        );
    }
}
