use std::io::{self, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use blanket_codec_common::{codec_error, validate_dimensions};
use pyo3::exceptions::{PyOSError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

struct Sink {
    object: Py<PyAny>,
    error: Arc<Mutex<Option<PyErr>>>,
    active: Arc<AtomicBool>,
}

impl Write for Sink {
    fn write(&mut self, data: &[u8]) -> io::Result<usize> {
        if !self.active.load(Ordering::Relaxed) {
            return Err(io::Error::other("closed PNG output"));
        }
        Python::attach(|py| {
            let result = self.object.call_method1(py, "write", (PyBytes::new(py, data),)).and_then(|count| count.extract::<usize>(py));
            match result {
                Ok(count) if count <= data.len() => Ok(count),
                Ok(_) => {
                    self.active.store(false, Ordering::Relaxed);
                    Err(io::Error::new(io::ErrorKind::InvalidData, "write() returned an invalid byte count"))
                }
                Err(error) => {
                    self.active.store(false, Ordering::Relaxed);
                    *self.error.lock().expect("sink error lock") = Some(error);
                    Err(io::Error::other("PNG output write failed"))
                }
            }
        })
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[pyclass]
pub struct PngStreamWriter {
    writer: Option<png::StreamWriter<'static, Sink>>,
    error: Arc<Mutex<Option<PyErr>>>,
    active: Arc<AtomicBool>,
    stride: usize,
    remaining: usize,
    depth: u8,
}

impl Drop for PngStreamWriter {
    fn drop(&mut self) {
        self.active.store(false, Ordering::Relaxed);
    }
}

fn layout(mode: &str, size: (u32, u32), bit_depth: u8, compress_level: u8) -> PyResult<(png::ColorType, usize, u8)> {
    let (width, height) = size;
    if width == 0 || height == 0 {
        return Err(PyValueError::new_err("PNG dimensions must be positive"));
    }
    validate_dimensions(width, height).map_err(PyValueError::new_err)?;
    if !matches!(bit_depth, 8 | 10 | 12 | 16) || compress_level > 9 {
        return Err(PyValueError::new_err("invalid bit depth or compression level"));
    }
    let (color, channels) = match mode {
        "1" if bit_depth == 8 => (png::ColorType::Grayscale, 1),
        "L" => (png::ColorType::Grayscale, 1),
        "LA" if bit_depth == 8 => (png::ColorType::GrayscaleAlpha, 2),
        "RGB" => (png::ColorType::Rgb, 3),
        "RGBA" => (png::ColorType::Rgba, 4),
        _ => return Err(PyValueError::new_err("unsupported mode for streaming PNG")),
    };
    Ok((color, channels, if mode == "1" { 1 } else { bit_depth }))
}

#[pymethods]
impl PngStreamWriter {
    #[staticmethod]
    fn validate(mode: &str, size: (u32, u32), bit_depth: u8, compress_level: u8) -> PyResult<()> {
        layout(mode, size, bit_depth, compress_level).map(|_| ())
    }

    #[new]
    fn new(sink: Py<PyAny>, mode: &str, size: (u32, u32), bit_depth: u8, compress_level: u8) -> PyResult<Self> {
        let (width, height) = size;
        let (color, channels, depth) = layout(mode, size, bit_depth, compress_level)?;
        let stride = if depth == 1 { (width as usize).div_ceil(8) } else { width as usize * channels * if depth > 8 { 2 } else { 1 } };
        let error = Arc::new(Mutex::new(None));
        let active = Arc::new(AtomicBool::new(true));
        let mut encoder = png::Encoder::new(Sink { object: sink, error: Arc::clone(&error), active: Arc::clone(&active) }, width, height);
        encoder.set_color(color);
        encoder.set_depth(if depth == 1 {
            png::BitDepth::One
        } else if depth > 8 {
            png::BitDepth::Sixteen
        } else {
            png::BitDepth::Eight
        });
        encoder.set_deflate_compression(if compress_level == 0 {
            png::DeflateCompression::NoCompression
        } else {
            png::DeflateCompression::Level(compress_level)
        });
        let result = (|| {
            let mut writer = encoder.write_header()?;
            if matches!(depth, 10 | 12) {
                writer.write_chunk(png::chunk::sBIT, &vec![depth; channels])?;
            }
            writer.into_stream_writer()
        })();
        let writer = result.map_err(|err| error.lock().expect("sink error lock").take().unwrap_or_else(|| codec_error(err)))?;
        Ok(Self { writer: Some(writer), error, active, stride, remaining: stride * height as usize, depth })
    }

    fn write(&mut self, data: &[u8]) -> PyResult<()> {
        if self.writer.is_none() {
            return Err(PyValueError::new_err("closed PNG writer"));
        }
        if !data.len().is_multiple_of(self.stride) || data.len() > self.remaining {
            return Err(PyValueError::new_err("write requires complete rows within the declared image size"));
        }
        if self.depth > 8 && data.as_chunks::<2>().0.iter().any(|&sample| u16::from_le_bytes(sample) as u32 >= (1_u32 << self.depth)) {
            return Err(PyValueError::new_err("sample exceeds bit depth"));
        }
        let result = if self.depth > 8 {
            let writer = self.writer.as_mut().expect("writer checked above");
            let mut result = Ok(());
            let mut converted = vec![0; self.stride];
            for row in data.chunks_exact(self.stride) {
                blanket_core::simd::samples_to_be16(row, &mut converted, self.depth);
                if let Err(error) = writer.write_all(&converted) {
                    result = Err(error);
                    break;
                }
            }
            result
        } else {
            self.writer.as_mut().expect("writer checked above").write_all(data)
        };
        if let Err(error) = result {
            let error = self.error.lock().expect("sink error lock").take().unwrap_or_else(|| codec_error(error));
            self.active.store(false, Ordering::Relaxed);
            self.writer = None;
            return Err(error);
        }
        self.remaining -= data.len();
        Ok(())
    }

    fn close(&mut self) -> PyResult<()> {
        let Some(writer) = self.writer.take() else { return Ok(()) };
        if self.remaining > 0 {
            self.active.store(false, Ordering::Relaxed);
            return Err(PyOSError::new_err("incomplete PNG rows"));
        }
        writer.finish().map_err(|error| self.error.lock().expect("sink error lock").take().unwrap_or_else(|| codec_error(error)))?;
        let error = self.error.lock().expect("sink error lock").take();
        if let Some(error) = error {
            return Err(error);
        }
        Ok(())
    }

    fn abort(&mut self) {
        self.active.store(false, Ordering::Relaxed);
        self.writer = None;
    }
}
