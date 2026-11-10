use blanket_codec_common::{DecodeError, decode_error};
use blanket_codec_jpeg::stream::JpegStreamDecoder;
use blanket_codec_jxl::stream::JxlStreamDecoder;
use blanket_core::Image;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyclass(unsendable)]
#[derive(Debug)]
pub struct StreamingDecoder {
    decoder: Option<Decoder>,
}

#[derive(Debug)]
enum Decoder {
    Jxl(JxlStreamDecoder),
    Jpeg(Box<JpegStreamDecoder>),
}

// SAFETY: native decoder state has no Python references. Detach keeps it on this thread.
unsafe impl pyo3::marker::Ungil for Decoder {}

impl Decoder {
    fn format(&self) -> &'static str {
        match self {
            Self::Jxl(_) => "JXL",
            Self::Jpeg(_) => "JPEG",
        }
    }
    fn feed(&mut self, data: &[u8]) -> Result<(), DecodeError> {
        match self {
            Self::Jxl(decoder) => decoder.feed(data),
            Self::Jpeg(decoder) => decoder.feed(data).map_err(DecodeError::from),
        }
    }

    fn image(&mut self) -> Result<Option<Image>, DecodeError> {
        match self {
            Self::Jxl(decoder) => decoder.image(),
            Self::Jpeg(decoder) => decoder.image().map_err(DecodeError::from),
        }
    }

    fn complete(&self) -> bool {
        match self {
            Self::Jxl(decoder) => decoder.is_complete(),
            Self::Jpeg(decoder) => decoder.complete,
        }
    }

    fn finish(self) -> Result<Option<Image>, DecodeError> {
        match self {
            Self::Jxl(decoder) => decoder.finish(),
            Self::Jpeg(decoder) => decoder.finish().map_err(DecodeError::from),
        }
    }
}

#[pymethods]
impl StreamingDecoder {
    #[new]
    fn new(py: Python<'_>, format: &str) -> PyResult<Self> {
        if !matches!(format, "JXL" | "JPEG") {
            return Err(PyValueError::new_err("streaming decode supports JXL and JPEG"));
        }
        let decoder = py
            .detach(|| match format {
                "JXL" => JxlStreamDecoder::new().map(Decoder::Jxl),
                _ => JpegStreamDecoder::new().map(|decoder| Decoder::Jpeg(Box::new(decoder))).map_err(DecodeError::from),
            })
            .map_err(|error| error.into_pyerr(Some(format)))?;
        Ok(Self { decoder: Some(decoder) })
    }

    fn feed(&mut self, py: Python<'_>, data: &[u8]) -> PyResult<()> {
        let decoder = self.decoder.as_mut().ok_or_else(|| PyValueError::new_err("closed decoder"))?;
        let format = decoder.format();
        let result = py.detach(|| decoder.feed(data));
        if let Err(error) = result {
            self.decoder = None;
            return Err(error.into_pyerr(Some(format)));
        }
        Ok(())
    }

    #[getter]
    fn image(&mut self, py: Python<'_>) -> PyResult<Option<Image>> {
        let decoder = self.decoder.as_mut().ok_or_else(|| PyValueError::new_err("closed decoder"))?;
        let format = decoder.format();
        py.detach(|| decoder.image()).map_err(|error| error.into_pyerr(Some(format)))
    }

    fn close(&mut self, py: Python<'_>) -> PyResult<Image> {
        let decoder = self.decoder.take().ok_or_else(|| PyValueError::new_err("closed decoder"))?;
        let format = decoder.format();
        if !decoder.complete() {
            return Err(decode_error("truncated image stream".into(), Some(format), "decode"));
        }
        py.detach(move || decoder.finish())
            .map_err(|error| error.into_pyerr(Some(format)))?
            .ok_or_else(|| decode_error("image stream has no pixels".into(), Some(format), "decode"))
    }
}
