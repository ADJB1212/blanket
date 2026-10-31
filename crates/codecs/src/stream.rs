use blanket_codec_common::decode_error;
use blanket_codec_jpeg::stream::JpegStreamDecoder;
use blanket_codec_jxl::stream::JxlStreamDecoder;
use blanket_core::Image;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyclass(unsendable)]
pub struct StreamingDecoder {
    decoder: Option<Decoder>,
}

enum Decoder {
    Jxl(JxlStreamDecoder),
    Jpeg(Box<JpegStreamDecoder>),
}

// Native decoder state has no Python references. Detach keeps it on this thread.
unsafe impl pyo3::marker::Ungil for Decoder {}

impl Decoder {
    fn format(&self) -> &'static str {
        match self {
            Self::Jxl(_) => "JXL",
            Self::Jpeg(_) => "JPEG",
        }
    }
    fn feed(&mut self, data: &[u8]) -> Result<(), String> {
        match self {
            Self::Jxl(decoder) => decoder.feed(data),
            Self::Jpeg(decoder) => decoder.feed(data),
        }
    }

    fn image(&mut self) -> Result<Option<Image>, String> {
        match self {
            Self::Jxl(decoder) => decoder.image(),
            Self::Jpeg(decoder) => decoder.image(),
        }
    }

    fn complete(&self) -> bool {
        match self {
            Self::Jxl(decoder) => decoder.complete,
            Self::Jpeg(decoder) => decoder.complete,
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
                _ => JpegStreamDecoder::new().map(|decoder| Decoder::Jpeg(Box::new(decoder))),
            })
            .map_err(|message| decode_error(message, Some(format), "decode"))?;
        Ok(Self { decoder: Some(decoder) })
    }

    fn feed(&mut self, py: Python<'_>, data: &[u8]) -> PyResult<()> {
        let decoder = self.decoder.as_mut().ok_or_else(|| PyValueError::new_err("closed decoder"))?;
        let format = decoder.format();
        let result = py.detach(|| decoder.feed(data));
        if let Err(error) = result {
            self.decoder = None;
            return Err(decode_error(error, Some(format), "decode"));
        }
        Ok(())
    }

    #[getter]
    fn image(&mut self, py: Python<'_>) -> PyResult<Option<Image>> {
        let decoder = self.decoder.as_mut().ok_or_else(|| PyValueError::new_err("closed decoder"))?;
        let format = decoder.format();
        py.detach(|| decoder.image()).map_err(|message| decode_error(message, Some(format), "decode"))
    }

    fn close(&mut self, py: Python<'_>) -> PyResult<Image> {
        let mut decoder = self.decoder.take().ok_or_else(|| PyValueError::new_err("closed decoder"))?;
        let format = decoder.format();
        if !decoder.complete() {
            return Err(decode_error("truncated image stream".into(), Some(format), "decode"));
        }
        py.detach(move || decoder.image())
            .map_err(|message| decode_error(message, Some(format), "decode"))?
            .ok_or_else(|| decode_error("image stream has no pixels".into(), Some(format), "decode"))
    }
}
