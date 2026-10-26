use blanket_codec_jpeg::stream::JpegStreamDecoder;
use blanket_codec_jxl::stream::JxlStreamDecoder;
use blanket_core::Image;
use pyo3::exceptions::{PyOSError, PyValueError};
use pyo3::prelude::*;

#[pyclass(unsendable)]
pub struct StreamingDecoder {
    decoder: Option<Decoder>,
}

enum Decoder {
    Jxl(JxlStreamDecoder),
    Jpeg(Box<JpegStreamDecoder>),
}

impl Decoder {
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
    fn new(format: &str) -> PyResult<Self> {
        let decoder = match format {
            "JXL" => Decoder::Jxl(JxlStreamDecoder::new().map_err(PyOSError::new_err)?),
            "JPEG" => Decoder::Jpeg(Box::new(JpegStreamDecoder::new().map_err(PyOSError::new_err)?)),
            _ => return Err(PyValueError::new_err("streaming decode supports JXL and JPEG")),
        };
        Ok(Self { decoder: Some(decoder) })
    }

    fn feed(&mut self, data: &[u8]) -> PyResult<()> {
        let result = self.decoder.as_mut().ok_or_else(|| PyValueError::new_err("closed decoder"))?.feed(data);
        if let Err(error) = result {
            self.decoder = None;
            return Err(PyOSError::new_err(error));
        }
        Ok(())
    }

    #[getter]
    fn image(&mut self) -> PyResult<Option<Image>> {
        self.decoder.as_mut().ok_or_else(|| PyValueError::new_err("closed decoder"))?.image().map_err(PyOSError::new_err)
    }

    fn close(&mut self) -> PyResult<Image> {
        let mut decoder = self.decoder.take().ok_or_else(|| PyValueError::new_err("closed decoder"))?;
        if !decoder.complete() {
            return Err(PyOSError::new_err("truncated image stream"));
        }
        decoder.image().map_err(PyOSError::new_err)?.ok_or_else(|| PyOSError::new_err("image stream has no pixels"))
    }
}
