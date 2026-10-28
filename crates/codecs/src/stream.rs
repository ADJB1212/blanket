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
    fn new(format: &str) -> PyResult<Self> {
        let decoder = match format {
            "JXL" => Decoder::Jxl(JxlStreamDecoder::new().map_err(|message| decode_error(message, Some(format), "decode"))?),
            "JPEG" => Decoder::Jpeg(Box::new(JpegStreamDecoder::new().map_err(|message| decode_error(message, Some(format), "decode"))?)),
            _ => return Err(PyValueError::new_err("streaming decode supports JXL and JPEG")),
        };
        Ok(Self { decoder: Some(decoder) })
    }

    fn feed(&mut self, data: &[u8]) -> PyResult<()> {
        let decoder = self.decoder.as_mut().ok_or_else(|| PyValueError::new_err("closed decoder"))?;
        let format = decoder.format();
        let result = decoder.feed(data);
        if let Err(error) = result {
            self.decoder = None;
            return Err(decode_error(error, Some(format), "decode"));
        }
        Ok(())
    }

    #[getter]
    fn image(&mut self) -> PyResult<Option<Image>> {
        let decoder = self.decoder.as_mut().ok_or_else(|| PyValueError::new_err("closed decoder"))?;
        let format = decoder.format();
        decoder.image().map_err(|message| decode_error(message, Some(format), "decode"))
    }

    fn close(&mut self) -> PyResult<Image> {
        let mut decoder = self.decoder.take().ok_or_else(|| PyValueError::new_err("closed decoder"))?;
        let format = decoder.format();
        if !decoder.complete() {
            return Err(decode_error("truncated image stream".into(), Some(format), "decode"));
        }
        decoder
            .image()
            .map_err(|message| decode_error(message, Some(format), "decode"))?
            .ok_or_else(|| decode_error("image stream has no pixels".into(), Some(format), "decode"))
    }
}
