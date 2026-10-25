use blanket_codec_jxl::stream::JxlStreamDecoder;
use blanket_core::Image;
use pyo3::exceptions::{PyOSError, PyValueError};
use pyo3::prelude::*;

#[pyclass(unsendable)]
pub struct StreamingDecoder {
    decoder: Option<JxlStreamDecoder>,
}

#[pymethods]
impl StreamingDecoder {
    #[new]
    fn new(format: &str) -> PyResult<Self> {
        if format != "JXL" {
            return Err(PyValueError::new_err("streaming decode supports JXL"));
        }
        let decoder = JxlStreamDecoder::new().map_err(PyOSError::new_err)?;
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
    fn image(&self) -> PyResult<Option<Image>> {
        self.decoder.as_ref().ok_or_else(|| PyValueError::new_err("closed decoder"))?.image().map_err(PyOSError::new_err)
    }

    fn close(&mut self) -> PyResult<Image> {
        let decoder = self.decoder.take().ok_or_else(|| PyValueError::new_err("closed decoder"))?;
        if !decoder.complete {
            return Err(PyOSError::new_err("truncated image stream"));
        }
        decoder.image().map_err(PyOSError::new_err)?.ok_or_else(|| PyOSError::new_err("image stream has no pixels"))
    }
}
