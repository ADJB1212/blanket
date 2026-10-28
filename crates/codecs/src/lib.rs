pub mod codecs;
pub use blanket_metadata as metadata;
pub mod stream;
pub use blanket_codec_common::{CodecErrorKind, decode_error, encode_error};
pub use blanket_codec_heif::backend_info as heif_backend_info;
pub use blanket_codec_jxl::backend_version as jxl_backend_version;
pub use blanket_codec_png::stream::PngStreamWriter;
pub use blanket_codec_webp::backend_version as webp_backend_version;
pub use codecs::{ImageFormat, SaveOptions, open_bytes};

#[pyo3::pyfunction]
#[pyo3(signature = (data, method, crop, trim, preserve_metadata))]
pub fn _transform_jpeg<'py>(
    py: pyo3::Python<'py>, data: &[u8], method: Option<u8>, crop: Option<(usize, usize, usize, usize)>, trim: bool, preserve_metadata: bool,
) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::types::PyBytes>> {
    let output = py.detach(|| blanket_codec_jpeg::transform_jpeg(data, method, crop, trim, preserve_metadata))?;
    Ok(pyo3::types::PyBytes::new(py, &output))
}
