pub mod codecs;
pub use blanket_metadata as metadata;
pub mod stream;
pub use blanket_codec_common::{CodecErrorKind, decode_error, encode_error};
pub use blanket_codec_heif::backend_info as heif_backend_info;
pub use blanket_codec_jxl::backend_version as jxl_backend_version;
pub use blanket_codec_png::encode_png_to;
pub use blanket_codec_png::stream::PngStreamWriter;
pub use blanket_codec_webp::backend_version as webp_backend_version;
pub use codecs::{ImageFormat, SaveOptions, open_bytes};

#[pyo3::pyfunction]
pub fn _draft_jpeg(py: pyo3::Python<'_>, data: &[u8], scale: u32, mode: &str) -> pyo3::PyResult<blanket_core::Image> {
    py.detach(|| blanket_codec_jpeg::decode_scaled(data, scale, mode)).map_err(|message| decode_error(message, Some("JPEG"), "decode"))
}

#[pyo3::pyfunction]
#[pyo3(signature = (data, method, crop, trim, preserve_metadata))]
pub fn _transform_jpeg<'py>(
    py: pyo3::Python<'py>, data: &[u8], method: Option<u8>, crop: Option<(usize, usize, usize, usize)>, trim: bool, preserve_metadata: bool,
) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::types::PyBytes>> {
    let options = blanket_codec_jpeg::TransformOptions { method, crop, trim, preserve_metadata };
    let output = py.detach(|| blanket_codec_jpeg::transform_jpeg(data, options))?;
    Ok(pyo3::types::PyBytes::new(py, &output))
}
