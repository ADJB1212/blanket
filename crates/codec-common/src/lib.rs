#![feature(default_field_values)]

use blanket_core::Image;
use pyo3::exceptions::PyOSError;
use pyo3::prelude::*;
use zerocopy::IntoBytes;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CodecErrorKind {
    Unsupported,
    Corrupt,
    MissingBackend,
    ResourceLimit,
    Other,
}

impl CodecErrorKind {
    pub fn into_pyerr(self, message: String, format: Option<&str>, operation: &str) -> PyErr {
        Python::attach(|py| {
            let result = (|| -> PyResult<PyErr> {
                let name = match self {
                    Self::Unsupported => "UnsupportedCodecError",
                    Self::Corrupt => "CorruptImageError",
                    Self::MissingBackend => "MissingBackendError",
                    Self::ResourceLimit => "CodecResourceLimitError",
                    Self::Other => "CodecError",
                };
                let kwargs = pyo3::types::PyDict::new(py);
                kwargs.set_item("format", format)?;
                kwargs.set_item("operation", operation)?;
                let exception = py.import("blanket.errors")?.getattr(name)?.call((message,), Some(&kwargs))?;
                Ok(PyErr::from_value(exception))
            })();
            result.unwrap_or_else(|error| error)
        })
    }
}

// Adapt diagnostic strings from backends that do not expose typed errors.
pub fn decode_error(message: String, format: Option<&str>, operation: &str) -> PyErr {
    diagnostic_kind(&message, CodecErrorKind::Corrupt).into_pyerr(message, format, operation)
}

pub fn encode_error(error: PyErr, format: &str) -> PyErr {
    Python::attach(|py| {
        if let Ok(class) = py.import("blanket.errors").and_then(|module| module.getattr("CodecError"))
            && error.value(py).is_instance(&class).unwrap_or(false)
        {
            return error;
        }
        if !error.is_instance_of::<PyOSError>(py) && !error.is_instance_of::<pyo3::exceptions::PyValueError>(py) {
            return error;
        }
        let message = error.value(py).str().map(|s| s.to_string()).unwrap_or_else(|_| error.to_string());
        let kind = diagnostic_kind(&message, CodecErrorKind::Other);
        if kind == CodecErrorKind::Other && error.is_instance_of::<pyo3::exceptions::PyValueError>(py) {
            return error;
        }
        let structured = kind.into_pyerr(message, Some(format), "encode");
        structured.set_cause(py, Some(error));
        structured
    })
}

/// A decoding failure with a machine-readable category and optional native cause.
#[derive(Debug)]
pub struct DecodeError {
    kind: CodecErrorKind,
    message: String,
    source: Option<Box<dyn std::error::Error + Send + Sync + pyo3::marker::Ungil>>,
}

impl DecodeError {
    /// Preserve a diagnostic chain when the native error cannot cross detached calls.
    pub fn with_diagnostic_source(error: impl std::error::Error) -> Self {
        fn snapshot(error: &dyn std::error::Error) -> DecodeError {
            let mut result = DecodeError::from(error.to_string());
            result.source = error.source().map(|source| Box::new(snapshot(source)) as _);
            result
        }
        let mut result = Self::from(error.to_string());
        result.source = Some(Box::new(snapshot(&error)));
        result
    }
    /// Category used when translating this error to a Python codec exception.
    pub fn kind(&self) -> CodecErrorKind {
        self.kind
    }

    /// Retain a native cause that can safely cross a detached Python call.
    pub fn with_source(error: impl std::error::Error + Send + Sync + pyo3::marker::Ungil + 'static) -> Self {
        let mut result = Self::from(error.to_string());
        result.source = Some(Box::new(error));
        result
    }

    /// Convert the diagnostic to the matching Python exception category.
    pub fn into_pyerr(self, format: Option<&str>) -> PyErr {
        self.kind.into_pyerr(self.message, format, "decode")
    }
}

impl From<String> for DecodeError {
    fn from(message: String) -> Self {
        Self { kind: diagnostic_kind(&message, CodecErrorKind::Corrupt), message, source: None }
    }
}

impl From<&str> for DecodeError {
    fn from(message: &str) -> Self {
        Self::from(message.to_owned())
    }
}

impl From<DecodeError> for String {
    fn from(error: DecodeError) -> Self {
        error.to_string()
    }
}

impl std::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for DecodeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.source.as_deref().map(|error| error as _)
    }
}

#[cfg(test)]
mod decode_error_tests {
    use std::error::Error;

    use super::*;

    #[test]
    fn decoding_errors_preserve_category_and_source() {
        let cause = "invalid".parse::<u32>().unwrap_err();
        let message = cause.to_string();
        let error = DecodeError::with_source(cause);
        assert_eq!(error.source().unwrap().to_string(), message);
        assert_eq!(DecodeError::from("unsupported channel layout").kind(), CodecErrorKind::Unsupported);
        assert_eq!(DecodeError::from("image exceeds Blanket limit").kind(), CodecErrorKind::ResourceLimit);
    }
}

fn diagnostic_kind(message: &str, fallback: CodecErrorKind) -> CodecErrorKind {
    let message = message.to_ascii_lowercase();
    if message.starts_with("eps reading requires ghostscript (gs):") {
        return CodecErrorKind::MissingBackend;
    }
    if message.contains("exceeds blanket limit")
        || message.contains("overflow addressable memory")
        || message.contains("frame pixel count overflow")
        || message.contains("memory limit")
        || message.contains("limits are exceeded")
        || message.contains("limit exceeded")
        || message.contains("exceeds streaming buffer limit")
    {
        CodecErrorKind::ResourceLimit
    } else if message.contains("unsupported")
        || message.contains("not supported")
        || message.contains("does not support")
        || message.contains("cannot write mode")
        || message.contains("read-only format")
        || message.contains("write-only format")
        || message.contains("high-bit-depth saving supports")
        || message.contains("encoding supports")
    {
        CodecErrorKind::Unsupported
    } else {
        fallback
    }
}

pub const MAX_IMAGE_PIXELS: usize = 178_956_970;

#[derive(Debug, PartialEq, Eq)]
pub struct ImageHeader {
    pub width: u32,
    pub height: u32,
    pub mode: &'static str,
    pub bit_depth: u8,
}

impl ImageHeader {
    pub fn new(width: u32, height: u32, mode: &'static str, bit_depth: u8) -> Result<Self, String> {
        if width == 0 || height == 0 {
            return Err("empty image header".into());
        }
        Ok(Self { width, height, mode, bit_depth })
    }
}

#[expect(unevaluated_default_field_value)]
pub struct Frame<I = Image> {
    pub image: I,
    pub duration: Option<f64> = None,
    pub disposal: u8 = 0,
    pub blend: u8 = 0,
}

#[expect(unevaluated_default_field_value)]
pub struct Sequence<I = Image> {
    pub frames: Vec<Frame<I>> = Vec::new(),
    pub loop_count: Option<u32> = None,
    pub default_image: bool = false,
}

impl<I> Default for Sequence<I> {
    fn default() -> Self {
        Self { .. }
    }
}

impl Sequence {
    pub fn single(image: Image) -> Self {
        Self { frames: vec![Frame { image, .. }], .. }
    }
}

#[derive(Default)]
pub struct SequenceInfo {
    pub count: usize,
    pub loop_count: Option<u32>,
    pub default_image: bool,
}

pub fn validate_frame_budget(total: &mut usize, width: u32, height: u32) -> Result<(), String> {
    validate_dimensions(width, height)?;
    *total = total.checked_add((width as usize) * (height as usize)).ok_or("frame pixel count overflow")?;
    if *total > MAX_IMAGE_PIXELS {
        return Err(format!("sequence exceeds Blanket limit of {MAX_IMAGE_PIXELS} decoded pixels"));
    }
    Ok(())
}

#[derive(Clone, Copy, Default)]
pub struct SaveOptions<'a> {
    pub quality: u8 = 90,
    pub compress_level: u8 = 6,
    pub lossless: bool = false,
    pub effort: u8 = 7,
    pub tiff_compression: u8 = 0,
    pub jpeg_subsampling: i8 = -1,
    pub jpeg_optimize: bool = false,
    pub jpeg_progressive: bool = false,
    pub jpeg_keep_rgb: bool = false,
    pub jpeg_smooth: i32 = 0,
    pub jpeg_qtables: &'a [[u16; 64]] = &[],
    pub jpeg_qtable_quality: i16 = -1,
    pub jpeg_restart_marker_blocks: u32 = 0,
    pub jpeg_restart_marker_rows: u32 = 0,
    pub jpeg_streamtype: i32 = 0,
    pub icc_profile: &'a [u8] = &[],
    pub exif: &'a [u8] = &[],
    pub xmp: &'a [u8] = &[],
}

pub fn validate_dimensions(width: u32, height: u32) -> Result<(), String> {
    let pixels = (width as usize).checked_mul(height as usize).ok_or_else(|| "image dimensions overflow addressable memory".to_owned())?;
    if pixels > MAX_IMAGE_PIXELS {
        return Err(format!("image size ({pixels} pixels) exceeds Blanket limit of {MAX_IMAGE_PIXELS} pixels"));
    }
    Ok(())
}

pub fn codec_error(error: impl std::fmt::Display) -> PyErr {
    PyOSError::new_err(error.to_string())
}

/// Borrow native-endian sample bytes without copying the sample buffer.
pub fn u16_bytes(samples: &[u16]) -> &[u8] {
    samples.as_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_diagnostics_have_stable_categories() {
        assert_eq!(diagnostic_kind("image size exceeds Blanket limit of pixels", CodecErrorKind::Corrupt), CodecErrorKind::ResourceLimit);
        assert_eq!(diagnostic_kind("unsupported HEIF bit depth: 9", CodecErrorKind::Corrupt), CodecErrorKind::Unsupported);
        assert_eq!(diagnostic_kind("truncated PNG data", CodecErrorKind::Corrupt), CodecErrorKind::Corrupt);
        assert_eq!(diagnostic_kind("invalid chunk offset overflow", CodecErrorKind::Corrupt), CodecErrorKind::Corrupt);
        assert_eq!(diagnostic_kind("encoder failed", CodecErrorKind::Other), CodecErrorKind::Other);
    }

    #[test]
    fn frame_and_sequence_defaults_do_not_require_default_images() {
        struct Raster;

        let frame = Frame { image: Raster, .. };
        assert_eq!(frame.duration, None);
        assert_eq!(frame.disposal, 0);
        assert_eq!(frame.blend, 0);

        for sequence in [Sequence::<Raster> { .. }, Sequence::<Raster>::default()] {
            assert!(sequence.frames.is_empty());
            assert_eq!(sequence.loop_count, None);
            assert!(!sequence.default_image);
        }
        let sequence = Sequence { frames: vec![frame], loop_count: Some(3), default_image: true };
        assert_eq!(sequence.frames.len(), 1);
        assert_eq!(sequence.loop_count, Some(3));
        assert!(sequence.default_image);
    }

    #[test]
    fn frame_budget_checks_cumulative_pixels_and_overflow() {
        let mut total = MAX_IMAGE_PIXELS - 2;
        assert!(validate_frame_budget(&mut total, 2, 1).is_ok());
        assert!(validate_frame_budget(&mut total, 1, 1).is_err());
        let mut overflowing = usize::MAX;
        assert!(validate_frame_budget(&mut overflowing, 1, 1).is_err());
        assert!(validate_frame_budget(&mut 0, u32::MAX, u32::MAX).is_err());
    }
}
