use blanket_core::{Image, PixelMode, UnidentifiedImageError};
use pyo3::exceptions::{PyOSError, PyValueError};
use pyo3::prelude::*;

use blanket_codec_avif::{decode_avif, encode_avif};
use blanket_codec_bmp_ico::{decode_bmp, decode_ico, encode_bmp, encode_ico};
pub use blanket_codec_common::SaveOptions;
use blanket_codec_gif::{decode_gif, encode_gif};
pub use blanket_codec_heif::{PreparedHeif, encode_heif_with_preset};
use blanket_codec_heif::{decode_heif, encode_heif};
pub use blanket_codec_jpeg::{JpegCoding, encode_jpeg_variant};
use blanket_codec_jpeg::{decode_jpeg, encode_jpeg};
pub use blanket_codec_jxl::{JXL_PARALLEL_MIN_BYTES, JxlThreads, PreparedJxl, jxl_candidate_threads};
use blanket_codec_jxl::{decode_jxl, encode_jxl};
use blanket_codec_pdf::encode_pdf;
use blanket_codec_png::{decode_png, encode_wide_png};
pub use blanket_codec_png::{encode_one_png, encode_palette_png, encode_png};
use blanket_codec_tiff::{decode_tiff, encode_float_tiff, encode_lab_tiff, encode_raw_tiff, encode_tiff, encode_wide_tiff, is_dng};
use blanket_codec_webp::{decode_webp, encode_webp};

const PNG_SIGNATURE: &[u8] = b"\x89PNG\r\n\x1a\n";
const JXL_CONTAINER_SIGNATURE: &[u8] = b"\0\0\0\x0cJXL \r\n\x87\n";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ImageFormat {
    Bmp,
    Gif,
    Ico,
    Png,
    Jpeg,
    Jxl,
    Tiff,
    Webp,
    Heif,
    Avif,
    Pdf,
}

impl ImageFormat {
    pub fn parse(value: &str) -> PyResult<Self> {
        match value.to_ascii_uppercase().as_str() {
            "BMP" => Ok(Self::Bmp),
            "GIF" => Ok(Self::Gif),
            "ICO" => Ok(Self::Ico),
            "PDF" => Ok(Self::Pdf),
            "AVIF" => Ok(Self::Avif),
            "TIFF" | "TIF" => Ok(Self::Tiff),
            "WEBP" => Ok(Self::Webp),
            "HEIF" | "HEIC" => Ok(Self::Heif),
            "PNG" => Ok(Self::Png),
            "JPEG" | "JPG" => Ok(Self::Jpeg),
            "JXL" | "JPEGXL" | "JPEG XL" => Ok(Self::Jxl),
            _ => Err(PyValueError::new_err(format!(
                "unsupported image format {value:?}; expected PNG, JPEG, JXL, TIFF, WEBP, HEIF, AVIF, PDF, BMP, GIF, or ICO"
            ))),
        }
    }

    const fn as_str(self) -> &'static str {
        match self {
            Self::Bmp => "BMP",
            Self::Gif => "GIF",
            Self::Ico => "ICO",
            Self::Pdf => "PDF",
            Self::Avif => "AVIF",
            Self::Tiff => "TIFF",
            Self::Webp => "WEBP",
            Self::Heif => "HEIF",
            Self::Png => "PNG",
            Self::Jpeg => "JPEG",
            Self::Jxl => "JXL",
        }
    }

    fn detect(data: &[u8]) -> Option<Self> {
        if data.starts_with(b"BM") {
            Some(Self::Bmp)
        } else if data.starts_with(b"GIF87a") || data.starts_with(b"GIF89a") {
            Some(Self::Gif)
        } else if data.starts_with(b"\0\0\x01\0") {
            Some(Self::Ico)
        } else if data.starts_with(PNG_SIGNATURE) {
            Some(Self::Png)
        } else if data.starts_with(&[0xff, 0xd8, 0xff]) {
            Some(Self::Jpeg)
        } else if data.starts_with(&[0xff, 0x0a]) || data.starts_with(JXL_CONTAINER_SIGNATURE) {
            Some(Self::Jxl)
        } else if data.starts_with(b"II*\0") || data.starts_with(b"MM\0*") || data.starts_with(b"II+\0") || data.starts_with(b"MM\0+") {
            if is_dng(data) { None } else { Some(Self::Tiff) }
        } else if data.starts_with(b"RIFF") && data.get(8..12) == Some(b"WEBP") {
            Some(Self::Webp)
        } else if is_avif(data) {
            Some(Self::Avif)
        } else if is_heif(data) {
            Some(Self::Heif)
        } else {
            None
        }
    }
}
#[pyfunction]
pub fn open_bytes(py: Python<'_>, data: &[u8], formats: Option<Vec<String>>) -> PyResult<Image> {
    let format = ImageFormat::detect(data).ok_or_else(|| UnidentifiedImageError::new_err("cannot identify image file"))?;
    if let Some(formats) = formats {
        let allowed = formats
            .iter()
            .filter_map(|name| ImageFormat::parse(name).ok())
            .any(|candidate| candidate == format);
        if !allowed {
            return Err(UnidentifiedImageError::new_err(format!(
                "image format {} is not in formats",
                format.as_str()
            )));
        }
    }

    py.detach(|| decode(data, format)).map_err(UnidentifiedImageError::new_err)
}

pub fn decode(data: &[u8], format: ImageFormat) -> Result<Image, String> {
    match format {
        ImageFormat::Bmp => decode_bmp(data),
        ImageFormat::Gif => decode_gif(data),
        ImageFormat::Ico => decode_ico(data),
        ImageFormat::Pdf => Err("PDF is a write-only format".into()),
        ImageFormat::Avif => decode_avif(data),
        ImageFormat::Tiff => decode_tiff(data),
        ImageFormat::Webp => decode_webp(data),
        ImageFormat::Heif => decode_heif(data),
        ImageFormat::Png => decode_png(data),
        ImageFormat::Jpeg => decode_jpeg(data),
        ImageFormat::Jxl => decode_jxl(data),
    }
}

pub fn encode(image: &Image, format: ImageFormat, options: SaveOptions) -> PyResult<Vec<u8>> {
    if image.mode == PixelMode::Lab {
        if format != ImageFormat::Tiff {
            return Err(PyOSError::new_err(format!("cannot write mode LAB as {}", format.as_str())));
        }
        return encode_lab_tiff(image);
    }
    if matches!(image.mode, PixelMode::Cmyk | PixelMode::YCbCr) {
        match (image.mode, format) {
            (_, ImageFormat::Jpeg) => return encode_jpeg(image, image.pixel_data()?, options.quality),
            (PixelMode::Cmyk, ImageFormat::Tiff) => return encode_raw_tiff(image, image.pixel_data()?),
            (PixelMode::Cmyk, ImageFormat::Pdf) => return encode_pdf(image, image.pixel_data()?),
            _ => {}
        }
        return Err(PyOSError::new_err(format!(
            "cannot write mode {} as {}",
            image.mode.as_str(),
            format.as_str()
        )));
    }
    if image.mode == PixelMode::F {
        if format != ImageFormat::Tiff {
            return Err(PyOSError::new_err(format!("cannot write mode F as {}", format.as_str())));
        }
        return encode_float_tiff(image);
    }
    if format == ImageFormat::Heif {
        return encode_heif(image, options);
    }
    if image.bit_depth > 8 {
        return encode_wide(image, format, options);
    }
    let pixels = image.pixel_data()?;
    match format {
        ImageFormat::Bmp => encode_bmp(image, pixels),
        ImageFormat::Gif => encode_gif(image, pixels),
        ImageFormat::Ico => encode_ico(image, pixels),
        ImageFormat::Pdf => encode_pdf(image, pixels),
        ImageFormat::Avif => encode_avif(image, pixels, options),
        ImageFormat::Heif => unreachable!(),
        ImageFormat::Tiff => encode_tiff(image, pixels),
        ImageFormat::Webp => encode_webp(image, pixels, options),
        ImageFormat::Png => encode_png(image, pixels, options.compress_level),
        ImageFormat::Jpeg => encode_jpeg(image, pixels, options.quality),
        ImageFormat::Jxl => encode_jxl(image, pixels, options),
    }
}
fn is_avif(data: &[u8]) -> bool {
    has_brand(data, &[b"avif", b"avis"])
}

fn has_brand(data: &[u8], accepted: &[&[u8; 4]]) -> bool {
    if data.get(4..8) != Some(b"ftyp") || data.len() < 16 {
        return false;
    }
    let size = u32::from_be_bytes(data[..4].try_into().unwrap()) as usize;
    if size < 16 || size > data.len() || !size.is_multiple_of(4) {
        return false;
    }
    std::iter::once(&data[8..12])
        .chain(data[16..size].as_chunks::<4>().0.iter().map(|v| v.as_slice()))
        .any(|brand| accepted.iter().any(|candidate| brand == candidate.as_slice()))
}

fn is_heif(data: &[u8]) -> bool {
    if data.get(4..8) != Some(b"ftyp") || data.len() < 16 {
        return false;
    }
    let size = u32::from_be_bytes(data[..4].try_into().unwrap()) as usize;
    if size < 16 || size > data.len() || !size.is_multiple_of(4) {
        return false;
    }
    let brands = std::iter::once(&data[8..12]).chain(data[16..size].as_chunks::<4>().0.iter().map(|v| v.as_slice()));
    brands.into_iter().any(|brand| {
        matches!(
            brand,
            b"heic" | b"heix" | b"hevc" | b"hevx" | b"heim" | b"heis" | b"hevm" | b"hevs" | b"mif1" | b"msf1"
        )
    })
}

fn encode_wide(image: &Image, format: ImageFormat, options: SaveOptions) -> PyResult<Vec<u8>> {
    let pixels = image.raw_data()?;
    if !matches!(format, ImageFormat::Png | ImageFormat::Tiff | ImageFormat::Jxl) {
        return Err(PyValueError::new_err(
            "high-bit-depth saving supports HEIF, PNG, TIFF, or JXL; convert to bit_depth=8 explicitly for this format",
        ));
    }
    if format == ImageFormat::Jxl {
        return PreparedJxl::new(image)?.encode(options);
    }
    let samples = blanket_core::simd::normalize_u16(pixels, image.bit_depth);
    match format {
        ImageFormat::Png => encode_wide_png(image, samples, options),
        ImageFormat::Tiff => encode_wide_tiff(image, samples),
        _ => unreachable!(),
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn detects_avif_brands_before_generic_heif() {
        for data in [
            b"\0\0\0\x14ftypavif\0\0\0\0mif1".as_slice(),
            b"\0\0\0\x14ftypmif1\0\0\0\0avif",
            b"\0\0\0\x10ftypavis\0\0\0\0",
        ] {
            assert_eq!(ImageFormat::detect(data), Some(ImageFormat::Avif));
        }
        assert_eq!(ImageFormat::detect(b"\0\0\0\x20ftypavif\0\0\0\0"), None);
        assert_eq!(ImageFormat::detect(b"\0\0\0\x10ftypxxxxavif"), None);
        assert_eq!(ImageFormat::detect(b"\0\0\0\x10ftypxxxx\0\0\0\0avif"), None);
    }

    #[test]
    fn detects_supported_signatures() {
        assert_eq!(ImageFormat::detect(PNG_SIGNATURE), Some(ImageFormat::Png));
        assert_eq!(ImageFormat::detect(&[0xff, 0xd8, 0xff]), Some(ImageFormat::Jpeg));
        assert_eq!(ImageFormat::detect(&[0xff, 0x0a]), Some(ImageFormat::Jxl));
        assert_eq!(ImageFormat::detect(b"not an image"), None);
        assert_eq!(ImageFormat::detect(b"\0\0\0\x14ftypxxxx\0\0\0\0heix"), Some(ImageFormat::Heif));
        assert_eq!(ImageFormat::detect(b"\0\0\0\x10ftypheic\0\0\0\0"), Some(ImageFormat::Heif));
        assert_eq!(ImageFormat::detect(b"\0\0\0\x20ftypheic\0\0\0\0"), None);
    }
}
