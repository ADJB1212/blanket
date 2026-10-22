use blanket_codec_avif::{decode_avif, encode_avif};
use blanket_codec_bmp_ico::{decode_bmp, decode_ico, encode_bmp, encode_ico};
pub use blanket_codec_common::SaveOptions;
use blanket_codec_extended::{decode_dds, decode_eps, decode_pnm, decode_qoi, decode_tga, encode_extended, encode_wide_pnm, is_tga};
use blanket_codec_gif::{decode_gif, encode_gif};
pub use blanket_codec_heif::{PreparedHeif, encode_heif_with_preset};
use blanket_codec_heif::{decode_heif, encode_heif};
pub use blanket_codec_jpeg::{JpegCoding, encode_jpeg_variant};
use blanket_codec_jpeg::{decode_jpeg, encode_jpeg};
use blanket_codec_jpeg2000::{decode_jpeg2000, encode_jpeg2000};
pub use blanket_codec_jxl::{JXL_PARALLEL_MIN_BYTES, JxlThreads, PreparedJxl, jxl_candidate_threads};
use blanket_codec_jxl::{decode_jxl, encode_jxl};
use blanket_codec_pdf::encode_pdf;
use blanket_codec_png::{decode_png, encode_wide_png};
pub use blanket_codec_png::{encode_one_png, encode_palette_png, encode_png};
use blanket_codec_tiff::{decode_tiff, encode_float_tiff, encode_lab_tiff, encode_raw_tiff, encode_tiff, encode_wide_tiff, is_dng};
use blanket_codec_webp::{decode_webp, encode_webp};
use blanket_core::{Image, PixelMode, UnidentifiedImageError};
use pyo3::exceptions::{PyOSError, PyValueError};
use pyo3::prelude::*;

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
    Ppm,
    Pam,
    Tga,
    Dds,
    Eps,
    Qoi,
    Jpeg2000,
    Pcx,
    Sgi,
    Xbm,
    Xpm,
    Fits,
    Spider,
    Pixar,
    Msp,
    Palm,
    Sun,
}

impl ImageFormat {
    pub fn parse(value: &str) -> PyResult<Self> {
        match value.to_ascii_uppercase().as_str() {
            "PCX" => Ok(Self::Pcx),
            "SGI" => Ok(Self::Sgi),
            "XBM" => Ok(Self::Xbm),
            "XPM" => Ok(Self::Xpm),
            "FITS" => Ok(Self::Fits),
            "SPIDER" => Ok(Self::Spider),
            "PIXAR" => Ok(Self::Pixar),
            "MSP" => Ok(Self::Msp),
            "PALM" => Ok(Self::Palm),
            "SUN" => Ok(Self::Sun),
            "BMP" => Ok(Self::Bmp),
            "PPM" | "PNM" | "PGM" | "PBM" => Ok(Self::Ppm),
            "PAM" => Ok(Self::Pam),
            "TGA" | "TARGA" => Ok(Self::Tga),
            "DDS" => Ok(Self::Dds),
            "EPS" | "PS" => Ok(Self::Eps),
            "QOI" => Ok(Self::Qoi),
            "JPEG2000" | "JP2" | "J2K" | "J2C" | "JPC" => Ok(Self::Jpeg2000),
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
            _ => Err(PyValueError::new_err(format!("unsupported image format {value:?}"))),
        }
    }

    const fn as_str(self) -> &'static str {
        match self {
            Self::Pcx => "PCX",
            Self::Sgi => "SGI",
            Self::Xbm => "XBM",
            Self::Xpm => "XPM",
            Self::Fits => "FITS",
            Self::Spider => "SPIDER",
            Self::Pixar => "PIXAR",
            Self::Msp => "MSP",
            Self::Palm => "PALM",
            Self::Sun => "SUN",
            Self::Bmp => "BMP",
            Self::Ppm => "PPM",
            Self::Pam => "PAM",
            Self::Tga => "TGA",
            Self::Dds => "DDS",
            Self::Eps => "EPS",
            Self::Qoi => "QOI",
            Self::Jpeg2000 => "JPEG2000",
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
        if let Some(format) = blanket_codec_extended::legacy::detect(data) {
            return Self::parse(format).ok();
        }
        if data.starts_with(b"qoif") {
            Some(Self::Qoi)
        } else if data.starts_with(b"DDS ") {
            Some(Self::Dds)
        } else if data.starts_with(b"%!PS") || data.starts_with(b"\xc5\xd0\xd3\xc6") {
            Some(Self::Eps)
        } else if data.starts_with(b"\0\0\0\x0cjP  \r\n\x87\n") || data.starts_with(b"\xff\x4f\xff\x51") {
            Some(Self::Jpeg2000)
        } else if data.len() >= 3 && data[0] == b'P' && matches!(data[1], b'1'..=b'7') && data[2].is_ascii_whitespace() {
            Some(Self::Ppm)
        } else if data.starts_with(b"BM") {
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
        } else if is_tga(data) {
            Some(Self::Tga)
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
            .any(|candidate| candidate == format || (candidate == ImageFormat::Pam && format == ImageFormat::Ppm && data.starts_with(b"P7")));
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
        ImageFormat::Pcx
        | ImageFormat::Sgi
        | ImageFormat::Xbm
        | ImageFormat::Xpm
        | ImageFormat::Fits
        | ImageFormat::Spider
        | ImageFormat::Pixar
        | ImageFormat::Msp
        | ImageFormat::Palm
        | ImageFormat::Sun => blanket_codec_extended::legacy::decode(data, format.as_str()),
        ImageFormat::Ppm | ImageFormat::Pam => decode_pnm(data),
        ImageFormat::Tga => decode_tga(data),
        ImageFormat::Dds => decode_dds(data),
        ImageFormat::Eps => decode_eps(data),
        ImageFormat::Qoi => decode_qoi(data),
        ImageFormat::Jpeg2000 => decode_jpeg2000(data),
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
    if format == ImageFormat::Tiff && options.tiff_compression != 0 {
        return blanket_codec_tiff::encode_compressed_tiff(image, options);
    }
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
    if format == ImageFormat::Jpeg2000 {
        return encode_jpeg2000(image);
    }
    if image.bit_depth > 8 {
        if matches!(format, ImageFormat::Ppm | ImageFormat::Pam) {
            return encode_wide_pnm(image, format == ImageFormat::Pam);
        }
        return encode_wide(image, format, options);
    }
    let pixels = image.pixel_data()?;
    match format {
        ImageFormat::Pcx
        | ImageFormat::Sgi
        | ImageFormat::Xbm
        | ImageFormat::Xpm
        | ImageFormat::Fits
        | ImageFormat::Spider
        | ImageFormat::Pixar
        | ImageFormat::Msp
        | ImageFormat::Palm
        | ImageFormat::Sun => blanket_codec_extended::legacy::encode(image, pixels, format.as_str()).map_err(PyOSError::new_err),
        ImageFormat::Ppm | ImageFormat::Pam | ImageFormat::Tga | ImageFormat::Dds | ImageFormat::Eps | ImageFormat::Qoi => {
            encode_extended(image, pixels, format.as_str())
        }
        ImageFormat::Jpeg2000 => unreachable!(),
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
        .chain(data[16..size].as_chunks::<4>().0.iter().map(<[u8; 4]>::as_slice))
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
    let brands = std::iter::once(&data[8..12]).chain(data[16..size].as_chunks::<4>().0.iter().map(<[u8; 4]>::as_slice));
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
    match format {
        ImageFormat::Png => encode_wide_png(image, blanket_core::simd::normalize_u16(pixels, image.bit_depth), options),
        ImageFormat::Tiff => encode_wide_tiff(image),
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
        for (data, format) in [
            (b"P1\n".as_slice(), ImageFormat::Ppm),
            (b"P7\n", ImageFormat::Ppm),
            (b"qoif", ImageFormat::Qoi),
            (b"DDS ", ImageFormat::Dds),
            (b"%!PS-Adobe-3.0 EPSF-3.0", ImageFormat::Eps),
            (b"\0\0\0\x0cjP  \r\n\x87\n", ImageFormat::Jpeg2000),
            (b"\xff\x4f\xff\x51", ImageFormat::Jpeg2000),
        ] {
            assert_eq!(ImageFormat::detect(data), Some(format));
        }
        assert_eq!(ImageFormat::detect(PNG_SIGNATURE), Some(ImageFormat::Png));
        assert_eq!(ImageFormat::detect(&[0xff, 0xd8, 0xff]), Some(ImageFormat::Jpeg));
        assert_eq!(ImageFormat::detect(&[0xff, 0x0a]), Some(ImageFormat::Jxl));
        assert_eq!(ImageFormat::detect(b"not an image"), None);
        assert_eq!(ImageFormat::detect(b"\0\0\0\x14ftypxxxx\0\0\0\0heix"), Some(ImageFormat::Heif));
        assert_eq!(ImageFormat::detect(b"\0\0\0\x10ftypheic\0\0\0\0"), Some(ImageFormat::Heif));
        assert_eq!(ImageFormat::detect(b"\0\0\0\x20ftypheic\0\0\0\0"), None);
    }
}
