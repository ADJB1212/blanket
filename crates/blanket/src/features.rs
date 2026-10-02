use pyo3::prelude::*;

#[pyfunction]
pub fn _feature_info(feature: &str) -> Option<(bool, Option<String>)> {
    let version = match feature {
        "blanket" => Some(env!("CARGO_PKG_VERSION").to_owned()),
        "png" => Some(format!("png {}", env!("BLANKET_BACKEND_PNG"))),
        "jpg" | "libjpeg_turbo" => Some(format!("turbojpeg {}", env!("BLANKET_BACKEND_TURBOJPEG"))),
        "libtiff" => Some(format!("tiff {}", env!("BLANKET_BACKEND_TIFF"))),
        "gif" => Some(format!("gif {}", env!("BLANKET_BACKEND_GIF"))),
        "bmp" | "ico" => Some(format!("image {}", env!("BLANKET_BACKEND_IMAGE"))),
        "avif" => Some(format!(
            "ravif {}; dav1d {}",
            env!("BLANKET_BACKEND_RAVIF"),
            env!("BLANKET_BACKEND_DAV1D")
        )),
        "jxl" => Some(blanket_codecs::jxl_backend_version()),
        "webp" => Some(blanket_codecs::webp_backend_version()),
        "littlecms2" => Some(blanket_core::littlecms_version()),
        "libheif" | "heif" | "heif_decoder" | "heif_encoder" => {
            let (decode, encode, version) = blanket_codecs::heif_backend_info();
            let available = match feature {
                "libheif" => true,
                "heif_decoder" => decode,
                "heif_encoder" => encode,
                _ => decode && encode,
            };
            return Some((available, available.then_some(version)));
        }
        "pdf" => None,
        "pil" | "tkinter" | "freetype2" | "jpg_2000" | "zlib" | "raqm" | "fribidi" | "harfbuzz" | "mozjpeg" | "zlib_ng" | "libimagequant" | "xcb" => {
            return Some((false, None));
        }
        _ => return None,
    };
    Some((true, version))
}
