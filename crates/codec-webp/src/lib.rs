use blanket_codec_common::{SaveOptions, codec_error, validate_dimensions};
use blanket_codec_image::decode_rust_image;
use blanket_core::{Image, PixelMode};
use image::ImageFormat as RustFormat;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub fn decode_webp(data: &[u8]) -> Result<Image, String> {
    let features = webpx::ImageInfo::from_webp(data).map_err(|error| error.to_string())?;
    validate_dimensions(features.width, features.height)?;
    // Preserve the existing first-frame behavior for animated containers.
    if features.has_animation {
        return decode_rust_image(data, RustFormat::WebP, "WEBP");
    }
    let (pixels, width, height) = if features.has_alpha {
        webpx::decode_rgba(data)
    } else {
        webpx::decode_rgb(data)
    }
    .map_err(|error| error.to_string())?;
    Image::from_pixels(
        width,
        height,
        if features.has_alpha { PixelMode::Rgba } else { PixelMode::Rgb },
        pixels,
        Some("WEBP".into()),
    )
    .map_err(|error| error.to_string())
}

pub fn encode_webp(image: &Image, pixels: &[u8], options: SaveOptions) -> PyResult<Vec<u8>> {
    let rgb;
    let encoder = match image.mode {
        PixelMode::L => {
            rgb = pixels.iter().flat_map(|v| [*v; 3]).collect::<Vec<_>>();
            webpx::Encoder::new_rgb(&rgb, image.width, image.height)
        }
        PixelMode::Rgb => webpx::Encoder::new_rgb(pixels, image.width, image.height),
        PixelMode::Rgba => webpx::Encoder::new_rgba(pixels, image.width, image.height),
        _ => return Err(PyValueError::new_err("unsupported mode for WebP")),
    };
    encoder
        .config(webpx::EncoderConfig::new().thread_level(1))
        .lossless(options.lossless)
        .quality(if options.lossless { 75.0 } else { f32::from(options.quality) })
        .encode(webpx::Unstoppable)
        .map_err(codec_error)
}
