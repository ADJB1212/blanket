use blanket_codec_common::codec_error;
use blanket_codec_image::{color_type, decode_rust_image};
use blanket_core::{Image, PixelMode};
use image::{ExtendedColorType, ImageFormat as RustFormat};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub fn decode_gif(data: &[u8]) -> Result<Image, String> {
    decode_rust_image(data, RustFormat::Gif, "GIF")
}

pub fn encode_gif(image: &Image, pixels: &[u8]) -> PyResult<Vec<u8>> {
    if image.width == 0 || image.height == 0 {
        return Err(PyValueError::new_err("cannot encode an empty image"));
    }
    if image.width > 65535 || image.height > 65535 {
        return Err(PyValueError::new_err("GIF dimensions must be between 1 and 65535"));
    }
    let mut output = Vec::new();
    let rgb;
    let (pixels, color) = if image.mode == PixelMode::L {
        rgb = pixels.iter().flat_map(|v| [*v; 3]).collect::<Vec<_>>();
        (rgb.as_slice(), ExtendedColorType::Rgb8)
    } else {
        (pixels, color_type(image.mode))
    };
    image::codecs::gif::GifEncoder::new(&mut output)
        .encode(pixels, image.width, image.height, color)
        .map_err(codec_error)?;

    Ok(output)
}
