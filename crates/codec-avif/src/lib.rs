use blanket_codec_common::{SaveOptions, codec_error};
use blanket_codec_image::{color_type, decode_rust_image};
use blanket_core::{Image, PixelMode};
use image::{ExtendedColorType, ImageEncoder, ImageFormat as RustFormat};
use pyo3::prelude::*;

pub fn decode_avif(data: &[u8]) -> Result<Image, String> {
    decode_rust_image(data, RustFormat::Avif, "AVIF")
}

pub fn encode_avif(image: &Image, pixels: &[u8], options: SaveOptions) -> PyResult<Vec<u8>> {
    let rgb;
    let (pixels, color) = if image.mode == PixelMode::L {
        rgb = pixels.iter().flat_map(|v| [*v; 3]).collect::<Vec<_>>();
        (rgb.as_slice(), ExtendedColorType::Rgb8)
    } else {
        (pixels, color_type(image.mode))
    };
    let mut output = Vec::new();
    image::codecs::avif::AvifEncoder::new_with_speed_quality(&mut output, 11 - options.effort, options.quality)
        .write_image(pixels, image.width, image.height, color)
        .map_err(codec_error)?;
    Ok(output)
}
