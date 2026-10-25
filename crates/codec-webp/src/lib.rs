use std::borrow::Borrow;

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
    let (pixels, width, height) =
        if features.has_alpha { webpx::decode_rgba(data) } else { webpx::decode_rgb(data) }.map_err(|error| error.to_string())?;
    Image::from_pixels(width, height, if features.has_alpha { PixelMode::Rgba } else { PixelMode::Rgb }, pixels, Some("WEBP".into()))
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

pub fn backend_version() -> String {
    let (major, minor, patch) = webpx::version();
    format!("{major}.{minor}.{patch}")
}

pub fn decode_sequence(data: &[u8]) -> Result<blanket_codec_common::Sequence, String> {
    use blanket_codec_common::{Frame, Sequence, validate_frame_budget};
    let info = webpx::ImageInfo::from_webp(data).map_err(|e| e.to_string())?;
    if !info.has_animation {
        return decode_webp(data).map(Sequence::single);
    }
    let mut decoder = webpx::AnimationDecoder::new(data).map_err(|e| e.to_string())?;
    let mut sequence = Sequence { loop_count: Some(decoder.info().loop_count), ..Sequence::default() };
    let mut timestamp = 0;
    let mut total = 0;
    while decoder.has_more_frames() {
        validate_frame_budget(&mut total, decoder.info().width, decoder.info().height)?;
        let frame = decoder.next_frame().map_err(|e| e.to_string())?.ok_or("missing WebP frame")?;
        let duration = frame.timestamp_ms.checked_sub(timestamp).filter(|d| *d >= 0).ok_or("invalid WebP timestamp")?;
        timestamp = frame.timestamp_ms;
        sequence.frames.push(Frame {
            image: Image::from_pixels(frame.width, frame.height, PixelMode::Rgba, frame.data, Some("WEBP".into())).map_err(|e| e.to_string())?,
            duration: Some(f64::from(duration)),
            disposal: 0,
            blend: 0,
        });
    }
    Ok(sequence)
}

pub fn encode_sequence(sequence: &blanket_codec_common::Sequence<impl Borrow<Image>>, options: SaveOptions) -> PyResult<Vec<u8>> {
    let first = sequence.frames[0].image.borrow();
    let mut encoder =
        webpx::AnimationEncoder::with_options(first.width, first.height, false, sequence.loop_count.unwrap_or(0)).map_err(codec_error)?;
    encoder.set_quality(f32::from(options.quality));
    encoder.set_lossless(options.lossless);
    let mut timestamp: i32 = 0;
    for frame in &sequence.frames {
        encoder.add_frame_rgba(frame.image.borrow().pixel_data()?, timestamp).map_err(codec_error)?;
        timestamp = timestamp.checked_add(frame.duration.unwrap_or(0.0) as i32).ok_or_else(|| PyValueError::new_err("WebP duration overflow"))?;
    }
    encoder.finish(timestamp).map_err(codec_error)
}
