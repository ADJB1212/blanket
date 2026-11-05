use std::borrow::Borrow;

use blanket_codec_common::{SaveOptions, codec_error, validate_dimensions};
use blanket_codec_image::decode_rust_image;
use blanket_core::{Image, PixelMode};
use image::ImageFormat as RustFormat;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub fn inspect(data: &[u8]) -> Result<blanket_codec_common::ImageHeader, String> {
    let info = webpx::ImageInfo::from_webp(data).map_err(|e| e.to_string())?;
    blanket_codec_common::ImageHeader::new(info.width, info.height, if info.has_alpha { "RGBA" } else { "RGB" }, 8)
}

pub fn decode_webp(data: &[u8]) -> Result<Image, String> {
    let features = webpx::ImageInfo::from_webp(data).map_err(|error| error.to_string())?;
    validate_dimensions(features.width, features.height)?;
    // Preserve the existing first-frame behavior for animated containers.
    if features.has_animation {
        return decode_rust_image(data, RustFormat::WebP, "WEBP");
    }
    let limits = webpx::Limits::default();
    limits.check_input_size(data.len() as u64).map_err(|error| format!("WebP limit exceeded: {error}"))?;
    limits.check_still_image(features.width, features.height).map_err(|error| format!("WebP limit exceeded: {error}"))?;
    let mode = if features.has_alpha { PixelMode::Rgba } else { PixelMode::Rgb };
    let stride = (features.width as usize).checked_mul(mode.channels()).ok_or("WebP row size overflow")?;
    let length = stride.checked_mul(features.height as usize).ok_or("WebP buffer size overflow")?;
    let stride = i32::try_from(stride).map_err(|error| error.to_string())?;
    let mut pixels = Vec::<u8>::new();
    pixels.try_reserve_exact(length).map_err(|error| error.to_string())?;
    // SAFETY: dimensions and capacity cover every packed output row. The Into
    // APIs write the entire buffer on success and never retain its pointer.
    let output = unsafe {
        if features.has_alpha {
            libwebp_sys::WebPDecodeRGBAInto(data.as_ptr(), data.len(), pixels.as_mut_ptr(), length, stride)
        } else {
            libwebp_sys::WebPDecodeRGBInto(data.as_ptr(), data.len(), pixels.as_mut_ptr(), length, stride)
        }
    };
    if output.is_null() {
        return Err("invalid WebP bitstream".into());
    }
    // SAFETY: successful decoding initialized all length bytes above.
    unsafe { pixels.set_len(length) };
    Image::from_pixels(features.width, features.height, mode, pixels, Some("WEBP".into())).map_err(|error| error.to_string())
}

pub fn encode_webp(image: &Image, pixels: &[u8], options: SaveOptions) -> PyResult<Vec<u8>> {
    let argb = (image.mode == PixelMode::L).then(|| pixels.iter().map(|&v| 0xff00_0000 | (u32::from(v) * 0x0001_0101)).collect::<Vec<_>>());
    let encoder = match image.mode {
        PixelMode::L => webpx::Encoder::new_argb(argb.as_deref().expect("grayscale pixels expanded above"), image.width, image.height),
        PixelMode::Rgb => webpx::Encoder::new_rgb(pixels, image.width, image.height),
        PixelMode::Rgba => webpx::Encoder::new_rgba(pixels, image.width, image.height),
        _ => return Err(PyValueError::new_err("unsupported mode for WebP")),
    };
    let encoder = encoder.config(webpx::EncoderConfig::new().thread_level(1)).lossless(options.lossless).quality(if options.lossless {
        75.0
    } else {
        f32::from(options.quality)
    });
    if image.mode != PixelMode::L {
        return encoder.encode(webpx::Unstoppable).map_err(codec_error);
    }
    let encoded = encoder.encode_owned(webpx::Unstoppable).map_err(codec_error)?;
    drop(argb);
    Ok(encoded.to_vec())
}

pub fn backend_version() -> String {
    let (major, minor, patch) = webpx::version();
    format!("{major}.{minor}.{patch}")
}

pub fn decode_sequence(data: &[u8]) -> Result<blanket_codec_common::Sequence, String> {
    decode_frames(data, None)
}

pub fn index_sequence(data: &[u8]) -> Result<blanket_codec_common::SequenceInfo, String> {
    use blanket_codec_common::{SequenceInfo, validate_frame_budget};
    let info = webpx::ImageInfo::from_webp(data).map_err(|e| e.to_string())?;
    validate_dimensions(info.width, info.height)?;
    let (count, loop_count) = if info.has_animation {
        let input = libwebp_sys::WebPData { bytes: data.as_ptr(), size: data.len() };
        // SAFETY: Input outlives the demuxer; a nonnull demuxer is deleted after querying it.
        unsafe {
            let demuxer = libwebp_sys::WebPDemuxInternal(&raw const input, 0, std::ptr::null_mut(), libwebp_sys::WEBP_DEMUX_ABI_VERSION as i32);
            if demuxer.is_null() {
                return Err("invalid WebP animation".into());
            }
            let count = libwebp_sys::WebPDemuxGetI(demuxer, libwebp_sys::WebPFormatFeature::WEBP_FF_FRAME_COUNT);
            let loop_count = libwebp_sys::WebPDemuxGetI(demuxer, libwebp_sys::WebPFormatFeature::WEBP_FF_LOOP_COUNT);
            libwebp_sys::WebPDemuxDelete(demuxer);
            (count, Some(loop_count))
        }
    } else {
        (1, None)
    };
    let mut total = 0;
    for _ in 0..count {
        validate_frame_budget(&mut total, info.width, info.height)?;
    }
    if count == 0 {
        return Err("WebP contains no frames".into());
    }
    Ok(SequenceInfo { count: count as usize, loop_count, default_image: false })
}

pub fn decode_frame(data: &[u8], target: usize) -> Result<blanket_codec_common::Sequence, String> {
    decode_frames(data, Some(target))
}

fn decode_frames(data: &[u8], target: Option<usize>) -> Result<blanket_codec_common::Sequence, String> {
    use blanket_codec_common::{Frame, Sequence, validate_frame_budget};
    let info = webpx::ImageInfo::from_webp(data).map_err(|e| e.to_string())?;
    if !info.has_animation {
        return decode_webp(data).map(Sequence::single);
    }
    let mut decoder = webpx::AnimationDecoder::new(data).map_err(|e| e.to_string())?;
    let mut sequence = Sequence { loop_count: Some(decoder.info().loop_count), ..Sequence::default() };
    let mut timestamp = 0;
    let mut total = 0;
    let mut frame_index = 0;
    while decoder.has_more_frames() {
        validate_frame_budget(&mut total, decoder.info().width, decoder.info().height)?;
        let frame = decoder.next_frame().map_err(|e| e.to_string())?.ok_or("missing WebP frame")?;
        let duration = frame.timestamp_ms.checked_sub(timestamp).filter(|d| *d >= 0).ok_or("invalid WebP timestamp")?;
        timestamp = frame.timestamp_ms;
        if target.is_none_or(|target| target == frame_index) {
            sequence.frames.push(Frame {
                image: Image::from_pixels(frame.width, frame.height, PixelMode::Rgba, frame.data, Some("WEBP".into())).map_err(|e| e.to_string())?,
                duration: Some(f64::from(duration)),
                disposal: 0,
                blend: 0,
            });
        }
        if target == Some(frame_index) {
            return Ok(sequence);
        }
        frame_index += 1;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grayscale_encoding_matches_rgb_import_for_lossy_and_lossless() {
        for (width, height) in [(1, 1), (17, 15), (257, 259)] {
            let pixels: Vec<_> = (0..width * height).map(|i| (i * 137 + i / 7) as u8).collect();
            let image = Image::from_pixels(width, height, PixelMode::L, pixels, None).unwrap();
            let pixels = image.pixel_data().unwrap();
            let rgb: Vec<_> = pixels.iter().flat_map(|&v| [v; 3]).collect();
            for lossless in [false, true] {
                for quality in [25, 85] {
                    let reference = webpx::Encoder::new_rgb(&rgb, width, height)
                        .config(webpx::EncoderConfig::new().thread_level(1))
                        .lossless(lossless)
                        .quality(if lossless { 75.0 } else { f32::from(quality) })
                        .encode(webpx::Unstoppable)
                        .unwrap();
                    let encoded = encode_webp(&image, pixels, SaveOptions { quality, lossless, ..Default::default() }).unwrap();
                    assert_eq!(encoded, reference);
                    assert_eq!(decode_webp(&encoded).unwrap().raw_data().unwrap(), webpx::decode_rgb(&encoded).unwrap().0);
                }
            }
        }
    }
}
