use std::borrow::Borrow;

use blanket_codec_common::{DecodeError, SaveOptions, codec_error, validate_dimensions};
use blanket_codec_image::decode_rust_image;
use blanket_core::parallel::deterministic;
use blanket_core::{Image, PixelMode};
use image::ImageFormat as RustFormat;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub fn inspect(data: &[u8]) -> Result<blanket_codec_common::ImageHeader, String> {
    let info = webpx::ImageInfo::from_webp(data).map_err(|e| e.to_string())?;
    blanket_codec_common::ImageHeader::new(info.width, info.height, if info.has_alpha { "RGBA" } else { "RGB" }, 8)
}

pub fn decode_webp(data: &[u8]) -> Result<Image, DecodeError> {
    decode_with_size(data, None)
}

pub fn decode_scaled(data: &[u8], size: (u32, u32)) -> Result<Image, DecodeError> {
    decode_with_size(data, Some(size))
}

fn decode_with_size(data: &[u8], size: Option<(u32, u32)>) -> Result<Image, DecodeError> {
    let features = webpx::ImageInfo::from_webp(data).map_err(DecodeError::with_diagnostic_source)?;
    validate_dimensions(features.width, features.height)?;
    // Preserve the existing first-frame behavior for animated containers.
    if features.has_animation {
        if size.is_some() {
            return Err("scaled decoding is unsupported for animated WebP".into());
        }
        return decode_rust_image(data, RustFormat::WebP, "WEBP");
    }
    let limits = webpx::Limits::default();
    limits.check_input_size(data.len() as u64).map_err(|error| format!("WebP limit exceeded: {error}"))?;
    limits.check_still_image(features.width, features.height).map_err(|error| format!("WebP limit exceeded: {error}"))?;
    let (width, height) = size.unwrap_or((features.width, features.height));
    if width == 0 || height == 0 || width > features.width || height > features.height {
        return Err("invalid scaled WebP dimensions".into());
    }
    let mode = if features.has_alpha { PixelMode::Rgba } else { PixelMode::Rgb };
    let stride = (width as usize).checked_mul(mode.channels()).ok_or("WebP row size overflow")?;
    let length = stride.checked_mul(height as usize).ok_or("WebP buffer size overflow")?;
    let stride = i32::try_from(stride).map_err(DecodeError::with_source)?;
    let mut pixels = Vec::<u8>::new();
    pixels.try_reserve_exact(length).map_err(DecodeError::with_source)?;
    if (width, height) != (features.width, features.height) {
        let mut config = libwebp_sys::WebPDecoderConfig::new().map_err(|()| "cannot initialize WebP decoder")?;
        config.options.use_scaling = 1;
        config.options.scaled_width = width as i32;
        config.options.scaled_height = height as i32;
        config.output.colorspace = if features.has_alpha { libwebp_sys::WEBP_CSP_MODE::MODE_RGBA } else { libwebp_sys::WEBP_CSP_MODE::MODE_RGB };
        config.output.is_external_memory = 1;
        config.output.u.RGBA = libwebp_sys::WebPRGBABuffer { rgba: pixels.as_mut_ptr(), stride, size: length };
        // SAFETY: the initialized decoder writes into the externally owned buffer,
        // whose capacity and stride cover the requested output. FreeDecBuffer never
        // frees external storage and is called on both success and failure.
        let status = unsafe {
            let status = libwebp_sys::WebPDecode(data.as_ptr(), data.len(), &raw mut config);
            libwebp_sys::WebPFreeDecBuffer(&raw mut config.output);
            status
        };
        if status != libwebp_sys::VP8StatusCode::VP8_STATUS_OK {
            if status == libwebp_sys::VP8StatusCode::VP8_STATUS_OUT_OF_MEMORY {
                return Err("WebP decoder memory limit exceeded".into());
            }
            if status == libwebp_sys::VP8StatusCode::VP8_STATUS_UNSUPPORTED_FEATURE {
                return Err("unsupported WebP scaled decoding feature".into());
            }
            return Err(format!("WebP scaled decode failed: {status:?}").into());
        }
        // SAFETY: successful decoding initialized every requested output sample.
        unsafe { pixels.set_len(length) };
        return Image::from_pixels(width, height, mode, pixels, Some("WEBP".into())).map_err(DecodeError::with_source);
    }
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
    Image::from_pixels(width, height, mode, pixels, Some("WEBP".into())).map_err(DecodeError::with_source)
}

pub fn encode_webp(image: &Image, pixels: &[u8], options: SaveOptions) -> PyResult<Vec<u8>> {
    encode_webp_config(image, pixels, options, webpx::EncoderConfig::new().thread_level(u8::from(!deterministic())))
}

pub fn encode_webp_variant(image: &Image, options: SaveOptions, method: u8) -> PyResult<Vec<u8>> {
    let config = webpx::EncoderConfig::new().thread_level(u8::from(!deterministic())).method(method).exact(true);
    encode_webp_config(image, image.pixel_data()?, options, config)
}

fn encode_webp_config(image: &Image, pixels: &[u8], options: SaveOptions, config: webpx::EncoderConfig) -> PyResult<Vec<u8>> {
    let argb = (image.pixel_mode() == PixelMode::L).then(|| pixels.iter().map(|&v| 0xff00_0000 | (u32::from(v) * 0x0001_0101)).collect::<Vec<_>>());
    let encoder = match image.pixel_mode() {
        PixelMode::L => webpx::Encoder::new_argb(argb.as_deref().expect("grayscale pixels expanded above"), image.width(), image.height()),
        PixelMode::Rgb => webpx::Encoder::new_rgb(pixels, image.width(), image.height()),
        PixelMode::Rgba => webpx::Encoder::new_rgba(pixels, image.width(), image.height()),
        _ => return Err(PyValueError::new_err("unsupported mode for WebP")),
    };
    let encoder = encoder.config(config).lossless(options.lossless).quality(if options.lossless { 75.0 } else { f32::from(options.quality) });
    if image.pixel_mode() != PixelMode::L {
        return encoder.encode(webpx::Unstoppable).map_err(codec_error);
    }
    let encoded = encoder.encode_owned(webpx::Unstoppable).map_err(codec_error)?;
    drop(argb);
    Ok(encoded.to_vec())
}

#[must_use]
pub fn backend_version() -> String {
    let (major, minor, patch) = webpx::version();
    format!("{major}.{minor}.{patch}")
}

pub fn decode_sequence(data: &[u8]) -> Result<blanket_codec_common::Sequence, DecodeError> {
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

pub fn decode_frame(data: &[u8], target: usize) -> Result<blanket_codec_common::Sequence, DecodeError> {
    decode_frames(data, Some(target))
}

fn decode_frames(data: &[u8], target: Option<usize>) -> Result<blanket_codec_common::Sequence, DecodeError> {
    use blanket_codec_common::{Frame, Sequence, validate_frame_budget};
    let info = webpx::ImageInfo::from_webp(data).map_err(DecodeError::with_diagnostic_source)?;
    if !info.has_animation {
        return decode_webp(data).map(Sequence::single);
    }
    let mut decoder = webpx::AnimationDecoder::new(data).map_err(DecodeError::with_diagnostic_source)?;
    let mut sequence = Sequence { loop_count: Some(decoder.info().loop_count), ..Sequence::default() };
    let mut timestamp = 0;
    let mut total = 0;
    let mut frame_index = 0;
    while decoder.has_more_frames() {
        validate_frame_budget(&mut total, decoder.info().width, decoder.info().height)?;
        let frame = decoder.next_frame().map_err(DecodeError::with_diagnostic_source)?.ok_or("missing WebP frame")?;
        let duration = frame.timestamp_ms.checked_sub(timestamp).filter(|d| *d >= 0).ok_or("invalid WebP timestamp")?;
        timestamp = frame.timestamp_ms;
        if target.is_none_or(|target| target == frame_index) {
            sequence.frames.push(Frame {
                image: Image::from_pixels(frame.width, frame.height, PixelMode::Rgba, frame.data, Some("WEBP".into()))
                    .map_err(DecodeError::with_source)?,
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

struct AnimEncoder(*mut libwebp_sys::WebPAnimEncoder);

impl AnimEncoder {
    fn error(&self, fallback: &str) -> PyErr {
        // SAFETY: the encoder is live; libwebp returns null or a NUL-terminated message it owns.
        let message = unsafe {
            let message = libwebp_sys::WebPAnimEncoderGetError(self.0);
            (!message.is_null()).then(|| std::ffi::CStr::from_ptr(message).to_string_lossy().into_owned())
        };
        PyValueError::new_err(message.filter(|message| !message.is_empty()).unwrap_or_else(|| fallback.into()))
    }
}

impl Drop for AnimEncoder {
    fn drop(&mut self) {
        // SAFETY: the pointer came from WebPAnimEncoderNewInternal and is deleted once.
        unsafe { libwebp_sys::WebPAnimEncoderDelete(self.0) };
    }
}

struct Picture(libwebp_sys::WebPPicture);

impl Drop for Picture {
    fn drop(&mut self) {
        // SAFETY: the picture was initialized by WebPPictureInitInternal.
        unsafe { libwebp_sys::WebPPictureFree(&raw mut self.0) };
    }
}

/// Encode RGB or RGBA frames with Pillow's animated defaults (method 0).
pub fn encode_sequence(sequence: &blanket_codec_common::Sequence<impl Borrow<Image>>, options: SaveOptions) -> PyResult<Vec<u8>> {
    let first = sequence.frames.first().ok_or_else(|| PyValueError::new_err("empty sequence"))?.image.borrow();
    let (width, height) = (first.width(), first.height());
    if width == 0 || height == 0 || width > 16383 || height > 16383 {
        return Err(PyValueError::new_err("invalid WebP dimensions"));
    }
    let loop_count = sequence.loop_count.unwrap_or(0);
    if loop_count > u32::from(u16::MAX) {
        return Err(PyValueError::new_err("WebP loop must be between 0 and 65535"));
    }
    let mut anim_options = std::mem::MaybeUninit::<libwebp_sys::WebPAnimEncoderOptions>::zeroed();
    let mut config = std::mem::MaybeUninit::<libwebp_sys::WebPConfig>::zeroed();
    // SAFETY: both structs start zeroed and are fully initialized by libwebp on success.
    let (mut anim_options, mut config) = unsafe {
        if libwebp_sys::WebPAnimEncoderOptionsInitInternal(anim_options.as_mut_ptr(), libwebp_sys::WEBP_MUX_ABI_VERSION as i32) == 0
            || libwebp_sys::WebPConfigInitInternal(
                config.as_mut_ptr(),
                libwebp_sys::WebPPreset::WEBP_PRESET_DEFAULT,
                f32::from(options.quality),
                libwebp_sys::WEBP_ENCODER_ABI_VERSION as i32,
            ) == 0
        {
            return Err(PyValueError::new_err("failed to initialize WebP encoder"));
        }
        (anim_options.assume_init(), config.assume_init())
    };
    anim_options.anim_params.loop_count = loop_count as i32;
    config.lossless = i32::from(options.lossless);
    config.method = 0;
    config.thread_level = i32::from(!deterministic());
    // SAFETY: options are initialized; a null result is handled below.
    let encoder = AnimEncoder(unsafe {
        libwebp_sys::WebPAnimEncoderNewInternal(width as i32, height as i32, &raw const anim_options, libwebp_sys::WEBP_MUX_ABI_VERSION as i32)
    });
    if encoder.0.is_null() {
        return Err(PyValueError::new_err("failed to allocate WebP encoder"));
    }
    let mut timestamp: i32 = 0;
    for frame in &sequence.frames {
        let image = frame.image.borrow();
        let channels = match image.pixel_mode() {
            PixelMode::Rgb => 3,
            PixelMode::Rgba => 4,
            _ => return Err(PyValueError::new_err("unsupported mode for WebP")),
        };
        let pixels = image.pixel_data()?;
        if image.width() != width || image.height() != height || pixels.len() < width as usize * height as usize * channels {
            return Err(PyValueError::new_err("WebP frames must share the canvas size"));
        }
        let mut picture = Picture(unsafe { std::mem::zeroed() });
        // SAFETY: the zeroed picture is initialized, then imports a buffer covering every row.
        let ok = unsafe {
            libwebp_sys::WebPPictureInitInternal(&raw mut picture.0, libwebp_sys::WEBP_ENCODER_ABI_VERSION as i32) != 0 && {
                picture.0.width = width as i32;
                picture.0.height = height as i32;
                picture.0.use_argb = 1;
                let stride = (width as usize * channels) as i32;
                if channels == 3 {
                    libwebp_sys::WebPPictureImportRGB(&raw mut picture.0, pixels.as_ptr(), stride) != 0
                } else {
                    libwebp_sys::WebPPictureImportRGBA(&raw mut picture.0, pixels.as_ptr(), stride) != 0
                }
            }
        };
        if !ok {
            return Err(PyValueError::new_err("failed to import WebP frame"));
        }
        // SAFETY: encoder, picture, and config are live and initialized.
        if unsafe { libwebp_sys::WebPAnimEncoderAdd(encoder.0, &raw mut picture.0, timestamp, &raw const config) } == 0 {
            return Err(encoder.error("failed to add WebP frame"));
        }
        timestamp = timestamp.checked_add(frame.duration.unwrap_or(0.0) as i32).ok_or_else(|| PyValueError::new_err("WebP duration overflow"))?;
    }
    let mut data = libwebp_sys::WebPData::default();
    // SAFETY: a null frame finalizes the encoder; assembled data is copied before being freed.
    unsafe {
        if libwebp_sys::WebPAnimEncoderAdd(encoder.0, std::ptr::null_mut(), timestamp, std::ptr::null()) == 0 {
            return Err(encoder.error("failed to finalize WebP animation"));
        }
        if libwebp_sys::WebPAnimEncoderAssemble(encoder.0, &raw mut data) == 0 || data.bytes.is_null() {
            return Err(encoder.error("failed to assemble WebP animation"));
        }
        let output = std::slice::from_raw_parts(data.bytes, data.size).to_vec();
        libwebp_sys::WebPDataClear(&mut data);
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scaled_decoding_preserves_full_size_samples_and_validates_output_bounds() {
        for mode in [PixelMode::Rgb, PixelMode::Rgba] {
            let pixels: Vec<_> = (0..17 * 15 * mode.channels()).map(|i| (i * 137 + i / 7) as u8).collect();
            let image = Image::from_pixels(17, 15, mode, pixels, None).unwrap();
            for lossless in [false, true] {
                let encoded = encode_webp(&image, image.pixel_data().unwrap(), SaveOptions { lossless, ..Default::default() }).unwrap();
                let normal = decode_webp(&encoded).unwrap();
                let scaled = decode_scaled(&encoded, (17, 15)).unwrap();
                assert_eq!(normal.raw_data().unwrap(), scaled.raw_data().unwrap());
                for size in [(1, 1), (3, 5), (9, 8)] {
                    let reduced = decode_scaled(&encoded, size).unwrap();
                    assert_eq!(reduced.size(), size);
                    assert_eq!(reduced.pixel_mode(), normal.pixel_mode());
                    assert_eq!(reduced.raw_data().unwrap().len(), size.0 as usize * size.1 as usize * normal.pixel_mode().channels());
                }
                for size in [(0, 1), (1, 0), (18, 15), (17, 16)] {
                    assert!(decode_scaled(&encoded, size).is_err());
                }
                assert!(decode_scaled(&encoded[..encoded.len() / 2], (3, 5)).is_err());
            }
        }
    }

    #[test]
    fn empty_animation_returns_an_error() {
        let sequence = blanket_codec_common::Sequence::<Image>::default();
        assert!(encode_sequence(&sequence, SaveOptions::default()).is_err());
    }

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
