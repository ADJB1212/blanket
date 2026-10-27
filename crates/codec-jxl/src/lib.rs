#![feature(default_field_values)]

use std::borrow::Cow;

mod encode;
pub mod stream;

use blanket_codec_common::{SaveOptions, codec_error, validate_dimensions};
use blanket_core::{Image, PixelMode};
use jpegxl_rs::decode::{Event, Events, PixelFormat};
use jpegxl_rs::decoder_builder;
use jpegxl_rs::encode::EncoderSpeed;
#[cfg(test)]
use jpegxl_rs::encode::{ColorEncoding, EncoderFrame};
#[cfg(test)]
use jpegxl_rs::encoder_builder;
use jpegxl_rs::parallel::ParallelRunner;
use jpegxl_rs::parallel::resizable_runner::ResizableRunner;
use jpegxl_rs::parallel::threads_runner::ThreadsRunner;
use pyo3::exceptions::{PyOSError, PyValueError};
use pyo3::prelude::*;
#[cfg(target_endian = "little")]
use zerocopy::FromBytes;

pub const JXL_PARALLEL_MIN_BYTES: usize = 32 * 1024;

pub fn backend_version() -> String {
    let version = unsafe { jpegxl_sys::decode::JxlDecoderVersion() };
    format!("{}.{}.{}", version / 1_000_000, version / 1_000 % 1_000, version % 1_000)
}

thread_local! {
    // A runner is used by only its owning calling thread. Reuse its workers
    // across decodes; each image still gets a fresh decoder and metadata state.
    static JXL_DECODE_RUNNER: Option<ResizableRunner<'static>> = ResizableRunner::new(None);
    // Reuse workers, but let libjxl size the pool for each frame. Tiny images
    // should not pay the synchronization cost of a full-machine thread pool.
    static JXL_ENCODE_RUNNER: std::cell::OnceCell<Option<ResizableRunner<'static>>> = const { std::cell::OnceCell::new() };
    // Fixed-size pools for search candidates that share the machine with
    // concurrent candidates. Each worker thread keeps its most recent pool.
    static JXL_CANDIDATE_RUNNER: std::cell::RefCell<Option<(usize, ThreadsRunner<'static>)>> = const { std::cell::RefCell::new(None) };
}

/// Thread use for one JPEG XL encoding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JxlThreads {
    /// The calling thread's resizable pool, sized by libjxl for the frame.
    Pool,
    /// Exactly this many worker threads; one means no runner at all.
    Fixed(usize),
}

pub fn decode_jxl(data: &[u8]) -> Result<Image, String> {
    use jpegxl_sys::common::types::{JxlDataType, JxlEndianness};

    let (info, pixels) = JXL_DECODE_RUNNER.with(|runner| -> Result<_, String> {
        let runner = runner.as_ref().ok_or_else(|| "cannot allocate JPEG XL thread pool".to_owned())?;
        let mut decoder = decoder_builder().parallel_runner(runner).build().map_err(|error| error.to_string())?;
        let mut session = decoder.session(Events::FULL_IMAGE).map_err(|error| error.to_string())?;
        let mut input = data;
        let mut info = None;
        let mut pixels = None;
        let mut data_type = None;
        loop {
            match session.process(&mut input).map_err(|error| error.to_string())? {
                Event::BasicInfo(header) => {
                    validate_dimensions(header.xsize, header.ysize)?;
                    // Ask libjxl for the display conversion during the first decode.
                    if header.exponent_bits_per_sample != 0 {
                        data_type = Some(JxlDataType::Uint8);
                    }
                    info = Some(header);
                }
                Event::NeedImageOutBuffer => {
                    let format = PixelFormat { endianness: JxlEndianness::Little, ..PixelFormat::default() };
                    session.alloc_image_buffer(format, data_type).map_err(|error| error.to_string())?;
                }
                Event::FullImage(image) => pixels = image,
                Event::Success => break,
                Event::NeedMoreInput => {
                    return Err(if info.is_none() { "incomplete JPEG XL header" } else { "incomplete JPEG XL image" }.into());
                }
                _ => (),
            }
        }
        Ok((info.ok_or("incomplete JPEG XL header")?, pixels.ok_or("no JPEG XL image decoded")?))
    })?;

    let wide = pixels.format.data_type == JxlDataType::Uint16;
    let (mode, pixels) = match (info.num_color_channels, info.alpha_bits > 0) {
        (1, true) if wide => {
            let mut rgba = Vec::with_capacity(pixels.data.len() * 2);
            for &[l0, l1, a0, a1] in pixels.data.as_chunks::<4>().0 {
                rgba.extend_from_slice(&[l0, l1, l0, l1, l0, l1, a0, a1]);
            }
            (PixelMode::Rgba, rgba)
        }
        (1, false) => (PixelMode::L, pixels.data),
        (3, false) => (PixelMode::Rgb, pixels.data),
        (1, true) => (PixelMode::Rgba, luma_alpha_to_rgba(&pixels.data)),
        (3, true) => (PixelMode::Rgba, pixels.data),
        (channels, has_alpha) => {
            return Err(format!("unsupported JPEG XL channel layout: {channels} color channels, alpha={has_alpha}"));
        }
    };
    if wide {
        Image::from_sample_bytes(info.xsize, info.ysize, mode, pixels, 16, Some("JXL".into()))
    } else {
        Image::from_pixels(info.xsize, info.ysize, mode, pixels, Some("JXL".into()))
    }
    .map_err(|error| error.to_string())
}

fn luma_alpha_to_rgba(luma_alpha: &[u8]) -> Vec<u8> {
    let mut rgba = Vec::with_capacity(luma_alpha.len().saturating_mul(2));
    for &[luma, alpha] in luma_alpha.as_chunks::<2>().0 {
        rgba.extend_from_slice(&[luma, luma, luma, alpha]);
    }
    rgba
}

pub fn encode_jxl(image: &Image, pixels: &[u8], options: SaveOptions) -> PyResult<Vec<u8>> {
    encode_jxl_samples(image, JxlSamples::Byte(pixels), options, JxlThreads::Pool)
}

#[derive(Clone, Copy)]
enum JxlSamples<'a> {
    Byte(&'a [u8]),
    Wide(&'a [u16]),
}

fn encode_jxl_samples(image: &Image, pixels: JxlSamples<'_>, options: SaveOptions, threads: JxlThreads) -> PyResult<Vec<u8>> {
    if !matches!(image.mode, PixelMode::L | PixelMode::Rgb | PixelMode::Rgba) {
        return Err(PyValueError::new_err("unsupported mode for JPEG XL"));
    }
    jxl_encoder_speed(options.effort)?;
    with_jxl_runner(threads, |runner| encode::encode(image, pixels, options, runner))
}

fn with_jxl_runner<T>(threads: JxlThreads, encode: impl FnOnce(Option<&dyn ParallelRunner>) -> PyResult<T>) -> PyResult<T> {
    match threads {
        // Do not even initialize a thread-local pool for single-threaded
        // Rayon candidates. Besides allocating unused state, pool creation
        // could fail despite the encoder not needing a runner at all.
        JxlThreads::Fixed(0 | 1) => encode(None),
        JxlThreads::Fixed(workers) => JXL_CANDIDATE_RUNNER.with(|cached| {
            let mut cached = cached.borrow_mut();
            if cached.as_ref().is_none_or(|(size, _)| *size != workers) {
                let runner = ThreadsRunner::new(None, Some(workers)).ok_or_else(|| PyOSError::new_err("cannot allocate JPEG XL thread pool"))?;
                *cached = Some((workers, runner));
            }
            encode(Some(&cached.as_ref().expect("pool cached above").1))
        }),
        JxlThreads::Pool => JXL_ENCODE_RUNNER.with(|runner| {
            let runner = runner.get_or_init(|| ResizableRunner::new(None));
            let runner = runner.as_ref().ok_or_else(|| PyOSError::new_err("cannot allocate JPEG XL thread pool"))?;
            encode(Some(runner))
        }),
    }
}

fn encode_native(
    encoder: &encode::Encoder, image: &Image, pixels: JxlSamples<'_>, options: SaveOptions, runner: Option<&dyn ParallelRunner>,
) -> PyResult<Vec<u8>> {
    use jpegxl_sys::common::types::{JxlDataType, JxlEndianness, JxlPixelFormat};
    use jpegxl_sys::encoder::encode::*;

    fn check(status: JxlEncoderStatus) -> PyResult<()> {
        if status == JxlEncoderStatus::Success { Ok(()) } else { Err(codec_error("JPEG XL profile encoding failed")) }
    }
    let mut info = std::mem::MaybeUninit::uninit();
    // SAFETY: libjxl initializes the entire basic-info struct.
    unsafe { JxlEncoderInitBasicInfo(info.as_mut_ptr()) };
    // SAFETY: initialized above.
    let mut info = unsafe { info.assume_init() };
    info.xsize = image.width;
    info.ysize = image.height;
    info.bits_per_sample = match pixels {
        JxlSamples::Byte(_) => 8,
        JxlSamples::Wide(_) => 16,
    };
    info.num_color_channels = if image.mode == PixelMode::L { 1 } else { 3 };
    info.uses_original_profile = (options.lossless || options.quality == 100 || !options.icc_profile.is_empty()).into();
    if image.mode == PixelMode::Rgba {
        info.num_extra_channels = 1;
        info.alpha_bits = info.bits_per_sample;
    }
    let (raw, length, data_type) = match pixels {
        JxlSamples::Byte(raw) => (raw.as_ptr().cast(), raw.len(), JxlDataType::Uint8),
        JxlSamples::Wide(raw) => (raw.as_ptr().cast(), std::mem::size_of_val(raw), JxlDataType::Uint16),
    };
    let format = JxlPixelFormat { num_channels: image.mode.channels() as u32, data_type, endianness: JxlEndianness::Native, align: 0 };
    if let Some(runner) = runner {
        runner.callback_basic_info(&info);
    }
    // SAFETY: the encoder and all input buffers stay alive through processing.
    unsafe {
        if let Some(runner) = runner {
            check(JxlEncoderSetParallelRunner(encoder.0, runner.runner(), runner.as_opaque_ptr()))?;
        }
        check(JxlEncoderSetBasicInfo(encoder.0, &raw const info))?;
        if options.icc_profile.is_empty() {
            let mut color = std::mem::MaybeUninit::uninit();
            JxlColorEncodingSetToSRGB(color.as_mut_ptr(), (image.mode == PixelMode::L).into());
            check(JxlEncoderSetColorEncoding(encoder.0, color.as_ptr()))?;
        } else {
            check(JxlEncoderSetICCProfile(encoder.0, options.icc_profile.as_ptr(), options.icc_profile.len()))?;
        }
        let settings = JxlEncoderFrameSettingsCreate(encoder.0, std::ptr::null());
        if settings.is_null() {
            return Err(codec_error("cannot allocate JPEG XL settings"));
        }
        check(JxlEncoderFrameSettingsSetOption(settings, JxlEncoderFrameSettingId::Effort, options.effort.into()))?;
        check(JxlEncoderSetFrameLossless(settings, options.lossless.into()))?;
        check(JxlEncoderSetFrameDistance(settings, JxlEncoderDistanceFromQuality(options.quality.into())))?;
        check(JxlEncoderFrameSettingsSetOption(settings, JxlEncoderFrameSettingId::DecodingSpeed, 0))?;
        check(JxlEncoderAddImageFrame(settings, &raw const format, raw, length))?;
        JxlEncoderCloseInput(encoder.0);
    }
    let mut output = Vec::<u8>::with_capacity(length.saturating_add(1024).clamp(4096, 512 * 1024));
    loop {
        let written = output.len();
        let spare = output.spare_capacity_mut();
        let capacity = spare.len();
        let mut next = spare.as_mut_ptr().cast::<u8>();
        let mut available = capacity;
        // SAFETY: next/available describe allocated spare capacity. Only bytes
        // reported written by successful processing are exposed through Vec's len.
        let status = unsafe { JxlEncoderProcessOutput(encoder.0, &raw mut next, &raw mut available) };
        if matches!(status, JxlEncoderStatus::Success | JxlEncoderStatus::NeedMoreOutput) {
            // SAFETY: libjxl initialized capacity - available additional bytes.
            unsafe { output.set_len(written + capacity - available) };
        }
        match status {
            JxlEncoderStatus::Success => {
                if output.capacity() > output.len().saturating_mul(4).max(4096) {
                    output.shrink_to_fit();
                }
                return Ok(output);
            }
            JxlEncoderStatus::NeedMoreOutput => output.reserve(output.capacity()),
            _ => return Err(codec_error("JPEG XL profile encoding failed")),
        }
    }
}

pub fn read_metadata(data: &[u8]) -> PyResult<Vec<(String, Vec<u8>)>> {
    use jpegxl_sys::common::types::JxlBoxType;
    use jpegxl_sys::decode::*;

    const LIMIT: usize = 16 * 1024 * 1024;
    struct Decoder(*mut JxlDecoder);
    impl Drop for Decoder {
        fn drop(&mut self) {
            // SAFETY: the guard owns a non-null decoder.
            unsafe { JxlDecoderDestroy(self.0) };
        }
    }
    fn check(status: JxlDecoderStatus) -> PyResult<()> {
        if status == JxlDecoderStatus::Success { Ok(()) } else { Err(codec_error("invalid JPEG XL metadata")) }
    }
    let mut result = Vec::new();
    let mut buffer = Vec::<u8>::new();
    let mut key: Option<&str> = None;
    // SAFETY: null selects the default allocator.
    let decoder = Decoder(unsafe { JxlDecoderCreate(std::ptr::null()) });
    if decoder.0.is_null() {
        return Err(codec_error("cannot allocate JPEG XL decoder"));
    }
    // SAFETY: the decoder and its borrowed input/output buffers remain alive.
    unsafe {
        check(JxlDecoderSubscribeEvents(
            decoder.0,
            JxlDecoderStatus::ColorEncoding as i32 | JxlDecoderStatus::Box as i32 | JxlDecoderStatus::BoxComplete as i32,
        ))?;
        check(JxlDecoderSetDecompressBoxes(decoder.0, true.into()))?;
        check(JxlDecoderSetInput(decoder.0, data.as_ptr(), data.len()))?;
        JxlDecoderCloseInput(decoder.0);
        loop {
            let status = JxlDecoderProcessInput(decoder.0);
            if matches!(status, JxlDecoderStatus::Box | JxlDecoderStatus::BoxComplete | JxlDecoderStatus::Success)
                && let Some(key) = key.take()
            {
                let remaining = JxlDecoderReleaseBoxBuffer(decoder.0);
                buffer.truncate(buffer.len() - remaining);
                let raw = if key == "exif" {
                    let offset = buffer.get(..4).map(|v| u32::from_be_bytes(v.try_into().expect("four bytes")) as usize);
                    offset.and_then(|v| v.checked_add(4)).and_then(|offset| buffer.get(offset..)).map(|v| [b"Exif\0\0".as_slice(), v].concat())
                } else {
                    Some(std::mem::take(&mut buffer))
                };
                if let Some(raw) = raw {
                    result.push((key.into(), raw));
                }
            }
            match status {
                JxlDecoderStatus::ColorEncoding => {
                    let mut size = 0;
                    check(JxlDecoderGetICCProfileSize(decoder.0, JxlColorProfileTarget::Original, &raw mut size))?;
                    if size > LIMIT {
                        return Err(codec_error("JPEG XL ICC profile exceeds 16 MiB"));
                    }
                    let mut profile = vec![0; size];
                    check(JxlDecoderGetColorAsICCProfile(decoder.0, JxlColorProfileTarget::Original, profile.as_mut_ptr(), size))?;
                    result.push(("icc_profile".into(), profile));
                }
                JxlDecoderStatus::Box => {
                    let mut kind = JxlBoxType([0; 4]);
                    check(JxlDecoderGetBoxType(decoder.0, &mut kind, true.into()))?;
                    key = match kind.0.map(|v| v.to_ne_bytes()[0]) {
                        [b'E', b'x', b'i', b'f'] => Some("exif"),
                        [b'x', b'm', b'l', b' '] => Some("xmp"),
                        _ => None,
                    };
                    if key.is_some() {
                        buffer = vec![0; 1024];
                        check(JxlDecoderSetBoxBuffer(decoder.0, buffer.as_mut_ptr(), buffer.len()))?;
                    }
                }
                JxlDecoderStatus::BoxNeedMoreOutput => {
                    let remaining = JxlDecoderReleaseBoxBuffer(decoder.0);
                    let written = buffer.len() - remaining;
                    if buffer.len() >= LIMIT {
                        return Err(codec_error("JPEG XL metadata box exceeds 16 MiB"));
                    }
                    buffer.resize(buffer.len() * 2, 0);
                    check(JxlDecoderSetBoxBuffer(decoder.0, buffer[written..].as_mut_ptr(), buffer.len() - written))?;
                }
                JxlDecoderStatus::Success => return Ok(result),
                JxlDecoderStatus::Error | JxlDecoderStatus::NeedMoreInput => return Err(codec_error("invalid JPEG XL metadata")),
                _ => {}
            }
        }
    }
}

/// Worker threads worth giving one search candidate when `workers` candidates
/// encode concurrently on the Rayon pool. libjxl parallelizes over 256-pixel
/// groups, so more threads than groups would only add synchronization.
pub fn jxl_candidate_threads(image: &Image, workers: usize) -> usize {
    let groups = (image.width as usize).div_ceil(256) * (image.height as usize).div_ceil(256);
    (rayon::current_num_threads() / workers.max(1)).clamp(1, groups.max(1))
}

/// Normalize wide samples once per effort search, rather than once per encode.
/// Eight-bit and aligned native-endian 16-bit input borrow the source image.
pub struct PreparedJxl<'a> {
    image: &'a Image,
    wide: Option<Cow<'a, [u16]>>,
}

fn jxl_wide_samples(raw: &[u8], depth: u8) -> Cow<'_, [u16]> {
    #[cfg(target_endian = "little")]
    if depth == 16
        && let Ok(samples) = <[u16]>::ref_from_bytes(raw)
    {
        return Cow::Borrowed(samples);
    }
    Cow::Owned(blanket_core::simd::normalize_u16(raw, depth))
}

impl<'a> PreparedJxl<'a> {
    pub fn new(image: &'a Image) -> PyResult<Self> {
        let raw = image.raw_data()?;
        let wide = (image.bit_depth > 8).then(|| jxl_wide_samples(raw, image.bit_depth));
        Ok(Self { image, wide })
    }

    pub fn encode(&self, options: SaveOptions) -> PyResult<Vec<u8>> {
        self.encode_with_threads(options, JxlThreads::Pool)
    }

    pub fn encode_candidate(&self, options: SaveOptions) -> PyResult<Vec<u8>> {
        // Match the outer search's small-image cutoff. Larger serial trials
        // still benefit from libjxl's pool; parallel trials size their own.
        let threads = if self.image.raw_data()?.len() < JXL_PARALLEL_MIN_BYTES { JxlThreads::Fixed(1) } else { JxlThreads::Pool };
        self.encode_with_threads(options, threads)
    }

    pub fn encode_with_threads(&self, options: SaveOptions, threads: JxlThreads) -> PyResult<Vec<u8>> {
        match &self.wide {
            Some(samples) => encode_jxl_samples(self.image, JxlSamples::Wide(samples), options, threads),
            None => encode_jxl_samples(self.image, JxlSamples::Byte(self.image.raw_data()?), options, threads),
        }
    }
}

fn jxl_encoder_speed(effort: u8) -> PyResult<EncoderSpeed> {
    match effort {
        1 => Ok(EncoderSpeed::Lightning),
        2 => Ok(EncoderSpeed::Thunder),
        3 => Ok(EncoderSpeed::Falcon),
        4 => Ok(EncoderSpeed::Cheetah),
        5 => Ok(EncoderSpeed::Hare),
        6 => Ok(EncoderSpeed::Wombat),
        7 => Ok(EncoderSpeed::Squirrel),
        8 => Ok(EncoderSpeed::Kitten),
        9 => Ok(EncoderSpeed::Tortoise),
        10 => Ok(EncoderSpeed::Glacier),
        _ => Err(PyValueError::new_err("effort must be between 1 and 10")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cached_encoding_matches_fresh_encoder_after_settings_and_profile_errors() {
        for mode in [PixelMode::L, PixelMode::Rgba, PixelMode::Rgb] {
            let pixels: Vec<_> = (0..67 * 19 * mode.channels()).map(|i| (i * 137 + i / 7) as u8).collect();
            let image = Image::from_pixels(67, 19, mode, pixels, None).unwrap();
            let pixels = image.pixel_data().unwrap();
            for lossless in [true, false] {
                for quality in [85, 100] {
                    let options = SaveOptions { quality, effort: 3, lossless, ..SaveOptions::default() };
                    let prepared = PreparedJxl::new(&image).unwrap();
                    let actual = prepared.encode_with_threads(options, JxlThreads::Fixed(1)).unwrap();
                    let mut reference = encoder_builder()
                        .has_alpha(mode == PixelMode::Rgba)
                        .lossless(lossless)
                        .speed(EncoderSpeed::Falcon)
                        .decoding_speed(0)
                        .use_container(false)
                        .jpeg_quality(f32::from(quality))
                        .uses_original_profile(lossless || quality == 100)
                        .color_encoding(if mode == PixelMode::L { ColorEncoding::SrgbLuma } else { ColorEncoding::Srgb })
                        .build()
                        .unwrap();
                    let expected = reference.encode_frame(&EncoderFrame::new(pixels).num_channels(mode.channels() as u32), 67, 19).unwrap();
                    assert_eq!(actual, expected);
                    assert!(prepared.encode_with_threads(SaveOptions { icc_profile: b"invalid profile", ..options }, JxlThreads::Fixed(1)).is_err());
                    assert_eq!(prepared.encode_with_threads(options, JxlThreads::Fixed(1)).unwrap(), expected);
                }
            }
        }
    }

    #[test]
    fn eager_decode_matches_typed_output_for_all_channel_layouts() {
        for channels in 1..=4 {
            for depth in [8, 16, 32] {
                let gray = channels <= 2;
                let alpha = channels == 2 || channels == 4;
                let mut encoder = encoder_builder()
                    .has_alpha(alpha)
                    .color_encoding(if gray { ColorEncoding::SrgbLuma } else { ColorEncoding::Srgb })
                    .build()
                    .unwrap();
                let samples: Vec<_> = (0..17 * 9 * channels).map(|i| (i * 137 + i / 7) as u16).collect();
                let encoded = match depth {
                    8 => {
                        let samples: Vec<_> = samples.iter().map(|&v| v as u8).collect();
                        encoder.encode_frame(&EncoderFrame::new(&samples).num_channels(channels as u32), 17, 9).unwrap()
                    }
                    16 => encoder.encode_frame(&EncoderFrame::new(&samples).num_channels(channels as u32), 17, 9).unwrap(),
                    _ => {
                        let samples: Vec<_> = samples.iter().map(|&v| f32::from(v) / 16384.0).collect();
                        encoder.encode_frame(&EncoderFrame::new(&samples).num_channels(channels as u32), 17, 9).unwrap()
                    }
                };
                let reference = decoder_builder().build().unwrap();
                let expected = if depth == 16 {
                    let (_, samples) = reference.decode_with::<u16>(&encoded).unwrap();
                    let samples = if channels == 2 { samples.as_chunks::<2>().0.iter().flat_map(|&[l, a]| [l, l, l, a]).collect() } else { samples };
                    samples.into_iter().flat_map(u16::to_le_bytes).collect::<Vec<_>>()
                } else {
                    let (_, samples) = reference.decode_with::<u8>(&encoded).unwrap();
                    if channels == 2 { luma_alpha_to_rgba(&samples) } else { samples }
                };
                let actual = decode_jxl(&encoded).unwrap();
                assert_eq!(actual.size(), (17, 9));
                assert_eq!(
                    actual.mode,
                    if alpha {
                        PixelMode::Rgba
                    } else if gray {
                        PixelMode::L
                    } else {
                        PixelMode::Rgb
                    }
                );
                assert_eq!(actual.bit_depth, if depth == 16 { 16 } else { 8 });
                assert_eq!(actual.raw_data().unwrap(), expected, "{channels}/{depth}");
                assert!(decode_jxl(&encoded[..encoded.len() / 2]).is_err());
                assert_eq!(decode_jxl(&encoded).unwrap().raw_data().unwrap(), expected);
            }
        }
    }

    #[test]
    fn oversized_header_is_rejected_without_frame_data() {
        let header = [0xff, 0x0a, 0xfc, 0x76, 0x01, 0xfe, 0xf9, 0x00, 0x22];
        assert!(decode_jxl(&header[..8]).err().unwrap().contains("incomplete JPEG XL header"));
        let error = decode_jxl(&header).err().unwrap();
        assert!(error.contains("192000000 pixels"));
        assert!(error.contains("exceeds Blanket limit"));
    }

    #[test]
    fn compressed_metadata_boxes_and_profile_are_read_without_pixel_output() {
        let mut encoder = encoder_builder().use_container(true).build().unwrap();
        let exif = b"\0\0\0\0II*\0\x08\0\0\0\0\0\0\0\0\0";
        let xmp = vec![b'x'; 9000];
        encoder.add_metadata(&jpegxl_rs::encode::Metadata::Exif(exif), true);
        encoder.add_metadata(&jpegxl_rs::encode::Metadata::Xmp(&xmp), true);
        let encoded = encoder.encode(&[30u8; 12], 2, 2).unwrap();
        let metadata = read_metadata(&encoded).unwrap();
        assert!(metadata.iter().any(|(key, raw)| key == "exif" && raw == &[b"Exif\0\0".as_slice(), &exif[4..]].concat()));
        assert!(metadata.iter().any(|(key, raw)| key == "xmp" && raw == &xmp));
        assert!(metadata.iter().any(|(key, raw)| key == "icc_profile" && !raw.is_empty()));
    }

    #[test]
    fn profiled_jxl_preserves_samples_and_original_icc() {
        for mode in [PixelMode::L, PixelMode::Rgb, PixelMode::Rgba] {
            for depth in [8, 16] {
                let samples = (0..7 * 5 * mode.channels()).map(|i| (i * 19) as u16).collect::<Vec<_>>();
                let image = if depth == 8 {
                    Image::from_pixels(7, 5, mode, samples.iter().map(|&v| v as u8).collect(), None).unwrap()
                } else {
                    Image::from_samples(7, 5, mode, samples, depth, None).unwrap()
                };
                let options = SaveOptions { lossless: true, ..Default::default() };
                let prepared = PreparedJxl::new(&image).unwrap();
                let baseline = prepared.encode(options).unwrap();
                let metadata = read_metadata(&baseline).unwrap();
                let profile = &metadata.iter().find(|(key, _)| key == "icc_profile").unwrap().1;
                let encoded = prepared.encode(SaveOptions { icc_profile: profile, ..options }).unwrap();
                let decoded = decode_jxl(&encoded).unwrap();
                assert_eq!(decoded.raw_data().unwrap(), image.raw_data().unwrap(), "{mode:?}/{depth}");
                assert!(read_metadata(&encoded).unwrap().iter().any(|(key, raw)| key == "icc_profile" && raw == profile));
            }
        }
    }

    #[test]
    fn jxl_wide_preparation_borrows_only_native_aligned_samples() {
        #[repr(align(2))]
        struct Aligned([u8; 10]);
        let storage = Aligned([0, 0, 255, 255, 1, 128, 17, 23, 42, 0]);
        for raw in [&storage.0[..8], &storage.0[1..9]] {
            let prepared = jxl_wide_samples(raw, 16);
            assert_eq!(prepared.as_ref(), blanket_core::simd::normalize_u16(raw, 16));
            assert_eq!(matches!(prepared, Cow::Borrowed(_)), cfg!(target_endian = "little") && raw.as_ptr().addr().is_multiple_of(2));
        }
        for depth in [10, 12] {
            let raw: Vec<_> = [0_u16, 1, 71, (1 << depth) - 1].into_iter().flat_map(u16::to_le_bytes).collect();
            let prepared = jxl_wide_samples(&raw, depth);
            assert!(matches!(prepared, Cow::Owned(_)));
            assert_eq!(prepared.as_ref(), blanket_core::simd::normalize_u16(&raw, depth));
            assert_eq!(prepared[3], u16::MAX);
        }
    }

    #[test]
    fn jxl_output_buffers_fit_small_images_and_grow_for_large_images() {
        for side in [17, 513] {
            let mut state = 19_u32;
            let pixels: Vec<_> = (0..side * side * 4)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 17;
                    state ^= state << 5;
                    state as u8
                })
                .collect();
            let image = Image::from_pixels(side, side, PixelMode::Rgba, pixels.clone(), None).unwrap();
            let options = SaveOptions { quality: 100, lossless: true, effort: 1, ..SaveOptions::default() };
            let output = PreparedJxl::new(&image).unwrap().encode_with_threads(options, JxlThreads::Fixed(1)).unwrap();
            if side == 17 {
                assert!(output.capacity() <= 4096);
            } else {
                assert!(output.len() > 512 * 1024);
                assert!(output.capacity() >= output.len());
            }
            assert_eq!(decode_jxl(&output).unwrap().raw_data().unwrap(), pixels);
            // Initial capacity must not affect the codestream, including
            // when incompressible data forces the buffer to grow repeatedly.
            let mut encoder = encoder_builder()
                .has_alpha(true)
                .lossless(true)
                .speed(EncoderSpeed::Lightning)
                .decoding_speed(0)
                .use_container(false)
                .jpeg_quality(100.0)
                .uses_original_profile(true)
                .color_encoding(ColorEncoding::Srgb)
                .build()
                .unwrap();
            let frame = EncoderFrame::new(&pixels).num_channels(4);
            let reference = encoder.encode_frame(&frame, side, side).unwrap();
            assert_eq!(output, reference);
        }
    }

    #[test]
    fn jxl_candidates_preserve_bytes_without_unused_thread_pools() {
        for mode in [PixelMode::L, PixelMode::Rgb, PixelMode::Rgba] {
            for depth in [8, 10, 12, 16] {
                for (width, height) in [(17, 19), (257, 129)] {
                    // A fresh thread makes pool initialization observable,
                    // independently of which other codec tests ran first.
                    std::thread::spawn(move || {
                        let samples: Vec<u16> = (0..width * height * mode.channels()).map(|i| ((i * 71 + i / 13) % (1 << depth)) as u16).collect();
                        let image = if depth == 8 {
                            Image::from_pixels(width as u32, height as u32, mode, samples.into_iter().map(|v| v as u8).collect(), None).unwrap()
                        } else {
                            Image::from_samples(width as u32, height as u32, mode, samples, depth, None).unwrap()
                        };
                        let prepared = PreparedJxl::new(&image).unwrap();
                        let options = SaveOptions { quality: 71, lossless: true, effort: 3, ..SaveOptions::default() };
                        let single = prepared.encode_with_threads(options, JxlThreads::Fixed(1)).unwrap();
                        JXL_ENCODE_RUNNER.with(|runner| assert!(runner.get().is_none()));
                        let candidate = prepared.encode_candidate(options).unwrap();
                        JXL_ENCODE_RUNNER.with(|runner| {
                            assert_eq!(runner.get().is_some(), image.raw_data().unwrap().len() >= JXL_PARALLEL_MIN_BYTES);
                        });
                        assert_eq!(candidate, single);
                        assert_eq!(candidate, prepared.encode(options).unwrap());
                        let lossy = SaveOptions { lossless: false, ..options };
                        assert_eq!(prepared.encode_candidate(lossy).unwrap(), prepared.encode(lossy).unwrap());
                        // Fixed-size candidate pools change only the thread
                        // count, never the bytes, and are reused per size.
                        for workers in [2, 3] {
                            JXL_CANDIDATE_RUNNER.with(|cached| cached.borrow_mut().take());
                            assert_eq!(prepared.encode_with_threads(options, JxlThreads::Fixed(workers)).unwrap(), single);
                            assert_eq!(prepared.encode_with_threads(lossy, JxlThreads::Fixed(workers)).unwrap(), prepared.encode(lossy).unwrap());
                            JXL_CANDIDATE_RUNNER.with(|cached| assert_eq!(cached.borrow().as_ref().map(|(size, _)| *size), Some(workers)));
                        }
                    })
                    .join()
                    .unwrap();
                }
            }
        }
    }

    #[test]
    fn jxl_candidate_threads_split_the_pool_and_stop_at_group_count() {
        let large = Image::from_pixels(1024, 768, PixelMode::L, vec![0; 1024 * 768], None).unwrap();
        let small = Image::from_pixels(200, 100, PixelMode::L, vec![0; 200 * 100], None).unwrap();
        let two_groups = Image::from_pixels(257, 100, PixelMode::L, vec![0; 257 * 100], None).unwrap();
        for threads in [1, 3, 8, 12] {
            rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap().install(|| {
                assert_eq!(jxl_candidate_threads(&large, 1), threads.min(12));
                assert_eq!(jxl_candidate_threads(&large, 3), (threads / 3).max(1));
                assert_eq!(jxl_candidate_threads(&large, threads), 1);
                assert_eq!(jxl_candidate_threads(&large, 0), threads.min(12));
                assert_eq!(jxl_candidate_threads(&small, 1), 1);
                assert_eq!(jxl_candidate_threads(&two_groups, 1), threads.min(2));
            });
        }
    }

    #[test]
    fn expands_luma_alpha_pixels_to_rgba() {
        assert_eq!(luma_alpha_to_rgba(&[0x10, 0x20, 0x30, 0x40]), [0x10, 0x10, 0x10, 0x20, 0x30, 0x30, 0x30, 0x40]);
    }
}
