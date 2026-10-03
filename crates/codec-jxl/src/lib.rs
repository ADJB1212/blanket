use blanket_codec_common::{SaveOptions, codec_error, validate_dimensions};
use blanket_core::{Image, PixelMode};
use jpegxl_rs::encode::{ColorEncoding, EncoderFrame, EncoderSpeed};
use jpegxl_rs::parallel::ParallelRunner;
use jpegxl_rs::parallel::resizable_runner::ResizableRunner;
use jpegxl_rs::parallel::threads_runner::ThreadsRunner;
use jpegxl_rs::{decoder_builder, encoder_builder};
use pyo3::exceptions::{PyOSError, PyValueError};
use pyo3::prelude::*;
use std::borrow::Cow;

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
    // Size the pool after reading basic info instead of starting one worker
    // per CPU even for small images with only a few independently coded groups.
    let (info, pixels) = JXL_DECODE_RUNNER.with(|runner| {
        let runner = runner.as_ref().ok_or_else(|| "cannot allocate JPEG XL thread pool".to_owned())?;
        let decoder = decoder_builder().parallel_runner(runner).build().map_err(|error| error.to_string())?;
        decoder.decode(data).map_err(|error| error.to_string())
    })?;
    validate_dimensions(info.width, info.height)?;

    let pixels = match pixels {
        jpegxl_rs::decode::Pixels::Uint8(pixels) => pixels,
        jpegxl_rs::decode::Pixels::Uint16(pixels) => {
            let (mode, samples) = match (info.num_color_channels, info.has_alpha_channel) {
                (1, false) => (PixelMode::L, pixels),
                (3, false) => (PixelMode::Rgb, pixels),
                (3, true) => (PixelMode::Rgba, pixels),
                (1, true) => (
                    PixelMode::Rgba,
                    pixels.as_chunks::<2>().0.iter().flat_map(|v| [v[0], v[0], v[0], v[1]]).collect(),
                ),
                _ => return Err("unsupported JPEG XL channel layout".into()),
            };
            return Image::from_samples(info.width, info.height, mode, samples, 16, Some("JXL".into())).map_err(|e| e.to_string());
        }
        // Retain the existing display conversion for floating-point inputs.
        _ => JXL_DECODE_RUNNER.with(|runner| {
            let runner = runner.as_ref().ok_or_else(|| "cannot allocate JPEG XL thread pool".to_owned())?;
            let decoder = decoder_builder().parallel_runner(runner).build().map_err(|e| e.to_string())?;
            decoder.decode_with::<u8>(data).map(|(_, pixels)| pixels).map_err(|e| e.to_string())
        })?,
    };

    let (mode, pixels) = match (info.num_color_channels, info.has_alpha_channel) {
        (1, false) => (PixelMode::L, pixels),
        (3, false) => (PixelMode::Rgb, pixels),
        (1, true) => (PixelMode::Rgba, luma_alpha_to_rgba(&pixels)),
        (3, true) => (PixelMode::Rgba, pixels),
        (channels, has_alpha) => {
            return Err(format!(
                "unsupported JPEG XL channel layout: {channels} color channels, alpha={has_alpha}"
            ));
        }
    };
    Image::from_pixels(info.width, info.height, mode, pixels, Some("JXL".to_owned())).map_err(|error| error.to_string())
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

enum JxlSamples<'a> {
    Byte(&'a [u8]),
    Wide(&'a [u16]),
}

fn encode_jxl_samples(image: &Image, pixels: JxlSamples<'_>, options: SaveOptions, threads: JxlThreads) -> PyResult<Vec<u8>> {
    let (color_encoding, has_alpha) = match image.mode {
        PixelMode::L => (ColorEncoding::SrgbLuma, false),
        PixelMode::Rgb => (ColorEncoding::Srgb, false),
        PixelMode::Rgba => (ColorEncoding::Srgb, true),
        _ => return Err(PyValueError::new_err("unsupported mode for JPEG XL")),
    };
    let speed = jxl_encoder_speed(options.effort)?;
    let input_bytes = match &pixels {
        JxlSamples::Byte(samples) => samples.len(),
        JxlSamples::Wide(samples) => std::mem::size_of_val(*samples),
    };
    let encode = |runner: Option<&dyn ParallelRunner>| {
        let builder = encoder_builder()
            // jpegxl-rs otherwise zero-fills 512 KiB even for tiny candidates.
            // It grows this buffer when needed and shrinks it after encoding.
            .init_buffer_size(input_bytes.saturating_add(1024).clamp(4096, 512 * 1024))
            .has_alpha(has_alpha)
            .lossless(options.lossless)
            .speed(speed)
            .decoding_speed(0)
            .use_container(false)
            .jpeg_quality(f32::from(options.quality))
            .uses_original_profile(options.lossless || options.quality == 100)
            .color_encoding(color_encoding);
        let mut encoder = if let Some(runner) = runner {
            builder.parallel_runner(runner).build()
        } else {
            builder.build()
        }
        .map_err(codec_error)?;
        match pixels {
            JxlSamples::Byte(pixels) => {
                let frame = EncoderFrame::new(pixels).num_channels(image.mode.channels() as u32);
                encoder.encode_frame(&frame, image.width, image.height).map_err(codec_error)
            }
            JxlSamples::Wide(pixels) => {
                let frame = EncoderFrame::new(pixels).num_channels(image.mode.channels() as u32);
                encoder.encode_frame(&frame, image.width, image.height).map_err(codec_error)
            }
        }
    };
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
    if depth == 16 {
        // SAFETY: every bit pattern is a valid u16; align_to checks alignment
        // and bounds. The immutable borrow cannot outlive the input buffer.
        let (prefix, samples, suffix) = unsafe { raw.align_to::<u16>() };
        if prefix.is_empty() && suffix.is_empty() {
            return Cow::Borrowed(samples);
        }
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
        let threads = if self.image.raw_data()?.len() < JXL_PARALLEL_MIN_BYTES {
            JxlThreads::Fixed(1)
        } else {
            JxlThreads::Pool
        };
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
    fn jxl_wide_preparation_borrows_only_native_aligned_samples() {
        #[repr(align(2))]
        struct Aligned([u8; 10]);
        let storage = Aligned([0, 0, 255, 255, 1, 128, 17, 23, 42, 0]);
        for raw in [&storage.0[..8], &storage.0[1..9]] {
            let prepared = jxl_wide_samples(raw, 16);
            assert_eq!(prepared.as_ref(), blanket_core::simd::normalize_u16(raw, 16));
            assert_eq!(
                matches!(prepared, Cow::Borrowed(_)),
                cfg!(target_endian = "little") && raw.as_ptr().addr().is_multiple_of(2)
            );
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
            let options = SaveOptions {
                quality: 100,
                compress_level: 6,
                lossless: true,
                effort: 1,
            };
            let output = PreparedJxl::new(&image)
                .unwrap()
                .encode_with_threads(options, JxlThreads::Fixed(1))
                .unwrap();
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
                        let samples: Vec<u16> = (0..width * height * mode.channels())
                            .map(|i| ((i * 71 + i / 13) % (1 << depth)) as u16)
                            .collect();
                        let image = if depth == 8 {
                            Image::from_pixels(width as u32, height as u32, mode, samples.into_iter().map(|v| v as u8).collect(), None).unwrap()
                        } else {
                            Image::from_samples(width as u32, height as u32, mode, samples, depth, None).unwrap()
                        };
                        let prepared = PreparedJxl::new(&image).unwrap();
                        let options = SaveOptions {
                            quality: 71,
                            compress_level: 6,
                            lossless: true,
                            effort: 3,
                        };
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
                            assert_eq!(
                                prepared.encode_with_threads(lossy, JxlThreads::Fixed(workers)).unwrap(),
                                prepared.encode(lossy).unwrap()
                            );
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
        assert_eq!(
            luma_alpha_to_rgba(&[0x10, 0x20, 0x30, 0x40]),
            [0x10, 0x10, 0x10, 0x20, 0x30, 0x30, 0x30, 0x40]
        );
    }
}
