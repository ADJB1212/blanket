use std::sync::LazyLock;

use blanket_codec_common::{SaveOptions, codec_error, validate_dimensions};
use blanket_core::{Image, PixelMode};
use pyo3::exceptions::{PyOSError, PyValueError};
use pyo3::prelude::*;

// libheif owns a process-wide plugin registry. Keep its initialization guard
// alive so each image does not tear down and recreate the codec plugins.
// Contexts, encoders, and pixel planes remain local to each operation.
static HEIF_LIBRARY: LazyLock<libheif_rs::LibHeif> = LazyLock::new(libheif_rs::LibHeif::new);

pub fn backend_info() -> (bool, bool, String) {
    let lib = &*HEIF_LIBRARY;
    let format = Some(libheif_rs::CompressionFormat::Hevc);
    let decode = !lib.decoder_descriptors(1, format).is_empty();
    let encode = !lib.encoder_descriptors(1, format, None).is_empty();
    let [major, minor, patch] = lib.version();
    (decode, encode, format!("{major}.{minor}.{patch}"))
}

pub fn inspect(data: &[u8]) -> Result<blanket_codec_common::ImageHeader, String> {
    let _lib = &*HEIF_LIBRARY;
    let context = libheif_rs::HeifContext::read_from_bytes(data).map_err(|e| e.to_string())?;
    let handle = context.primary_image_handle().map_err(|e| e.to_string())?;
    blanket_codec_common::ImageHeader::new(
        handle.width(),
        handle.height(),
        if handle.has_alpha_channel() { "RGBA" } else { "RGB" },
        handle.luma_bits_per_pixel(),
    )
}

pub fn decode_heif(data: &[u8]) -> Result<Image, String> {
    use libheif_rs::HeifContext;
    let _lib = &*HEIF_LIBRARY;
    let context = HeifContext::read_from_bytes(data).map_err(|e| e.to_string())?;
    let handle = context.primary_image_handle().map_err(|e| e.to_string())?;
    decode_handle(&handle)
}

pub fn previews(data: &[u8]) -> Result<Vec<Image>, String> {
    let _library = &*HEIF_LIBRARY;
    let context = libheif_rs::HeifContext::read_from_bytes(data).map_err(|e| e.to_string())?;
    let primary = context.primary_image_handle().map_err(|e| e.to_string())?;
    let mut ids = vec![0; primary.number_of_thumbnails()];
    let count = primary.thumbnail_ids(&mut ids);
    let mut total = 0;
    let mut result = Vec::new();
    for id in ids.into_iter().take(count) {
        let handle = primary.thumbnail(id).map_err(|e| e.to_string())?;
        blanket_codec_common::validate_frame_budget(&mut total, handle.width(), handle.height())?;
        result.push(decode_handle(&handle)?);
    }
    Ok(result)
}

fn decode_handle(handle: &libheif_rs::ImageHandle) -> Result<Image, String> {
    use libheif_rs::{ColorSpace, DecodingOptions, RgbChroma};
    let lib = &*HEIF_LIBRARY;
    validate_dimensions(handle.width(), handle.height())?;
    let depth = handle.luma_bits_per_pixel();
    if !matches!(depth, 8 | 10 | 12 | 16) {
        return Err(format!("unsupported HEIF bit depth: {depth}"));
    }
    let alpha = handle.has_alpha_channel();
    let chroma = match (depth > 8, alpha) {
        (false, false) => RgbChroma::Rgb,
        (false, true) => RgbChroma::Rgba,
        (true, false) => RgbChroma::HdrRgbLe,
        (true, true) => RgbChroma::HdrRgbaLe,
    };
    let mut options = DecodingOptions::new().ok_or("cannot allocate HEIF decoding options")?;
    // The context's tile-thread limit does not enable HEVC codec workers.
    // Small images do not have enough CTU rows to amortize starting workers.
    let threads = if u64::from(handle.width()) * u64::from(handle.height()) >= 256 * 1024 {
        std::thread::available_parallelism().map_or(1, |count| count.get().min(8))
    } else {
        1
    };
    options.set_num_codec_threads(threads as u32);
    let decoded = lib.decode(handle, ColorSpace::Rgb(chroma), Some(options)).map_err(|e| e.to_string())?;
    validate_dimensions(decoded.width(), decoded.height())?;
    let mode = if alpha { PixelMode::Rgba } else { PixelMode::Rgb };
    let plane = decoded.planes().interleaved.ok_or("missing HEIF pixel plane")?;
    let row = decoded.width() as usize * mode.channels() * if depth > 8 { 2 } else { 1 };
    if row > plane.stride || plane.stride == 0 {
        return Err("invalid HEIF row stride".into());
    }
    let mut pixels = Vec::with_capacity(row * decoded.height() as usize);
    if row == plane.stride {
        pixels.extend_from_slice(plane.data.get(..row * decoded.height() as usize).ok_or("truncated HEIF pixel plane")?);
    } else {
        for source in plane.data.chunks_exact(plane.stride).take(decoded.height() as usize) {
            pixels.extend_from_slice(&source[..row]);
        }
    }
    if depth == 8 {
        Image::from_pixels(decoded.width(), decoded.height(), mode, pixels, Some("HEIF".into()))
    } else {
        Image::from_sample_bytes(decoded.width(), decoded.height(), mode, pixels, depth, Some("HEIF".into()))
    }
    .map_err(|e| e.to_string())
}

pub fn encode_heif(image: &Image, options: SaveOptions) -> PyResult<Vec<u8>> {
    encode_heif_with_preset(image, options, "medium")
}

pub fn encode_heif_with_preset(image: &Image, options: SaveOptions, preset: &str) -> PyResult<Vec<u8>> {
    PreparedHeif::new(image)?.encode(options, preset)
}

/// Own pixel planes and encoder parameter metadata once for a preset search.
/// Every encoding resets its quality/preset and gets a fresh container.
pub struct PreparedHeif {
    native: libheif_rs::Image,
    encoder: libheif_rs::Encoder<'static>,
    presets: bool,
}

impl PreparedHeif {
    pub fn new(image: &Image) -> PyResult<Self> {
        use libheif_rs::{Channel, ColorSpace, RgbChroma};

        let pixels = image.raw_data()?;
        if image.width == 0 || image.height == 0 {
            return Err(PyValueError::new_err("cannot encode empty HEIF image"));
        }
        if !matches!(image.bit_depth, 8 | 10 | 12) {
            return Err(PyValueError::new_err("HEIF encoding supports 8, 10, or 12 bits per channel"));
        }
        let _library = &*HEIF_LIBRARY;
        let chroma = match (image.bit_depth > 8, image.mode == PixelMode::Rgba) {
            (false, false) => RgbChroma::Rgb,
            (false, true) => RgbChroma::Rgba,
            (true, false) => RgbChroma::HdrRgbLe,
            (true, true) => RgbChroma::HdrRgbaLe,
        };
        let grayscale = image.mode == PixelMode::L;
        let color_space = if grayscale { ColorSpace::Monochrome } else { ColorSpace::Rgb(chroma) };
        let channel = if grayscale { Channel::Y } else { Channel::Interleaved };
        let mut native = libheif_rs::Image::new(image.width, image.height, color_space).map_err(codec_error)?;
        native.create_plane(channel, image.width, image.height, image.bit_depth).map_err(codec_error)?;
        let row = image.width as usize * image.mode.channels() * if image.bit_depth > 8 { 2 } else { 1 };
        let planes = native.planes_mut();
        let plane = if grayscale { planes.y } else { planes.interleaved }.ok_or_else(|| PyOSError::new_err("missing HEIF pixel plane"))?;
        // Monochrome HEVC avoids RGB expansion, color conversion, and encoding
        // two constant chroma planes. Wide monochrome planes use native endian.
        if grayscale && image.bit_depth > 8 && cfg!(target_endian = "big") {
            for (source, target) in pixels.chunks_exact(row).zip(plane.data.chunks_exact_mut(plane.stride)) {
                for (sample, dest) in source.as_chunks::<2>().0.iter().zip(target[..row].as_chunks_mut::<2>().0) {
                    *dest = u16::from_le_bytes(*sample).to_ne_bytes();
                }
            }
        } else if row == plane.stride {
            plane.data[..pixels.len()].copy_from_slice(pixels);
        } else {
            for (source, target) in pixels.chunks_exact(row).zip(plane.data.chunks_exact_mut(plane.stride)) {
                target[..row].copy_from_slice(source);
            }
        }
        let encoder = HEIF_LIBRARY.encoder_for_format(libheif_rs::CompressionFormat::Hevc).map_err(codec_error)?;
        let presets = encoder.name().starts_with("x265 ");
        Ok(Self { native, encoder, presets })
    }

    pub fn supports_presets(&self) -> bool {
        self.presets
    }

    pub fn encode(&mut self, options: SaveOptions, preset: &str) -> PyResult<Vec<u8>> {
        use libheif_rs::{EncoderParameterValue, EncoderQuality, HeifContext};

        // Other HEVC plugins keep their defaults rather than receiving an
        // x265-specific parameter.
        if self.presets {
            self.encoder.set_parameter_value("preset", EncoderParameterValue::String(preset.into())).map_err(codec_error)?;
        }
        self.encoder
            .set_quality(if options.lossless { EncoderQuality::LossLess } else { EncoderQuality::Lossy(options.quality) })
            .map_err(codec_error)?;
        let mut context = HeifContext::new().map_err(codec_error)?;
        if !options.icc_profile.is_empty() {
            self.native
                .set_color_profile_raw(&libheif_rs::ColorProfileRaw::new(libheif_rs::color_profile_types::PROF, options.icc_profile.to_vec()))
                .map_err(codec_error)?;
        }
        let handle = context.encode_image(&self.native, &mut self.encoder, None).map_err(codec_error)?;
        if !options.exif.is_empty() {
            context.add_exif_metadata(&handle, options.exif.strip_prefix(b"Exif\0\0").unwrap_or(options.exif)).map_err(codec_error)?;
        }
        if !options.xmp.is_empty() {
            context.add_xmp_metadata(&handle, options.xmp).map_err(codec_error)?;
        }
        context.write_to_bytes().map_err(codec_error)
    }
}

pub fn read_metadata(data: &[u8]) -> PyResult<Vec<(String, Vec<u8>)>> {
    let _library = &*HEIF_LIBRARY;
    let context = libheif_rs::HeifContext::read_from_bytes(data).map_err(codec_error)?;
    let handle = context.primary_image_handle().map_err(codec_error)?;
    let mut result = Vec::new();
    if let Some(profile) = handle.color_profile_raw() {
        result.push(("icc_profile".into(), profile.data));
    }
    let mut ids = vec![0; handle.number_of_metadata_blocks(0).max(0) as usize];
    let count = handle.metadata_block_ids(&mut ids, 0);
    for id in ids.into_iter().take(count) {
        let key = match handle.metadata_type(id) {
            Some("Exif") => "exif",
            Some("mime") if handle.metadata_content_type(id) == Some("application/rdf+xml") => "xmp",
            _ => continue,
        };
        let mut raw = handle.metadata(id).map_err(codec_error)?;
        if key == "exif" {
            if raw.len() < 4 {
                continue;
            }
            let offset = u32::from_be_bytes(raw[..4].try_into().expect("four bytes")) as usize;
            let Some(tiff) = offset.checked_add(4).and_then(|offset| raw.get(offset..)) else {
                continue;
            };
            raw = [b"Exif\0\0".as_slice(), tiff].concat();
        }
        result.push((key.into(), raw));
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_previews_decode_only_thumbnail_handles_and_retain_precision() {
        if !backend_info().1 {
            return;
        }
        for depth in [8, 10, 12] {
            let image = if depth == 8 {
                Image::from_pixels(128, 96, PixelMode::Rgb, vec![85; 128 * 96 * 3], None).unwrap()
            } else {
                Image::from_samples(128, 96, PixelMode::Rgb, vec![(1 << depth) / 3; 128 * 96 * 3], depth, None).unwrap()
            };
            let mut prepared = PreparedHeif::new(&image).unwrap();
            let mut context = libheif_rs::HeifContext::new().unwrap();
            let primary = context.encode_image(&prepared.native, &mut prepared.encoder, None).unwrap();
            let thumbnail = context.encode_thumbnail(&prepared.native, &primary, 32, &mut prepared.encoder, None).unwrap().unwrap();
            let size = (thumbnail.width(), thumbnail.height());
            let data = context.write_to_bytes().unwrap();
            let images = previews(&data).unwrap();
            assert_eq!(images.len(), 1);
            assert_eq!((images[0].width, images[0].height), size);
            assert_eq!(images[0].bit_depth, depth);
            assert_ne!(size, (128, 96));
        }
    }

    #[test]
    fn reused_heif_planes_match_fresh_planes_across_presets_and_quality() {
        for mode in [PixelMode::L, PixelMode::Rgb, PixelMode::Rgba] {
            for depth in [8, 10, 12] {
                let count = 17 * 19 * mode.channels();
                let samples: Vec<u16> = (0..count).map(|i| ((i * 37) % (1 << depth)) as u16).collect();
                let image = if depth == 8 {
                    Image::from_pixels(17, 19, mode, samples.iter().map(|&v| v as u8).collect(), None).unwrap()
                } else {
                    Image::from_samples(17, 19, mode, samples, depth, None).unwrap()
                };
                let mut prepared = PreparedHeif::new(&image).unwrap();
                for (preset, lossless) in [("fast", false), ("slow", true), ("fast", false)] {
                    let options = SaveOptions { quality: 71, lossless, ..SaveOptions::default() };
                    let reused = prepared.encode(options, preset).unwrap();
                    let fresh = PreparedHeif::new(&image).unwrap().encode(options, preset).unwrap();
                    assert_eq!(reused, fresh, "mode={mode:?}, depth={depth}, preset={preset}");
                }
            }
        }
    }
}
