use std::borrow::Cow;
use std::io::Cursor;

use blanket_codec_common::{SaveOptions, codec_error, u16_bytes, validate_dimensions};
use blanket_codec_image::{color_type, decode_rust_image};
use blanket_core::{Image, PixelMode};
use image::{ExtendedColorType, ImageEncoder, ImageFormat as RustFormat};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

pub fn decode_region(data: &[u8], bounds: (u32, u32, u32, u32)) -> Result<Option<Image>, String> {
    use tiff::tags::Tag;
    let mut decoder = tiff::decoder::Decoder::new(Cursor::new(data)).map_err(|e| e.to_string())?;
    if decoder.get_tag_u32(Tag::Orientation).is_ok_and(|value| (2..=8).contains(&value)) {
        return Ok(None);
    }
    let (width, height) = decoder.dimensions().map_err(|e| e.to_string())?;
    validate_dimensions(width, height)?;
    let (left, top, right, bottom) = bounds;
    if left >= right || top >= bottom || right > width || bottom > height {
        return Err("region must be a nonempty rectangle inside the image".into());
    }
    if decoder.get_tag_u32(Tag::PlanarConfiguration).unwrap_or(1) != 1
        || decoder.get_tag_u32(Tag::PhotometricInterpretation).is_ok_and(|v| !matches!(v, 1 | 2))
        || decoder.get_tag_u16_vec(Tag::SampleFormat).is_ok_and(|v| v.iter().any(|&f| f != 1))
        || decoder.get_tag_u16_vec(Tag::ExtraSamples).is_ok_and(|v| v.contains(&1))
    {
        return Ok(None);
    }
    let (mode, depth) = match decoder.colortype().map_err(|e| e.to_string())? {
        tiff::ColorType::Gray(8) => (PixelMode::L, 8),
        tiff::ColorType::Gray(16) => (PixelMode::I16, 16),
        tiff::ColorType::RGB(d @ (8 | 16)) => (PixelMode::Rgb, d),
        tiff::ColorType::RGBA(d @ (8 | 16)) => (PixelMode::Rgba, d),
        _ => return Ok(None),
    };
    let (cw, ch) = decoder.chunk_dimensions();
    if cw == 0 || ch == 0 {
        return Err("invalid TIFF chunk dimensions".into());
    }
    let across = width.div_ceil(cw);
    let stride = mode.channels() * (depth as usize / 8);
    let ow = (right - left) as usize;
    let mut pixels = blanket_core::raster::alloc_pixels(ow * (bottom - top) as usize * stride).map_err(|e| e.to_string())?;
    for cy in top / ch..=(bottom - 1) / ch {
        for cx in left / cw..=(right - 1) / cw {
            let index = cy * across + cx;
            let decoded = decoder.read_chunk(index).map_err(|e| e.to_string())?;
            let byte_count = match &decoded {
                tiff::decoder::DecodingResult::U8(v) => v.len(),
                tiff::decoder::DecodingResult::U16(v) => v.len() * 2,
                _ => return Err("unexpected TIFF sample type".into()),
            };
            let (w, h) = decoder.chunk_data_dimensions(index);
            if byte_count != w as usize * h as usize * stride {
                return Err("invalid TIFF chunk sample count".into());
            }
            let x0 = left.max(cx * cw);
            let x1 = right.min(cx * cw + w);
            for y in top.max(cy * ch)..bottom.min(cy * ch + h) {
                let src = ((y - cy * ch) as usize * w as usize + (x0 - cx * cw) as usize) * stride;
                let dst = ((y - top) as usize * ow + (x0 - left) as usize) * stride;
                let len = (x1 - x0) as usize * stride;
                let output = &mut pixels[dst..dst + len];
                match &decoded {
                    tiff::decoder::DecodingResult::U8(bytes) => output.copy_from_slice(&bytes[src..src + len]),
                    tiff::decoder::DecodingResult::U16(values) => {
                        for (bytes, value) in output.as_chunks_mut::<2>().0.iter_mut().zip(&values[src / 2..(src + len) / 2]) {
                            *bytes = value.to_le_bytes();
                        }
                    }
                    _ => unreachable!("validated TIFF sample type"),
                }
            }
        }
    }
    let result = if depth == 8 {
        Image::from_pixels(right - left, bottom - top, mode, pixels, Some("TIFF".into()))
    } else if mode == PixelMode::I16 {
        Image::from_integer_bytes(right - left, bottom - top, mode, pixels).map(|mut image| {
            image.format = Some("TIFF".into());
            image
        })
    } else {
        Image::from_sample_bytes(right - left, bottom - top, mode, pixels, depth, Some("TIFF".into()))
    };
    result.map(Some).map_err(|e| e.to_string())
}

pub fn inspect(data: &[u8]) -> Result<blanket_codec_common::ImageHeader, String> {
    use tiff::tags::Tag;
    let mut decoder = tiff::decoder::Decoder::new(Cursor::new(data)).map_err(|e| e.to_string())?;
    let (mut width, mut height) = decoder.dimensions().map_err(|e| e.to_string())?;
    if decoder.get_tag_u32(Tag::Orientation).is_ok_and(|value| (5..=8).contains(&value)) {
        std::mem::swap(&mut width, &mut height);
    }
    let bits = decoder.get_tag_u16_vec(Tag::BitsPerSample).unwrap_or_else(|_| vec![1]);
    let depth = *bits.first().ok_or("missing TIFF sample depth")?;
    if bits.iter().any(|&b| b != depth) {
        return Err("mixed TIFF sample depths".into());
    }
    let channels = decoder.get_tag_u32(Tag::SamplesPerPixel).unwrap_or(1);
    let photo = decoder.get_tag_u32(Tag::PhotometricInterpretation).map_err(|e| e.to_string())?;
    let sample_format = decoder.get_tag_u16_vec(Tag::SampleFormat).unwrap_or_else(|_| vec![1]);
    let sample_format = *sample_format.first().ok_or("missing TIFF sample format")?;
    let mode = match (photo, channels, depth, sample_format) {
        (0 | 1, 1, 1, _) => "1",
        (0 | 1, 1, 32 | 64, 3) => "F",
        (2, 3, 32, 3) => "RGB;F",
        (2, 4, 32, 3) => "RGBA;F",
        (0 | 1, 1, 32, _) => "I",
        (0 | 1, 1, 16, 1) => "I;16",
        (0 | 1, 1, _, _) => "L",
        (0 | 1, 2, _, _) => "LA",
        (2 | 6, 3, _, _) => "RGB",
        (2, 4, _, _) => "RGBA",
        (3, 1, _, _) => "P",
        (5, 4, _, _) => "CMYK",
        (8 | 9, 3, _, _) => "LAB",
        _ => return Err("unsupported TIFF header layout".into()),
    };
    blanket_codec_common::ImageHeader::new(width, height, mode, depth.try_into().map_err(|_| "invalid TIFF depth")?)
}

fn decode_special_tiff(data: &[u8], decoder: &mut tiff::decoder::Decoder<Cursor<&[u8]>>) -> Result<Option<Image>, String> {
    let photometric = decoder.get_tag_u32(tiff::tags::Tag::PhotometricInterpretation).ok();
    if decoder.get_tag_u32(tiff::tags::Tag::Compression).ok() == Some(7) {
        return decode_jpeg_tiff(data, decoder).map(Some);
    }
    if matches!(photometric, Some(8 | 9)) {
        let signed = photometric == Some(8);
        // Decode LAB as raw three-channel samples; the TIFF crate cannot read LAB buffers.
        let normalized = lab_tiff_as_rgb(data)?;
        let mut decoder = tiff::decoder::Decoder::new(Cursor::new(normalized.as_slice())).map_err(|error| error.to_string())?;
        if decoder.colortype().map_err(|error| error.to_string())? != tiff::ColorType::RGB(8) {
            return Err("only 8-bit LAB TIFF images are supported".into());
        }
        let (width, height) = decoder.dimensions().map_err(|error| error.to_string())?;
        validate_dimensions(width, height)?;
        let tiff::decoder::DecodingResult::U8(pixels) = decoder.read_image().map_err(|error| error.to_string())? else {
            return Err("invalid LAB TIFF samples".into());
        };
        let pixels = if signed { blanket_core::raster::lab_raw_bytes(&pixels) } else { pixels };
        return Image::from_pixels(width, height, PixelMode::Lab, pixels, Some("TIFF".into())).map(Some).map_err(|error| error.to_string());
    }
    if decoder.colortype().ok() == Some(tiff::ColorType::CMYK(8)) {
        let (width, height) = decoder.dimensions().map_err(|error| error.to_string())?;
        validate_dimensions(width, height)?;
        let tiff::decoder::DecodingResult::U8(pixels) = decoder.read_image().map_err(|error| error.to_string())? else {
            return Err("invalid CMYK TIFF samples".into());
        };
        return Image::from_pixels(width, height, PixelMode::Cmyk, pixels, Some("TIFF".into())).map(Some).map_err(|error| error.to_string());
    }
    if decoder.get_tag_u32(tiff::tags::Tag::Compression).ok() == Some(5)
        && let Some(image) = decode_lzw_strips(data, decoder)?
    {
        return Ok(Some(image));
    }
    if let Some(image) = decode_direct_tiff(decoder)? {
        return Ok(Some(image));
    }
    let mode = match decoder.colortype().ok() {
        Some(tiff::ColorType::Gray(32)) => PixelMode::F,
        Some(tiff::ColorType::RGB(32)) => PixelMode::RgbF,
        Some(tiff::ColorType::RGBA(32)) => PixelMode::RgbaF,
        _ => return Ok(None),
    };
    let (width, height) = decoder.dimensions().map_err(|error| error.to_string())?;
    validate_dimensions(width, height)?;
    let tiff::decoder::DecodingResult::F32(values) = decoder.read_image().map_err(|error| error.to_string())? else {
        return Ok(None);
    };
    let mut image = Image::from_color_float_bytes(width, height, mode, values.into_iter().flat_map(f32::to_le_bytes).collect())
        .map_err(|error| error.to_string())?;
    image.format = Some("TIFF".to_owned());
    Ok(Some(image))
}

fn decode_direct_tiff(decoder: &mut tiff::decoder::Decoder<Cursor<&[u8]>>) -> Result<Option<Image>, String> {
    use tiff::tags::SampleFormat;

    let (mode, depth) = match decoder.colortype().map_err(|error| error.to_string())? {
        tiff::ColorType::Gray(depth @ (8 | 16)) => (PixelMode::L, depth),
        tiff::ColorType::GrayA(8) => (PixelMode::La, 8),
        tiff::ColorType::RGB(depth @ (8 | 16)) => (PixelMode::Rgb, depth),
        tiff::ColorType::RGBA(depth @ (8 | 16)) => (PixelMode::Rgba, depth),
        tiff::ColorType::Gray(32) => (PixelMode::F, 32),
        tiff::ColorType::RGB(32) => (PixelMode::RgbF, 32),
        tiff::ColorType::RGBA(32) => (PixelMode::RgbaF, 32),
        _ => return Ok(None),
    };
    let layout = decoder.image_buffer_layout().map_err(|error| error.to_string())?;
    let sample_format = if mode.is_float() { SampleFormat::IEEEFP } else { SampleFormat::Uint };
    if layout.planes != 1 || layout.sample_format != sample_format {
        return Ok(None);
    }
    let (width, height) = decoder.dimensions().map_err(|error| error.to_string())?;
    validate_dimensions(width, height)?;
    let sample_bytes = usize::from(depth / 8);
    let length = (width as usize)
        .checked_mul(height as usize)
        .and_then(|n| n.checked_mul(mode.channels() * sample_bytes))
        .ok_or("TIFF image buffer size overflow")?;
    if layout.complete_len != length {
        return Err("invalid TIFF image buffer layout".into());
    }
    let limit = if mode.is_float() {
        tiff::decoder::Limits::default().decoding_buffer_size as u64
    } else {
        image::Limits::default().max_alloc.unwrap_or(u64::MAX)
    };
    if length as u64 > limit {
        return Err("TIFF decoding buffer limit exceeded".into());
    }
    let mut pixels = blanket_core::buffers::buffer((width, height), mode.channels() * sample_bytes).map_err(|error| error.to_string())?;
    decoder.read_image_bytes(&mut pixels).map_err(|error| error.to_string())?;
    if cfg!(target_endian = "big") && sample_bytes > 1 {
        for sample in pixels.chunks_exact_mut(sample_bytes) {
            sample.reverse();
        }
    }
    let image = if mode.is_float() {
        Image::from_color_float_bytes(width, height, mode, pixels).map(|mut image| {
            image.format = Some("TIFF".into());
            image
        })
    } else if depth == 8 {
        Image::from_pixels(width, height, mode, pixels, Some("TIFF".into()))
    } else {
        Image::from_sample_bytes(width, height, mode, pixels, depth, Some("TIFF".into()))
    };
    image.map(Some).map_err(|error| error.to_string())
}

fn decode_lzw_strips(data: &[u8], decoder: &mut tiff::decoder::Decoder<Cursor<&[u8]>>) -> Result<Option<Image>, String> {
    use tiff::tags::Tag;

    let mode = match decoder.colortype().map_err(|e| e.to_string())? {
        tiff::ColorType::Gray(8) => PixelMode::L,
        tiff::ColorType::RGB(8) => PixelMode::Rgb,
        tiff::ColorType::RGBA(8) => PixelMode::Rgba,
        _ => return Ok(None),
    };
    if decoder.get_tag_u32(Tag::PlanarConfiguration).unwrap_or(1) != 1 || decoder.get_tag_u32(Tag::TileWidth).is_ok() {
        return Ok(None);
    }
    let (width, height) = decoder.dimensions().map_err(|e| e.to_string())?;
    validate_dimensions(width, height)?;
    let rows = decoder.get_tag_u32(Tag::RowsPerStrip).unwrap_or(height).min(height);
    if width == 0 || rows == 0 {
        return Ok(None);
    }
    let stride = width as usize * mode.channels();
    let length = stride * height as usize;
    if !blanket_core::parallel::should_parallel(length, stride * rows as usize, 256 * 1024) {
        return Ok(None);
    }
    if decoder.strip_count().map_err(|e| e.to_string())? != height.div_ceil(rows) {
        return Err("invalid TIFF strip count".into());
    }
    let mut pixels = vec![0; length];
    let strips_per_worker = height.div_ceil(rows).div_ceil(rayon::current_num_threads() as u32) as usize;
    pixels.par_chunks_mut(stride * rows as usize * strips_per_worker).enumerate().try_for_each(|(index, output)| {
        let mut decoder = tiff::decoder::Decoder::new(Cursor::new(data)).map_err(|e| e.to_string())?;
        for (strip, output) in output.chunks_mut(stride * rows as usize).enumerate() {
            decoder.read_chunk_bytes((index * strips_per_worker + strip) as u32, output).map_err(|e| e.to_string())?;
        }
        Ok::<_, String>(())
    })?;
    Image::from_pixels(width, height, mode, pixels, Some("TIFF".into())).map(Some).map_err(|e| e.to_string())
}

fn decode_jpeg_tiff(data: &[u8], decoder: &mut tiff::decoder::Decoder<Cursor<&[u8]>>) -> Result<Image, String> {
    use tiff::tags::Tag;

    let (width, height) = decoder.dimensions().map_err(|e| e.to_string())?;
    validate_dimensions(width, height)?;
    if width == 0 || height == 0 || decoder.get_tag_u32(Tag::PlanarConfiguration).unwrap_or(1) != 1 {
        return Err("unsupported JPEG TIFF layout".into());
    }
    let samples = decoder.get_tag_u32(Tag::SamplesPerPixel).unwrap_or(1);
    let mode = match samples {
        1 => PixelMode::L,
        3 => PixelMode::Rgb,
        _ => return Err("unsupported JPEG TIFF channels".into()),
    };
    if decoder.get_tag_u16_vec(Tag::BitsPerSample).map_err(|e| e.to_string())?.iter().any(|&v| v != 8) {
        return Err("JPEG TIFF requires 8-bit samples".into());
    }
    let tiled = decoder.get_tag_u32(Tag::TileWidth).is_ok();
    let (chunk_width, chunk_height) = if tiled {
        (decoder.get_tag_u32(Tag::TileWidth).map_err(|e| e.to_string())?, decoder.get_tag_u32(Tag::TileLength).map_err(|e| e.to_string())?)
    } else {
        (width, decoder.get_tag_u32(Tag::RowsPerStrip).unwrap_or(height).min(height))
    };
    if chunk_width == 0 || chunk_height == 0 {
        return Err("invalid JPEG TIFF chunk size".into());
    }
    let across = width.div_ceil(chunk_width);
    let count = across.checked_mul(height.div_ceil(chunk_height)).ok_or("invalid JPEG TIFF chunk count")?;
    let offsets = decoder.get_tag_u64_vec(if tiled { Tag::TileOffsets } else { Tag::StripOffsets }).map_err(|e| e.to_string())?;
    let lengths = decoder.get_tag_u64_vec(if tiled { Tag::TileByteCounts } else { Tag::StripByteCounts }).map_err(|e| e.to_string())?;
    if offsets.len() != count as usize || lengths.len() != offsets.len() {
        return Err("invalid JPEG TIFF chunk table".into());
    }
    let tables = decoder.get_tag_u8_vec(Tag::JPEGTables).unwrap_or_default();
    if !tables.is_empty() && (!tables.starts_with(b"\xff\xd8") || !tables.ends_with(b"\xff\xd9")) {
        return Err("invalid TIFF JPEG tables".into());
    }
    let channels = mode.channels();
    let mut pixels = vec![0; width as usize * height as usize * channels];
    for (index, (&offset, &length)) in offsets.iter().zip(&lengths).enumerate() {
        let start = usize::try_from(offset).map_err(|_| "invalid JPEG TIFF offset")?;
        let end = usize::try_from(offset.checked_add(length).ok_or("invalid JPEG TIFF length")?).map_err(|_| "invalid JPEG TIFF length")?;
        let source = data.get(start..end).ok_or("truncated JPEG TIFF chunk")?;
        let combined;
        let jpeg = if tables.is_empty() {
            source
        } else {
            combined = [tables[..tables.len() - 2].as_ref(), source.strip_prefix(b"\xff\xd8").unwrap_or(source)].concat();
            &combined
        };
        let decoded = blanket_codec_jpeg::decode_jpeg(jpeg)?;
        let x = (index as u32 % across) * chunk_width;
        let y = (index as u32 / across) * chunk_height;
        let rows = chunk_height.min(height - y);
        let columns = chunk_width.min(width - x);
        if decoded.mode != mode || decoded.width < columns || decoded.height < rows || decoded.width > chunk_width || decoded.height > chunk_height {
            return Err("invalid JPEG TIFF chunk dimensions".into());
        }
        let raw = decoded.pixel_data().map_err(|e| e.to_string())?;
        for row in 0..rows as usize {
            let dst = ((y as usize + row) * width as usize + x as usize) * channels;
            let src = row * decoded.width as usize * channels;
            pixels[dst..dst + columns as usize * channels].copy_from_slice(&raw[src..src + columns as usize * channels]);
        }
    }
    Image::from_pixels(width, height, mode, pixels, Some("TIFF".into())).map_err(|e| e.to_string())
}

fn lab_tiff_as_rgb(data: &[u8]) -> Result<Vec<u8>, String> {
    let little = data.starts_with(b"II");
    let read = |offset: usize, count: usize| -> Result<usize, String> {
        let bytes = data.get(offset..offset.checked_add(count).ok_or("invalid TIFF offset")?).ok_or("truncated TIFF directory")?;
        let value = if little {
            bytes.iter().rev().fold(0_u64, |value, &byte| (value << 8) | u64::from(byte))
        } else {
            bytes.iter().fold(0_u64, |value, &byte| (value << 8) | u64::from(byte))
        };
        usize::try_from(value).map_err(|_| "invalid TIFF offset".into())
    };
    let big = read(2, 2)? == 43;
    let (offset, count_bytes, entry_bytes, value_offset) = if big { (read(8, 8)?, 8, 20, 12) } else { (read(4, 4)?, 2, 12, 8) };
    let count = read(offset, count_bytes)?;
    let start = offset.checked_add(count_bytes).ok_or("invalid TIFF offset")?;
    if count > data.len().saturating_sub(start) / entry_bytes {
        return Err("truncated TIFF directory".into());
    }
    for i in 0..count {
        let entry = start + i * entry_bytes;
        if read(entry, 2)? == 262 {
            if read(entry + 2, 2)? != 3 || read(entry + 4, if big { 8 } else { 4 })? != 1 {
                return Err("invalid TIFF photometric tag".into());
            }
            let mut normalized = data.to_vec();
            normalized[entry + value_offset..entry + value_offset + 2].copy_from_slice(&if little {
                2_u16.to_le_bytes()
            } else {
                2_u16.to_be_bytes()
            });
            return Ok(normalized);
        }
    }
    Err("missing TIFF photometric tag".into())
}

// DNGVersion identifies raw camera containers in the first classic TIFF IFD.
const DNG_VERSION_TAG: u16 = 50706;
// Classic TIFF directory entries contain a tag, type, count, and value/offset.
const IFD_ENTRY_BYTES: usize = 12;

// Read only the bounded directory, never scan compressed image payloads for tag bytes.
// Reject DNG containers instead of opening their previews as ordinary TIFFs.
pub fn is_dng(data: &[u8]) -> bool {
    if !data.starts_with(b"II*\0") && !data.starts_with(b"MM\0*") {
        return false;
    }
    let le = data.starts_with(b"II");
    let u16_at = |offset: usize| -> Option<u16> {
        let bytes = data.get(offset..offset.checked_add(2)?)?.try_into().ok()?;
        Some(if le { u16::from_le_bytes(bytes) } else { u16::from_be_bytes(bytes) })
    };
    let Some(bytes) = data.get(4..8).and_then(|v| v.try_into().ok()) else {
        return false;
    };
    let offset = if le { u32::from_le_bytes(bytes) } else { u32::from_be_bytes(bytes) } as usize;
    let Some(count) = u16_at(offset) else { return false };
    (0..usize::from(count)).any(|i| offset.checked_add(2 + i * IFD_ENTRY_BYTES).and_then(u16_at) == Some(DNG_VERSION_TAG))
}

/// Uncompressed little-endian TIFF with one strip preceding the directory.
pub fn encode_raw_tiff(image: &Image, pixels: &[u8]) -> PyResult<Vec<u8>> {
    if image.width == 0 || image.height == 0 {
        return Err(PyValueError::new_err("cannot encode an empty image"));
    }
    let (samples, bits, photometric) = match image.mode {
        PixelMode::L => (1_u16, 8_u16, 1_u16),
        PixelMode::F => (1, 32, 1),
        PixelMode::RgbF => (3, 32, 2),
        PixelMode::RgbaF => (4, 32, 2),
        PixelMode::Rgb => (3, 8, 2),
        PixelMode::Rgba => (4, 8, 2),
        PixelMode::Cmyk => (4, 8, 5),
        _ => unreachable!("caller validates raw TIFF modes"),
    };
    let too_large = || PyValueError::new_err("image is too large for TIFF");
    let length = u32::try_from(pixels.len()).map_err(|_| too_large())?;
    let bits_offset = 8 + pixels.len().next_multiple_of(2);
    let ifd_offset = bits_offset + if samples > 2 { usize::from(samples) * 2 } else { 0 };
    let mut entries: Vec<(u16, u16, u32, u32)> = vec![
        (256, 4, 1, image.width),
        (257, 4, 1, image.height),
        (258, 3, u32::from(samples), if samples > 2 { u32::try_from(bits_offset).map_err(|_| too_large())? } else { u32::from(bits) }),
        (259, 3, 1, 1),
        (262, 3, 1, u32::from(photometric)),
        (273, 4, 1, 8),
        (277, 3, 1, u32::from(samples)),
        (278, 4, 1, image.height),
        (279, 4, 1, length),
        (284, 3, 1, 1),
    ];
    if matches!(image.mode, PixelMode::Rgba | PixelMode::RgbaF) {
        entries.push((338, 3, 1, 2));
    }
    if image.mode.is_float() {
        entries.push((339, 3, 1, 3));
    }
    let size = ifd_offset + 2 + entries.len() * 12 + 4;
    u32::try_from(size).map_err(|_| too_large())?;
    let mut output = Vec::with_capacity(size);
    output.extend_from_slice(b"II*\0");
    output.extend_from_slice(&(ifd_offset as u32).to_le_bytes());
    output.extend_from_slice(pixels);
    output.resize(bits_offset, 0);
    if samples > 2 {
        for _ in 0..samples {
            output.extend_from_slice(&bits.to_le_bytes());
        }
    }
    output.extend_from_slice(&(entries.len() as u16).to_le_bytes());
    for (tag, kind, count, value) in entries {
        output.extend_from_slice(&tag.to_le_bytes());
        output.extend_from_slice(&kind.to_le_bytes());
        output.extend_from_slice(&count.to_le_bytes());
        if kind == 3 && count == 1 {
            output.extend_from_slice(&(value as u16).to_le_bytes());
            output.extend_from_slice(&[0, 0]);
        } else {
            output.extend_from_slice(&value.to_le_bytes());
        }
    }
    output.extend_from_slice(&[0; 4]);
    Ok(output)
}

pub fn decode_tiff(data: &[u8]) -> Result<Image, String> {
    let mut decoder = tiff::decoder::Decoder::new(Cursor::new(data)).map_err(|error| error.to_string())?;
    let orientation = decoder.get_tag_u32(tiff::tags::Tag::Orientation).unwrap_or(1);
    let mut image = decode_special_tiff(data, &mut decoder)?.map_or_else(|| decode_rust_image(data, RustFormat::Tiff, "TIFF"), Ok)?;
    if image.mode == PixelMode::L && image.bit_depth == 16 {
        image.mode = PixelMode::I16;
    }
    if !(2..=8).contains(&orientation) {
        return Ok(image);
    }
    let source = image.raw_data().map_err(|error| error.to_string())?;
    let size = if orientation >= 5 { (image.height, image.width) } else { (image.width, image.height) };
    let mut pixels = blanket_core::raster::alloc_pixels(source.len()).map_err(|error| error.to_string())?;
    macro_rules! orient {
        ($channels:literal) => {
            blanket_core::geometry::transpose::<$channels>(source, &mut pixels, image.width as usize, image.height as usize, orientation as u8)
        };
    }
    match image.raw_bytes_per_pixel() {
        1 => orient!(1),
        2 => orient!(2),
        3 => orient!(3),
        4 => orient!(4),
        6 => orient!(6),
        8 => orient!(8),
        12 => orient!(12),
        16 => orient!(16),
        _ => return Err("unsupported TIFF pixel layout".into()),
    }
    let mut result = blanket_core::buffers::output(&image, size, pixels).map_err(|error| error.to_string())?;
    result.format = Some("TIFF".into());
    Ok(result)
}

pub fn decode_sequence(data: &[u8]) -> Result<blanket_codec_common::Sequence, String> {
    use blanket_codec_common::{Frame, Sequence};
    let mut sequence = Sequence::default();
    visit_pages(data, |page| {
        sequence.frames.push(Frame { image: decode_tiff(page)?, duration: None, disposal: 0, blend: 0 });
        Ok(())
    })?;
    Ok(sequence)
}

pub fn index_sequence(data: &[u8]) -> Result<blanket_codec_common::SequenceInfo, String> {
    let mut sequence = blanket_codec_common::SequenceInfo::default();
    visit_pages(data, |_| {
        sequence.count += 1;
        Ok(())
    })?;
    Ok(sequence)
}

pub fn decode_frame(data: &[u8], target: usize) -> Result<blanket_codec_common::Sequence, String> {
    use blanket_codec_common::{Frame, Sequence};
    let mut sequence = Sequence::default();
    let mut index = 0;
    visit_pages(data, |page| {
        if index == target {
            sequence.frames.push(Frame { image: decode_tiff(page)?, duration: None, disposal: 0, blend: 0 });
        }
        index += 1;
        Ok(())
    })?;
    Ok(sequence)
}

fn visit_pages(data: &[u8], mut visit: impl FnMut(&[u8]) -> Result<(), String>) -> Result<(), String> {
    use blanket_codec_common::validate_frame_budget;
    let little = data.starts_with(b"II");
    let read = |offset: usize, count: usize| -> Result<usize, String> {
        let bytes = data.get(offset..offset.checked_add(count).ok_or("invalid TIFF offset")?).ok_or("truncated TIFF directory")?;
        let value = if little {
            bytes.iter().rev().fold(0_u64, |v, &b| (v << 8) | u64::from(b))
        } else {
            bytes.iter().fold(0_u64, |v, &b| (v << 8) | u64::from(b))
        };
        usize::try_from(value).map_err(|_| "invalid TIFF offset".into())
    };
    let big = read(2, 2)? == 43;
    let (header, offset_bytes, count_bytes, entry_bytes) = if big { (8, 8, 8, 20) } else { (4, 4, 2, 12) };
    let mut offset = read(header, offset_bytes)?;
    let mut seen = rustc_hash::FxHashSet::default();
    let mut page_data = Cow::Borrowed(data);
    let mut total = 0;
    while offset != 0 {
        if !seen.insert(offset) {
            return Err("cyclic TIFF page directories".into());
        }
        let count = read(offset, count_bytes)?;
        let next = offset
            .checked_add(count_bytes)
            .and_then(|v| count.checked_mul(entry_bytes).and_then(|n| v.checked_add(n)))
            .ok_or("TIFF directory overflow")?;
        let next_offset = read(next, offset_bytes)?;
        let bytes = if little { (offset as u64).to_le_bytes() } else { (offset as u64).to_be_bytes() };
        let offset_data = if little { &bytes[..offset_bytes] } else { &bytes[8 - offset_bytes..] };
        if &page_data[header..header + offset_bytes] != offset_data {
            page_data.to_mut()[header..header + offset_bytes].copy_from_slice(offset_data);
        }
        let mut decoder = tiff::decoder::Decoder::new(Cursor::new(page_data.as_ref())).map_err(|e| e.to_string())?;
        let (width, height) = decoder.dimensions().map_err(|e| e.to_string())?;
        validate_frame_budget(&mut total, width, height)?;
        visit(&page_data)?;
        offset = next_offset;
    }
    if seen.is_empty() {
        return Err("TIFF contains no pages".into());
    }
    Ok(())
}
pub fn encode_lab_tiff(image: &Image) -> PyResult<Vec<u8>> {
    if image.width == 0 || image.height == 0 {
        return Err(PyValueError::new_err("cannot encode an empty image"));
    }
    let mut output = Cursor::new(Vec::new());
    {
        let mut encoder = tiff::encoder::TiffEncoder::new(&mut output).map_err(codec_error)?;
        let mut frame = encoder.new_image::<tiff::encoder::colortype::RGB8>(image.width, image.height).map_err(codec_error)?;
        frame.encoder().write_tag(tiff::tags::Tag::PhotometricInterpretation, 8_u16).map_err(codec_error)?;
        frame.write_data(&blanket_core::raster::lab_raw_bytes(image.pixel_data()?)).map_err(codec_error)?;
    }
    return Ok(output.into_inner());
}

pub fn encode_float_tiff(image: &Image) -> PyResult<Vec<u8>> {
    if image.width == 0 || image.height == 0 {
        return Err(PyValueError::new_err("cannot encode an empty image"));
    }
    encode_raw_tiff(image, image.raw_data()?)
}

pub fn encode_tiff(image: &Image, pixels: &[u8]) -> PyResult<Vec<u8>> {
    if matches!(image.mode, PixelMode::L | PixelMode::Rgb | PixelMode::Rgba) {
        return encode_raw_tiff(image, pixels);
    }
    let mut output = Cursor::new(Vec::with_capacity(pixels.len().saturating_add(1024)));
    image::codecs::tiff::TiffEncoder::new(&mut output).write_image(pixels, image.width, image.height, color_type(image.mode)).map_err(codec_error)?;
    Ok(output.into_inner())
}

fn float_samples(bytes: &[u8]) -> Cow<'_, [f32]> {
    if cfg!(target_endian = "little") {
        // SAFETY: all f32 bit patterns are valid; align_to checks the bounds.
        let (prefix, values, suffix) = unsafe { bytes.align_to::<f32>() };
        if prefix.is_empty() && suffix.is_empty() {
            return Cow::Borrowed(values);
        }
    }
    Cow::Owned(bytes.as_chunks::<4>().0.iter().map(|bytes| f32::from_le_bytes(*bytes)).collect())
}

pub fn encode_compressed_tiff(image: &Image, options: SaveOptions) -> PyResult<Vec<u8>> {
    use tiff::encoder::{Compression, TiffEncoder, colortype};
    use tiff::tags::Tag;

    if image.width == 0 || image.height == 0 {
        return Err(PyValueError::new_err("cannot encode an empty image"));
    }
    let mut output = Cursor::new(Vec::new());
    if options.tiff_compression == 3 {
        if image.bit_depth != 8 || !matches!(image.mode, PixelMode::L | PixelMode::Rgb) {
            return Err(blanket_codec_common::CodecErrorKind::Unsupported.into_pyerr(
                "JPEG TIFF requires 8-bit L or RGB".into(),
                Some("TIFF"),
                "encode",
            ));
        }
        let jpeg = blanket_codec_jpeg::encode_jpeg(image, image.pixel_data()?, options.quality)?;
        {
            let mut encoder = TiffEncoder::new(&mut output).map_err(codec_error)?;
            let mut directory = encoder.image_directory().map_err(codec_error)?;
            let offset = u32::try_from(directory.write_data(jpeg.as_slice()).map_err(codec_error)?).map_err(codec_error)?;
            let channels = image.mode.channels() as u16;
            for (tag, value) in [
                (Tag::ImageWidth, image.width),
                (Tag::ImageLength, image.height),
                (Tag::RowsPerStrip, image.height),
                (Tag::StripOffsets, offset),
                (Tag::StripByteCounts, jpeg.len() as u32),
            ] {
                directory.write_tag(tag, value).map_err(codec_error)?;
            }
            directory.write_tag(Tag::BitsPerSample, vec![8_u16; channels as usize].as_slice()).map_err(codec_error)?;
            directory.write_tag(Tag::Compression, 7_u16).map_err(codec_error)?;
            directory.write_tag(Tag::PhotometricInterpretation, if channels == 1 { 1_u16 } else { 6 }).map_err(codec_error)?;
            directory.write_tag(Tag::SamplesPerPixel, channels).map_err(codec_error)?;
            directory.write_tag(Tag::PlanarConfiguration, 1_u16).map_err(codec_error)?;
            if channels == 3 {
                directory.write_tag(Tag::Unknown(530), &[2_u16, 2][..]).map_err(codec_error)?;
            }
            directory.finish().map_err(codec_error)?;
        }
        return Ok(output.into_inner());
    }
    let compression = match options.tiff_compression {
        1 => Compression::Lzw,
        2 => Compression::Deflate(Default::default()),
        _ => return Err(PyValueError::new_err("unsupported TIFF compression")),
    };
    {
        let mut encoder = TiffEncoder::new(&mut output).map_err(codec_error)?.with_compression(compression);
        if image.mode.is_float() {
            let values = float_samples(image.raw_data()?);
            match image.mode {
                PixelMode::F => encoder.write_image::<colortype::Gray32Float>(image.width, image.height, &values),
                PixelMode::RgbF => encoder.write_image::<colortype::RGB32Float>(image.width, image.height, &values),
                _ => encoder.write_image::<colortype::RGBA32Float>(image.width, image.height, &values),
            }
            .map_err(codec_error)?;
        } else if image.bit_depth > 8 {
            let values = if image.bit_depth == 16 {
                blanket_core::samples::view(image)?
            } else {
                Cow::Owned(blanket_core::simd::normalize_u16(image.raw_data()?, image.bit_depth))
            };
            match image.mode {
                PixelMode::L => encoder.write_image::<colortype::Gray16>(image.width, image.height, &values),
                PixelMode::Rgb => encoder.write_image::<colortype::RGB16>(image.width, image.height, &values),
                PixelMode::Rgba => encoder.write_image::<colortype::RGBA16>(image.width, image.height, &values),
                _ => return Err(PyValueError::new_err("unsupported compressed TIFF mode")),
            }
            .map_err(codec_error)?;
        } else if image.mode == PixelMode::Lab {
            let mut frame = encoder.new_image::<colortype::RGB8>(image.width, image.height).map_err(codec_error)?;
            frame.encoder().write_tag(Tag::PhotometricInterpretation, 8_u16).map_err(codec_error)?;
            frame.write_data(&blanket_core::raster::lab_raw_bytes(image.pixel_data()?)).map_err(codec_error)?;
        } else {
            let pixels = image.pixel_data()?;
            match image.mode {
                PixelMode::L | PixelMode::One => encoder.write_image::<colortype::Gray8>(image.width, image.height, pixels),
                PixelMode::Rgb => encoder.write_image::<colortype::RGB8>(image.width, image.height, pixels),
                PixelMode::Rgba => encoder.write_image::<colortype::RGBA8>(image.width, image.height, pixels),
                PixelMode::Cmyk => encoder.write_image::<colortype::CMYK8>(image.width, image.height, pixels),
                _ => return Err(PyValueError::new_err("unsupported compressed TIFF mode")),
            }
            .map_err(codec_error)?;
        }
    }
    Ok(output.into_inner())
}

pub fn encode_wide_tiff(image: &Image) -> PyResult<Vec<u8>> {
    let color = match image.mode {
        PixelMode::L => ExtendedColorType::L16,
        PixelMode::Rgb => ExtendedColorType::Rgb16,
        PixelMode::Rgba => ExtendedColorType::Rgba16,
        _ => return Err(PyValueError::new_err("unsupported high-bit-depth mode")),
    };
    let raw = image.raw_data()?;
    let normalized;
    let samples = if image.bit_depth == 16 && cfg!(target_endian = "little") {
        raw
    } else {
        normalized = blanket_core::simd::normalize_u16(raw, image.bit_depth);
        u16_bytes(&normalized)
    };
    let mut output = Cursor::new(Vec::with_capacity(samples.len().saturating_add(1024)));
    image::codecs::tiff::TiffEncoder::new(&mut output).write_image(samples, image.width, image.height, color).map_err(codec_error)?;
    return Ok(output.into_inner());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float_samples_borrow_only_aligned_native_storage() {
        let bits = [0_u32, 0x8000_0000, 0x3f80_0000, 0x7fc0_1234, 0x7f80_0000, 0xff80_0000];
        for offset in 0..8 {
            let mut storage = vec![173; bits.len() * 4 + 8];
            for (dst, value) in storage[offset..offset + bits.len() * 4].as_chunks_mut::<4>().0.iter_mut().zip(bits) {
                *dst = value.to_le_bytes();
            }
            let bytes = &storage[offset..offset + bits.len() * 4];
            let samples = float_samples(bytes);
            assert_eq!(samples.iter().map(|value| value.to_bits()).collect::<Vec<_>>(), bits);
            assert_eq!(matches!(samples, Cow::Borrowed(_)), cfg!(target_endian = "little") && bytes.as_ptr().align_offset(4) == 0);
            assert_eq!(storage[offset + bits.len() * 4], 173);
        }
    }

    #[test]
    fn direct_decoding_preserves_common_modes_and_depths() {
        for mode in [PixelMode::L, PixelMode::Rgb, PixelMode::Rgba] {
            for depth in [8, 10, 12, 16] {
                let maximum = (1_u32 << depth) - 1;
                let values: Vec<u16> = (0..257 * 259 * mode.channels()).map(|i| ((i as u32 * 719 + 257) & maximum) as u16).collect();
                let image = if depth == 8 {
                    Image::from_pixels(257, 259, mode, values.into_iter().map(|v| v as u8).collect(), None).unwrap()
                } else {
                    Image::from_samples(257, 259, mode, values, depth, None).unwrap()
                };
                for compression in [0, 1, 2] {
                    let encoded = if compression != 0 {
                        encode_compressed_tiff(&image, SaveOptions { tiff_compression: compression, ..Default::default() }).unwrap()
                    } else if depth == 8 {
                        encode_tiff(&image, image.raw_data().unwrap()).unwrap()
                    } else {
                        encode_wide_tiff(&image).unwrap()
                    };
                    let reference = decode_rust_image(&encoded, RustFormat::Tiff, "TIFF").unwrap();
                    let decoded = decode_tiff(&encoded).unwrap();
                    assert_eq!(decoded.bit_depth, reference.bit_depth);
                    assert_eq!(decoded.raw_data().unwrap(), reference.raw_data().unwrap());
                }
            }
        }
    }

    #[test]
    fn lzw_strip_decoding_matches_serial_with_partial_last_strip() {
        fn check<C: tiff::encoder::colortype::ColorType<Inner = u8>>(mode: PixelMode, rows_per_strip: u32) {
            let (width, height) = (513, 517);
            let pixels: Vec<_> = (0..width * height * mode.channels()).map(|i| (i * 37 + i / 7) as u8).collect();
            let mut data = Cursor::new(Vec::new());
            {
                let mut encoder = tiff::encoder::TiffEncoder::new(&mut data).unwrap().with_compression(tiff::encoder::Compression::Lzw);
                let mut image = encoder.new_image::<C>(width as u32, height as u32).unwrap();
                image.rows_per_strip(rows_per_strip).unwrap();
                image.write_data(&pixels).unwrap();
            }
            for threads in [1, 4] {
                rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap().install(|| {
                    let decoded = decode_tiff(data.get_ref()).unwrap();
                    assert_eq!(decoded.mode, mode);
                    assert_eq!(decoded.pixel_data().unwrap(), pixels);
                    let mut decoder = tiff::decoder::Decoder::new(Cursor::new(data.get_ref().as_slice())).unwrap();
                    assert_eq!(decode_lzw_strips(data.get_ref(), &mut decoder).unwrap().is_some(), threads > 1);
                });
            }
        }
        for rows in [1, 17, 128] {
            check::<tiff::encoder::colortype::Gray8>(PixelMode::L, rows);
            check::<tiff::encoder::colortype::RGB8>(PixelMode::Rgb, rows);
            check::<tiff::encoder::colortype::RGBA8>(PixelMode::Rgba, rows);
        }
    }

    #[test]
    fn lab_tiff_directory_bounds_and_byte_order() {
        for little in [false, true] {
            for big in [false, true] {
                let mut data = if little { b"II".to_vec() } else { b"MM".to_vec() };
                let push = |data: &mut Vec<u8>, value: u64, count: usize| {
                    if little {
                        data.extend_from_slice(&value.to_le_bytes()[..count]);
                    } else {
                        data.extend_from_slice(&value.to_be_bytes()[8 - count..]);
                    }
                };
                push(&mut data, if big { 43 } else { 42 }, 2);
                if big {
                    push(&mut data, 8, 2);
                    push(&mut data, 0, 2);
                }
                push(&mut data, if big { 16 } else { 8 }, if big { 8 } else { 4 });
                push(&mut data, 1, if big { 8 } else { 2 });
                push(&mut data, 262, 2);
                push(&mut data, 3, 2);
                push(&mut data, 1, if big { 8 } else { 4 });
                let value_offset = data.len();
                push(&mut data, 8, 2);
                push(&mut data, 0, if big { 6 } else { 2 });
                let mut expected = data.clone();
                expected[value_offset..value_offset + 2].copy_from_slice(&if little { 2_u16.to_le_bytes() } else { 2_u16.to_be_bytes() });
                assert_eq!(lab_tiff_as_rgb(&data).unwrap(), expected);
                for length in 0..data.len() {
                    assert!(lab_tiff_as_rgb(&data[..length]).is_err());
                }
            }
        }
    }
}
