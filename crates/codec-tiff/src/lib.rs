use std::borrow::Cow;
use std::io::Cursor;

use blanket_codec_common::{SaveOptions, codec_error, u16_bytes, validate_dimensions};
use blanket_codec_image::{color_type, decode_rust_image};
use blanket_core::{Image, PixelMode};
use image::{ExtendedColorType, ImageEncoder, ImageFormat as RustFormat};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

fn decode_special_tiff(data: &[u8]) -> Result<Option<Image>, String> {
    let Ok(mut decoder) = tiff::decoder::Decoder::new(Cursor::new(data)) else {
        return Ok(None);
    };
    let photometric = decoder.get_tag_u32(tiff::tags::Tag::PhotometricInterpretation).ok();
    if decoder.get_tag_u32(tiff::tags::Tag::Compression).ok() == Some(7) {
        return decode_jpeg_tiff(data, &mut decoder).map(Some);
    }
    if matches!(photometric, Some(8 | 9)) {
        let signed = photometric == Some(8);
        // Decode LAB as raw three-channel samples; the TIFF crate cannot read LAB buffers.
        let normalized = lab_tiff_as_rgb(data)?;
        decoder = tiff::decoder::Decoder::new(Cursor::new(normalized.as_slice())).map_err(|error| error.to_string())?;
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
        && let Some(image) = decode_lzw_strips(data, &mut decoder)?
    {
        return Ok(Some(image));
    }
    if decoder.colortype().ok() != Some(tiff::ColorType::Gray(32)) {
        return Ok(None);
    }
    let (width, height) = decoder.dimensions().map_err(|error| error.to_string())?;
    validate_dimensions(width, height)?;
    let tiff::decoder::DecodingResult::F32(values) = decoder.read_image().map_err(|error| error.to_string())? else {
        return Ok(None);
    };
    let mut image =
        Image::from_float_bytes(width, height, values.into_iter().flat_map(f32::to_le_bytes).collect()).map_err(|error| error.to_string())?;
    image.format = Some("TIFF".to_owned());
    Ok(Some(image))
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
    if image.mode == PixelMode::Rgba {
        entries.push((338, 3, 1, 2));
    }
    if image.mode == PixelMode::F {
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
    decode_special_tiff(data)?.map_or_else(|| decode_rust_image(data, RustFormat::Tiff, "TIFF"), Ok)
}

pub fn decode_sequence(data: &[u8]) -> Result<blanket_codec_common::Sequence, String> {
    use blanket_codec_common::{Frame, Sequence, validate_frame_budget};
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
    let mut seen = std::collections::HashSet::new();
    let mut page_data = Cow::Borrowed(data);
    let mut sequence = Sequence::default();
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
        let image = decode_tiff(&page_data)?;
        sequence.frames.push(Frame { image, duration: None, disposal: 0, blend: 0 });
        offset = next_offset;
    }
    if sequence.frames.is_empty() {
        return Err("TIFF contains no pages".into());
    }
    Ok(sequence)
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
    if cfg!(target_endian = "little") {
        return encode_raw_tiff(image, image.raw_data()?);
    }
    let samples: Vec<f32> = image.raw_data()?.as_chunks::<4>().0.iter().map(|bytes| f32::from_le_bytes(*bytes)).collect();
    let mut output = Cursor::new(Vec::new());
    tiff::encoder::TiffEncoder::new(&mut output)
        .map_err(codec_error)?
        .write_image::<tiff::encoder::colortype::Gray32Float>(image.width, image.height, &samples)
        .map_err(codec_error)?;
    return Ok(output.into_inner());
}

pub fn encode_tiff(image: &Image, pixels: &[u8]) -> PyResult<Vec<u8>> {
    if matches!(image.mode, PixelMode::L | PixelMode::Rgb | PixelMode::Rgba) {
        return encode_raw_tiff(image, pixels);
    }
    let mut output = Cursor::new(Vec::with_capacity(pixels.len().saturating_add(1024)));
    image::codecs::tiff::TiffEncoder::new(&mut output).write_image(pixels, image.width, image.height, color_type(image.mode)).map_err(codec_error)?;
    Ok(output.into_inner())
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
            return Err(PyValueError::new_err("JPEG TIFF requires 8-bit L or RGB"));
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
        if image.mode == PixelMode::F {
            let values = image.raw_data()?.as_chunks::<4>().0.iter().map(|v| f32::from_le_bytes(*v)).collect::<Vec<_>>();
            encoder.write_image::<colortype::Gray32Float>(image.width, image.height, &values).map_err(codec_error)?;
        } else if image.bit_depth > 8 {
            let values = blanket_core::simd::normalize_u16(image.raw_data()?, image.bit_depth);
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
