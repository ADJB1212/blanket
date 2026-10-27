use std::borrow::Cow;
use std::io::Write;
use std::simd::{Simd, simd_swizzle};

use blanket_codec_common::codec_error;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

pub type PngChunk = (Vec<u8>, Vec<u8>, bool);
type TiffEntry = (u16, u16, u32, usize);

#[pyfunction]
pub fn _read_native_metadata<'py>(py: Python<'py>, data: &[u8], format: &str) -> PyResult<Bound<'py, PyDict>> {
    let values = py.detach(|| match format {
        "HEIF" => blanket_codec_heif::read_metadata(data),
        "JXL" => blanket_codec_jxl::read_metadata(data),
        _ => Ok(Vec::new()),
    })?;
    let result = PyDict::new(py);
    for (key, value) in values {
        if !value.is_empty() {
            result.set_item(key, PyBytes::new(py, &value))?;
        }
    }
    Ok(result)
}

#[pyfunction]
pub fn _write_metadata<'py>(
    py: Python<'py>, data: &[u8], format: &str, exif: &[u8], xmp: &[u8], icc_profile: &[u8], chunks: Vec<PngChunk>,
) -> PyResult<Bound<'py, PyBytes>> {
    let output = py.detach(|| write_metadata(data, format, exif, xmp, icc_profile, &chunks))?;
    Ok(PyBytes::new(py, &output))
}

fn chunk(output: &mut Vec<u8>, kind: &[u8], payload: &[u8], little: bool) -> PyResult<()> {
    let length = u32::try_from(payload.len()).map_err(|_| PyValueError::new_err("metadata chunk exceeds 4 GiB"))?;
    if little {
        output.extend_from_slice(kind);
        output.extend_from_slice(&length.to_le_bytes());
        output.extend_from_slice(payload);
        if payload.len() & 1 != 0 {
            output.push(0);
        }
    } else {
        output.extend_from_slice(&length.to_be_bytes());
        output.extend_from_slice(kind);
        output.extend_from_slice(payload);
    }
    Ok(())
}

fn png_chunk(output: &mut Vec<u8>, kind: &[u8], payload: &[u8]) -> PyResult<()> {
    chunk(output, kind, payload, false)?;
    let mut crc = crc32fast::Hasher::new();
    crc.update(kind);
    crc.update(payload);
    output.extend_from_slice(&crc.finalize().to_be_bytes());
    Ok(())
}

fn jpeg_segment(output: &mut Vec<u8>, marker: u8, prefix: &[u8], payload: &[u8]) -> PyResult<()> {
    let length = u16::try_from(prefix.len() + payload.len() + 2).map_err(|_| PyValueError::new_err("JPEG metadata segment exceeds 65533 bytes"))?;
    output.extend_from_slice(&[255, marker]);
    output.extend_from_slice(&length.to_be_bytes());
    output.extend_from_slice(prefix);
    output.extend_from_slice(payload);
    Ok(())
}

fn jxl_box(output: &mut Vec<u8>, kind: &[u8], payload: &[u8]) -> PyResult<()> {
    let length = u32::try_from(payload.len() + 8).map_err(codec_error)?;
    output.extend_from_slice(&length.to_be_bytes());
    output.extend_from_slice(kind);
    output.extend_from_slice(payload);
    Ok(())
}

fn write_metadata(data: &[u8], format: &str, exif: &[u8], xmp: &[u8], icc: &[u8], chunks: &[PngChunk]) -> PyResult<Vec<u8>> {
    write_metadata_buffer(Cow::Borrowed(data), format, exif, xmp, icc, chunks)
}

fn insert_metadata(data: Cow<'_, [u8]>, before: usize, after: usize, prefix: &[u8], suffix: &[u8]) -> Vec<u8> {
    let copy = |data: &[u8]| {
        let mut output = Vec::with_capacity(data.len() + prefix.len() + suffix.len());
        output.extend_from_slice(&data[..before]);
        output.extend_from_slice(prefix);
        output.extend_from_slice(&data[before..after]);
        output.extend_from_slice(suffix);
        output.extend_from_slice(&data[after..]);
        output
    };
    match data {
        Cow::Borrowed(data) => copy(data),
        Cow::Owned(mut output) => {
            if !prefix.is_empty() && output.capacity() - output.len() < prefix.len() + suffix.len() {
                return copy(&output);
            }
            output.reserve(prefix.len() + suffix.len());
            if !suffix.is_empty() {
                drop(output.splice(after..after, suffix.iter().copied()));
            }
            if !prefix.is_empty() {
                drop(output.splice(before..before, prefix.iter().copied()));
            }
            output
        }
    }
}

fn write_metadata_buffer(data: Cow<'_, [u8]>, format: &str, exif: &[u8], xmp: &[u8], icc: &[u8], chunks: &[PngChunk]) -> PyResult<Vec<u8>> {
    if exif.is_empty() && xmp.is_empty() && icc.is_empty() && chunks.is_empty() {
        return Ok(data.into_owned());
    }
    if (format == "JPEG" && !data.starts_with(b"\xff\xd8"))
        || (format == "PNG" && (data.len() < 45 || !data.starts_with(b"\x89PNG\r\n\x1a\n")))
        || (format == "WEBP" && (data.len() < 12 || !data.starts_with(b"RIFF") || &data[8..12] != b"WEBP"))
    {
        return Err(codec_error("invalid image container for metadata"));
    }
    let tiff = exif.strip_prefix(b"Exif\0\0").unwrap_or(exif);
    match format {
        "JPEG" => {
            let mut output = Vec::with_capacity(data.len() + exif.len() + xmp.len() + icc.len() + 64 + icc.len().div_ceil(65519) * 18);
            output.extend_from_slice(&data[..2]);
            if !exif.is_empty() {
                jpeg_segment(&mut output, 0xe1, b"Exif\0\0", tiff)?;
            }
            if !xmp.is_empty() {
                jpeg_segment(&mut output, 0xe1, b"http://ns.adobe.com/xap/1.0/\0", xmp)?;
            }
            let count = icc.len().div_ceil(65519);
            if count > 255 {
                return Err(PyValueError::new_err("JPEG ICC profile exceeds 255 segments"));
            }
            let mut prefix = *b"ICC_PROFILE\0\0\0";
            prefix[13] = count as u8;
            for (i, part) in icc.chunks(65519).enumerate() {
                prefix[12] = i as u8 + 1;
                jpeg_segment(&mut output, 0xe2, &prefix, part)?;
            }
            output.extend_from_slice(&data[2..]);
            Ok(output)
        }
        "PNG" => {
            let mut metadata = Vec::new();
            if !tiff.is_empty() {
                png_chunk(&mut metadata, b"eXIf", tiff)?;
            }
            if !icc.is_empty() {
                let mut encoder = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
                encoder.write_all(icc).map_err(codec_error)?;
                png_chunk(&mut metadata, b"iCCP", &[b"ICC Profile\0\0".as_slice(), &encoder.finish().map_err(codec_error)?].concat())?;
            }
            if !xmp.is_empty() {
                png_chunk(&mut metadata, b"iTXt", &[b"XML:com.adobe.xmp\0\0\0\0\0".as_slice(), xmp].concat())?;
            }
            let mut after = Vec::new();
            for (kind, payload, after_idat) in chunks {
                if kind.len() != 4
                    || !kind.iter().all(u8::is_ascii_alphabetic)
                    || !kind[0].is_ascii_lowercase()
                    || !kind[2].is_ascii_uppercase()
                    || matches!(kind.as_slice(), b"acTL" | b"fcTL" | b"fdAT" | b"eXIf" | b"iCCP")
                {
                    return Err(PyValueError::new_err("invalid custom PNG chunk"));
                }
                png_chunk(if *after_idat { &mut after } else { &mut metadata }, kind, payload)?;
            }
            let end = data.len() - 12;
            Ok(insert_metadata(data, 33, end, &metadata, &after))
        }
        "WEBP" => {
            let mut flags = 0;
            if !icc.is_empty() {
                flags |= 0x20;
            }
            if !exif.is_empty() {
                flags |= 0x08;
            }
            if !xmp.is_empty() {
                flags |= 0x04;
            }
            let mut pos = 12;
            let mut extended = None;
            while pos + 8 <= data.len() {
                let kind = &data[pos..pos + 4];
                let length = u32::from_le_bytes(data[pos + 4..pos + 8].try_into().expect("four bytes")) as usize;
                let end =
                    pos.checked_add(8 + length + (length & 1)).filter(|&end| end <= data.len()).ok_or_else(|| codec_error("invalid WebP chunk"))?;
                if kind == b"VP8X" {
                    extended = Some(data[pos + 8..pos + 8 + length].to_vec());
                }
                pos = end;
            }
            let mut header = if let Some(header) = extended {
                header
            } else {
                let (width, height, alpha) = webp_dimensions(&data)?;
                let mut header = vec![if alpha { 0x10 } else { 0 }, 0, 0, 0];
                header.extend_from_slice(&(width - 1).to_le_bytes()[..3]);
                header.extend_from_slice(&(height - 1).to_le_bytes()[..3]);
                header
            };
            if header.len() != 10 {
                return Err(codec_error("invalid WebP extended header"));
            }
            header[0] |= flags;
            let mut output = Vec::with_capacity(data.len() + exif.len() + xmp.len() + icc.len() + 48);
            output.extend_from_slice(b"RIFF\0\0\0\0WEBP");
            chunk(&mut output, b"VP8X", &header, true)?;
            if !icc.is_empty() {
                chunk(&mut output, b"ICCP", icc, true)?;
            }
            pos = 12;
            while pos + 8 <= data.len() {
                let length = u32::from_le_bytes(data[pos + 4..pos + 8].try_into().expect("validated chunk")) as usize;
                let end = pos + 8 + length + (length & 1);
                if &data[pos..pos + 4] != b"VP8X" {
                    output.extend_from_slice(&data[pos..end]);
                }
                pos = end;
            }
            if !exif.is_empty() {
                chunk(&mut output, b"EXIF", exif, true)?;
            }
            if !xmp.is_empty() {
                chunk(&mut output, b"XMP ", xmp, true)?;
            }
            let size = u32::try_from(output.len() - 8).map_err(codec_error)?;
            output[4..8].copy_from_slice(&size.to_le_bytes());
            Ok(output)
        }
        "JXL" => {
            let container = data.starts_with(b"\0\0\0\x0cJXL \r\n\x87\n") && data.len() >= 32;
            let mut output = Vec::with_capacity(data.len() + exif.len() + xmp.len() + 60);
            output.extend_from_slice(if container { &data[..32] } else { b"\0\0\0\x0cJXL \r\n\x87\n" });
            if !container {
                jxl_box(&mut output, b"ftyp", b"jxl \0\0\0\0jxl ")?;
            }
            if !exif.is_empty() {
                jxl_box(&mut output, b"Exif", &[&[0; 4], tiff].concat())?;
            }
            if !xmp.is_empty() {
                jxl_box(&mut output, b"xml ", xmp)?;
            }
            if container {
                output.extend_from_slice(&data[32..]);
            } else {
                jxl_box(&mut output, b"jxlc", &data)?;
            }
            Ok(output)
        }
        "HEIF" => Ok(data.into_owned()),
        "TIFF" => write_tiff(data, tiff, xmp, icc),
        _ => Err(PyValueError::new_err(format!("metadata saving is not supported for {format}"))),
    }
}

pub fn encode_metadata(data: Vec<u8>, format: &str, exif: &[u8], xmp: &[u8], icc: &[u8], chunks: &[PngChunk]) -> PyResult<Vec<u8>> {
    if format == "HEIF" || (exif.is_empty() && xmp.is_empty() && icc.is_empty() && chunks.is_empty()) {
        return Ok(data);
    }
    write_metadata_buffer(Cow::Owned(data), format, exif, xmp, icc, chunks)
}

fn webp_dimensions(data: &[u8]) -> PyResult<(u32, u32, bool)> {
    if data.get(12..16) == Some(b"VP8L") && data.len() >= 25 {
        let bits = u32::from_le_bytes(data[21..25].try_into().expect("four bytes"));
        Ok(((bits & 0x3fff) + 1, ((bits >> 14) & 0x3fff) + 1, bits & (1 << 28) != 0))
    } else if data.get(12..16) == Some(b"VP8 ") && data.len() >= 30 {
        Ok((
            u16::from_le_bytes(data[26..28].try_into().expect("two bytes")) as u32 & 0x3fff,
            u16::from_le_bytes(data[28..30].try_into().expect("two bytes")) as u32 & 0x3fff,
            false,
        ))
    } else {
        Err(codec_error("missing WebP dimensions"))
    }
}

struct Tiff<'a> {
    data: &'a [u8],
    little: bool,
}

impl<'a> Tiff<'a> {
    fn new(data: &'a [u8]) -> PyResult<Self> {
        let this = Self { data, little: data.starts_with(b"II") };
        if !(data.starts_with(b"II") || data.starts_with(b"MM")) || this.number(2, 2)? != 42 {
            return Err(codec_error("invalid classic TIFF metadata"));
        }
        Ok(this)
    }
    fn number(&self, pos: usize, size: usize) -> PyResult<u32> {
        let bytes = self.data.get(pos..pos + size).ok_or_else(|| codec_error("truncated TIFF metadata"))?;
        Ok(if self.little { bytes.iter().rev().fold(0, |n, &b| (n << 8) | b as u32) } else { bytes.iter().fold(0, |n, &b| (n << 8) | b as u32) })
    }
    fn entries(&self, offset: usize) -> PyResult<(Vec<TiffEntry>, usize)> {
        let count = self.number(offset, 2)? as usize;
        let mut entries = Vec::with_capacity(count);
        for i in 0..count {
            let pos = offset + 2 + i * 12;
            entries.push((self.number(pos, 2)? as u16, self.number(pos + 2, 2)? as u16, self.number(pos + 4, 4)?, pos + 8));
        }
        Ok((entries, self.number(offset + 2 + count * 12, 4)? as usize))
    }
}

fn append_tiff(output: &mut Vec<u8>, raw: &[u8]) -> PyResult<u32> {
    if output.len() & 1 != 0 {
        output.push(0);
    }
    let offset = u32::try_from(output.len()).map_err(codec_error)?;
    output.extend_from_slice(raw);
    Ok(offset)
}

fn record(tag: u16, kind: u16, count: u32, field: &[u8], little: bool) -> [u8; 12] {
    let mut result = [0; 12];
    result[..2].copy_from_slice(&if little { tag.to_le_bytes() } else { tag.to_be_bytes() });
    result[2..4].copy_from_slice(&if little { kind.to_le_bytes() } else { kind.to_be_bytes() });
    result[4..8].copy_from_slice(&word(count, little));
    result[8..8 + field.len()].copy_from_slice(field);
    result
}

fn word(value: u32, little: bool) -> [u8; 4] {
    if little { value.to_le_bytes() } else { value.to_be_bytes() }
}

fn swap_tiff_bytes<const N: usize>(data: &mut [u8]) {
    let (blocks, tail) = data.as_chunks_mut::<16>();
    for block in blocks {
        let bytes = Simd::<u8, 16>::from_slice(block);
        let swapped = match N {
            2 => simd_swizzle!(bytes, [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14]),
            4 => simd_swizzle!(bytes, [3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12]),
            8 => simd_swizzle!(bytes, [7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8]),
            _ => unreachable!(),
        };
        swapped.copy_to_slice(block);
    }
    for value in tail.as_chunks_mut::<N>().0 {
        value.reverse();
    }
}

fn copy_ifd(source: &Tiff<'_>, offset: usize, output: &mut Vec<u8>, little: bool, depth: usize) -> PyResult<(Vec<[u8; 12]>, u32)> {
    if depth > 16 {
        return Err(codec_error("EXIF IFD nesting exceeds 16"));
    }
    let (entries, next) = source.entries(offset)?;
    let mut result = Vec::with_capacity(entries.len());
    for &(tag, kind, count, pos) in &entries {
        if depth == 0
            && matches!(tag, 256..=266 | 273 | 277..=279 | 284 | 317 | 320 | 322..=325 | 338 | 339 | 347 | 512..=521 | 529..=532 | 700 | 34675)
        {
            continue;
        }
        let unit = match kind {
            1 | 2 | 6 | 7 => 1,
            3 | 8 => 2,
            4 | 9 | 11 | 13 => 4,
            5 | 10 | 12 => 8,
            _ => continue,
        };
        let length = (count as usize).checked_mul(unit).ok_or_else(|| codec_error("EXIF value overflow"))?;
        let start = if length > 4 { source.number(pos, 4)? as usize } else { pos };
        let end = start.checked_add(length).ok_or_else(|| codec_error("EXIF offset overflow"))?;
        let raw = source.data.get(start..end).ok_or_else(|| codec_error("truncated EXIF value"))?;
        if matches!(tag, 34665 | 34853 | 40965) && count == 1 {
            let (nested, following) = copy_ifd(source, source.number(pos, 4)? as usize, output, little, depth + 1)?;
            let pointer = write_ifd(output, nested, following, little)?;
            result.push(record(tag, 4, 1, &word(pointer, little), little));
            continue;
        }
        if tag == 513 && count == 1 {
            let size_pos = entries.iter().find(|entry| entry.0 == 514).ok_or_else(|| codec_error("missing EXIF thumbnail length"))?.3;
            let start = source.number(pos, 4)? as usize;
            let size = source.number(size_pos, 4)? as usize;
            let thumbnail = source.data.get(start..start.saturating_add(size)).ok_or_else(|| codec_error("truncated EXIF thumbnail"))?;
            let pointer = append_tiff(output, thumbnail)?;
            result.push(record(tag, 4, 1, &word(pointer, little), little));
            continue;
        }
        let mut field = [0; 4];
        let value = if length > 4 {
            let pointer = append_tiff(output, raw)?;
            field = word(pointer, little);
            &mut output[pointer as usize..pointer as usize + length]
        } else {
            field[..length].copy_from_slice(raw);
            &mut field[..length]
        };
        if source.little != little {
            match if matches!(kind, 5 | 10) { 4 } else { unit } {
                2 => swap_tiff_bytes::<2>(value),
                4 => swap_tiff_bytes::<4>(value),
                8 => swap_tiff_bytes::<8>(value),
                _ => {}
            }
        }
        result.push(record(tag, kind, count, &field, little));
    }
    let following = if next != 0 {
        let (entries, following) = copy_ifd(source, next, output, little, depth + 1)?;
        write_ifd(output, entries, following, little)?
    } else {
        0
    };
    Ok((result, following))
}

fn write_ifd(output: &mut Vec<u8>, mut entries: Vec<[u8; 12]>, following: u32, little: bool) -> PyResult<u32> {
    entries.sort_by_key(|entry| if little { u16::from_le_bytes([entry[0], entry[1]]) } else { u16::from_be_bytes([entry[0], entry[1]]) });
    let count = u16::try_from(entries.len()).map_err(codec_error)?;
    let offset = append_tiff(output, &[])?;
    output.reserve(2 + entries.len() * 12 + 4);
    output.extend_from_slice(&if little { count.to_le_bytes() } else { count.to_be_bytes() });
    for entry in entries {
        output.extend_from_slice(&entry);
    }
    output.extend_from_slice(&word(following, little));
    Ok(offset)
}

fn write_tiff(data: Cow<'_, [u8]>, exif: &[u8], xmp: &[u8], icc: &[u8]) -> PyResult<Vec<u8>> {
    let source = Tiff::new(&data)?;
    let little = source.little;
    let (base, _) = source.entries(source.number(4, 4)? as usize)?;
    let extra = exif.len() + xmp.len() + icc.len() + 128;
    let mut output = match data {
        Cow::Owned(mut data) => {
            data.reserve(extra);
            data
        }
        Cow::Borrowed(data) => {
            let mut output = Vec::with_capacity(data.len() + extra);
            output.extend_from_slice(data);
            output
        }
    };
    let (mut entries, following) = if exif.is_empty() {
        (Vec::new(), 0)
    } else {
        let metadata = Tiff::new(exif)?;
        copy_ifd(&metadata, metadata.number(4, 4)? as usize, &mut output, little, 0)?
    };
    if following != 0 {
        entries.push(record(330, 4, 1, &word(following, little), little));
    }
    for (tag, _, _, pos) in base {
        entries.retain(|entry| &entry[..2] != if little { tag.to_le_bytes() } else { tag.to_be_bytes() }.as_slice());
        entries.push(output[pos - 8..pos + 4].try_into().expect("validated TIFF entry"));
    }
    for (tag, raw) in [(700u16, xmp), (34675, icc)] {
        if !raw.is_empty() {
            let count = u32::try_from(raw.len()).map_err(codec_error)?;
            let field = if raw.len() > 4 { word(append_tiff(&mut output, raw)?, little).to_vec() } else { raw.to_vec() };
            entries.retain(|entry| &entry[..2] != if little { tag.to_le_bytes() } else { tag.to_be_bytes() }.as_slice());
            entries.push(record(tag, 1, count, &field, little));
        }
    }
    let root = write_ifd(&mut output, entries, 0, little)?;
    output[4..8].copy_from_slice(&word(root, little));
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn owned_metadata_matches_borrowed_output_and_preserves_decoded_pixels() {
        use blanket_core::{Image, PixelMode};

        use crate::codecs::{ImageFormat, SaveOptions, decode, encode};

        let image = Image::from_pixels(17, 19, PixelMode::Rgb, (0..17 * 19 * 3).map(|i| (i * 137 + i / 7) as u8).collect(), None).unwrap();
        let exif = b"Exif\0\0II*\0\x08\0\0\0\0\0\0\0\0\0";
        let xmp = b"<description>_width _height</description>";
        for format in [ImageFormat::Png, ImageFormat::Jpeg, ImageFormat::Jxl, ImageFormat::Tiff, ImageFormat::Webp] {
            let data = encode(&image, format, SaveOptions { effort: 1, quality: 85, ..Default::default() }).unwrap();
            let mut sources = vec![data];
            if format == ImageFormat::Jxl {
                sources.push(write_metadata(&sources[0], "JXL", &[], b"existing metadata", &[], &[]).unwrap());
            }
            for data in sources {
                let expected = write_metadata(&data, format.as_str(), exif, xmp, &[], &[]).unwrap();
                let original = decode(&data, format).unwrap();
                for spare in [0, 4096] {
                    let mut input = Vec::with_capacity(data.len() + spare);
                    input.extend_from_slice(&data);
                    let actual = encode_metadata(input, format.as_str(), exif, xmp, &[], &[]).unwrap();
                    assert_eq!(actual, expected, "{format:?}, spare={spare}");
                    let decoded = decode(&actual, format).unwrap();
                    assert_eq!(decoded.mode, original.mode);
                    assert_eq!(decoded.size(), original.size());
                    assert_eq!(decoded.raw_data().unwrap(), original.raw_data().unwrap());
                }
            }
        }
    }

    #[test]
    fn owned_png_metadata_preserves_chunk_order_crc_and_image_data() {
        use blanket_core::{Image, PixelMode};

        use crate::codecs::{ImageFormat, SaveOptions, decode, encode};

        let image = Image::from_pixels(17, 19, PixelMode::L, vec![137; 17 * 19], None).unwrap();
        let data = encode(&image, ImageFormat::Png, SaveOptions::default()).unwrap();
        for before in [false, true] {
            for after in [false, true] {
                let mut chunks = Vec::new();
                if before {
                    chunks.push((b"tEXt".to_vec(), b"before\0header".to_vec(), false));
                }
                if after {
                    chunks.push((b"tEXt".to_vec(), b"after\0image".to_vec(), true));
                }
                let expected = write_metadata(&data, "PNG", &[], &[], &[], &chunks).unwrap();
                for spare in [0, 4096] {
                    let mut input = Vec::with_capacity(data.len() + spare);
                    input.extend_from_slice(&data);
                    let actual = encode_metadata(input, "PNG", &[], &[], &[], &chunks).unwrap();
                    assert_eq!(actual, expected);
                    assert_eq!(decode(&actual, ImageFormat::Png).unwrap().raw_data().unwrap(), image.raw_data().unwrap());
                }
            }
        }
    }

    #[test]
    fn tiff_byte_swaps_preserve_bits_tails_and_guards() {
        for unit in [2, 4, 8] {
            for offset in 0..16 {
                for count in 0..66 {
                    let length = count * unit;
                    let mut data: Vec<u8> = (0..offset + length + 19).map(|i| (i * 71 + 255) as u8).collect();
                    let mut expected = data.clone();
                    for value in expected[offset..offset + length].chunks_exact_mut(unit) {
                        value.reverse();
                    }
                    let values = &mut data[offset..offset + length];
                    match unit {
                        2 => swap_tiff_bytes::<2>(values),
                        4 => swap_tiff_bytes::<4>(values),
                        8 => swap_tiff_bytes::<8>(values),
                        _ => unreachable!(),
                    }
                    assert_eq!(data, expected, "unit={unit}, offset={offset}, count={count}");
                }
            }
        }
    }

    #[test]
    fn webp_preserves_chunk_bytes_and_validates_later_chunks() {
        let mut data = b"RIFF\0\0\0\0WEBP".to_vec();
        let mut header = [0; 10];
        header[0] = 0x12;
        chunk(&mut data, b"VP8X", &header, true).unwrap();
        let body_start = data.len();
        chunk(&mut data, b"ANIM", &[0; 6], true).unwrap();
        chunk(&mut data, b"ANMF", &vec![17; 65537], true).unwrap();
        chunk(&mut data, b"JUNK", &[42; 3], true).unwrap();
        let output = write_metadata(&data, "WEBP", b"exif", b"xmp", b"icc", &[]).unwrap();
        assert_eq!(output[20], 0x3e);
        assert_eq!(&output[30..34], b"ICCP");
        assert_eq!(&output[42..42 + data.len() - body_start], &data[body_start..]);
        assert_eq!(u32::from_le_bytes(output[4..8].try_into().unwrap()) as usize, output.len() - 8);
        data.extend_from_slice(b"JUNK\xff\xff\xff\xff");
        assert!(write_metadata(&data, "WEBP", &[], b"xmp", &[], &[]).is_err());
    }

    #[test]
    fn metadata_rejects_truncated_containers_and_critical_png_chunks() {
        for format in ["JPEG", "PNG", "WEBP", "TIFF"] {
            assert!(write_metadata(&[], format, b"Exif\0\0", &[], &[], &[]).is_err());
            assert!(encode_metadata(Vec::new(), format, b"Exif\0\0", &[], &[], &[]).is_err());
        }
        let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
        png.resize(45, 0);
        for kind in [b"IDAT", b"acTL", b"eXIf", b"bad!"] {
            let chunks = vec![(kind.to_vec(), Vec::new(), false)];
            assert!(write_metadata(&png, "PNG", &[], &[], &[], &chunks).is_err());
            assert!(encode_metadata(png.clone(), "PNG", &[], &[], &[], &chunks).is_err());
        }
    }

    #[test]
    fn tiff_rejects_cyclic_and_truncated_ifds() {
        let mut tiff = b"II*\0\x08\0\0\0\0\0\0\0\0\0".to_vec();
        tiff[10..14].copy_from_slice(&8u32.to_le_bytes());
        let source = Tiff::new(&tiff).unwrap();
        assert!(copy_ifd(&source, 8, &mut Vec::new(), true, 0).is_err());
        for length in 0..14 {
            assert!(write_tiff(Cow::Borrowed(&tiff[..length]), &[], &[], b"profile").is_err());
        }
    }
}
