use std::io::Write;

use blanket_codec_common::codec_error;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

type PngChunk = (Vec<u8>, Vec<u8>, bool);
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

fn jpeg_segment(output: &mut Vec<u8>, marker: u8, payload: &[u8]) -> PyResult<()> {
    let length = u16::try_from(payload.len() + 2).map_err(|_| PyValueError::new_err("JPEG metadata segment exceeds 65533 bytes"))?;
    output.extend_from_slice(&[255, marker]);
    output.extend_from_slice(&length.to_be_bytes());
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
    if exif.is_empty() && xmp.is_empty() && icc.is_empty() && chunks.is_empty() {
        return Ok(data.to_vec());
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
            let mut output = data[..2].to_vec();
            if !exif.is_empty() {
                jpeg_segment(&mut output, 0xe1, &[b"Exif\0\0".as_slice(), tiff].concat())?;
            }
            if !xmp.is_empty() {
                jpeg_segment(&mut output, 0xe1, &[b"http://ns.adobe.com/xap/1.0/\0".as_slice(), xmp].concat())?;
            }
            let count = icc.len().div_ceil(65519);
            if count > 255 {
                return Err(PyValueError::new_err("JPEG ICC profile exceeds 255 segments"));
            }
            for (i, part) in icc.chunks(65519).enumerate() {
                jpeg_segment(
                    &mut output,
                    0xe2,
                    &[b"ICC_PROFILE\0".as_slice(), &[i as u8 + 1, count as u8], part].concat(),
                )?;
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
                png_chunk(
                    &mut metadata,
                    b"iCCP",
                    &[b"ICC Profile\0\0".as_slice(), &encoder.finish().map_err(codec_error)?].concat(),
                )?;
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
            let mut output = data[..33].to_vec();
            output.extend_from_slice(&metadata);
            output.extend_from_slice(&data[33..data.len() - 12]);
            output.extend_from_slice(&after);
            output.extend_from_slice(&data[data.len() - 12..]);
            Ok(output)
        }
        "WEBP" => {
            let mut body = Vec::new();
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
                let end = pos
                    .checked_add(8 + length + (length & 1))
                    .filter(|&end| end <= data.len())
                    .ok_or_else(|| codec_error("invalid WebP chunk"))?;
                if kind == b"VP8X" {
                    extended = Some(data[pos + 8..pos + 8 + length].to_vec());
                } else {
                    body.extend_from_slice(&data[pos..end]);
                }
                pos = end;
            }
            let mut header = if let Some(header) = extended {
                header
            } else {
                let (width, height, alpha) = webp_dimensions(data)?;
                let mut header = vec![if alpha { 0x10 } else { 0 }, 0, 0, 0];
                header.extend_from_slice(&(width - 1).to_le_bytes()[..3]);
                header.extend_from_slice(&(height - 1).to_le_bytes()[..3]);
                header
            };
            if header.len() != 10 {
                return Err(codec_error("invalid WebP extended header"));
            }
            header[0] |= flags;
            let mut output = b"RIFF\0\0\0\0WEBP".to_vec();
            chunk(&mut output, b"VP8X", &header, true)?;
            if !icc.is_empty() {
                chunk(&mut output, b"ICCP", icc, true)?;
            }
            output.extend_from_slice(&body);
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
            let mut output = if container {
                data[..32].to_vec()
            } else {
                b"\0\0\0\x0cJXL \r\n\x87\n".to_vec()
            };
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
                jxl_box(&mut output, b"jxlc", data)?;
            }
            Ok(output)
        }
        "HEIF" => Ok(data.to_vec()),
        "TIFF" => write_tiff(data, tiff, xmp, icc),
        _ => Err(PyValueError::new_err(format!("metadata saving is not supported for {format}"))),
    }
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
        let this = Self {
            data,
            little: data.starts_with(b"II"),
        };
        if !(data.starts_with(b"II") || data.starts_with(b"MM")) || this.number(2, 2)? != 42 {
            return Err(codec_error("invalid classic TIFF metadata"));
        }
        Ok(this)
    }
    fn number(&self, pos: usize, size: usize) -> PyResult<u32> {
        let bytes = self.data.get(pos..pos + size).ok_or_else(|| codec_error("truncated TIFF metadata"))?;
        Ok(if self.little {
            bytes.iter().rev().fold(0, |n, &b| (n << 8) | b as u32)
        } else {
            bytes.iter().fold(0, |n, &b| (n << 8) | b as u32)
        })
    }
    fn entries(&self, offset: usize) -> PyResult<(Vec<TiffEntry>, usize)> {
        let count = self.number(offset, 2)? as usize;
        let mut entries = Vec::with_capacity(count);
        for i in 0..count {
            let pos = offset + 2 + i * 12;
            entries.push((
                self.number(pos, 2)? as u16,
                self.number(pos + 2, 2)? as u16,
                self.number(pos + 4, 4)?,
                pos + 8,
            ));
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

fn record(tag: u16, kind: u16, count: u32, field: &[u8], little: bool) -> Vec<u8> {
    let mut result = Vec::with_capacity(12);
    result.extend_from_slice(&if little { tag.to_le_bytes() } else { tag.to_be_bytes() });
    result.extend_from_slice(&if little { kind.to_le_bytes() } else { kind.to_be_bytes() });
    result.extend_from_slice(&if little { count.to_le_bytes() } else { count.to_be_bytes() });
    result.extend_from_slice(field);
    result.resize(12, 0);
    result
}

fn word(value: u32, little: bool) -> [u8; 4] {
    if little { value.to_le_bytes() } else { value.to_be_bytes() }
}

fn copy_ifd(source: &Tiff<'_>, offset: usize, output: &mut Vec<u8>, little: bool, depth: usize) -> PyResult<(Vec<Vec<u8>>, u32)> {
    if depth > 16 {
        return Err(codec_error("EXIF IFD nesting exceeds 16"));
    }
    let (entries, next) = source.entries(offset)?;
    let mut result = Vec::new();
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
        let mut raw = source.data.get(start..end).ok_or_else(|| codec_error("truncated EXIF value"))?.to_vec();
        if matches!(tag, 34665 | 34853 | 40965) && count == 1 {
            let (nested, following) = copy_ifd(source, source.number(pos, 4)? as usize, output, little, depth + 1)?;
            let pointer = write_ifd(output, nested, following, little)?;
            result.push(record(tag, 4, 1, &word(pointer, little), little));
            continue;
        }
        if tag == 513 && count == 1 {
            let size_pos = entries
                .iter()
                .find(|entry| entry.0 == 514)
                .ok_or_else(|| codec_error("missing EXIF thumbnail length"))?
                .3;
            let start = source.number(pos, 4)? as usize;
            let size = source.number(size_pos, 4)? as usize;
            let thumbnail = source
                .data
                .get(start..start.saturating_add(size))
                .ok_or_else(|| codec_error("truncated EXIF thumbnail"))?;
            let pointer = append_tiff(output, thumbnail)?;
            result.push(record(tag, 4, 1, &word(pointer, little), little));
            continue;
        }
        if source.little != little && unit > 1 {
            for value in raw.chunks_exact_mut(if matches!(kind, 5 | 10) { 4 } else { unit }) {
                value.reverse();
            }
        }
        let field = if length > 4 {
            word(append_tiff(output, &raw)?, little).to_vec()
        } else {
            raw
        };
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

fn write_ifd(output: &mut Vec<u8>, mut entries: Vec<Vec<u8>>, following: u32, little: bool) -> PyResult<u32> {
    entries.sort_by_key(|entry| {
        if little {
            u16::from_le_bytes([entry[0], entry[1]])
        } else {
            u16::from_be_bytes([entry[0], entry[1]])
        }
    });
    let count = u16::try_from(entries.len()).map_err(codec_error)?;
    let mut raw = if little { count.to_le_bytes() } else { count.to_be_bytes() }.to_vec();
    for entry in entries {
        raw.extend_from_slice(&entry);
    }
    raw.extend_from_slice(&word(following, little));
    append_tiff(output, &raw)
}

fn write_tiff(data: &[u8], exif: &[u8], xmp: &[u8], icc: &[u8]) -> PyResult<Vec<u8>> {
    let source = Tiff::new(data)?;
    let little = source.little;
    let (base, _) = source.entries(source.number(4, 4)? as usize)?;
    let mut output = data.to_vec();
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
        entries.push(data[pos - 8..pos + 4].to_vec());
    }
    for (tag, raw) in [(700u16, xmp), (34675, icc)] {
        if !raw.is_empty() {
            let count = u32::try_from(raw.len()).map_err(codec_error)?;
            let field = if raw.len() > 4 {
                word(append_tiff(&mut output, raw)?, little).to_vec()
            } else {
                raw.to_vec()
            };
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
    fn metadata_rejects_truncated_containers_and_critical_png_chunks() {
        for format in ["JPEG", "PNG", "WEBP", "TIFF"] {
            assert!(write_metadata(&[], format, b"Exif\0\0", &[], &[], &[]).is_err());
        }
        let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
        png.resize(45, 0);
        for kind in [b"IDAT", b"acTL", b"eXIf", b"bad!"] {
            let chunks = vec![(kind.to_vec(), Vec::new(), false)];
            assert!(write_metadata(&png, "PNG", &[], &[], &[], &chunks).is_err());
        }
    }

    #[test]
    fn tiff_rejects_cyclic_and_truncated_ifds() {
        let mut tiff = b"II*\0\x08\0\0\0\0\0\0\0\0\0".to_vec();
        tiff[10..14].copy_from_slice(&8u32.to_le_bytes());
        let source = Tiff::new(&tiff).unwrap();
        assert!(copy_ifd(&source, 8, &mut Vec::new(), true, 0).is_err());
        for length in 0..14 {
            assert!(write_tiff(&tiff[..length], &[], &[], b"profile").is_err());
        }
    }
}
