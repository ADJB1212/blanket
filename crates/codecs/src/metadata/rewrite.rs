use blanket_codec_common::codec_error;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use super::{PngChunk, jxl_box, write_metadata};

#[pyfunction]
#[pyo3(signature = (data, format, exif, xmp, icc, text, dpi, remove_resolution))]
#[allow(clippy::too_many_arguments)]
pub fn _rewrite_metadata<'py>(
    py: Python<'py>, data: &[u8], format: &str, exif: Option<Vec<u8>>, xmp: Option<Vec<u8>>, icc: Option<Vec<u8>>, text: Option<Vec<PngChunk>>,
    dpi: Option<(f64, f64)>, remove_resolution: bool,
) -> PyResult<Bound<'py, PyBytes>> {
    let output = py.detach(|| rewrite(data, format, exif.as_deref(), xmp.as_deref(), icc.as_deref(), text.as_deref(), dpi, remove_resolution))?;
    Ok(PyBytes::new(py, &output))
}

fn density(value: f64, scale: f64, maximum: u32) -> PyResult<u32> {
    let value = (value * scale).round();
    if !value.is_finite() || value < 1.0 || value > f64::from(maximum) {
        return Err(PyValueError::new_err("DPI is outside the container's density range"));
    }
    Ok(value as u32)
}

#[allow(clippy::too_many_arguments)]
fn rewrite(
    data: &[u8], format: &str, exif: Option<&[u8]>, xmp: Option<&[u8]>, icc: Option<&[u8]>, text: Option<&[PngChunk]>, dpi: Option<(f64, f64)>,
    remove_resolution: bool,
) -> PyResult<Vec<u8>> {
    let mut chunks = text.unwrap_or_default().to_vec();
    let output = match format {
        "JPEG" => jpeg(data, exif.is_some(), xmp.is_some(), icc.is_some(), text.is_some(), dpi, remove_resolution)?,
        "PNG" => {
            let output = png(data, exif.is_some(), xmp.is_some(), icc.is_some(), text.is_some(), dpi.is_some() || remove_resolution)?;
            if let Some((x, y)) = dpi {
                let mut payload = density(x, 1.0 / 0.0254, u32::MAX)?.to_be_bytes().to_vec();
                payload.extend_from_slice(&density(y, 1.0 / 0.0254, u32::MAX)?.to_be_bytes());
                payload.push(1);
                chunks.push((b"pHYs".to_vec(), payload, false));
            }
            output
        }
        "WEBP" => webp(data, exif.is_some(), xmp.is_some(), icc.is_some())?,
        "JXL" => {
            if icc.is_some() {
                return Err(PyValueError::new_err("JPEG XL ICC profiles are part of the codestream and cannot be rewritten without re-encoding"));
            }
            return jxl(data, exif, xmp);
        }
        _ => return Err(PyValueError::new_err(format!("metadata-only rewriting is not supported for {format}"))),
    };
    write_metadata(&output, format, exif.unwrap_or_default(), xmp.unwrap_or_default(), icc.unwrap_or_default(), &chunks)
}

#[allow(clippy::too_many_arguments)]
fn jpeg(data: &[u8], exif: bool, xmp: bool, icc: bool, text: bool, dpi: Option<(f64, f64)>, remove_resolution: bool) -> PyResult<Vec<u8>> {
    if !data.starts_with(b"\xff\xd8") {
        return Err(codec_error("invalid JPEG container"));
    }
    let mut output = data[..2].to_vec();
    let mut pos = 2;
    let mut has_density = false;
    let mut in_scan = false;
    let mut ended = false;
    while pos < data.len() {
        if in_scan {
            let start = pos;
            loop {
                if pos >= data.len() {
                    return Err(codec_error("truncated JPEG scan"));
                }
                if data[pos] == 255 {
                    let mut next = pos + 1;
                    while data.get(next) == Some(&255) {
                        next += 1;
                    }
                    let marker = *data.get(next).ok_or_else(|| codec_error("truncated JPEG marker"))?;
                    if marker != 0 && !(0xd0..=0xd7).contains(&marker) {
                        break;
                    }
                    pos = next + 1;
                } else {
                    pos += 1;
                }
            }
            output.extend_from_slice(&data[start..pos]);
            in_scan = false;
        }
        let start = pos;
        if data.get(pos) != Some(&255) {
            return Err(codec_error("invalid JPEG marker"));
        }
        while data.get(pos) == Some(&255) {
            pos += 1;
        }
        let marker = *data.get(pos).ok_or_else(|| codec_error("truncated JPEG marker"))?;
        pos += 1;
        if marker == 0xd9 {
            output.extend_from_slice(&data[start..]);
            ended = true;
            break;
        }
        if marker == 1 || (0xd0..=0xd7).contains(&marker) {
            output.extend_from_slice(&data[start..pos]);
            continue;
        }
        let length = data.get(pos..pos + 2).ok_or_else(|| codec_error("truncated JPEG segment"))?;
        let length = usize::from(u16::from_be_bytes(length.try_into().expect("two bytes")));
        let end =
            pos.checked_add(length).filter(|&end| length >= 2 && end <= data.len()).ok_or_else(|| codec_error("invalid JPEG segment length"))?;
        let payload = &data[pos + 2..end];
        let remove = (marker == 0xe1
            && ((exif && payload.starts_with(b"Exif\0\0"))
                || (xmp
                    && (payload.starts_with(b"http://ns.adobe.com/xap/1.0/\0") || payload.starts_with(b"http://ns.adobe.com/xmp/extension/\0")))))
            || (marker == 0xe2 && icc && payload.starts_with(b"ICC_PROFILE\0"))
            || (marker == 0xfe && text);
        if marker == 0xe0 && payload.starts_with(b"JFIF\0") && (dpi.is_some() || remove_resolution) {
            if payload.len() < 14 {
                return Err(codec_error("truncated JFIF density"));
            }
            let mut segment = data[start..end].to_vec();
            let base = pos + 2 - start;
            if let Some((x, y)) = dpi {
                segment[base + 7] = 1;
                segment[base + 8..base + 10].copy_from_slice(&(density(x, 1.0, 65535)? as u16).to_be_bytes());
                segment[base + 10..base + 12].copy_from_slice(&(density(y, 1.0, 65535)? as u16).to_be_bytes());
            } else {
                segment[base + 7..base + 12].copy_from_slice(&[0, 0, 1, 0, 1]);
            }
            output.extend_from_slice(&segment);
            has_density = true;
        } else if !remove {
            output.extend_from_slice(&data[start..end]);
        }
        pos = end;
        in_scan = marker == 0xda;
    }
    if !ended {
        return Err(codec_error("missing JPEG end marker"));
    }
    if let Some((x, y)) = dpi
        && !has_density
    {
        let mut segment = b"\xff\xe0\0\x10JFIF\0\x01\x02\x01".to_vec();
        segment.extend_from_slice(&(density(x, 1.0, 65535)? as u16).to_be_bytes());
        segment.extend_from_slice(&(density(y, 1.0, 65535)? as u16).to_be_bytes());
        segment.extend_from_slice(&[0, 0]);
        output.splice(2..2, segment);
    }
    Ok(output)
}

fn png(data: &[u8], exif: bool, xmp: bool, icc: bool, text: bool, resolution: bool) -> PyResult<Vec<u8>> {
    if !data.starts_with(b"\x89PNG\r\n\x1a\n") {
        return Err(codec_error("invalid PNG container"));
    }
    let mut output = data[..8].to_vec();
    let mut pos = 8;
    let mut first = true;
    while pos < data.len() {
        let header = data.get(pos..pos + 8).ok_or_else(|| codec_error("truncated PNG chunk"))?;
        let length = u32::from_be_bytes(header[..4].try_into().expect("four bytes")) as usize;
        let end = pos
            .checked_add(12)
            .and_then(|p| p.checked_add(length))
            .filter(|&end| end <= data.len())
            .ok_or_else(|| codec_error("truncated PNG chunk"))?;
        let kind = &header[4..];
        if first && (kind != b"IHDR" || length != 13) {
            return Err(codec_error("invalid PNG header"));
        }
        first = false;
        let payload = &data[pos + 8..end - 4];
        let crc = u32::from_be_bytes(data[end - 4..end].try_into().expect("four bytes"));
        if crc32fast::hash(&data[pos + 4..end - 4]) != crc {
            return Err(codec_error("invalid PNG chunk CRC"));
        }
        let is_text = matches!(kind, b"tEXt" | b"zTXt" | b"iTXt");
        let keyword = payload.split(|&b| b == 0).next().unwrap_or_default();
        let is_xmp = is_text && keyword == b"XML:com.adobe.xmp";
        let is_exif_text = is_text && keyword == b"Raw profile type exif";
        let remove = (exif && (kind == b"eXIf" || is_exif_text))
            || (icc && kind == b"iCCP")
            || (xmp && is_xmp)
            || (text && is_text && !is_xmp && !is_exif_text)
            || (resolution && kind == b"pHYs");
        if !remove {
            output.extend_from_slice(&data[pos..end]);
        }
        pos = end;
        if kind == b"IEND" {
            if length != 0 || pos != data.len() {
                return Err(codec_error("invalid PNG end chunk"));
            }
            return Ok(output);
        }
    }
    Err(codec_error("missing PNG end chunk"))
}

fn webp(data: &[u8], exif: bool, xmp: bool, icc: bool) -> PyResult<Vec<u8>> {
    if !data.starts_with(b"RIFF") || data.get(8..12) != Some(b"WEBP") {
        return Err(codec_error("invalid WebP container"));
    }
    let declared = u32::from_le_bytes(data[4..8].try_into().expect("four bytes")) as usize;
    if declared.checked_add(8) != Some(data.len()) {
        return Err(codec_error("invalid WebP RIFF size"));
    }
    let mut output = data[..12].to_vec();
    let mut pos = 12;
    while pos < data.len() {
        let header = data.get(pos..pos + 8).ok_or_else(|| codec_error("truncated WebP chunk"))?;
        let length = u32::from_le_bytes(header[4..].try_into().expect("four bytes")) as usize;
        let end = pos
            .checked_add(8)
            .and_then(|p| p.checked_add(length))
            .and_then(|p| p.checked_add(length & 1))
            .filter(|&end| end <= data.len())
            .ok_or_else(|| codec_error("truncated WebP chunk"))?;
        let kind = &header[..4];
        let remove = (exif && kind == b"EXIF") || (xmp && kind == b"XMP ") || (icc && kind == b"ICCP");
        if !remove {
            let start = output.len();
            output.extend_from_slice(&data[pos..end]);
            if kind == b"VP8X" {
                if length != 10 {
                    return Err(codec_error("invalid WebP extended header"));
                }
                output[start + 8] &= !(if exif { 8 } else { 0 } | if xmp { 4 } else { 0 } | if icc { 32 } else { 0 });
            }
        }
        pos = end;
    }
    let size = u32::try_from(output.len() - 8).map_err(codec_error)?;
    output[4..8].copy_from_slice(&size.to_le_bytes());
    Ok(output)
}

fn jxl(data: &[u8], exif: Option<&[u8]>, xmp: Option<&[u8]>) -> PyResult<Vec<u8>> {
    let signature = b"\0\0\0\x0cJXL \r\n\x87\n";
    let mut output = signature.to_vec();
    let mut metadata = Vec::new();
    if let Some(raw) = exif
        && !raw.is_empty()
    {
        jxl_box(&mut metadata, b"Exif", &[&[0; 4], raw.strip_prefix(b"Exif\0\0").unwrap_or(raw)].concat())?;
    }
    if let Some(raw) = xmp
        && !raw.is_empty()
    {
        jxl_box(&mut metadata, b"xml ", raw)?;
    }
    if data.starts_with(b"\xff\x0a") {
        jxl_box(&mut output, b"ftyp", b"jxl \0\0\0\0jxl ")?;
        output.extend_from_slice(&metadata);
        jxl_box(&mut output, b"jxlc", data)?;
        return Ok(output);
    }
    if !data.starts_with(signature) {
        return Err(codec_error("invalid JPEG XL container"));
    }
    let mut pos = 12;
    let mut first = true;
    while pos < data.len() {
        let header = data.get(pos..pos + 8).ok_or_else(|| codec_error("truncated JPEG XL box"))?;
        let length = u32::from_be_bytes(header[..4].try_into().expect("four bytes"));
        let (length, header_size) = match length {
            0 => ((data.len() - pos) as u64, 8),
            1 => (
                u64::from_be_bytes(data.get(pos + 8..pos + 16).ok_or_else(|| codec_error("truncated JPEG XL box"))?.try_into().expect("eight bytes")),
                16,
            ),
            n => (u64::from(n), 8),
        };
        let length = usize::try_from(length).map_err(codec_error)?;
        let end = pos
            .checked_add(length)
            .filter(|&end| length >= header_size && end <= data.len())
            .ok_or_else(|| codec_error("invalid JPEG XL box length"))?;
        let kind = &header[4..];
        if first && kind != b"ftyp" {
            return Err(codec_error("missing JPEG XL file type box"));
        }
        let payload = &data[pos + header_size..end];
        let actual_kind = if kind == b"brob" { payload.get(..4).ok_or_else(|| codec_error("truncated compressed JPEG XL box"))? } else { kind };
        if !((exif.is_some() && actual_kind == b"Exif") || (xmp.is_some() && actual_kind == b"xml ")) {
            output.extend_from_slice(&data[pos..end]);
        }
        if first {
            output.extend_from_slice(&metadata);
            first = false;
        }
        pos = end;
    }
    if first {
        return Err(codec_error("missing JPEG XL file type box"));
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jpeg_rewrites_markers_between_scans_without_touching_entropy() {
        let first = b"\xff\xda\0\x02abc\xff\0\x12\xff\xd0def";
        let second = b"\xff\xda\0\x02ghi\xff\0\x34\xff\xd9";
        let mut input = b"\xff\xd8".to_vec();
        input.extend_from_slice(first);
        super::super::jpeg_segment(&mut input, 0xe1, b"Exif\0\0", b"private").unwrap();
        input.extend_from_slice(second);
        let output = rewrite(&input, "JPEG", Some(&[]), None, None, None, None, false).unwrap();
        assert_eq!(output, [b"\xff\xd8".as_slice(), first, second].concat());
        assert!(rewrite(&input[..input.len() - 1], "JPEG", None, None, None, None, None, false).is_err());
    }

    #[test]
    fn jxl_filters_compressed_metadata_and_preserves_extended_codestream_boxes() {
        let mut input = b"\0\0\0\x0cJXL \r\n\x87\n".to_vec();
        jxl_box(&mut input, b"ftyp", b"jxl \0\0\0\0jxl ").unwrap();
        jxl_box(&mut input, b"brob", b"Exifopaque compressed payload").unwrap();
        jxl_box(&mut input, b"xml ", b"private").unwrap();
        let mut codestream = b"\0\0\0\x01jxlc".to_vec();
        codestream.extend_from_slice(&21_u64.to_be_bytes());
        codestream.extend_from_slice(b"image");
        input.extend_from_slice(&codestream);
        let output = rewrite(&input, "JXL", Some(&[]), Some(b"public"), None, None, None, false).unwrap();
        assert!(output.ends_with(&codestream));
        assert!(!output.windows(7).any(|v| v == b"private"));
        assert!(!output.windows(4).any(|v| v == b"brob"));
        assert!(output.windows(6).any(|v| v == b"public"));
        assert!(rewrite(&input[..input.len() - 1], "JXL", None, None, None, None, None, false).is_err());
    }
}
