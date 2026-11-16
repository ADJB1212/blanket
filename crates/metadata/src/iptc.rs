//! IPTC IIM datasets and lossless JPEG/TIFF record edits.

use pyo3::prelude::*;
use pyo3::types::PyBytes;

use super::tiff::{Tiff, invalid, slice};

type Dataset = (u8, u8, Vec<u8>);
type PyDataset<'py> = (u8, u8, Bound<'py, PyBytes>);

pub(super) struct Segment {
    pub marker: u8,
    pub start: usize,
    pub payload: usize,
    pub end: usize,
}

pub(super) fn jpeg_segments(data: &[u8]) -> PyResult<Vec<Segment>> {
    if !data.starts_with(b"\xff\xd8") {
        return Err(invalid("invalid JPEG header"));
    }
    let mut segments = Vec::new();
    let mut pos = 2;
    while pos < data.len() {
        let start = pos;
        if data[pos] != 255 {
            return Err(invalid("invalid JPEG marker"));
        }
        while data.get(pos) == Some(&255) {
            pos += 1;
        }
        let marker = slice(data, pos, 1)?[0];
        pos += 1;
        if marker == 0xd9 {
            return Ok(segments);
        }
        if marker == 1 || (0xd0..=0xd7).contains(&marker) {
            continue;
        }
        let length = usize::from(u16::from_be_bytes(slice(data, pos, 2)?.try_into().expect("two bytes")));
        if length < 2 {
            return Err(invalid("invalid JPEG segment length"));
        }
        slice(data, pos, length)?;
        let end = pos + length;
        segments.push(Segment { marker, start, payload: pos + 2, end });
        pos = end;
        if marker == 0xda {
            loop {
                pos += memchr::memchr(255, &data[pos..]).ok_or_else(|| invalid("truncated JPEG scan"))?;
                let mut next = pos + 1;
                while data.get(next) == Some(&255) {
                    next += 1;
                }
                let marker = slice(data, next, 1)?[0];
                if marker != 0 && !(0xd0..=0xd7).contains(&marker) {
                    break;
                }
                pos = next + 1;
            }
        }
    }
    Err(invalid("missing JPEG end marker"))
}

fn decode(raw: &[u8]) -> PyResult<Vec<Dataset>> {
    let mut result = Vec::new();
    let mut pos = 0;
    while pos < raw.len() {
        if raw[pos] == 0 && raw[pos..].iter().all(|v| *v == 0) {
            break;
        }
        let header = slice(raw, pos, 5)?;
        if header[0] != 0x1c {
            return Err(invalid("invalid IPTC IIM header"));
        }
        let (record, dataset) = (header[1], header[2]);
        let mut length = usize::from(u16::from_be_bytes([header[3], header[4]]));
        pos += 5;
        if length & 0x8000 != 0 {
            let count = length & 0x7fff;
            if !(1..=4).contains(&count) {
                return Err(invalid("invalid IPTC IIM extended length"));
            }
            length = slice(raw, pos, count)?.iter().fold(0_usize, |value, digit| (value << 8) | usize::from(*digit));
            pos += count;
        }
        let value = slice(raw, pos, length)?.to_vec();
        result.push((record, dataset, value));
        pos += length;
    }
    Ok(result)
}

fn encode(datasets: &[Dataset]) -> PyResult<Vec<u8>> {
    let mut result = Vec::new();
    for (record, dataset, bytes) in datasets {
        let length = u32::try_from(bytes.len()).map_err(|_| invalid("IPTC IIM dataset exceeds 4 GiB"))?;
        result.extend_from_slice(&[0x1c, *record, *dataset]);
        if length < 0x8000 {
            result.extend_from_slice(&(length as u16).to_be_bytes());
        } else {
            result.extend_from_slice(&[0x80, 4]);
            result.extend_from_slice(&length.to_be_bytes());
        }
        result.extend_from_slice(bytes);
    }
    Ok(result)
}

#[derive(Debug)]
struct Resource {
    payload: usize,
    length: usize,
    end: usize,
}

fn resources(raw: &[u8]) -> PyResult<Vec<Resource>> {
    let mut result = Vec::new();
    let mut pos = 0;
    while pos < raw.len() {
        let header = slice(raw, pos, 7)?;
        if &header[..4] != b"8BIM" {
            return Err(invalid("invalid Photoshop image resource"));
        }
        let kind = u16::from_be_bytes([header[4], header[5]]);
        let name_size = usize::from(header[6]) + 1;
        pos += 6 + name_size + (name_size & 1);
        let length = u32::from_be_bytes(slice(raw, pos, 4)?.try_into().expect("four bytes")) as usize;
        let payload = pos + 4;
        slice(raw, payload, length.checked_add(length & 1).ok_or_else(|| invalid("IPTC resource size overflow"))?)?;
        pos = payload + length + (length & 1);
        if kind == 0x0404 {
            result.push(Resource { payload, length, end: pos });
        }
    }
    Ok(result)
}

fn replace_resource(raw: &[u8], resource: &Resource, replacement: &[u8]) -> PyResult<Vec<u8>> {
    let mut output = raw[..resource.payload - 4].to_vec();
    let count = u32::try_from(replacement.len()).map_err(|_| invalid("IPTC record exceeds 4 GiB"))?;
    output.extend_from_slice(&count.to_be_bytes());
    output.extend_from_slice(replacement);
    if replacement.len() & 1 != 0 {
        output.push(0);
    }
    output.extend_from_slice(&raw[resource.end..]);
    Ok(output)
}

fn record(data: &[u8], replacement: Option<&[u8]>) -> PyResult<(Option<Vec<u8>>, Vec<u8>)> {
    if data.starts_with(b"II") || data.starts_with(b"MM") {
        let tiff = Tiff::new(data)?;
        let direct = tiff.find(tiff.root, 33723)?;
        let mut photoshop = None;
        if let Some(pos) = tiff.find(tiff.root, 34377)? {
            let raw = tiff.value(pos)?;
            let mut records = resources(raw)?;
            if records.len() > 1 {
                return Err(invalid("multiple IPTC IIM records are ambiguous"));
            }
            if let Some(resource) = records.pop() {
                photoshop = Some((pos, raw, resource));
            }
        }
        if direct.is_some() && photoshop.is_some() {
            return Err(invalid("multiple IPTC IIM records are ambiguous"));
        }
        let pos = direct.or_else(|| photoshop.as_ref().map(|(pos, _, _)| *pos));
        let Some(pos) = pos else {
            if replacement.is_some() {
                return Err(invalid("container has no existing IPTC IIM record"));
            }
            return Ok((None, Vec::new()));
        };
        if !matches!(tiff.short(pos + 2)?, 1 | 4 | 7) {
            return Err(invalid("unsupported TIFF IPTC field type"));
        }
        let raw = if let Some((_, raw, resource)) = &photoshop {
            raw[resource.payload..resource.payload + resource.length].to_vec()
        } else {
            tiff.value(pos)?.to_vec()
        };
        decode(&raw)?;
        let Some(replacement) = replacement else { return Ok((Some(raw), Vec::new())) };
        let resource_bytes;
        let replacement = if let Some((_, raw, resource)) = &photoshop {
            resource_bytes = replace_resource(raw, resource, replacement)?;
            resource_bytes.as_slice()
        } else {
            replacement
        };
        let mut result = data.to_vec();
        let value = if replacement.len() <= 4 {
            let mut value = [0; 4];
            value[..replacement.len()].copy_from_slice(replacement);
            value
        } else {
            if result.len() & 1 != 0 {
                result.push(0);
            }
            let offset = u32::try_from(result.len()).map_err(|_| invalid("TIFF metadata exceeds 4 GiB"))?;
            result.extend_from_slice(replacement);
            tiff.put_long(offset)
        };
        result[pos + 2..pos + 4].copy_from_slice(&tiff.put_short(1));
        let count = u32::try_from(replacement.len()).map_err(|_| invalid("TIFF IPTC exceeds 4 GiB"))?;
        result[pos + 4..pos + 8].copy_from_slice(&tiff.put_long(count));
        result[pos + 8..pos + 12].copy_from_slice(&value);
        return Ok((Some(raw), result));
    }
    if !data.starts_with(b"\xff\xd8") {
        return Err(invalid("IPTC IIM supports JPEG and classic TIFF only"));
    }
    let prefix = b"Photoshop 3.0\0";
    let mut found = None;
    let mut edit = None;
    for segment in jpeg_segments(data)? {
        let payload = &data[segment.payload..segment.end];
        if segment.marker != 0xed || !payload.starts_with(prefix) {
            continue;
        }
        let raw = &payload[prefix.len()..];
        for resource in resources(raw)? {
            if found.is_some() {
                return Err(invalid("multiple IPTC IIM records are ambiguous"));
            }
            let value = &raw[resource.payload..resource.payload + resource.length];
            decode(value)?;
            found = Some(value.to_vec());
            if let Some(replacement) = replacement {
                let mut payload = prefix.to_vec();
                payload.extend_from_slice(&replace_resource(raw, &resource, replacement)?);
                let length = u16::try_from(payload.len() + 2).map_err(|_| invalid("IPTC exceeds JPEG APP13 limit"))?;
                let mut output = data[..segment.start].to_vec();
                output.extend_from_slice(&[255, 0xed]);
                output.extend_from_slice(&length.to_be_bytes());
                output.extend_from_slice(&payload);
                output.extend_from_slice(&data[segment.end..]);
                edit = Some(output);
            }
        }
    }
    if replacement.is_some() && found.is_none() {
        return Err(invalid("container has no existing IPTC IIM record"));
    }
    Ok((found, edit.unwrap_or_default()))
}

#[pyfunction]
pub fn _get_iptc<'py>(py: Python<'py>, data: &[u8]) -> PyResult<Vec<PyDataset<'py>>> {
    let values = py.detach(|| -> PyResult<Vec<Dataset>> {
        let (raw, _) = record(data, None)?;
        raw.as_deref().map(decode).transpose().map(Option::unwrap_or_default)
    })?;
    Ok(values.into_iter().map(|(record, dataset, value)| (record, dataset, PyBytes::new(py, &value))).collect())
}

#[pyfunction]
pub fn _rewrite_iptc<'py>(py: Python<'py>, data: &[u8], datasets: Vec<Dataset>) -> PyResult<Bound<'py, PyBytes>> {
    let output = py.detach(|| -> PyResult<Vec<u8>> {
        let replacement = encode(&datasets)?;
        Ok(record(data, Some(&replacement))?.1)
    })?;
    Ok(PyBytes::new(py, &output))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_iim_extended_lengths_and_binary_values() {
        for length in [0, 1, 32767, 32768, 65535, 65536] {
            let values = vec![(1, 90, b"\x1b%G".to_vec()), (2, 25, vec![255; length]), (2, 25, b"repeat".to_vec())];
            let bytes = encode(&values).unwrap();
            assert_eq!(decode(&bytes).unwrap(), values);
            if length >= 32768 {
                assert_eq!(&bytes[11..13], &[0x80, 4]);
            }
        }
    }

    #[test]
    fn native_iim_checks_all_truncation_points() {
        let bytes = encode(&[(2, 120, vec![1; 32768])]).unwrap();
        for length in [1, 2, 3, 4, 5, 6, 7, 8, 9, 100, bytes.len() - 1] {
            assert!(decode(&bytes[..length]).is_err());
        }
        for raw in [b"\x1c\x02\x78\x80\x00".as_slice(), b"\x1c\x02\x78\x80\x05", b"\x1c\x02\x78\x80\x04\xff\xff\xff\xff"] {
            assert!(decode(raw).is_err());
        }
    }

    #[test]
    fn native_jpeg_checks_scan_stuffing_and_later_resources() {
        let raw = b"\xff\xd8\xff\xda\x00\x02\x01\xff\x00\xff\xd0\x02\xff\xed\x00\x04ab\xff\xd9";
        let segments = jpeg_segments(raw).unwrap();
        assert_eq!(segments.len(), 2);
        assert_eq!(&raw[segments[1].payload..segments[1].end], b"ab");
        for length in 0..raw.len() {
            assert!(jpeg_segments(&raw[..length]).is_err());
        }
    }
}
