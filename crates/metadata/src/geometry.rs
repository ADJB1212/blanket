//! Unapplied TIFF/EXIF crop bounds and primary HEIF clap/pasp properties.

use pyo3::prelude::*;

use super::iptc::jpeg_segments;
use super::tiff::{Tiff, invalid, slice};

type Rect = (f64, f64, f64, f64);
type Geometry = (Option<Rect>, Option<f64>, Option<&'static str>);
type BoxValue<'a> = (&'a [u8], &'a [u8]);

fn boxes(data: &[u8]) -> PyResult<Vec<BoxValue<'_>>> {
    let mut result = Vec::new();
    let mut pos = 0;
    while pos < data.len() {
        let header = slice(data, pos, 8)?;
        let mut length = u64::from(u32::from_be_bytes(header[..4].try_into().expect("four bytes")));
        let kind = &header[4..8];
        let mut header_size = 8;
        if length == 1 {
            length = u64::from_be_bytes(slice(data, pos + 8, 8)?.try_into().expect("eight bytes"));
            header_size = 16;
        } else if length == 0 {
            length = (data.len() - pos) as u64;
        }
        let length = usize::try_from(length).map_err(|_| invalid("geometry box size overflow"))?;
        if length < header_size {
            return Err(invalid("invalid geometry box size"));
        }
        let value = slice(data, pos, length)?;
        result.push((kind, &value[header_size..]));
        pos += length;
    }
    Ok(result)
}

fn integer(data: &[u8], pos: usize, length: usize) -> PyResult<u32> {
    Ok(slice(data, pos, length)?.iter().fold(0_u32, |value, byte| (value << 8) | u32::from(*byte)))
}

fn primary_properties(data: &[u8]) -> PyResult<Vec<BoxValue<'_>>> {
    for (kind, meta) in boxes(data)? {
        if kind != b"meta" {
            continue;
        }
        if slice(meta, 0, 4)? != [0; 4] {
            return Err(invalid("unsupported HEIF meta version"));
        }
        let children = boxes(&meta[4..])?;
        let mut primary = None;
        for (kind, value) in &children {
            if *kind == b"pitm" {
                let version = slice(value, 0, 4)?[0];
                let width = match version {
                    0 => 2,
                    1 => 4,
                    _ => return Err(invalid("unsupported primary item version")),
                };
                if value.len() != 4 + width {
                    return Err(invalid("invalid primary item size"));
                }
                primary = Some(integer(value, 4, width)?);
            }
        }
        let Some(primary) = primary else { return Ok(Vec::new()) };
        let mut result = Vec::new();
        for (kind, value) in children {
            if kind != b"iprp" {
                continue;
            }
            let mut properties = Vec::new();
            let mut indices = Vec::new();
            for (kind, value) in boxes(value)? {
                if kind == b"ipco" {
                    if !properties.is_empty() {
                        return Err(invalid("duplicate HEIF property container"));
                    }
                    properties = boxes(value)?;
                } else if kind == b"ipma" {
                    let header = slice(value, 0, 8)?;
                    let width = match header[0] {
                        0 => 2,
                        1 => 4,
                        _ => return Err(invalid("unsupported property association version")),
                    };
                    let large = header[3] & 1 != 0;
                    let count = integer(value, 4, 4)?;
                    let mut pos = 8;
                    for _ in 0..count {
                        let item = integer(value, pos, width)?;
                        let count = integer(value, pos + width, 1)?;
                        pos += width + 1;
                        for _ in 0..count {
                            let size = if large { 2 } else { 1 };
                            let index = integer(value, pos, size)? & if large { 0x7fff } else { 0x7f };
                            pos += size;
                            if item == primary && index != 0 {
                                indices.push(index as usize);
                            }
                        }
                    }
                    if pos != value.len() {
                        return Err(invalid("invalid property association size"));
                    }
                }
            }
            for index in indices {
                result.push(*properties.get(index - 1).ok_or_else(|| invalid("invalid HEIF property index"))?);
            }
        }
        return Ok(result);
    }
    Ok(Vec::new())
}

fn heif(data: &[u8]) -> PyResult<Geometry> {
    let mut dimensions = None;
    let mut aspect = None;
    let mut aperture = None;
    for (kind, value) in primary_properties(data)? {
        match kind {
            b"ispe" => {
                if value.len() != 12 || value[..4] != [0; 4] {
                    return Err(invalid("invalid HEIF spatial extent"));
                }
                dimensions = Some((f64::from(integer(value, 4, 4)?), f64::from(integer(value, 8, 4)?)));
            }
            b"pasp" => {
                if value.len() != 8 || aspect.is_some() {
                    return Err(invalid("invalid or duplicate HEIF pixel aspect"));
                }
                let x = integer(value, 0, 4)?;
                let y = integer(value, 4, 4)?;
                if x == 0 || y == 0 {
                    return Err(invalid("pixel aspect must be positive"));
                }
                aspect = Some(f64::from(x) / f64::from(y));
            }
            b"clap" => {
                if value.len() != 32 || aperture.is_some() {
                    return Err(invalid("invalid or duplicate HEIF aperture"));
                }
                let ratio = |pos: usize, signed: bool| -> PyResult<f64> {
                    let numerator = integer(value, pos, 4)?;
                    let denominator = integer(value, pos + 4, 4)?;
                    if denominator == 0 {
                        return Err(invalid("zero HEIF aperture denominator"));
                    }
                    let numerator = if signed { f64::from(numerator as i32) } else { f64::from(numerator) };
                    Ok(numerator / f64::from(denominator))
                };
                let width = ratio(0, false)?;
                let height = ratio(8, false)?;
                if width <= 0.0 || height <= 0.0 {
                    return Err(invalid("aperture dimensions must be positive"));
                }
                aperture = Some((width, height, ratio(16, true)?, ratio(24, true)?));
            }
            _ => {}
        }
    }
    let Some((width, height, x, y)) = aperture else { return Ok((None, aspect, None)) };
    let (stored_width, stored_height) = dimensions.ok_or_else(|| invalid("HEIF aperture requires stored dimensions"))?;
    let left = (stored_width - width) / 2.0 + x;
    let top = (stored_height - height) / 2.0 + y;
    Ok((Some((left, top, left + width, top + height)), aspect, Some("heif:clap")))
}

fn crop(tiff: &Tiff<'_>, format: &str) -> PyResult<Geometry> {
    let aspect = tiff_density(tiff).map(|(aspect, _)| aspect);
    let values = |tag: u16, count: usize| -> PyResult<Option<Vec<f64>>> {
        let values = tiff.numbers(tiff.root, tag)?;
        if values.as_ref().is_some_and(|v| v.len() != count || !v.iter().all(|n| n.is_finite())) {
            return Err(invalid("invalid TIFF aperture field"));
        }
        Ok(values)
    };
    let active = values(50829, 4)?;
    if let Some(v) = &active
        && (v[0] < 0.0 || v[1] < 0.0 || v[2] <= v[0] || v[3] <= v[1])
    {
        return Err(invalid("invalid TIFF active area"));
    }
    if let Some(size) = values(50720, 2)? {
        let origin = values(50719, 2)?.unwrap_or_else(|| vec![0.0, 0.0]);
        if origin.iter().any(|v| *v < 0.0) || size.iter().any(|v| *v <= 0.0) {
            return Err(invalid("invalid TIFF default crop"));
        }
        let x = origin[0] + active.as_ref().map_or(0.0, |v| v[1]);
        let y = origin[1] + active.as_ref().map_or(0.0, |v| v[0]);
        return Ok((Some((x, y, x + size[0], y + size[1])), aspect, Some(if format == "TIFF" { "tiff:default_crop" } else { "exif:default_crop" })));
    }
    if let Some(v) = active {
        return Ok((Some((v[1], v[0], v[3], v[2])), aspect, Some(if format == "TIFF" { "tiff:active_area" } else { "exif:active_area" })));
    }
    Ok((None, aspect, None))
}

fn tiff_density(tiff: &Tiff<'_>) -> Option<(f64, bool)> {
    let x = tiff.numbers(tiff.root, 282).ok()??;
    let y = tiff.numbers(tiff.root, 283).ok()??;
    let unit = tiff.numbers(tiff.root, 296).ok()?.unwrap_or_else(|| vec![2.0]);
    let ([x], [y], [unit]) = (x.as_slice(), y.as_slice(), unit.as_slice()) else { return None };
    let aspect = y / x;
    if !x.is_finite() || !y.is_finite() || *x <= 0.0 || *y <= 0.0 || !aspect.is_finite() || aspect <= 0.0 || !matches!(*unit, 1.0 | 2.0 | 3.0) {
        return None;
    }
    Some((aspect, *unit != 1.0))
}

#[pyfunction]
pub fn _get_geometry(py: Python<'_>, data: &[u8], format: &str) -> PyResult<Geometry> {
    py.detach(|| match format {
        "TIFF" => crop(&Tiff::new(data)?, format),
        "JPEG" => {
            let mut result = (None, None, None);
            let mut density = None;
            let mut container = None;
            for segment in jpeg_segments(data)? {
                let raw = &data[segment.payload..segment.end];
                if segment.marker == 0xe1 && raw.starts_with(b"Exif\0\0") {
                    let tiff = Tiff::new(raw)?;
                    result = crop(&tiff, format)?;
                    density = tiff_density(&tiff);
                } else if segment.marker == 0xe0 && raw.starts_with(b"JFIF\0") {
                    let raw = slice(raw, 0, 14)?;
                    let x = u16::from_be_bytes([raw[8], raw[9]]);
                    let y = u16::from_be_bytes([raw[10], raw[11]]);
                    if x != 0 && y != 0 && raw[7] <= 2 {
                        container = Some((f64::from(y) / f64::from(x), raw[7] != 0));
                    }
                }
            }
            result.1 = match (container, density) {
                (Some((aspect, true)), _)
                | (Some((aspect, false)), None | Some((_, false)))
                | (None, Some((aspect, _)))
                | (Some((_, false)), Some((aspect, true))) => Some(aspect),
                (None, None) => None,
            };
            Ok(result)
        }
        "HEIF" | "AVIF" => heif(data),
        _ => Err(invalid("geometry metadata supports JPEG, TIFF, and HEIF/AVIF only")),
    })
}
