//! Physical density normalization and conflict resolution.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

type RawDensity = (f64, f64, i64, String);
type ResolutionValue = (Option<(f64, f64)>, f64, Option<String>, Option<String>, Vec<String>);

struct Density {
    dpi: Option<(f64, f64)>,
    aspect: f64,
    source: String,
    unit: &'static str,
}

fn normalize((x, y, unit, source): RawDensity) -> Option<Density> {
    if !x.is_finite() || !y.is_finite() || x <= 0.0 || y <= 0.0 {
        return None;
    }
    let (unit, factor) = match (source.as_str(), unit) {
        ("png" | "jfif", 0) | ("exif" | "tiff", 1) => ("relative", None),
        ("png" | "bmp", 1) => ("meter", Some(0.0254)),
        ("jfif", 1) | ("exif" | "tiff", 2) => ("inch", Some(1.0)),
        ("jfif", 2) | ("exif" | "tiff", 3) => ("cm", Some(2.54)),
        _ => return None,
    };
    let dpi = factor.map(|factor| (x * factor, y * factor));
    let aspect = y / x;
    if !aspect.is_finite() || aspect <= 0.0 || dpi.is_some_and(|(x, y)| !x.is_finite() || !y.is_finite() || x <= 0.0 || y <= 0.0) {
        return None;
    }
    Some(Density { dpi, aspect, source, unit })
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= (1e-3 * a).abs().max((1e-3 * b).abs())
}

#[pyfunction]
pub fn _resolve_resolution(data: &[u8], format: &str, mut candidates: Vec<RawDensity>, conflict: &str) -> PyResult<ResolutionValue> {
    if !matches!(conflict, "container" | "exif" | "error") {
        return Err(PyValueError::new_err("conflict must be 'container', 'exif', or 'error'"));
    }
    if let Some(density) = container_density(data, format) {
        candidates.insert(0, density);
    }
    let candidates: Vec<_> = candidates.into_iter().filter_map(normalize).collect();
    let Some(first) = candidates.first() else {
        return Ok((None, 1.0, None, None, Vec::new()));
    };
    let mut physical = candidates.iter().filter(|candidate| candidate.dpi.is_some());
    let selected =
        if conflict == "exif" { physical.next_back().or_else(|| candidates.last()).unwrap_or(first) } else { physical.next().unwrap_or(first) };
    let mut conflicts = Vec::new();
    for candidate in &candidates {
        if std::ptr::eq(candidate, selected) {
            continue;
        }
        let dpi_differs = match (candidate.dpi, selected.dpi) {
            (Some((ax, ay)), Some((bx, by))) => !close(ax, bx) || !close(ay, by),
            _ => false,
        };
        if !close(candidate.aspect, selected.aspect) || dpi_differs {
            conflicts.push(candidate.source.clone());
        }
    }
    if !conflicts.is_empty() && conflict == "error" {
        return Err(PyValueError::new_err("conflicting resolution metadata"));
    }
    Ok((selected.dpi, selected.aspect, Some(selected.source.clone()), Some(selected.unit.into()), conflicts))
}

fn container_density(data: &[u8], format: &str) -> Option<RawDensity> {
    match format {
        "PNG" => {
            let mut remaining = data.get(8..)?;
            while remaining.len() >= 12 {
                let length = u32::from_be_bytes(remaining[..4].try_into().ok()?) as usize;
                let end = length.checked_add(12)?;
                let chunk = remaining.get(..end)?;
                if &chunk[4..8] == b"IEND" {
                    break;
                }
                if &chunk[4..8] == b"pHYs" && length == 9 {
                    let x = u32::from_be_bytes(chunk[8..12].try_into().ok()?);
                    let y = u32::from_be_bytes(chunk[12..16].try_into().ok()?);
                    return Some((f64::from(x), f64::from(y), i64::from(chunk[16]), "png".into()));
                }
                remaining = &remaining[end..];
            }
        }
        "JPEG" => {
            let mut remaining = data.get(2..)?;
            while remaining.len() >= 4 && remaining[0] == 255 {
                while remaining.first() == Some(&255) {
                    remaining = &remaining[1..];
                }
                let (&marker, rest) = remaining.split_first()?;
                remaining = rest;
                if matches!(marker, 0xda | 0xd9) {
                    break;
                }
                if marker == 1 || (0xd0..=0xd8).contains(&marker) {
                    continue;
                }
                let length = usize::from(u16::from_be_bytes(remaining.get(..2)?.try_into().ok()?));
                if length < 2 {
                    break;
                }
                let segment = remaining.get(..length)?;
                if marker == 0xe0 && length >= 16 && &segment[2..7] == b"JFIF\0" {
                    let x = u16::from_be_bytes(segment[10..12].try_into().ok()?);
                    let y = u16::from_be_bytes(segment[12..14].try_into().ok()?);
                    return Some((f64::from(x), f64::from(y), i64::from(segment[9]), "jfif".into()));
                }
                remaining = &remaining[length..];
            }
        }
        "BMP" if data.len() >= 46 && u32::from_le_bytes(data[14..18].try_into().ok()?) >= 40 => {
            let x = i32::from_le_bytes(data[38..42].try_into().ok()?);
            let y = i32::from_le_bytes(data[42..46].try_into().ok()?);
            return Some((f64::from(x), f64::from(y), 1, "bmp".into()));
        }
        _ => {}
    }
    None
}
