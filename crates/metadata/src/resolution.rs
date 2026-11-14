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
pub fn _resolve_resolution(candidates: Vec<RawDensity>, conflict: &str) -> PyResult<ResolutionValue> {
    if !matches!(conflict, "container" | "exif" | "error") {
        return Err(PyValueError::new_err("conflict must be 'container', 'exif', or 'error'"));
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
