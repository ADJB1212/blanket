//! Geometry calculations for the public image API.

use pyo3::exceptions::{PyOverflowError, PyValueError};
use pyo3::prelude::*;

type BoxF = (f64, f64, f64, f64);
type Rotation = ((u32, u32), (f64, f64, f64, f64, f64, f64));

pub(super) struct Reduction {
    pub factor: (u32, u32),
    pub safe: (u32, u32, u32, u32),
    pub bounds: BoxF,
}

pub(super) fn reduction(source: (u32, u32), size: (u32, u32), bounds: BoxF, method: u8, gap: f64) -> Option<Reduction> {
    let fx = (((bounds.2 - bounds.0) / f64::from(size.0) / gap) as u32).max(1);
    let fy = (((bounds.3 - bounds.1) / f64::from(size.1) / gap) as u32).max(1);
    if fx == 1 && fy == 1 {
        return None;
    }
    let support = match method {
        1 => 3.0,
        3 => 2.0,
        4 => 0.5,
        6 => 2.5,
        7 => 4.5,
        _ => 1.0,
    } - 0.5;
    let sx = support * (bounds.2 - bounds.0) / f64::from(size.0);
    let sy = support * (bounds.3 - bounds.1) / f64::from(size.1);
    let safe = (
        (bounds.0 - sx).max(0.0) as u32,
        (bounds.1 - sy).max(0.0) as u32,
        (bounds.2 + sx).ceil().min(f64::from(source.0)) as u32,
        (bounds.3 + sy).ceil().min(f64::from(source.1)) as u32,
    );
    Some(Reduction {
        factor: (fx, fy),
        safe,
        bounds: (
            (bounds.0 - f64::from(safe.0)) / f64::from(fx),
            (bounds.1 - f64::from(safe.1)) / f64::from(fy),
            (bounds.2 - f64::from(safe.0)) / f64::from(fx),
            (bounds.3 - f64::from(safe.1)) / f64::from(fy),
        ),
    })
}

#[pyfunction]
pub(super) fn ops_thumbnail_size(source: (u32, u32), target: (f64, f64)) -> PyResult<Option<(u32, u32)>> {
    if target.0.is_nan() || target.1.is_nan() {
        return Err(PyValueError::new_err("cannot convert float NaN to integer"));
    }
    if !target.0.is_finite() || !target.1.is_finite() {
        return Err(PyOverflowError::new_err("cannot convert float infinity to integer"));
    }
    let (mut x, mut y) = (target.0.floor(), target.1.floor());
    if x <= 0.0 || y <= 0.0 {
        return Err(PyValueError::new_err("height and width must be > 0"));
    }
    let (width, height) = (f64::from(source.0), f64::from(source.1));
    if (x >= width && y >= height) || width == 0.0 || height == 0.0 {
        return Ok(None);
    }
    let aspect = width / height;
    if x / y >= aspect {
        let value = y * aspect;
        let (low, high) = (value.floor(), value.ceil());
        x = if (aspect - low / y).abs() <= (aspect - high / y).abs() { low } else { high }.max(1.0);
    } else {
        let value = x / aspect;
        let (low, high) = (value.floor(), value.ceil());
        let error = |n: f64| if n == 0.0 { 0.0 } else { (aspect - x / n).abs() };
        y = if error(low) <= error(high) { low } else { high }.max(1.0);
    }
    Ok(Some((x as u32, y as u32)))
}

#[pyfunction]
pub(super) fn ops_rotation_geometry(
    size: (u32, u32), angle: f64, expand: bool, center: Option<(f64, f64)>, translate: Option<(f64, f64)>,
) -> PyResult<Rotation> {
    let (mut w, mut h) = (f64::from(size.0), f64::from(size.1));
    let (cx, cy) = center.unwrap_or((w / 2.0, h / 2.0));
    let (tx, ty) = translate.unwrap_or((0.0, 0.0));
    if ![cx, cy, tx, ty, angle].iter().all(|v| v.is_finite()) {
        return Err(PyValueError::new_err("center and translate must contain finite coordinates"));
    }
    let radians = -angle.to_radians();
    let round = |v: f64| format!("{v:.15}").parse::<f64>().expect("rounded float");
    let (a, b) = (round(radians.cos()), round(radians.sin()));
    let (d, e) = (-b, a);
    let mut c = a * (-cx - tx) + b * (-cy - ty) + cx;
    let mut f = d * (-cx - tx) + e * (-cy - ty) + cy;
    if expand {
        let corners = [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)].map(|(x, y)| (a * x + b * y + c, d * x + e * y + f));
        let nw =
            corners.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max).ceil() - corners.iter().map(|p| p.0).fold(f64::INFINITY, f64::min).floor();
        let nh =
            corners.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max).ceil() - corners.iter().map(|p| p.1).fold(f64::INFINITY, f64::min).floor();
        let (dx, dy) = (-(nw - w) / 2.0, -(nh - h) / 2.0);
        (c, f) = (a * dx + b * dy + c, d * dx + e * dy + f);
        (w, h) = (nw, nh);
    }
    if !w.is_finite() || !h.is_finite() || w > f64::from(u32::MAX) || h > f64::from(u32::MAX) {
        return Err(PyOverflowError::new_err("rotation dimensions are too large"));
    }
    Ok(((w as u32, h as u32), (a, b, c, d, e, f)))
}
