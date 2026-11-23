//! Connected Bezier geometry and coverage rasterization.

use blanket_core::Image;
use kurbo::PathEl;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use tiny_skia::{FillRule, LineCap, LineJoin, Mask, Path, PathBuilder, PathSegment, Point, Stroke, Transform};

use crate::draw::{check_coverage, coverage_paint, ink};

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<BezierPath>()?;
    module.add_function(wrap_pyfunction!(draw_bezier_path, module)?)
}

/// Mutable geometry for ImageDraw.path().
#[pyclass(module = "blanket.ImageDraw")]
#[derive(Debug, Default)]
struct BezierPath {
    builder: PathBuilder,
}

impl BezierPath {
    fn check_segment(&self, coordinates: &[f32]) -> PyResult<()> {
        if self.builder.is_empty() {
            return Err(PyValueError::new_err("start the path with move_to()"));
        }
        check_coordinates(coordinates)
    }
}

fn check_coordinates(coordinates: &[f32]) -> PyResult<()> {
    if coordinates.iter().any(|value| !value.is_finite() || value.abs() >= i64::MAX as f32) {
        return Err(PyValueError::new_err("path coordinates must be finite and within the signed 64-bit range"));
    }
    Ok(())
}

#[pymethods]
impl BezierPath {
    #[new]
    fn new() -> Self {
        Self::default()
    }

    /// Start a contour at (x, y).
    fn move_to(&mut self, x: f32, y: f32) -> PyResult<()> {
        check_coordinates(&[x, y])?;
        self.builder.move_to(x, y);
        Ok(())
    }

    /// Add a straight segment to (x, y).
    fn line_to(&mut self, x: f32, y: f32) -> PyResult<()> {
        self.check_segment(&[x, y])?;
        self.builder.line_to(x, y);
        Ok(())
    }

    /// Add a quadratic curve with control point (cx, cy) and endpoint (x, y).
    fn quadratic_to(&mut self, cx: f32, cy: f32, x: f32, y: f32) -> PyResult<()> {
        self.check_segment(&[cx, cy, x, y])?;
        self.builder.quad_to(cx, cy, x, y);
        Ok(())
    }

    /// Add a cubic curve with two control points and endpoint (x, y).
    fn cubic_to(&mut self, cx1: f32, cy1: f32, cx2: f32, cy2: f32, x: f32, y: f32) -> PyResult<()> {
        self.check_segment(&[cx1, cy1, cx2, cy2, x, y])?;
        self.builder.cubic_to(cx1, cy1, cx2, cy2, x, y);
        Ok(())
    }

    /// Close the contour with a segment back to its start.
    fn close(&mut self) {
        self.builder.close();
    }
}

fn stroke_style((width, join, cap, miter_limit): (f32, &str, &str, f32)) -> PyResult<Stroke> {
    if !width.is_finite() || width < 0.0 || width >= i64::MAX as f32 {
        return Err(PyValueError::new_err("path width must be finite, nonnegative, and within the signed 64-bit range"));
    }
    if !miter_limit.is_finite() || miter_limit < 1.0 {
        return Err(PyValueError::new_err("miter_limit must be finite and at least one"));
    }
    let line_join = match join {
        "round" => LineJoin::Round,
        "bevel" => LineJoin::Bevel,
        "miter" => LineJoin::Miter,
        _ => return Err(PyValueError::new_err("join must be 'round', 'bevel', or 'miter'")),
    };
    let line_cap = match cap {
        "butt" => LineCap::Butt,
        "round" => LineCap::Round,
        "square" => LineCap::Square,
        _ => return Err(PyValueError::new_err("cap must be 'butt', 'round', or 'square'")),
    };
    Ok(Stroke { width, line_join, line_cap, miter_limit, dash: None })
}

fn dash_path(path: &Path, lengths: &[f64], offset: f64) -> PyResult<Option<Path>> {
    const MAX_DASH_ELEMENTS: usize = 1_000_000;
    let point = |p: Point| kurbo::Point::new(f64::from(p.x), f64::from(p.y));
    let elements = path.segments().map(|segment| match segment {
        PathSegment::MoveTo(p) => PathEl::MoveTo(point(p)),
        PathSegment::LineTo(p) => PathEl::LineTo(point(p)),
        PathSegment::QuadTo(a, p) => PathEl::QuadTo(point(a), point(p)),
        PathSegment::CubicTo(a, b, p) => PathEl::CurveTo(point(a), point(b), point(p)),
        PathSegment::Close => PathEl::ClosePath,
    });
    let mut builder = PathBuilder::new();
    // Preserve a closed join when one dash covers the whole contour.
    for (index, element) in kurbo::dash(elements, offset, lengths).enumerate() {
        if index >= MAX_DASH_ELEMENTS {
            return Err(PyValueError::new_err("dashed path contains too many segments"));
        }
        match element {
            PathEl::MoveTo(p) => builder.move_to(p.x as f32, p.y as f32),
            PathEl::LineTo(p) => builder.line_to(p.x as f32, p.y as f32),
            PathEl::QuadTo(a, p) => builder.quad_to(a.x as f32, a.y as f32, p.x as f32, p.y as f32),
            PathEl::CurveTo(a, b, p) => builder.cubic_to(a.x as f32, a.y as f32, b.x as f32, b.y as f32, p.x as f32, p.y as f32),
            PathEl::ClosePath => builder.close(),
        }
    }
    Ok(builder.finish())
}

#[pyfunction]
fn draw_bezier_path(
    image: &mut Image, path: PyRef<'_, BezierPath>, inks: (Option<Bound<'_, PyAny>>, Option<Bound<'_, PyAny>>), style: (f32, &str, &str, f32),
    dashes: (Vec<f64>, f64), fill_rule: &str, context: (bool, bool),
) -> PyResult<()> {
    let (fill, outline) = inks;
    let (blend, antialias) = context;
    let stroke = stroke_style(style)?;
    let (mut lengths, dash_offset) = dashes;
    if !dash_offset.is_finite() {
        return Err(PyValueError::new_err("dash_offset must be finite"));
    }
    if lengths.iter().any(|&length| !length.is_finite() || length <= 0.0) {
        return Err(PyValueError::new_err("dash lengths must be finite and positive"));
    }
    if lengths.len() % 2 != 0 {
        lengths.extend_from_within(..);
    }
    if !lengths.iter().sum::<f64>().is_finite() {
        return Err(PyValueError::new_err("dash pattern is out of range"));
    }
    let fill_rule = match fill_rule {
        "nonzero" => FillRule::Winding,
        "evenodd" => FillRule::EvenOdd,
        _ => return Err(PyValueError::new_err("fill_rule must be 'nonzero' or 'evenodd'")),
    };
    let fill = fill.map(|value| ink(image, &value, blend)).transpose()?;
    let outline = outline.filter(|_| stroke.width > 0.0).map(|value| ink(image, &value, blend)).transpose()?;
    if antialias {
        check_coverage(image)?;
    }
    let py = path.py();
    let path = path.builder.clone().finish();
    image.ensure_owned()?;
    py.detach(|| {
        let Some(path) = path else {
            return Ok(());
        };
        // Dash and stroke before clipping so the phase and joins survive tile boundaries.
        let dashed = if outline.is_some() && !lengths.is_empty() { dash_path(&path, &lengths, dash_offset)? } else { None };
        let centerline = if lengths.is_empty() { Some(&path) } else { dashed.as_ref() };
        if let Some(fill) = fill {
            rasterize(image, &path, fill_rule, &fill, blend, antialias)?;
        }
        if let Some((centerline, outline)) = centerline.zip(outline)
            && let Some(stroke) = centerline.stroke(&stroke, 1.0)
        {
            rasterize(image, &stroke, FillRule::Winding, &outline, blend, antialias)?;
        }
        Ok(())
    })
}

fn rasterize(image: &mut Image, path: &Path, rule: FillRule, ink: &[u8], blend: bool, antialias: bool) -> PyResult<()> {
    const TILE: u32 = 512;
    let bounds = path.bounds();
    // The rasterizer samples at half-integers. Blanket pixel centers are integers.
    let left = bounds.left().floor() as u32;
    let top = bounds.top().floor() as u32;
    let right = ((f64::from(bounds.right()) + 1.0).ceil() as u32).min(image.width());
    let bottom = ((f64::from(bounds.bottom()) + 1.0).ceil() as u32).min(image.height());
    if left >= right || top >= bottom {
        return Ok(());
    }
    let width = TILE.min(right - left);
    let height = TILE.min(bottom - top);
    let mut mask = Mask::new(width, height).expect("bounded nonempty mask");
    for y in (top..bottom).step_by(height as usize) {
        for x in (left..right).step_by(width as usize) {
            mask.clear();
            let transform = Transform::from_translate((0.5 - f64::from(x)) as f32, (0.5 - f64::from(y)) as f32);
            mask.fill_path(path, rule, antialias, transform);
            for (row, values) in mask.data().chunks_exact(width as usize).take((bottom - y) as usize).enumerate() {
                for (column, &coverage) in values.iter().take((right - x) as usize).enumerate() {
                    coverage_paint(image, i64::from(x) + column as i64, i64::from(y) + row as i64, ink, blend, coverage)?;
                }
            }
        }
    }
    Ok(())
}
