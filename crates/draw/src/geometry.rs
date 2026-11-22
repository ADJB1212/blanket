//! Geometry preparation shared by the Python drawing adapters.

use std::f64::consts::PI;

use pyo3::exceptions::{PyOverflowError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

type Point = (f64, f64);
type IntPoint = (i64, i64);
pub(crate) type Edge = (IntPoint, IntPoint);
type Bounds = (f64, f64, f64, f64);
type IntBounds = (i64, i64, i64, i64);
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<Outline>()?;
    module.add_function(wrap_pyfunction!(draw_regular_vertices, module)?)?;
    module.add_function(wrap_pyfunction!(draw_arc_points, module)?)?;
    module.add_function(wrap_pyfunction!(draw_rounded_points, module)?)?;
    module.add_function(wrap_pyfunction!(draw_line_joints, module)?)?;
    module.add_function(wrap_pyfunction!(draw_text_positions, module)?)?;
    Ok(())
}

pub(crate) fn integer(value: f64) -> PyResult<i64> {
    if value.is_nan() {
        return Err(PyValueError::new_err("cannot convert float NaN to integer"));
    }
    if !value.is_finite() || !(-9_223_372_036_854_775_808.0..9_223_372_036_854_775_808.0).contains(&value.trunc()) {
        return Err(PyOverflowError::new_err("drawing coordinate is too large"));
    }
    Ok(value as i64)
}

fn int_point(point: Point) -> PyResult<IntPoint> {
    Ok((integer(point.0)?, integer(point.1)?))
}

const BEZIER_STEPS: usize = 32;

pub(crate) fn flatten_bezier(points: &[Point]) -> PyResult<Vec<Point>> {
    if !matches!(points.len(), 3 | 4) {
        return Err(PyValueError::new_err("a Bezier curve requires three or four points"));
    }
    for &point in points {
        int_point(point)?;
    }
    let mut vertices = vec![points[0]];
    match *points {
        [a, b, c] => subdivide_bezier([a, b, c], &mut vertices, 0),
        [a, b, c, d] => subdivide_bezier([a, b, c, d], &mut vertices, 0),
        _ => unreachable!(),
    }
    Ok(vertices)
}

fn subdivide_bezier<const N: usize>(mut points: [Point; N], vertices: &mut Vec<Point>, depth: u8) {
    const MAX_DEPTH: u8 = 16;
    const TOLERANCE: f64 = 0.0625;
    let start = points[0];
    let end = points[N - 1];
    // The control-point deviations bound the curve's deviation from its chord.
    let flat = (1..N - 1).all(|i| {
        let t = i as f64 / (N - 1) as f64;
        let dx = (points[i].0 - start.0) - t * (end.0 - start.0);
        let dy = (points[i].1 - start.1) - t * (end.1 - start.1);
        dx * dx + dy * dy <= TOLERANCE * TOLERANCE
    });
    if flat || depth == MAX_DEPTH {
        vertices.push(end);
        return;
    }
    let mut left = [start; N];
    let mut right = [end; N];
    for level in 1..N {
        for i in 0..N - level {
            points[i] = (points[i].0.midpoint(points[i + 1].0), points[i].1.midpoint(points[i + 1].1));
        }
        left[level] = points[0];
        right[N - 1 - level] = points[N - 1 - level];
    }
    subdivide_bezier(left, vertices, depth + 1);
    subdivide_bezier(right, vertices, depth + 1);
}

fn bezier_edges(start: Point, controls: [Point; 3]) -> PyResult<([Edge; BEZIER_STEPS], Point)> {
    let [(x1, y1), (x2, y2), (x3, y3)] = controls;
    let mut previous = int_point(start)?;
    let mut end = start;
    let mut edges = [((0, 0), (0, 0)); BEZIER_STEPS];
    for (i, edge) in edges.iter_mut().enumerate() {
        let t = (i + 1) as f64 / BEZIER_STEPS as f64;
        let t2 = t * t;
        let t3 = t2 * t;
        let u = 1.0 - t;
        let u2 = u * u;
        let u3 = u2 * u;
        end = (start.0 * u3 + 3.0 * (x1 * t * u2 + x2 * t2 * u) + x3 * t3 + 0.5, start.1 * u3 + 3.0 * (y1 * t * u2 + y2 * t2 * u) + y3 * t3 + 0.5);
        let point = int_point(end)?;
        *edge = (previous, point);
        previous = point;
    }
    Ok((edges, end))
}

/// Flattened cubic Bezier and line segments for ImageDraw.shape().
///
/// Edges are integer point pairs. `x` and `y` give the current position;
/// `x0` and `y0` give the origin used by `close()`.
#[pyclass(name = "Outline", module = "blanket.ImageDraw")]
#[derive(Default)]
pub(crate) struct Outline {
    pub(crate) edges: Vec<Edge>,
    #[pyo3(get)]
    x: f64,
    #[pyo3(get)]
    y: f64,
    #[pyo3(get)]
    x0: f64,
    #[pyo3(get)]
    y0: f64,
}

#[pymethods]
impl Outline {
    /// Initialize an empty Outline at the origin.
    #[new]
    fn new() -> Self {
        Self::default()
    }

    /// Move the current position and origin to (x0, y0) without drawing an edge.
    #[pyo3(name = "move")]
    fn move_to(&mut self, x0: f64, y0: f64) {
        (self.x, self.y, self.x0, self.y0) = (x0, y0, x0, y0);
    }

    /// Add a line segment from the current position to (x1, y1).
    fn line(&mut self, x1: f64, y1: f64) -> PyResult<()> {
        self.edges.push((int_point((self.x, self.y))?, int_point((x1, y1))?));
        (self.x, self.y) = (x1, y1);
        Ok(())
    }

    /// Add a 32-segment cubic Bezier curve through two control points to (x3, y3).
    fn curve(&mut self, x1: f64, y1: f64, x2: f64, y2: f64, x3: f64, y3: f64) -> PyResult<()> {
        let (edges, position) = bezier_edges((self.x, self.y), [(x1, y1), (x2, y2), (x3, y3)])?;
        self.edges.extend_from_slice(&edges);
        (self.x, self.y) = position;
        Ok(())
    }

    /// Connect the current position back to the origin unless they are equal.
    fn close(&mut self) -> PyResult<()> {
        if self.x == self.x0 && self.y == self.y0 {
            return Ok(());
        }
        self.line(self.x0, self.y0)
    }

    /// Apply the affine matrix `(a, b, c, d, e, f)` to every point, truncating to integers.
    fn transform(&mut self, matrix: &Bound<'_, PyAny>) -> PyResult<()> {
        let describe = || matrix.repr().map(|text| text.to_string()).unwrap_or_default();
        if matrix.len().ok() != Some(6) {
            return Err(PyTypeError::new_err(format!("transform matrix must contain six values, got {}", describe())));
        }
        let [a, b, c, d, e, f] = if let Ok(tuple) = matrix.cast_exact::<PyTuple>() {
            let mut values = [0.0; 6];
            for (value, item) in values.iter_mut().zip(tuple.as_slice()) {
                *value = item.extract()?;
            }
            values
        } else {
            let values = matrix.try_iter()?.map(|value| value?.extract::<f64>()).collect::<PyResult<Vec<_>>>()?;
            let Ok(values) = <[f64; 6]>::try_from(values) else {
                return Err(PyTypeError::new_err(format!("transform matrix must contain six values, got {}", describe())));
            };
            values
        };
        if ![a, b, c, d, e, f].iter().all(|value| value.is_finite()) {
            return Err(PyValueError::new_err(format!("transform matrix must be finite, got {}", describe())));
        }
        let point = |(x, y): Point| int_point((a * x + b * y + c, d * x + e * y + f));
        let edges = self
            .edges
            .iter()
            .map(|&(start, end)| Ok((point((start.0 as f64, start.1 as f64))?, point((end.0 as f64, end.1 as f64))?)))
            .collect::<PyResult<Vec<_>>>()?;
        let (position, origin) = (point((self.x, self.y))?, point((self.x0, self.y0))?);
        self.edges = edges;
        (self.x, self.y) = (position.0 as f64, position.1 as f64);
        (self.x0, self.y0) = (origin.0 as f64, origin.1 as f64);
        Ok(())
    }

    /// A copy of the edges as `((x0, y0), (x1, y1))` integer pairs.
    #[getter]
    fn edges(&self) -> Vec<Edge> {
        self.edges.clone()
    }

    /// Return the polygon vertices traced by the edges.
    fn _points(&self) -> Vec<IntPoint> {
        let mut points: Vec<_> = self.edges.iter().map(|edge| edge.0).collect();
        points.extend(self.edges.last().map(|edge| edge.1));
        points
    }
}

fn round_hundredths(value: f64) -> f64 {
    // Decimal formatting rounds the binary value directly, as Python round(x, 2) does.
    format!("{value:.2}").parse().expect("formatted float")
}

#[pyfunction]
fn draw_regular_vertices(center: Point, radius: f64, sides: usize, rotation: f64) -> PyResult<Vec<Point>> {
    if sides < 3 {
        return Err(PyValueError::new_err("n_sides must be greater than two"));
    }
    if rotation.is_infinite() {
        return Err(PyValueError::new_err("math domain error"));
    }
    let degrees = 360.0 / sides as f64;
    let mut angle = (270.0 - 0.5 * degrees) + rotation;
    let mut points = Vec::new();
    points.try_reserve_exact(sides).map_err(|_| pyo3::exceptions::PyMemoryError::new_err("polygon is too large"))?;
    for _ in 0..sides {
        let radians = (360.0 - angle) * (PI / 180.0);
        points.push((
            round_hundredths(radius * radians.cos() - 0.0 * radians.sin() + center.0),
            round_hundredths(0.0 * radians.cos() + radius * radians.sin() + center.1),
        ));
        angle += degrees;
        if angle > 360.0 {
            angle -= 360.0;
        }
    }
    Ok(points)
}

#[pyfunction(signature = (bounds, start, end, style="arc"))]
fn draw_arc_points(bounds: IntBounds, start: f64, end: f64, style: &str) -> PyResult<Vec<Point>> {
    if !start.is_finite() || !end.is_finite() {
        return Err(PyValueError::new_err("angles must be finite"));
    }
    let difference = end - start;
    let sweep = if difference >= 360.0 { 360.0 } else { difference.rem_euclid(360.0) };
    if sweep.is_nan() {
        return Err(PyValueError::new_err("cannot convert float NaN to integer"));
    }
    if sweep == 0.0 {
        return Ok(Vec::new());
    }
    let (left, top, right, bottom) = bounds;
    let (rx, ry) = ((i128::from(right) - i128::from(left)) as f64 / 2.0, (i128::from(bottom) - i128::from(top)) as f64 / 2.0);
    let (x0, y0) = (left as f64, top as f64);
    let count = (rx.max(ry) * sweep * PI / 180.0 * 2.0).ceil().clamp(1.0, 65536.0) as u32;
    let mut points: Vec<_> = (0..=count)
        .map(|i| {
            let radians = (start + sweep * f64::from(i) / f64::from(count)) * PI / 180.0;
            (x0 + rx + rx * radians.cos(), y0 + ry + ry * radians.sin())
        })
        .collect();
    if style == "pie" && difference.abs() < 360.0 {
        points.push(((i128::from(left) + i128::from(right)) as f64 / 2.0, (i128::from(top) + i128::from(bottom)) as f64 / 2.0));
    }
    Ok(points)
}

#[pyfunction]
fn draw_rounded_points(bounds: Bounds, radius: f64, corners: [bool; 4]) -> Vec<Point> {
    let (x0, y0, x1, y1) = bounds;
    let radius = radius.min((x1 - x0) / 2.0).min((y1 - y0) / 2.0);
    let mut points = Vec::new();
    let centers = [
        (x0 + radius, y0 + radius, 180.0, (x0, y0)),
        (x1 - radius, y0 + radius, 270.0, (x1, y0)),
        (x1 - radius, y1 - radius, 0.0, (x1, y1)),
        (x0 + radius, y1 - radius, 90.0, (x0, y1)),
    ];
    for (enabled, (cx, cy, start, corner)) in corners.into_iter().zip(centers) {
        if enabled && radius != 0.0 {
            let count = (radius * PI).ceil().clamp(1.0, 16384.0) as u32;
            points.extend((0..=count).map(|i| {
                let radians = (start + 90.0 * f64::from(i) / f64::from(count)) * PI / 180.0;
                (cx + radius * radians.cos(), cy + radius * radians.sin())
            }));
        } else {
            points.push(corner);
        }
    }
    points
}

type Joint = (Bounds, f64, f64, Vec<IntPoint>);

#[pyfunction]
fn draw_line_joints(points: Vec<IntPoint>, width: i64) -> PyResult<Vec<Joint>> {
    let mut joints = Vec::new();
    for triple in points.windows(3) {
        let [previous, center, following] = [triple[0], triple[1], triple[2]];
        let point = (center.0 as f64, center.1 as f64);
        let angle = |start: IntPoint, end: IntPoint| {
            let dx = (i128::from(end.0) - i128::from(start.0)) as f64;
            let dy = (i128::from(start.1) - i128::from(end.1)) as f64;
            dx.atan2(dy).to_degrees().rem_euclid(360.0)
        };
        let angles = [angle(previous, center), angle(center, following)];
        if angles[0] == angles[1] {
            continue;
        }
        let flipped = (angles[1] > angles[0] && angles[1] - 180.0 > angles[0]) || (angles[1] < angles[0] && angles[1] + 180.0 > angles[0]);
        let (start, end) = if flipped { (angles[1] + 90.0, angles[0] + 90.0) } else { (angles[0] - 90.0, angles[1] - 90.0) };
        let half = width as f64 / 2.0;
        let bounds = (point.0 - half + 1.0, point.1 - half + 1.0, point.0 + half - 1.0, point.1 + half - 1.0);
        let mut gap = Vec::new();
        if width > 8 {
            for angle in angles {
                let radians = (angle + if flipped { 90.0 } else { -90.0 } - 90.0) * PI / 180.0;
                let (dx, dy) = ((half - 1.0) * radians.cos(), (half - 1.0) * radians.sin());
                let x = center.0.checked_add(integer(dx)?);
                let y = center.1.checked_add(integer(dy)?);
                gap.push(x.zip(y).ok_or_else(|| PyOverflowError::new_err("drawing coordinate is too large"))?);
            }
            gap.insert(1, center);
        }
        joints.push((bounds, start - 90.0, end - 90.0, gap));
    }
    Ok(joints)
}

#[pyfunction]
fn draw_text_positions(xy: Point, widths: Vec<f64>, height: f64, stroke: f64, spacing: f64, anchor: &str, align: &str) -> PyResult<Vec<Point>> {
    if widths.is_empty() {
        return Ok(Vec::new());
    }
    let max_width = widths.iter().copied().reduce(|a, b| if b > a { b } else { a }).unwrap();
    let line_height = height + stroke + spacing;
    let mut y = xy.1;
    match anchor.as_bytes().get(1) {
        Some(b'a' | b's') => (),
        Some(b'm') => y -= (widths.len() - 1) as f64 * line_height / 2.0,
        Some(b'd') => y -= (widths.len() - 1) as f64 * line_height,
        _ => return Err(PyValueError::new_err("invalid multiline text anchor")),
    }
    widths
        .into_iter()
        .enumerate()
        .map(|(i, width)| {
            let difference = max_width - width;
            let mut shift = match align {
                "left" => 0.0,
                "center" => difference / 2.0,
                "right" => difference,
                _ => return Err(PyValueError::new_err("align must be left, center, or right")),
            };
            shift -= match anchor.as_bytes().first() {
                Some(b'l') => 0.0,
                Some(b'm') => difference / 2.0,
                Some(b'r') => difference,
                _ => return Err(PyValueError::new_err("invalid multiline text anchor")),
            };
            Ok((xy.0 + shift, y + i as f64 * line_height))
        })
        .collect()
}
