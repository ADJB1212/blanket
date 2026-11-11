// Raster geometry derived from Pillow's libImaging/Draw.c; see licenses/Pillow-LICENSE.
use blanket_core::Image;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

struct Quarter {
    a: i64,
    b: i64,
    x: i64,
    y: i64,
    done: bool,
}

impl Quarter {
    fn new(a: i64, b: i64) -> Self {
        Self { a, b, x: a, y: b % 2, done: a < 0 || b < 0 }
    }

    fn delta(&self, x: i64, y: i64) -> i128 {
        let (a, b, x, y) = (i128::from(self.a), i128::from(self.b), i128::from(x), i128::from(y));
        (a * a * y * y + b * b * x * x - a * a * b * b).abs()
    }

    fn next(&mut self) -> Option<(i64, i64)> {
        if self.done {
            return None;
        }
        let result = (self.x, self.y);
        if self.x == self.a % 2 && self.y == self.b {
            self.done = true;
        } else {
            let (mut x, mut y) = (self.x, self.y + 2);
            let mut delta = self.delta(x, y);
            if x > 1 {
                let d = self.delta(self.x - 2, self.y + 2);
                if delta > d {
                    x = self.x - 2;
                    y = self.y + 2;
                    delta = d;
                }
                if delta > self.delta(self.x - 2, self.y) {
                    x = self.x - 2;
                    y = self.y;
                }
            }
            self.x = x;
            self.y = y;
        }
        Some(result)
    }
}

fn ellipse_spans(a: i64, b: i64, width: i64, mut emit: impl FnMut(i64, i64, i64)) {
    if width < 1 {
        return;
    }
    let mut outer = Quarter::new(a, b);
    let Some((mut right, mut y)) = outer.next() else {
        return;
    };
    let mut inner = Quarter::new(a - 2 * (width - 1), b - 2 * (width - 1));
    let mut left = a % 2;
    loop {
        let (row, mut l, r) = (y, left, right);
        let mut next = outer.next();
        while next.is_some_and(|p| p.1 <= row) {
            next = outer.next();
        }
        if let Some((x, ny)) = next {
            right = x;
            y = ny;
        }
        let mut inside = inner.next();
        while let Some((x, iy)) = inside {
            if iy > row {
                break;
            }
            l = x;
            inside = inner.next();
        }
        left = inside.map_or(a % 2, |p| p.0);
        let positive = if l == 0 { 2 } else { l };
        emit(-r, -row, -l);
        if l > 0 || l < r {
            emit(positive, -row, r);
        }
        if row > 0 {
            emit(-r, row, -l);
            if l > 0 || l < r {
                emit(positive, row, r);
            }
        }
        if next.is_none() {
            break;
        }
    }
}

enum Clip {
    Pass,
    Plane(f64, f64, f64),
    And(Box<Clip>, Box<Clip>),
    Or(Box<Clip>, Box<Clip>),
}

impl Clip {
    fn and(self, other: Self) -> Self {
        Self::And(Box::new(self), Box::new(other))
    }
    fn or(self, other: Self) -> Self {
        Self::Or(Box::new(self), Box::new(other))
    }
    fn combine(self, other: Self, intersection: bool) -> Self {
        if intersection { self.and(other) } else { self.or(other) }
    }
    fn transpose(self) -> Self {
        match self {
            Self::Pass => Self::Pass,
            Self::Plane(a, b, c) => Self::Plane(b, a, c),
            Self::And(a, b) => a.transpose().and(b.transpose()),
            Self::Or(a, b) => a.transpose().or(b.transpose()),
        }
    }

    fn spans(&self, mut left: i64, y: i64, mut right: i64) -> Vec<(i64, i64)> {
        match self {
            Self::Pass => vec![(left, right)],
            Self::Plane(a, b, c) => {
                let row = b * y as f64 + c;
                if a.abs() < 1e-9 {
                    if row < -1e-9 {
                        return Vec::new();
                    }
                } else {
                    let x = -row / a;
                    if a * left as f64 + row < 1e-9 {
                        left = (left as f64).max(x).round() as i64;
                    }
                    if a * right as f64 + row < 1e-9 {
                        right = (right as f64).min(x).round() as i64;
                    }
                }
                if left <= right { vec![(left, right)] } else { Vec::new() }
            }
            Self::And(a, b) => a.spans(left, y, right).into_iter().flat_map(|(l, r)| b.spans(l, y, r)).collect(),
            Self::Or(a, b) => {
                let mut spans = a.spans(left, y, right);
                spans.extend(b.spans(left, y, right));
                spans.sort_unstable();
                let mut result: Vec<(i64, i64)> = Vec::new();
                for (l, r) in spans {
                    if let Some(last) = result.last_mut().filter(|last| last.1 >= l) {
                        last.1 = last.1.max(r);
                    } else {
                        result.push((l, r));
                    }
                }
                result
            }
        }
    }
}

fn normalize(start: f32, end: f32) -> (f32, f32) {
    if end - start >= 360.0 {
        (0.0, 360.0)
    } else {
        let start = start.rem_euclid(360.0);
        (start, start + (end - start).rem_euclid(360.0))
    }
}

fn arc_clip(a: i64, b: i64, start: f32, end: f32) -> Clip {
    if a < b {
        return arc_clip(b, a, 90.0 - end, 90.0 - start).transpose();
    }
    let (start, end) = normalize(start, end);
    if end == start + 360.0 {
        return Clip::Pass;
    }
    let (al, ar) = (f64::from(start).to_radians(), f64::from(end).to_radians());
    let (a, b) = (a as f64, b as f64);
    let l = Clip::Plane(-a * al.sin(), b * al.cos(), (a * a - b * b) * (al * 2.0).sin() / 2.0);
    let r = Clip::Plane(a * ar.sin(), -b * ar.cos(), (b * b - a * a) * (ar * 2.0).sin() / 2.0);
    if start % 180.0 == 0.0 || end % 180.0 == 0.0 {
        l.combine(r, end - start < 180.0)
    } else if ((start / 180.0) as i32 + (end / 180.0) as i32) % 2 == 1 {
        let bottom = |angle: f32| Clip::Plane(0.0, if (angle / 180.0) as i32 % 2 == 0 { 1.0 } else { -1.0 }, 0.0);
        bottom(start).and(l).or(bottom(end).and(r))
    } else {
        l.combine(r, end - start < 180.0)
            .combine(Clip::Plane(0.0, if !(180.0..=540.0).contains(&end) { 1.0 } else { -1.0 }, 0.0), end - start < 180.0)
    }
}

fn endpoints(a: i64, b: i64, start: f32, end: f32) -> (f64, f64, f64, f64) {
    let (al, ar) = (f64::from(start).to_radians(), f64::from(end).to_radians());
    (a as f64 * al.cos(), b as f64 * al.sin(), a as f64 * ar.cos(), b as f64 * ar.sin())
}

fn pie_clip(a: i64, b: i64, start: f32, end: f32) -> Clip {
    let (xl, yl, xr, yr) = endpoints(a, b, start, end);
    let clip = Clip::Plane(-yl, xl, 0.0).combine(Clip::Plane(yr, -xr, 0.0), end - start < 180.0);
    if end - start < 90.0 { clip.and(Clip::Plane((xl + xr) / 2.0, (yl + yr) / 2.0, 0.0)) } else { clip }
}

fn render(image: &mut Image, bounds: (i64, i64, i64, i64), width: i64, clip: &Clip, ink: &[u8], blend: bool) {
    let (x0, y0, x1, y1) = bounds;
    let (a, b) = (x1 - x0, y1 - y0);
    ellipse_spans(a, b, width, |l, y, r| {
        let row = y0 + (y + b) / 2;
        if row < 0 || row >= i64::from(image.height()) {
            return;
        }
        if matches!(clip, Clip::Pass) {
            super::draw::hline(image, x0 + (l + a) / 2, x0 + (r + a) / 2, row, ink, blend);
            return;
        }
        for (left, right) in clip.spans(l, y, r) {
            super::draw::hline(image, x0 + (left + a) / 2, x0 + (right + a) / 2, row, ink, blend);
        }
    });
}

pub(crate) fn ellipse(image: &mut Image, bounds: (i64, i64, i64, i64), width: i64, ink: &[u8], blend: bool) {
    let width = if width == 0 { bounds.2 - bounds.0 + bounds.3 - bounds.1 } else { width };
    render(image, bounds, width, &Clip::Pass, ink, blend);
}

#[pyfunction]
fn draw_curve(
    image: &mut Image, bounds: (i64, i64, i64, i64), start: f32, end: f32, width: i64, style: &str, filled: bool, value: &Bound<'_, PyAny>,
    blend: bool,
) -> PyResult<()> {
    if !start.is_finite() || !end.is_finite() {
        return Err(PyValueError::new_err("angles must be finite"));
    }
    let ink = super::draw::ink(image, value, blend)?;
    image.ensure_owned()?;
    let (start, end) = normalize(start, end);
    let (a, b) = (i128::from(bounds.2) - i128::from(bounds.0), i128::from(bounds.3) - i128::from(bounds.1));
    if start == end || a < 0 || b < 0 || (!filled && width < 1) {
        return Ok(());
    }
    if a > 1_000_000 || b > 1_000_000 {
        return Err(PyValueError::new_err("ellipse bounds are too large"));
    }
    if bounds.2 < 0 || bounds.3 < 0 || bounds.0 >= i64::from(image.width()) || bounds.1 >= i64::from(image.height()) {
        return Ok(());
    }
    let (a, b) = (a as i64, b as i64);
    let width = width.min(a.max(b) + 1);
    value.py().detach(|| {
        if end == start + 360.0 {
            ellipse(image, bounds, if filled { 0 } else { width }, &ink, blend);
            return;
        }
        let (xl, yl, xr, yr) = endpoints(a, b, start, end);
        let line_a = yr - yl;
        let line_b = xl - xr;
        let line_c = -(line_a * xl + line_b * yl);
        let clip = match style {
            "arc" => arc_clip(a, b, start, end),
            "chord" => Clip::Plane(line_a, line_b, line_c),
            _ => pie_clip(a, b, start, end),
        };
        if !filled && style == "chord" {
            let chord = Clip::Plane(line_a, line_b, line_c).and(Clip::Plane(-line_a, -line_b, 2.0 * width as f64 * line_a.hypot(line_b) - line_c));
            render(image, bounds, a + b + 1, &chord, &ink, blend);
        } else if !filled && style == "pie" {
            for (x, y) in [(xl, yl), (xr, yr)] {
                let c = width as f64 * x.hypot(y);
                let side = Clip::Plane(-y, x, c).and(Clip::Plane(y, -x, c)).and(Clip::Plane(x, y, 0.0));
                render(image, bounds, a + b + 1, &side, &ink, blend);
            }
            let x = ((bounds.0 + bounds.2 - width) as f64 / 2.0).round() as i64;
            let y = ((bounds.1 + bounds.3 - width) as f64 / 2.0).round() as i64;
            ellipse(image, (x, y, x + width - 1, y + width - 1), 0, &ink, blend);
        }
        render(image, bounds, if filled { a + b + 1 } else { width }, &clip, &ink, blend);
    });
    Ok(())
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(draw_curve, module)?)?;
    module.add_function(wrap_pyfunction!(draw_polygon, module)?)
}

#[derive(Clone)]
struct Edge {
    x: i64,
    y: i64,
    xmin: i64,
    xmax: i64,
    ymin: i64,
    ymax: i64,
    slope: f32,
}

impl Edge {
    fn new(a: (i64, i64), b: (i64, i64)) -> Self {
        Self {
            x: a.0,
            y: a.1,
            xmin: a.0.min(b.0),
            xmax: a.0.max(b.0),
            ymin: a.1.min(b.1),
            ymax: a.1.max(b.1),
            slope: if a.1 == b.1 { 0.0 } else { (i128::from(b.0) - i128::from(a.0)) as f32 / (i128::from(b.1) - i128::from(a.1)) as f32 },
        }
    }
    fn at(&self, y: i64) -> f32 {
        let offset = (i128::from(y) - i128::from(self.y)) as f32;
        if cfg!(target_arch = "aarch64") { offset.mul_add(self.slope, self.x as f32) } else { offset * self.slope + self.x as f32 }
    }
}

fn round_down(value: f64) -> i64 {
    if value >= 0.0 { (value - 0.5).ceil() as i64 } else { -((-value - 0.5).ceil() as i64) }
}

fn polygon_spans(points: &[(i64, i64)], height: u32, blend: bool, mut emit: impl FnMut(i64, i64, i64)) {
    if points.len() < 2 {
        return;
    }
    let mut edges: Vec<Edge> = Vec::with_capacity(points.len());
    for (i, pair) in points.windows(2).enumerate() {
        if i > 0 && pair[0].1 == pair[1].1 && pair[0].1 == points[i - 1].1 {
            let last = edges.last_mut().unwrap();
            if pair[1].0 > pair[0].0 && pair[0].0 > points[i - 1].0 {
                last.xmax = pair[1].0;
                continue;
            }
            if pair[1].0 < pair[0].0 && pair[0].0 < points[i - 1].0 {
                last.xmin = pair[1].0;
                continue;
            }
        }
        edges.push(Edge::new(pair[0], pair[1]));
    }
    if points.last() != points.first() {
        edges.push(Edge::new(*points.last().unwrap(), points[0]));
    }
    let first = edges.iter().map(|e| e.ymin).min().unwrap().max(0);
    let last = edges.iter().map(|e| e.ymax).max().unwrap().min(i64::from(height));
    if !blend {
        for edge in &edges {
            if edge.ymin == edge.ymax {
                emit(edge.xmin, edge.ymin, edge.xmax);
            }
        }
    }
    let lines: Vec<_> = edges.iter().filter(|e| e.ymin != e.ymax).collect();
    let mut intersections = Vec::new();
    let mut spans = Vec::new();
    let mut horizontal: Vec<_> = if blend { edges.iter().filter(|e| e.ymin == e.ymax).collect() } else { Vec::new() };
    horizontal.sort_unstable_by_key(|edge| edge.ymin);
    let mut first_horizontal = 0;
    for y in first..=last {
        intersections.clear();
        for (i, edge) in lines.iter().enumerate() {
            if y < edge.ymin || y > edge.ymax {
                continue;
            }
            let mut x = edge.at(y);
            if y == edge.ymax && y < last {
                intersections.extend([x, x]);
                continue;
            }
            if (y == edge.ymin || y == edge.ymax) && edge.slope != 0.0 {
                for other in &lines[..i] {
                    if (y != other.ymin && y != other.ymax) || other.slope == 0.0 {
                        continue;
                    }
                    if x.round() == other.at(y).round() {
                        let adjacent_y = y + if y == edge.ymax { -1 } else { 1 };
                        let adjacent = edge.at(adjacent_y);
                        if adjacent_y >= other.ymin && adjacent_y <= other.ymax {
                            let other_adjacent = other.at(adjacent_y);
                            if x > adjacent + 1.0 && x > other_adjacent + 1.0 {
                                x = adjacent.max(other_adjacent).round() + 1.0;
                            } else if x < adjacent - 1.0 && x < other_adjacent - 1.0 {
                                x = adjacent.min(other_adjacent).round() - 1.0;
                            }
                            break;
                        }
                    }
                }
            }
            intersections.push(x);
        }
        intersections.sort_unstable_by(f32::total_cmp);
        spans.clear();
        let row_spans = intersections
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| (f64::from(pair[0]).round() as i64, round_down(f64::from(pair[1]))))
            .filter(|&(l, r)| l <= r);
        if !blend {
            for (l, r) in row_spans {
                emit(l, y, r);
            }
            continue;
        }
        spans.extend(row_spans);
        while first_horizontal < horizontal.len() && horizontal[first_horizontal].ymin < y {
            first_horizontal += 1;
        }
        while first_horizontal < horizontal.len() && horizontal[first_horizontal].ymin == y {
            let edge = horizontal[first_horizontal];
            spans.push((edge.xmin, edge.xmax));
            first_horizontal += 1;
        }
        spans.sort_unstable();
        let mut kept = 0;
        for i in 0..spans.len() {
            let (l, r) = spans[i];
            if kept != 0 && spans[kept - 1].1.saturating_add(1) >= l {
                spans[kept - 1].1 = spans[kept - 1].1.max(r);
            } else {
                spans[kept] = (l, r);
                kept += 1;
            }
        }
        spans.truncate(kept);
        for &(l, r) in &spans {
            emit(l, y, r);
        }
    }
}

fn wide_line(a: (i64, i64), b: (i64, i64), width: i64, height: u32, blend: bool, emit: &mut impl FnMut(i64, i64, i64)) {
    let (dx, dy) = ((i128::from(b.0) - i128::from(a.0)) as f64, (i128::from(b.1) - i128::from(a.1)) as f64);
    if dx == 0.0 && dy == 0.0 {
        emit(a.0, a.1, a.0);
        return;
    }
    let half = (width - 1) as f64 / 2.0;
    let max = half.round() / dx.hypot(dy);
    let min = round_down(half) as f64 / dx.hypot(dy);
    let (xmin, xmax, ymin, ymax) = (round_down(min * dy), round_down(max * dy), round_down(min * dx), round_down(max * dx));
    polygon_spans(
        &[
            (a.0.saturating_sub(xmin), a.1.saturating_add(ymax)),
            (b.0.saturating_sub(xmin), b.1.saturating_add(ymax)),
            (b.0.saturating_add(xmax), b.1.saturating_sub(ymin)),
            (a.0.saturating_add(xmax), a.1.saturating_sub(ymin)),
        ],
        height,
        blend,
        emit,
    );
}

#[pyfunction]
fn draw_polygon(
    image: &mut Image, points: Vec<(i64, i64)>, width: i64, filled: bool, closed: bool, value: &Bound<'_, PyAny>, blend: bool,
) -> PyResult<()> {
    let ink = super::draw::ink(image, value, blend)?;
    image.ensure_owned()?;
    if points.is_empty() || (!filled && width < 1) {
        return Ok(());
    }
    let (w, h) = (image.width() as usize, image.height());
    let mut mask = if closed && !filled && width > 1 { Some(blanket_core::raster::alloc_pixels(w * h as usize)?) } else { None };
    value.py().detach(|| {
        if !filled && width == 1 {
            let line = |image: &mut Image, a: (i64, i64), b: (i64, i64)| {
                super::draw::thin_line(image, (a.0 as f64, a.1 as f64), (b.0 as f64, b.1 as f64), &ink, blend)
            };
            for pair in points.windows(2) {
                line(image, pair[0], pair[1]);
            }
            if closed {
                line(image, *points.last().unwrap(), points[0]);
            } else {
                let p = *points.last().unwrap();
                super::draw::paint(image, p.0, p.1, &ink, blend);
            }
            return;
        }
        if let Some(mask) = &mut mask {
            polygon_spans(&points, h, false, |l, y, r| {
                if y >= 0 && y < i64::from(h) {
                    let (l, r) = (l.max(0) as usize, r.min(w as i64 - 1));
                    if r >= l as i64 {
                        mask[y as usize * w + l..=y as usize * w + r as usize].fill(1);
                    }
                }
            });
        }
        let mut emit = |l: i64, y: i64, r: i64| {
            if let Some(mask) = &mask {
                if y < 0 || y >= i64::from(h) {
                    return;
                }
                for x in l.max(0)..=r.min(w as i64 - 1) {
                    if mask[y as usize * w + x as usize] != 0 {
                        super::draw::paint(image, x, y, &ink, blend);
                    }
                }
            } else {
                super::draw::hline(image, l, r, y, &ink, blend);
            }
        };
        if filled {
            polygon_spans(&points, h, blend, emit);
        } else {
            let width = if mask.is_some() { width.saturating_mul(2) - 1 } else { width };
            for pair in points.windows(2) {
                wide_line(pair[0], pair[1], width, h, blend, &mut emit);
            }
            if closed {
                wide_line(*points.last().unwrap(), points[0], width, h, blend, &mut emit);
            }
        }
    });
    Ok(())
}
