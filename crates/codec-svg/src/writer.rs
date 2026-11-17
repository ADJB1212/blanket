use visioncortex::{CompoundPath, CompoundPathElement, PointF64};

pub(crate) fn encode_path(path: &CompoundPath, precision: u8) -> String {
    let mut emitter = Emitter::new(precision);
    let factor = 10_f64.powi(i32::from(precision));
    let point = |p: PointF64| PointF64 { x: (p.x * factor).round() / factor, y: (p.y * factor).round() / factor };
    for element in path.iter() {
        match element {
            CompoundPathElement::PathI32(path) => {
                if let Some(first) = path.path.first() {
                    emitter.move_to(point(PointF64 { x: f64::from(first.x), y: f64::from(first.y) }));
                    for p in &path.path[1..] {
                        emitter.line_to(point(PointF64 { x: f64::from(p.x), y: f64::from(p.y) }));
                    }
                    emitter.close();
                }
            }
            CompoundPathElement::PathF64(path) => {
                if let Some(&first) = path.path.first() {
                    emitter.move_to(point(first));
                    for &p in &path.path[1..] {
                        emitter.line_to(point(p));
                    }
                    emitter.close();
                }
            }
            CompoundPathElement::Spline(spline) => {
                if let Some(&first) = spline.points.first() {
                    emitter.move_to(point(first));
                    for cubic in spline.points[1..].as_chunks::<3>().0 {
                        emitter.cubic_to(point(cubic[0]), point(cubic[1]), point(cubic[2]));
                    }
                    emitter.close();
                }
            }
        }
    }
    emitter.finish()
}

struct Emitter {
    precision: u8,
    out: String,
    cur: PointF64,
    subpath_start: PointF64,
    started: bool,
    prev_cubic_c2: Option<PointF64>,
}

impl Emitter {
    fn new(precision: u8) -> Self {
        Self { precision, out: String::new(), cur: PointF64::default(), subpath_start: PointF64::default(), started: false, prev_cubic_c2: None }
    }

    fn finish(self) -> String {
        self.out
    }

    fn close(&mut self) {
        self.out.push('Z');
        self.cur = self.subpath_start;
        self.prev_cubic_c2 = None;
    }

    fn move_to(&mut self, p: PointF64) {
        if !self.started {
            let token = format!("M{}", self.coord(p));
            self.out.push_str(&token);
            self.started = true;
        } else {
            let abs = format!("M{}", self.coord(p));
            let rel = format!("m{}", self.coord_delta(p));
            let token = shorter(abs, rel);
            self.out.push_str(&token);
        }
        self.cur = p;
        self.subpath_start = p;
        self.prev_cubic_c2 = None;
    }

    fn line_to(&mut self, p: PointF64) {
        let mut candidates: Vec<String> = Vec::new();

        if p.y == self.cur.y {
            candidates.push(format!("H{}", self.num(p.x)));
            candidates.push(format!("h{}", self.num(p.x - self.cur.x)));
        }
        if p.x == self.cur.x {
            candidates.push(format!("V{}", self.num(p.y)));
            candidates.push(format!("v{}", self.num(p.y - self.cur.y)));
        }

        candidates.push(format!("L{}", self.coord(p)));
        candidates.push(format!("l{}", self.coord_delta(p)));

        self.out.push_str(&shortest(candidates));
        self.cur = p;
        self.prev_cubic_c2 = None;
    }

    fn cubic_to(&mut self, c1: PointF64, c2: PointF64, e: PointF64) {
        let mut candidates: Vec<String> = Vec::new();

        if let Some(prev_c2) = self.prev_cubic_c2 {
            let reflection = PointF64 { x: 2.0 * self.cur.x - prev_c2.x, y: 2.0 * self.cur.y - prev_c2.y };
            if reflection == c1 {
                candidates.push(format!("S{}", self.coord_list(&[c2, e])));
                candidates.push(format!("s{}", self.delta_list(&[c2, e])));
            }
        }

        candidates.push(format!("C{}", self.coord_list(&[c1, c2, e])));
        candidates.push(format!("c{}", self.delta_list(&[c1, c2, e])));

        self.out.push_str(&shortest(candidates));
        self.cur = e;
        self.prev_cubic_c2 = Some(c2);
    }

    fn num(&self, v: f64) -> String {
        fmt_num(v, self.precision)
    }

    fn coord(&self, p: PointF64) -> String {
        join_nums(&[self.num(p.x), self.num(p.y)])
    }

    fn coord_delta(&self, p: PointF64) -> String {
        join_nums(&[self.num(p.x - self.cur.x), self.num(p.y - self.cur.y)])
    }

    fn coord_list(&self, pts: &[PointF64]) -> String {
        let mut nums = Vec::with_capacity(pts.len() * 2);
        for p in pts {
            nums.push(self.num(p.x));
            nums.push(self.num(p.y));
        }
        join_nums(&nums)
    }

    fn delta_list(&self, pts: &[PointF64]) -> String {
        let mut nums = Vec::with_capacity(pts.len() * 2);
        for p in pts {
            nums.push(self.num(p.x - self.cur.x));
            nums.push(self.num(p.y - self.cur.y));
        }
        join_nums(&nums)
    }
}

fn shorter(a: String, b: String) -> String {
    if b.len() < a.len() { b } else { a }
}

fn shortest(candidates: Vec<String>) -> String {
    candidates.into_iter().min_by_key(|s| s.len()).unwrap_or_default()
}

fn join_nums(nums: &[String]) -> String {
    let mut s = String::new();
    for (i, n) in nums.iter().enumerate() {
        if i > 0 && !n.starts_with('-') {
            s.push(',');
        }
        s.push_str(n);
    }
    s
}

fn fmt_num(v: f64, precision: u8) -> String {
    let factor = 10_f64.powi(i32::from(precision));
    let v = (v * factor).round() / factor;
    if v == 0.0 {
        return "0".to_string();
    }

    let mut s = format!("{:.*}", usize::from(precision), v);

    if s.contains('.') {
        while s.ends_with('0') {
            s.pop();
        }
        if s.ends_with('.') {
            s.pop();
        }
    }

    if let Some(rest) = s.strip_prefix("0.") {
        s = format!(".{rest}");
    } else if let Some(rest) = s.strip_prefix("-0.") {
        s = format!("-.{rest}");
    }

    s
}

#[cfg(test)]
mod tests {
    use visioncortex::PathF64;

    use super::*;

    #[test]
    fn relative_moves_after_close_use_subpath_start() {
        let mut emitter = Emitter::new(2);
        emitter.move_to(PointF64 { x: 1000.0, y: 1000.0 });
        emitter.line_to(PointF64 { x: 2000.0, y: 2000.0 });
        emitter.close();
        emitter.move_to(PointF64 { x: 1001.0, y: 1002.0 });
        assert!(emitter.finish().ends_with("Zm1,2"));
    }

    #[test]
    fn rounding_absolute_points_prevents_relative_drift() {
        let mut path = PathF64::new();
        for i in 0..100 {
            path.add(PointF64 { x: 1000.0 + f64::from(i) * 0.004, y: 1000.0 });
        }
        let mut compound = CompoundPath::new();
        compound.add_path_f64(path);
        let data = encode_path(&compound, 2);
        let mut x = 0.0;
        for command in svgtypes::PathParser::from(data.as_str()) {
            match command.unwrap() {
                svgtypes::PathSegment::MoveTo { x: px, .. } => x = px,
                svgtypes::PathSegment::HorizontalLineTo { abs, x: px } => x = if abs { px } else { x + px },
                svgtypes::PathSegment::ClosePath { .. } => break,
                other => panic!("unexpected path command {other:?}"),
            }
        }
        assert!((x - 1000.4).abs() < 1e-9);
    }

    #[test]
    fn smooth_cubics_require_reflection_and_reset_after_lines() {
        let p = |x, y| PointF64 { x, y };
        let mut emitter = Emitter::new(2);
        emitter.move_to(p(1000.0, 1000.0));
        emitter.cubic_to(p(1001.0, 1000.0), p(1002.0, 1001.0), p(1003.0, 1001.0));
        emitter.cubic_to(p(1004.0, 1001.0), p(1005.0, 1002.0), p(1006.0, 1002.0));
        assert!(emitter.out.ends_with("s2,1,3,1"));
        emitter.line_to(p(1007.0, 1002.0));
        emitter.cubic_to(p(1008.0, 1002.0), p(1009.0, 1003.0), p(1010.0, 1003.0));
        assert!(emitter.finish().ends_with("c1,0,2,1,3,1"));
    }
}
