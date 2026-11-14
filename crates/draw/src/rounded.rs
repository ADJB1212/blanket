//! Pillow-compatible rounded rectangle decomposition.

use blanket_core::Image;
use pyo3::prelude::*;

type Bounds = (f64, f64, f64, f64);

enum Part {
    Rectangle(Bounds),
    Curve(Bounds, f32, f32),
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(draw_rounded_rectangle, module)?)
}

fn bounds(values: Bounds) -> PyResult<(i64, i64, i64, i64)> {
    use crate::geometry::integer;
    Ok((integer(values.0)?, integer(values.1)?, integer(values.2)?, integer(values.3)?))
}

#[pyfunction]
fn draw_rounded_rectangle(
    image: &mut Image, original: Bounds, radius: f64, width: i64, corners: [bool; 4], filled: bool, value: &Bound<'_, PyAny>, blend: bool,
) -> PyResult<()> {
    let (x0, y0, x1, y1) = original;
    let mut diameter = (x1 - x0).min(y1 - y0).min(radius * 2.0);
    let (x0, y0, x1, y1) = (x0.round_ties_even(), y0.round_ties_even(), x1.round_ties_even(), y1.round_ties_even());
    let (mut full_x, mut full_y) = (false, false);
    if corners.iter().all(|&enabled| enabled) {
        full_x = diameter >= x1 - x0 - 1.0;
        if full_x {
            diameter = x1 - x0;
        }
        full_y = diameter >= y1 - y0 - 1.0;
        if full_y {
            diameter = y1 - y0;
        }
        if full_x && full_y {
            return crate::draw::draw_ellipse(image, bounds(original)?, value, if filled { 0 } else { width }, blend, false);
        }
    }
    if diameter == 0.0 || !corners.iter().any(|&enabled| enabled) {
        return crate::draw::draw_rectangle(image, bounds(original)?, value, if filled { 0 } else { width }, blend);
    }
    let r = (diameter / 2.0).floor();
    let mut parts = if full_x {
        vec![Part::Curve((x0, y0, x0 + diameter, y0 + diameter), 180.0, 360.0), Part::Curve((x0, y1 - diameter, x0 + diameter, y1), 0.0, 180.0)]
    } else if full_y {
        vec![Part::Curve((x0, y0, x0 + diameter, y0 + diameter), 90.0, 270.0), Part::Curve((x1 - diameter, y0, x1, y0 + diameter), 270.0, 90.0)]
    } else {
        [
            ((x0, y0, x0 + diameter, y0 + diameter), 180.0, 270.0),
            ((x1 - diameter, y0, x1, y0 + diameter), 270.0, 360.0),
            ((x1 - diameter, y1 - diameter, x1, y1), 0.0, 90.0),
            ((x0, y1 - diameter, x0 + diameter, y1), 90.0, 180.0),
        ]
        .into_iter()
        .zip(corners)
        .filter_map(|((bounds, start, end), enabled)| enabled.then_some(Part::Curve(bounds, start, end)))
        .collect()
    };
    let corner = |i: usize| if corners[i] { r + 1.0 } else { 0.0 };
    if filled {
        if full_x {
            parts.push(Part::Rectangle((x0, y0 + r + 1.0, x1, y1 - r - 1.0)));
        } else if x1 - r - 1.0 >= x0 + r + 1.0 {
            parts.push(Part::Rectangle((x0 + r + 1.0, y0, x1 - r - 1.0, y1)));
        }
        if !full_x && !full_y {
            parts.push(Part::Rectangle((x0, y0 + corner(0), x0 + r, y1 - corner(3))));
            parts.push(Part::Rectangle((x1 - r, y0 + corner(1), x1, y1 - corner(2))));
        }
    } else {
        let width = width as f64;
        if !full_x {
            parts.push(Part::Rectangle((x0 + corner(0), y0, x1 - corner(1), y0 + width - 1.0)));
            parts.push(Part::Rectangle((x0 + corner(3), y1 - width + 1.0, x1 - corner(2), y1)));
        }
        if !full_y {
            parts.push(Part::Rectangle((x0, y0 + corner(0), x0 + width - 1.0, y1 - corner(3))));
            parts.push(Part::Rectangle((x1 - width + 1.0, y0 + corner(1), x1, y1 - corner(2))));
        }
    }
    for part in parts {
        match part {
            Part::Rectangle(rect) => crate::draw::draw_rectangle(image, bounds(rect)?, value, 0, blend)?,
            Part::Curve(rect, start, end) => crate::pillow::draw_curve(
                image,
                bounds(rect)?,
                start,
                end,
                if filled { 0 } else { width },
                if filled { "pie" } else { "arc" },
                filled,
                value,
                blend,
            )?,
        }
    }
    Ok(())
}
