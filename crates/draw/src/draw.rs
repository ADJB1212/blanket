//! Native primitives used by the Python `ImageDraw` API.

use blanket_core::raster::{Image, PixelMode};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(draw_points, module)?)?;
    module.add_function(wrap_pyfunction!(draw_rectangle, module)?)?;
    module.add_function(wrap_pyfunction!(draw_path, module)?)?;
    module.add_function(wrap_pyfunction!(draw_ellipse, module)?)?;
    module.add_function(wrap_pyfunction!(draw_mask, module)?)?;
    module.add_function(wrap_pyfunction!(raster_font_mask, module)?)?;
    module.add_function(wrap_pyfunction!(draw_floodfill, module)?)?;
    Ok(())
}

fn ink(image: &Image, value: &Bound<'_, PyAny>, blend: bool) -> PyResult<Vec<u8>> {
    if blend && image.mode != PixelMode::Rgb {
        return Err(PyValueError::new_err("mode mismatch"));
    }
    image.raw_data()?;
    if !image.mode.is_wide_scalar() {
        image.pixel_data()?;
    }
    let mode = if blend { PixelMode::Rgba } else { image.mode };
    let mut pixel = Image {
        width: 1,
        height: 1,
        mode,
        pixels: Some(vec![0; mode.channels() * mode.sample_bytes()]),
        bit_depth: image.bit_depth,
        format: None,
        palette: None,
    };
    pixel.putpixel((0, 0), value)?;
    Ok(pixel.pixels.unwrap())
}

fn paint(image: &mut Image, x: i64, y: i64, ink: &[u8], blend: bool) {
    if x < 0 || y < 0 || x >= i64::from(image.width) || y >= i64::from(image.height) {
        return;
    }
    let stride = image.mode.channels() * image.mode.sample_bytes();
    let offset = (y as usize * image.width as usize + x as usize) * stride;
    let pixel = &mut image.pixels.as_mut().unwrap()[offset..offset + stride];
    if blend {
        let alpha = u32::from(ink[3]);
        for (dst, src) in pixel.iter_mut().zip(ink) {
            *dst = ((u32::from(*dst) * (255 - alpha) + u32::from(*src) * alpha + 127) / 255) as u8;
        }
    } else {
        pixel.copy_from_slice(ink);
    }
}

#[pyfunction]
fn draw_points(image: &mut Image, points: Vec<(i64, i64)>, value: &Bound<'_, PyAny>, blend: bool) -> PyResult<()> {
    let mut ink = ink(image, value, blend)?;
    // Pillow points use native byte order even for explicit-endian modes.
    if (cfg!(target_endian = "little") && image.mode == PixelMode::I16B)
        || (cfg!(target_endian = "big") && matches!(image.mode, PixelMode::I16 | PixelMode::I16L))
    {
        ink.reverse();
    }
    for (x, y) in points {
        paint(image, x, y, &ink, blend);
    }
    Ok(())
}

#[pyfunction]
fn draw_rectangle(image: &mut Image, bounds: (i64, i64, i64, i64), value: &Bound<'_, PyAny>, width: i64, blend: bool) -> PyResult<()> {
    let ink = ink(image, value, blend)?;
    let (left, top, right, bottom) = bounds;
    if width == 0 {
        for y in top.max(0)..=bottom.min(i64::from(image.height) - 1) {
            hline(image, left, right, y, &ink, blend);
        }
    } else {
        for i in 0..width {
            hline(image, left, right, top.saturating_add(i), &ink, blend);
            hline(image, left, right, bottom.saturating_sub(i), &ink, blend);
            let start = top.saturating_add(width);
            let end = bottom.saturating_sub(width).saturating_add(1);
            let (first, last) = if start < end { (start, end - 1) } else { (end.saturating_add(1), start) };
            for y in first.max(0)..=last.min(i64::from(image.height) - 1) {
                paint(image, right.saturating_sub(i), y, &ink, blend);
                paint(image, left.saturating_add(i), y, &ink, blend);
            }
        }
    }
    Ok(())
}

fn hline(image: &mut Image, left: i64, right: i64, y: i64, ink: &[u8], blend: bool) {
    if y < 0 || y >= i64::from(image.height) {
        return;
    }
    for x in left.max(0)..=right.min(i64::from(image.width) - 1) {
        paint(image, x, y, ink, blend);
    }
}

fn coverage_paint(image: &mut Image, x: i64, y: i64, ink: &[u8], blend: bool, coverage: u8) {
    if coverage == 255 {
        paint(image, x, y, ink, blend);
    } else if coverage != 0 && x >= 0 && y >= 0 && x < i64::from(image.width) && y < i64::from(image.height) {
        let stride = image.mode.channels();
        let offset = (y as usize * image.width as usize + x as usize) * stride;
        let pixel = &mut image.pixels.as_mut().unwrap()[offset..offset + stride];
        let alpha = if blend {
            (u32::from(coverage) * u32::from(ink[3]) + 127) / 255
        } else {
            u32::from(coverage)
        };
        for (dst, src) in pixel.iter_mut().zip(ink) {
            *dst = ((u32::from(*dst) * (255 - alpha) + u32::from(*src) * alpha + 127) / 255) as u8;
        }
    }
}

fn check_coverage(image: &Image) -> PyResult<()> {
    if image.mode.is_wide_scalar() || image.palette.is_some() || matches!(image.mode, PixelMode::Pa | PixelMode::One) {
        return Err(PyValueError::new_err("anti-aliasing requires an 8-bit non-indexed image"));
    }
    Ok(())
}

fn distance_squared(point: (f64, f64), a: (f64, f64), b: (f64, f64)) -> f64 {
    let dx = b.0 - a.0;
    let dy = b.1 - a.1;
    let length = dx * dx + dy * dy;
    let t = if length == 0.0 {
        0.0
    } else {
        (((point.0 - a.0) * dx + (point.1 - a.1) * dy) / length).clamp(0.0, 1.0)
    };
    (point.0 - a.0 - t * dx).powi(2) + (point.1 - a.1 - t * dy).powi(2)
}

fn contains(points: &[(f64, f64)], point: (f64, f64)) -> bool {
    let mut inside = false;
    let mut previous = points[points.len() - 1];
    for &current in points {
        if distance_squared(point, previous, current) < 1e-12 {
            return true;
        }
        if (current.1 > point.1) != (previous.1 > point.1)
            && point.0 < (previous.0 - current.0) * (point.1 - current.1) / (previous.1 - current.1) + current.0
        {
            inside = !inside;
        }
        previous = current;
    }
    inside
}

fn thin_line(image: &mut Image, a: (f64, f64), b: (f64, f64), ink: &[u8], blend: bool) {
    let dx = b.0 - a.0;
    let dy = b.1 - a.1;
    let horizontal = dx.abs() > dy.abs();
    let (start, delta, minor, limit) = if horizontal {
        (a.0, dx, dy, image.width)
    } else {
        (a.1, dy, dx, image.height)
    };
    if delta == 0.0 {
        return;
    }
    let length = delta.abs();
    let sign = delta.signum();
    for coordinate in 0..limit {
        let step = (f64::from(coordinate) - start) * sign;
        if step < 0.0 || step >= length {
            continue;
        }
        let offset = (step * minor.abs() / length + 0.5).floor() * minor.signum();
        let point = if horizontal {
            (i64::from(coordinate), (a.1 + offset) as i64)
        } else {
            ((a.0 + offset) as i64, i64::from(coordinate))
        };
        paint(image, point.0, point.1, ink, blend);
    }
}

#[pyfunction]
fn draw_ellipse(image: &mut Image, bounds: (i64, i64, i64, i64), value: &Bound<'_, PyAny>, width: i64, blend: bool, antialias: bool) -> PyResult<()> {
    let ink = ink(image, value, blend)?;
    if antialias {
        check_coverage(image)?;
    }
    let (left, top, right, bottom) = bounds;
    let rx = (right as f64 - left as f64 + 1.0) / 2.0;
    let ry = (bottom as f64 - top as f64 + 1.0) / 2.0;
    let cx = (right as f64 + left as f64) / 2.0;
    let cy = (bottom as f64 + top as f64) / 2.0;
    let inner_x = rx - width as f64;
    let inner_y = ry - width as f64;
    let samples = if antialias { 4 } else { 1 };
    for y in top.max(0)..=bottom.min(i64::from(image.height) - 1) {
        for x in left.max(0)..=right.min(i64::from(image.width) - 1) {
            let mut covered = 0;
            for sy in 0..samples {
                for sx in 0..samples {
                    let dx = x as f64 + (f64::from(sx) + 0.5) / f64::from(samples) - 0.5 - cx;
                    let dy = y as f64 + (f64::from(sy) + 0.5) / f64::from(samples) - 0.5 - cy;
                    let outer = (dx / rx).powi(2) + (dy / ry).powi(2) <= 1.0;
                    let inner = width != 0 && inner_x > 0.0 && inner_y > 0.0 && (dx / inner_x).powi(2) + (dy / inner_y).powi(2) <= 1.0;
                    covered += i32::from(outer && !inner);
                }
            }
            coverage_paint(
                image,
                x,
                y,
                &ink,
                blend,
                ((covered * 255 + samples * samples / 2) / (samples * samples)) as u8,
            );
        }
    }
    Ok(())
}

#[pyfunction]
fn draw_path(
    image: &mut Image, points: Vec<(f64, f64)>, value: &Bound<'_, PyAny>, width: f64, style: &str, blend: bool, antialias: bool,
) -> PyResult<()> {
    let (closed, filled) = match style {
        "fill" => (true, true),
        "outline" => (true, false),
        "line" => (false, false),
        _ => return Err(PyValueError::new_err("invalid path style")),
    };
    let ink = ink(image, value, blend)?;
    if antialias {
        check_coverage(image)?;
    }
    if !width.is_finite() || points.iter().any(|&(x, y)| !x.is_finite() || !y.is_finite()) {
        return Err(PyValueError::new_err("coordinates and width must be finite"));
    }
    if points.is_empty() || (!filled && width <= 0.0) {
        return Ok(());
    }
    if !filled && !closed && !antialias && width == 1.0 && points.iter().all(|&(x, y)| x.fract() == 0.0 && y.fract() == 0.0) {
        for pair in points.windows(2) {
            thin_line(image, pair[0], pair[1], &ink, blend);
        }
        let last = points[points.len() - 1];
        paint(image, last.0 as i64, last.1 as i64, &ink, blend);
        return Ok(());
    }
    let radius = width.max(1.0) / 2.0;
    let left = points.iter().map(|p| p.0).fold(f64::INFINITY, f64::min) - radius;
    let right = points.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max) + radius;
    let top = points.iter().map(|p| p.1).fold(f64::INFINITY, f64::min) - radius;
    let bottom = points.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max) + radius;
    let samples = if antialias { 4 } else { 1 };
    for y in (top.floor() as i64).max(0)..=(bottom.ceil() as i64).min(i64::from(image.height) - 1) {
        for x in (left.floor() as i64).max(0)..=(right.ceil() as i64).min(i64::from(image.width) - 1) {
            let mut covered = 0;
            for sy in 0..samples {
                for sx in 0..samples {
                    let point = (
                        x as f64 + (f64::from(sx) + 0.5) / f64::from(samples) - 0.5,
                        y as f64 + (f64::from(sy) + 0.5) / f64::from(samples) - 0.5,
                    );
                    let hit = if filled {
                        contains(&points, point)
                    } else {
                        points
                            .windows(2)
                            .any(|pair| distance_squared(point, pair[0], pair[1]) < radius * radius + 1e-12)
                            || (closed && distance_squared(point, points[points.len() - 1], points[0]) < radius * radius + 1e-12)
                    };
                    covered += i32::from(hit);
                }
            }
            coverage_paint(
                image,
                x,
                y,
                &ink,
                blend,
                ((covered * 255 + samples * samples / 2) / (samples * samples)) as u8,
            );
        }
    }
    Ok(())
}

#[pyfunction]
fn draw_mask(
    image: &mut Image, position: (i64, i64), size: (usize, usize), mask: Vec<u8>, value: &Bound<'_, PyAny>, blend: bool, stroke: usize,
) -> PyResult<()> {
    let ink = ink(image, value, blend)?;
    if size.0.checked_mul(size.1) != Some(mask.len()) {
        return Err(PyValueError::new_err("invalid mask size"));
    }
    if mask.iter().any(|&sample| sample != 0 && sample != 255) {
        check_coverage(image)?;
    }
    let stroke = i64::try_from(stroke).map_err(|_| PyValueError::new_err("stroke is too large"))?;
    for y in (-stroke).max(position.1.saturating_neg())
        ..(size.1 as i64)
            .saturating_add(stroke)
            .min(i64::from(image.height).saturating_sub(position.1))
    {
        for x in (-stroke).max(position.0.saturating_neg())
            ..(size.0 as i64)
                .saturating_add(stroke)
                .min(i64::from(image.width).saturating_sub(position.0))
        {
            let mut coverage = 0;
            for my in y.saturating_sub(stroke).max(0)..=y.saturating_add(stroke).min(size.1 as i64 - 1) {
                for mx in x.saturating_sub(stroke).max(0)..=x.saturating_add(stroke).min(size.0 as i64 - 1) {
                    if (mx as f64 - x as f64).powi(2) + (my as f64 - y as f64).powi(2) <= (stroke as f64).powi(2) {
                        coverage = coverage.max(mask[my as usize * size.0 + mx as usize]);
                    }
                }
            }
            if image.mode == PixelMode::Rgba && coverage != 0 {
                let offset = ((position.1 + y) as usize * image.width as usize + (position.0 + x) as usize) * 4;
                let dst = &image.pixels.as_ref().unwrap()[offset..offset + 4];
                let mut pixel = [0; 4];
                for channel in 0..4 {
                    let weight = if channel < 3 && dst[3] == 0 { 255 } else { u32::from(coverage) };
                    pixel[channel] = ((u32::from(ink[channel]) * weight + u32::from(dst[channel]) * (255 - weight) + 127) / 255) as u8;
                }
                paint(image, position.0.saturating_add(x), position.1.saturating_add(y), &pixel, false);
            } else {
                coverage_paint(image, position.0.saturating_add(x), position.1.saturating_add(y), &ink, blend, coverage);
            }
        }
    }
    Ok(())
}

#[pyfunction]
fn raster_font_mask(glyphs: Vec<Vec<u8>>, scale: usize) -> PyResult<(usize, usize, Vec<u8>)> {
    if scale == 0 || glyphs.iter().any(|rows| rows.len() != 7) {
        return Err(PyValueError::new_err("invalid bitmap glyph or scale"));
    }
    let width = glyphs
        .len()
        .checked_mul(6)
        .and_then(|value| value.checked_mul(scale))
        .ok_or_else(|| PyValueError::new_err("font mask is too large"))?;
    let height = if glyphs.is_empty() {
        0
    } else {
        7usize.checked_mul(scale).ok_or_else(|| PyValueError::new_err("font mask is too large"))?
    };
    let length = width
        .checked_mul(height)
        .filter(|&length| length <= 100_000_000)
        .ok_or_else(|| PyValueError::new_err("font mask is too large"))?;
    let mut mask = vec![0; length];
    for (i, rows) in glyphs.iter().enumerate() {
        for (y, &row) in rows.iter().enumerate() {
            for x in 0..5 {
                if row & (1 << (4 - x)) != 0 {
                    for dy in 0..scale {
                        let offset = (y * scale + dy) * width + (i * 6 + x) * scale;
                        mask[offset..offset + scale].fill(255);
                    }
                }
            }
        }
    }
    Ok((width, height, mask))
}

fn sample_distance(mode: PixelMode, a: &[u8], b: &[u8]) -> f64 {
    match mode {
        PixelMode::I => (f64::from(i32::from_ne_bytes(a.try_into().unwrap())) - f64::from(i32::from_ne_bytes(b.try_into().unwrap()))).abs(),
        PixelMode::F => (f64::from(f32::from_ne_bytes(a.try_into().unwrap())) - f64::from(f32::from_ne_bytes(b.try_into().unwrap()))).abs(),
        PixelMode::I16 | PixelMode::I16L => {
            (f64::from(u16::from_le_bytes(a.try_into().unwrap())) - f64::from(u16::from_le_bytes(b.try_into().unwrap()))).abs()
        }
        PixelMode::I16B => (f64::from(u16::from_be_bytes(a.try_into().unwrap())) - f64::from(u16::from_be_bytes(b.try_into().unwrap()))).abs(),
        _ => a.iter().zip(b).map(|(&x, &y)| (f64::from(x) - f64::from(y)).abs()).sum(),
    }
}

#[pyfunction]
fn draw_floodfill(
    image: &mut Image, position: (i64, i64), value: &Bound<'_, PyAny>, border: Option<&Bound<'_, PyAny>>, threshold: f64,
) -> PyResult<()> {
    let fill = ink(image, value, false)?;
    let border = border.map(|value| ink(image, value, false)).transpose()?;
    let (x, y) = position;
    if x < 0 || y < 0 || x >= i64::from(image.width) || y >= i64::from(image.height) {
        return Ok(());
    }
    let stride = image.mode.channels() * image.mode.sample_bytes();
    let seed_index = y as usize * image.width as usize + x as usize;
    let pixels = image.pixels.as_mut().unwrap();
    let seed = pixels[seed_index * stride..(seed_index + 1) * stride].to_vec();
    if sample_distance(image.mode, &seed, &fill) <= threshold {
        return Ok(());
    }
    let mut visited = vec![false; pixels.len() / stride];
    let mut pending = vec![seed_index];
    while let Some(index) = pending.pop() {
        if visited[index] {
            continue;
        }
        visited[index] = true;
        let pixel = &mut pixels[index * stride..(index + 1) * stride];
        let eligible = match &border {
            Some(border) => pixel != border && pixel != fill,
            None => sample_distance(image.mode, pixel, &seed) <= threshold,
        };
        if !eligible {
            continue;
        }
        pixel.copy_from_slice(&fill);
        let width = image.width as usize;
        if index % width > 0 {
            pending.push(index - 1);
        }
        if index % width + 1 < width {
            pending.push(index + 1);
        }
        if index >= width {
            pending.push(index - width);
        }
        if index + width < visited.len() {
            pending.push(index + width);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn polygon_includes_edges_and_uses_even_odd_fill() {
        let square = [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)];
        assert!(contains(&square, (0.0, 2.0)));
        assert!(contains(&square, (4.0, 4.0)));
        assert!(contains(&square, (2.0, 2.0)));
        assert!(!contains(&square, (5.0, 2.0)));
        let twice: Vec<_> = square.iter().chain(&square).copied().collect();
        assert!(!contains(&twice, (2.0, 2.0)));
    }

    #[test]
    fn bitmap_masks_scale_without_touching_spacing() {
        let (width, height, mask) = raster_font_mask(vec![vec![16; 7], vec![1; 7]], 2).unwrap();
        assert_eq!((width, height), (24, 14));
        for row in mask.chunks_exact(width) {
            assert_eq!(&row[..2], &[255, 255]);
            assert!(row[2..20].iter().all(|&value| value == 0));
            assert_eq!(&row[20..22], &[255, 255]);
            assert_eq!(&row[22..], &[0, 0]);
        }
        assert_eq!(raster_font_mask(vec![], 1).unwrap(), (0, 0, vec![]));
    }

    #[test]
    fn flood_distance_retains_scalar_values() {
        assert_eq!(
            sample_distance(PixelMode::I, &i32::MIN.to_ne_bytes(), &i32::MAX.to_ne_bytes()),
            4_294_967_295.0
        );
        assert_eq!(sample_distance(PixelMode::I16B, &1000u16.to_be_bytes(), &1001u16.to_be_bytes()), 1.0);
        assert_eq!(sample_distance(PixelMode::F, &1.25f32.to_ne_bytes(), &1.5f32.to_ne_bytes()), 0.25);
        assert!(sample_distance(PixelMode::F, &f32::NAN.to_ne_bytes(), &1.0f32.to_ne_bytes()).is_nan());
    }
}
