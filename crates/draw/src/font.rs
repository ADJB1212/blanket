//! Outline font shaping and coverage masks without FreeType.

use std::collections::HashMap;

use ab_glyph::{Font, FontVec, Glyph, GlyphId, Outline, OutlinedGlyph, PxScale, ScaleFont, point};
use pyo3::exceptions::{PyOSError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use rustybuzz::{Direction, Feature, UnicodeBuffer};
use unicode_bidi::{BidiInfo, Level};
use unicode_script::{Script, UnicodeScript};

type Bounds = (i32, i32, i32, i32);
type MaskResult = (f32, Bounds, Py<PyBytes>);

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeFont>()?;
    module.add_function(wrap_pyfunction!(_font_stroke, module)?)?;
    Ok(())
}

#[pyclass(module = "blanket._blanket", frozen)]
struct NativeFont {
    font: FontVec,
    index: u32,
}

#[pymethods]
impl NativeFont {
    #[new]
    fn new(data: &[u8], index: u32) -> PyResult<Self> {
        let font = FontVec::try_from_vec_and_index(data.to_vec(), index).map_err(|_| PyOSError::new_err("invalid or unsupported font"))?;
        if rustybuzz::Face::from_slice(font.font_data(), index).is_none() {
            return Err(PyOSError::new_err("invalid font face"));
        }
        Ok(Self { font, index })
    }

    fn metrics(&self, size: f32) -> PyResult<(f32, f32)> {
        self.validate_size(size)?;
        let factor = size / self.font.units_per_em().unwrap();
        Ok((self.font.ascent_unscaled() * factor, -self.font.descent_unscaled() * factor))
    }

    fn name(&self) -> (String, String) {
        let face = rustybuzz::Face::from_slice(self.font.font_data(), self.index).unwrap();
        let find = |id| {
            face.names()
                .into_iter()
                .filter(|name| name.name_id == id)
                .find_map(|name| name.to_string())
                .unwrap_or_default()
        };
        (find(1), find(2))
    }

    #[pyo3(signature = (text, size, direction, features, language, render, start, advance_only=false))]
    #[allow(clippy::too_many_arguments)]
    fn layout(
        &self, py: Python<'_>, text: &str, size: f32, direction: Option<&str>, features: Vec<String>, language: Option<&str>, render: bool,
        start: (f32, f32), advance_only: bool,
    ) -> PyResult<MaskResult> {
        self.validate_size(size)?;
        if !matches!(direction, None | Some("ltr" | "rtl" | "ttb")) {
            return Err(PyValueError::new_err("direction must be ltr, rtl, or ttb"));
        }
        if !start.0.is_finite() || !start.1.is_finite() || start.0.abs() > 1.0 || start.1.abs() > 1.0 {
            return Err(PyValueError::new_err("start must contain fractional pixel offsets"));
        }
        let features: Vec<Feature> = features
            .iter()
            .map(|value| value.parse().map_err(|_| PyValueError::new_err("invalid OpenType feature")))
            .collect::<PyResult<_>>()?;
        let language = language
            .map(|value| value.parse().map_err(|_| PyValueError::new_err("invalid language tag")))
            .transpose()?;
        let (advance, bounds, data) = py.detach(|| {
            let face = rustybuzz::Face::from_slice(self.font.font_data(), self.index).unwrap();
            let factor = size / face.units_per_em() as f32;
            let scale = PxScale::from(size * self.font.height_unscaled() / face.units_per_em() as f32);
            let scale_factor = self.font.as_scaled(scale).scale_factor();
            let vertical = direction == Some("ttb");
            let base_level = direction.and_then(|value| match value {
                "ltr" => Some(Level::ltr()),
                "rtl" => Some(Level::rtl()),
                _ => None,
            });
            let bidi = BidiInfo::new(text, base_level);
            let mut segments = Vec::new();
            if vertical {
                segments.extend(script_runs(text).into_iter().map(|range| (range, Direction::TopToBottom)));
            } else {
                for paragraph in &bidi.paragraphs {
                    let (levels, runs) = bidi.visual_runs(paragraph, paragraph.range.clone());
                    for range in runs {
                        let rtl = levels[range.start].is_rtl();
                        let mut scripts = script_runs(&text[range.clone()]);
                        if rtl {
                            scripts.reverse();
                        }
                        segments.extend(scripts.into_iter().map(|script| {
                            (
                                range.start + script.start..range.start + script.end,
                                if rtl { Direction::RightToLeft } else { Direction::LeftToRight },
                            )
                        }));
                    }
                }
            }
            let mut pen = point(0.0, 0.0);
            let mut glyphs = Vec::new();
            let mut glyph_outlines: HashMap<GlyphId, Option<Outline>> = HashMap::new();
            let mut ink: Option<Bounds> = None;
            let mut buffer = UnicodeBuffer::new();
            for (range, direction) in segments {
                buffer.push_str(&text[range]);
                buffer.set_direction(direction);
                if let Some(language) = language.clone() {
                    buffer.set_language(language);
                }
                buffer.guess_segment_properties();
                let shaped = rustybuzz::shape(&face, &features, buffer);
                for (info, position) in shaped.glyph_infos().iter().zip(shaped.glyph_positions()) {
                    let glyph = Glyph {
                        id: GlyphId(info.glyph_id as u16),
                        scale,
                        position: point(
                            pen.x + position.x_offset as f32 * factor + start.0,
                            pen.y - position.y_offset as f32 * factor + start.1,
                        ),
                    };
                    if !advance_only && let Some(outline) = glyph_outlines.entry(glyph.id).or_insert_with(|| self.font.outline(glyph.id)) {
                        let rect = outline.px_bounds(scale_factor, glyph.position);
                        if [rect.min.x, rect.min.y, rect.max.x, rect.max.y]
                            .iter()
                            .any(|value| !value.is_finite() || value.abs() > 100_000_000.0)
                        {
                            return Err(PyValueError::new_err("glyph bounds are too large"));
                        }
                        let bounds = (rect.min.x as i32, rect.min.y as i32, rect.max.x as i32, rect.max.y as i32);
                        ink = Some(match ink {
                            None => bounds,
                            Some(old) => (old.0.min(bounds.0), old.1.min(bounds.1), old.2.max(bounds.2), old.3.max(bounds.3)),
                        });
                        if render {
                            glyphs.push(glyph);
                        }
                    }
                    pen.x += position.x_advance as f32 * factor;
                    pen.y -= position.y_advance as f32 * factor;
                    if !pen.x.is_finite() || !pen.y.is_finite() || pen.x.abs().max(pen.y.abs()) > 100_000_000.0 {
                        return Err(PyValueError::new_err("text layout is too large"));
                    }
                }
                buffer = shaped.clear();
            }
            let advance = if vertical { pen.y } else { pen.x };
            let mut bounds = ink.unwrap_or((0, 0, 0, 0));
            if vertical {
                bounds.1 = bounds.1.min(0);
                bounds.3 = bounds.3.max(advance.ceil() as i32);
            } else {
                bounds.0 = bounds.0.min(0);
                bounds.2 = bounds.2.max(advance.ceil() as i32);
            }
            let mut data = if render { allocate_mask(bounds)? } else { Vec::new() };
            if render {
                let width = (bounds.2 - bounds.0) as usize;
                for glyph in glyphs {
                    let raw_outline = glyph_outlines[&glyph.id].as_ref().unwrap().clone();
                    let outline = OutlinedGlyph::new(glyph, raw_outline, scale_factor);
                    let rect = outline.px_bounds();
                    let x = (rect.min.x as i32 - bounds.0) as usize;
                    let y = (rect.min.y as i32 - bounds.1) as usize;
                    outline.draw(|gx, gy, coverage| {
                        let dst = &mut data[(y + gy as usize) * width + x + gx as usize];
                        let value = (coverage * 255.0).round() as u8;
                        *dst = (u32::from(*dst) + u32::from(value) - (u32::from(*dst) * u32::from(value) + 127) / 255) as u8;
                    });
                }
            }
            Ok((advance, bounds, data))
        })?;
        Ok((advance, bounds, PyBytes::new(py, &data).unbind()))
    }
}

impl NativeFont {
    fn validate_size(&self, size: f32) -> PyResult<()> {
        if !size.is_finite() || size <= 0.0 || size > 1_000_000.0 {
            return Err(PyValueError::new_err("font size must be finite, positive, and at most 1000000"));
        }
        Ok(())
    }
}

fn script_runs(text: &str) -> Vec<std::ops::Range<usize>> {
    let mut runs = Vec::new();
    let mut start = 0;
    let mut script = Script::Common;
    for (offset, ch) in text.char_indices() {
        let next = ch.script();
        if matches!(next, Script::Common | Script::Inherited | Script::Unknown) {
            continue;
        }
        if script != Script::Common && script != next {
            runs.push(start..offset);
            start = offset;
        }
        script = next;
    }
    if start < text.len() {
        runs.push(start..text.len());
    }
    runs
}

fn allocate_mask(bounds: Bounds) -> PyResult<Vec<u8>> {
    let width = i64::from(bounds.2) - i64::from(bounds.0);
    let height = i64::from(bounds.3) - i64::from(bounds.1);
    if width < 0 || height < 0 {
        return Err(PyValueError::new_err("invalid font mask bounds"));
    }
    let length = width
        .checked_mul(height)
        .filter(|&length| (0..=100_000_000).contains(&length))
        .ok_or_else(|| PyValueError::new_err("font mask is too large"))?;
    Ok(vec![0; length as usize])
}

#[pyfunction]
fn _font_stroke(py: Python<'_>, data: &[u8], size: (usize, usize), stroke: usize) -> PyResult<Py<PyBytes>> {
    let (width, height) = size;
    if width.checked_mul(height) != Some(data.len()) || stroke > 1024 || width > 100_000_000 || height > 100_000_000 {
        return Err(PyValueError::new_err("invalid font mask or stroke size"));
    }
    let output_width = width + 2 * stroke;
    let output_height = height + 2 * stroke;
    let mut output = allocate_mask((0, 0, output_width as i32, output_height as i32))?;
    if width == 0 || height == 0 {
        return Ok(PyBytes::new(py, &output).unbind());
    }
    py.detach(|| stroke_mask(data, width, stroke, &mut output));
    Ok(PyBytes::new(py, &output).unbind())
}

fn stroke_mask(data: &[u8], width: usize, stroke: usize, output: &mut [u8]) {
    if stroke == 0 {
        output.copy_from_slice(data);
        return;
    }
    let stride = width + 2 * stroke;
    let mut horizontal = vec![0; stride];
    let mut span = stroke;
    let mut spans = Vec::with_capacity(stroke + 1);
    for dy in 0..=stroke {
        while span * span + dy * dy > stroke * stroke {
            span -= 1;
        }
        spans.push(span);
    }
    for (y, source) in data.chunks_exact(width).enumerate() {
        let mut previous_span = None;
        for (dy, &span) in spans.iter().enumerate() {
            let row_length = width + 2 * span;
            if previous_span != Some(span) {
                crate::font_simd::horizontal_max(source, span, &mut horizontal[..row_length]);
                previous_span = Some(span);
            }
            let start = (y + stroke - dy) * stride + stroke - span;
            crate::font_simd::max_assign(&mut output[start..start + row_length], &horizontal[..row_length]);
            if dy != 0 {
                let start = (y + stroke + dy) * stride + stroke - span;
                crate::font_simd::max_assign(&mut output[start..start + row_length], &horizontal[..row_length]);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_scripts_at_utf8_boundaries_and_keeps_marks() {
        let text = "Latin a\u{301} Ελληνικά Русский";
        let runs = script_runs(text);
        assert_eq!(
            runs.iter().map(|range| &text[range.clone()]).collect::<Vec<_>>(),
            ["Latin a\u{301} ", "Ελληνικά ", "Русский"]
        );
        assert!(script_runs("").is_empty());
    }

    #[test]
    fn rejects_oversized_masks_without_allocating() {
        assert!(allocate_mask((0, 0, 100_000_000, 2)).is_err());
        assert!(allocate_mask((i32::MIN, 0, i32::MAX, 2)).is_err());
        assert_eq!(allocate_mask((0, 0, 0, 0)).unwrap(), Vec::<u8>::new());
    }

    #[test]
    fn disk_strokes_match_scalar_reference_at_vector_boundaries() {
        for width in [1, 15, 16, 17, 31, 32, 33, 63, 65] {
            for height in [1, 3, 9] {
                let input: Vec<u8> = (0..width * height + 3).map(|i| (i * 73 + i / 7) as u8).collect();
                let source = &input[3..];
                for radius in [0, 1, 2, 4, 5, 8, 17] {
                    let stride = width + 2 * radius;
                    let length = stride * (height + 2 * radius);
                    let mut expected = vec![0; length];
                    for sy in 0..height {
                        for sx in 0..width {
                            for dy in 0..=2 * radius {
                                for dx in 0..=2 * radius {
                                    if dx.abs_diff(radius).pow(2) + dy.abs_diff(radius).pow(2) <= radius * radius {
                                        let dst = &mut expected[(sy + dy) * stride + sx + dx];
                                        *dst = (*dst).max(source[sy * width + sx]);
                                    }
                                }
                            }
                        }
                    }
                    let mut actual = vec![87; length + 10];
                    actual[5..5 + length].fill(0);
                    stroke_mask(source, width, radius, &mut actual[5..5 + length]);
                    assert_eq!(&actual[5..5 + length], expected, "{width}x{height}, radius {radius}");
                    assert_eq!(&actual[..5], &[87; 5]);
                    assert_eq!(&actual[5 + length..], &[87; 5]);
                }
            }
        }
    }
}
