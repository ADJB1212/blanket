//! APNG frame selection and delta planning.

use blanket_core::{Image, PixelMode};
use blanket_image_ops::convert::{convert, convert_palette_alpha, convert_transparent_alpha};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyInt};

type Bounds = (usize, usize, usize, usize);
type PlannedFrame = (usize, Option<Bounds>, f64, u8, u8);
type FramePlan = ((u32, u32), Vec<PlannedFrame>);

fn rgba(py: Python<'_>, image: &Image, transparency: Option<&Bound<'_, PyAny>>) -> PyResult<Image> {
    if image.palette().is_some() {
        if let Some(value) = transparency {
            if value.is_instance_of::<PyInt>() {
                return convert_palette_alpha(py, image, "RGBA", None, Some(value.extract()?), None);
            }
            if value.is_instance_of::<PyBytes>() {
                return convert_palette_alpha(py, image, "RGBA", Some(value.extract()?), None, None);
            }
        }
    } else if matches!(image.pixel_mode(), PixelMode::L | PixelMode::Rgb)
        && let Some(value) = transparency
    {
        let key = if value.is_instance_of::<PyInt>() { vec![value.extract()?] } else { value.extract()? };
        return convert_transparent_alpha(py, image, "RGBA", key);
    }
    convert(py, image, "RGBA", None)
}

fn is_opaque(py: Python<'_>, image: &Image, has_transparency_key: bool) -> PyResult<bool> {
    if has_transparency_key {
        return Ok(false);
    }
    if let Some((mode, _)) = image.palette() {
        return Ok(*mode == PixelMode::Rgb);
    }
    let mode = image.pixel_mode();
    if matches!(mode, PixelMode::La | PixelMode::Rgba) {
        let pixels = image.pixel_data()?;
        let channels = mode.channels();
        return Ok(py.detach(|| pixels.chunks_exact(channels).all(|pixel| pixel[channels - 1] == 255)));
    }
    Ok(true)
}

#[pyfunction]
pub(crate) fn _png_frame_plan(
    py: Python<'_>, images: Vec<PyRef<'_, Image>>, durations: Vec<f64>, disposals: Vec<u8>, blends: Vec<u8>,
    transparencies: Vec<Option<Bound<'_, PyAny>>>, default_image: bool, duration_known: bool,
) -> PyResult<FramePlan> {
    let start = usize::from(default_image);
    if images.len() <= start || [durations.len(), disposals.len(), blends.len(), transparencies.len()].iter().any(|&n| n != images.len()) {
        return Err(PyValueError::new_err("invalid frame arrays"));
    }
    if images.iter().any(|image| image.bit_depth() != 8) {
        return Err(PyValueError::new_err("APNG delta planning requires 8-bit frames"));
    }
    let canvas = images.iter().fold((0, 0), |(w, h), image| (w.max(image.width()), h.max(image.height())));
    let mut records: Vec<PlannedFrame> = Vec::new();
    let mut comparison_base: Option<(usize, Option<Bounds>)> = None;
    for (position, frame) in images.iter().enumerate().skip(start) {
        let mut bbox = None;
        let (duration, disposal, blend) = (durations[position], disposals[position], blends[position]);
        let replaces_pixels = blend == 0 || is_opaque(py, frame, transparencies[position].is_some())?;
        if let Some((base_index, clear)) = comparison_base
            // Preserve Pillow-compatible delta bounds for differently sized frames.
            && (frame.size() != images[base_index].size() || replaces_pixels)
        {
            let base = &images[base_index];
            let clear = clear.map(|(l, t, r, b)| (l as u32, t as u32, r as u32, b as u32));
            let direct = frame.pixel_mode() == base.pixel_mode()
                && frame.palette().is_none()
                && base.palette().is_none()
                && (clear.is_none() || matches!(frame.pixel_mode(), PixelMode::La | PixelMode::Rgba))
                && transparencies[position].is_none()
                && transparencies[base_index].is_none();
            if direct {
                bbox = blanket_composite::chops::_difference_bbox(py, frame, base, clear)?;
            } else {
                let current = rgba(py, frame, transparencies[position].as_ref())?;
                let comparison = rgba(py, base, transparencies[base_index].as_ref())?;
                bbox = blanket_composite::chops::_difference_bbox(py, &current, &comparison, clear)?;
            }
            if bbox.is_none()
                && frame.size() == base.size()
                && duration_known
                && disposal == 0
                && let Some((_, _, previous_duration, 0, previous_blend)) = records.last_mut()
                && *previous_blend == blend
                && blanket_codecs::codecs::frame_delay(*previous_duration + duration).is_ok()
            {
                *previous_duration += duration;
                continue;
            }
        }
        // PREVIOUS leaves the pre-frame canvas intact, even across consecutive frames.
        if disposal == 1 || (disposal == 2 && records.is_empty()) {
            let (width, height) = if records.is_empty() && !default_image { canvas } else { frame.size() };
            comparison_base = Some((position, Some(bbox.unwrap_or((0, 0, width as usize, height as usize)))));
        } else if disposal == 0 {
            comparison_base = replaces_pixels.then_some((position, None));
        }
        records.push((position, bbox, duration, disposal, blend));
    }
    Ok((canvas, records))
}
