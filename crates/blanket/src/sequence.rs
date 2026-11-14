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
    for (position, frame) in images.iter().enumerate().skip(start) {
        let mut bbox = None;
        let (duration, disposal, blend) = (durations[position], disposals[position], blends[position]);
        if let Some(&(previous, previous_bbox, previous_duration, previous_disposal, previous_blend)) = records.last() {
            let previous_disposal = if previous_disposal == 2 && records.len() < 2 { 1 } else { previous_disposal };
            let base_index = if previous_disposal == 2 { records[records.len() - 2].0 } else { previous };
            let base = &images[base_index];
            let mut clear = if previous_disposal == 1 {
                Some(previous_bbox.unwrap_or_else(|| (0, 0, images[0].width() as usize, images[0].height() as usize)))
            } else {
                None
            };
            let direct = frame.pixel_mode() == base.pixel_mode()
                && frame.palette().is_none()
                && base.palette().is_none()
                && matches!(frame.pixel_mode(), PixelMode::One | PixelMode::L | PixelMode::La | PixelMode::Rgb | PixelMode::Rgba)
                && transparencies[position].is_none()
                && transparencies[base_index].is_none();
            let bounds = |clear: Option<Bounds>| clear.map(|(l, t, r, b)| (l as u32, t as u32, r as u32, b as u32));
            if direct {
                bbox = blanket_composite::chops::_difference_bbox(py, frame, base, bounds(clear))?;
            } else {
                let mut cleared;
                let comparison_base = if let Some((left, top, right, bottom)) = clear
                    && base.pixel_mode() != PixelMode::Rgba
                {
                    cleared = base.copy(py)?;
                    cleared.ensure_owned()?;
                    let channels = base.pixel_mode().channels();
                    let width = base.width() as usize;
                    let height = base.height() as usize;
                    let pixels = cleared.pixels_mut().expect("owned pixel storage");
                    py.detach(|| {
                        for y in top.min(height)..bottom.min(height) {
                            let row = &mut pixels[y * width * channels..(y + 1) * width * channels];
                            row[left.min(width) * channels..right.min(width) * channels].fill(0);
                        }
                    });
                    clear = None;
                    &cleared
                } else {
                    base
                };
                let current = rgba(py, frame, transparencies[position].as_ref())?;
                let comparison = rgba(py, comparison_base, transparencies[base_index].as_ref())?;
                bbox = blanket_composite::chops::_difference_bbox(py, &current, &comparison, bounds(clear))?;
            }
            let combined = previous_duration + duration;
            if bbox.is_none()
                && previous_disposal == disposal
                && previous_blend == blend
                && duration_known
                && blanket_codecs::codecs::frame_delay(combined).is_ok()
            {
                records.last_mut().expect("previous frame").2 = combined;
                continue;
            }
        }
        records.push((position, bbox, duration, disposal, blend));
    }
    Ok((canvas, records))
}
