//! Saliency-based placement of a fixed-aspect crop on a bounded thumbnail.

use blanket_core::{Image, PixelMode};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(ops_smart_crop, module)?)?;
    Ok(())
}

#[pyfunction]
fn ops_smart_crop(py: Python<'_>, image: &Image, fractions: (f64, f64)) -> PyResult<(f64, f64)> {
    let fractions_array: [f64; 2] = fractions.into();
    if image.pixel_mode() != PixelMode::Rgba
        || image.bit_depth() != 8
        || image.width() == 0
        || image.height() == 0
        || image.width() > 128
        || image.height() > 128
        || !fractions_array.iter().all(|v| v.is_finite() && *v > 0.0 && *v <= 1.0)
    {
        return Err(PyValueError::new_err("expected a nonempty 8-bit RGBA thumbnail up to 128x128 and crop fractions in (0, 1]"));
    }
    let pixels = image.pixel_data()?;
    let width = image.width() as usize;
    let height = image.height() as usize;
    Ok(py.detach(|| placement(pixels, width, height, fractions)))
}

fn placement(pixels: &[u8], width: usize, height: usize, fractions: (f64, f64)) -> (f64, f64) {
    let mut mean = [0.0; 3];
    let mut alpha_sum = 0.0;
    for pixel in pixels.as_chunks::<4>().0 {
        let alpha = f64::from(pixel[3]) / 255.0;
        alpha_sum += alpha;
        for c in 0..3 {
            mean[c] += f64::from(pixel[c]) * alpha;
        }
    }
    if alpha_sum == 0.0 {
        return (0.5, 0.5);
    }
    for value in &mut mean {
        *value /= alpha_sum;
    }
    let mean_alpha = alpha_sum / (width * height) as f64;
    let mut columns = vec![0.0; width];
    let mut rows = vec![0.0; height];
    for (y, row) in rows.iter_mut().enumerate() {
        for (x, column) in columns.iter_mut().enumerate() {
            let offset = (y * width + x) * 4;
            let pixel = &pixels[offset..offset + 4];
            let alpha = f64::from(pixel[3]) / 255.0;
            let contrast: f64 = (0..3).map(|c| (f64::from(pixel[c]) - mean[c]).abs()).sum();
            let mut edge = 0.0;
            for (nx, ny) in [(x.saturating_sub(1), y), ((x + 1).min(width - 1), y), (x, y.saturating_sub(1)), (x, (y + 1).min(height - 1))] {
                let neighbor = &pixels[(ny * width + nx) * 4..][..4];
                let neighbor_alpha = f64::from(neighbor[3]) / 255.0;
                edge += (0..3).map(|c| (f64::from(pixel[c]) - f64::from(neighbor[c])).abs()).sum::<f64>() * neighbor_alpha;
            }
            let score = alpha * (contrast + edge * 0.25 + 255.0 * (alpha - mean_alpha).abs());
            *column += score;
            *row += score;
        }
    }
    (axis_position(&columns, fractions.0), axis_position(&rows, fractions.1))
}

fn axis_position(scores: &[f64], fraction: f64) -> f64 {
    if fraction >= 1.0 || scores.windows(2).all(|pair| pair[0] == pair[1]) {
        return 0.5;
    }
    let extent = scores.len() as f64;
    let window = extent * fraction;
    let travel = extent - window;
    let score = |position: f64| {
        let center = position * travel + window * 0.5;
        scores
            .iter()
            .enumerate()
            .map(|(i, value)| {
                let distance = ((i as f64 + 0.5 - center) / (window.max(1.0) * 0.5)).abs();
                value * (1.0 - distance * distance).max(0.0)
            })
            .sum::<f64>()
    };
    let mut best_position = 0.5;
    let mut best_score = score(best_position);
    for step in 0..=128 {
        let position = f64::from(step) / 128.0;
        let candidate = score(position);
        if candidate > best_score || (candidate == best_score && (position - 0.5).abs() < (best_position - 0.5).abs()) {
            best_score = candidate;
            best_position = position;
        }
    }
    best_position
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_and_transparent_images_are_centered() {
        for (width, height) in [(1, 1), (1, 17), (31, 1), (128, 127)] {
            for pixel in [[70, 70, 70, 255], [255, 0, 128, 0]] {
                let pixels = pixel.repeat(width * height);
                assert_eq!(placement(&pixels, width, height, (0.2, 1.0)), (0.5, 0.5));
            }
        }
    }

    #[test]
    fn saliency_tracks_both_axes_and_ignores_hidden_rgb() {
        for (width, height, fractions) in [(31, 9, (0.25, 1.0)), (9, 31, (1.0, 0.25))] {
            let mut pixels = [0, 0, 0, 255].repeat(width * height);
            for y in 0..height {
                for x in 0..width {
                    if (width > height && x > width - 6) || (height > width && y > height - 6) {
                        pixels[(y * width + x) * 4..][..4].copy_from_slice(&[255, 0, 0, 255]);
                    }
                }
            }
            let position = placement(&pixels, width, height, fractions);
            assert!(if width > height { position.0 > 0.8 } else { position.1 > 0.8 });
        }
        let mut pixels = [0, 0, 0, 0].repeat(31);
        pixels[4..8].copy_from_slice(&[255, 255, 255, 255]);
        let expected = placement(&pixels, 31, 1, (0.25, 1.0));
        for pixel in pixels.as_chunks_mut::<4>().0.iter_mut().skip(2) {
            pixel[..3].copy_from_slice(&[255, 128, 255]);
        }
        assert_eq!(placement(&pixels, 31, 1, (0.25, 1.0)), expected);
        assert!(expected.0 < 0.1);
    }
}
