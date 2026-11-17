use std::fmt::Write as _;

use blanket_codec_common::{CodecErrorKind, SaveOptions};
use blanket_core::{Image, PixelMode};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rustc_hash::FxHashSet;
use visioncortex::color_clusters::{HIERARCHICAL_MAX, KeyingAction, Runner, RunnerConfig};
use visioncortex::{Color, ColorImage, PathSimplifyMode};

use crate::writer::encode_path;

pub fn encode(image: &Image, options: SaveOptions<'_>) -> PyResult<Vec<u8>> {
    let mode = match options.svg_mode {
        "pixel" => PathSimplifyMode::None,
        "polygon" => PathSimplifyMode::Polygon,
        "spline" => PathSimplifyMode::Spline,
        _ => return Err(PyValueError::new_err("svg_mode must be pixel, polygon, or spline")),
    };
    if !(1..=8).contains(&options.svg_color_precision)
        || options.svg_filter_speckle > 4096
        || options.svg_layer_difference > 765
        || options.svg_path_precision > 8
        || options.svg_alpha_threshold == 0
        || !options.svg_corner_threshold.is_finite()
        || !(0.0..=180.0).contains(&options.svg_corner_threshold)
        || !options.svg_splice_threshold.is_finite()
        || !(0.0..=180.0).contains(&options.svg_splice_threshold)
        || !options.svg_length_threshold.is_finite()
        || options.svg_length_threshold < 1.0
    {
        return Err(PyValueError::new_err("invalid SVG tracing options"));
    }
    if image.bit_depth() != 8 || !matches!(image.pixel_mode(), PixelMode::L | PixelMode::La | PixelMode::Rgb | PixelMode::Rgba) {
        return Err(CodecErrorKind::Unsupported.into_pyerr("SVG tracing requires 8-bit L, LA, RGB, or RGBA".into(), Some("SVG"), "encode"));
    }
    let (width, height) = image.size();
    if width == 0 || height == 0 {
        return Err(PyValueError::new_err("cannot trace an empty image"));
    }
    if width > 4096 || height > 4096 || u64::from(width) * u64::from(height) > 4 << 20 {
        return Err(CodecErrorKind::ResourceLimit.into_pyerr("SVG tracing exceeds 4096 per axis or 4194304 pixels".into(), Some("SVG"), "encode"));
    }
    let mut pixels = Vec::with_capacity(width as usize * height as usize * 4);
    for pixel in image.pixel_data()?.chunks_exact(image.pixel_mode().channels()) {
        let rgba = match image.pixel_mode() {
            PixelMode::L => [pixel[0], pixel[0], pixel[0], 255],
            PixelMode::La => [pixel[0], pixel[0], pixel[0], pixel[1]],
            PixelMode::Rgb => [pixel[0], pixel[1], pixel[2], 255],
            PixelMode::Rgba => [pixel[0], pixel[1], pixel[2], pixel[3]],
            _ => unreachable!(),
        };
        pixels.extend_from_slice(&rgba);
    }
    let key = transparency_key(&mut pixels, options.svg_alpha_threshold);
    let clusters = Runner::new(
        RunnerConfig {
            diagonal: options.svg_layer_difference == 0,
            hierarchical: HIERARCHICAL_MAX,
            batch_size: 25600,
            good_min_area: options.svg_filter_speckle as usize * options.svg_filter_speckle as usize,
            good_max_area: width as usize * height as usize,
            is_same_color_a: i32::from(8 - options.svg_color_precision),
            is_same_color_b: 1,
            deepen_diff: options.svg_layer_difference as i32,
            hollow_neighbours: 1,
            key_color: key,
            keying_action: KeyingAction::Discard,
        },
        ColorImage { pixels, width: width as usize, height: height as usize },
    )
    .run();
    let view = clusters.view();
    let mut output = format!("<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\" viewBox=\"0 0 {width} {height}\">\n");
    for &index in view.clusters_output.iter().rev() {
        let cluster = view.get_cluster(index);
        // Smoothing a single pixel can collapse its contour to zero area.
        let fitting = if cluster.area() == 1 { PathSimplifyMode::None } else { mode };
        let path = cluster.to_compound_path(
            &view,
            false,
            fitting,
            options.svg_corner_threshold.to_radians(),
            options.svg_length_threshold,
            10,
            options.svg_splice_threshold.to_radians(),
        );
        let data = encode_path(&path, options.svg_path_precision);
        if !data.is_empty() {
            writeln!(output, "<path fill=\"{}\" d=\"{data}\"/>", cluster.residue_color().to_hex_string()).unwrap();
        }
    }
    output.push_str("</svg>\n");
    Ok(output.into_bytes())
}

fn transparency_key(pixels: &mut [u8], threshold: u8) -> Color {
    if !pixels.as_chunks::<4>().0.iter().any(|pixel| pixel[3] < threshold) {
        return Color::default();
    }
    let used: FxHashSet<u32> =
        pixels.as_chunks::<4>().0.iter().filter(|p| p[3] >= threshold).map(|p| u32::from_be_bytes([0, p[0], p[1], p[2]])).collect();
    // The pixel limit guarantees an unused color in the 24-bit RGB cube.
    let value = (0..=used.len() as u32).find(|v| !used.contains(v)).unwrap();
    let [_, r, g, b] = value.to_be_bytes();
    let key = Color::new(r, g, b);
    for pixel in pixels.as_chunks_mut::<4>().0 {
        if pixel[3] < threshold {
            pixel.copy_from_slice(&[r, g, b, 255]);
        } else {
            pixel[3] = 255;
        }
    }
    key
}
