use blanket_codec_common::{SaveOptions, codec_error};
use blanket_core::{Image, PixelMode};
use libjpeg_turbo_rs::{Encoder, PixelFormat, Subsampling};
use pyo3::exceptions::{PyOSError, PyValueError};
use pyo3::prelude::*;

use super::JpegCoding;

pub(super) fn configured(image: &Image, pixels: &[u8], options: SaveOptions, coding: JpegCoding) -> PyResult<Vec<u8>> {
    if options.jpeg_qtables.len() > 4 {
        return Err(PyValueError::new_err("invalid JPEG quantization table count"));
    }
    let keep_rgb = options.jpeg_keep_rgb && image.pixel_mode() == PixelMode::Rgb;
    if keep_rgb && !matches!(options.jpeg_subsampling, -1 | 0) {
        return Err(PyOSError::new_err("RGB JPEG does not support chroma subsampling"));
    }
    encode_components(image, pixels, options, coding)
}

fn encode_components(image: &Image, pixels: &[u8], options: SaveOptions, coding: JpegCoding) -> PyResult<Vec<u8>> {
    use libjpeg_turbo_rs::{read_coefficients, write_coefficients, write_coefficients_optimized, write_coefficients_progressive};

    let (width, height) = (image.width() as usize, image.height() as usize);
    if width == 0 || height == 0 || width > 65500 || height > 65500 {
        return Err(PyOSError::new_err("invalid JPEG dimensions"));
    }
    let cmyk = image.pixel_mode() == PixelMode::Cmyk;
    let rgb = image.pixel_mode() == PixelMode::Rgb && options.jpeg_keep_rgb;
    let gray = image.pixel_mode() == PixelMode::L;
    if !matches!(image.pixel_mode(), PixelMode::L | PixelMode::Rgb | PixelMode::YCbCr | PixelMode::Cmyk) {
        return Err(PyOSError::new_err(format!("cannot write mode {} as JPEG", image.pixel_mode().as_str())));
    }
    let (horizontal, vertical) = if rgb || ((cmyk || gray) && options.jpeg_subsampling == -1) {
        (1, 1)
    } else {
        match options.jpeg_subsampling {
            0 => (1, 1),
            1 => (2, 1),
            _ => (2, 2),
        }
    };
    let planes = smooth_planes(image, pixels, horizontal, vertical, options.jpeg_smooth, rgb);
    let mut result = None;
    for (index, plane) in planes.iter().enumerate() {
        let table_index = if options.jpeg_qtables.is_empty() {
            usize::from(index > 0 && !(cmyk || rgb || gray))
        } else {
            index.min(options.jpeg_qtables.len() - 1)
        };
        let (h, v) = if index == 0 || gray { (1, 1) } else { (horizontal, vertical) };
        let (plane_width, plane_height) = (width.div_ceil(h).next_multiple_of(8), height.div_ceil(v).next_multiple_of(8));
        let mut encoder = Encoder::new(plane, plane_width, plane_height, PixelFormat::Grayscale)
            .subsampling(Subsampling::S444)
            .quality(options.quality)
            .force_baseline(options.jpeg_qtables.is_empty());
        if let Some(table) = options.jpeg_qtables.get(table_index) {
            encoder = encoder.quant_table(0, scale_table(*table, options.jpeg_qtable_quality));
        } else if table_index == 1 {
            encoder = encoder.quant_table(0, chroma_table(options.quality));
        }
        let encoded = encoder.encode().map_err(codec_error)?;
        let mut coefficients = read_coefficients(&encoded).map_err(codec_error)?;
        let quant_table = coefficients.quant_tables[0];
        let mut component = coefficients.components.remove(0);
        component.component_id = if cmyk {
            b"CMYK"[index]
        } else if rgb {
            b"RGB"[index]
        } else {
            index as u8 + 1
        };
        component.quant_table_index = table_index as u8;
        if index == 0 {
            component.h_sampling = horizontal as u8;
            component.v_sampling = vertical as u8;
            let blocks_x = component.blocks_x.next_multiple_of(horizontal);
            let blocks_y = component.blocks_y.next_multiple_of(vertical);
            if (blocks_x, blocks_y) != (component.blocks_x, component.blocks_y) {
                let mut blocks = vec![[0; 64]; blocks_x * blocks_y];
                for row in 0..component.blocks_y {
                    blocks[row * blocks_x..row * blocks_x + component.blocks_x]
                        .copy_from_slice(&component.blocks[row * component.blocks_x..(row + 1) * component.blocks_x]);
                }
                component.blocks = blocks;
                component.blocks_x = blocks_x;
                component.blocks_y = blocks_y;
            }
        }
        let output = result.get_or_insert_with(|| {
            coefficients.width = image.width() as u16;
            coefficients.height = image.height() as u16;
            coefficients.saw_jfif_marker = !(cmyk || rgb);
            coefficients.adobe_transform = (cmyk || rgb).then_some(0);
            coefficients.quant_tables.clear();
            coefficients
        });
        if output.quant_tables.len() <= table_index {
            output.quant_tables.push(quant_table);
        }
        output.components.push(component);
    }
    let mut coefficients = result.expect("JPEG modes have components");
    coefficients.restart_interval = if options.jpeg_restart_marker_rows != 0 {
        (u64::from(options.jpeg_restart_marker_rows) * width.div_ceil(8 * if gray { 1 } else { horizontal }) as u64).min(65535) as u16
    } else {
        options.jpeg_restart_marker_blocks as u16
    };
    if coding.progressive {
        let rows = (options.jpeg_restart_marker_rows != 0).then_some(options.jpeg_restart_marker_rows.min(65535) as u16);
        write_coefficients_progressive(&coefficients, rows).map_err(codec_error)
    } else if coding.optimize {
        write_coefficients_optimized(&coefficients).map_err(codec_error)
    } else {
        write_coefficients(&coefficients).map_err(codec_error)
    }
}

fn scale_table(table: [u16; 64], quality: i16) -> [u16; 64] {
    if quality < 0 {
        return table.map(|value| value.clamp(1, 32767));
    }
    let scale = libjpeg_turbo_rs::quality_scaling(quality.clamp(1, 100) as u8);
    table.map(|value| ((u32::from(value) * scale + 50) / 100).clamp(1, 255) as u16)
}

fn chroma_table(quality: u8) -> [u16; 64] {
    let base = [
        17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99, 24, 26, 56, 99, 99, 99, 99, 99, 47, 66, 99, 99, 99, 99, 99, 99, 99, 99, 99,
        99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
    ];
    let scale = libjpeg_turbo_rs::quality_scaling(quality);
    base.map(|value| ((value * scale + 50) / 100).clamp(1, 255) as u16)
}

fn smooth_planes(image: &Image, pixels: &[u8], horizontal: usize, vertical: usize, factor: i32, keep_rgb: bool) -> Vec<Vec<u8>> {
    let (width, height) = (image.width() as usize, image.height() as usize);
    let channels = image.pixel_mode().channels();
    let mut planes = vec![Vec::with_capacity(width * height); channels];
    for pixel in pixels.chunks_exact(channels) {
        if image.pixel_mode() == PixelMode::Rgb && !keep_rgb {
            let [r, g, b] = [i32::from(pixel[0]), i32::from(pixel[1]), i32::from(pixel[2])];
            let values = [
                blanket_core::simd::pillow_luma(pixel[0], pixel[1], pixel[2]),
                ((-11059 * r - 21709 * g + 32768 * b + 8421375) >> 16) as u8,
                ((32768 * r - 27439 * g - 5329 * b + 8421375) >> 16) as u8,
            ];
            for (plane, value) in planes.iter_mut().zip(values) {
                plane.push(value);
            }
        } else {
            for (plane, &sample) in planes.iter_mut().zip(pixel) {
                plane.push(if image.pixel_mode() == PixelMode::Cmyk { 255 - sample } else { sample });
            }
        }
    }
    planes
        .into_iter()
        .enumerate()
        .map(|(index, plane)| {
            let (h, v) = if index == 0 || channels == 1 { (1, 1) } else { (horizontal, vertical) };
            smooth_plane(&plane, width, height, h, v, factor)
        })
        .collect()
}

fn smooth_plane(plane: &[u8], width: usize, height: usize, horizontal: usize, vertical: usize, factor: i32) -> Vec<u8> {
    let output_width = width.div_ceil(horizontal).next_multiple_of(8);
    let output_height = height.div_ceil(vertical).next_multiple_of(8);
    let sample = |x: isize, y: isize| i64::from(plane[y.clamp(0, height as isize - 1) as usize * width + x.clamp(0, width as isize - 1) as usize]);
    let factor = i64::from(factor);
    let mut output = vec![0; output_width * output_height];
    for (row, destination) in output.chunks_exact_mut(output_width).enumerate() {
        let y = (row * vertical) as isize;
        for (column, value) in destination.iter_mut().enumerate() {
            let x = (column * horizontal) as isize;
            let result = if horizontal == 1 {
                let center = sample(x, y);
                let mut neighbors = -center;
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        neighbors += sample(x + dx, y + dy);
                    }
                }
                (center * (65536 - factor * 512) + neighbors * factor * 64 + 32768) >> 16
            } else if vertical == 1 {
                (sample(x, y) + sample(x + 1, y) + (column as i64 & 1)) >> 1
            } else {
                let members = sample(x, y) + sample(x + 1, y) + sample(x, y + 1) + sample(x + 1, y + 1);
                if factor == 0 {
                    *value = ((members + 1 + (column as i64 & 1)) >> 2) as u8;
                    continue;
                }
                let edges = sample(x, y - 1)
                    + sample(x + 1, y - 1)
                    + sample(x, y + 2)
                    + sample(x + 1, y + 2)
                    + sample(x - 1, y)
                    + sample(x + 2, y)
                    + sample(x - 1, y + 1)
                    + sample(x + 2, y + 1);
                let corners = sample(x - 1, y - 1) + sample(x + 2, y - 1) + sample(x - 1, y + 2) + sample(x + 2, y + 2);
                (members * (16384 - factor * 80) + (edges * 2 + corners) * factor * 16 + 32768) >> 16
            };
            *value = result as u8;
        }
    }
    output
}
