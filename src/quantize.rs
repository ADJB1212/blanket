//! Color palette generation and mapping. All pixel work runs without the GIL.
use std::collections::BTreeMap;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use crate::raster::{Image, PixelMode};

type Color = [u8; 4];
type Histogram = Vec<(Color, u64)>;

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(quantize, module)?)?;
    Ok(())
}

fn distance(a: Color, b: Color) -> u32 {
    a.iter().zip(b).map(|(&a, b)| (i32::from(a) - i32::from(b)).unsigned_abs().pow(2)).sum()
}

fn nearest(color: Color, palette: &[Color]) -> usize {
    palette
        .iter()
        .enumerate()
        .min_by_key(|(_, entry)| distance(color, **entry))
        .map_or(0, |(i, _)| i)
}

fn average(histogram: &[(Color, u64)]) -> Color {
    let count: u64 = histogram.iter().map(|(_, n)| n).sum();
    std::array::from_fn(|c| {
        let sum: u64 = histogram.iter().map(|(color, n)| u64::from(color[c]) * n).sum();
        ((sum + count / 2) / count.max(1)) as u8
    })
}

// Pillow's supplied-palette converter searches on a six-bit RGB grid.
fn palette_nearest(color: Color, palette: &[Color]) -> usize {
    nearest([color[0] & 252, color[1] & 252, color[2] & 252, 255], palette)
}

fn median_cut(histogram: Histogram, colors: usize) -> Vec<Color> {
    let mut boxes = vec![histogram];
    while boxes.len() < colors {
        let Some(i) = boxes
            .iter()
            .enumerate()
            .filter(|(_, b)| b.len() > 1)
            .max_by_key(|(_, b)| b.iter().map(|(_, n)| n).sum::<u64>())
            .map(|(i, _)| i)
        else {
            break;
        };
        let b = &mut boxes[i];
        let channel = (0..3)
            .max_by_key(|&c| {
                let low = b.iter().map(|(v, _)| v[c]).min().unwrap();
                let high = b.iter().map(|(v, _)| v[c]).max().unwrap();
                (high - low, 3 - c)
            })
            .unwrap();
        b.sort_by_key(|(v, _)| v[channel]);
        let half = b.iter().map(|(_, n)| n).sum::<u64>().div_ceil(2);
        let mut count = 0;
        let mut split = 1;
        for (j, (_, n)) in b.iter().enumerate().take(b.len() - 1) {
            count += n;
            split = j + 1;
            if count >= half {
                break;
            }
        }
        let right = b.split_off(split);
        boxes.push(right);
    }
    boxes.iter().map(|b| average(b)).collect()
}

fn maximum_coverage(histogram: &Histogram, colors: usize) -> Vec<Color> {
    let mean = average(histogram);
    let first = histogram.iter().max_by_key(|(color, _)| distance(*color, mean)).unwrap().0;
    let mut palette = vec![first];
    let mut distances = vec![u32::MAX; histogram.len()];
    while palette.len() < colors.min(histogram.len()) {
        let last = *palette.last().unwrap();
        for (d, (color, _)) in distances.iter_mut().zip(histogram) {
            *d = (*d).min(distance(*color, last));
        }
        let Some((i, &d)) = distances.iter().enumerate().max_by_key(|(_, d)| *d) else {
            break;
        };
        if d == 0 {
            break;
        }
        palette.push(histogram[i].0);
    }
    palette
}

// Merge the least-populated deepest octree branch until the palette fits.
fn octree(histogram: Histogram, colors: usize) -> Vec<Color> {
    let mut bins: BTreeMap<(u8, Color), Histogram> = histogram.into_iter().map(|entry| ((8, entry.0), vec![entry])).collect();
    for depth in (1..=8).rev() {
        if bins.len() <= colors {
            break;
        }
        let shift = 9 - depth;
        let mask = if shift == 8 { 0 } else { 255u8 << shift };
        let mut parents: BTreeMap<Color, (u64, Vec<(u8, Color)>)> = BTreeMap::new();
        for ((d, color), entries) in &bins {
            if *d == depth {
                let key = color.map(|v| v & mask);
                let group = parents.entry(key).or_default();
                group.0 += entries.iter().map(|(_, n)| n).sum::<u64>();
                group.1.push((*d, *color));
            }
        }
        let mut parents: Vec<_> = parents.into_iter().collect();
        parents.sort_by_key(|(parent, (count, _))| (*count, *parent));
        for (parent, (_, keys)) in parents {
            if bins.len() <= colors {
                break;
            }
            let mut merged = bins.remove(&(depth - 1, parent)).unwrap_or_default();
            for key in keys {
                merged.extend(bins.remove(&key).unwrap());
            }
            bins.insert((depth - 1, parent), merged);
        }
    }
    bins.values().map(|b| average(b)).collect()
}

fn refine(histogram: &Histogram, palette: &mut [Color], threshold: u64) {
    let mut previous = vec![usize::MAX; histogram.len()];
    loop {
        let mut sums = vec![[0u64; 4]; palette.len()];
        let mut counts = vec![0u64; palette.len()];
        let mut changed = 0u64;
        for ((color, count), old) in histogram.iter().zip(&mut previous) {
            let i = nearest(*color, palette);
            if i != *old {
                changed += count;
                *old = i;
            }
            counts[i] += count;
            for c in 0..4 {
                sums[i][c] += u64::from(color[c]) * count;
            }
        }
        for (i, entry) in palette.iter_mut().enumerate() {
            if counts[i] != 0 {
                *entry = std::array::from_fn(|c| ((sums[i][c] + counts[i] / 2) / counts[i]) as u8);
            }
        }
        if changed <= threshold {
            break;
        }
    }
}

#[pyfunction]
#[pyo3(signature = (image, colors, method, kmeans, palette=None, dither=3))]
fn quantize(py: Python<'_>, image: &Image, colors: usize, method: u8, kmeans: u64, palette: Option<&Image>, dither: i32) -> PyResult<Image> {
    let source = image.pixel_data()?;
    if !(1..=256).contains(&colors) {
        return Err(PyValueError::new_err("bad number of colors"));
    }
    if method > 3 {
        return Err(PyValueError::new_err("quantization error"));
    }
    if method == 3 && palette.is_none() {
        return Err(PyRuntimeError::new_err(
            "dependency required by this method was not enabled at compile time",
        ));
    }
    if image.mode == PixelMode::Rgba && method < 2 {
        return Err(PyValueError::new_err("only FASTOCTREE and LIBIMAGEQUANT support RGBA"));
    }
    if let Some(palette) = palette {
        palette.pixel_data()?;
    }
    py.detach(|| {
        let channels = image.mode.channels();
        let pixels: Vec<Color> = source
            .chunks_exact(channels)
            .map(|p| match image.mode {
                PixelMode::L => [p[0], p[0], p[0], 255],
                PixelMode::Rgb => [p[0], p[1], p[2], 255],
                PixelMode::Rgba => [p[0], p[1], p[2], p[3]],
            })
            .collect();
        let mut counts = BTreeMap::new();
        for &pixel in &pixels {
            *counts.entry(pixel).or_insert(0u64) += 1;
        }
        let histogram: Histogram = counts.into_iter().collect();
        let palette_mode = if image.mode == PixelMode::Rgba {
            PixelMode::Rgba
        } else {
            PixelMode::Rgb
        };
        let mut entries = if let Some(reference) = palette {
            let (mode, data) = reference
                .palette
                .as_ref()
                .ok_or_else(|| PyValueError::new_err("bad mode for palette image"))?;
            data.chunks_exact(mode.channels()).map(|v| [v[0], v[1], v[2], 255]).collect()
        } else if histogram.is_empty() {
            Vec::new()
        } else {
            match method {
                0 => median_cut(histogram.clone(), colors),
                1 => maximum_coverage(&histogram, colors),
                2 => octree(histogram.clone(), colors),
                _ => unreachable!(),
            }
        };
        if palette.is_some() && entries.is_empty() {
            entries.push([0, 0, 0, 255]);
        }
        if kmeans > 0 && method < 2 && palette.is_none() {
            refine(&histogram, &mut entries, kmeans);
        }
        let mut indices = Vec::with_capacity(pixels.len());
        if palette.is_some() && image.mode == PixelMode::L {
            indices.extend_from_slice(source);
        } else if palette.is_some() && dither != 0 && image.mode == PixelMode::Rgb {
            let width = image.width as usize;
            let mut errors = vec![[0i32; 3]; width + 2];
            for row in pixels.chunks(width.max(1)) {
                let mut next = vec![[0i32; 3]; width + 2];
                let mut blue_errors = [0; 2];
                for (x, &pixel) in row.iter().enumerate() {
                    let mut adjusted = pixel;
                    for c in 0..3 {
                        adjusted[c] = (i32::from(pixel[c]) + errors[x + 1][c] / 16).clamp(0, 255) as u8;
                    }
                    let i = palette_nearest(adjusted, &entries);
                    indices.push(i as u8);
                    for c in 0..3 {
                        let error = i32::from(adjusted[c]) - i32::from(entries[i][c]);
                        errors[x + 2][c] += error * 7;
                        next[x][c] += error * 3;
                        next[x + 1][c] += error * 5;
                        next[x + 2][c] += error;
                        if c == 2 {
                            blue_errors = [blue_errors[1], error];
                        }
                    }
                }
                // Preserve Pillow's right-edge carry convention: its final
                // three error slots are populated from the blue accumulators.
                next[width] = [5 * blue_errors[1] + blue_errors[0], blue_errors[1], blue_errors[1]];
                errors = next;
            }
        } else {
            let lookup: BTreeMap<Color, u8> = histogram
                .iter()
                .map(|(color, _)| {
                    (
                        *color,
                        if palette.is_some() {
                            palette_nearest(*color, &entries)
                        } else {
                            nearest(*color, &entries)
                        } as u8,
                    )
                })
                .collect();
            indices.extend(pixels.iter().map(|color| lookup[color]));
        }
        let mut result = Image::from_pixels(image.width, image.height, PixelMode::L, indices, None)?;
        result.palette = Some(if let Some(reference) = palette {
            reference.palette.clone().unwrap()
        } else {
            (
                palette_mode,
                entries.iter().flat_map(|v| v[..palette_mode.channels()].iter().copied()).collect(),
            )
        });
        Ok(result)
    })
}
