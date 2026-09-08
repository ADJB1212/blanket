//! Color palette generation and mapping. All pixel work runs without the GIL.
use std::cmp::Reverse;
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

use rayon::prelude::*;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use crate::parallel::{CHUNK_PIXELS, MIN_PARALLEL_BYTES, chunks_mut};
use crate::quantize_simd::PaletteSearch;
use crate::raster::{Image, PixelMode};

type Color = [u8; 4];
type Histogram = Vec<(Color, u64)>;

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(quantize, module)?)?;
    Ok(())
}

/// Packed colors are already well-distributed 32-bit keys. One multiply and
/// fold is far cheaper than SipHash for the millions of lookups a photo needs.
#[derive(Default)]
struct ColorHasher(u64);

impl Hasher for ColorHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.write_u64(u64::from(byte));
        }
    }

    fn write_u32(&mut self, n: u32) {
        self.write_u64(u64::from(n));
    }

    fn write_u64(&mut self, n: u64) {
        let mixed = (self.0 ^ n).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        // hashbrown indexes with the low bits and tags with the top seven.
        self.0 = mixed ^ (mixed >> 32);
    }
}

type ColorMap<V> = HashMap<u32, V, BuildHasherDefault<ColorHasher>>;

fn key(color: Color) -> u32 {
    u32::from_ne_bytes(color)
}

/// Expand an L, RGB, or RGBA pixel to the RGBA color used for palette work.
#[inline(always)]
fn color_of<const C: usize>(pixel: &[u8; C]) -> Color {
    std::array::from_fn(|c| {
        if c == 3 {
            pixel.get(3).copied().unwrap_or(255)
        } else {
            pixel[c.min(C - 1)]
        }
    })
}

fn distance(a: Color, b: Color) -> u32 {
    a.iter().zip(b).map(|(&a, b)| (i32::from(a) - i32::from(b)).unsigned_abs().pow(2)).sum()
}

fn average(histogram: &[(Color, u64)]) -> Color {
    let count: u64 = histogram.iter().map(|(_, n)| n).sum();
    std::array::from_fn(|c| {
        let sum: u64 = histogram.iter().map(|(color, n)| u64::from(color[c]) * n).sum();
        ((sum + count / 2) / count.max(1)) as u8
    })
}

// Pillow's supplied-palette converter searches on a six-bit RGB grid.
fn palette_nearest(color: Color, palette: &PaletteSearch<'_>, cache: &mut [u16]) -> usize {
    let key = ((color[0] as usize >> 2) << 12) | ((color[1] as usize >> 2) << 6) | (color[2] as usize >> 2);
    if cache[key] == 256 {
        cache[key] = palette.nearest([color[0] & 252, color[1] & 252, color[2] & 252, 255]) as u16;
    }
    cache[key] as usize
}

fn color_histogram<const C: usize>(source: &[u8]) -> Histogram {
    let pixels = source.as_chunks::<C>().0;
    let count = |pixels: &[[u8; C]]| {
        let mut counts = ColorMap::<u64>::default();
        // Neighboring pixels usually repeat; count a run with one lookup.
        let mut run: Option<(u32, u64)> = None;
        for pixel in pixels {
            let color = key(color_of(pixel));
            match &mut run {
                Some((current, n)) if *current == color => *n += 1,
                _ => {
                    if let Some((current, n)) = run {
                        *counts.entry(current).or_insert(0) += n;
                    }
                    run = Some((color, 1));
                }
            }
        }
        if let Some((current, n)) = run {
            *counts.entry(current).or_insert(0) += n;
        }
        counts
    };
    let counts = if pixels.len() >= MIN_PARALLEL_BYTES {
        pixels.par_chunks(CHUNK_PIXELS * 4).map(count).reduce(ColorMap::default, |mut a, mut b| {
            if a.len() < b.len() {
                std::mem::swap(&mut a, &mut b);
            }
            for (color, n) in b {
                *a.entry(color).or_default() += n;
            }
            a
        })
    } else {
        count(pixels)
    };
    let mut histogram: Histogram = counts.into_iter().map(|(color, n)| (color.to_ne_bytes(), n)).collect();
    // Tree ordering is part of the existing palette selection/tie behavior.
    histogram.sort_unstable_by_key(|(color, _)| *color);
    histogram
}

/// Map every pixel through the exact color table built from the histogram.
fn map_colors<const C: usize>(source: &[u8], lookup: &ColorMap<u8>, indices: &mut [u8]) {
    let pixels = source.as_chunks::<C>().0;
    chunks_mut(indices, CHUNK_PIXELS, |chunk, out| {
        let start = chunk * CHUNK_PIXELS;
        let mut last: Option<(u32, u8)> = None;
        for (index, pixel) in out.iter_mut().zip(&pixels[start..]) {
            let color = key(color_of(pixel));
            *index = match last {
                Some((current, index)) if current == color => index,
                _ => {
                    let index = lookup[&color];
                    last = Some((color, index));
                    index
                }
            };
        }
    });
}

fn fixed_palette(image: &Image, source: &[u8], reference: &Image, dither: i32) -> PyResult<Image> {
    let (mode, data) = reference
        .palette
        .as_ref()
        .ok_or_else(|| PyValueError::new_err("bad mode for palette image"))?;
    let mut entries: Vec<Color> = data.chunks_exact(mode.channels()).map(|v| [v[0], v[1], v[2], 255]).collect();
    if entries.is_empty() {
        entries.push([0, 0, 0, 255]);
    }
    let search = PaletteSearch::new(&entries);
    let mut indices = vec![0; image.width as usize * image.height as usize];
    if image.mode == PixelMode::L {
        indices.copy_from_slice(source);
    } else if image.mode != PixelMode::Rgb {
        return Err(PyValueError::new_err("only RGB or L mode images can be quantized to a palette"));
    } else if dither == 0 {
        // Each worker owns its cache and output. No atomics or per-pixel locks.
        chunks_mut(&mut indices, CHUNK_PIXELS * 8, |chunk, out| {
            let mut cache = vec![256u16; 64 * 64 * 64];
            let start = chunk * CHUNK_PIXELS * 8 * 3;
            for (pixel, index) in source[start..start + out.len() * 3].as_chunks::<3>().0.iter().zip(out) {
                *index = palette_nearest([pixel[0], pixel[1], pixel[2], 255], &search, &mut cache) as u8;
            }
        });
    } else if !indices.is_empty() {
        // Error diffusion is ordered across rows; parallelizing it changes the
        // result. Cache and SIMD accelerate palette searches within that order.
        let width = image.width as usize;
        let mut cache = vec![256u16; 64 * 64 * 64];
        // Like Pillow, keep one error row: entry x + 1 still holds the previous
        // row's contribution to this pixel while entry x receives the next
        // row's. Same-row error stays in registers instead of memory.
        let mut errors = vec![[0i32; 3]; width + 1];
        for (row, out) in source.chunks_exact(width * 3).zip(indices.chunks_exact_mut(width)) {
            let (mut right, mut below, mut below_right) = ([0i32; 3], [0i32; 3], [0i32; 3]);
            for (x, (pixel, index)) in row.as_chunks::<3>().0.iter().zip(out.iter_mut()).enumerate() {
                let mut adjusted = [0, 0, 0, 255];
                for c in 0..3 {
                    adjusted[c] = (i32::from(pixel[c]) + (right[c] + errors[x + 1][c]) / 16).clamp(0, 255) as u8;
                }
                let i = palette_nearest(adjusted, &search, &mut cache);
                *index = i as u8;
                for c in 0..3 {
                    let error = i32::from(adjusted[c]) - i32::from(entries[i][c]);
                    errors[x][c] = 3 * error + below[c];
                    below[c] = 5 * error + below_right[c];
                    below_right[c] = error;
                    right[c] = 7 * error;
                }
            }
            // Pillow seeds the row end with its blue accumulators for all bands.
            errors[width] = [below[2], below_right[2], below_right[2]];
        }
    }
    let mut result = Image::from_pixels(image.width, image.height, PixelMode::L, indices, None)?;
    result.palette = reference.palette.clone();
    Ok(result)
}

fn median_cut(histogram: Histogram, colors: usize) -> Vec<Color> {
    let population = |b: &[(Color, u64)]| b.iter().map(|(_, n)| n).sum::<u64>();
    // Carry each box's population so choosing the next split is O(boxes).
    let mut boxes = vec![(population(&histogram), histogram)];
    while boxes.len() < colors {
        let Some(i) = boxes
            .iter()
            .enumerate()
            .filter(|(_, (_, b))| b.len() > 1)
            .max_by_key(|(_, (n, _))| *n)
            .map(|(i, _)| i)
        else {
            break;
        };
        let (total, b) = &mut boxes[i];
        let channel = (0..3)
            .max_by_key(|&c| {
                let low = b.iter().map(|(v, _)| v[c]).min().unwrap();
                let high = b.iter().map(|(v, _)| v[c]).max().unwrap();
                (high - low, 3 - c)
            })
            .unwrap();
        b.sort_by_key(|(v, _)| v[channel]);
        let half = total.div_ceil(2);
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
        let moved = population(&right);
        *total -= moved;
        boxes.push((moved, right));
    }
    boxes.iter().map(|(_, b)| average(b)).collect()
}

fn maximum_coverage(histogram: &Histogram, colors: usize) -> Vec<Color> {
    let mean = average(histogram);
    let first = histogram.iter().max_by_key(|(color, _)| distance(*color, mean)).unwrap().0;
    let mut palette = vec![first];
    let mut distances = vec![u32::MAX; histogram.len()];
    while palette.len() < colors.min(histogram.len()) {
        let last = *palette.last().unwrap();
        // Ties select the last maximum, matching a serial scan.
        let farthest = |offset: usize, distances: &mut [u32], entries: &[(Color, u64)]| {
            let mut best = (0, 0);
            for (i, (d, (color, _))) in distances.iter_mut().zip(entries).enumerate() {
                *d = (*d).min(distance(*color, last));
                if *d >= best.0 {
                    best = (*d, offset + i);
                }
            }
            best
        };
        let later = |a: (u32, usize), b: (u32, usize)| if b.0 > a.0 || (b.0 == a.0 && b.1 > a.1) { b } else { a };
        let (d, i) = if histogram.len() >= CHUNK_PIXELS {
            distances
                .par_chunks_mut(CHUNK_PIXELS)
                .zip(histogram.par_chunks(CHUNK_PIXELS))
                .enumerate()
                .map(|(chunk, (distances, entries))| farthest(chunk * CHUNK_PIXELS, distances, entries))
                .reduce(|| (0, 0), later)
        } else {
            farthest(0, &mut distances, histogram)
        };
        if d == 0 {
            break;
        }
        palette.push(histogram[i].0);
    }
    palette
}

/// Pillow's FASTOCTREE: accumulate colors into a fixed fine cube, keep the
/// used coarse cells plus the most populous fine cells, then map every pixel
/// through a lookup cube. Both passes are one indexed access per pixel.
#[derive(Clone, Copy, Default)]
struct Bucket {
    count: u64,
    sums: [u64; 4],
}

impl Bucket {
    fn add(&mut self, other: &Bucket) {
        self.count += other.count;
        for (sum, other) in self.sums.iter_mut().zip(other.sums) {
            *sum += other;
        }
    }

    fn subtract(&mut self, other: &Bucket) {
        self.count = self.count.saturating_sub(other.count);
        for (sum, other) in self.sums.iter_mut().zip(other.sums) {
            *sum = sum.saturating_sub(other);
        }
    }

    fn average(&self) -> Color {
        if self.count == 0 {
            return [0; 4];
        }
        // Truncate like Pillow, but exactly rather than in single precision.
        self.sums.map(|sum| (sum / self.count) as u8)
    }
}

struct Cube {
    bits: [u32; 4],
    buckets: Vec<Bucket>,
}

impl Cube {
    fn new(bits: [u32; 4]) -> Self {
        Self {
            bits,
            buckets: vec![Bucket::default(); 1 << bits.iter().sum::<u32>()],
        }
    }

    #[inline(always)]
    fn offset(&self, coordinates: [u32; 4]) -> usize {
        let [_, g, b, a] = self.bits;
        ((coordinates[0] << (g + b + a)) | (coordinates[1] << (b + a)) | (coordinates[2] << a) | coordinates[3]) as usize
    }

    #[inline(always)]
    fn index(&self, color: Color) -> usize {
        self.offset(std::array::from_fn(|c| u32::from(color[c]) >> (8 - self.bits[c])))
    }

    fn used(&self) -> usize {
        self.buckets.iter().filter(|bucket| bucket.count > 0).count()
    }

    /// Shrink or expand to another resolution, summing or replicating cells.
    fn resized(&self, bits: [u32; 4]) -> Cube {
        let mut result = Cube::new(bits);
        let source_shift: [u32; 4] = std::array::from_fn(|c| bits[c].saturating_sub(self.bits[c]));
        let target_shift: [u32; 4] = std::array::from_fn(|c| self.bits[c].saturating_sub(bits[c]));
        let width: [u32; 4] = std::array::from_fn(|c| 1 << self.bits[c].max(bits[c]));
        for r in 0..width[0] {
            for g in 0..width[1] {
                for b in 0..width[2] {
                    for a in 0..width[3] {
                        let position = [r, g, b, a];
                        let source = self.offset(std::array::from_fn(|c| position[c] >> source_shift[c]));
                        let target = result.offset(std::array::from_fn(|c| position[c] >> target_shift[c]));
                        result.buckets[target].add(&self.buckets[source]);
                    }
                }
            }
        }
        result
    }

    /// Cells by descending population; equal counts keep cube order.
    fn sorted(&self) -> Vec<Bucket> {
        let mut buckets = self.buckets.clone();
        buckets.sort_by_key(|bucket| Reverse(bucket.count));
        buckets
    }

    fn subtract(&mut self, buckets: &[Bucket]) {
        for bucket in buckets.iter().filter(|bucket| bucket.count > 0) {
            let index = self.index(bucket.average());
            self.buckets[index].subtract(bucket);
        }
    }
}

fn octree<const C: usize>(source: &[u8], colors: usize, alpha: bool) -> (Vec<Color>, Vec<u8>) {
    let (fine_bits, coarse_bits) = if alpha {
        ([3, 4, 3, 3], [2, 2, 2, 2])
    } else {
        ([4, 4, 4, 0], [2, 2, 2, 0])
    };
    let pixels = source.as_chunks::<C>().0;
    let accumulate = |pixels: &[[u8; C]]| {
        let mut cube = Cube::new(fine_bits);
        for pixel in pixels {
            let color = color_of(pixel);
            let index = cube.index(color);
            let bucket = &mut cube.buckets[index];
            bucket.count += 1;
            for (sum, value) in bucket.sums.iter_mut().zip(color) {
                *sum += u64::from(value);
            }
        }
        cube
    };
    let fine = if pixels.len() >= MIN_PARALLEL_BYTES {
        pixels.par_chunks(CHUNK_PIXELS * 16).map(accumulate).reduce(
            || Cube::new(fine_bits),
            |mut a, b| {
                for (a, b) in a.buckets.iter_mut().zip(&b.buckets) {
                    a.add(b);
                }
                a
            },
        )
    } else {
        accumulate(pixels)
    };

    let mut coarse = fine.resized(coarse_bits);
    let mut coarse_count = coarse.used().min(colors);
    let mut fine_count = colors - coarse_count;
    let fine_palette = fine.sorted();
    // Fine cells replace their share of the coarse cells. Cleared coarse
    // cells free palette slots for further fine cells.
    coarse.subtract(&fine_palette[..fine_count]);
    while coarse_count > coarse.used() {
        let subtracted = fine_count;
        coarse_count = coarse.used();
        fine_count = colors - coarse_count;
        coarse.subtract(&fine_palette[subtracted..fine_count]);
    }
    let coarse_palette = coarse.sorted();
    let palette: Vec<Bucket> = coarse_palette[..coarse_count]
        .iter()
        .chain(&fine_palette[..fine_count])
        .copied()
        .collect();

    // Coarse entries claim whole coarse cells; fine entries then override
    // their own cell. Lower indices win ties, as in Pillow's reverse loop.
    let mut coarse_lookup = Cube::new(coarse_bits);
    for (i, bucket) in palette[..coarse_count].iter().enumerate().rev() {
        let index = coarse_lookup.index(bucket.average());
        coarse_lookup.buckets[index].count = i as u64;
    }
    let mut lookup = coarse_lookup.resized(fine_bits);
    for (i, bucket) in palette.iter().enumerate().skip(coarse_count).rev() {
        let index = lookup.index(bucket.average());
        lookup.buckets[index].count = i as u64;
    }
    let table: Vec<u8> = lookup.buckets.iter().map(|bucket| bucket.count as u8).collect();

    let mut indices = vec![0; pixels.len()];
    chunks_mut(&mut indices, CHUNK_PIXELS, |chunk, out| {
        let start = chunk * CHUNK_PIXELS;
        for (index, pixel) in out.iter_mut().zip(&pixels[start..]) {
            *index = table[lookup.index(color_of(pixel))];
        }
    });
    (palette.iter().map(Bucket::average).collect(), indices)
}

fn refine(histogram: &Histogram, palette: &mut [Color], threshold: u64) {
    let mut previous = vec![usize::MAX; histogram.len()];
    loop {
        let search = PaletteSearch::new(palette);
        let nearest = |(color, _): &(Color, u64)| search.nearest(*color);
        let assignments: Vec<usize> = if histogram.len() >= CHUNK_PIXELS {
            histogram.par_iter().map(nearest).collect()
        } else {
            histogram.iter().map(nearest).collect()
        };
        let mut sums = vec![[0u64; 4]; palette.len()];
        let mut counts = vec![0u64; palette.len()];
        let mut changed = 0u64;
        for (((color, count), old), i) in histogram.iter().zip(&mut previous).zip(assignments) {
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

/// Median cut and maximum coverage work on the exact color histogram.
fn exact<const C: usize>(source: &[u8], colors: usize, method: u8, kmeans: u64) -> (Vec<Color>, Vec<u8>) {
    let histogram = color_histogram::<C>(source);
    let mut entries = match method {
        0 => median_cut(histogram.clone(), colors),
        1 => maximum_coverage(&histogram, colors),
        _ => unreachable!(),
    };
    if kmeans > 0 {
        refine(&histogram, &mut entries, kmeans);
    }
    let search = PaletteSearch::new(&entries);
    let entry = |(color, _): &(Color, u64)| (key(*color), search.nearest(*color) as u8);
    let lookup: ColorMap<u8> = if histogram.len() >= CHUNK_PIXELS {
        histogram.par_iter().map(entry).collect()
    } else {
        histogram.iter().map(entry).collect()
    };
    let mut indices = vec![0; source.len() / C];
    map_colors::<C>(source, &lookup, &mut indices);
    (entries, indices)
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
        if let Some(reference) = palette {
            return fixed_palette(image, source, reference, dither);
        }
        let palette_mode = if image.mode == PixelMode::Rgba {
            PixelMode::Rgba
        } else {
            PixelMode::Rgb
        };
        let (entries, indices) = if source.is_empty() {
            (Vec::new(), Vec::new())
        } else if method == 2 {
            match image.mode {
                PixelMode::L => octree::<1>(source, colors, false),
                PixelMode::Rgb => octree::<3>(source, colors, false),
                PixelMode::Rgba => octree::<4>(source, colors, true),
            }
        } else {
            match image.mode {
                PixelMode::L => exact::<1>(source, colors, method, kmeans),
                PixelMode::Rgb => exact::<3>(source, colors, method, kmeans),
                PixelMode::Rgba => exact::<4>(source, colors, method, kmeans),
            }
        };
        let mut result = Image::from_pixels(image.width, image.height, PixelMode::L, indices, None)?;
        result.palette = Some((
            palette_mode,
            entries.iter().flat_map(|v| v[..palette_mode.channels()].iter().copied()).collect(),
        ));
        Ok(result)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn colors_expand_from_every_mode() {
        assert_eq!(color_of(&[7]), [7, 7, 7, 255]);
        assert_eq!(color_of(&[1, 2, 3]), [1, 2, 3, 255]);
        assert_eq!(color_of(&[1, 2, 3, 4]), [1, 2, 3, 4]);
    }

    #[test]
    fn histogram_counts_runs_and_sorts_by_color() {
        let source = [5, 5, 5, 1, 1, 5, 2];
        let histogram = color_histogram::<1>(&source);
        assert_eq!(histogram, vec![([1, 1, 1, 255], 2), ([2, 2, 2, 255], 1), ([5, 5, 5, 255], 4)]);
        assert!(color_histogram::<3>(&[]).is_empty());
    }

    #[test]
    fn octree_keeps_distinct_coarse_colors_exactly() {
        let source = [255, 0, 0, 0, 255, 0, 255, 0, 0, 0, 0, 255];
        let (palette, indices) = octree::<3>(&source, 3, false);
        assert_eq!(palette.len(), 3);
        for (pixel, &index) in source.as_chunks::<3>().0.iter().zip(&indices) {
            assert_eq!(&palette[usize::from(index)][..3], pixel);
        }
        // Population order: red appears twice and sorts first.
        assert_eq!(palette[0], [255, 0, 0, 255]);
    }

    #[test]
    fn octree_fills_unused_entries_and_alpha_cells() {
        let source = [10, 20, 30, 40, 10, 20, 30, 40, 200, 100, 50, 255];
        let (palette, indices) = octree::<4>(&source, 4, true);
        assert_eq!(palette.len(), 4);
        assert_eq!(indices, [0, 0, 1]);
        assert_eq!(palette[0], [10, 20, 30, 40]);
        assert_eq!(palette[1], [200, 100, 50, 255]);
        assert_eq!(palette[2..], [[0; 4], [0; 4]]);
    }

    #[test]
    fn cube_resizing_sums_and_replicates() {
        let mut fine = Cube::new([4, 4, 4, 0]);
        for color in [[0, 0, 0, 255], [15, 15, 15, 255], [16, 16, 16, 255]] {
            let index = fine.index(color);
            fine.buckets[index].count += 1;
        }
        let coarse = fine.resized([2, 2, 2, 0]);
        assert_eq!(coarse.used(), 1);
        assert_eq!(coarse.buckets[0].count, 3);
        // Expansion replicates the coarse cell into every nested fine cell.
        let expanded = coarse.resized([4, 4, 4, 0]);
        for (offset, count) in [(0x000, 3), (0x111, 3), (0x333, 3), (0x444, 0), (0x00F, 0)] {
            assert_eq!(expanded.buckets[offset].count, count);
        }
    }

    #[test]
    fn median_cut_and_coverage_split_populations() {
        let histogram: Histogram = vec![([0, 0, 0, 255], 4), ([255, 0, 0, 255], 1), ([0, 255, 0, 255], 1), ([0, 0, 255, 255], 1)];
        assert_eq!(median_cut(histogram.clone(), 1), vec![[36, 36, 36, 255]]);
        let two = median_cut(histogram.clone(), 2);
        assert_eq!(two.len(), 2);
        let coverage = maximum_coverage(&histogram, 4);
        assert_eq!(coverage.len(), 4);
        assert!(coverage.iter().all(|color| histogram.iter().any(|(c, _)| c == color)));
    }
}
