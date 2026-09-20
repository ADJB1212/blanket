//! Lossless optimization applied only during image saving.
//!
//! The normal save is the size and fidelity baseline. JPEG optimization works
//! on that save's DCT coefficients; JXL and HEIF candidates must decode to exactly
//! the same samples. This does not undo loss introduced by the requested save,
//! and does not reuse an opened image's original encoded source.

use rayon::prelude::*;
use std::borrow::Cow;
#[cfg(test)]
use std::collections::HashMap;
use std::sync::OnceLock;

use pyo3::exceptions::{PyOSError, PyValueError};
use pyo3::prelude::*;

use crate::codecs::{self, ImageFormat, SaveOptions};
use crate::raster::{Image, PixelMode};
use crate::{compressor_simd as pixels, parallel};

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<LosslessImageCompressor>()
}

#[pyclass(name = "_LosslessImageCompressor", module = "blanket._blanket", frozen)]
pub(crate) struct LosslessImageCompressor {
    #[pyo3(get)]
    effort: u8,
}

#[pymethods]
impl LosslessImageCompressor {
    #[new]
    #[pyo3(signature = (*, effort=7))]
    fn new(effort: u8) -> PyResult<Self> {
        if !(1..=10).contains(&effort) {
            return Err(PyValueError::new_err("effort must be between 1 and 10"));
        }
        Ok(Self { effort })
    }
}

impl LosslessImageCompressor {
    pub(crate) fn encode(&self, image: &Image, format: ImageFormat, options: SaveOptions) -> PyResult<Vec<u8>> {
        match format {
            ImageFormat::Png if image.bit_depth == 8 => self.optimize_png(image, options.compress_level),
            ImageFormat::Jpeg => self.optimize_jpeg(codecs::encode(image, format, options)?, image.raw_data()?.len()),
            ImageFormat::Jxl | ImageFormat::Heif => self.optimize_sample_codec(image, format, options),
            _ => codecs::encode(image, format, options),
        }
    }

    fn optimize_jpeg(&self, baseline: Vec<u8>, work_bytes: usize) -> PyResult<Vec<u8>> {
        // Keep ordinary Huffman JPEG compatibility; arithmetic coding is not
        // supported by many consumers. Both candidates retain all coefficients,
        // quantization tables, subsampling, partial edge blocks, and markers.
        let jobs = &[false, true][..if self.effort >= 3 { 2 } else { 1 }];
        let candidate = parallel::best_candidate(jobs, work_bytes, baseline.len(), |&progressive| {
            let mut transformer = turbojpeg::Transformer::new().map_err(codec_error)?;
            let mut transform = turbojpeg::Transform::default();
            transform.optimize = true;
            transform.progressive = progressive;
            let candidate = transformer.transform_to_owned(&transform, &baseline).map_err(codec_error)?;
            Ok::<_, PyErr>((candidate.len() < baseline.len()).then(|| candidate.to_vec()))
        })?;
        Ok(candidate.unwrap_or(baseline))
    }

    fn optimize_sample_codec(&self, image: &Image, format: ImageFormat, options: SaveOptions) -> PyResult<Vec<u8>> {
        // Candidate encoders use one thread each, avoiding nested libjxl pools.
        // Account conservatively for libjxl's larger working set when deciding
        // whether multiple encoders fit the candidate memory budget.
        let work_bytes = image.raw_data()?.len().saturating_mul(4);
        let candidate_count = usize::from(self.effort) - usize::from(options.effort <= self.effort);
        if format == ImageFormat::Jxl && image.raw_data()?.len() >= 32 * 1024 && parallel::candidate_workers(work_bytes, candidate_count) > 1 {
            return self.optimize_jxl_parallel(image, options, work_bytes);
        }
        let mut prepared = PreparedSampleCodec::new(image, format)?;
        let mut output = prepared.encode(options, true)?;
        let mut reference = None;
        let mut priority = 0;
        // More thorough encoders usually set the tightest size bound. Try them
        // first to avoid decoding intermediate winners, but retain the old
        // ascending-effort tie priority so output remains deterministic.
        for effort in (1..=self.effort).rev() {
            let candidate_options = SaveOptions { effort, ..options };
            // The normal JXL encoder already tried this setting. HEIF's normal
            // preset is medium (effort 6 in the search below).
            if !prepared.supports_effort() || (format == ImageFormat::Jxl && effort == options.effort) || (format == ImageFormat::Heif && effort == 6)
            {
                continue;
            }
            let candidate = prepared.encode(candidate_options, false)?;
            keep_equivalent_ranked(&mut output, candidate, &mut reference, format, &mut priority, effort)?;
        }
        drop(prepared);
        if !options.lossless {
            // A lossless representation of a lossy save's decoded samples can
            // be smaller, especially for flat graphics. Never recompress those
            // samples lossily: even unchanged quality would introduce loss.
            if reference.is_none() {
                reference = Some(codecs::decode(&output, format).map_err(codec_error)?);
            }
            let reference_image = reference.as_ref().expect("baseline decoded above");
            let mut prepared = PreparedSampleCodec::new(reference_image, format)?;
            let efforts = if prepared.supports_effort() { self.effort } else { 1 };
            // Lossless trials follow source trials when encoded sizes tie.
            let mut lossless_priority = priority;
            for effort in (1..=efforts).rev() {
                let candidate = prepared.encode(
                    SaveOptions {
                        lossless: true,
                        quality: 100,
                        effort,
                        ..options
                    },
                    false,
                )?;
                let rank = self.effort + effort;
                if candidate.len() < output.len() || (candidate.len() == output.len() && rank < lossless_priority) {
                    let before = output.len();
                    if candidate.len() == before {
                        if equivalent(&candidate, reference_image, format)? {
                            output = candidate;
                            lossless_priority = rank;
                        }
                    } else {
                        keep_equivalent(&mut output, candidate, reference_image, format)?;
                        if output.len() < before {
                            lossless_priority = rank;
                        }
                    }
                }
            }
        }
        Ok(output)
    }

    fn optimize_jxl_parallel(&self, image: &Image, options: SaveOptions, work_bytes: usize) -> PyResult<Vec<u8>> {
        let prepared = codecs::PreparedJxl::new(image)?;
        let baseline = prepared.encode(options)?;
        let reference = OnceLock::new();
        let jobs: Vec<_> = (1..=self.effort).filter(|&effort| effort != options.effort).collect();
        let candidate = parallel::best_candidate(&jobs, work_bytes, baseline.len(), |&effort| {
            let candidate = prepared.encode_with_threads(SaveOptions { effort, ..options }, true)?;
            if candidate.len() >= baseline.len() {
                return Ok(None);
            }
            // Decode the baseline at most once, and only if a candidate could
            // win. All workers compare against the same immutable samples.
            let reference = reference
                .get_or_init(|| codecs::decode(&baseline, ImageFormat::Jxl))
                .as_ref()
                .map_err(codec_error)?;
            Ok::<_, PyErr>(equivalent(&candidate, reference, ImageFormat::Jxl)?.then_some(candidate))
        })?;
        let mut output = candidate.unwrap_or(baseline);
        drop(prepared);
        if !options.lossless {
            let reference = reference
                .get_or_init(|| codecs::decode(&output, ImageFormat::Jxl))
                .as_ref()
                .map_err(codec_error)?;
            let prepared = codecs::PreparedJxl::new(reference)?;
            let jobs: Vec<_> = (1..=self.effort).collect();
            let candidate = parallel::best_candidate(&jobs, work_bytes, output.len(), |&effort| {
                let candidate = prepared.encode_with_threads(
                    SaveOptions {
                        effort,
                        lossless: true,
                        quality: 100,
                        ..options
                    },
                    true,
                )?;
                if candidate.len() >= output.len() {
                    return Ok(None);
                }
                Ok::<_, PyErr>(equivalent(&candidate, reference, ImageFormat::Jxl)?.then_some(candidate))
            })?;
            if let Some(candidate) = candidate {
                output = candidate;
            }
        }
        Ok(output)
    }

    fn optimize_png(&self, image: &Image, compress_level: u8) -> PyResult<Vec<u8>> {
        let pixels = image.pixel_data()?;
        // Keep the normal save as the size baseline, then search each pixel
        // representation here rather than repeating the codec's own search.
        let mut output = codecs::encode_png(image, pixels, compress_level)?;
        self.search_png(
            image,
            &PngCandidate {
                color: match image.mode {
                    PixelMode::L => png::ColorType::Grayscale,
                    PixelMode::Rgb => png::ColorType::Rgb,
                    PixelMode::Rgba => png::ColorType::Rgba,
                },
                depth: png::BitDepth::Eight,
                pixels,
                palette: &[],
                transparency: &[],
            },
            compress_level,
            true,
            &mut output,
        )?;
        self.optimize_palette_png(image, pixels, compress_level, &mut output)?;
        self.optimize_gray_or_key(image, pixels, compress_level, &mut output)?;
        Ok(output)
    }

    fn png_levels(&self, compress_level: u8) -> Vec<u8> {
        let mut levels = vec![compress_level.max(self.effort.min(9))];
        let extras: &[u8] = match self.effort {
            10 => &[1, 2, 3, 4, 5, 6, 7, 8, 9],
            9 => &[6, 9],
            _ => &[],
        };
        for &level in extras {
            if !levels.contains(&level) {
                levels.push(level);
            }
        }
        levels
    }

    fn png_jobs(&self, compress_level: u8, baseline: bool) -> Vec<(u8, png::Filter)> {
        let filters = [
            png::Filter::Adaptive,
            png::Filter::NoFilter,
            png::Filter::Sub,
            png::Filter::Up,
            png::Filter::MinEntropy,
            png::Filter::Avg,
            png::Filter::Paeth,
        ];
        let count = match self.effort {
            1..=2 => 1,
            3..=4 => 2,
            5..=6 => 5,
            _ => filters.len(),
        };
        self.png_levels(compress_level)
            .into_iter()
            .flat_map(|level| filters[..count].iter().map(move |&filter| (level, filter)))
            // The normal save already encoded this exact representation with
            // Adaptive filtering. Level zero uses a different encoder path,
            // and reduced representations must still try every candidate.
            .filter(|&(level, filter)| !(baseline && compress_level != 0 && level == compress_level && filter == png::Filter::Adaptive))
            .collect()
    }

    fn search_png(&self, image: &Image, candidate: &PngCandidate<'_>, compress_level: u8, baseline: bool, output: &mut Vec<u8>) -> PyResult<()> {
        let jobs = self.png_jobs(compress_level, baseline);
        parallel::try_candidates(output, &jobs, candidate.pixels.len(), |&(level, filter)| {
            candidate.encode(image, filter, level)
        })
    }

    fn optimize_gray_or_key(&self, image: &Image, pixels: &[u8], compress_level: u8, output: &mut Vec<u8>) -> PyResult<()> {
        let channels = image.mode.channels();
        let (grayscale, opaque) = pixels::properties(pixels, channels);
        if image.mode == PixelMode::Rgba && (opaque || grayscale) {
            // Retain every alpha and hidden color sample. Gray+alpha supports
            // arbitrary transparency even when a single tRNS key cannot.
            let (color, reduced) = if opaque {
                (png::ColorType::Rgb, crate::simd::convert(pixels, PixelMode::Rgba, PixelMode::Rgb))
            } else {
                (png::ColorType::GrayscaleAlpha, pixels::select(pixels, channels, true, 0))
            };
            self.search_png(
                image,
                &PngCandidate {
                    color,
                    depth: png::BitDepth::Eight,
                    pixels: &reduced,
                    palette: &[],
                    transparency: &[],
                },
                compress_level,
                false,
                output,
            )?;
        }
        let mut key: Option<&[u8]> = None;
        if image.mode == PixelMode::Rgba && !opaque {
            let first = pixels.as_chunks::<4>().0.iter().find(|p| p[3] != 255).expect("non-opaque image");
            if first[3] != 0 || !pixels::matches_key(pixels, [first[0], first[1], first[2]]) {
                return Ok(());
            }
            key = Some(&first[..3]);
        }
        if !grayscale && key.is_none() {
            return Ok(());
        }
        let gray = if grayscale && channels == 1 {
            Cow::Borrowed(pixels)
        } else if grayscale {
            Cow::Owned(pixels::select(pixels, channels, false, 0))
        } else {
            Cow::Borrowed(&[][..])
        };
        let depths: &[usize] = if grayscale { &[1, 2, 4, 8] } else { &[8] };
        // Exact gray depths are nested: every 1-bit value also fits 2/4/8,
        // and every 2-bit value fits 4/8. Validate only the minimum once.
        let minimum_depth = depths
            .iter()
            .copied()
            .find(|&bits| pixels::fits_depth(&gray, bits))
            .expect("8-bit samples always fit");
        for &bits in depths {
            if bits == 8 && image.mode == PixelMode::L {
                // Already searched the unchanged source representation.
                continue;
            }
            // Pillow expands 2/4-bit gray pixels to L8 but leaves the tRNS
            // sample unscaled. Keep interoperability by using a palette or
            // 8-bit gray for nonzero keys (1-bit mode handles scaling).
            if matches!(bits, 2 | 4) && key.is_some_and(|key| key[0] != 0) {
                continue;
            }
            let step = 255 / ((1 << bits) - 1);
            if bits < minimum_depth {
                continue;
            }
            let packed = if grayscale && bits < 8 {
                Cow::Owned(pixels::pack_gray_rows(&gray, image.width as usize, bits))
            } else if grayscale {
                Cow::Borrowed(gray.as_ref())
            } else {
                Cow::Owned(crate::simd::convert(pixels, PixelMode::Rgba, PixelMode::Rgb))
            };
            let transparency: Vec<u8> = match key {
                Some(key) if grayscale => ((usize::from(key[0]) / step) as u16).to_be_bytes().to_vec(),
                Some(key) => key.iter().flat_map(|&v| u16::from(v).to_be_bytes()).collect(),
                None => Vec::new(),
            };
            self.search_png(
                image,
                &PngCandidate {
                    color: if grayscale { png::ColorType::Grayscale } else { png::ColorType::Rgb },
                    depth: png_depth(bits),
                    pixels: &packed,
                    palette: &[],
                    transparency: &transparency,
                },
                compress_level,
                false,
                output,
            )?;
        }
        Ok(())
    }

    /// Build an exact palette, never quantizing or discarding invisible colors.
    fn optimize_palette_png(&self, image: &Image, pixels: &[u8], compress_level: u8, output: &mut Vec<u8>) -> PyResult<()> {
        let channels = image.mode.channels();
        let mut lookup = ColorLookup::new();
        let mut palette = Vec::new();
        let mut alpha = Vec::new();
        let mut frequencies = Vec::new();
        let Some(entries) = palette_entries(pixels, channels) else {
            return Ok(());
        };
        let colors = entries.len();
        let mut gray_mapping = [0; 256];
        for (index, (pixel, count)) in entries.into_iter().enumerate() {
            lookup.insert(pixel_key(pixel), index);
            if channels == 1 {
                gray_mapping[usize::from(pixel[0])] = index as u8;
            }
            frequencies.push(count);
            if channels == 1 {
                palette.extend_from_slice(&[pixel[0]; 3]);
            } else {
                palette.extend_from_slice(&pixel[..3]);
            }
            alpha.push(if channels == 4 { pixel[3] } else { 255 });
        }
        let bits = match colors {
            0 => return Ok(()),
            1..=2 => 1,
            3..=4 => 2,
            5..=16 => 4,
            _ => 8,
        };
        let indices = if channels == 1 {
            pixels::remap(pixels, &gray_mapping)
        } else {
            let mut indices = vec![0; pixels.len() / channels];
            parallel::chunks_mut_above(&mut indices, 64 * 1024, 256 * 1024, |i, dst| {
                let mut previous = None;
                for (pixel, index) in pixels[i * 64 * 1024 * channels..].chunks_exact(channels).zip(dst) {
                    let key = pixel_key(pixel);
                    *index = match previous {
                        Some((old, value)) if old == key => value,
                        _ => {
                            let value = lookup.find(key).expect("palette contains every pixel") as u8;
                            previous = Some((key, value));
                            value
                        }
                    };
                }
            });
            indices
        };
        let original: Vec<usize> = (0..colors).collect();
        let mut orders = vec![original.clone()];
        if self.effort >= 3 {
            let mut frequency = original.clone();
            // Put transparent entries first to shorten tRNS, then favor common
            // colors. Stable ties make output independent of HashMap iteration.
            frequency.sort_by_key(|&i| (alpha[i] == 255, std::cmp::Reverse(frequencies[i])));
            if !orders.contains(&frequency) {
                orders.push(frequency);
            }
        }
        if self.effort >= 5 {
            let mut color = original;
            color.sort_by_key(|&i| (alpha[i] == 255, palette[3 * i], palette[3 * i + 1], palette[3 * i + 2], alpha[i]));
            if !orders.contains(&color) {
                orders.push(color);
            }
        }
        for (order_index, order) in orders.into_iter().enumerate() {
            let mut mapping = [0_u8; 256];
            let mut rgb = Vec::new();
            let mut transparency = Vec::new();
            for (new, &old) in order.iter().enumerate() {
                mapping[old] = new as u8;
                rgb.extend_from_slice(&palette[3 * old..3 * old + 3]);
                transparency.push(alpha[old]);
            }
            transparency.truncate(transparency.iter().rposition(|&a| a != 255).map_or(0, |i| i + 1));
            // First-occurrence ordering already has these indices. Borrow it
            // instead of allocating and applying an identity LUT to the image.
            let fuse_mapping = order_index != 0 && self.effort < 8 && bits < 8;
            let remapped = if order_index == 0 || fuse_mapping {
                Cow::Borrowed(indices.as_slice())
            } else {
                Cow::Owned(pixels::remap(&indices, &mapping))
            };
            // Wider indices can produce more compressible filter residuals.
            // Keep this extra search at high effort, retaining packed trials.
            for depth in [1, 2, 4, 8] {
                if depth < bits || (depth != bits && self.effort < 8) {
                    continue;
                }
                let packed;
                let samples = if depth == 8 {
                    remapped.as_ref()
                } else {
                    packed = if fuse_mapping {
                        pixels::pack_mapped_rows(&indices, image.width as usize, depth, &mapping)
                    } else {
                        pack_rows(&remapped, image.width as usize, depth)
                    };
                    &packed
                };
                self.search_png(
                    image,
                    &PngCandidate {
                        color: png::ColorType::Indexed,
                        depth: png_depth(depth),
                        pixels: samples,
                        palette: &rgb,
                        transparency: &transparency,
                    },
                    compress_level,
                    false,
                    output,
                )?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
type PaletteCounts<'a> = HashMap<&'a [u8], (usize, usize)>;

#[cfg(test)]
fn count_colors(pixels: &[u8], channels: usize, base: usize) -> Option<PaletteCounts<'_>> {
    let mut counts = HashMap::new();
    for (i, pixel) in pixels.chunks_exact(channels).enumerate() {
        let entry = counts.entry(pixel).or_insert((0, base + i));
        entry.0 += 1;
        if counts.len() > 256 {
            return None;
        }
    }
    Some(counts)
}

fn pixel_key(pixel: &[u8]) -> u32 {
    match pixel.len() {
        1 => u32::from(pixel[0]),
        3 => u32::from_le_bytes([pixel[0], pixel[1], pixel[2], 255]),
        4 => u32::from_le_bytes(pixel.try_into().expect("RGBA pixel")),
        _ => unreachable!("validated pixel mode"),
    }
}

/// At most 256 exact colors, with at most 25% occupancy. Integer keys avoid
/// hashing a slice (and its length) for each pixel. Even colliding input has a
/// fixed, small bound; overflow rejects the palette rather than growing it.
struct ColorLookup {
    keys: [u32; 1024],
    values: [u16; 1024],
}

impl ColorLookup {
    fn new() -> Self {
        Self {
            keys: [0; 1024],
            values: [u16::MAX; 1024],
        }
    }

    fn slot(&self, key: u32) -> usize {
        let mixed = (key ^ (key >> 16)).wrapping_mul(0x7feb_352d);
        let mut slot = ((mixed ^ (mixed >> 15)) as usize) & 1023;
        while self.values[slot] != u16::MAX && self.keys[slot] != key {
            slot = (slot + 1) & 1023;
        }
        slot
    }

    fn find(&self, key: u32) -> Option<usize> {
        let value = self.values[self.slot(key)];
        (value != u16::MAX).then_some(usize::from(value))
    }

    fn insert(&mut self, key: u32, value: usize) {
        let slot = self.slot(key);
        self.keys[slot] = key;
        self.values[slot] = value as u16;
    }
}

struct ColorCounts {
    lookup: ColorLookup,
    entries: Vec<(u32, usize, usize)>,
}

impl ColorCounts {
    fn new() -> Self {
        Self {
            lookup: ColorLookup::new(),
            entries: Vec::with_capacity(256),
        }
    }

    fn add(&mut self, key: u32, count: usize, first: usize) -> Option<usize> {
        if let Some(index) = self.lookup.find(key) {
            let entry = &mut self.entries[index];
            entry.1 += count;
            entry.2 = entry.2.min(first);
            Some(index)
        } else if self.entries.len() < 256 {
            let index = self.entries.len();
            self.lookup.insert(key, index);
            self.entries.push((key, count, first));
            Some(index)
        } else {
            None
        }
    }

    fn count(pixels: &[u8], channels: usize, base: usize) -> Option<Self> {
        let mut result = Self::new();
        let mut previous: Option<(u32, usize)> = None;
        for (i, pixel) in pixels.chunks_exact(channels).enumerate() {
            let key = pixel_key(pixel);
            if let Some((old, index)) = previous
                && old == key
            {
                result.entries[index].1 += 1;
            } else {
                previous = Some((key, result.add(key, 1, base + i)?));
            }
        }
        Some(result)
    }

    fn merge(mut self, other: Self) -> Option<Self> {
        for (color, count, first) in other.entries {
            self.add(color, count, first)?;
        }
        Some(self)
    }
}

fn palette_entries(pixels: &[u8], channels: usize) -> Option<Vec<(&[u8], usize)>> {
    if channels == 1 {
        // The sample itself is a bounded key: avoid hashing every gray pixel.
        let mut counts = [0_usize; 256];
        let mut first = Vec::with_capacity(256);
        for (i, &value) in pixels.iter().enumerate() {
            let count = &mut counts[usize::from(value)];
            if *count == 0 {
                first.push(i);
            }
            *count += 1;
        }
        return Some(first.into_iter().map(|i| (&pixels[i..i + 1], counts[usize::from(pixels[i])])).collect());
    }
    let chunk = 64 * 1024 * channels;
    let counts = if parallel::should_parallel(pixels.len(), chunk, 256 * 1024) {
        // Most photographic images exceed 256 colors immediately. Reject them
        // before scheduling tasks or allocating per-image index buffers.
        let prefix_len = pixels.len().min(4096 * channels);
        let prefix = ColorCounts::count(&pixels[..prefix_len], channels, 0)?;
        let remaining = pixels[prefix_len..]
            .par_chunks(chunk)
            .enumerate()
            .map(|(i, src)| ColorCounts::count(src, channels, prefix_len / channels + i * 64 * 1024))
            .try_reduce(ColorCounts::new, ColorCounts::merge)?;
        prefix.merge(remaining)?
    } else {
        ColorCounts::count(pixels, channels, 0)?
    };
    let mut entries = counts.entries;
    entries.sort_unstable_by_key(|&(_, _, first)| first);
    Some(
        entries
            .into_iter()
            .map(|(_, count, first)| (&pixels[first * channels..(first + 1) * channels], count))
            .collect(),
    )
}

struct PngCandidate<'a> {
    color: png::ColorType,
    depth: png::BitDepth,
    pixels: &'a [u8],
    palette: &'a [u8],
    transparency: &'a [u8],
}

impl PngCandidate<'_> {
    fn encode(&self, image: &Image, filter: png::Filter, level: u8) -> PyResult<Vec<u8>> {
        let mut output = Vec::new();
        {
            let mut encoder = png::Encoder::new(&mut output, image.width, image.height);
            encoder.set_color(self.color);
            encoder.set_depth(self.depth);
            encoder.set_filter(filter);
            encoder.set_deflate_compression(png::DeflateCompression::Level(level));
            if !self.palette.is_empty() {
                encoder.set_palette(self.palette);
            }
            if !self.transparency.is_empty() {
                encoder.set_trns(self.transparency);
            }
            let mut writer = encoder.write_header().map_err(codec_error)?;
            writer.write_image_data(self.pixels).map_err(codec_error)?;
            writer.finish().map_err(codec_error)?;
        }
        Ok(output)
    }
}

fn png_depth(bits: usize) -> png::BitDepth {
    match bits {
        1 => png::BitDepth::One,
        2 => png::BitDepth::Two,
        4 => png::BitDepth::Four,
        8 => png::BitDepth::Eight,
        _ => unreachable!("PNG packing uses 1, 2, 4, or 8 bits"),
    }
}

fn pack_rows(samples: &[u8], width: usize, bits: usize) -> Vec<u8> {
    pixels::pack_rows(samples, width, bits)
}

fn codec_error(error: impl std::fmt::Display) -> PyErr {
    PyOSError::new_err(error.to_string())
}

fn keep_equivalent(output: &mut Vec<u8>, candidate: Vec<u8>, reference: &Image, format: ImageFormat) -> PyResult<()> {
    if candidate.len() < output.len() && equivalent(&candidate, reference, format)? {
        *output = candidate;
    }
    Ok(())
}

fn equivalent(candidate: &[u8], reference: &Image, format: ImageFormat) -> PyResult<bool> {
    let decoded = codecs::decode(candidate, format).map_err(codec_error)?;
    Ok(
        (decoded.width, decoded.height, decoded.mode, decoded.bit_depth) == (reference.width, reference.height, reference.mode, reference.bit_depth)
            && pixels::equal(decoded.raw_data()?, reference.raw_data()?),
    )
}

fn keep_equivalent_ranked(
    output: &mut Vec<u8>, candidate: Vec<u8>, reference: &mut Option<Image>, format: ImageFormat, priority: &mut u8, rank: u8,
) -> PyResult<()> {
    let before = output.len();
    if candidate.len() < before {
        keep_equivalent_lazy(output, candidate, reference, format)?;
        if output.len() < before {
            *priority = rank;
        }
    } else if candidate.len() == before && rank < *priority {
        let reference = reference.as_ref().expect("a prior winner was verified");
        if candidate == *output || equivalent(&candidate, reference, format)? {
            *output = candidate;
            *priority = rank;
        }
    }
    Ok(())
}

fn keep_equivalent_lazy(output: &mut Vec<u8>, candidate: Vec<u8>, reference: &mut Option<Image>, format: ImageFormat) -> PyResult<()> {
    if candidate.len() >= output.len() {
        return Ok(());
    }
    if reference.is_none() {
        // Decode before replacing output, so the reference always describes
        // the normal save, even after several successful replacements.
        *reference = Some(codecs::decode(output, format).map_err(codec_error)?);
    }
    keep_equivalent(output, candidate, reference.as_ref().expect("baseline decoded above"), format)
}

enum PreparedSampleCodec<'a> {
    Jxl(codecs::PreparedJxl<'a>),
    Heif(codecs::PreparedHeif),
}

impl<'a> PreparedSampleCodec<'a> {
    fn new(image: &'a Image, format: ImageFormat) -> PyResult<Self> {
        match format {
            ImageFormat::Jxl => Ok(Self::Jxl(codecs::PreparedJxl::new(image)?)),
            ImageFormat::Heif => Ok(Self::Heif(codecs::PreparedHeif::new(image)?)),
            _ => unreachable!("sample codec search only supports JXL and HEIF"),
        }
    }

    fn supports_effort(&self) -> bool {
        match self {
            Self::Jxl(_) => true,
            Self::Heif(prepared) => prepared.supports_presets(),
        }
    }

    fn encode(&mut self, options: SaveOptions, baseline: bool) -> PyResult<Vec<u8>> {
        const PRESETS: [&str; 10] = [
            "ultrafast",
            "superfast",
            "veryfast",
            "faster",
            "fast",
            "medium",
            "slow",
            "slower",
            "veryslow",
            "placebo",
        ];
        match self {
            Self::Jxl(prepared) => prepared.encode(options),
            Self::Heif(prepared) => prepared.encode(options, if baseline { "medium" } else { PRESETS[usize::from(options.effort - 1)] }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::raster::PixelMode;

    #[test]
    fn parallel_jxl_search_matches_serial_candidates() {
        let compressor = LosslessImageCompressor { effort: 7 };
        let serial = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
        let parallel = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
        for mode in [PixelMode::L, PixelMode::Rgb, PixelMode::Rgba] {
            for depth in [8, 10, 12, 16] {
                let samples: Vec<u16> = (0..257 * 129 * mode.channels())
                    .map(|i| ((i * 71 + i / 13) % (1 << depth)) as u16)
                    .collect();
                let image = if depth == 8 {
                    Image::from_pixels(257, 129, mode, samples.into_iter().map(|v| v as u8).collect(), None).unwrap()
                } else {
                    Image::from_samples(257, 129, mode, samples, depth, None).unwrap()
                };
                for lossless in [false, true] {
                    let options = SaveOptions {
                        quality: 71,
                        compress_level: 6,
                        lossless,
                        effort: 1,
                    };
                    let expected = serial.install(|| compressor.encode(&image, ImageFormat::Jxl, options)).unwrap();
                    let actual = parallel
                        .install(|| compressor.optimize_jxl_parallel(&image, options, image.raw_data().unwrap().len() * 4))
                        .unwrap();
                    assert_eq!(actual, expected, "mode={mode:?}, depth={depth}, lossless={lossless}");
                }
            }
        }
    }

    #[test]
    fn reordered_sample_search_matches_ascending_reference() {
        let compressor = LosslessImageCompressor { effort: 3 };
        for format in [ImageFormat::Jxl, ImageFormat::Heif] {
            for mode in [PixelMode::L, PixelMode::Rgb, PixelMode::Rgba] {
                for depth in [8, 12] {
                    let samples: Vec<u16> = (0..17 * 19 * mode.channels())
                        .map(|i| ((i * 71 + i / 13) % (1 << depth)) as u16)
                        .collect();
                    let image = if depth == 8 {
                        Image::from_pixels(17, 19, mode, samples.into_iter().map(|v| v as u8).collect(), None).unwrap()
                    } else {
                        Image::from_samples(17, 19, mode, samples, depth, None).unwrap()
                    };
                    for lossless in [false, true] {
                        let options = SaveOptions {
                            quality: 71,
                            compress_level: 6,
                            lossless,
                            effort: 2,
                        };
                        let mut expected = codecs::encode(&image, format, options).unwrap();
                        let reference = codecs::decode(&expected, format).unwrap();
                        // Fresh preparation per candidate is the old behavior
                        // for JXL; normal encoding also gives fresh HEIF state.
                        for effort in 1..=compressor.effort {
                            if format == ImageFormat::Jxl && effort == options.effort {
                                continue;
                            }
                            let candidate = PreparedSampleCodec::new(&image, format)
                                .unwrap()
                                .encode(SaveOptions { effort, ..options }, false)
                                .unwrap();
                            keep_equivalent(&mut expected, candidate, &reference, format).unwrap();
                        }
                        if !lossless {
                            for effort in 1..=compressor.effort {
                                let candidate = PreparedSampleCodec::new(&reference, format)
                                    .unwrap()
                                    .encode(
                                        SaveOptions {
                                            lossless: true,
                                            quality: 100,
                                            effort,
                                            ..options
                                        },
                                        false,
                                    )
                                    .unwrap();
                                keep_equivalent(&mut expected, candidate, &reference, format).unwrap();
                            }
                        }
                        assert_eq!(
                            compressor.encode(&image, format, options).unwrap(),
                            expected,
                            "{format:?}, {mode:?}, depth={depth}, lossless={lossless}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn ranked_verification_preserves_first_candidate_ties() {
        let encode = |tag: &str, value: u8| {
            let mut output = Vec::new();
            let mut encoder = png::Encoder::new(&mut output, 1, 1);
            encoder.set_color(png::ColorType::Grayscale);
            encoder.set_depth(png::BitDepth::Eight);
            encoder.add_text_chunk("rank".into(), tag.into()).unwrap();
            let mut writer = encoder.write_header().unwrap();
            writer.write_image_data(&[value]).unwrap();
            writer.finish().unwrap();
            output
        };
        let first = encode("one", 71);
        let second = encode("two", 71);
        assert_eq!(first.len(), second.len());
        let mut output = first.clone();
        let mut reference = Some(codecs::decode(&first, ImageFormat::Png).unwrap());
        let mut priority = 3;
        keep_equivalent_ranked(&mut output, second.clone(), &mut reference, ImageFormat::Png, &mut priority, 2).unwrap();
        assert_eq!(output, second);
        keep_equivalent_ranked(&mut output, first.clone(), &mut reference, ImageFormat::Png, &mut priority, 1).unwrap();
        assert_eq!(output, first);
        keep_equivalent_ranked(&mut output, encode("bad", 72), &mut reference, ImageFormat::Png, &mut priority, 0).unwrap();
        assert_eq!(output, first);
        assert_eq!(priority, 1);
    }

    #[test]
    fn integer_palette_counts_match_reference_with_collisions_and_runs() {
        let empty = ColorLookup::new();
        let keys: Vec<_> = (0..1_000_000_u32).filter(|&key| empty.slot(key) == 0).take(257).collect();
        assert_eq!(keys.len(), 257);
        for channels in [3, 4] {
            for count in [1, 2, 16, 256, 257] {
                let source: Vec<u8> = keys[..count]
                    .iter()
                    .chain(keys[..count].iter().rev())
                    .flat_map(|key| key.to_le_bytes()[..channels].repeat(17))
                    .collect();
                let expected = count_colors(&source, channels, 0).map(|counts| {
                    let mut entries: Vec<_> = counts.into_iter().collect();
                    entries.sort_unstable_by_key(|(_, (_, first))| *first);
                    entries.into_iter().map(|(pixel, (count, _))| (pixel, count)).collect::<Vec<_>>()
                });
                assert_eq!(palette_entries(&source, channels), expected);
            }
        }
    }

    #[test]
    fn png_search_skips_only_the_existing_baseline() {
        for effort in 1..=10 {
            let compressor = LosslessImageCompressor { effort };
            for level in 0..=9 {
                let all = compressor.png_jobs(level, false);
                let remaining = compressor.png_jobs(level, true);
                let expected: Vec<_> = all
                    .into_iter()
                    .filter(|&(candidate_level, filter)| !(level != 0 && candidate_level == level && filter == png::Filter::Adaptive))
                    .collect();
                assert_eq!(remaining, expected);
            }
        }
        assert!(LosslessImageCompressor { effort: 1 }.png_jobs(6, true).is_empty());
    }

    #[test]
    fn skipped_png_candidates_are_identical_to_normal_saves() {
        for (mode, color) in [
            (PixelMode::L, png::ColorType::Grayscale),
            (PixelMode::Rgb, png::ColorType::Rgb),
            (PixelMode::Rgba, png::ColorType::Rgba),
        ] {
            for (width, height) in [(1, 1), (17, 9), (257, 129)] {
                let samples = (0..width * height * mode.channels()).map(|i| (i * 71 + i / 13) as u8).collect();
                let image = Image::from_pixels(width as u32, height as u32, mode, samples, None).unwrap();
                let candidate = PngCandidate {
                    color,
                    depth: png::BitDepth::Eight,
                    pixels: image.pixel_data().unwrap(),
                    palette: &[],
                    transparency: &[],
                };
                for level in 1..=9 {
                    assert_eq!(
                        codecs::encode_png(&image, candidate.pixels, level).unwrap(),
                        candidate.encode(&image, png::Filter::Adaptive, level).unwrap()
                    );
                }
            }
        }
    }

    #[test]
    fn lazy_verification_decodes_only_smaller_candidates_and_keeps_original_reference() {
        let image = Image::from_pixels(32, 32, PixelMode::Rgba, [19, 73, 151, 0].repeat(1024), None).unwrap();
        let baseline = codecs::encode_png(&image, image.pixel_data().unwrap(), 0).unwrap();
        let mut output = baseline.clone();
        let mut reference = None;
        keep_equivalent_lazy(&mut output, baseline.clone(), &mut reference, ImageFormat::Png).unwrap();
        assert!(reference.is_none());
        let changed = Image::from_pixels(32, 32, PixelMode::Rgba, [0, 0, 0, 0].repeat(1024), None).unwrap();
        let changed = codecs::encode_png(&changed, changed.pixel_data().unwrap(), 9).unwrap();
        assert!(changed.len() < baseline.len());
        keep_equivalent_lazy(&mut output, changed, &mut reference, ImageFormat::Png).unwrap();
        assert_eq!(output, baseline);
        let candidate = codecs::encode_png(&image, image.pixel_data().unwrap(), 9).unwrap();
        assert!(candidate.len() < baseline.len());
        keep_equivalent_lazy(&mut output, candidate.clone(), &mut reference, ImageFormat::Png).unwrap();
        assert_eq!(output, candidate);
        assert_eq!(reference.unwrap().raw_data().unwrap(), image.raw_data().unwrap());
    }

    #[test]
    fn gray_palette_histogram_preserves_counts_and_first_occurrence() {
        for count in [0, 1, 255, 256, 257, 131_073] {
            let source: Vec<u8> = (0..count).map(|i| (255 - (i * 71 % 256)) as u8).collect();
            let entries = palette_entries(&source, 1).unwrap();
            let mut expected: Vec<_> = count_colors(&source, 1, 0).unwrap().into_iter().collect();
            expected.sort_unstable_by_key(|(_, (_, first))| *first);
            assert_eq!(
                entries,
                expected.into_iter().map(|(pixel, (count, _))| (pixel, count)).collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn palette_counts_preserve_first_occurrence_across_workers() {
        let mut source: Vec<u8> = (0..131_073).flat_map(|i| [(i % 256) as u8, 19, 37, 255]).collect();
        let run = || palette_entries(&source, 4).unwrap();
        let serial = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap().install(run);
        let parallel = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap().install(run);
        assert_eq!(serial, parallel);
        assert_eq!(serial[0], (&[0, 19, 37, 255][..], 513));
        assert_eq!(serial[255], (&[255, 19, 37, 255][..], 512));
        source.extend_from_slice(&[0, 20, 37, 255]);
        // Overflow only in the last chunk must also reject the whole palette.
        assert!(
            rayon::ThreadPoolBuilder::new()
                .num_threads(4)
                .build()
                .unwrap()
                .install(|| palette_entries(&source, 4))
                .is_none()
        );
    }

    #[test]
    fn rejects_smaller_candidates_that_change_samples_or_layout() {
        let reference = Image::from_pixels(2, 1, PixelMode::Rgba, vec![1, 2, 3, 0, 4, 5, 6, 255], None).unwrap();
        let mut invisible_color = reference.clone();
        invisible_color.pixels.as_mut().unwrap()[0] = 0;
        let mut alpha = reference.clone();
        alpha.pixels.as_mut().unwrap()[3] = 1;
        let mut layout = reference.clone();
        layout.width = 1;
        layout.height = 2;
        for changed in [invisible_color, alpha, layout] {
            let candidate = codecs::encode_png(&changed, changed.pixel_data().unwrap(), 6).unwrap();
            let mut output = vec![42; candidate.len() + 1];
            let baseline = output.clone();
            keep_equivalent(&mut output, candidate, &reference, ImageFormat::Png).unwrap();
            assert_eq!(output, baseline);
        }
    }

    #[test]
    fn accepts_only_strictly_smaller_equivalent_candidates() {
        let reference = Image::from_pixels(1, 1, PixelMode::L, vec![71], None).unwrap();
        let candidate = codecs::encode_png(&reference, reference.pixel_data().unwrap(), 6).unwrap();
        let mut output = vec![42; candidate.len() + 1];
        keep_equivalent(&mut output, candidate.clone(), &reference, ImageFormat::Png).unwrap();
        assert_eq!(output, candidate);
        for length in [candidate.len(), candidate.len() - 1] {
            let mut output = vec![42; length];
            let baseline = output.clone();
            keep_equivalent(&mut output, candidate.clone(), &reference, ImageFormat::Png).unwrap();
            assert_eq!(output, baseline);
        }
    }
}
