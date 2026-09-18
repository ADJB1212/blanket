//! Lossless optimization applied only during image saving.
//!
//! The normal save is the size and fidelity baseline. JPEG optimization works
//! on that save's DCT coefficients; JXL and HEIF candidates must decode to exactly
//! the same samples. This does not undo loss introduced by the requested save,
//! and does not reuse an opened image's original encoded source.

use rayon::prelude::*;
use std::collections::HashMap;

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
        let mut output = baseline.clone();
        // Keep ordinary Huffman JPEG compatibility; arithmetic coding is not
        // supported by many consumers. Both candidates retain all coefficients,
        // quantization tables, subsampling, partial edge blocks, and markers.
        let jobs = &[false, true][..if self.effort >= 3 { 2 } else { 1 }];
        parallel::try_candidates(&mut output, jobs, work_bytes, |&progressive| {
            let mut transformer = turbojpeg::Transformer::new().map_err(codec_error)?;
            let mut transform = turbojpeg::Transform::default();
            transform.optimize = true;
            transform.progressive = progressive;
            transformer.transform_to_vec(&transform, &baseline).map_err(codec_error)
        })?;
        Ok(output)
    }

    fn optimize_sample_codec(&self, image: &Image, format: ImageFormat, options: SaveOptions) -> PyResult<Vec<u8>> {
        let prepared = if format == ImageFormat::Heif {
            Some(codecs::PreparedHeif::new(image)?)
        } else {
            None
        };
        let mut output = match &prepared {
            Some(prepared) => prepared.encode(options, "medium")?,
            None => codecs::encode(image, format, options)?,
        };
        let mut reference = None;
        for effort in 1..=self.effort {
            let candidate_options = SaveOptions { effort, ..options };
            // The normal JXL encoder already tried this setting. HEIF's normal
            // preset is medium (effort 6 in the search below).
            if (format == ImageFormat::Jxl && effort == options.effort) || (format == ImageFormat::Heif && effort == 6) {
                continue;
            }
            let candidate = encode_candidate(image, format, candidate_options, prepared.as_ref())?;
            keep_equivalent_lazy(&mut output, candidate, &mut reference, format)?;
        }
        drop(prepared);
        if !options.lossless {
            // A lossless representation of a lossy save's decoded samples can
            // be smaller, especially for flat graphics. Never recompress those
            // samples lossily: even unchanged quality would introduce loss.
            if reference.is_none() {
                reference = Some(codecs::decode(&output, format).map_err(codec_error)?);
            }
            let reference = reference.as_ref().expect("baseline decoded above");
            let prepared = if format == ImageFormat::Heif {
                Some(codecs::PreparedHeif::new(reference)?)
            } else {
                None
            };
            for effort in 1..=self.effort {
                let candidate = encode_candidate(
                    reference,
                    format,
                    SaveOptions {
                        lossless: true,
                        quality: 100,
                        effort,
                        ..options
                    },
                    prepared.as_ref(),
                )?;
                keep_equivalent(&mut output, candidate, reference, format)?;
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

    fn search_png(&self, image: &Image, candidate: &PngCandidate<'_>, compress_level: u8, output: &mut Vec<u8>) -> PyResult<()> {
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
        let jobs: Vec<_> = self
            .png_levels(compress_level)
            .into_iter()
            .flat_map(|level| filters[..count].iter().map(move |&filter| (level, filter)))
            .collect();
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
        let gray = if grayscale {
            pixels::select(pixels, channels, false, 0)
        } else {
            Vec::new()
        };
        let depths: &[usize] = if grayscale { &[1, 2, 4, 8] } else { &[8] };
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
            if grayscale && !pixels::fits_depth(&gray, bits) {
                continue;
            }
            let samples: Vec<u8> = if grayscale {
                pixels::select(&gray, 1, false, (8 - bits) as u8)
            } else {
                crate::simd::convert(pixels, PixelMode::Rgba, PixelMode::Rgb)
            };
            let packed = if bits < 8 {
                pack_rows(&samples, image.width as usize, bits)
            } else {
                samples
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
                output,
            )?;
        }
        Ok(())
    }

    /// Build an exact palette, never quantizing or discarding invisible colors.
    fn optimize_palette_png(&self, image: &Image, pixels: &[u8], compress_level: u8, output: &mut Vec<u8>) -> PyResult<()> {
        let channels = image.mode.channels();
        let mut lookup = HashMap::new();
        let mut palette = Vec::new();
        let mut alpha = Vec::new();
        let mut frequencies = Vec::new();
        let Some(entries) = palette_entries(pixels, channels) else {
            return Ok(());
        };
        for (pixel, count) in entries {
            lookup.insert(pixel, lookup.len() as u8);
            frequencies.push(count);
            if channels == 1 {
                palette.extend_from_slice(&[pixel[0]; 3]);
            } else {
                palette.extend_from_slice(&pixel[..3]);
            }
            alpha.push(if channels == 4 { pixel[3] } else { 255 });
        }
        let bits = match lookup.len() {
            0 => return Ok(()),
            1..=2 => 1,
            3..=4 => 2,
            5..=16 => 4,
            _ => 8,
        };
        let indices = if channels == 1 {
            let mut mapping = [0; 256];
            for (pixel, &index) in &lookup {
                mapping[usize::from(pixel[0])] = index;
            }
            pixels::remap(pixels, &mapping)
        } else {
            let mut indices = vec![0; pixels.len() / channels];
            parallel::chunks_mut_above(&mut indices, 64 * 1024, 256 * 1024, |i, dst| {
                for (pixel, index) in pixels[i * 64 * 1024 * channels..].chunks_exact(channels).zip(dst) {
                    *index = lookup[pixel];
                }
            });
            indices
        };
        let original: Vec<usize> = (0..lookup.len()).collect();
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
        for order in orders {
            let mut mapping = [0_u8; 256];
            let mut rgb = Vec::new();
            let mut transparency = Vec::new();
            for (new, &old) in order.iter().enumerate() {
                mapping[old] = new as u8;
                rgb.extend_from_slice(&palette[3 * old..3 * old + 3]);
                transparency.push(alpha[old]);
            }
            transparency.truncate(transparency.iter().rposition(|&a| a != 255).map_or(0, |i| i + 1));
            let remapped = pixels::remap(&indices, &mapping);
            // Wider indices can produce more compressible filter residuals.
            // Keep this extra search at high effort, retaining packed trials.
            for depth in [1, 2, 4, 8] {
                if depth < bits || (depth != bits && self.effort < 8) {
                    continue;
                }
                let packed;
                let samples = if depth == 8 {
                    remapped.as_slice()
                } else {
                    packed = pack_rows(&remapped, image.width as usize, depth);
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
                    output,
                )?;
            }
        }
        Ok(())
    }
}

type PaletteCounts<'a> = HashMap<&'a [u8], (usize, usize)>;

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
        count_colors(&pixels[..pixels.len().min(4096 * channels)], channels, 0)?;
        pixels
            .par_chunks(chunk)
            .enumerate()
            .map(|(i, src)| count_colors(src, channels, i * 64 * 1024))
            .try_reduce(HashMap::new, |mut a, b| {
                for (color, (count, first)) in b {
                    let entry = a.entry(color).or_insert((0, first));
                    entry.0 += count;
                    entry.1 = entry.1.min(first);
                    if a.len() > 256 {
                        return None;
                    }
                }
                Some(a)
            })?
    } else {
        count_colors(pixels, channels, 0)?
    };
    let mut entries: Vec<_> = counts.into_iter().collect();
    entries.sort_unstable_by_key(|(_, (_, first))| *first);
    Some(entries.into_iter().map(|(color, (count, _))| (color, count)).collect())
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
    if candidate.len() < output.len() {
        let decoded = codecs::decode(&candidate, format).map_err(codec_error)?;
        if (decoded.width, decoded.height, decoded.mode, decoded.bit_depth)
            == (reference.width, reference.height, reference.mode, reference.bit_depth)
            && pixels::equal(decoded.raw_data()?, reference.raw_data()?)
        {
            *output = candidate;
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

fn encode_candidate(image: &Image, format: ImageFormat, options: SaveOptions, prepared: Option<&codecs::PreparedHeif>) -> PyResult<Vec<u8>> {
    if format == ImageFormat::Heif {
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
        prepared
            .expect("HEIF input prepared before search")
            .encode(options, PRESETS[usize::from(options.effort - 1)])
    } else {
        codecs::encode(image, format, options)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::raster::PixelMode;

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
