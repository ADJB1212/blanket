use std::simd::cmp::SimdPartialEq;
use std::simd::num::{SimdFloat, SimdUint};
use std::simd::{Select, Simd, StdFloat};

use blanket_core::pixels::{Samples, load_samples, store_sample_bytes, store_samples};
use blanket_core::{Image, PixelMode, samples};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

enum MaskSamples<'a> {
    Bytes(&'a [u8]),
    Wide(std::borrow::Cow<'a, [u16]>),
}

impl MaskSamples<'_> {
    fn get(&self, index: usize) -> u16 {
        match self {
            Self::Bytes(values) => u16::from(values[index]),
            Self::Wide(values) => values[index],
        }
    }

    fn load(&self, index: usize, channels: usize) -> Samples {
        match self {
            Self::Bytes(values) => {
                if channels == 1 {
                    blanket_core::pixels::Bytes::from_slice(&values[index..]).cast()
                } else {
                    blanket_core::pixels::load::<4>(&values[index..])[3].cast()
                }
            }
            Self::Wide(values) => {
                if channels == 1 {
                    Samples::from_slice(&values[index..])
                } else {
                    load_samples::<4>(&values[index..])[3]
                }
            }
        }
    }

    fn slice(&self, index: usize) -> MaskSamples<'_> {
        match self {
            Self::Bytes(values) => MaskSamples::Bytes(&values[index..]),
            Self::Wide(values) => MaskSamples::Wide(std::borrow::Cow::Borrowed(&values[index..])),
        }
    }
}

fn matching(first: &Image, second: &Image) -> PyResult<()> {
    if first.mode != second.mode || first.bit_depth != second.bit_depth {
        return Err(PyValueError::new_err("images do not match"));
    }
    Ok(())
}

pub(crate) fn paste(image: &mut Image, source: &Image, position: (i64, i64), mask: Option<&Image>, fill: bool) -> PyResult<()> {
    matching(image, source)?;
    let source_bytes = source.raw_data()?;
    image.raw_data()?;
    let mask_values = mask
        .map(|mask| {
            if mask.palette.is_some()
                || !matches!(mask.mode, PixelMode::L | PixelMode::Rgba)
                || (mask.width, mask.height) != (source.width, source.height)
                || mask.bit_depth != 8 && mask.bit_depth != image.bit_depth
            {
                return Err(PyValueError::new_err("bad transparency mask"));
            }
            let values = if mask.bit_depth == 8 { MaskSamples::Bytes(mask.pixel_data()?) } else { MaskSamples::Wide(samples::view(mask)?) };
            Ok((values, mask.mode.channels(), (1_u64 << mask.bit_depth) - 1))
        })
        .transpose()?;
    let channels = image.mode.channels();
    let (left, top, right, bottom) =
        super::composite::paste_bounds(position, image.width as usize, image.height as usize, source.width, source.height);
    if right <= left || bottom <= top {
        return Ok(());
    }
    let source_values = if mask.is_some() { samples::view(source)? } else { std::borrow::Cow::Borrowed(&[][..]) };
    let big_endian = image.mode.is_big_endian();
    let width = image.width as usize;
    image.ensure_owned();
    let pixels = image.pixels.as_mut().unwrap();
    let count = (right - left) as usize;
    let mut values = if mask.is_some() { vec![0_u16; count * channels] } else { Vec::new() };
    for y in top..bottom {
        let src_start = ((y - position.1) as usize * source.width as usize + (left - position.0) as usize) * channels;
        let dst_start = (y as usize * width + left as usize) * channels * 2;
        let destination = &mut pixels[dst_start..dst_start + count * channels * 2];
        if mask.is_none() {
            destination.copy_from_slice(&source_bytes[src_start * 2..(src_start + count * channels) * 2]);
            continue;
        }
        for (value, &bytes) in values.iter_mut().zip(destination.as_chunks::<2>().0) {
            *value = if big_endian { u16::from_be_bytes(bytes) } else { u16::from_le_bytes(bytes) };
        }
        let dst = &mut values;
        let src = &source_values[src_start..src_start + count * channels];
        let end = if let Some((mask, mc, maximum)) = &mask_values {
            let mask = mask.slice(src_start / channels * mc);
            match channels {
                1 => paste_row::<1>(dst, src, &mask, *mc, *maximum as u32, fill),
                3 => paste_row::<3>(dst, src, &mask, *mc, *maximum as u32, fill),
                4 => paste_row::<4>(dst, src, &mask, *mc, *maximum as u32, fill),
                _ => unreachable!(),
            }
        } else {
            dst.copy_from_slice(src);
            count
        };
        for x in left + end as i64..right {
            let src = ((y - position.1) as usize * source.width as usize + (x - position.0) as usize) * channels;
            let dst = (x - left) as usize * channels;
            let (alpha, maximum) = mask_values
                .as_ref()
                .map_or((1, 1), |(values, channels_mask, maximum)| (u64::from(values.get((src / channels + 1) * channels_mask - 1)), *maximum));
            let replace_rgb =
                fill && channels == 4 && mask_values.as_ref().is_some_and(|(_, channels, _)| *channels == 1) && values[dst + 3] == 0 && alpha != 0;
            for c in 0..channels {
                let alpha = if replace_rgb && c < 3 { maximum } else { alpha };
                values[dst + c] =
                    ((u64::from(values[dst + c]) * (maximum - alpha) + u64::from(source_values[src + c]) * alpha + maximum / 2) / maximum) as u16;
            }
        }
        for (&value, bytes) in values.iter().zip(destination.as_chunks_mut::<2>().0) {
            *bytes = if big_endian { value.to_be_bytes() } else { value.to_le_bytes() };
        }
    }
    Ok(())
}

fn paste_row<const C: usize>(dst: &mut [u16], src: &[u16], mask: &MaskSamples<'_>, mc: usize, maximum: u32, fill: bool) -> usize {
    match maximum {
        255 => paste_row_max::<C, 255>(dst, src, mask, mc, fill),
        1023 => paste_row_max::<C, 1023>(dst, src, mask, mc, fill),
        4095 => paste_row_max::<C, 4095>(dst, src, mask, mc, fill),
        65535 => paste_row_max::<C, 65535>(dst, src, mask, mc, fill),
        _ => unreachable!("validated mask depth"),
    }
}

fn paste_row_max<const C: usize, const MAXIMUM: u32>(dst: &mut [u16], src: &[u16], mask: &MaskSamples<'_>, mc: usize, fill: bool) -> usize {
    let end = dst.len() / C / 16 * 16;
    for i in (0..end).step_by(16) {
        let original = load_samples::<C>(&dst[i * C..]);
        let source = load_samples::<C>(&src[i * C..]);
        let alpha = mask.load(i * mc, mc).cast::<u32>();
        let replace = if C == 4 && mc == 1 && fill {
            original[3].simd_eq(Samples::splat(0)).cast::<i32>() & alpha.simd_ne(Simd::splat(0))
        } else {
            std::simd::Mask::splat(false)
        };
        let result: [Samples; C] = std::array::from_fn(|c| {
            let a = if c < 3 { replace.select(Simd::splat(MAXIMUM), alpha) } else { alpha };
            ((original[c].cast::<u32>() * (Simd::splat(MAXIMUM) - a) + source[c].cast::<u32>() * a + Simd::splat(MAXIMUM / 2)) / Simd::splat(MAXIMUM))
                .cast()
        });
        store_samples(&mut dst[i * C..], result);
    }
    end
}

pub(crate) fn composite(background: &Image, overlay: &Image) -> PyResult<Image> {
    matching(background, overlay)?;
    let values = samples::view(background)?;
    let source = samples::view(overlay)?;
    let maximum = u64::from(samples::maximum(background));
    let mut pixels = blanket_core::raster::alloc_pixels(values.len() * 2)?;
    composite_samples(&values, &source, &mut pixels, maximum);
    samples::output_bytes(background, pixels)
}

fn composite_samples(background: &[u16], source: &[u16], output: &mut [u8], maximum: u64) {
    const CHUNK: usize = 16 * 1024 * 8;
    blanket_core::parallel::chunks_mut_above(output, CHUNK, 256 * 1024, |chunk, dst| {
        let start = chunk * CHUNK / 2;
        let count = dst.len() / 2;
        composite_chunk(&background[start..start + count], &source[start..start + count], dst, maximum);
    });
}

fn composite_chunk(background: &[u16], source: &[u16], output: &mut [u8], maximum: u64) {
    let end = background.len() / 64 * 64;
    for i in (0..end).step_by(64) {
        let dst = load_samples::<4>(&background[i..]);
        let src = load_samples::<4>(&source[i..]);
        let a = src[3].cast::<f64>();
        let b = dst[3].cast::<f64>() * (Simd::splat(maximum as f64) - a);
        let alpha = a * Simd::splat(maximum as f64) + b;
        let denominator = alpha.simd_max(Simd::splat(1.0));
        // Numerators fit 48 bits; f64 division cannot cross sample boundaries.
        let result: [Samples; 4] = std::array::from_fn(|c| {
            let value = if c == 3 {
                (alpha + Simd::splat((maximum / 2) as f64)) / Simd::splat(maximum as f64)
            } else {
                (src[c].cast::<f64>() * a * Simd::splat(maximum as f64) + dst[c].cast::<f64>() * b + (denominator * Simd::splat(0.5)).floor())
                    / denominator
            };
            src[3].simd_eq(Samples::splat(0)).select(dst[c], value.cast())
        });
        store_sample_bytes(&mut output[i * 2..], result);
    }
    for ((original, src), output) in
        background[end..].as_chunks::<4>().0.iter().zip(source[end..].as_chunks::<4>().0).zip(output[end * 2..].as_chunks_mut::<8>().0)
    {
        let mut dst = *original;
        let a = u64::from(src[3]);
        if a != 0 {
            let b = u64::from(dst[3]) * (maximum - a);
            let alpha = a * maximum + b;
            for c in 0..3 {
                dst[c] = ((u64::from(src[c]) * a * maximum + u64::from(dst[c]) * b + alpha / 2) / alpha) as u16;
            }
            dst[3] = ((alpha + maximum / 2) / maximum) as u16;
        }
        for (dst, value) in output.as_chunks_mut::<2>().0.iter_mut().zip(dst) {
            *dst = value.to_le_bytes();
        }
    }
}

pub(crate) fn putalpha(image: &mut Image, alpha: &Image) -> PyResult<()> {
    if alpha.bit_depth != 8 && alpha.bit_depth != image.bit_depth {
        return Err(PyValueError::new_err("images do not match"));
    }
    let source = samples::view(image)?;
    let maximum = u32::from(samples::maximum(image));
    let alpha_values = if alpha.bit_depth == 8 { MaskSamples::Bytes(alpha.pixel_data()?) } else { MaskSamples::Wide(samples::view(alpha)?) };
    let mut pixels = blanket_core::buffers::buffer(image.size(), 8)?;
    match image.mode.channels() {
        3 => putalpha_samples::<3>(&source, &alpha_values, &mut pixels, maximum),
        4 => putalpha_samples::<4>(&source, &alpha_values, &mut pixels, maximum),
        _ => unreachable!(),
    }
    let result = Image::from_sample_bytes(image.width, image.height, PixelMode::Rgba, pixels, image.bit_depth, None)?;
    image.pixels = result.pixels;
    image.mode = PixelMode::Rgba;
    Ok(())
}

fn putalpha_samples<const C: usize>(source: &[u16], alpha: &MaskSamples<'_>, output: &mut [u8], maximum: u32) {
    const CHUNK: usize = 16 * 1024;
    blanket_core::parallel::chunks_mut(output, CHUNK * 8, |chunk, dst| {
        let start = chunk * CHUNK;
        let count = dst.len() / 8;
        putalpha_chunk::<C>(&source[start * C..][..count * C], &alpha.slice(start), dst, maximum);
    });
}

fn putalpha_chunk<const C: usize>(source: &[u16], alpha: &MaskSamples<'_>, output: &mut [u8], maximum: u32) {
    let count = output.len() / 8;
    let end = count / 16 * 16;
    for i in (0..end).step_by(16) {
        let input = load_samples::<C>(&source[i * C..]);
        let values = alpha.load(i, 1).cast::<u32>();
        let values =
            if matches!(alpha, MaskSamples::Bytes(_)) { (values * Simd::splat(maximum) + Simd::splat(127)) / Simd::splat(255) } else { values };
        store_sample_bytes(&mut output[i * 8..], [input[0], input[1], input[2], values.cast()]);
    }
    for (i, (dst, src)) in output[end * 8..].as_chunks_mut::<8>().0.iter_mut().zip(source[end * C..].as_chunks::<C>().0).enumerate() {
        let a = u32::from(alpha.get(end + i));
        let a = if matches!(alpha, MaskSamples::Bytes(_)) { (a * maximum + 127) / 255 } else { a } as u16;
        for (dst, value) in dst.as_chunks_mut::<2>().0.iter_mut().zip([src[0], src[1], src[2], a]) {
            *dst = value.to_le_bytes();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn paste_case<const C: usize>(count: usize, maximum: u32, mc: usize, fill: bool) {
        let source: Vec<_> = (0..count * C + 1).map(|i| ((i * 719 + 257) % 65536) as u16).collect();
        let mask: Vec<_> = (0..count * mc + 1).map(|i| ((i * 127 + i / 3) % (maximum as usize + 1)) as u16).collect();
        let mut output = vec![173; count * C + 2];
        for (i, dst) in output[1..count * C + 1].iter_mut().enumerate() {
            *dst = if i % 4 == 3 { 0 } else { 65535 };
        }
        let original = output.clone();
        let byte_mask: Vec<u8> = mask.iter().map(|&v| v as u8).collect();
        let samples = if maximum == 255 { MaskSamples::Bytes(&byte_mask[1..]) } else { MaskSamples::Wide(std::borrow::Cow::Borrowed(&mask[1..])) };
        let end = paste_row::<C>(&mut output[1..count * C + 1], &source[1..], &samples, mc, maximum, fill);
        for i in 0..end {
            let a = u64::from(mask[1 + (i + 1) * mc - 1]);
            let replace = fill && C == 4 && mc == 1 && original[1 + i * C + 3] == 0 && a != 0;
            for c in 0..C {
                let a = if replace && c < 3 { u64::from(maximum) } else { a };
                let expected =
                    (u64::from(original[1 + i * C + c]) * (u64::from(maximum) - a) + u64::from(source[1 + i * C + c]) * a + u64::from(maximum / 2))
                        / u64::from(maximum);
                assert_eq!(output[1 + i * C + c], expected as u16);
            }
        }
        assert_eq!(&output[1 + end * C..], &original[1 + end * C..]);
        assert_eq!(output[0], 173);
    }

    #[test]
    fn masked_paste_preserves_rounding_for_every_alpha() {
        for maximum in [255_u32, 1023, 4095, 65535] {
            let count = maximum as usize + 1;
            let mask: Vec<_> = (0..count).map(|value| value as u16).collect();
            for (source, destination) in [(0, 65535), (65535, 0), (32767, 65535)] {
                let source = vec![source; count];
                let mut output = vec![destination; count];
                let samples = MaskSamples::Wide(std::borrow::Cow::Borrowed(&mask));
                assert_eq!(paste_row::<1>(&mut output, &source, &samples, 1, maximum, false), count);
                for (i, &value) in output.iter().enumerate() {
                    let expected =
                        (u64::from(destination) * (u64::from(maximum) - i as u64) + u64::from(source[i]) * i as u64 + u64::from(maximum / 2))
                            / u64::from(maximum);
                    assert_eq!(value, expected as u16);
                }
            }
        }
    }

    #[test]
    fn masked_paste_preserves_wide_products_alpha_unaligned_slices_and_guards() {
        for count in [0, 1, 15, 16, 17, 31, 32, 33] {
            for maximum in [255, 1023, 4095, 65535] {
                for mc in [1, 4] {
                    for fill in [false, true] {
                        paste_case::<1>(count, maximum, mc, fill);
                        paste_case::<3>(count, maximum, mc, fill);
                        paste_case::<4>(count, maximum, mc, fill);
                    }
                }
            }
        }
    }

    #[test]
    fn source_over_matches_integer_reference_for_every_source_alpha() {
        for maximum in [1023_u64, 4095, 65535] {
            for destination_alpha in [0, 1, 2, maximum / 2, maximum - 1, maximum] {
                let count = maximum as usize + 1;
                let mut original = vec![0; count * 4];
                let mut source = original.clone();
                for (i, (dst, src)) in original.as_chunks_mut::<4>().0.iter_mut().zip(source.as_chunks_mut::<4>().0).enumerate() {
                    *dst = [maximum as u16, 0, ((i * 719) as u64 % (maximum + 1)) as u16, destination_alpha as u16];
                    *src = [0, maximum as u16, ((i * 313 + 619) as u64 % (maximum + 1)) as u16, i as u16];
                }
                let mut pixels = vec![0; count * 8];
                composite_samples(&original, &source, &mut pixels, maximum);
                let output: Vec<_> = pixels.as_chunks::<2>().0.iter().map(|&bytes| u16::from_le_bytes(bytes)).collect();
                for ((actual, dst), src) in output.as_chunks::<4>().0.iter().zip(original.as_chunks::<4>().0).zip(source.as_chunks::<4>().0) {
                    let a = u64::from(src[3]);
                    if a == 0 {
                        assert_eq!(actual, dst);
                        continue;
                    }
                    let b = destination_alpha * (maximum - a);
                    let alpha = a * maximum + b;
                    for c in 0..3 {
                        assert_eq!(actual[c], ((u64::from(src[c]) * a * maximum + u64::from(dst[c]) * b + alpha / 2) / alpha) as u16);
                    }
                    assert_eq!(actual[3], ((alpha + maximum / 2) / maximum) as u16);
                }
            }
        }
    }

    #[test]
    fn source_over_preserves_alpha_extremes_tails_and_guards() {
        for maximum in [1023_u64, 4095, 65535] {
            for count in [0, 1, 15, 16, 17, 31, 32, 33, 65] {
                let mut output = vec![173; count * 4 + 2];
                let mut source = vec![0; count * 4 + 1];
                for i in 0..count * 4 {
                    output[1 + i] = ((i * 719 + 257) as u64 % (maximum + 1)) as u16;
                    source[1 + i] = ((i * 313 + 619) as u64 % (maximum + 1)) as u16;
                    if i % 8 == 3 {
                        source[1 + i] = 0;
                        output[1 + i] = 0;
                    }
                    if i % 12 == 7 {
                        source[1 + i] = maximum as u16;
                    }
                }
                let mut expected = output.clone();
                for (dst, src) in expected[1..count * 4 + 1].as_chunks_mut::<4>().0.iter_mut().zip(source[1..].as_chunks::<4>().0) {
                    let a = u64::from(src[3]);
                    if a == 0 {
                        continue;
                    }
                    let b = u64::from(dst[3]) * (maximum - a);
                    let alpha = a * maximum + b;
                    for c in 0..3 {
                        dst[c] = ((u64::from(src[c]) * a * maximum + u64::from(dst[c]) * b + alpha / 2) / alpha) as u16;
                    }
                    dst[3] = ((alpha + maximum / 2) / maximum) as u16;
                }
                let mut pixels = vec![173; count * 8 + 2];
                composite_samples(&output[1..count * 4 + 1], &source[1..], &mut pixels[1..count * 8 + 1], maximum);
                let actual: Vec<_> = pixels[1..count * 8 + 1].as_chunks::<2>().0.iter().map(|&bytes| u16::from_le_bytes(bytes)).collect();
                assert_eq!(actual, expected[1..count * 4 + 1]);
                assert_eq!((pixels[0], pixels[count * 8 + 1]), (173, 173));
            }
        }
    }

    fn putalpha_case<const C: usize>(count: usize) {
        let source: Vec<_> = (0..count * C + 1).map(|i| (i * 719 + 257) as u16).collect();
        let alpha: Vec<_> = (0..count + 1).map(|i| (i * 313 + 619) as u16).collect();
        let byte_alpha: Vec<u8> = alpha.iter().map(|&v| v as u8).collect();
        for maximum in [1023, 4095, 65535] {
            for byte in [false, true] {
                let samples = if byte { MaskSamples::Bytes(&byte_alpha[1..]) } else { MaskSamples::Wide(std::borrow::Cow::Borrowed(&alpha[1..])) };
                let mut output = vec![173; count * 8 + 2];
                putalpha_samples::<C>(&source[1..], &samples, &mut output[1..count * 8 + 1], maximum);
                for i in 0..count {
                    let a = if byte { ((u32::from(byte_alpha[1 + i]) * maximum + 127) / 255) as u16 } else { alpha[1 + i] };
                    let expected = [source[1 + i * C], source[2 + i * C], source[3 + i * C], a];
                    for (bytes, value) in output[1 + i * 8..1 + (i + 1) * 8].as_chunks::<2>().0.iter().zip(expected) {
                        assert_eq!(u16::from_le_bytes(*bytes), value);
                    }
                }
                assert_eq!((output[0], output[count * 8 + 1]), (173, 173));
            }
        }
    }

    #[test]
    fn alpha_replacement_preserves_channels_tails_and_guards() {
        for count in [0, 1, 15, 16, 17, 31, 32, 33, 65, 65537] {
            putalpha_case::<3>(count);
            putalpha_case::<4>(count);
        }
    }
}
