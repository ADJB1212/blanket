use std::simd::cmp::{SimdOrd, SimdPartialEq};
use std::simd::num::SimdUint;
use std::simd::{Select, Simd};

use blanket_core::pixels::{Samples, load_samples, store_samples};
use blanket_core::{Image, PixelMode, samples};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn matching(first: &Image, second: &Image) -> PyResult<()> {
    if first.mode != second.mode || first.bit_depth != second.bit_depth {
        return Err(PyValueError::new_err("images do not match"));
    }
    Ok(())
}

pub(crate) fn paste(image: &mut Image, source: &Image, position: (i64, i64), mask: Option<&Image>, fill: bool) -> PyResult<()> {
    matching(image, source)?;
    let source_values = samples::view(source)?;
    let mut values = samples::read(image)?;
    let mask_values = mask
        .map(|mask| {
            if mask.palette.is_some()
                || !matches!(mask.mode, PixelMode::L | PixelMode::Rgba)
                || (mask.width, mask.height) != (source.width, source.height)
                || mask.bit_depth != 8 && mask.bit_depth != image.bit_depth
            {
                return Err(PyValueError::new_err("bad transparency mask"));
            }
            let values = if mask.bit_depth == 8 {
                std::borrow::Cow::Owned(mask.pixel_data()?.iter().map(|&v| u16::from(v)).collect::<Vec<_>>())
            } else {
                samples::view(mask)?
            };
            Ok((values, mask.mode.channels(), (1_u64 << mask.bit_depth) - 1))
        })
        .transpose()?;
    let channels = image.mode.channels();
    let (left, top, right, bottom) =
        super::composite::paste_bounds(position, image.width as usize, image.height as usize, source.width, source.height);
    if right <= left || bottom <= top {
        return Ok(());
    }
    for y in top..bottom {
        let count = (right - left) as usize;
        let src_start = ((y - position.1) as usize * source.width as usize + (left - position.0) as usize) * channels;
        let dst_start = (y as usize * image.width as usize + left as usize) * channels;
        let dst = &mut values[dst_start..dst_start + count * channels];
        let src = &source_values[src_start..src_start + count * channels];
        let end = if let Some((mask, mc, maximum)) = &mask_values {
            let mask = &mask[src_start / channels * mc..];
            match channels {
                1 => paste_row::<1>(dst, src, mask, *mc, *maximum as u32, fill),
                3 => paste_row::<3>(dst, src, mask, *mc, *maximum as u32, fill),
                4 => paste_row::<4>(dst, src, mask, *mc, *maximum as u32, fill),
                _ => unreachable!(),
            }
        } else {
            dst.copy_from_slice(src);
            count
        };
        for x in left + end as i64..right {
            let src = ((y - position.1) as usize * source.width as usize + (x - position.0) as usize) * channels;
            let dst = (y as usize * image.width as usize + x as usize) * channels;
            let (alpha, maximum) = mask_values
                .as_ref()
                .map_or((1, 1), |(values, channels_mask, maximum)| (u64::from(values[(src / channels + 1) * channels_mask - 1]), *maximum));
            let replace_rgb =
                fill && channels == 4 && mask_values.as_ref().is_some_and(|(_, channels, _)| *channels == 1) && values[dst + 3] == 0 && alpha != 0;
            for c in 0..channels {
                let alpha = if replace_rgb && c < 3 { maximum } else { alpha };
                values[dst + c] =
                    ((u64::from(values[dst + c]) * (maximum - alpha) + u64::from(source_values[src + c]) * alpha + maximum / 2) / maximum) as u16;
            }
        }
    }
    image.pixels = samples::output(image, &values)?.pixels;
    Ok(())
}

fn paste_row<const C: usize>(dst: &mut [u16], src: &[u16], mask: &[u16], mc: usize, maximum: u32, fill: bool) -> usize {
    match maximum {
        255 => paste_row_max::<C, 255>(dst, src, mask, mc, fill),
        1023 => paste_row_max::<C, 1023>(dst, src, mask, mc, fill),
        4095 => paste_row_max::<C, 4095>(dst, src, mask, mc, fill),
        65535 => paste_row_max::<C, 65535>(dst, src, mask, mc, fill),
        _ => unreachable!("validated mask depth"),
    }
}

fn paste_row_max<const C: usize, const MAXIMUM: u32>(dst: &mut [u16], src: &[u16], mask: &[u16], mc: usize, fill: bool) -> usize {
    let end = dst.len() / C / 16 * 16;
    for i in (0..end).step_by(16) {
        let original = load_samples::<C>(&dst[i * C..]);
        let source = load_samples::<C>(&src[i * C..]);
        let alpha = if mc == 1 { Samples::from_slice(&mask[i..]) } else { load_samples::<4>(&mask[i * 4..])[3] }.cast::<u32>();
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
    let mut values = samples::read(background)?;
    let source = samples::view(overlay)?;
    let maximum = u64::from(samples::maximum(background));
    composite_samples(&mut values, &source, maximum);
    samples::output(background, &values)
}

fn composite_samples(values: &mut [u16], source: &[u16], maximum: u64) {
    let end = values.len() / 64 * 64;
    for i in (0..end).step_by(64) {
        let dst = load_samples::<4>(&values[i..]);
        let src = load_samples::<4>(&source[i..]);
        let a = src[3].cast::<u64>();
        let b = dst[3].cast::<u64>() * (Simd::splat(maximum) - a);
        let alpha = a * Simd::splat(maximum) + b;
        let denominator = alpha.simd_max(Simd::splat(1));
        let result: [Samples; 4] = std::array::from_fn(|c| {
            let value = if c == 3 {
                (alpha + Simd::splat(maximum / 2)) / Simd::splat(maximum)
            } else {
                (src[c].cast::<u64>() * a * Simd::splat(maximum) + dst[c].cast::<u64>() * b + denominator / Simd::splat(2)) / denominator
            };
            src[3].simd_eq(Samples::splat(0)).select(dst[c], value.cast())
        });
        store_samples(&mut values[i..], result);
    }
    for (dst, src) in values[end..].as_chunks_mut::<4>().0.iter_mut().zip(source[end..].as_chunks::<4>().0) {
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
}

pub(crate) fn putalpha(image: &mut Image, alpha: &Image) -> PyResult<()> {
    if alpha.bit_depth != 8 && alpha.bit_depth != image.bit_depth {
        return Err(PyValueError::new_err("images do not match"));
    }
    let source = samples::view(image)?;
    let maximum = u32::from(samples::maximum(image));
    let alpha_values = if alpha.bit_depth == 8 {
        std::borrow::Cow::Owned(alpha.pixel_data()?.iter().map(|&a| ((u32::from(a) * maximum + 127) / 255) as u16).collect::<Vec<_>>())
    } else {
        samples::view(alpha)?
    };
    let mut values = vec![0; alpha_values.len() * 4];
    match image.mode.channels() {
        3 => putalpha_samples::<3>(&source, &alpha_values, &mut values),
        4 => putalpha_samples::<4>(&source, &alpha_values, &mut values),
        _ => unreachable!(),
    }
    let result = Image::from_samples(image.width, image.height, PixelMode::Rgba, values, image.bit_depth, None)?;
    image.pixels = result.pixels;
    image.mode = PixelMode::Rgba;
    Ok(())
}

fn putalpha_samples<const C: usize>(source: &[u16], alpha: &[u16], output: &mut [u16]) {
    let end = alpha.len() / 16 * 16;
    for i in (0..end).step_by(16) {
        let input = load_samples::<C>(&source[i * C..]);
        store_samples(&mut output[i * 4..], [input[0], input[1], input[2], Samples::from_slice(&alpha[i..])]);
    }
    for ((dst, src), &a) in output[end * 4..].as_chunks_mut::<4>().0.iter_mut().zip(source[end * C..].as_chunks::<C>().0).zip(&alpha[end..]) {
        *dst = [src[0], src[1], src[2], a];
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
        let end = paste_row::<C>(&mut output[1..count * C + 1], &source[1..], &mask[1..], mc, maximum, fill);
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
                assert_eq!(paste_row::<1>(&mut output, &source, &mask, 1, maximum, false), count);
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
                composite_samples(&mut output[1..count * 4 + 1], &source[1..], maximum);
                assert_eq!(output, expected);
            }
        }
    }

    fn putalpha_case<const C: usize>(count: usize) {
        let source: Vec<_> = (0..count * C + 1).map(|i| (i * 719 + 257) as u16).collect();
        let alpha: Vec<_> = (0..count + 1).map(|i| (i * 313 + 619) as u16).collect();
        let mut output = vec![173; count * 4 + 2];
        putalpha_samples::<C>(&source[1..], &alpha[1..], &mut output[1..count * 4 + 1]);
        for i in 0..count {
            assert_eq!(&output[1 + i * 4..1 + i * 4 + 4], &[source[1 + i * C], source[2 + i * C], source[3 + i * C], alpha[1 + i]]);
        }
        assert_eq!((output[0], output[count * 4 + 1]), (173, 173));
    }

    #[test]
    fn alpha_replacement_preserves_channels_tails_and_guards() {
        for count in [0, 1, 15, 16, 17, 31, 32, 33, 65] {
            putalpha_case::<3>(count);
            putalpha_case::<4>(count);
        }
    }
}
