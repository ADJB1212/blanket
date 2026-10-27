use std::simd::cmp::SimdOrd;
use std::simd::num::{SimdFloat, SimdUint};
use std::simd::{Select, Simd, ToBytes};

use blanket_core::color::{LUMA_BITS, LUMA_BLUE, LUMA_GREEN, LUMA_RED, LUMA_ROUND};
use blanket_core::pixels::{Samples, load_samples, store_samples};
use blanket_core::{Image, PixelMode, samples};
use pyo3::prelude::*;

pub(crate) fn degenerate(py: Python<'_>, image: &Image, operation: &str) -> PyResult<Image> {
    let source = samples::view(image)?;
    let channels = image.mode.channels();
    let color_channels = if image.mode == PixelMode::Rgba { 3 } else { channels };
    let luma = |pixel: &[u16]| -> u16 {
        if channels == 1 {
            pixel[0]
        } else {
            ((u64::from(pixel[0]) * u64::from(LUMA_RED)
                + u64::from(pixel[1]) * u64::from(LUMA_GREEN)
                + u64::from(pixel[2]) * u64::from(LUMA_BLUE)
                + u64::from(LUMA_ROUND))
                >> LUMA_BITS) as u16
        }
    };
    let values = py.detach(|| {
        let mut values = source.to_vec();
        let end = source.len() / (channels * 16) * channels * 16;
        let mean = if operation != "contrast" || source.is_empty() {
            0
        } else {
            let mut sum = 0_u64;
            for i in (0..end).step_by(channels * 16) {
                let gray = match channels {
                    1 => Samples::from_slice(&source[i..]).cast::<u32>(),
                    3 => luminance(load_samples::<3>(&source[i..])).cast(),
                    4 => luminance(load_samples::<4>(&source[i..])).cast(),
                    _ => unreachable!(),
                };
                sum += u64::from(gray.reduce_sum());
            }
            sum += source[end..].chunks_exact(channels).map(|pixel| u64::from(luma(pixel))).sum::<u64>();
            (sum as f64 / (source.len() / channels) as f64 + 0.5) as u16
        };
        if operation == "sharpness" {
            smooth(&source, &mut values, image.width as usize, image.height as usize, channels);
            return values;
        }
        for i in (0..end).step_by(channels * 16) {
            match channels {
                1 => {
                    let value = if operation == "color" {
                        Samples::from_slice(&source[i..])
                    } else {
                        Samples::splat(if operation == "brightness" { 0 } else { mean })
                    };
                    value.copy_to_slice(&mut values[i..i + 16]);
                }
                3 => degenerate_block::<3>(&source[i..], &mut values[i..], operation, mean),
                4 => degenerate_block::<4>(&source[i..], &mut values[i..], operation, mean),
                _ => unreachable!(),
            }
        }
        for (pixel_index, pixel) in values[end..].chunks_exact_mut(channels).enumerate() {
            let i = end / channels + pixel_index;
            match operation {
                "brightness" => pixel[..color_channels].fill(0),
                "color" => pixel[..color_channels].fill(luma(&source[i * channels..])),
                "contrast" => pixel[..color_channels].fill(mean),
                _ => unreachable!(),
            }
        }
        values
    });
    samples::output(image, &values)
}

pub(crate) fn blend(py: Python<'_>, first: &Image, second: &Image, factor: f32) -> PyResult<Image> {
    let a = samples::view(first)?;
    let b = samples::view(second)?;
    if std::ptr::eq(first, second) || factor == 0.0 {
        return samples::output(first, &a);
    }
    if factor == 1.0 {
        return samples::output(second, &b);
    }
    let maximum = samples::maximum(first);
    let mut pixels = blanket_core::raster::alloc_pixels(a.len() * 2)?;
    py.detach(|| blend_samples(&a, &b, &mut pixels, factor, maximum, first.mode == PixelMode::I16B));
    samples::output_bytes(first, pixels)
}

fn luminance<const C: usize>(pixel: [Samples; C]) -> Samples {
    ((pixel[0].cast::<u32>() * Simd::splat(LUMA_RED)
        + pixel[1].cast::<u32>() * Simd::splat(LUMA_GREEN)
        + pixel[2].cast::<u32>() * Simd::splat(LUMA_BLUE)
        + Simd::splat(LUMA_ROUND))
        >> LUMA_BITS)
        .cast()
}

fn degenerate_block<const C: usize>(source: &[u16], output: &mut [u16], operation: &str, mean: u16) {
    let mut pixel = load_samples::<C>(source);
    let gray = if operation == "color" { luminance(pixel) } else { Samples::splat(if operation == "brightness" { 0 } else { mean }) };
    pixel[..3].fill(gray);
    store_samples(output, pixel);
}

fn smooth(source: &[u16], output: &mut [u16], width: usize, height: usize, channels: usize) {
    if width < 3 || height < 3 {
        return;
    }
    let stride = width * channels;
    blanket_core::parallel::chunks_mut_above(output, stride * 32, 256 * 1024, |band, rows| {
        for (row_index, row) in rows.chunks_exact_mut(stride).enumerate() {
            let y = band * 32 + row_index;
            if y == 0 || y + 1 == height {
                continue;
            }
            let first = channels;
            let last = stride - channels;
            let end = first + (last - first) / 16 * 16;
            for x in (first..end).step_by(16) {
                let i = y * stride + x;
                let original = Samples::from_slice(&source[i..]);
                let mut sum = original.cast::<u32>() * Simd::splat(4);
                for row in [i - stride, i, i + stride] {
                    for start in [row - channels, row, row + channels] {
                        sum += Samples::from_slice(&source[start..]).cast::<u32>();
                    }
                }
                let value = ((sum + Simd::splat(6)) / Simd::splat(13)).cast::<u16>();
                let alpha = std::simd::Mask::<i16, 16>::from_array(std::array::from_fn(|lane| channels == 4 && (i + lane) % 4 == 3));
                alpha.select(original, value).copy_to_slice(&mut row[x..x + 16]);
            }
            for x in end..last {
                let i = y * stride + x;
                if channels == 4 && i % 4 == 3 {
                    continue;
                }
                let mut sum = u32::from(source[i]) * 4;
                for row in [i - stride, i, i + stride] {
                    for start in [row - channels, row, row + channels] {
                        sum += u32::from(source[start]);
                    }
                }
                row[x] = ((sum + 6) / 13) as u16;
            }
        }
    });
}

fn blend_samples(a: &[u16], b: &[u16], output: &mut [u8], factor: f32, maximum: u16, big_endian: bool) {
    const CHUNK: usize = 128 * 1024;
    blanket_core::parallel::chunks_mut_above(output, CHUNK, 1024 * 1024, |chunk, dst| {
        let start = chunk * CHUNK / 2;
        let a = &a[start..][..dst.len() / 2];
        let b = &b[start..][..dst.len() / 2];
        if big_endian {
            blend_chunk::<true>(a, b, dst, factor, maximum);
        } else {
            blend_chunk::<false>(a, b, dst, factor, maximum);
        }
    });
}

fn blend_chunk<const BIG_ENDIAN: bool>(a: &[u16], b: &[u16], output: &mut [u8], factor: f32, maximum: u16) {
    let end = output.len() / 32 * 16;
    for i in (0..end).step_by(16) {
        let a = Samples::from_slice(&a[i..]).cast::<f32>();
        let b = Samples::from_slice(&b[i..]).cast::<f32>();
        #[cfg(target_arch = "aarch64")]
        let value = {
            use std::simd::StdFloat;
            Simd::splat(factor).mul_add(b - a, a)
        };
        #[cfg(not(target_arch = "aarch64"))]
        let value = a + Simd::splat(factor) * (b - a);
        let value = value.cast::<u16>().simd_min(Samples::splat(maximum));
        let bytes = if BIG_ENDIAN { value.to_be_bytes() } else { value.to_le_bytes() };
        bytes.copy_to_slice(&mut output[i * 2..i * 2 + 32]);
    }
    for ((&a, &b), dst) in a[end..].iter().zip(&b[end..]).zip(output[end * 2..].as_chunks_mut::<2>().0) {
        let (a, b) = (f32::from(a), f32::from(b));
        #[cfg(target_arch = "aarch64")]
        let value = factor.mul_add(b - a, a);
        #[cfg(not(target_arch = "aarch64"))]
        let value = a + factor * (b - a);
        let value = value.clamp(0.0, f32::from(maximum)) as u16;
        *dst = if BIG_ENDIAN { value.to_be_bytes() } else { value.to_le_bytes() };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blend_preserves_scalar_rounding_nonfinite_values_tails_and_guards() {
        for count in [0, 1, 15, 16, 17, 31, 32, 33, 65, 524289] {
            for maximum in [1023, 4095, 65535] {
                let a: Vec<_> = (0..count + 1).map(|i| ((i * 719) % (usize::from(maximum) + 1)) as u16).collect();
                let b: Vec<_> = a.iter().rev().copied().collect();
                for factor in [-2.0, 0.0, 0.1, 0.5, 1.0, 1.75, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                    for big_endian in [false, true] {
                        let mut output = vec![173; count * 2 + 2];
                        blend_samples(&a[1..], &b[1..], &mut output[1..count * 2 + 1], factor, maximum, big_endian);
                        for i in 0..count {
                            let (a, b) = (f32::from(a[i + 1]), f32::from(b[i + 1]));
                            #[cfg(target_arch = "aarch64")]
                            let expected = factor.mul_add(b - a, a);
                            #[cfg(not(target_arch = "aarch64"))]
                            let expected = a + factor * (b - a);
                            let bytes = [output[i * 2 + 1], output[i * 2 + 2]];
                            let actual = if big_endian { u16::from_be_bytes(bytes) } else { u16::from_le_bytes(bytes) };
                            assert_eq!(actual, expected.clamp(0.0, f32::from(maximum)) as u16);
                        }
                        assert_eq!((output[0], output[count * 2 + 1]), (173, 173));
                    }
                }
            }
        }
    }

    #[test]
    fn smooth_preserves_parallel_bands_borders_alpha_and_guards() {
        for threads in [1, 4] {
            rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap().install(|| {
                let (width, height) = (517, 263);
                for channels in [1, 3, 4] {
                    let count = width * height * channels;
                    let storage: Vec<_> = (0..count + 1).map(|i| (i * 719 + 257) as u16).collect();
                    let source = &storage[1..];
                    let mut output = vec![173; count + 2];
                    output[1..count + 1].copy_from_slice(source);
                    smooth(source, &mut output[1..count + 1], width, height, channels);
                    for y in 0..height {
                        for x in 0..width {
                            for channel in 0..channels {
                                let i = (y * width + x) * channels + channel;
                                let expected = if x == 0 || x + 1 == width || y == 0 || y + 1 == height || channels == 4 && channel == 3 {
                                    source[i]
                                } else {
                                    let mut sum = u32::from(source[i]) * 4;
                                    for yy in y - 1..=y + 1 {
                                        for xx in x - 1..=x + 1 {
                                            sum += u32::from(source[(yy * width + xx) * channels + channel]);
                                        }
                                    }
                                    ((sum + 6) / 13) as u16
                                };
                                assert_eq!(output[i + 1], expected);
                            }
                        }
                    }
                    assert_eq!((output[0], output[count + 1]), (173, 173));
                }
            });
        }
    }

    #[test]
    fn smooth_preserves_borders_alpha_and_scalar_rounding() {
        for width in [1, 3, 7, 15, 16, 17, 33] {
            for channels in [1, 3, 4] {
                let height = 5;
                let count = width * height * channels;
                let backing: Vec<_> = (0..count + 1).map(|i| (i * 719 + 257) as u16).collect();
                let source = &backing[1..];
                let mut output = vec![173; count + 2];
                output[1..count + 1].copy_from_slice(source);
                smooth(source, &mut output[1..count + 1], width, height, channels);
                for y in 0..height {
                    for x in 0..width {
                        for c in 0..channels {
                            let i = (y * width + x) * channels + c;
                            let expected = if x == 0 || y == 0 || x + 1 == width || y + 1 == height || channels == 4 && c == 3 {
                                source[i]
                            } else {
                                let mut sum = u32::from(source[i]) * 4;
                                for yy in y - 1..=y + 1 {
                                    for xx in x - 1..=x + 1 {
                                        sum += u32::from(source[(yy * width + xx) * channels + c]);
                                    }
                                }
                                ((sum + 6) / 13) as u16
                            };
                            assert_eq!(output[i + 1], expected);
                        }
                    }
                }
                assert_eq!((output[0], output[count + 1]), (173, 173));
            }
        }
    }
}
