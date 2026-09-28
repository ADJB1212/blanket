use crate::parallel::{CHUNK_PIXELS, chunks_mut};
use crate::pixels::{Bytes, load, store};
use crate::raster::PixelMode;
use std::simd::{
    Simd,
    cmp::SimdOrd,
    num::{SimdFloat, SimdInt},
};

pub fn float_to_integer(source: &[u8]) -> Vec<u8> {
    let mut output = vec![0; source.len()];
    chunks_mut(&mut output, CHUNK_PIXELS * 4, |chunk, dst| {
        let src = &source[chunk * CHUNK_PIXELS * 4..][..dst.len()];
        let mut offset = 0;
        while offset + 16 <= dst.len() {
            let values = Simd::<f32, 4>::from_array(std::array::from_fn(|i| {
                f32::from_le_bytes(src[offset + i * 4..offset + i * 4 + 4].try_into().unwrap())
            }));
            for (dst, value) in dst[offset..offset + 16]
                .as_chunks_mut::<4>()
                .0
                .iter_mut()
                .zip(values.cast::<i32>().to_array())
            {
                *dst = value.to_le_bytes();
            }
            offset += 16;
        }
        for (src, dst) in src[offset..].as_chunks::<4>().0.iter().zip(dst[offset..].as_chunks_mut::<4>().0) {
            *dst = (f32::from_le_bytes(*src) as i32).to_le_bytes();
        }
    });
    output
}

pub fn integer_to_l(source: &[u8], mode: PixelMode) -> Vec<u8> {
    let stride = mode.sample_bytes();
    let mut output = vec![0; source.len() / stride];
    crate::parallel::chunks_mut_above(&mut output, CHUNK_PIXELS, 2 * 1024 * 1024, |chunk, dst| {
        let src = &source[chunk * CHUNK_PIXELS * stride..(chunk * CHUNK_PIXELS + dst.len()) * stride];
        match mode {
            PixelMode::I => {
                for (pixel, bytes) in dst.iter_mut().zip(src.as_chunks::<4>().0) {
                    *pixel = i32::from_le_bytes(*bytes).clamp(0, 255) as u8;
                }
            }
            PixelMode::I16B => {
                for (pixel, bytes) in dst.iter_mut().zip(src.as_chunks::<2>().0) {
                    *pixel = u16::from_be_bytes(*bytes).min(255) as u8;
                }
            }
            _ => {
                for (pixel, bytes) in dst.iter_mut().zip(src.as_chunks::<2>().0) {
                    *pixel = u16::from_le_bytes(*bytes).min(255) as u8;
                }
            }
        }
    });
    output
}

pub fn integer_to_color(source: &[u8], mode: PixelMode, channels: usize) -> Vec<u8> {
    debug_assert!(matches!(channels, 3 | 4));
    let stride = mode.sample_bytes();
    let mut output = vec![0; source.len() / stride * channels];
    chunks_mut(&mut output, CHUNK_PIXELS * channels, |chunk, dst| {
        let start = chunk * CHUNK_PIXELS * stride;
        let src = &source[start..start + dst.len() / channels * stride];
        for (bytes, pixel) in src.chunks_exact(stride).zip(dst.chunks_exact_mut(channels)) {
            let value = match mode {
                PixelMode::I => i32::from_le_bytes(bytes.try_into().unwrap()).clamp(0, 255) as u8,
                PixelMode::I16B => u16::from_be_bytes(bytes.try_into().unwrap()).min(255) as u8,
                _ => u16::from_le_bytes(bytes.try_into().unwrap()).min(255) as u8,
            };
            pixel[..3].fill(value);
            if channels == 4 {
                pixel[3] = 255;
            }
        }
    });
    output
}

pub fn convert_integer(source: &[u8], from: PixelMode, to: PixelMode) -> Vec<u8> {
    let source_bytes = from.sample_bytes();
    let target_bytes = to.sample_bytes();
    let mut output = vec![0; source.len() / source_bytes * target_bytes];
    chunks_mut(&mut output, CHUNK_PIXELS * target_bytes, |chunk, dst| {
        let src = &source[chunk * CHUNK_PIXELS * source_bytes..(chunk * CHUNK_PIXELS + dst.len() / target_bytes) * source_bytes];
        let offset = dst.len() / target_bytes / 8 * 8;
        for i in (0..offset).step_by(8) {
            let values = Simd::<i32, 8>::from_array(std::array::from_fn(|lane| {
                let input = &src[(i + lane) * source_bytes..(i + lane + 1) * source_bytes];
                match from {
                    PixelMode::I => i32::from_le_bytes(input.try_into().unwrap()),
                    PixelMode::I16B => i32::from(u16::from_be_bytes(input.try_into().unwrap())),
                    _ => i32::from(u16::from_le_bytes(input.try_into().unwrap())),
                }
            }));
            if to == PixelMode::I {
                for (dst, value) in dst[i * 4..(i + 8) * 4].as_chunks_mut::<4>().0.iter_mut().zip(values.to_array()) {
                    *dst = value.to_le_bytes();
                }
            } else {
                let values = values.simd_clamp(Simd::splat(0), Simd::splat(65535)).cast::<u16>();
                for (dst, value) in dst[i * 2..(i + 8) * 2].as_chunks_mut::<2>().0.iter_mut().zip(values.to_array()) {
                    *dst = if to == PixelMode::I16B {
                        value.to_be_bytes()
                    } else {
                        value.to_le_bytes()
                    };
                }
            }
        }
        for (input, output) in src[offset * source_bytes..]
            .chunks_exact(source_bytes)
            .zip(dst[offset * target_bytes..].chunks_exact_mut(target_bytes))
        {
            let value = match from {
                PixelMode::I => i64::from(i32::from_le_bytes(input.try_into().unwrap())),
                PixelMode::I16B => i64::from(u16::from_be_bytes(input.try_into().unwrap())),
                _ => i64::from(u16::from_le_bytes(input.try_into().unwrap())),
            };
            if to == PixelMode::I {
                output.copy_from_slice(&(value as i32).to_le_bytes());
            } else {
                let value = value.clamp(0, 65535) as u16;
                let bytes = if to == PixelMode::I16B {
                    value.to_be_bytes()
                } else {
                    value.to_le_bytes()
                };
                output.copy_from_slice(&bytes);
            }
        }
    });
    output
}

pub fn convert(source: &[u8], from: PixelMode, to: PixelMode) -> Vec<u8> {
    debug_assert_ne!(from, to, "caller should short-circuit identity conversion");

    match (from, to) {
        (PixelMode::Rgba, PixelMode::Rgb) => convert_layout::<4, 3>(source),
        (PixelMode::Rgb, PixelMode::Rgba) => convert_layout::<3, 4>(source),
        (PixelMode::L, PixelMode::Rgb) => convert_layout::<1, 3>(source),
        (PixelMode::L, PixelMode::Rgba) => convert_layout::<1, 4>(source),
        (PixelMode::Rgb, PixelMode::L) => color_to_gray::<3>(source),
        (PixelMode::Rgba, PixelMode::L) => color_to_gray::<4>(source),
        _ => unreachable!("all mode pairs are covered"),
    }
}

fn map_blocks<const S: usize, const D: usize>(
    source: &[u8], parallel_bytes: usize, block: impl Fn([Bytes; S]) -> [Bytes; D] + Sync, pixel: impl Fn(&[u8; S]) -> [u8; D] + Sync,
) -> Vec<u8> {
    let mut output = vec![0; source.len() / S * D];
    let chunk_pixels = 64 * 1024;
    crate::parallel::chunks_mut_above(&mut output, chunk_pixels * D, parallel_bytes, |i, dst| {
        let src = &source[i * chunk_pixels * S..(i * chunk_pixels + dst.len() / D) * S];
        let end = src.len() / S / 16 * 16;
        for i in (0..end).step_by(16) {
            store(&mut dst[i * D..], block(load::<S>(&src[i * S..])));
        }
        for (src, dst) in src[end * S..].as_chunks::<S>().0.iter().zip(dst[end * D..].as_chunks_mut::<D>().0) {
            *dst = pixel(src);
        }
    });
    output
}

pub fn gray_to_hsv<const C: usize>(source: &[u8]) -> Vec<u8> {
    map_blocks::<C, 3>(source, 5 * 1024 * 1024, |p| [Bytes::splat(0), Bytes::splat(0), p[0]], |p| [0, 0, p[0]])
}

/// Pillow's L to CMYK stores inverted gray as K.
pub fn gray_to_cmyk(source: &[u8]) -> Vec<u8> {
    map_blocks::<1, 4>(
        source,
        4 * 1024 * 1024,
        |p| [Bytes::splat(0), Bytes::splat(0), Bytes::splat(0), !p[0]],
        |p| [0, 0, 0, 255 - p[0]],
    )
}

pub fn gray_to_ycbcr(source: &[u8]) -> Vec<u8> {
    map_blocks::<1, 3>(
        source,
        4 * 1024 * 1024,
        |p| [p[0], Bytes::splat(128), Bytes::splat(128)],
        |p| [p[0], 128, 128],
    )
}

pub fn color_to_cmyk<const S: usize>(source: &[u8]) -> Vec<u8> {
    map_blocks::<S, 4>(
        source,
        4 * 1024 * 1024,
        |p| [!p[0], !p[1], !p[2], Bytes::splat(0)],
        |p| [255 - p[0], 255 - p[1], 255 - p[2], 0],
    )
}

pub fn extract_channel<const S: usize>(source: &[u8], channel: usize) -> Vec<u8> {
    map_blocks::<S, 1>(source, 1024 * 1024, |p| [p[channel]], |p| [p[channel]])
}

pub fn convert_la(source: &[u8], to: PixelMode) -> Vec<u8> {
    let channels = to.channels();
    let mut output = vec![0; source.len() / 2 * channels];
    crate::parallel::chunks_mut_above(&mut output, CHUNK_PIXELS * channels, 2 * 1024 * 1024, |chunk, dst| {
        let start = chunk * CHUNK_PIXELS * 2;
        let src = &source[start..start + dst.len() / channels * 2];
        let offset = src.len() / 2 / 16 * 16;
        for i in (0..offset).step_by(16) {
            let [l, a] = load::<2>(&src[i * 2..]);
            let dst = &mut dst[i * channels..];
            match to {
                PixelMode::L => l.copy_to_slice(&mut dst[..16]),
                PixelMode::Rgb => store(dst, [l; 3]),
                PixelMode::Rgba => store(dst, [l, l, l, a]),
                _ => unreachable!(),
            }
        }
        for (pair, pixel) in src[offset * 2..]
            .as_chunks::<2>()
            .0
            .iter()
            .zip(dst[offset * channels..].chunks_exact_mut(channels))
        {
            match to {
                PixelMode::L => pixel[0] = pair[0],
                PixelMode::Rgb => pixel.copy_from_slice(&[pair[0]; 3]),
                PixelMode::Rgba => pixel.copy_from_slice(&[pair[0], pair[0], pair[0], pair[1]]),
                _ => unreachable!(),
            }
        }
    });
    output
}

pub fn extract_two_channel(source: &[u8], channel: usize) -> Vec<u8> {
    extract_channel::<2>(source, channel)
}

fn convert_layout<const S: usize, const C: usize>(source: &[u8]) -> Vec<u8> {
    map_blocks::<S, C>(
        source,
        2 * 1024 * 1024,
        |p| std::array::from_fn(|c| if c == 3 { Bytes::splat(255) } else { p[if S == 1 { 0 } else { c }] }),
        |p| std::array::from_fn(|c| if c == 3 { 255 } else { p[if S == 1 { 0 } else { c }] }),
    )
}

fn color_to_gray<const SOURCE_CHANNELS: usize>(source: &[u8]) -> Vec<u8> {
    let pixel_count = source.len() / SOURCE_CHANNELS;
    let mut output = vec![0u8; pixel_count];
    let (colors, remainder) = source.as_chunks::<SOURCE_CHANNELS>();
    debug_assert!(remainder.is_empty());
    chunks_mut(&mut output, CHUNK_PIXELS, |i, dst| {
        for (color, gray) in colors[i * CHUNK_PIXELS..].iter().zip(dst) {
            *gray = pillow_luma(color[0], color[1], color[2]);
        }
    });
    output
}

#[inline(always)]
pub fn pillow_luma(r: u8, g: u8, b: u8) -> u8 {
    ((u32::from(r) * 19_595 + u32::from(g) * 38_470 + u32::from(b) * 7_471 + 0x8000) >> 16) as u8
}

/// Fixed divisors let LLVM vectorize exact wide-sample normalization. Preserve
/// the existing rounded full-range mapping, including 16-bit identity input.
pub fn normalize_u16(source: &[u8], depth: u8) -> Vec<u16> {
    use crate::parallel::should_parallel;
    use rayon::prelude::*;

    const CHUNK_PIXELS: usize = 64 * 1024;
    const MEMORY_PARALLEL_BYTES: usize = 4 * 1024 * 1024;

    fn convert<const MAX: u32>(source: &[u8]) -> Vec<u16> {
        let mut output = vec![0; source.len() / 2];
        let fill = |i: usize, dst: &mut [u16]| {
            let src = &source[i * CHUNK_PIXELS * 2..(i * CHUNK_PIXELS + dst.len()) * 2];
            for (v, dst) in src.as_chunks::<2>().0.iter().zip(dst) {
                *dst = ((u32::from(u16::from_le_bytes(*v)) * 65535 + MAX / 2) / MAX) as u16;
            }
        };
        if should_parallel(source.len(), CHUNK_PIXELS * 2, MEMORY_PARALLEL_BYTES) {
            output.par_chunks_mut(CHUNK_PIXELS).enumerate().for_each(|(i, dst)| fill(i, dst));
        } else {
            output.chunks_mut(CHUNK_PIXELS).enumerate().for_each(|(i, dst)| fill(i, dst));
        }
        output
    }
    match depth {
        10 => convert::<1023>(source),
        12 => convert::<4095>(source),
        16 => convert::<65535>(source),
        _ => unreachable!("validated wide depth"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float_cast_matches_scalar_at_vector_boundaries() {
        let values = [
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            -0.0,
            0.0,
            1.75,
            -1.75,
            2147483648.0,
            -2147483648.0,
            2147483520.0,
        ];
        for count in [0, 1, 3, 4, 5, 15, 16, 17, 33] {
            let source: Vec<_> = std::iter::once(93)
                .chain((0..count).flat_map(|i| values[i % values.len()].to_le_bytes()))
                .collect();
            let expected: Vec<_> = (0..count).flat_map(|i| (values[i % values.len()] as i32).to_le_bytes()).collect();
            assert_eq!(float_to_integer(&source[1..]), expected);
        }
    }

    #[test]
    fn rgba_to_rgb_basic() {
        let rgba: Vec<u8> = (0..256).flat_map(|i| [i as u8, (i * 2) as u8, (i * 3) as u8, 0xAA]).collect();
        let result = convert(&rgba, PixelMode::Rgba, PixelMode::Rgb);
        for i in 0..256 {
            assert_eq!(result[i * 3], rgba[i * 4], "R mismatch at pixel {i}");
            assert_eq!(result[i * 3 + 1], rgba[i * 4 + 1], "G mismatch at pixel {i}");
            assert_eq!(result[i * 3 + 2], rgba[i * 4 + 2], "B mismatch at pixel {i}");
        }
    }

    #[test]
    fn integer_to_color_matches_gray_expansion() {
        for mode in [PixelMode::I, PixelMode::I16, PixelMode::I16L, PixelMode::I16B] {
            let values = [-10_i32, 0, 1, 127, 255, 256, 65535, 100000];
            for count in [0, 1, 15, 16, 17, 257] {
                let source: Vec<u8> = values
                    .iter()
                    .cycle()
                    .take(count)
                    .flat_map(|&value| match mode {
                        PixelMode::I => value.to_le_bytes().to_vec(),
                        PixelMode::I16B => (value as u16).to_be_bytes().to_vec(),
                        _ => (value as u16).to_le_bytes().to_vec(),
                    })
                    .collect();
                let gray = integer_to_l(&source, mode);
                for destination in [PixelMode::Rgb, PixelMode::Rgba] {
                    assert_eq!(
                        integer_to_color(&source, mode, destination.channels()),
                        convert(&gray, PixelMode::L, destination)
                    );
                }
            }
        }
    }

    #[test]
    fn rgb_to_rgba_basic() {
        let rgb: Vec<u8> = (0..256).flat_map(|i| [i as u8, (i * 2) as u8, (i * 3) as u8]).collect();
        let result = convert(&rgb, PixelMode::Rgb, PixelMode::Rgba);
        for i in 0..256 {
            assert_eq!(result[i * 4], rgb[i * 3], "R mismatch at pixel {i}");
            assert_eq!(result[i * 4 + 1], rgb[i * 3 + 1], "G mismatch at pixel {i}");
            assert_eq!(result[i * 4 + 2], rgb[i * 3 + 2], "B mismatch at pixel {i}");
            assert_eq!(result[i * 4 + 3], 255, "A mismatch at pixel {i}");
        }
    }

    #[test]
    fn l_to_rgb_basic() {
        let luma: Vec<u8> = (0..=255).collect();
        let result = convert(&luma, PixelMode::L, PixelMode::Rgb);
        for (i, &l) in luma.iter().enumerate() {
            assert_eq!(result[i * 3], l);
            assert_eq!(result[i * 3 + 1], l);
            assert_eq!(result[i * 3 + 2], l);
        }
    }

    #[test]
    fn l_to_rgba_basic() {
        let luma: Vec<u8> = (0..=255).collect();
        let result = convert(&luma, PixelMode::L, PixelMode::Rgba);
        for (i, &l) in luma.iter().enumerate() {
            assert_eq!(result[i * 4], l);
            assert_eq!(result[i * 4 + 1], l);
            assert_eq!(result[i * 4 + 2], l);
            assert_eq!(result[i * 4 + 3], 255);
        }
    }

    #[test]
    fn rgb_to_l_matches_pillow_luminance() {
        let rgb: Vec<u8> = (0u8..=u8::MAX)
            .flat_map(|value| [value, value.wrapping_mul(37), value.wrapping_add(113)])
            .collect();
        let result = convert(&rgb, PixelMode::Rgb, PixelMode::L);
        let expected = scalar_luminance_rgb(&rgb);
        assert_eq!(result, expected);
    }

    #[test]
    fn rgba_to_l_matches_pillow_luminance() {
        let rgba: Vec<u8> = (0u8..=u8::MAX)
            .flat_map(|value| [value, value.wrapping_mul(37), value.wrapping_add(113), value.wrapping_mul(19)])
            .collect();
        let result = convert(&rgba, PixelMode::Rgba, PixelMode::L);
        let expected = scalar_luminance_rgba(&rgba);
        assert_eq!(result, expected);
    }

    #[test]
    fn handles_non_simd_aligned_sizes() {
        let rgba: Vec<u8> = (0..68).collect();
        let result = convert(&rgba, PixelMode::Rgba, PixelMode::Rgb);
        assert_eq!(result.len(), 51);
        for i in 0..17 {
            assert_eq!(result[i * 3], rgba[i * 4]);
            assert_eq!(result[i * 3 + 1], rgba[i * 4 + 1]);
            assert_eq!(result[i * 3 + 2], rgba[i * 4 + 2]);
        }
    }

    #[test]
    fn handles_empty_input() {
        assert!(convert(&[], PixelMode::Rgba, PixelMode::Rgb).is_empty());
        assert!(convert(&[], PixelMode::Rgb, PixelMode::L).is_empty());
        assert!(convert(&[], PixelMode::L, PixelMode::Rgba).is_empty());
    }

    #[test]
    fn handles_single_pixel() {
        assert_eq!(convert(&[10, 20, 30, 40], PixelMode::Rgba, PixelMode::Rgb), [10, 20, 30]);
        assert_eq!(convert(&[10, 20, 30], PixelMode::Rgb, PixelMode::Rgba), [10, 20, 30, 255]);
        assert_eq!(convert(&[42], PixelMode::L, PixelMode::Rgb), [42, 42, 42]);
    }

    fn scalar_luminance_rgb(source: &[u8]) -> Vec<u8> {
        source
            .as_chunks::<3>()
            .0
            .iter()
            .map(|px| ((px[0] as u32 * 19_595 + px[1] as u32 * 38_470 + px[2] as u32 * 7_471 + 0x8000) >> 16) as u8)
            .collect()
    }

    fn scalar_luminance_rgba(source: &[u8]) -> Vec<u8> {
        source
            .as_chunks::<4>()
            .0
            .iter()
            .map(|px| ((px[0] as u32 * 19_595 + px[1] as u32 * 38_470 + px[2] as u32 * 7_471 + 0x8000) >> 16) as u8)
            .collect()
    }

    #[test]
    fn wide_normalization_matches_original_rounding_exhaustively() {
        for depth in [10, 12, 16] {
            let maximum = (1_u32 << depth) - 1;
            let src: Vec<_> = (0..=maximum).flat_map(|v| (v as u16).to_le_bytes()).collect();
            let expected: Vec<_> = (0..=maximum).map(|v| ((v * 65535 + maximum / 2) / maximum) as u16).collect();
            assert_eq!(normalize_u16(&src, depth), expected);
        }
    }
}
