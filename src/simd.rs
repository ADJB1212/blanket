use crate::parallel::{CHUNK_PIXELS, chunks_mut};
use crate::raster::PixelMode;

pub(crate) fn convert(source: &[u8], from: PixelMode, to: PixelMode) -> Vec<u8> {
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

fn convert_layout<const S: usize, const C: usize>(source: &[u8]) -> Vec<u8> {
    let count = source.len() / S;
    #[cfg(not(target_arch = "aarch64"))]
    if S != 1 || C == 4 {
        // Keep garb's runtime-dispatched SIMD on other architectures.
        let conversion = match (S, C) {
            (3, 4) => garb::bytes::rgb_to_rgba,
            (4, 3) => garb::bytes::rgba_to_rgb,
            (1, 4) => garb::bytes::gray_to_rgba,
            _ => unreachable!(),
        };
        let mut output = vec![0; count * C];
        crate::parallel::chunks_mut_above(&mut output, 256 * 1024 * C, 2 * 1024 * 1024, |i, dst| {
            let start = i * 256 * 1024 * S;
            conversion(&source[start..start + dst.len() / C * S], dst).expect("validated image buffers have matching pixel counts");
        });
        return output;
    }
    let mut output = Vec::<u8>::with_capacity(count * C);
    let spare = &mut output.spare_capacity_mut()[..count * C];
    let chunk_pixels = if S == 4 { 64 * 1024 } else { 256 * 1024 };
    let fill = |i: usize, dst: &mut [std::mem::MaybeUninit<u8>]| {
        let src = &source[i * chunk_pixels * S..(i * chunk_pixels + dst.len() / C) * S];
        #[cfg(target_arch = "aarch64")]
        let offset = {
            use std::arch::aarch64::*;
            let mut offset = 0;
            // NEON is mandatory on AArch64. These stores initialize exactly
            // 16 complete pixels within the allocated spare capacity.
            unsafe {
                while offset + 16 <= src.len() / S {
                    let ptr = src.as_ptr().add(offset * S);
                    if S == 4 && C == 3 {
                        // Compact 16 packed RGBA pixels with byte tables,
                        // avoiding a full deinterleave followed by interleave.
                        let rgba = vld1q_u8_x4(ptr);
                        let a = vld1q_u8([0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20].as_ptr());
                        let b = vld1q_u8([21, 22, 24, 25, 26, 28, 29, 30, 32, 33, 34, 36, 37, 38, 40, 41].as_ptr());
                        let c = vld1q_u8([42, 44, 45, 46, 48, 49, 50, 52, 53, 54, 56, 57, 58, 60, 61, 62].as_ptr());
                        vst1q_u8_x3(
                            dst.as_mut_ptr().cast::<u8>().add(offset * C),
                            uint8x16x3_t(vqtbl4q_u8(rgba, a), vqtbl4q_u8(rgba, b), vqtbl4q_u8(rgba, c)),
                        );
                        offset += 16;
                        continue;
                    }
                    let (r, g, b) = if S == 1 {
                        let gray = vld1q_u8(ptr);
                        (gray, gray, gray)
                    } else if S == 3 {
                        let rgb = vld3q_u8(ptr);
                        (rgb.0, rgb.1, rgb.2)
                    } else {
                        let rgba = vld4q_u8(ptr);
                        (rgba.0, rgba.1, rgba.2)
                    };
                    let ptr = dst.as_mut_ptr().cast::<u8>().add(offset * C);
                    if C == 3 {
                        vst3q_u8(ptr, uint8x16x3_t(r, g, b));
                    } else {
                        vst4q_u8(ptr, uint8x16x4_t(r, g, b, vdupq_n_u8(255)));
                    }
                    offset += 16;
                }
            }
            offset
        };
        #[cfg(not(target_arch = "aarch64"))]
        let offset = 0;
        for (src, pixel) in src[offset * S..].as_chunks::<S>().0.iter().zip(dst[offset * C..].as_chunks_mut::<C>().0) {
            for (channel, value) in pixel.iter_mut().enumerate() {
                value.write(if channel == 3 { 255 } else { src[if S == 1 { 0 } else { channel }] });
            }
        }
    };
    // Cheap grayscale expansion stays serial while its output fits in cache.
    let parallel_bytes = match S {
        1 => 5 * 1024 * 1024,
        4 => 2 * 1024 * 1024,
        _ => 2 * 1024 * 1024,
    };
    if spare.len() >= parallel_bytes {
        use rayon::prelude::*;
        spare.par_chunks_mut(chunk_pixels * C).enumerate().for_each(|(i, dst)| fill(i, dst));
    } else {
        spare.chunks_mut(chunk_pixels * C).enumerate().for_each(|(i, dst)| fill(i, dst));
    }
    // Every byte was initialized above, including the scalar tail of each
    // disjoint chunk. A panic before completion leaves the vector length zero.
    unsafe {
        output.set_len(count * C);
    }
    output
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
pub(crate) fn pillow_luma(r: u8, g: u8, b: u8) -> u8 {
    ((u32::from(r) * 19_595 + u32::from(g) * 38_470 + u32::from(b) * 7_471 + 0x8000) >> 16) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
