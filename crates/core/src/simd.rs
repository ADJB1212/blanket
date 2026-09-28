use crate::parallel::{CHUNK_PIXELS, chunks_mut};
use crate::raster::PixelMode;

pub fn float_to_integer(source: &[u8]) -> Vec<u8> {
    let mut output = vec![0; source.len()];
    chunks_mut(&mut output, CHUNK_PIXELS * 4, |chunk, dst| {
        let src = &source[chunk * CHUNK_PIXELS * 4..][..dst.len()];
        let mut offset = 0;
        #[cfg(target_arch = "aarch64")]
        unsafe {
            use std::arch::aarch64::*;
            while offset + 16 <= dst.len() {
                let values = vld1q_f32(src.as_ptr().add(offset).cast());
                vst1q_s32(dst.as_mut_ptr().add(offset).cast(), vcvtq_s32_f32(values));
                offset += 16;
            }
        }
        #[cfg(target_arch = "x86_64")]
        unsafe {
            use std::arch::x86_64::*;
            while offset + 16 <= dst.len() {
                let values = _mm_loadu_ps(src.as_ptr().add(offset).cast());
                let converted = _mm_cvttps_epi32(values);
                let high = _mm_castps_si128(_mm_cmpge_ps(values, _mm_set1_ps(2147483648.0)));
                let valid = _mm_castps_si128(_mm_cmpord_ps(values, values));
                let result = _mm_and_si128(
                    valid,
                    _mm_or_si128(_mm_and_si128(high, _mm_set1_epi32(i32::MAX)), _mm_andnot_si128(high, converted)),
                );
                _mm_storeu_si128(dst.as_mut_ptr().add(offset).cast(), result);
                offset += 16;
            }
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
    chunks_mut(&mut output, CHUNK_PIXELS, |chunk, dst| {
        let src = &source[chunk * CHUNK_PIXELS * stride..(chunk * CHUNK_PIXELS + dst.len()) * stride];
        #[cfg(target_arch = "aarch64")]
        let offset = {
            use std::arch::aarch64::*;
            let mut offset = 0;
            unsafe {
                while offset + 16 <= dst.len() {
                    let ptr = src.as_ptr().add(offset * stride);
                    let values = if mode == PixelMode::I {
                        let a = vqmovun_s32(vld1q_s32(ptr.cast()));
                        let b = vqmovun_s32(vld1q_s32(ptr.add(16).cast()));
                        let c = vqmovun_s32(vld1q_s32(ptr.add(32).cast()));
                        let d = vqmovun_s32(vld1q_s32(ptr.add(48).cast()));
                        vcombine_u8(vqmovn_u16(vcombine_u16(a, b)), vqmovn_u16(vcombine_u16(c, d)))
                    } else {
                        let a = vld1q_u8(ptr);
                        let b = vld1q_u8(ptr.add(16));
                        let a = if mode == PixelMode::I16B { vrev16q_u8(a) } else { a };
                        let b = if mode == PixelMode::I16B { vrev16q_u8(b) } else { b };
                        vcombine_u8(vqmovn_u16(vreinterpretq_u16_u8(a)), vqmovn_u16(vreinterpretq_u16_u8(b)))
                    };
                    vst1q_u8(dst.as_mut_ptr().add(offset), values);
                    offset += 16;
                }
            }
            offset
        };
        #[cfg(not(target_arch = "aarch64"))]
        let offset = 0;
        for (pixel, bytes) in dst[offset..].iter_mut().zip(src[offset * stride..].chunks_exact(stride)) {
            *pixel = match mode {
                PixelMode::I => i32::from_le_bytes(bytes.try_into().unwrap()).clamp(0, 255) as u8,
                PixelMode::I16B => u16::from_be_bytes(bytes.try_into().unwrap()).min(255) as u8,
                _ => u16::from_le_bytes(bytes.try_into().unwrap()).min(255) as u8,
            };
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
        #[cfg(target_arch = "aarch64")]
        let offset = if to == PixelMode::I && source_bytes == 2 {
            use std::arch::aarch64::*;
            let mut offset = 0;
            unsafe {
                while offset + 8 <= dst.len() / 4 {
                    let input = vld1q_u8(src.as_ptr().add(offset * 2));
                    let input = if from == PixelMode::I16B { vrev16q_u8(input) } else { input };
                    let input = vreinterpretq_u16_u8(input);
                    vst1q_u32(dst.as_mut_ptr().add(offset * 4).cast(), vmovl_u16(vget_low_u16(input)));
                    vst1q_u32(dst.as_mut_ptr().add((offset + 4) * 4).cast(), vmovl_u16(vget_high_u16(input)));
                    offset += 8;
                }
            }
            offset
        } else {
            0
        };
        #[cfg(not(target_arch = "aarch64"))]
        let offset = 0;
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

pub fn gray_to_hsv<const C: usize>(source: &[u8]) -> Vec<u8> {
    let length = source.len() / C * 3;
    let mut output = Vec::<u8>::with_capacity(length);
    let spare = &mut output.spare_capacity_mut()[..length];
    let chunk_pixels = 256 * 1024;
    crate::parallel::chunks_mut_above(spare, chunk_pixels * 3, 5 * 1024 * 1024, |chunk, dst| {
        let start = chunk * chunk_pixels * C;
        let src = &source[start..start + dst.len() / 3 * C];
        #[cfg(target_arch = "aarch64")]
        let offset = {
            use std::arch::aarch64::*;
            let mut offset = 0;
            unsafe {
                let zero = vdupq_n_u8(0);
                while offset + 16 <= src.len() / C {
                    let ptr = src.as_ptr().add(offset * C);
                    let value = if C == 1 { vld1q_u8(ptr) } else { vld2q_u8(ptr).0 };
                    vst3q_u8(dst.as_mut_ptr().cast::<u8>().add(offset * 3), uint8x16x3_t(zero, zero, value));
                    offset += 16;
                }
            }
            offset
        };
        #[cfg(not(target_arch = "aarch64"))]
        let offset = 0;
        for (src, dst) in src[offset * C..].as_chunks::<C>().0.iter().zip(dst[offset * 3..].as_chunks_mut::<3>().0) {
            dst[0].write(0);
            dst[1].write(0);
            dst[2].write(src[0]);
        }
    });
    // Every output byte is initialized by the vector loop or scalar tail.
    unsafe { output.set_len(length) };
    output
}

/// Map `S`-byte pixels to `D` bytes in uninitialized output. `block` converts
/// 16 pixels at a time where supported; `pixel` handles the remainder.
fn map_blocks<const S: usize, const D: usize>(
    source: &[u8], parallel_bytes: usize, block: impl Fn(*const u8, *mut u8) + Sync, pixel: impl Fn(&[u8; S]) -> [u8; D] + Sync,
) -> Vec<u8> {
    let count = source.len() / S;
    let mut output = Vec::<u8>::with_capacity(count * D);
    let chunk_pixels = 64 * 1024;
    crate::parallel::chunks_mut_above(
        &mut output.spare_capacity_mut()[..count * D],
        chunk_pixels * D,
        parallel_bytes,
        |i, dst| {
            let src = &source[i * chunk_pixels * S..(i * chunk_pixels + dst.len() / D) * S];
            let blocks = if cfg!(target_arch = "aarch64") { src.len() / S / 16 } else { 0 };
            for b in 0..blocks {
                block(src[b * 16 * S..].as_ptr(), dst[b * 16 * D..].as_mut_ptr().cast::<u8>());
            }
            for (src, dst) in src[blocks * 16 * S..]
                .as_chunks::<S>()
                .0
                .iter()
                .zip(dst[blocks * 16 * D..].as_chunks_mut::<D>().0)
            {
                for (value, converted) in dst.iter_mut().zip(pixel(src)) {
                    value.write(converted);
                }
            }
        },
    );
    // Blocks and the scalar tail initialize every byte of each disjoint chunk.
    unsafe { output.set_len(count * D) };
    output
}

/// Pillow's L to CMYK stores inverted gray as K.
pub fn gray_to_cmyk(source: &[u8]) -> Vec<u8> {
    map_blocks::<1, 4>(
        source,
        4 * 1024 * 1024,
        |src, dst| {
            #[cfg(target_arch = "aarch64")]
            unsafe {
                use std::arch::aarch64::*;
                let zero = vdupq_n_u8(0);
                vst4q_u8(dst, uint8x16x4_t(zero, zero, zero, vmvnq_u8(vld1q_u8(src))));
            }
            #[cfg(not(target_arch = "aarch64"))]
            let _ = (src, dst);
        },
        |p| [0, 0, 0, 255 - p[0]],
    )
}

pub fn gray_to_ycbcr(source: &[u8]) -> Vec<u8> {
    map_blocks::<1, 3>(
        source,
        4 * 1024 * 1024,
        |src, dst| {
            #[cfg(target_arch = "aarch64")]
            unsafe {
                use std::arch::aarch64::*;
                let half = vdupq_n_u8(128);
                vst3q_u8(dst, uint8x16x3_t(vld1q_u8(src), half, half));
            }
            #[cfg(not(target_arch = "aarch64"))]
            let _ = (src, dst);
        },
        |p| [p[0], 128, 128],
    )
}

/// RGB or RGBA to CMYK by inverting color channels, with zero K.
pub fn color_to_cmyk<const S: usize>(source: &[u8]) -> Vec<u8> {
    map_blocks::<S, 4>(
        source,
        4 * 1024 * 1024,
        |src, dst| {
            #[cfg(target_arch = "aarch64")]
            unsafe {
                use std::arch::aarch64::*;
                let (r, g, b) = if S == 3 {
                    let p = vld3q_u8(src);
                    (p.0, p.1, p.2)
                } else {
                    let p = vld4q_u8(src);
                    (p.0, p.1, p.2)
                };
                vst4q_u8(dst, uint8x16x4_t(vmvnq_u8(r), vmvnq_u8(g), vmvnq_u8(b), vdupq_n_u8(0)));
            }
            #[cfg(not(target_arch = "aarch64"))]
            let _ = (src, dst);
        },
        |p| [255 - p[0], 255 - p[1], 255 - p[2], 0],
    )
}

pub fn extract_channel<const S: usize>(source: &[u8], channel: usize) -> Vec<u8> {
    map_blocks::<S, 1>(
        source,
        1024 * 1024,
        |src, dst| {
            #[cfg(target_arch = "aarch64")]
            unsafe {
                use std::arch::aarch64::*;
                let value = if S == 3 {
                    let p = vld3q_u8(src);
                    match channel {
                        0 => p.0,
                        1 => p.1,
                        _ => p.2,
                    }
                } else {
                    let p = vld4q_u8(src);
                    match channel {
                        0 => p.0,
                        1 => p.1,
                        2 => p.2,
                        _ => p.3,
                    }
                };
                vst1q_u8(dst, value);
            }
            #[cfg(not(target_arch = "aarch64"))]
            let _ = (src, dst);
        },
        |p| [p[channel]],
    )
}

pub fn convert_la(source: &[u8], to: PixelMode) -> Vec<u8> {
    let channels = to.channels();
    let mut output = vec![0; source.len() / 2 * channels];
    chunks_mut(&mut output, CHUNK_PIXELS * channels, |chunk, dst| {
        let start = chunk * CHUNK_PIXELS * 2;
        let src = &source[start..start + dst.len() / channels * 2];
        #[cfg(target_arch = "aarch64")]
        let offset = {
            use std::arch::aarch64::*;
            let mut offset = 0;
            unsafe {
                while offset + 16 <= src.len() / 2 {
                    let pair = vld2q_u8(src.as_ptr().add(offset * 2));
                    let out = dst.as_mut_ptr().add(offset * channels);
                    match to {
                        PixelMode::L => vst1q_u8(out, pair.0),
                        PixelMode::Rgb => vst3q_u8(out, uint8x16x3_t(pair.0, pair.0, pair.0)),
                        PixelMode::Rgba => vst4q_u8(out, uint8x16x4_t(pair.0, pair.0, pair.0, pair.1)),
                        _ => unreachable!(),
                    }
                    offset += 16;
                }
            }
            offset
        };
        #[cfg(not(target_arch = "aarch64"))]
        let offset = 0;
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
    let mut output = vec![0; source.len() / 2];
    chunks_mut(&mut output, CHUNK_PIXELS, |chunk, dst| {
        let src = &source[chunk * CHUNK_PIXELS * 2..(chunk * CHUNK_PIXELS + dst.len()) * 2];
        #[cfg(target_arch = "aarch64")]
        let offset = {
            use std::arch::aarch64::*;
            let mut offset = 0;
            unsafe {
                while offset + 16 <= dst.len() {
                    let pair = vld2q_u8(src.as_ptr().add(offset * 2));
                    vst1q_u8(dst.as_mut_ptr().add(offset), if channel == 0 { pair.0 } else { pair.1 });
                    offset += 16;
                }
            }
            offset
        };
        #[cfg(not(target_arch = "aarch64"))]
        let offset = 0;
        for (value, pair) in dst[offset..].iter_mut().zip(src[offset * 2..].as_chunks::<2>().0) {
            *value = pair[channel];
        }
    });
    output
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
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64")))]
        let offset = 0;
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        let offset = if S == 1 && C == 3 && std::arch::is_x86_feature_detected!("ssse3") {
            // SAFETY: SSSE3 detected; spare capacity holds three bytes per input.
            unsafe { crate::x86_pixels::gray_to_rgb(src, dst) }
        } else {
            0
        };
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
    if crate::parallel::should_parallel(spare.len(), chunk_pixels * C, parallel_bytes) {
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
