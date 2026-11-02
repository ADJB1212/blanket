use std::simd::cmp::{SimdOrd, SimdPartialEq};
use std::simd::num::SimdUint;
use std::simd::{Select, Simd};

use crate::pixels::{load, store};

pub(crate) fn convert<const C: usize>(source: &[u8], output: &mut [u8], premultiply: bool) {
    let end = source.len() / (16 * C) * (16 * C);
    for offset in (0..end).step_by(16 * C) {
        let mut values = load::<C>(&source[offset..]);
        let alpha = values[C - 1].cast::<u16>();
        for value in &mut values[..C - 1] {
            let sample = value.cast::<u16>();
            *value = if premultiply {
                let product = sample * alpha + Simd::splat(128);
                ((product + (product >> 8)) >> 8).cast()
            } else {
                let restored = (sample * Simd::splat(255) / alpha.simd_max(Simd::splat(1))).simd_min(Simd::splat(255));
                alpha.simd_eq(Simd::splat(0)).select(sample, restored).cast()
            };
        }
        store::<C>(&mut output[offset..], values);
    }
    for (source, output) in source[end..].as_chunks::<C>().0.iter().zip(output[end..].as_chunks_mut::<C>().0) {
        let alpha = u16::from(source[C - 1]);
        for c in 0..C - 1 {
            let sample = u16::from(source[c]);
            output[c] = if premultiply {
                let product = sample * alpha + 128;
                ((product + (product >> 8)) >> 8) as u8
            } else {
                (sample * 255).checked_div(alpha).unwrap_or(sample).min(255) as u8
            };
        }
        output[C - 1] = alpha as u8;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check<const C: usize>() {
        for premultiply in [false, true] {
            for offset in 0..4 {
                for count in [0, 1, 15, 16, 17, 31, 32, 33, 65536] {
                    let mut source = vec![93; offset + count * C + 1];
                    for (i, pixel) in source[offset..offset + count * C].as_chunks_mut::<C>().0.iter_mut().enumerate() {
                        pixel.fill((i % 256) as u8);
                        pixel[C - 1] = (i / 256) as u8;
                    }
                    let mut output = vec![77; source.len()];
                    convert::<C>(&source[offset..offset + count * C], &mut output[offset..offset + count * C], premultiply);
                    assert!(output[..offset].iter().all(|&v| v == 77));
                    assert_eq!(output[offset + count * C], 77);
                    for (src, dst) in source[offset..offset + count * C].as_chunks::<C>().0.iter().zip(output[offset..].as_chunks::<C>().0) {
                        let alpha = u32::from(src[C - 1]);
                        for c in 0..C - 1 {
                            let sample = u32::from(src[c]);
                            let expected =
                                if premultiply { (sample * alpha + 127) / 255 } else { (sample * 255).checked_div(alpha).unwrap_or(sample).min(255) };
                            assert_eq!(u32::from(dst[c]), expected);
                        }
                        assert_eq!(dst[C - 1], src[C - 1]);
                    }
                    assert_eq!(source[0], if offset == 0 && count > 0 { 0 } else { 93 });
                    assert_eq!(source[offset + count * C], 93);
                }
            }
        }
    }

    #[test]
    fn alpha_arithmetic_matches_scalar_with_unaligned_buffers_and_guards() {
        check::<2>();
        check::<4>();
    }
}
