use std::simd::Simd;
use std::simd::num::SimdUint;

use blanket_core::parallel::chunks_mut_above;
use blanket_core::pixels::{load, store};

pub(super) fn swap_red_blue<const C: usize>(source: &[u8], output: &mut [u8]) {
    let end = output.len() / (16 * C) * (16 * C);
    for i in (0..end).step_by(16 * C) {
        let mut channels = load::<C>(&source[i..]);
        channels.swap(0, 2);
        store(&mut output[i..], channels);
    }
    for (src, dst) in source[end..].as_chunks::<C>().0.iter().zip(output[end..].as_chunks_mut::<C>().0) {
        dst.copy_from_slice(src);
        dst.swap(0, 2);
    }
}

pub(super) fn wide_samples(source: &[u8], output: &mut [u8], depth: u8) {
    fn convert<const DEPTH: u16>(source: &[u8], output: &mut [u8]) {
        chunks_mut_above(output, 128 * 1024, 4 * 1024 * 1024, |chunk, dst| {
            let src = &source[chunk * 128 * 1024..][..dst.len()];
            let (src_groups, src_tail) = src.as_chunks::<32>();
            let (dst_groups, dst_tail) = dst.as_chunks_mut::<32>();
            for (src, dst) in src_groups.iter().zip(dst_groups) {
                let samples = Simd::<u16, 16>::from_array(std::array::from_fn(|lane| u16::from_le_bytes([src[lane * 2], src[lane * 2 + 1]])));
                let shift = 16 - DEPTH;
                let numerator = samples * Simd::splat((1 << shift) - 1) + Simd::splat(((1 << DEPTH) - 1) / 2);
                // Exact division by 2^DEPTH - 1 for this reduced numerator.
                let correction = (numerator + Simd::splat(1) + (numerator >> DEPTH)) >> DEPTH;
                let scaled = (samples << shift) + correction;
                let (first, second) = (scaled >> 8).cast::<u8>().interleave(scaled.cast::<u8>());
                first.copy_to_slice(&mut dst[..16]);
                second.copy_to_slice(&mut dst[16..]);
            }
            let maximum = (1 << DEPTH) - 1;
            for (src, dst) in src_tail.as_chunks::<2>().0.iter().zip(dst_tail.as_chunks_mut::<2>().0) {
                let sample = (u32::from(u16::from_le_bytes(*src)) * 65535 + maximum / 2) / maximum;
                *dst = (sample as u16).to_be_bytes();
            }
        });
    }
    match depth {
        10 => convert::<10>(source, output),
        12 => convert::<12>(source, output),
        16 => {
            for (src, dst) in source.as_chunks::<2>().0.iter().zip(output.as_chunks_mut::<2>().0) {
                *dst = [src[1], src[0]];
            }
        }
        _ => unreachable!("wide codecs require 10-, 12-, or 16-bit samples"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn red_blue_swap_handles_unaligned_vectors_tails_and_guards() {
        fn check<const C: usize>() {
            for count in 0..=65 {
                for offset in 1..=3 {
                    let length = count * C;
                    let source: Vec<_> = (0..length + offset + 1).map(|i| (i * 37) as u8).collect();
                    let mut output = vec![0xa5; length + 2];
                    swap_red_blue::<C>(&source[offset..offset + length], &mut output[1..length + 1]);
                    for (src, dst) in source[offset..offset + length]
                        .as_chunks::<C>()
                        .0
                        .iter()
                        .zip(output[1..length + 1].as_chunks::<C>().0)
                    {
                        let mut expected = *src;
                        expected.swap(0, 2);
                        assert_eq!(dst, &expected);
                    }
                    assert_eq!(output[0], 0xa5);
                    assert_eq!(output[length + 1], 0xa5);
                }
            }
        }
        check::<3>();
        check::<4>();
    }

    #[test]
    fn wide_conversion_preserves_rounding_endianness_and_partition_boundaries() {
        for depth in [10, 12, 16] {
            let maximum = (1_u32 << depth) - 1;
            for count in [0, 1, 7, 8, 9, 15, 16, 17, 65535, 65536, 65537, 2_097_153] {
                let samples: Vec<_> = (0..count).map(|i| (i as u32 * 719 % (maximum + 1)) as u16).collect();
                let mut input = vec![0x5a];
                input.extend(samples.iter().flat_map(|v| v.to_le_bytes()));
                let mut output = vec![0xa5; count * 2 + 2];
                wide_samples(&input[1..], &mut output[1..count * 2 + 1], depth);
                for (&sample, bytes) in samples.iter().zip(output[1..count * 2 + 1].as_chunks::<2>().0) {
                    assert_eq!(u32::from(u16::from_be_bytes(*bytes)), (u32::from(sample) * 65535 + maximum / 2) / maximum);
                }
                assert_eq!(output[0], 0xa5);
                assert_eq!(output[count * 2 + 1], 0xa5);
            }
        }
    }
}
