use std::simd::Simd;
use std::simd::cmp::SimdPartialOrd;

use blanket_core::pixels::{load, store};

pub(super) fn pcx_literals(source: &[u8]) -> usize {
    let mut offset = 0;
    for chunk in source.as_chunks::<32>().0 {
        let runs = Simd::from_array(*chunk).simd_ge(Simd::splat(0xc0));
        if let Some(index) = runs.first_set() {
            return offset + index;
        }
        offset += 32;
    }
    offset + source[offset..].iter().position(|&v| v >= 0xc0).unwrap_or(source.len() - offset)
}

pub(super) fn extract_channel<const C: usize>(source: &[u8], output: &mut [u8], channel: usize) {
    let end = output.len() / 16 * 16;
    for i in (0..end).step_by(16) {
        load::<C>(&source[i * C..])[channel].copy_to_slice(&mut output[i..i + 16]);
    }
    for (src, dst) in source[end * C..].as_chunks::<C>().0.iter().zip(&mut output[end..]) {
        *dst = src[channel];
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pcx_literal_spans_handle_unaligned_vectors_and_tails() {
        for length in 0..=97 {
            for offset in 0..=3 {
                let mut source = vec![0x7f; length + offset];
                assert_eq!(pcx_literals(&source[offset..]), length);
                for position in 0..length {
                    for marker in [0xc0, 0xc1, 0xff] {
                        source[offset + position] = marker;
                        assert_eq!(pcx_literals(&source[offset..]), position);
                    }
                    source[offset + position] = 0xbf;
                }
            }
        }
    }

    #[test]
    fn channel_extraction_preserves_unaligned_tails_and_guards() {
        fn check<const C: usize>() {
            for count in 0..=65 {
                for offset in 1..=3 {
                    let source: Vec<_> = (0..count * C + offset + 1).map(|i| (i * 37) as u8).collect();
                    for channel in 0..C {
                        let mut output = vec![0xa5; count + 2];
                        extract_channel::<C>(&source[offset..offset + count * C], &mut output[1..count + 1], channel);
                        for i in 0..count {
                            assert_eq!(output[i + 1], source[offset + i * C + channel]);
                        }
                        assert_eq!(output[0], 0xa5);
                        assert_eq!(output[count + 1], 0xa5);
                    }
                }
            }
        }
        check::<3>();
        check::<4>();
    }

    #[test]
    fn red_blue_swap_handles_unaligned_vectors_tails_and_guards() {
        fn check<const C: usize>() {
            for count in 0..=65 {
                for offset in 1..=3 {
                    let length = count * C;
                    let source: Vec<_> = (0..length + offset + 1).map(|i| (i * 37) as u8).collect();
                    let mut output = vec![0xa5; length + 2];
                    swap_red_blue::<C>(&source[offset..offset + length], &mut output[1..length + 1]);
                    for (src, dst) in source[offset..offset + length].as_chunks::<C>().0.iter().zip(output[1..length + 1].as_chunks::<C>().0) {
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
}
