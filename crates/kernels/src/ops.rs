//! Shared lookup-table and pixel layout kernels.
use std::simd::Simd;
use std::simd::cmp::SimdPartialEq;
use std::simd::num::SimdUint;

use crate::pixels::{self, Bytes, load, store};

pub fn constant_pixel<const C: usize>(source: &[u8]) -> Option<[u8; C]> {
    let pixels = source.as_chunks::<C>().0;
    let first = *pixels.first()?;
    let (blocks, tail) = pixels.as_chunks::<16>();
    for block in blocks {
        let values = load::<C>(block.as_flattened());
        if (0..C).any(|c| values[c].simd_ne(Bytes::splat(first[c])).any()) {
            return None;
        }
    }
    tail.iter().all(|pixel| *pixel == first).then_some(first)
}

pub fn reverse_gray(source: &[u8], output: &mut [std::mem::MaybeUninit<u8>]) {
    let count = output.len();
    assert_eq!(source.len(), count);
    let (blocks, tail) = output.as_chunks_mut::<64>();
    for (i, block) in blocks.iter_mut().enumerate() {
        *block = Simd::<u8, 64>::from_slice(&source[count - (i + 1) * 64..count - i * 64]).reverse().to_array().map(std::mem::MaybeUninit::new);
    }
    let length = tail.len();
    for (dst, &src) in tail.iter_mut().zip(source[..length].iter().rev()) {
        dst.write(src);
    }
}

pub fn reverse_packed<const C: usize>(source: &[u8], output: &mut [std::mem::MaybeUninit<u8>]) {
    assert!(C == 2 || C == 4);
    assert_eq!(source.len(), output.len());
    assert!(source.len().is_multiple_of(C));
    let count = source.len() / C;
    let done = count / 16 * 16;
    for i in (0..done).step_by(16) {
        let start = (count - i - 16) * C;
        let input = &source[start..start + 16 * C];
        let dst = &mut output[i * C..(i + 16) * C];
        if C == 2 {
            let values = Simd::<u16, 16>::from_array(std::array::from_fn(|j| u16::from_ne_bytes(input[j * 2..j * 2 + 2].try_into().unwrap())));
            dst.write_copy_of_slice(values.reverse().to_array().map(u16::to_ne_bytes).as_flattened());
        } else {
            let values = Simd::<u32, 16>::from_array(std::array::from_fn(|j| u32::from_ne_bytes(input[j * 4..j * 4 + 4].try_into().unwrap())));
            dst.write_copy_of_slice(values.reverse().to_array().map(u32::to_ne_bytes).as_flattened());
        }
    }
    for (dst, src) in output[done * C..].as_chunks_mut::<C>().0.iter_mut().zip(source[..(count - done) * C].as_chunks::<C>().0.iter().rev()) {
        dst.write_copy_of_slice(src);
    }
}

#[inline]
pub fn transpose_rgba_four(rows: [&[u8]; 4]) -> [[u8; 16]; 4] {
    use std::simd::simd_swizzle;

    let [a, b, c, d] = rows.map(Bytes::from_slice);
    let ab0 = simd_swizzle!(a, b, [0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23]);
    let ab1 = simd_swizzle!(a, b, [8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31]);
    let cd0 = simd_swizzle!(c, d, [0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23]);
    let cd1 = simd_swizzle!(c, d, [8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31]);
    [
        simd_swizzle!(ab0, cd0, [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23]),
        simd_swizzle!(ab0, cd0, [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31]),
        simd_swizzle!(ab1, cd1, [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23]),
        simd_swizzle!(ab1, cd1, [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31]),
    ]
    .map(Simd::to_array)
}

#[inline]
pub fn transpose_rgb_four(rows: [&[u8]; 4]) -> [[u8; 12]; 4] {
    use std::simd::simd_swizzle;

    let [a, b, c, d] = rows.map(|row| Bytes::from_array(std::array::from_fn(|i| if i < 12 { row[i] } else { 0 })));
    let ab0 = simd_swizzle!(a, b, [0, 1, 2, 16, 17, 18, 3, 4, 5, 19, 20, 21, 0, 0, 0, 0]);
    let ab1 = simd_swizzle!(a, b, [6, 7, 8, 22, 23, 24, 9, 10, 11, 25, 26, 27, 0, 0, 0, 0]);
    let cd0 = simd_swizzle!(c, d, [0, 1, 2, 16, 17, 18, 3, 4, 5, 19, 20, 21, 0, 0, 0, 0]);
    let cd1 = simd_swizzle!(c, d, [6, 7, 8, 22, 23, 24, 9, 10, 11, 25, 26, 27, 0, 0, 0, 0]);
    [
        simd_swizzle!(ab0, cd0, [0, 1, 2, 3, 4, 5, 16, 17, 18, 19, 20, 21, 0, 0, 0, 0]),
        simd_swizzle!(ab0, cd0, [6, 7, 8, 9, 10, 11, 22, 23, 24, 25, 26, 27, 0, 0, 0, 0]),
        simd_swizzle!(ab1, cd1, [0, 1, 2, 3, 4, 5, 16, 17, 18, 19, 20, 21, 0, 0, 0, 0]),
        simd_swizzle!(ab1, cd1, [6, 7, 8, 9, 10, 11, 22, 23, 24, 25, 26, 27, 0, 0, 0, 0]),
    ]
    .map(|v| v.to_array()[..12].try_into().unwrap())
}

pub fn reverse_rgb(source: &[u8], output: &mut [u8]) {
    assert_eq!(source.len(), output.len());
    assert!(source.len().is_multiple_of(3));
    let pixels = source.len() / 3;
    let done = pixels::reverse_rgb(source, output);
    for (i, dst) in output[done * 3..].as_chunks_mut::<3>().0.iter_mut().enumerate() {
        let start = (pixels - done - i - 1) * 3;
        dst.copy_from_slice(&source[start..start + 3]);
    }
}

pub fn lut<const C: usize>(source: &[u8], output: &mut [u8], tables: &[u8]) {
    assert_eq!(source.len(), output.len());
    assert!(tables.len() == 256 || tables.len() == C << 8);
    let end = source.len() / (16 * C) * (16 * C);
    for i in (0..end).step_by(16 * C) {
        let values = load::<C>(&source[i..]);
        let mapped = std::array::from_fn::<_, C, _>(|c| {
            let base = if tables.len() == 256 { 0 } else { c * 256 };
            Bytes::gather_or_default(&tables[base..base + 256], values[c].cast::<usize>())
        });
        store(&mut output[i..], mapped);
    }
    scalar_lut::<C>(&source[end..], &mut output[end..], tables);
}

fn scalar_lut<const C: usize>(source: &[u8], output: &mut [u8], tables: &[u8]) {
    for (i, (&src, dst)) in source.iter().zip(output).enumerate() {
        let base = if tables.len() == 256 { 0 } else { i % C * 256 };
        *dst = tables[base + usize::from(src)];
    }
}

pub fn colorize(source: &[u8], output: &mut [u8], tables: &[u8]) {
    assert_eq!(Some(output.len()), source.len().checked_mul(3));
    assert_eq!(tables.len(), 768);
    let start = dispatch_colorize(source, output, tables);
    for (&v, dst) in source[start..].iter().zip(output[start * 3..].as_chunks_mut::<3>().0) {
        let index = usize::from(v);
        *dst = [tables[index], tables[256 + index], tables[512 + index]];
    }
}

fn dispatch_colorize(source: &[u8], output: &mut [u8], tables: &[u8]) -> usize {
    let end = source.len() / 16 * 16;
    for i in (0..end).step_by(16) {
        let indices = Bytes::from_slice(&source[i..i + 16]).cast::<usize>();
        store(&mut output[i * 3..], std::array::from_fn::<_, 3, _>(|c| Bytes::gather_or_default(&tables[c * 256..(c + 1) * 256], indices)));
    }
    end
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_reversals_preserve_bytes_tails_and_guards() {
        fn check<const C: usize>() {
            for count in [0, 1, 15, 16, 17, 31, 32, 33, 127] {
                for offset in 0..16 {
                    let source: Vec<u8> = (0..offset + count * C).map(|i| (i * 37) as u8).collect();
                    let mut output = vec![std::mem::MaybeUninit::new(93); offset + count * C + 16];
                    reverse_packed::<C>(&source[offset..], &mut output[offset..offset + count * C]);
                    // SAFETY: guards start initialized and the kernel writes every pixel byte.
                    let output: Vec<_> = output.into_iter().map(|value| unsafe { value.assume_init() }).collect();
                    for (dst, src) in
                        output[offset..offset + count * C].as_chunks::<C>().0.iter().zip(source[offset..].as_chunks::<C>().0.iter().rev())
                    {
                        assert_eq!(dst, src);
                    }
                    assert!(output[..offset].iter().chain(&output[offset + count * C..]).all(|&value| value == 93));
                }
            }
        }
        check::<2>();
        check::<4>();
    }

    #[test]
    fn constant_pixels_check_every_channel_lane_unaligned_tail_and_guard() {
        fn check<const C: usize>() {
            let pixel: [u8; C] = std::array::from_fn(|c| (c * 71 + 3) as u8);
            for count in [0, 1, 15, 16, 17, 31, 32, 33, 65537] {
                let mut source = vec![91];
                source.extend(std::iter::repeat_n(pixel, count).flatten());
                source.push(93);
                assert_eq!(constant_pixel::<C>(&source[1..source.len() - 1]), (count != 0).then_some(pixel));
                for i in (0..count.min(33)).chain((count > 33).then(|| count - 1)) {
                    for c in 0..C {
                        let index = 1 + i * C + c;
                        source[index] ^= 1;
                        assert!(constant_pixel::<C>(&source[1..source.len() - 1]).is_none() || count == 1);
                        source[index] ^= 1;
                    }
                }
                assert_eq!(source[0], 91);
                assert_eq!(source[source.len() - 1], 93);
            }
        }
        check::<1>();
        check::<2>();
        check::<3>();
        check::<4>();
    }

    #[test]
    fn lut_matches_scalar_at_vector_boundaries() {
        check_lut::<1>();
        check_lut::<3>();
        check_lut::<4>();
    }

    fn check_lut<const C: usize>() {
        for n in [0, 1, 7, 8, 9, 15, 16, 17, 47, 48, 49, 255, 256, 257] {
            let input: Vec<u8> = (0..n * C).map(|i| (i * 37) as u8).collect();
            for channels in [1, C] {
                let tables: Vec<u8> = (0..channels * 256).map(|i| (i * 53 + i / 256 * 13) as u8).collect();
                let mut expected = vec![0; input.len()];
                let mut actual = expected.clone();
                scalar_lut::<C>(&input, &mut expected, &tables);
                lut::<C>(&input, &mut actual, &tables);
                assert_eq!(actual, expected);
            }
        }
    }

    #[test]
    fn colorize_dispatch_preserves_tails_and_guards() {
        let tables: Vec<u8> = (0..768).map(|i| (i * 29 + i / 256 * 7) as u8).collect();
        for n in [0, 1, 7, 8, 9, 15, 16, 17, 255, 256, 257] {
            let source: Vec<u8> = (0..n + 1).map(|i| (i * 37) as u8).collect();
            let mut output = vec![123; n * 3 + 2];
            let count = dispatch_colorize(&source[1..], &mut output[1..1 + n * 3], &tables);
            assert_eq!(count, n / 16 * 16);
            for i in 0..count {
                let v = usize::from(source[1 + i]);
                assert_eq!(&output[1 + i * 3..1 + (i + 1) * 3], &[tables[v], tables[256 + v], tables[512 + v]]);
            }
            assert_eq!(output[0], 123);
            assert!(output[1 + count * 3..].iter().all(|&v| v == 123));
        }
    }

    #[test]
    fn colorize_matches_scalar_at_vector_boundaries() {
        let tables: Vec<u8> = (0..768).map(|i| (i * 29 + i / 256 * 7) as u8).collect();
        for n in [0, 1, 7, 8, 9, 15, 16, 17, 255, 256, 257] {
            let source: Vec<u8> = (0..n).map(|i| (i * 37) as u8).collect();
            let mut output = vec![0; n * 3];
            colorize(&source, &mut output, &tables);
            for (pixel, &v) in output.as_chunks::<3>().0.iter().zip(&source) {
                let i = usize::from(v);
                assert_eq!(*pixel, [tables[i], tables[256 + i], tables[512 + i]]);
            }
        }
    }

    #[test]
    fn reverse_rgb_vector_boundaries_and_unaligned_slices() {
        for n in [0, 1, 15, 16, 17, 31, 32, 33, 127] {
            let source: Vec<u8> = (0..n * 3 + 2).map(|i| (i * 37) as u8).collect();
            let mut output = vec![123; n * 3 + 2];
            reverse_rgb(&source[1..1 + n * 3], &mut output[1..1 + n * 3]);
            for i in 0..n {
                assert_eq!(output[1 + i * 3..1 + (i + 1) * 3], source[1 + (n - i - 1) * 3..1 + (n - i) * 3]);
            }
            assert_eq!(output[0], 123);
            assert_eq!(output[n * 3 + 1], 123);
        }
    }
}
