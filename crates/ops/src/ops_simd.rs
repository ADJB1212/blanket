//! Portable SIMD kernels with scalar tails.
use blanket_core::pixels::{self, Bytes, load, store};
use std::simd::{
    Simd,
    cmp::SimdOrd,
    num::{SimdFloat, SimdInt, SimdUint},
};

pub(crate) fn reduce_float_two(upper: &[u8], lower: &[u8], output: &mut [u8]) -> usize {
    let end = (upper.len().min(lower.len()) / 16 * 2).min(output.len() / 8 * 2);
    for x in (0..end).step_by(2) {
        let mut sum = Simd::<f64, 2>::splat(0.0);
        for row in [upper, lower] {
            for column in 0..2 {
                let values = Simd::<f32, 2>::from_array(std::array::from_fn(|i| {
                    let start = (x + i) * 8 + column * 4;
                    f32::from_le_bytes(row[start..start + 4].try_into().unwrap())
                }));
                sum += values.cast::<f64>();
            }
        }
        let values = (sum * Simd::splat(0.25)).cast::<f32>().to_array();
        for (dst, value) in output[x * 4..(x + 2) * 4].as_chunks_mut::<4>().0.iter_mut().zip(values) {
            *dst = value.to_le_bytes();
        }
    }
    end
}

/// Reusable grayscale gather map for an axis-aligned nearest transform.
pub(crate) struct NearestL {
    columns: Vec<usize>,
    width: usize,
}

impl NearestL {
    pub(crate) fn new(columns: &[Option<usize>], width: usize) -> Option<Self> {
        let columns: Vec<usize> = columns.iter().copied().collect::<Option<_>>()?;
        if columns.iter().any(|&x| x >= width) {
            return None;
        }
        Some(Self { columns, width })
    }

    pub(crate) fn is_vectorized(&self) -> bool {
        self.columns.len() >= 16
    }

    pub(crate) fn sample(&self, source: &[u8], output: &mut [u8]) {
        assert_eq!(source.len(), self.width);
        assert_eq!(output.len(), self.columns.len());
        let done = output.len() / 16 * 16;
        for i in (0..done).step_by(16) {
            Bytes::gather_or_default(source, Simd::from_slice(&self.columns[i..i + 16])).copy_to_slice(&mut output[i..i + 16]);
        }
        for (dst, &x) in output[done..].iter_mut().zip(&self.columns[done..]) {
            *dst = source[x];
        }
    }
}

pub(crate) fn reverse_rgb(source: &[u8], output: &mut [u8]) {
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
    assert!(tables.len() == 256 || tables.len() == C * 256);
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

pub(crate) fn colorize(source: &[u8], output: &mut [u8], tables: &[u8]) {
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
        store(
            &mut output[i * 3..],
            std::array::from_fn::<_, 3, _>(|c| Bytes::gather_or_default(&tables[c * 256..(c + 1) * 256], indices)),
        );
    }
    end
}

/// Returns the byte count processed without overflowing fixed-point accumulators.
pub(crate) fn vertical(source: &[u8], output: &mut [u8], stride: usize, weights: &[i32]) -> usize {
    let magnitude: i64 = weights.iter().map(|&w| i64::from(w).abs()).sum();
    if magnitude * 255 + (1 << 21) > i64::from(i32::MAX) {
        return 0;
    }
    assert!(weights.is_empty() || source.len() >= (weights.len() - 1).saturating_mul(stride).saturating_add(output.len()));
    let end = output.len() / 8 * 8;
    for x in (0..end).step_by(8) {
        let mut acc = Simd::<i32, 8>::splat(1 << 21);
        for (i, &weight) in weights.iter().enumerate() {
            let values = Simd::<u8, 8>::from_slice(&source[i * stride + x..i * stride + x + 8]);
            acc += values.cast::<i32>() * Simd::splat(weight);
        }
        (acc >> 22)
            .simd_clamp(Simd::splat(0), Simd::splat(255))
            .cast::<u8>()
            .copy_to_slice(&mut output[x..x + 8]);
    }
    end
}

pub(crate) fn nearest_half<const C: usize>(source: &[u8], output: &mut [u8]) {
    assert_eq!(source.len(), output.len() * 2);
    let done = output.len() / C / 16 * 16 * C;
    for i in (0..done).step_by(16 * C) {
        let first = load::<C>(&source[i * 2..]);
        let second = load::<C>(&source[i * 2 + 16 * C..]);
        let values = std::array::from_fn::<_, C, _>(|c| first[c].deinterleave(second[c]).1);
        store(&mut output[i..], values);
    }
    for (dst, pair) in output[done..]
        .as_chunks_mut::<C>()
        .0
        .iter_mut()
        .zip(source[done * 2..].as_chunks::<C>().0.as_chunks::<2>().0)
    {
        *dst = pair[1];
    }
}

pub(crate) fn reduce_three_l(upper: &[u8], middle: &[u8], lower: &[u8], output: &mut [u8]) -> usize {
    assert_eq!(upper.len(), output.len() * 3);
    assert_eq!(middle.len(), upper.len());
    assert_eq!(lower.len(), upper.len());
    pixels::reduce_three_l([upper, middle, lower], output)
}

pub(crate) fn reduce_two_l(upper: &[u8], lower: &[u8], output: &mut [u8]) -> usize {
    assert_eq!(upper.len(), output.len() * 2);
    assert_eq!(lower.len(), upper.len());
    let end = output.len() / 8 * 8;
    for i in (0..end).step_by(8) {
        let mut sum = Simd::<u16, 8>::splat(2);
        for row in [upper, lower] {
            let values = Simd::<u8, 16>::from_slice(&row[i * 2..i * 2 + 16]);
            let even = std::simd::simd_swizzle!(values, [0, 2, 4, 6, 8, 10, 12, 14]);
            let odd = std::simd::simd_swizzle!(values, [1, 3, 5, 7, 9, 11, 13, 15]);
            sum += even.cast::<u16>() + odd.cast::<u16>();
        }
        (sum >> 2).cast::<u8>().copy_to_slice(&mut output[i..i + 8]);
    }
    end
}

#[cfg(test)]
mod tests {
    #[test]
    fn float_reduction_preserves_odd_edges_and_accumulation_order() {
        for width in 0_usize..=33 {
            let samples = [
                0.0_f32,
                -0.0,
                f32::MAX,
                -f32::MAX,
                1.25,
                -2.75,
                f32::MIN_POSITIVE,
                f32::INFINITY,
                f32::NAN,
            ];
            let rows: [Vec<u8>; 2] = std::array::from_fn(|r| {
                std::iter::once(93)
                    .chain((0..width).flat_map(|i| samples[(i + r * 3) % samples.len()].to_le_bytes()))
                    .collect()
            });
            let mut output = vec![93; width.div_ceil(2) * 4 + 2];
            let len = output.len();
            let done = super::reduce_float_two(&rows[0][1..], &rows[1][1..], &mut output[1..len - 1]);
            assert_eq!(done, width / 4 * 2);
            for x in 0..done {
                let mut sum = 0.0_f64;
                for row in &rows {
                    for c in 0..2 {
                        let start = 1 + (x * 2 + c) * 4;
                        sum += f64::from(f32::from_le_bytes(row[start..start + 4].try_into().unwrap()));
                    }
                }
                let expected = (sum * 0.25) as f32;
                let actual = f32::from_le_bytes(output[1 + x * 4..5 + x * 4].try_into().unwrap());
                assert!(actual.to_bits() == expected.to_bits() || (actual.is_nan() && expected.is_nan()));
            }
            assert_eq!(output[0], 93);
            assert!(output[1 + done * 4..].iter().all(|&v| v == 93));
        }
    }

    #[test]
    fn reduce_two_l_rounding_tails_and_guards() {
        for count in [0, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33] {
            for value in 0..=255 {
                let upper = vec![value; count * 2 + 1];
                let lower: Vec<u8> = (0..count * 2 + 1).map(|i| (i * 37 + value as usize) as u8).collect();
                let mut output = vec![73; count + 2];
                let done = super::reduce_two_l(&upper[1..], &lower[1..], &mut output[1..count + 1]);
                for i in 0..done {
                    let sum: u32 = [&upper, &lower]
                        .iter()
                        .flat_map(|r| &r[1 + i * 2..1 + (i + 1) * 2])
                        .map(|&v| u32::from(v))
                        .sum();
                    assert_eq!(output[i + 1], ((sum + 2) / 4) as u8);
                }
                assert_eq!(output[0], 73);
                assert!(output[done + 1..].iter().all(|&v| v == 73));
            }
        }
    }

    #[test]
    fn reduce_three_l_rounding_and_guards() {
        for count in [0, 1, 15, 16, 17, 31, 32, 33] {
            for value in 0..=255 {
                let upper = vec![value; count * 3 + 1];
                let middle: Vec<_> = (0..count * 3 + 1).map(|i| if value % 2 == 0 { (i * 37) as u8 } else { value }).collect();
                let lower = vec![value; count * 3 + 1];
                let mut output = vec![73; count + 2];
                let done = super::reduce_three_l(&upper[1..], &middle[1..], &lower[1..], &mut output[1..count + 1]);
                assert_eq!(done, count / 16 * 16);
                for i in 0..done {
                    let sum: u32 = [&upper, &middle, &lower]
                        .iter()
                        .flat_map(|r| &r[1 + i * 3..1 + (i + 1) * 3])
                        .map(|&v| u32::from(v))
                        .sum();
                    assert_eq!(output[i + 1], (((sum + 4) * 1_864_135) >> 24) as u8);
                }
                assert_eq!(output[0], 73);
                assert!(output[done + 1..].iter().all(|&v| v == 73));
            }
        }
    }

    #[test]
    fn nearest_half_vector_tails_and_guards() {
        fn check<const C: usize>() {
            for count in [0, 1, 15, 16, 17, 31, 32, 33, 400] {
                let source: Vec<u8> = (0..count * C * 2 + 1).map(|i| (i * 37 + i / 11) as u8).collect();
                let mut output = vec![199; count * C + 2];
                super::nearest_half::<C>(&source[1..], &mut output[1..count * C + 1]);
                let expected: Vec<_> = (0..count)
                    .flat_map(|i| (0..C).map(move |c| 1 + (i * 2 + 1) * C + c))
                    .map(|i| source[i])
                    .collect();
                assert_eq!(&output[1..count * C + 1], expected);
                assert_eq!(output[0], 199);
                assert_eq!(output[count * C + 1], 199);
            }
        }
        check::<1>();
        check::<2>();
        check::<3>();
        check::<4>();
    }

    use super::*;

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
    fn vertical_matches_signed_fixed_point() {
        let weights = [-400_000, 2_000_000, 3_000_000, -405_696];
        let source: Vec<u8> = (0..4 * 31).map(|i| (i * 67) as u8).collect();
        let mut actual = vec![0; 31];
        let count = vertical(&source, &mut actual, 31, &weights);
        assert_eq!(count, 24);
        for x in 0..count {
            let sum = weights
                .iter()
                .enumerate()
                .fold(1_i64 << 21, |sum, (i, &w)| sum + i64::from(source[i * 31 + x]) * i64::from(w));
            assert_eq!(actual[x], (sum >> 22).clamp(0, 255) as u8);
        }
    }

    #[test]
    fn vertical_preserves_tails_and_guards() {
        for width in [0, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33] {
            for weights in [
                &[][..],
                &[1 << 22][..],
                &[-(1 << 22)][..],
                &[2 << 22][..],
                &[-400_000, 2_000_000, 3_000_000, -405_696][..],
            ] {
                let stride = width + 3;
                // Offset both buffers by one to exercise unaligned loads/stores.
                let source: Vec<u8> = (0..1 + weights.len() * stride).map(|i| (i * 67) as u8).collect();
                let mut actual = vec![123; width + 2];
                let count = vertical(&source[1..], &mut actual[1..1 + width], stride, weights);
                assert_eq!(count, width / 8 * 8);
                for x in 0..count {
                    let sum = weights
                        .iter()
                        .enumerate()
                        .fold(1_i64 << 21, |sum, (i, &w)| sum + i64::from(source[1 + i * stride + x]) * i64::from(w));
                    assert_eq!(actual[1 + x], (sum >> 22).clamp(0, 255) as u8);
                }
                assert_eq!(actual[0], 123);
                assert!(actual[1 + count..].iter().all(|&v| v == 123));
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
    fn vertical_leaves_output_for_scalar_when_weights_can_overflow() {
        for weight in [i32::MIN, i32::MAX] {
            let mut output = [123; 16];
            assert_eq!(vertical(&[255; 16], &mut output, 16, &[weight]), 0);
            assert_eq!(output, [123; 16]);
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

    #[test]
    fn nearest_l_gathers_and_scalar_tails() {
        let source: Vec<u8> = (0..258).map(|i| (i * 37) as u8).collect();
        for n in [0, 1, 15, 16, 17, 31, 32, 33, 127] {
            for step in [0, 1, 2, 5] {
                for reverse in [false, true] {
                    let columns: Vec<_> = (0..n)
                        .map(|i| {
                            let x = i * step % 256;
                            Some(if reverse { 255 - x } else { x })
                        })
                        .collect();
                    let map = NearestL::new(&columns, 256).unwrap();
                    let mut output = vec![123; n + 2];
                    map.sample(&source[1..257], &mut output[1..n + 1]);
                    for (i, column) in columns.iter().enumerate() {
                        assert_eq!(output[i + 1], source[1 + column.unwrap()]);
                    }
                    assert_eq!(output[0], 123);
                    assert_eq!(output[n + 1], 123);
                }
            }
        }
        assert!(NearestL::new(&[None], 256).is_none());
        assert!(NearestL::new(&[Some(256)], 256).is_none());
    }

    #[test]
    fn nearest_l_detects_vector_support() {
        let columns: Vec<_> = (0..16).map(Some).collect();
        let map = NearestL::new(&columns, 32).unwrap();
        assert!(map.is_vectorized());
    }
}
