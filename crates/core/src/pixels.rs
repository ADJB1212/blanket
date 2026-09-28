//! Portable kernels for packed byte pixels.
use std::simd::{Select, Simd, cmp::SimdPartialEq, num::SimdUint};

pub type Bytes = Simd<u8, 16>;

#[inline]
pub fn load<const C: usize>(source: &[u8]) -> [Bytes; C] {
    assert!(source.len() >= 16 * C);
    std::array::from_fn(|c| Bytes::gather_or_default(source, Simd::from_array(std::array::from_fn(|i| i * C + c))))
}

#[inline]
pub fn store<const C: usize>(output: &mut [u8], values: [Bytes; C]) {
    assert!(output.len() >= 16 * C);
    for (c, value) in values.into_iter().enumerate() {
        value.scatter(output, Simd::from_array(std::array::from_fn(|i| i * C + c)));
    }
}

pub fn reverse_rgb(src: &[u8], dst: &mut [u8]) -> usize {
    let end = dst.len() / 3 / 16 * 16;
    for i in (0..end).step_by(16) {
        let values = load::<3>(&src[src.len() - (i + 16) * 3..]);
        store(&mut dst[i * 3..], values.map(Bytes::reverse));
    }
    end
}

pub fn nearest_half_rgb(src: &[u8], dst: &mut [u8]) -> usize {
    let end = dst.len() / 48 * 48;
    for i in (0..end).step_by(48) {
        let pairs = load::<6>(&src[i * 2..]);
        store(&mut dst[i..], [pairs[3], pairs[4], pairs[5]]);
    }
    end
}

pub fn reduce_three_l(rows: [&[u8]; 3], dst: &mut [u8]) -> usize {
    let end = dst.len() / 16 * 16;
    for i in (0..end).step_by(16) {
        let mut sum = Simd::<u32, 16>::splat(4);
        for row in rows {
            for v in load::<3>(&row[i * 3..]) {
                sum += v.cast();
            }
        }
        ((sum * Simd::splat(1_864_135)) >> 24).cast::<u8>().copy_to_slice(&mut dst[i..i + 16]);
    }
    end
}

pub fn gray_to_rgb(src: &[u8], dst: &mut [std::mem::MaybeUninit<u8>]) -> usize {
    let end = src.len() / 16 * 16;
    for i in (0..end).step_by(16) {
        let mut block = [0; 48];
        store(&mut block, [Bytes::from_slice(&src[i..i + 16]); 3]);
        for (dst, v) in dst[i * 3..(i + 16) * 3].iter_mut().zip(block) {
            dst.write(v);
        }
    }
    end
}

pub fn merge<const C: usize>(bands: [&[u8]; C], dst: &mut [[u8; C]]) -> usize {
    let end = dst.len() / 16 * 16;
    for i in (0..end).step_by(16) {
        store(dst[i..i + 16].as_flattened_mut(), bands.map(|b| Bytes::from_slice(&b[i..i + 16])));
    }
    end
}

pub fn putalpha_rgb(src: &[u8], alpha: &[u8], dst: &mut [u8]) -> usize {
    let end = alpha.len() / 16 * 16;
    for i in (0..end).step_by(16) {
        let [r, g, b] = load::<3>(&src[i * 3..]);
        store(&mut dst[i * 4..], [r, g, b, Bytes::from_slice(&alpha[i..i + 16])]);
    }
    end
}

pub fn putalpha_rgba(dst: &mut [u8], alpha: &[u8]) -> usize {
    let end = alpha.len() / 16 * 16;
    for i in (0..end).step_by(16) {
        let [r, g, b, _] = load::<4>(&dst[i * 4..]);
        store(&mut dst[i * 4..], [r, g, b, Bytes::from_slice(&alpha[i..i + 16])]);
    }
    end
}

fn blend(d: Bytes, s: Bytes, a: Bytes) -> Bytes {
    let a = a.cast::<u16>();
    let sum = d.cast::<u16>() * (Simd::splat(255) - a) + s.cast::<u16>() * a + Simd::splat(128);
    ((sum + (sum >> 8)) >> 8).cast()
}

pub fn paste_masked<const C: usize, const M: usize>(dst: &mut [u8], src: &[u8], mask: &[u8], fill: bool) -> usize {
    let end = dst.len() / C / 16 * 16;
    for i in (0..end).step_by(16) {
        let d = load::<C>(&dst[i * C..]);
        let s = load::<C>(&src[i * C..]);
        let a = load::<M>(&mask[i * M..])[M - 1];
        let rgb_a = if fill && C == 4 && M == 1 {
            (d[C - 1].simd_eq(Bytes::splat(0)) & a.simd_ne(Bytes::splat(0))).select(Bytes::splat(255), a)
        } else {
            a
        };
        store(
            &mut dst[i * C..],
            std::array::from_fn::<_, C, _>(|c| blend(d[c], s[c], if c == 3 { a } else { rgb_a })),
        );
    }
    end
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sampling_matches_scalar_with_unaligned_buffers_and_tails() {
        for n in [0, 1, 15, 16, 17, 31, 32, 33, 257] {
            let src: Vec<_> = (0..n * 6 + 1).map(|i| (i * 37 + i / 11) as u8).collect();
            let mut dst = vec![93; n * 3 + 2];

            let end = nearest_half_rgb(&src[1..], &mut dst[1..n * 3 + 1]);
            assert_eq!(end, n / 16 * 48);
            for i in 0..end / 3 {
                assert_eq!(&dst[1 + i * 3..4 + i * 3], &src[4 + i * 6..7 + i * 6]);
            }
            assert_eq!(dst[0], 93);
            assert!(dst[1 + end..].iter().all(|&v| v == 93));
            dst.fill(93);

            let end = reverse_rgb(&src[1..n * 3 + 1], &mut dst[1..n * 3 + 1]);
            assert_eq!(end, n / 16 * 16);
            for i in 0..end {
                let j = 1 + (n - i - 1) * 3;
                assert_eq!(&dst[1 + i * 3..4 + i * 3], &src[j..j + 3]);
            }
            assert_eq!(dst[0], 93);
            assert!(dst[1 + end * 3..].iter().all(|&v| v == 93));
        }
    }

    #[test]
    fn reduction_matches_every_possible_sum() {
        // The reciprocal product exceeds i32::MAX at high pixel sums.
        // Match the scalar kernel's unsigned accumulator throughout.
        for sum in 0_u32..=9 * 255 {
            let mut remaining = sum;
            let values: [u8; 9] = std::array::from_fn(|_| {
                let v = remaining.min(255);
                remaining -= v;
                v as u8
            });
            let rows: [Vec<u8>; 3] = std::array::from_fn(|r| (0..52).map(|i| if i == 0 { 93 } else { values[r * 3 + (i - 1) % 3] }).collect());
            let mut dst = [93; 19];

            let end = reduce_three_l([&rows[0][1..], &rows[1][1..], &rows[2][1..]], &mut dst[1..18]);
            assert_eq!(end, 16);
            assert_eq!(&dst[1..17], &[(((sum + 4) * 1_864_135) >> 24) as u8; 16]);
            assert_eq!(dst[0], 93);
            assert_eq!(&dst[17..], &[93; 2]);
        }
    }

    #[test]
    fn layouts_and_alpha_match_scalar_with_guards() {
        for n in [0, 1, 15, 16, 17, 31, 32, 33, 257] {
            let bands: [Vec<u8>; 4] = std::array::from_fn(|c| (0..n + 1).map(|i| (i * 37 + c * 61) as u8).collect());
            let rgb: Vec<_> = (0..n * 3 + 1).map(|i| (i * 53) as u8).collect();
            let end = n / 16 * 16;
            let mut dst = vec![93; n * 4 + 2];

            unsafe {
                assert_eq!(putalpha_rgb(&rgb[1..], &bands[3][1..], &mut dst[1..n * 4 + 1]), end);
                for i in 0..end {
                    assert_eq!(&dst[1 + i * 4..4 + i * 4], &rgb[1 + i * 3..4 + i * 3]);
                    assert_eq!(dst[4 + i * 4], bands[3][1 + i]);
                }
                assert_eq!(putalpha_rgba(&mut dst[1..n * 4 + 1], &bands[0][1..]), end);
                for i in 0..end {
                    assert_eq!(&dst[1 + i * 4..4 + i * 4], &rgb[1 + i * 3..4 + i * 3]);
                    assert_eq!(dst[4 + i * 4], bands[0][1 + i]);
                }
                assert_eq!(dst[0], 93);
                assert!(dst[1 + end * 4..].iter().all(|&v| v == 93));

                let mut gray = vec![std::mem::MaybeUninit::new(93); n * 3 + 2];
                assert_eq!(gray_to_rgb(&bands[0][1..], &mut gray[1..n * 3 + 1]), end);
                for (i, v) in gray.iter().enumerate() {
                    let expected = if (1..1 + end * 3).contains(&i) { bands[0][1 + (i - 1) / 3] } else { 93 };
                    assert_eq!(v.assume_init(), expected);
                }
                check_merge::<1>(&bands, n);
                check_merge::<2>(&bands, n);
                check_merge::<3>(&bands, n);
                check_merge::<4>(&bands, n);
            }
        }
    }

    fn check_merge<const C: usize>(bands: &[Vec<u8>; 4], n: usize) {
        let mut dst = vec![[93; C]; n + 2];
        let end = merge::<C>(std::array::from_fn(|c| &bands[c][1..]), &mut dst[1..n + 1]);
        assert_eq!(end, n / 16 * 16);
        for i in 0..end {
            assert_eq!(dst[1 + i], std::array::from_fn(|c| bands[c][1 + i]));
        }
        assert_eq!(dst[0], [93; C]);
        assert!(dst[1 + end..].iter().all(|v| *v == [93; C]));
    }

    #[test]
    fn masked_paste_matches_scalar_all_modes_and_alpha_values() {
        check_paste::<1, 1>();
        check_paste::<1, 4>();
        check_paste::<2, 1>();
        check_paste::<2, 4>();
        check_paste::<3, 1>();
        check_paste::<3, 4>();
        check_paste::<4, 1>();
        check_paste::<4, 4>();
    }

    fn check_paste<const C: usize, const M: usize>() {
        for n in [0, 1, 15, 16, 17, 31, 32, 33, 257] {
            for alpha in 0..=255u8 {
                for fill in [false, true] {
                    let src: Vec<_> = (0..n * C + 1).map(|i| (i * 37) as u8).collect();
                    let mask: Vec<_> = (0..n * M + 1).map(|i| if i % M == 0 { alpha } else { 71 }).collect();
                    let mut dst: Vec<_> = (0..n * C + 2).map(|i| if i % 3 == 0 { 0 } else { (i * 61) as u8 }).collect();
                    let mut expected = dst.clone();
                    for i in 0..n / 16 * 16 {
                        let a = u32::from(mask[(i + 1) * M]);
                        let replace = fill && C == 4 && M == 1 && dst[(i + 1) * C] == 0 && a != 0;
                        for c in 0..C {
                            let a = if replace && c < 3 { 255 } else { a };
                            let j = 1 + i * C + c;
                            expected[j] = ((u32::from(dst[j]) * (255 - a) + u32::from(src[j]) * a + 127) / 255) as u8;
                        }
                    }

                    let end = paste_masked::<C, M>(&mut dst[1..n * C + 1], &src[1..], &mask[1..], fill);
                    assert_eq!(end, n / 16 * 16);
                    assert_eq!(dst, expected, "C={C} M={M} n={n} alpha={alpha} fill={fill}");
                }
            }
        }
    }
}
