//! SSSE3 packed-pixel kernels. Callers detect SSSE3 before entering and
//! provide matching, complete pixel buffers. All tails remain with callers.

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// Deinterleave/interleave exactly 16 pixels without padded buffer accesses.
#[target_feature(enable = "ssse3")]
unsafe fn load<const C: usize>(ptr: *const u8) -> [__m128i; C] {
    unsafe {
        let packed: [_; C] = std::array::from_fn(|b| _mm_loadu_si128(ptr.add(b * 16).cast()));
        std::array::from_fn(|c| {
            let mut value = _mm_setzero_si128();
            for (b, &bytes) in packed.iter().enumerate() {
                let mask: [i8; 16] = std::array::from_fn(|i| {
                    let index = i * C + c;
                    if index / 16 == b { (index % 16) as i8 } else { -1 }
                });
                value = _mm_or_si128(value, _mm_shuffle_epi8(bytes, _mm_loadu_si128(mask.as_ptr().cast())));
            }
            value
        })
    }
}

#[target_feature(enable = "ssse3")]
unsafe fn store<const C: usize>(ptr: *mut u8, bands: [__m128i; C]) {
    unsafe {
        for b in 0..C {
            let mut value = _mm_setzero_si128();
            for (c, &band) in bands.iter().enumerate() {
                let mask: [i8; 16] = std::array::from_fn(|i| {
                    let index = b * 16 + i;
                    if index % C == c { (index / C) as i8 } else { -1 }
                });
                value = _mm_or_si128(value, _mm_shuffle_epi8(band, _mm_loadu_si128(mask.as_ptr().cast())));
            }
            _mm_storeu_si128(ptr.add(b * 16).cast(), value);
        }
    }
}

#[target_feature(enable = "ssse3")]
pub(crate) unsafe fn reverse_rgb(src: &[u8], dst: &mut [u8]) -> usize {
    let end = src.len() / 3 / 16 * 16;
    unsafe {
        let reverse = _mm_setr_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        for i in (0..end).step_by(16) {
            let bands = load::<3>(src.as_ptr().add(src.len() - (i + 16) * 3));
            store(dst.as_mut_ptr().add(i * 3), bands.map(|v| _mm_shuffle_epi8(v, reverse)));
        }
    }
    end
}

#[target_feature(enable = "ssse3")]
pub(crate) unsafe fn nearest_half_rgb(src: &[u8], dst: &mut [u8]) -> usize {
    let end = dst.len() / 48 * 48;
    unsafe {
        let odd = _mm_setr_epi8(1, 3, 5, 7, 9, 11, 13, 15, -1, -1, -1, -1, -1, -1, -1, -1);
        for i in (0..end).step_by(48) {
            let a = load::<3>(src.as_ptr().add(i * 2));
            let b = load::<3>(src.as_ptr().add(i * 2 + 48));
            store::<3>(
                dst.as_mut_ptr().add(i),
                std::array::from_fn(|c| _mm_unpacklo_epi64(_mm_shuffle_epi8(a[c], odd), _mm_shuffle_epi8(b[c], odd))),
            );
        }
    }
    end
}

#[target_feature(enable = "ssse3")]
pub(crate) unsafe fn reduce_three_l(rows: [&[u8]; 3], dst: &mut [u8]) -> usize {
    let end = dst.len() / 16 * 16;
    unsafe {
        let zero = _mm_setzero_si128();
        for i in (0..end).step_by(16) {
            let mut lo = _mm_set1_epi16(4);
            let mut hi = lo;
            for row in rows {
                for v in load::<3>(row.as_ptr().add(i * 3)) {
                    lo = _mm_add_epi16(lo, _mm_unpacklo_epi8(v, zero));
                    hi = _mm_add_epi16(hi, _mm_unpackhi_epi8(v, zero));
                }
            }
            // Match the scalar Pillow-compatible reciprocal ((sum * 1_864_135) >> 24).
            let divisor = _mm_set1_epi16(7281);
            let value = _mm_packus_epi16(_mm_mulhi_epu16(lo, divisor), _mm_mulhi_epu16(hi, divisor));
            _mm_storeu_si128(dst.as_mut_ptr().add(i).cast(), value);
        }
    }
    end
}

#[target_feature(enable = "ssse3")]
pub(crate) unsafe fn gray_to_rgb(src: &[u8], dst: &mut [std::mem::MaybeUninit<u8>]) -> usize {
    let end = src.len() / 16 * 16;
    unsafe {
        for i in (0..end).step_by(16) {
            let gray = _mm_loadu_si128(src.as_ptr().add(i).cast());
            store(dst.as_mut_ptr().cast::<u8>().add(i * 3), [gray; 3]);
        }
    }
    end
}

#[target_feature(enable = "ssse3")]
pub(crate) unsafe fn merge<const C: usize>(bands: [&[u8]; C], dst: &mut [[u8; C]]) -> usize {
    let end = dst.len() / 16 * 16;
    unsafe {
        for i in (0..end).step_by(16) {
            store(dst.as_mut_ptr().add(i).cast(), bands.map(|b| _mm_loadu_si128(b.as_ptr().add(i).cast())));
        }
    }
    end
}

#[target_feature(enable = "ssse3")]
pub(crate) unsafe fn putalpha_rgb(src: &[u8], alpha: &[u8], dst: &mut [u8]) -> usize {
    let end = alpha.len() / 16 * 16;
    unsafe {
        for i in (0..end).step_by(16) {
            let [r, g, b] = load::<3>(src.as_ptr().add(i * 3));
            let a = _mm_loadu_si128(alpha.as_ptr().add(i).cast());
            store(dst.as_mut_ptr().add(i * 4), [r, g, b, a]);
        }
    }
    end
}

#[target_feature(enable = "ssse3")]
pub(crate) unsafe fn putalpha_rgba(dst: &mut [u8], alpha: &[u8]) -> usize {
    let end = alpha.len() / 16 * 16;
    unsafe {
        for i in (0..end).step_by(16) {
            let ptr = dst.as_mut_ptr().add(i * 4);
            let [r, g, b, _] = load::<4>(ptr);
            let a = _mm_loadu_si128(alpha.as_ptr().add(i).cast());
            store(ptr, [r, g, b, a]);
        }
    }
    end
}

#[target_feature(enable = "ssse3")]
unsafe fn blend(d: __m128i, s: __m128i, a: __m128i) -> __m128i {
    let zero = _mm_setzero_si128();
    let inv = _mm_xor_si128(a, _mm_set1_epi8(-1));
    let round = _mm_set1_epi16(128);
    let lo = _mm_add_epi16(
        round,
        _mm_add_epi16(
            _mm_mullo_epi16(_mm_unpacklo_epi8(d, zero), _mm_unpacklo_epi8(inv, zero)),
            _mm_mullo_epi16(_mm_unpacklo_epi8(s, zero), _mm_unpacklo_epi8(a, zero)),
        ),
    );
    let hi = _mm_add_epi16(
        round,
        _mm_add_epi16(
            _mm_mullo_epi16(_mm_unpackhi_epi8(d, zero), _mm_unpackhi_epi8(inv, zero)),
            _mm_mullo_epi16(_mm_unpackhi_epi8(s, zero), _mm_unpackhi_epi8(a, zero)),
        ),
    );
    _mm_packus_epi16(
        _mm_srli_epi16::<8>(_mm_add_epi16(lo, _mm_srli_epi16::<8>(lo))),
        _mm_srli_epi16::<8>(_mm_add_epi16(hi, _mm_srli_epi16::<8>(hi))),
    )
}

#[target_feature(enable = "ssse3")]
pub(crate) unsafe fn paste_masked<const C: usize, const M: usize>(dst: &mut [u8], src: &[u8], mask: &[u8], fill: bool) -> usize {
    let end = dst.len() / C / 16 * 16;
    unsafe {
        for i in (0..end).step_by(16) {
            let ptr = dst.as_mut_ptr().add(i * C);
            let d = load::<C>(ptr);
            let s = load::<C>(src.as_ptr().add(i * C));
            let a = load::<M>(mask.as_ptr().add(i * M))[M - 1];
            let rgb_a = if fill && C == 4 && M == 1 {
                let zero = _mm_setzero_si128();
                _mm_or_si128(a, _mm_andnot_si128(_mm_cmpeq_epi8(a, zero), _mm_cmpeq_epi8(d[C - 1], zero)))
            } else {
                a
            };
            store::<C>(ptr, std::array::from_fn(|c| blend(d[c], s[c], if c == 3 { a } else { rgb_a })));
        }
    }
    end
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sampling_matches_scalar_with_unaligned_buffers_and_tails() {
        if !is_x86_feature_detected!("ssse3") {
            return;
        }
        for n in [0, 1, 15, 16, 17, 31, 32, 33, 257] {
            let src: Vec<_> = (0..n * 6 + 1).map(|i| (i * 37 + i / 11) as u8).collect();
            let mut dst = vec![93; n * 3 + 2];
            // SAFETY: detected SSSE3; input has twice as many RGB pixels.
            let end = unsafe { nearest_half_rgb(&src[1..], &mut dst[1..n * 3 + 1]) };
            assert_eq!(end, n / 16 * 48);
            for i in 0..end / 3 {
                assert_eq!(&dst[1 + i * 3..4 + i * 3], &src[4 + i * 6..7 + i * 6]);
            }
            assert_eq!(dst[0], 93);
            assert!(dst[1 + end..].iter().all(|&v| v == 93));
            dst.fill(93);
            // SAFETY: detected SSSE3; input and output have matching RGB sizes.
            let end = unsafe { reverse_rgb(&src[1..n * 3 + 1], &mut dst[1..n * 3 + 1]) };
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
        if !is_x86_feature_detected!("ssse3") {
            return;
        }
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
            // SAFETY: detected SSSE3; each unaligned row has 3 * 17 bytes.
            let end = unsafe { reduce_three_l([&rows[0][1..], &rows[1][1..], &rows[2][1..]], &mut dst[1..18]) };
            assert_eq!(end, 16);
            assert_eq!(&dst[1..17], &[(((sum + 4) * 1_864_135) >> 24) as u8; 16]);
            assert_eq!(dst[0], 93);
            assert_eq!(&dst[17..], &[93; 2]);
        }
    }

    #[test]
    fn layouts_and_alpha_match_scalar_with_guards() {
        if !is_x86_feature_detected!("ssse3") {
            return;
        }
        for n in [0, 1, 15, 16, 17, 31, 32, 33, 257] {
            let bands: [Vec<u8>; 4] = std::array::from_fn(|c| (0..n + 1).map(|i| (i * 37 + c * 61) as u8).collect());
            let rgb: Vec<_> = (0..n * 3 + 1).map(|i| (i * 53) as u8).collect();
            let end = n / 16 * 16;
            let mut dst = vec![93; n * 4 + 2];
            // SAFETY: detected SSSE3; slices contain matching pixel counts.
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
                check_merge::<3>(&bands, n);
                check_merge::<4>(&bands, n);
            }
        }
    }

    unsafe fn check_merge<const C: usize>(bands: &[Vec<u8>; 4], n: usize) {
        let mut dst = vec![[93; C]; n + 2];
        let end = unsafe { merge::<C>(std::array::from_fn(|c| &bands[c][1..]), &mut dst[1..n + 1]) };
        assert_eq!(end, n / 16 * 16);
        for i in 0..end {
            assert_eq!(dst[1 + i], std::array::from_fn(|c| bands[c][1 + i]));
        }
        assert_eq!(dst[0], [93; C]);
        assert!(dst[1 + end..].iter().all(|v| *v == [93; C]));
    }

    #[test]
    fn masked_paste_matches_scalar_all_modes_and_alpha_values() {
        if !is_x86_feature_detected!("ssse3") {
            return;
        }
        check_paste::<1, 1>();
        check_paste::<1, 4>();
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
                    // SAFETY: detected SSSE3 and matching complete pixel slices.
                    let end = unsafe { paste_masked::<C, M>(&mut dst[1..n * C + 1], &src[1..], &mask[1..], fill) };
                    assert_eq!(end, n / 16 * 16);
                    assert_eq!(dst, expected, "C={C} M={M} n={n} alpha={alpha} fill={fill}");
                }
            }
        }
    }
}
