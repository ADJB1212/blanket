//! SIMD kernels with scalar tails and portable fallbacks.
//!
//! Nightly uses portable SIMD; other toolchains use NEON on AArch64 and
//! scalar kernels on other architectures.

pub(crate) fn lut<const C: usize>(source: &[u8], output: &mut [u8], tables: &[u8]) {
    assert_eq!(source.len(), output.len());
    assert!(tables.len() == 256 || tables.len() == C * 256);
    dispatch_lut::<C>(source, output, tables);
}

#[cfg(RUSTC_IS_NIGHTLY)]
fn dispatch_lut<const C: usize>(source: &[u8], output: &mut [u8], tables: &[u8]) {
    portable::lut::<C>(source, output, tables);
}

#[cfg(not(RUSTC_IS_NIGHTLY))]
fn dispatch_lut<const C: usize>(source: &[u8], output: &mut [u8], tables: &[u8]) {
    #[cfg(target_arch = "aarch64")]
    // SAFETY: NEON is mandatory on AArch64; the kernel bounds every load
    // and store and only reads complete 256-entry lookup tables.
    unsafe {
        neon::lut::<C>(source, output, tables)
    };
    #[cfg(not(target_arch = "aarch64"))]
    scalar_lut::<C>(source, output, tables);
}

fn scalar_lut<const C: usize>(source: &[u8], output: &mut [u8], tables: &[u8]) {
    if tables.len() == 256 {
        for (dst, &src) in output.iter_mut().zip(source) {
            *dst = tables[usize::from(src)];
        }
    } else {
        for (src, dst) in source
            .as_chunks::<C>()
            .0
            .iter()
            .zip(output.as_chunks_mut::<C>().0)
        {
            for c in 0..C {
                dst[c] = tables[c * 256 + usize::from(src[c])];
            }
        }
    }
}

pub(crate) fn colorize(source: &[u8], output: &mut [u8], tables: &[u8]) {
    assert_eq!(Some(output.len()), source.len().checked_mul(3));
    assert_eq!(tables.len(), 768);
    let start = dispatch_colorize(source, output, tables);
    for (&v, dst) in source[start..]
        .iter()
        .zip(output[start * 3..].as_chunks_mut::<3>().0)
    {
        let index = usize::from(v);
        *dst = [tables[index], tables[256 + index], tables[512 + index]];
    }
}

#[cfg(RUSTC_IS_NIGHTLY)]
fn dispatch_colorize(source: &[u8], output: &mut [u8], tables: &[u8]) -> usize {
    portable::colorize(source, output, tables)
}

#[cfg(not(RUSTC_IS_NIGHTLY))]
fn dispatch_colorize(source: &[u8], output: &mut [u8], tables: &[u8]) -> usize {
    #[cfg(target_arch = "aarch64")]
    // SAFETY: NEON is guaranteed, and input/output/table extents are validated.
    return unsafe { neon::colorize(source, output, tables) };
    #[cfg(not(target_arch = "aarch64"))]
    {
        let _ = (source, output, tables);
        0
    }
}

/// Resample contiguous lanes from several source rows, retaining fixed-point
/// coefficient rounding. Returns the byte count processed by SIMD.
pub(crate) fn vertical(source: &[u8], output: &mut [u8], stride: usize, weights: &[i32]) -> usize {
    dispatch_vertical(source, output, stride, weights)
}

#[cfg(RUSTC_IS_NIGHTLY)]
fn dispatch_vertical(source: &[u8], output: &mut [u8], stride: usize, weights: &[i32]) -> usize {
    let magnitude: i64 = weights.iter().map(|&w| i64::from(w).abs()).sum();
    if magnitude * 255 + (1 << 21) <= i64::from(i32::MAX) {
        assert!(
            weights.is_empty()
                || source.len()
                    >= (weights.len() - 1)
                        .saturating_mul(stride)
                        .saturating_add(output.len())
        );
        return portable::vertical(source, output, stride, weights);
    }
    0
}

#[cfg(not(RUSTC_IS_NIGHTLY))]
fn dispatch_vertical(source: &[u8], output: &mut [u8], stride: usize, weights: &[i32]) -> usize {
    #[cfg(target_arch = "aarch64")]
    {
        let magnitude: i64 = weights.iter().map(|&w| i64::from(w).abs()).sum();
        if magnitude * 255 + (1 << 21) <= i64::from(i32::MAX) {
            // SAFETY: Each referenced source row spans output.len() bytes.
            // Loads/stores operate only on complete groups of eight bytes.
            assert!(
                weights.is_empty()
                    || source.len()
                        >= (weights.len() - 1)
                            .saturating_mul(stride)
                            .saturating_add(output.len())
            );
            return unsafe { neon::vertical(source, output, stride, weights) };
        }
    }
    let _ = (source, output, stride, weights);
    0
}

#[cfg(RUSTC_IS_NIGHTLY)]
mod portable {
    use std::simd::Simd;
    use std::simd::prelude::*;

    const LANES: usize = 16;
    const VLANES: usize = 8;

    pub(super) fn lut<const C: usize>(source: &[u8], output: &mut [u8], tables: &[u8]) {
        let mut offset = 0;
        if tables.len() == 256 {
            while offset + LANES <= source.len() {
                let idx: Simd<usize, LANES> =
                    Simd::from_array(std::array::from_fn(|i| usize::from(source[offset + i])));
                let vals: Simd<u8, LANES> = Simd::gather_or_default(tables, idx);
                output[offset..offset + LANES].copy_from_slice(&vals.to_array());
                offset += LANES;
            }
            super::scalar_lut::<1>(&source[offset..], &mut output[offset..], tables);
        } else {
            while offset + LANES * C <= source.len() {
                for c in 0..C {
                    let src_idx: Simd<usize, LANES> =
                        Simd::from_array(std::array::from_fn(|i| offset + i * C + c));
                    let src_bytes: Simd<u8, LANES> = Simd::gather_or_default(source, src_idx);
                    let idx: Simd<usize, LANES> = src_bytes.cast();
                    let table_slice = &tables[c * 256..c * 256 + 256];
                    let vals: Simd<u8, LANES> = Simd::gather_or_default(table_slice, idx);
                    let arr = vals.to_array();
                    for i in 0..LANES {
                        output[offset + i * C + c] = arr[i];
                    }
                }
                offset += LANES * C;
            }
            super::scalar_lut::<C>(&source[offset..], &mut output[offset..], tables);
        }
    }

    pub(super) fn colorize(source: &[u8], output: &mut [u8], tables: &[u8]) -> usize {
        let mut offset = 0;
        while offset + LANES <= source.len() {
            let idx: Simd<usize, LANES> =
                Simd::from_array(std::array::from_fn(|i| usize::from(source[offset + i])));
            for c in 0..3 {
                let table_slice = &tables[c * 256..c * 256 + 256];
                let vals: Simd<u8, LANES> = Simd::gather_or_default(table_slice, idx);
                let arr = vals.to_array();
                for i in 0..LANES {
                    output[(offset + i) * 3 + c] = arr[i];
                }
            }
            offset += LANES;
        }
        offset
    }

    pub(super) fn vertical(
        source: &[u8],
        output: &mut [u8],
        stride: usize,
        weights: &[i32],
    ) -> usize {
        let end = output.len() / VLANES * VLANES;
        for x in (0..end).step_by(VLANES) {
            let mut acc: Simd<i32, VLANES> = Simd::splat(1 << 21);
            for (i, &weight) in weights.iter().enumerate() {
                let bytes: Simd<u8, VLANES> =
                    Simd::from_slice(&source[i * stride + x..i * stride + x + VLANES]);
                let widened: Simd<i32, VLANES> = bytes.cast();
                acc += widened * Simd::splat(weight);
            }
            let shifted = acc >> Simd::splat(22);
            let clamped = shifted.simd_clamp(Simd::splat(0), Simd::splat(255));
            let narrowed: Simd<u8, VLANES> = clamped.cast();
            output[x..x + VLANES].copy_from_slice(&narrowed.to_array());
        }
        end
    }
}

#[cfg(all(target_arch = "aarch64", not(RUSTC_IS_NIGHTLY)))]
mod neon {
    use std::arch::aarch64::*;

    #[inline]
    unsafe fn load_table(table: &[u8]) -> [uint8x16x4_t; 4] {
        // SAFETY: callers supply at least 256 bytes; each load reads 16.
        unsafe {
            std::array::from_fn(|i| {
                let p = table.as_ptr().add(i * 64);
                uint8x16x4_t(
                    vld1q_u8(p),
                    vld1q_u8(p.add(16)),
                    vld1q_u8(p.add(32)),
                    vld1q_u8(p.add(48)),
                )
            })
        }
    }

    #[inline]
    unsafe fn lookup(table: &[uint8x16x4_t; 4], index: uint8x16_t) -> uint8x16_t {
        // SAFETY: NEON is guaranteed. Out-of-table indices produce zero;
        // exactly one of the four 64-entry tables contributes to each lane.
        unsafe {
            let a = vqtbl4q_u8(table[0], index);
            let b = vqtbl4q_u8(table[1], vsubq_u8(index, vdupq_n_u8(64)));
            let c = vqtbl4q_u8(table[2], vsubq_u8(index, vdupq_n_u8(128)));
            let d = vqtbl4q_u8(table[3], vsubq_u8(index, vdupq_n_u8(192)));
            vorrq_u8(vorrq_u8(a, b), vorrq_u8(c, d))
        }
    }

    pub(super) unsafe fn lut<const C: usize>(source: &[u8], output: &mut [u8], tables: &[u8]) {
        let mut offset = 0;
        // SAFETY: every loop checks the full interleaved load/store size;
        // the remaining bytes are handled by the scalar implementation.
        unsafe {
            if tables.len() == 256 {
                let table = load_table(tables);
                while offset + 16 <= source.len() {
                    vst1q_u8(
                        output.as_mut_ptr().add(offset),
                        lookup(&table, vld1q_u8(source.as_ptr().add(offset))),
                    );
                    offset += 16;
                }
                super::scalar_lut::<1>(&source[offset..], &mut output[offset..], tables);
            } else if C == 3 {
                let r = load_table(&tables[..256]);
                let g = load_table(&tables[256..512]);
                let b = load_table(&tables[512..768]);
                while offset + 48 <= source.len() {
                    let src = vld3q_u8(source.as_ptr().add(offset));
                    vst3q_u8(
                        output.as_mut_ptr().add(offset),
                        uint8x16x3_t(lookup(&r, src.0), lookup(&g, src.1), lookup(&b, src.2)),
                    );
                    offset += 48;
                }
                super::scalar_lut::<C>(&source[offset..], &mut output[offset..], tables);
            } else {
                super::scalar_lut::<C>(source, output, tables);
            }
        }
    }

    pub(super) unsafe fn vertical(
        source: &[u8],
        output: &mut [u8],
        stride: usize,
        weights: &[i32],
    ) -> usize {
        let end = output.len() / 8 * 8;
        // SAFETY: caller checks source row extents and accumulator bounds.
        unsafe {
            for x in (0..end).step_by(8) {
                let mut low = vdupq_n_s32(1 << 21);
                let mut high = low;
                for (i, &weight) in weights.iter().enumerate() {
                    let bytes = vmovl_u8(vld1_u8(source.as_ptr().add(i * stride + x)));
                    let lo = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(bytes)));
                    let hi = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(bytes)));
                    low = vmlaq_n_s32(low, lo, weight);
                    high = vmlaq_n_s32(high, hi, weight);
                }
                let shorts = vcombine_u16(
                    vqmovun_s32(vshrq_n_s32::<22>(low)),
                    vqmovun_s32(vshrq_n_s32::<22>(high)),
                );
                vst1_u8(output.as_mut_ptr().add(x), vqmovn_u16(shorts));
            }
        }
        end
    }

    pub(super) unsafe fn colorize(source: &[u8], output: &mut [u8], tables: &[u8]) -> usize {
        let end = source.len() / 16 * 16;
        // SAFETY: table slices contain 256 bytes each; the loop loads 16 input
        // bytes and writes exactly 48 output bytes, with tails handled outside.
        unsafe {
            let r = load_table(&tables[..256]);
            let g = load_table(&tables[256..512]);
            let b = load_table(&tables[512..768]);
            for i in (0..end).step_by(16) {
                let v = vld1q_u8(source.as_ptr().add(i));
                vst3q_u8(
                    output.as_mut_ptr().add(i * 3),
                    uint8x16x3_t(lookup(&r, v), lookup(&g, v), lookup(&b, v)),
                );
            }
        }
        end
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lut_matches_scalar_at_vector_boundaries() {
        check_lut::<1>();
        check_lut::<3>();
        check_lut::<4>();
    }

    fn check_lut<const C: usize>() {
        for n in [0, 1, 15, 16, 17, 47, 48, 49, 255, 256, 257] {
            let input: Vec<u8> = (0..n * C).map(|i| (i * 37) as u8).collect();
            for channels in [1, C] {
                let tables: Vec<u8> = (0..channels * 256)
                    .map(|i| (i * 53 + i / 256 * 13) as u8)
                    .collect();
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
        #[cfg(any(RUSTC_IS_NIGHTLY, target_arch = "aarch64"))]
        assert_eq!(count, 24);
        for x in 0..count {
            let sum = weights
                .iter()
                .enumerate()
                .fold(1_i64 << 21, |sum, (i, &w)| {
                    sum + i64::from(source[i * 31 + x]) * i64::from(w)
                });
            assert_eq!(actual[x], (sum >> 22).clamp(0, 255) as u8);
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
        for n in [0, 1, 15, 16, 17, 255, 256, 257] {
            let source: Vec<u8> = (0..n).map(|i| (i * 37) as u8).collect();
            let mut output = vec![0; n * 3];
            colorize(&source, &mut output, &tables);
            for (pixel, &v) in output.as_chunks::<3>().0.iter().zip(&source) {
                let i = usize::from(v);
                assert_eq!(*pixel, [tables[i], tables[256 + i], tables[512 + i]]);
            }
        }
    }
}
