//! SIMD kernels with scalar tails and portable fallbacks.
//!
//! Nightly uses portable SIMD; other toolchains use NEON on AArch64 and
//! runtime-detected AVX2 on x86/x86_64, with scalar fallbacks elsewhere.
//! Nearest grayscale gathers also support x86 CPUs with SSSE3.

/// Reusable grayscale gather map for an axis-aligned nearest transform.
pub(crate) struct NearestL {
    columns: Vec<usize>,
    width: usize,
    #[cfg(any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64"))]
    blocks: Vec<Option<(usize, [u8; 16])>>,
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    ssse3: bool,
}

impl NearestL {
    pub(crate) fn new(columns: &[Option<usize>], width: usize) -> Option<Self> {
        let columns: Vec<usize> = columns.iter().copied().collect::<Option<_>>()?;
        if columns.iter().any(|&x| x >= width) {
            return None;
        }
        #[cfg(any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64"))]
        let blocks = columns
            .as_chunks::<16>()
            .0
            .iter()
            .map(|xs| {
                let base = *xs.iter().min().unwrap();
                // Two table registers gather 16 nearby pixels without scalar
                // loads. Wide reductions and right-edge loads use the fallback.
                (width - base >= 32 && xs.iter().all(|&x| x - base < 32))
                    .then(|| (base, xs.map(|x| (x - base) as u8)))
            })
            .collect();
        Some(Self {
            columns,
            width,
            #[cfg(any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64"))]
            blocks,
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            ssse3: std::arch::is_x86_feature_detected!("ssse3"),
        })
    }

    pub(crate) fn is_vectorized(&self) -> bool {
        #[cfg(target_arch = "aarch64")]
        {
            self.blocks.iter().any(Option::is_some)
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            self.ssse3 && self.blocks.iter().any(Option::is_some)
        }
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64")))]
        {
            false
        }
    }

    pub(crate) fn sample(&self, source: &[u8], output: &mut [u8]) {
        assert_eq!(source.len(), self.width);
        assert_eq!(output.len(), self.columns.len());
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64")))]
        let done = 0;
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        let done = if self.ssse3 {
            // SAFETY: CPU support is checked when constructing the map; the
            // source and output lengths are checked above.
            unsafe { self.sample_ssse3(source, output) }
        } else {
            0
        };
        #[cfg(target_arch = "aarch64")]
        let done = {
            use std::arch::aarch64::*;
            for (i, block) in self.blocks.iter().enumerate() {
                if let Some((base, indices)) = block {
                    // SAFETY: new() checks both 16-byte loads fit the source;
                    // each block owns 16 output bytes. NEON is mandatory here.
                    unsafe {
                        let table = uint8x16x2_t(
                            vld1q_u8(source.as_ptr().add(*base)),
                            vld1q_u8(source.as_ptr().add(*base + 16)),
                        );
                        vst1q_u8(
                            output.as_mut_ptr().add(i * 16),
                            vqtbl2q_u8(table, vld1q_u8(indices.as_ptr())),
                        );
                    }
                } else {
                    for j in i * 16..(i + 1) * 16 {
                        output[j] = source[self.columns[j]];
                    }
                }
            }
            self.blocks.len() * 16
        };
        for (dst, &x) in output[done..].iter_mut().zip(&self.columns[done..]) {
            *dst = source[x];
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "ssse3")]
    unsafe fn sample_ssse3(&self, source: &[u8], output: &mut [u8]) -> usize {
        #[cfg(target_arch = "x86")]
        use std::arch::x86::*;
        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::*;

        for (i, block) in self.blocks.iter().enumerate() {
            if let Some((base, indices)) = block {
                // SAFETY: new() validates the complete 32-byte table. Each
                // block has 16 indices and writes 16 disjoint output bytes.
                unsafe {
                    let low = _mm_loadu_si128(source.as_ptr().add(*base).cast());
                    let high = _mm_loadu_si128(source.as_ptr().add(*base + 16).cast());
                    let indices = _mm_loadu_si128(indices.as_ptr().cast());
                    // PSHUFB zeros lanes whose index has its high bit set.
                    // Select 0..15 from low, 16..31 from high, then combine.
                    let low_indices =
                        _mm_or_si128(indices, _mm_cmpgt_epi8(indices, _mm_set1_epi8(15)));
                    let high_indices = _mm_sub_epi8(indices, _mm_set1_epi8(16));
                    let values = _mm_or_si128(
                        _mm_shuffle_epi8(low, low_indices),
                        _mm_shuffle_epi8(high, high_indices),
                    );
                    _mm_storeu_si128(output.as_mut_ptr().add(i * 16).cast(), values);
                }
            } else {
                for j in i * 16..(i + 1) * 16 {
                    output[j] = source[self.columns[j]];
                }
            }
        }
        self.blocks.len() * 16
    }
}

/// Reverse packed RGB pixels without reversing their channel order.
pub(crate) fn reverse_rgb(source: &[u8], output: &mut [u8]) {
    assert_eq!(source.len(), output.len());
    assert!(source.len().is_multiple_of(3));
    let pixels = source.len() / 3;
    #[cfg(not(target_arch = "aarch64"))]
    let done = 0;
    #[cfg(target_arch = "aarch64")]
    let done = {
        let mut done = 0;
        use std::arch::aarch64::*;
        // SAFETY: NEON is mandatory on AArch64. Each iteration loads and
        // stores exactly 16 complete RGB pixels within the provided slices.
        unsafe {
            while done + 16 <= pixels {
                let src = vld3q_u8(source.as_ptr().add((pixels - done - 16) * 3));
                let reverse = |v| {
                    let v = vrev64q_u8(v);
                    vextq_u8::<8>(v, v)
                };
                vst3q_u8(
                    output.as_mut_ptr().add(done * 3),
                    uint8x16x3_t(reverse(src.0), reverse(src.1), reverse(src.2)),
                );
                done += 16;
            }
        }
        done
    };
    for (i, dst) in output[done * 3..]
        .as_chunks_mut::<3>()
        .0
        .iter_mut()
        .enumerate()
    {
        let start = (pixels - done - i - 1) * 3;
        dst.copy_from_slice(&source[start..start + 3]);
    }
}

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
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 is detected at runtime; the kernel bounds all accesses.
        unsafe {
            if tables.len() == 256 {
                x86::lut::<1>(source, output, tables);
            } else {
                x86::lut::<C>(source, output, tables);
            }
        }
        return;
    }
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
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 is detected and input/output/table extents are validated.
        return unsafe { x86::colorize(source, output, tables) };
    }
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
    #[cfg(any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64"))]
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
            #[cfg(target_arch = "aarch64")]
            return unsafe { neon::vertical(source, output, stride, weights) };
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            if std::arch::is_x86_feature_detected!("avx2") {
                // SAFETY: AVX2 is detected; row extents and accumulator bounds
                // were checked above. The kernel processes eight bytes at a time.
                return unsafe { x86::vertical(source, output, stride, weights) };
            }
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

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(RUSTC_IS_NIGHTLY)
))]
mod x86 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    /// Narrow eight nonnegative i32 lanes to bytes, preserving lane order.
    #[target_feature(enable = "avx2")]
    unsafe fn pack_bytes(values: __m256i) -> __m128i {
        let shorts = _mm_packus_epi32(
            _mm256_castsi256_si128(values),
            _mm256_extracti128_si256::<1>(values),
        );
        _mm_packus_epi16(shorts, _mm_setzero_si128())
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn lut<const C: usize>(source: &[u8], output: &mut [u8], tables: &[u8]) {
        let end = source.len() / (8 * C) * (8 * C);
        if end == 0 {
            super::scalar_lut::<C>(source, output, tables);
            return;
        }
        // AVX2 gathers i32 entries, so widen once instead of reading four bytes
        // at each u8 entry (which would overread the end of the original table).
        let table: [[i32; 256]; C] =
            std::array::from_fn(|c| std::array::from_fn(|i| i32::from(tables[c * 256 + i])));
        let channels: [[i32; 8]; C] = std::array::from_fn(|block| {
            std::array::from_fn(|i| (((block * 8 + i) % C) * 256) as i32)
        });
        // SAFETY: each block loads/stores eight bytes. Indices select a byte's
        // channel and value in the widened table. Tails start on a pixel boundary.
        unsafe {
            for offset in (0..end).step_by(8 * C) {
                for (block, channel) in channels.iter().enumerate() {
                    let offset = offset + block * 8;
                    let bytes = _mm_loadl_epi64(source.as_ptr().add(offset).cast());
                    let indices = _mm256_add_epi32(
                        _mm256_cvtepu8_epi32(bytes),
                        _mm256_loadu_si256(channel.as_ptr().cast()),
                    );
                    let values =
                        _mm256_i32gather_epi32::<4>(table.as_flattened().as_ptr(), indices);
                    _mm_storel_epi64(output.as_mut_ptr().add(offset).cast(), pack_bytes(values));
                }
            }
        }
        super::scalar_lut::<C>(&source[end..], &mut output[end..], tables);
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn colorize(source: &[u8], output: &mut [u8], tables: &[u8]) -> usize {
        let end = source.len() / 8 * 8;
        if end == 0 {
            return 0;
        }
        let table: [i32; 256] = std::array::from_fn(|i| {
            i32::from(tables[i])
                | (i32::from(tables[256 + i]) << 8)
                | (i32::from(tables[512 + i]) << 16)
        });
        let shuffle = _mm_setr_epi8(0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, -1, -1, -1, -1);
        // SAFETY: each gather indexes a complete i32 entry. Each input block
        // contains eight bytes and writes two groups of twelve RGB bytes.
        unsafe {
            for offset in (0..end).step_by(8) {
                let bytes = _mm_loadl_epi64(source.as_ptr().add(offset).cast());
                let values =
                    _mm256_i32gather_epi32::<4>(table.as_ptr(), _mm256_cvtepu8_epi32(bytes));
                for (half, values) in [
                    _mm256_castsi256_si128(values),
                    _mm256_extracti128_si256::<1>(values),
                ]
                .into_iter()
                .enumerate()
                {
                    let packed = _mm_shuffle_epi8(values, shuffle);
                    let mut rgb = [0_u8; 16];
                    _mm_storeu_si128(rgb.as_mut_ptr().cast(), packed);
                    let start = offset * 3 + half * 12;
                    output[start..start + 12].copy_from_slice(&rgb[..12]);
                }
            }
        }
        end
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn vertical(
        source: &[u8],
        output: &mut [u8],
        stride: usize,
        weights: &[i32],
    ) -> usize {
        let end = output.len() / 8 * 8;
        // SAFETY: caller checks row extents and accumulator bounds. Only full
        // groups of eight bytes are loaded/stored; the caller handles the tail.
        unsafe {
            for x in (0..end).step_by(8) {
                let mut acc = _mm256_set1_epi32(1 << 21);
                for (i, &weight) in weights.iter().enumerate() {
                    let bytes = _mm_loadl_epi64(source.as_ptr().add(i * stride + x).cast());
                    acc = _mm256_add_epi32(
                        acc,
                        _mm256_mullo_epi32(_mm256_cvtepu8_epi32(bytes), _mm256_set1_epi32(weight)),
                    );
                }
                let clamped = _mm256_min_epi32(
                    _mm256_max_epi32(_mm256_srai_epi32::<22>(acc), _mm256_setzero_si256()),
                    _mm256_set1_epi32(255),
                );
                _mm_storel_epi64(output.as_mut_ptr().add(x).cast(), pack_bytes(clamped));
            }
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
        for n in [0, 1, 7, 8, 9, 15, 16, 17, 47, 48, 49, 255, 256, 257] {
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
        #[cfg(all(
            not(RUSTC_IS_NIGHTLY),
            any(target_arch = "x86", target_arch = "x86_64")
        ))]
        assert_eq!(
            count,
            if std::arch::is_x86_feature_detected!("avx2") {
                24
            } else {
                0
            }
        );
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
                let source: Vec<u8> = (0..1 + weights.len() * stride)
                    .map(|i| (i * 67) as u8)
                    .collect();
                let mut actual = vec![123; width + 2];
                let count = vertical(&source[1..], &mut actual[1..1 + width], stride, weights);
                let enabled = cfg!(any(RUSTC_IS_NIGHTLY, target_arch = "aarch64"));
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                let enabled = enabled || std::arch::is_x86_feature_detected!("avx2");
                assert_eq!(count, if enabled { width / 8 * 8 } else { 0 });
                for x in 0..count {
                    let sum = weights
                        .iter()
                        .enumerate()
                        .fold(1_i64 << 21, |sum, (i, &w)| {
                            sum + i64::from(source[1 + i * stride + x]) * i64::from(w)
                        });
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
            #[cfg(any(RUSTC_IS_NIGHTLY, target_arch = "aarch64"))]
            assert_eq!(count, n / 16 * 16);
            #[cfg(all(
                not(RUSTC_IS_NIGHTLY),
                any(target_arch = "x86", target_arch = "x86_64")
            ))]
            assert_eq!(
                count,
                if std::arch::is_x86_feature_detected!("avx2") {
                    n / 8 * 8
                } else {
                    0
                }
            );
            for i in 0..count {
                let v = usize::from(source[1 + i]);
                assert_eq!(
                    &output[1 + i * 3..1 + (i + 1) * 3],
                    &[tables[v], tables[256 + v], tables[512 + v]]
                );
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
                assert_eq!(
                    output[1 + i * 3..1 + (i + 1) * 3],
                    source[1 + (n - i - 1) * 3..1 + (n - i) * 3]
                );
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
                    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                    {
                        let mut scalar_map = NearestL::new(&columns, 256).unwrap();
                        scalar_map.ssse3 = false;
                        assert!(!scalar_map.is_vectorized());
                        let mut scalar = vec![0; n];
                        scalar_map.sample(&source[1..257], &mut scalar);
                        assert_eq!(scalar, output[1..n + 1]);
                    }
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
        #[cfg(target_arch = "aarch64")]
        assert!(map.is_vectorized());
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        assert_eq!(
            map.is_vectorized(),
            std::arch::is_x86_feature_detected!("ssse3")
        );
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64")))]
        assert!(!map.is_vectorized());
    }
}
