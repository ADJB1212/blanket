//! Exact compression preparation: bounded SIMD loads, scalar tails, and
//! coarse Rayon partitions. Codec libraries handle their own entropy SIMD.

use rayon::prelude::*;

use blanket_core::parallel::{chunks_mut_above, should_parallel};

const MEMORY_PARALLEL_BYTES: usize = 4 * 1024 * 1024;
const CHUNK_PIXELS: usize = 64 * 1024;

pub(crate) fn matches_key(source: &[u8], key: [u8; 3]) -> bool {
    if should_parallel(source.len(), CHUNK_PIXELS * 4, MEMORY_PARALLEL_BYTES) {
        source.par_chunks(CHUNK_PIXELS * 4).all(|src| matches_key_chunk(src, key))
    } else {
        matches_key_chunk(source, key)
    }
}

fn matches_key_chunk(source: &[u8], key: [u8; 3]) -> bool {
    let mut offset = 0;
    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;
        // SAFETY: mandatory NEON, loading only complete 16-pixel blocks.
        unsafe {
            while offset + 64 <= source.len() {
                let p = vld4q_u8(source.as_ptr().add(offset));
                let same = vandq_u8(
                    vandq_u8(vceqq_u8(p.0, vdupq_n_u8(key[0])), vceqq_u8(p.1, vdupq_n_u8(key[1]))),
                    vceqq_u8(p.2, vdupq_n_u8(key[2])),
                );
                let valid = vorrq_u8(
                    vandq_u8(same, vceqq_u8(p.3, vdupq_n_u8(0))),
                    vbicq_u8(vceqq_u8(p.3, vdupq_n_u8(255)), same),
                );
                if vminvq_u8(valid) != 255 {
                    return false;
                }
                offset += 64;
            }
        }
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if std::arch::is_x86_feature_detected!("sse2") {
        // SAFETY: CPU feature checked; kernel bounds all loads.
        match unsafe { x86::matches_key(source, key) } {
            Some(done) => offset = done,
            None => return false,
        }
    }
    source[offset..]
        .as_chunks::<4>()
        .0
        .iter()
        .all(|p| (p[3] == 0 && p[..3] == key) || (p[3] == 255 && p[..3] != key))
}

/// Fixed divisors let LLVM vectorize exact wide-sample normalization. Preserve
/// the existing rounded full-range mapping, including 16-bit identity input.
pub(crate) fn normalize_u16(source: &[u8], depth: u8) -> Vec<u16> {
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

pub(crate) fn properties(source: &[u8], channels: usize) -> (bool, bool) {
    if channels == 1 {
        return (true, true);
    }
    let chunk = CHUNK_PIXELS * channels;
    if should_parallel(source.len(), chunk, MEMORY_PARALLEL_BYTES) {
        // Photographic RGB and nonopaque RGBA usually disprove both possible
        // reductions immediately. Avoid scheduling a scan of every chunk.
        let prefix = properties_chunk(&source[..source.len().min(64 * channels)], channels);
        if !prefix.0 && (!prefix.1 || channels == 3) {
            return prefix;
        }
        source
            .par_chunks(chunk)
            .map(|src| properties_chunk(src, channels))
            .reduce(|| (true, true), |a, b| (a.0 && b.0, a.1 && b.1))
    } else {
        properties_chunk(source, channels)
    }
}

fn properties_chunk(source: &[u8], channels: usize) -> (bool, bool) {
    let mut gray = true;
    let mut opaque = true;
    let mut offset = 0;
    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;
        // SAFETY: NEON is mandatory. Each load covers 16 complete pixels.
        unsafe {
            while offset + 16 * channels <= source.len() {
                let (r, g, b, a) = if channels == 3 {
                    let p = vld3q_u8(source.as_ptr().add(offset));
                    (p.0, p.1, p.2, vdupq_n_u8(255))
                } else {
                    let p = vld4q_u8(source.as_ptr().add(offset));
                    (p.0, p.1, p.2, p.3)
                };
                gray &= vminvq_u8(vandq_u8(vceqq_u8(r, g), vceqq_u8(r, b))) == 255;
                opaque &= vminvq_u8(a) == 255;
                offset += 16 * channels;
                if !gray && (!opaque || channels == 3) {
                    return (gray, opaque);
                }
            }
        }
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if std::arch::is_x86_feature_detected!("ssse3") {
        // SAFETY: CPU support checked; kernel bounds loads and handles tails.
        let result = unsafe { x86::properties(source, channels) };
        (offset, gray, opaque) = result;
    }
    for pixel in source[offset..].chunks_exact(channels) {
        gray &= pixel[0] == pixel[1] && pixel[0] == pixel[2];
        opaque &= channels == 3 || pixel[3] == 255;
        if !gray && (!opaque || channels == 3) {
            break;
        }
    }
    (gray, opaque)
}

/// Extract the red/gray channel (plus alpha for LA), after grayscale validation.
/// Exact sub-byte samples are obtained by shifting, avoiding integer division.
pub(crate) fn select(source: &[u8], channels: usize, alpha: bool, shift: u8) -> Vec<u8> {
    let target_channels = if alpha { 2 } else { 1 };
    let mut output = vec![0; source.len() / channels * target_channels];
    chunks_mut_above(
        &mut output,
        CHUNK_PIXELS * target_channels,
        MEMORY_PARALLEL_BYTES / channels * target_channels,
        |i, dst| {
            let start = i * CHUNK_PIXELS * channels;
            let src = &source[start..start + dst.len() / target_channels * channels];
            select_chunk(src, dst, channels, alpha, shift);
        },
    );
    output
}

fn select_chunk(source: &[u8], output: &mut [u8], channels: usize, alpha: bool, shift: u8) {
    let target_channels = if alpha { 2 } else { 1 };
    let mut done = 0;
    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;
        // SAFETY: all loads/stores cover exactly 16 complete pixels. Callers
        // use alpha only for RGBA -> LA and shift only for single-channel output.
        unsafe {
            while (done + 16) * channels <= source.len() {
                let ptr = source.as_ptr().add(done * channels);
                let (gray, a) = match channels {
                    1 => (vld1q_u8(ptr), vdupq_n_u8(255)),
                    3 => (vld3q_u8(ptr).0, vdupq_n_u8(255)),
                    _ => {
                        let p = vld4q_u8(ptr);
                        (p.0, p.3)
                    }
                };
                let gray = vshlq_u8(gray, vdupq_n_s8(-(shift as i8)));
                let ptr = output.as_mut_ptr().add(done * target_channels);
                if alpha {
                    vst2q_u8(ptr, uint8x16x2_t(gray, a));
                } else {
                    vst1q_u8(ptr, gray);
                }
                done += 16;
            }
        }
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if std::arch::is_x86_feature_detected!("ssse3") {
        // SAFETY: feature checked; kernel bounds source and destination access.
        done = unsafe { x86::select(source, output, channels, alpha, shift) };
    }
    for (src, dst) in source[done * channels..]
        .chunks_exact(channels)
        .zip(output[done * target_channels..].chunks_exact_mut(target_channels))
    {
        dst[0] = src[0] >> shift;
        if alpha {
            dst[1] = src[3];
        }
    }
}

pub(crate) fn fits_depth(gray: &[u8], bits: usize) -> bool {
    if bits == 8 {
        return true;
    }
    let check = |src: &[u8]| fits_depth_chunk(src, bits);
    if should_parallel(gray.len(), CHUNK_PIXELS, MEMORY_PARALLEL_BYTES) {
        gray.par_chunks(CHUNK_PIXELS).all(check)
    } else {
        check(gray)
    }
}

fn fits_depth_chunk(source: &[u8], bits: usize) -> bool {
    let mut offset = 0;
    let step = (255 / ((1 << bits) - 1)) as u8;
    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;
        // SAFETY: each load stays inside the slice; NEON is mandatory.
        unsafe {
            while offset + 16 <= source.len() {
                let v = vld1q_u8(source.as_ptr().add(offset));
                let reduced = vshlq_u8(v, vdupq_n_s8(bits as i8 - 8));
                if vminvq_u8(vceqq_u8(v, vmulq_u8(reduced, vdupq_n_u8(step)))) != 255 {
                    return false;
                }
                offset += 16;
            }
        }
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if std::arch::is_x86_feature_detected!("sse2") {
        // SAFETY: feature checked and all vector loads bounded in the kernel.
        match unsafe { x86::fits_depth(source, bits) } {
            Some(done) => offset = done,
            None => return false,
        }
    }
    source[offset..].iter().all(|&v| v % step == 0)
}

pub(crate) fn remap(source: &[u8], table: &[u8; 256]) -> Vec<u8> {
    let mut output = vec![0; source.len()];
    chunks_mut_above(&mut output, CHUNK_PIXELS, MEMORY_PARALLEL_BYTES, |i, dst| {
        blanket_ops::ops_simd::lut::<1>(&source[i * CHUNK_PIXELS..i * CHUNK_PIXELS + dst.len()], dst, table);
    });
    output
}

pub(crate) fn pack_rows(samples: &[u8], width: usize, bits: usize) -> Vec<u8> {
    pack_rows_impl(samples, width, bits, PackTransform::Identity)
}

pub(crate) fn pack_gray_rows(samples: &[u8], width: usize, bits: usize) -> Vec<u8> {
    pack_rows_impl(samples, width, bits, PackTransform::Shift((8 - bits) as u8))
}

pub(crate) fn pack_mapped_rows(samples: &[u8], width: usize, bits: usize, mapping: &[u8; 256]) -> Vec<u8> {
    pack_rows_impl(samples, width, bits, PackTransform::Map(mapping))
}

#[derive(Clone, Copy)]
enum PackTransform<'a> {
    Identity,
    Shift(u8),
    Map(&'a [u8; 256]),
}

/// Transform in cache-sized scratch buffers, then immediately pack the rows.
/// This avoids writing and rereading a full image of temporary byte indices.
fn pack_rows_impl(samples: &[u8], width: usize, bits: usize, transform: PackTransform<'_>) -> Vec<u8> {
    if bits == 8 {
        return match transform {
            PackTransform::Identity | PackTransform::Shift(0) => samples.to_vec(),
            PackTransform::Shift(shift) => select(samples, 1, false, shift),
            PackTransform::Map(mapping) => remap(samples, mapping),
        };
    }
    let row_bytes = (width * bits).div_ceil(8);
    let rows_per_chunk = (CHUNK_PIXELS / width).max(1);
    let mut output = vec![0; row_bytes * (samples.len() / width)];
    let fill = |i: usize, dst: &mut [u8], scratch: &mut Vec<u8>| {
        let start = i * rows_per_chunk * width;
        let source = &samples[start..start + dst.len() / row_bytes * width];
        let source = match transform {
            PackTransform::Identity => source,
            _ => {
                scratch.resize(source.len(), 0);
                match transform {
                    PackTransform::Shift(shift) => select_chunk(source, scratch, 1, false, shift),
                    PackTransform::Map(mapping) => blanket_ops::ops_simd::lut::<1>(source, scratch, mapping),
                    PackTransform::Identity => unreachable!(),
                }
                scratch.as_slice()
            }
        };
        for (src, dst) in source.chunks_exact(width).zip(dst.chunks_exact_mut(row_bytes)) {
            pack_row(src, dst, bits);
        }
    };
    if should_parallel(samples.len(), rows_per_chunk * width, MEMORY_PARALLEL_BYTES) {
        output
            .par_chunks_mut(rows_per_chunk * row_bytes)
            .enumerate()
            .for_each_init(Vec::new, |scratch, (i, dst)| fill(i, dst, scratch));
    } else {
        let mut scratch = Vec::new();
        output
            .chunks_mut(rows_per_chunk * row_bytes)
            .enumerate()
            .for_each(|(i, dst)| fill(i, dst, &mut scratch));
    }
    output
}

fn pack_row(source: &[u8], output: &mut [u8], bits: usize) {
    let mut done = 0;
    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;
        // SAFETY: loads are bounded by complete blocks; each block emits only
        // its packed bytes. Scalar tails handle row padding independently.
        unsafe {
            if bits == 4 {
                while done + 32 <= source.len() {
                    let v = vld2q_u8(source.as_ptr().add(done));
                    vst1q_u8(output.as_mut_ptr().add(done / 2), vorrq_u8(vshlq_n_u8::<4>(v.0), v.1));
                    done += 32;
                }
            } else if bits == 2 {
                while done + 64 <= source.len() {
                    let v = vld4q_u8(source.as_ptr().add(done));
                    let packed = vorrq_u8(vorrq_u8(vshlq_n_u8::<6>(v.0), vshlq_n_u8::<4>(v.1)), vorrq_u8(vshlq_n_u8::<2>(v.2), v.3));
                    vst1q_u8(output.as_mut_ptr().add(done / 4), packed);
                    done += 64;
                }
            } else {
                let weights = vld1q_u8([128, 64, 32, 16, 8, 4, 2, 1, 128, 64, 32, 16, 8, 4, 2, 1].as_ptr());
                while done + 16 <= source.len() {
                    let v = vmulq_u8(vld1q_u8(source.as_ptr().add(done)), weights);
                    output[done / 8] = vaddv_u8(vget_low_u8(v));
                    output[done / 8 + 1] = vaddv_u8(vget_high_u8(v));
                    done += 16;
                }
            }
        }
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if std::arch::is_x86_feature_detected!("ssse3") {
        // SAFETY: checked CPU support; kernel bounds all accesses.
        done = unsafe { x86::pack(source, output, bits) };
    }
    for (x, &sample) in source.iter().enumerate().skip(done) {
        output[x * bits / 8] |= sample << (8 - bits - x * bits % 8);
    }
}

/// Slice equality already uses the platform's optimized SIMD memcmp. Only
/// distribute very large scans, preserving the cheap early-mismatch path.
pub(crate) fn equal(left: &[u8], right: &[u8]) -> bool {
    if left.len() != right.len() {
        return false;
    }
    if !should_parallel(left.len(), 256 * 1024, 8 * 1024 * 1024) {
        return left == right;
    }
    left[..1024] == right[..1024] && left.par_chunks(256 * 1024).zip(right.par_chunks(256 * 1024)).all(|(a, b)| a == b)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    #[target_feature(enable = "sse2")]
    pub(super) unsafe fn matches_key(src: &[u8], key: [u8; 3]) -> Option<usize> {
        let mut done = 0;
        unsafe {
            let key = _mm_set1_epi32(i32::from_le_bytes([key[0], key[1], key[2], 0]));
            let rgb = _mm_set1_epi32(0x00ff_ffff);
            let alpha = _mm_set1_epi32(0xff00_0000_u32 as i32);
            while done + 16 <= src.len() {
                let v = _mm_loadu_si128(src.as_ptr().add(done).cast());
                let same = _mm_cmpeq_epi32(_mm_and_si128(v, rgb), key);
                let a = _mm_and_si128(v, alpha);
                let valid = _mm_or_si128(
                    _mm_and_si128(same, _mm_cmpeq_epi32(a, _mm_setzero_si128())),
                    _mm_andnot_si128(same, _mm_cmpeq_epi32(a, alpha)),
                );
                if _mm_movemask_epi8(valid) != 0xffff {
                    return None;
                }
                done += 16;
            }
        }
        Some(done)
    }

    #[target_feature(enable = "ssse3")]
    pub(super) unsafe fn properties(src: &[u8], channels: usize) -> (usize, bool, bool) {
        let mut offset = 0;
        let (mut gray, mut opaque) = (true, true);
        unsafe {
            let r = if channels == 3 { [0, 3, 6, 9] } else { [0, 4, 8, 12] };
            let mask = |delta: i8| {
                _mm_setr_epi8(
                    r[0] + delta,
                    r[1] + delta,
                    r[2] + delta,
                    r[3] + delta,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                )
            };
            while offset + 16 <= src.len() {
                let v = _mm_loadu_si128(src.as_ptr().add(offset).cast());
                let red = _mm_shuffle_epi8(v, mask(0));
                gray &= _mm_movemask_epi8(_mm_and_si128(
                    _mm_cmpeq_epi8(red, _mm_shuffle_epi8(v, mask(1))),
                    _mm_cmpeq_epi8(red, _mm_shuffle_epi8(v, mask(2))),
                )) == 0xffff;
                if channels == 4 {
                    opaque &= _mm_movemask_epi8(_mm_cmpeq_epi8(_mm_shuffle_epi8(v, mask(3)), _mm_set1_epi8(-1))) & 15 == 15;
                }
                offset += 4 * channels;
                if !gray && (!opaque || channels == 3) {
                    break;
                }
            }
        }
        (offset, gray, opaque)
    }

    #[target_feature(enable = "ssse3")]
    pub(super) unsafe fn select(src: &[u8], dst: &mut [u8], channels: usize, alpha: bool, shift: u8) -> usize {
        let mut done = 0;
        let count = if channels == 1 { 16 } else { 4 };
        let target = if alpha { 2 } else { 1 };
        unsafe {
            let mask = if alpha {
                _mm_setr_epi8(0, 3, 4, 7, 8, 11, 12, 15, -1, -1, -1, -1, -1, -1, -1, -1)
            } else {
                _mm_setr_epi8(
                    0,
                    channels as i8,
                    (2 * channels) as i8,
                    (3 * channels) as i8,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                )
            };
            while done * channels + 16 <= src.len() {
                let mut v = _mm_loadu_si128(src.as_ptr().add(done * channels).cast());
                if channels != 1 {
                    v = _mm_shuffle_epi8(v, mask);
                }
                v = _mm_and_si128(
                    _mm_srl_epi16(v, _mm_cvtsi32_si128(i32::from(shift))),
                    _mm_set1_epi8((255_u8 >> shift) as i8),
                );
                let mut buffer = [0_u8; 16];
                _mm_storeu_si128(buffer.as_mut_ptr().cast(), v);
                dst[done * target..(done + count) * target].copy_from_slice(&buffer[..count * target]);
                done += count;
            }
        }
        done
    }

    #[target_feature(enable = "sse2")]
    pub(super) unsafe fn fits_depth(src: &[u8], bits: usize) -> Option<usize> {
        let mut done = 0;
        unsafe {
            while done + 16 <= src.len() {
                let v = _mm_loadu_si128(src.as_ptr().add(done).cast());
                let shift = _mm_cvtsi32_si128((8 - bits) as i32);
                let mut restored = _mm_and_si128(_mm_srl_epi16(v, shift), _mm_set1_epi8(((1 << bits) - 1) as i8));
                if bits == 1 {
                    restored = _mm_sub_epi8(_mm_setzero_si128(), restored);
                } else {
                    if bits == 2 {
                        restored = _mm_or_si128(restored, _mm_slli_epi16::<2>(restored));
                    }
                    restored = _mm_or_si128(restored, _mm_slli_epi16::<4>(restored));
                }
                if _mm_movemask_epi8(_mm_cmpeq_epi8(v, restored)) != 0xffff {
                    return None;
                }
                done += 16;
            }
        }
        Some(done)
    }

    #[target_feature(enable = "ssse3")]
    pub(super) unsafe fn pack(src: &[u8], dst: &mut [u8], bits: usize) -> usize {
        let mut done = 0;
        unsafe {
            while done + 16 <= src.len() {
                let v = _mm_loadu_si128(src.as_ptr().add(done).cast());
                if bits == 1 {
                    let mask = _mm_movemask_epi8(_mm_slli_epi16::<7>(v)) as u16;
                    dst[done / 8] = (mask as u8).reverse_bits();
                    dst[done / 8 + 1] = ((mask >> 8) as u8).reverse_bits();
                } else {
                    let weights = if bits == 4 { _mm_set1_epi16(0x0110) } else { _mm_set1_epi32(0x01041040) };
                    let mut packed = _mm_maddubs_epi16(v, weights);
                    if bits == 2 {
                        packed = _mm_hadd_epi16(packed, _mm_setzero_si128());
                    }
                    packed = _mm_packus_epi16(packed, _mm_setzero_si128());
                    let mut buffer = [0_u8; 16];
                    _mm_storeu_si128(buffer.as_mut_ptr().cast(), packed);
                    dst[done * bits / 8..(done + 16) * bits / 8].copy_from_slice(&buffer[..2 * bits]);
                }
                done += 16;
            }
        }
        done
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn channel_kernels_match_scalar_at_unaligned_boundaries() {
        for channels in [1, 3, 4] {
            for count in [0, 1, 3, 4, 5, 15, 16, 17, 31, 32, 33, 63, 64, 65, 257] {
                let backing: Vec<u8> = (0..count * channels + 1).map(|i| (i * 71 + i / 7) as u8).collect();
                let src = &backing[1..];
                let expected_gray = channels == 1 || src.chunks_exact(channels).all(|p| p[0] == p[1] && p[0] == p[2]);
                let expected_opaque = channels != 4 || src.as_chunks::<4>().0.iter().all(|p| p[3] == 255);
                assert_eq!(properties(src, channels), (expected_gray, expected_opaque));
                for shift in [0, 4, 6, 7] {
                    assert_eq!(
                        select(src, channels, false, shift),
                        src.chunks_exact(channels).map(|p| p[0] >> shift).collect::<Vec<_>>()
                    );
                }
                if channels == 4 {
                    assert_eq!(
                        select(src, channels, true, 0),
                        src.as_chunks::<4>().0.iter().flat_map(|p| [p[0], p[3]]).collect::<Vec<_>>()
                    );
                }
            }
        }
        for channels in [3, 4] {
            let mut src = vec![255; 65 * channels];
            assert_eq!(properties(&src, channels), (true, true));
            for i in 0..65 {
                src[i * channels] = 254;
                assert_eq!(properties(&src, channels), (false, true));
                src[i * channels] = 255;
                if channels == 4 {
                    src[i * channels + 3] = 127;
                    assert_eq!(properties(&src, channels), (true, false));
                    src[i * channels + 3] = 255;
                }
            }
        }
    }

    #[test]
    fn exact_depth_checks_cover_every_sample_and_vector_lane() {
        for bits in [1, 2, 4, 8] {
            for value in 0..=255_u8 {
                for lane in 0..33 {
                    let mut src = [0; 33];
                    src[lane] = value;
                    assert_eq!(fits_depth(&src, bits), usize::from(value) % (255 / ((1 << bits) - 1)) == 0);
                }
            }
        }
    }

    #[test]
    fn packing_matches_scalar_with_odd_rows_and_vector_tails() {
        for bits in [1_usize, 2, 4, 8] {
            for width in [1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 257] {
                let src: Vec<_> = (0..width * 5).map(|i| ((i * 37 + i / 3) % (1 << bits)) as u8).collect();
                let row_bytes = (width * bits).div_ceil(8);
                let mut expected = vec![0_u8; row_bytes * 5];
                for (row, samples) in src.chunks_exact(width).enumerate() {
                    for (x, &sample) in samples.iter().enumerate() {
                        expected[row * row_bytes + x * bits / 8] |= sample << (8 - bits - x * bits % 8);
                    }
                }
                assert_eq!(pack_rows(&src, width, bits), expected, "width={width}, bits={bits}");
            }
        }
    }

    #[test]
    fn fused_packing_matches_separate_transforms() {
        for bits in [1_usize, 2, 4, 8] {
            let mask = ((1_usize << bits) - 1) as u8;
            let mapping = std::array::from_fn(|i| (255 - i) as u8 & mask);
            for width in [1, 7, 17, 65, 257, 65_537] {
                let source: Vec<_> = (0..width * 3).map(|i| (i * 71 + i / 7) as u8).collect();
                assert_eq!(
                    pack_mapped_rows(&source, width, bits, &mapping),
                    pack_rows(&remap(&source, &mapping), width, bits)
                );
                assert_eq!(
                    pack_gray_rows(&source, width, bits),
                    pack_rows(&select(&source, 1, false, (8 - bits) as u8), width, bits)
                );
            }
        }
    }

    #[test]
    fn transparency_key_rejects_changed_alpha_and_invisible_colors() {
        let key = [19, 37, 71];
        let mut src: Vec<_> = (0..65).flat_map(|i| if i % 2 == 0 { [19, 37, 71, 0] } else { [0, 0, 0, 255] }).collect();
        assert!(matches_key(&src, key));
        for i in 0..65 {
            let start = i * 4;
            let saved: [u8; 4] = src[start..start + 4].try_into().unwrap();
            for invalid in [[19, 37, 71, 255], [19, 37, 72, 0], [19, 37, 71, 127]] {
                src[start..start + 4].copy_from_slice(&invalid);
                assert!(!matches_key(&src, key));
            }
            src[start..start + 4].copy_from_slice(&saved);
        }
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

    #[test]
    fn large_parallel_kernels_match_single_worker() {
        let source: Vec<_> = (0..1024 * 1025)
            .flat_map(|i| {
                let v = (i % 16 * 17) as u8;
                [v, v, v, 255]
            })
            .collect();
        let table = std::array::from_fn(|i| (255 - i) as u8);
        let packed_table = std::array::from_fn(|i| (i % 16) as u8);
        let run = || {
            (
                properties(&source, 4),
                select(&source, 4, false, 4),
                remap(&source, &table),
                pack_rows(&source.iter().map(|v| v >> 4).collect::<Vec<_>>(), 1024, 4),
                pack_gray_rows(&source, 1024, 4),
                pack_mapped_rows(&source, 1024, 4, &packed_table),
            )
        };
        let serial = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap().install(run);
        let parallel = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap().install(run);
        assert_eq!(serial, parallel);
        let left = vec![42; 8 * 1024 * 1024 + 1];
        let mut right = left.clone();
        assert!(equal(&left, &right));
        right[left.len() - 1] = 0;
        assert!(!equal(&left, &right));
        assert!(!equal(&left, &right[..right.len() - 1]));
    }

    #[test]
    fn property_preflight_does_not_hide_late_alpha_or_color() {
        let mut source = [19, 37, 71, 255].repeat(1024 * 1025);
        assert_eq!(properties(&source, 4), (false, true));
        *source.last_mut().unwrap() = 0;
        assert_eq!(properties(&source, 4), (false, false));
        source.chunks_exact_mut(4).for_each(|p| p.copy_from_slice(&[71, 71, 71, 0]));
        assert_eq!(properties(&source, 4), (true, false));
        let last = source.len() - 4;
        source[last] = 72;
        assert_eq!(properties(&source, 4), (false, false));
    }
}
