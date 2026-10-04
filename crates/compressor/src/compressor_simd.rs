//! Exact compression preparation: bounded SIMD loads, scalar tails, and
//! coarse Rayon partitions. Codec libraries handle their own entropy SIMD.

use std::simd::Simd;
use std::simd::cmp::SimdPartialEq;
use std::simd::num::SimdUint;

use blanket_core::parallel::{chunks_mut_above, should_parallel};
use blanket_core::pixels::{Bytes, load, store};
use rayon::prelude::*;

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
    while offset + 64 <= source.len() {
        let [r, g, b, a] = load::<4>(&source[offset..]);
        let same = r.simd_eq(Bytes::splat(key[0])) & g.simd_eq(Bytes::splat(key[1])) & b.simd_eq(Bytes::splat(key[2]));
        let valid = (same & a.simd_eq(Bytes::splat(0))) | (!same & a.simd_eq(Bytes::splat(255)));
        if !valid.all() {
            return false;
        }
        offset += 64;
    }
    source[offset..]
        .as_chunks::<4>()
        .0
        .iter()
        .all(|p| (p[3] == 0 && p[..3] == key) || (p[3] == 255 && p[..3] != key))
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
    while offset + 16 * channels <= source.len() {
        let (r, g, b, a) = if channels == 3 {
            let [r, g, b] = load::<3>(&source[offset..]);
            (r, g, b, Bytes::splat(255))
        } else {
            let [r, g, b, a] = load::<4>(&source[offset..]);
            (r, g, b, a)
        };
        gray &= (r.simd_eq(g) & r.simd_eq(b)).all();
        opaque &= a.simd_eq(Bytes::splat(255)).all();
        offset += 16 * channels;
        if !gray && (!opaque || channels == 3) {
            return (gray, opaque);
        }
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
    while (done + 16) * channels <= source.len() {
        let src = &source[done * channels..];
        let (gray, a) = match channels {
            1 => (Bytes::from_slice(&src[..16]), Bytes::splat(255)),
            3 => (load::<3>(src)[0], Bytes::splat(255)),
            _ => {
                let p = load::<4>(src);
                (p[0], p[3])
            }
        };
        let gray = gray >> shift;
        let dst = &mut output[done * target_channels..];
        if alpha {
            store(dst, [gray, a]);
        } else {
            gray.copy_to_slice(&mut dst[..16]);
        }
        done += 16;
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
    while offset + 16 <= source.len() {
        let v = Bytes::from_slice(&source[offset..offset + 16]);
        if !v.simd_eq((v >> (8 - bits) as u8) * Bytes::splat(step)).all() {
            return false;
        }
        offset += 16;
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
        let source = if let PackTransform::Identity = transform {
            source
        } else {
            scratch.resize(source.len(), 0);
            match transform {
                PackTransform::Shift(shift) => select_chunk(source, scratch, 1, false, shift),
                PackTransform::Map(mapping) => blanket_ops::ops_simd::lut::<1>(source, scratch, mapping),
                PackTransform::Identity => unreachable!(),
            }
            scratch.as_slice()
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
    if bits == 4 {
        while done + 32 <= source.len() {
            let [a, b] = load::<2>(&source[done..]);
            ((a << 4) | b).copy_to_slice(&mut output[done / 2..done / 2 + 16]);
            done += 32;
        }
    } else if bits == 2 {
        while done + 64 <= source.len() {
            let [a, b, c, d] = load::<4>(&source[done..]);
            ((a << 6) | (b << 4) | (c << 2) | d).copy_to_slice(&mut output[done / 4..done / 4 + 16]);
            done += 64;
        }
    } else {
        let weights = Simd::<u8, 8>::from_array([128, 64, 32, 16, 8, 4, 2, 1]);
        while done + 8 <= source.len() {
            output[done / 8] = (Simd::<u8, 8>::from_slice(&source[done..done + 8]) * weights).reduce_sum();
            done += 8;
        }
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
        source.as_chunks_mut::<4>().0.fill([71, 71, 71, 0]);
        assert_eq!(properties(&source, 4), (true, false));
        let last = source.len() - 4;
        source[last] = 72;
        assert_eq!(properties(&source, 4), (false, false));
    }
}
