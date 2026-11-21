const COLOR_HASH_MULTIPLIER: u64 = 0x9e37_79b9_7f4a_7c15;
// Twice the 256-color palette limit keeps linear probing at most half full.
const COLOR_TABLE_BITS: u32 = 9;
const COLOR_TABLE_SIZE: usize = 1 << COLOR_TABLE_BITS;

pub(super) type ColorCounts = [(u64, usize, usize); COLOR_TABLE_SIZE];

pub(super) fn bounding_box_l(pixels: &[u8], width: usize) -> Option<(usize, usize, usize, usize)> {
    bounding_box_packed::<1>(pixels, width, false)
}

pub(super) fn bounding_box_packed<const C: usize>(pixels: &[u8], width: usize, alpha: bool) -> Option<(usize, usize, usize, usize)> {
    let pixels = pixels.as_chunks::<C>().0;
    let rows = pixels.chunks_exact(width);
    let occupied = |pixel: &[u8; C]| {
        if alpha { pixel[C - 1] != 0 } else { *pixel != [0; C] }
    };
    let edge = |row: &[[u8; C]], reverse: bool| {
        use std::simd::cmp::SimdPartialEq;

        if reverse && row.last().is_some_and(occupied) {
            return Some(row.len() - 1);
        }
        if !reverse && row.first().is_some_and(occupied) {
            return Some(0);
        }
        let (blocks, tail) = row.as_chunks::<16>();
        let block_occupied = |block: &[[u8; C]; 16]| {
            let bytes = block.as_flattened();
            if alpha {
                crate::pixels::load::<C>(bytes)[C - 1].simd_ne(crate::pixels::Bytes::splat(0)).any()
            } else {
                bytes.as_chunks::<16>().0.iter().any(|&chunk| crate::pixels::Bytes::from_array(chunk).simd_ne(crate::pixels::Bytes::splat(0)).any())
            }
        };
        let offset = blocks.len() * 16;
        if reverse {
            tail.iter().rposition(occupied).map(|i| offset + i).or_else(|| {
                let i = blocks.iter().rposition(block_occupied)?;
                blocks[i].iter().rposition(occupied).map(|x| i * 16 + x)
            })
        } else {
            blocks
                .iter()
                .position(block_occupied)
                .and_then(|i| blocks[i].iter().position(occupied).map(|x| i * 16 + x))
                .or_else(|| tail.iter().position(occupied).map(|i| offset + i))
        }
    };
    let (top, mut left) = rows.clone().enumerate().find_map(|(y, row)| edge(row, false).map(|x| (y, x)))?;
    let bottom = rows.clone().rposition(|row| edge(row, true).is_some()).unwrap();
    let mut right = edge(&pixels[top * width..(top + 1) * width], true).unwrap() + 1;
    for row in rows.skip(top + 1).take(bottom - top) {
        if left == 0 && right == width {
            break;
        }
        if let Some(x) = edge(&row[..left], false) {
            left = x;
        }
        if let Some(x) = edge(&row[right..], true) {
            right += x + 1;
        }
    }
    Some((left, top, right, bottom + 1))
}

pub(super) fn extrema<const C: usize>(data: &[u8]) -> Vec<(u16, u16)> {
    use std::simd::Simd;
    use std::simd::cmp::SimdOrd;
    use std::simd::num::SimdUint;

    use crate::pixels::{Bytes, load};

    let mut minimum = [255_u8; C];
    let mut maximum = [0_u8; C];
    for block in data.as_chunks::<C>().0.chunks(256) {
        // The scalar RGB loop already vectorizes efficiently.
        let done = if C == 3 {
            0
        } else {
            let bytes = block.as_flattened();
            let done = bytes.len() / 64 * 64;
            let mut low = Simd::<u8, 64>::splat(255);
            let mut high = Simd::<u8, 64>::splat(0);
            for values in bytes[..done].as_chunks::<64>().0 {
                let values = Simd::from_array(*values);
                low = low.simd_min(values);
                high = high.simd_max(values);
            }
            if C == 1 {
                minimum[0] = minimum[0].min(low.reduce_min());
                maximum[0] = maximum[0].max(high.reduce_max());
            } else {
                let low = low.to_array();
                let high = high.to_array();
                let mut low_channels = minimum.map(Bytes::splat);
                let mut high_channels = maximum.map(Bytes::splat);
                for i in (0..64).step_by(16 * C) {
                    let low = load::<C>(&low[i..]);
                    let high = load::<C>(&high[i..]);
                    for c in 0..C {
                        low_channels[c] = low_channels[c].simd_min(low[c]);
                        high_channels[c] = high_channels[c].simd_max(high[c]);
                    }
                }
                minimum = low_channels.map(SimdUint::reduce_min);
                maximum = high_channels.map(SimdUint::reduce_max);
            }
            done / C
        };
        for pixel in &block[done..] {
            for c in 0..C {
                minimum[c] = minimum[c].min(pixel[c]);
                maximum[c] = maximum[c].max(pixel[c]);
            }
        }
        if minimum == [0; C] && maximum == [255; C] {
            break;
        }
    }
    minimum.into_iter().zip(maximum).map(|(a, b)| (u16::from(a), u16::from(b))).collect()
}

pub(super) fn sample_extrema<const C: usize>(data: &[u8], maximum: u16) -> Vec<(u16, u16)> {
    use std::simd::cmp::SimdOrd;
    use std::simd::num::SimdUint;

    use crate::pixels::{Samples, load_sample_bytes};

    let mut minimum = [u16::MAX; C];
    let mut high = [0; C];
    for block in data.chunks(256 * C * 2) {
        let done = block.len() / (32 * C) * (32 * C);
        let mut low_vectors = minimum.map(Samples::splat);
        let mut high_vectors = high.map(Samples::splat);
        for offset in (0..done).step_by(32 * C) {
            let values = load_sample_bytes::<C>(&block[offset..]);
            for c in 0..C {
                low_vectors[c] = low_vectors[c].simd_min(values[c]);
                high_vectors[c] = high_vectors[c].simd_max(values[c]);
            }
        }
        minimum = low_vectors.map(SimdUint::reduce_min);
        high = high_vectors.map(SimdUint::reduce_max);
        for pixel in block[done..].chunks_exact(C * 2) {
            for c in 0..C {
                let value = u16::from_le_bytes([pixel[c * 2], pixel[c * 2 + 1]]);
                minimum[c] = minimum[c].min(value);
                high[c] = high[c].max(value);
            }
        }
        if minimum == [0; C] && high == [maximum; C] {
            break;
        }
    }
    minimum.into_iter().zip(high).collect()
}

pub(super) fn integer_extrema(data: &[u8], mode: super::PixelMode) -> (i64, i64) {
    use std::simd::cmp::SimdOrd;
    use std::simd::num::{SimdInt, SimdUint};
    use std::simd::{Simd, ToBytes};

    let (lo, hi, tail) = if mode == super::PixelMode::I {
        let mut lo = Simd::<i32, 16>::splat(i32::MAX);
        let mut hi = Simd::<i32, 16>::splat(i32::MIN);
        let (blocks, tail) = data.as_chunks::<64>();
        for &block in blocks {
            let values = Simd::from_le_bytes(Simd::from_array(block));
            lo = lo.simd_min(values);
            hi = hi.simd_max(values);
        }
        (i64::from(lo.reduce_min()), i64::from(hi.reduce_max()), tail)
    } else {
        let mut lo = crate::pixels::Samples::splat(u16::MAX);
        let mut hi = crate::pixels::Samples::splat(u16::MIN);
        let (blocks, tail) = data.as_chunks::<32>();
        let big_endian = mode.is_big_endian();
        for block in blocks {
            let values = if big_endian { crate::pixels::load_u16::<true>(block) } else { crate::pixels::load_u16::<false>(block) };
            lo = lo.simd_min(values);
            hi = hi.simd_max(values);
        }
        (i64::from(lo.reduce_min()), i64::from(hi.reduce_max()), tail)
    };
    tail.chunks_exact(mode.sample_bytes()).fold((lo, hi), |(lo, hi), bytes| {
        let value = super::pixels::integer_value(mode, bytes);
        (lo.min(value), hi.max(value))
    })
}

pub(super) fn count_colors<const N: usize>(data: &[u8], maxcolors: usize) -> Option<ColorCounts> {
    const CHUNK: usize = 128 << 10;
    if maxcolors >= 64 && crate::parallel::should_parallel(data.len() / N, CHUNK, CHUNK * 2) {
        use rayon::prelude::*;
        let tables: Option<Vec<_>> = data.par_chunks(CHUNK * N).map(|chunk| count_colors_serial::<N>(chunk, maxcolors)).collect();
        let mut merged = [(0, 0, 0); COLOR_TABLE_SIZE];
        let mut used = 0;
        for (chunk, table) in tables?.into_iter().enumerate() {
            for (key, count, index) in table.into_iter().filter(|entry| entry.1 != 0) {
                let mut slot = (key.wrapping_mul(COLOR_HASH_MULTIPLIER) >> (u64::BITS - COLOR_TABLE_BITS)) as usize;
                loop {
                    let entry = &mut merged[slot];
                    if entry.1 == 0 {
                        used += 1;
                        if used > maxcolors {
                            return None;
                        }
                        *entry = (key, count, chunk * CHUNK + index);
                        break;
                    }
                    if entry.0 == key {
                        entry.1 += count;
                        break;
                    }
                    slot = (slot + 1) & (COLOR_TABLE_SIZE - 1);
                }
            }
        }
        return Some(merged);
    }
    count_colors_serial::<N>(data, maxcolors)
}

/// Unbounded counts keyed by packed pixel bytes, as (count, first index).
pub(super) fn count_colors_map<const N: usize>(data: &[u8], maxcolors: usize) -> Option<Vec<(usize, usize)>> {
    const CHUNK: usize = 128 << 10;
    if crate::parallel::should_parallel(data.len() / N, CHUNK, CHUNK * 2) {
        use rayon::prelude::*;
        let maps: Option<Vec<_>> = data.par_chunks(CHUNK * N).map(|chunk| count_colors_map_serial::<N>(chunk, maxcolors)).collect();
        let mut maps = maps?.into_iter();
        let mut merged = maps.next()?;
        for (chunk, map) in maps.enumerate() {
            for (key, (count, index)) in map {
                let entry = merged.entry(key).or_insert((0, (chunk + 1) * CHUNK + index));
                entry.0 += count;
            }
            if merged.len() > maxcolors {
                return None;
            }
        }
        return Some(merged.into_values().collect());
    }
    Some(count_colors_map_serial::<N>(data, maxcolors)?.into_values().collect())
}

fn count_colors_map_serial<const N: usize>(data: &[u8], maxcolors: usize) -> Option<rustc_hash::FxHashMap<u64, (usize, usize)>> {
    let mut counts = rustc_hash::FxHashMap::<u64, (usize, usize)>::default();
    counts.reserve(maxcolors.min(data.len() / N).min(1 << 16));
    for (i, pixel) in data.as_chunks::<N>().0.iter().enumerate() {
        let mut bytes = [0; 8];
        bytes[..N].copy_from_slice(pixel);
        let entry = counts.entry(u64::from_le_bytes(bytes)).or_insert((0, i));
        entry.0 += 1;
        if entry.0 == 1 && counts.len() > maxcolors {
            return None;
        }
    }
    Some(counts)
}

pub(super) fn count_colors_serial<const N: usize>(data: &[u8], maxcolors: usize) -> Option<ColorCounts> {
    let mut table = [(0_u64, 0_usize, 0_usize); COLOR_TABLE_SIZE];
    let mut used = 0;
    for (i, pixel) in data.as_chunks::<N>().0.iter().enumerate() {
        let mut bytes = [0; 8];
        bytes[..N].copy_from_slice(pixel);
        let key = u64::from_le_bytes(bytes);
        let mut slot = (key.wrapping_mul(COLOR_HASH_MULTIPLIER) >> (u64::BITS - COLOR_TABLE_BITS)) as usize;
        loop {
            let entry = &mut table[slot];
            if entry.1 == 0 {
                used += 1;
                if used > maxcolors {
                    return None;
                }
                *entry = (key, 1, i);
                break;
            }
            if entry.0 == key {
                entry.1 += 1;
                break;
            }
            slot = (slot + 1) & (COLOR_TABLE_SIZE - 1);
        }
    }
    Some(table)
}
