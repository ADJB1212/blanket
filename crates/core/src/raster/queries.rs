const COLOR_HASH_MULTIPLIER: u64 = 0x9e37_79b9_7f4a_7c15;
// Twice the 256-color palette limit keeps linear probing at most half full.
const COLOR_TABLE_BITS: u32 = 9;
const COLOR_TABLE_SIZE: usize = 1 << COLOR_TABLE_BITS;

pub(super) type ColorCounts = [(u64, usize, usize); COLOR_TABLE_SIZE];

pub(super) fn bounding_box_l(pixels: &[u8], width: usize) -> Option<(usize, usize, usize, usize)> {
    let top = pixels.chunks_exact(width).position(|row| row.iter().any(|&v| v != 0))?;
    let bottom = pixels.chunks_exact(width).rposition(|row| row.iter().any(|&v| v != 0)).unwrap();
    let first = &pixels[top * width..(top + 1) * width];
    let mut left = first.iter().position(|&v| v != 0).unwrap();
    let mut right = first.iter().rposition(|&v| v != 0).unwrap() + 1;
    let remaining = &pixels[(top + 1) * width..(bottom + 1) * width];
    if left > 8 || width - right > 8 {
        // Wide empty margins are cheaper to scan contiguously by row.
        for row in remaining.chunks_exact(width) {
            if let Some(x) = row[..left].iter().position(|&v| v != 0) {
                left = x;
            }
            if let Some(x) = row[right..].iter().rposition(|&v| v != 0) {
                right += x + 1;
            }
            if left == 0 && right == width {
                break;
            }
        }
        return Some((left, top, right, bottom + 1));
    }
    // Search only columns outside the first occupied row's bounds. Separate
    // edges avoid revisiting an already complete right edge on every row.
    for x in 0..left {
        if remaining.chunks_exact(width).any(|row| row[x] != 0) {
            left = x;
            break;
        }
    }
    for x in (right..width).rev() {
        if remaining.chunks_exact(width).any(|row| row[x] != 0) {
            right = x + 1;
            break;
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
