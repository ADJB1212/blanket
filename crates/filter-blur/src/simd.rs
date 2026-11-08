use std::simd::Simd;
use std::simd::num::SimdUint;

use crate::{BOX_WEIGHT_BITS, BOX_WEIGHT_ROUND, box_value};

type Bytes = Simd<u8, 16>;

pub(crate) fn box_column<const R: usize>(source: &[u8], output: &mut [u8], y: usize, weight: u32, far_weight: u32) {
    let stride = output.len();
    let height = source.len() / stride;
    let row = |tap: usize| {
        let start = (y + tap).saturating_sub(R + 1).min(height - 1) * stride;
        &source[start..start + stride]
    };
    let upper = row(0);
    let lower = row(2 * R + 2);
    let end = stride / 16 * 16;
    for x in (0..end).step_by(16) {
        let mut sum = Simd::<u16, 16>::splat(0);
        for tap in 1..=2 * R + 1 {
            sum += Bytes::from_slice(&row(tap)[x..]).cast::<u16>();
        }
        let ends = Bytes::from_slice(&upper[x..]).cast::<u16>() + Bytes::from_slice(&lower[x..]).cast::<u16>();
        let value = sum.cast::<u32>() * Simd::splat(weight) + ends.cast::<u32>() * Simd::splat(far_weight) + Simd::splat(BOX_WEIGHT_ROUND as u32);
        (value >> BOX_WEIGHT_BITS).cast::<u8>().copy_to_slice(&mut output[x..x + 16]);
    }
    for x in end..stride {
        let sum = (1..=2 * R + 1).map(|tap| u32::from(row(tap)[x])).sum();
        output[x] = box_value(sum, u32::from(upper[x]) + u32::from(lower[x]), weight, far_weight);
    }
}

pub(crate) fn box_row<const C: usize, const R: usize>(source: &[u8], output: &mut [u8], weight: u32, far_weight: u32) {
    let width = source.len() / C;
    let first = ((R + 1) * C).min(source.len());
    let end = first + width.saturating_sub(2 * (R + 1)) * C / 16 * 16;
    for x in (first..end).step_by(16) {
        let mut sum = Simd::<u32, 16>::splat(0);
        for tap in 0..2 * R + 1 {
            let start = x + tap * C - R * C;
            sum += Bytes::from_slice(&source[start..start + 16]).cast::<u32>();
        }
        let left = Bytes::from_slice(&source[x - (R + 1) * C..][..16]).cast::<u32>();
        let right = Bytes::from_slice(&source[x + (R + 1) * C..][..16]).cast::<u32>();
        let result = (sum * Simd::splat(weight) + (left + right) * Simd::splat(far_weight) + Simd::splat(BOX_WEIGHT_ROUND as u32)) >> BOX_WEIGHT_BITS;
        result.cast::<u8>().copy_to_slice(&mut output[x..x + 16]);
    }
    for x in (0..first).chain(end..source.len()) {
        let sample = |tap: usize| u32::from(source[(x / C + tap).saturating_sub(R + 1).min(width - 1) * C + x % C]);
        let sum: u32 = (1..=2 * R + 1).map(sample).sum();
        output[x] = box_value(sum, sample(0) + sample(2 * R + 2), weight, far_weight);
    }
}
