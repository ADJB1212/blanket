//! Portable SIMD kernels with scalar tails.
use std::simd::Simd;
use std::simd::cmp::{SimdOrd, SimdPartialEq};
use std::simd::num::{SimdFloat, SimdInt, SimdUint};

use crate::pixels::{self, Bytes, load, store};

// Fixed-point resampling coefficients shared by scalar and vector passes.
pub const RESAMPLE_BITS: u32 = 22;
pub const RESAMPLE_SCALE: i32 = 1 << RESAMPLE_BITS;
pub const RESAMPLE_ROUND: i32 = RESAMPLE_SCALE / 2;

#[cfg(test)]
mod float_resize_tests {
    use super::*;

    #[test]
    fn float_passes_preserve_order_and_guards() {
        let weights = [123456, -654321, 4725169];
        let samples = [-2.0_f32, 0.0, 0.25, 1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY];
        for width in 1..19 {
            for offset in 0..4 {
                let source: Vec<f64> = (0..width * 3 + offset).map(|i| f64::from(samples[i % samples.len()])).collect();
                let source = &source[offset..];
                let mut output = vec![123; offset + width * 4 + 4];
                vertical_float(source, &mut output[offset..offset + width * 4], width, &weights);
                for x in 0..width {
                    let expected = weights
                        .iter()
                        .enumerate()
                        .map(|(i, &w)| source[i * width + x] * f64::from(w) / f64::from(crate::ops::RESAMPLE_SCALE))
                        .sum::<f64>() as f32;
                    let actual = f32::from_le_bytes(output[offset + x * 4..offset + x * 4 + 4].try_into().unwrap());
                    assert!(actual.to_bits() == expected.to_bits() || actual.is_nan() && expected.is_nan());
                }
                assert!(output[..offset].iter().chain(&output[offset + width * 4..]).all(|&v| v == 123));
                let bytes: Vec<_> = (0..12).flat_map(|i| samples[(i + width) % samples.len()].to_le_bytes()).collect();
                let rows = std::array::from_fn(|i| &bytes[i * 12..i * 12 + 12]);
                let actual = horizontal_float_four(rows, &weights);
                for r in 0..4 {
                    let expected: f64 = weights
                        .iter()
                        .enumerate()
                        .map(|(i, &w)| {
                            f64::from(f32::from_le_bytes(rows[r][i * 4..i * 4 + 4].try_into().unwrap())) * f64::from(w)
                                / f64::from(crate::ops::RESAMPLE_SCALE)
                        })
                        .sum();
                    assert!(actual[r].to_bits() == expected.to_bits() || actual[r].is_nan() && expected.is_nan());
                }
            }
        }
    }
}

#[inline]
pub fn sample_color<const C: usize>(source: &[u8], output: &mut [u8], size: (usize, usize), position: (i64, i64), fraction: (f64, f64), method: u8) {
    let x = |dx: i64| (position.0 + dx).clamp(0, size.0 as i64 - 1) as usize * C;
    let y = |dy: i64| (position.1 + dy).clamp(0, size.1 as i64 - 1) as usize * size.0 * C;
    let get = |x: usize, y: usize| -> [u8; C] { source[x + y..x + y + C].try_into().unwrap() };
    let interpolate = |a: f64, b: f64, t: f64| {
        if cfg!(target_arch = "aarch64") { (b - a).mul_add(t, a) } else { a + (b - a) * t }
    };
    let cubic = |v: [f64; 4], t: f64| {
        let a = -v[0] + v[1] - v[2] + v[3];
        let b = 2.0 * v[0] - 2.0 * v[1] + v[2] - v[3];
        let c = -v[0] + v[2];
        ((a * t + b) * t + c) * t + v[1]
    };
    if method == 2 {
        let [x0, x1] = [0, 1].map(x);
        let [y0, y1] = [0, 1].map(y);
        let [a, b, c, d] = [get(x0, y0), get(x1, y0), get(x0, y1), get(x1, y1)];
        for channel in 0..C {
            let top = interpolate(f64::from(a[channel]), f64::from(b[channel]), fraction.0);
            let bottom = interpolate(f64::from(c[channel]), f64::from(d[channel]), fraction.0);
            output[channel] = interpolate(top, bottom, fraction.1).clamp(0.0, 255.0) as u8;
        }
    } else {
        let xs = [-1, 0, 1, 2].map(x);
        let rows: [[f64; C]; 4] = [-1, 0, 1, 2].map(y).map(|y| {
            let pixels = xs.map(|x| get(x, y));
            std::array::from_fn(|channel| cubic(pixels.map(|pixel| f64::from(pixel[channel])), fraction.0))
        });
        for channel in 0..C {
            output[channel] = cubic(rows.map(|row| row[channel]), fraction.1).clamp(0.0, 255.0) as u8;
        }
    }
}

pub fn constant_pixel<const C: usize>(source: &[u8]) -> Option<[u8; C]> {
    let pixels = source.as_chunks::<C>().0;
    let first = *pixels.first()?;
    let (blocks, tail) = pixels.as_chunks::<16>();
    for block in blocks {
        let values = load::<C>(block.as_flattened());
        if (0..C).any(|c| values[c].simd_ne(Bytes::splat(first[c])).any()) {
            return None;
        }
    }
    tail.iter().all(|pixel| *pixel == first).then_some(first)
}

pub fn box_half<const C: usize>(upper: &[u8], lower: &[u8], output: &mut [u8]) {
    let end = output.len() / (16 * C) * (16 * C);
    for i in (0..end).step_by(16 * C) {
        let horizontal = |row: &[u8]| {
            let first = load::<C>(&row[i * 2..]);
            let second = load::<C>(&row[i * 2 + 16 * C..]);
            std::array::from_fn::<_, C, _>(|c| {
                let (even, odd) = first[c].deinterleave(second[c]);
                (even.cast::<u16>() + odd.cast::<u16>() + Simd::splat(1)) >> 1
            })
        };
        let upper = horizontal(upper);
        let lower = horizontal(lower);
        store::<C>(&mut output[i..], std::array::from_fn(|c| ((upper[c] + lower[c] + Simd::splat(1)) >> 1).cast::<u8>()));
    }
    for (x, dst) in output[end..].iter_mut().enumerate() {
        let x = end + x;
        let start = x / C * 2 * C + x % C;
        let average = |row: &[u8]| (u16::from(row[start]) + u16::from(row[start + C]) + 1) >> 1;
        *dst = ((average(upper) + average(lower) + 1) >> 1) as u8;
    }
}

pub fn reverse_gray(source: &[u8], output: &mut [std::mem::MaybeUninit<u8>]) {
    let count = output.len();
    assert_eq!(source.len(), count);
    let (blocks, tail) = output.as_chunks_mut::<64>();
    for (i, block) in blocks.iter_mut().enumerate() {
        *block = Simd::<u8, 64>::from_slice(&source[count - (i + 1) * 64..count - i * 64]).reverse().to_array().map(std::mem::MaybeUninit::new);
    }
    let length = tail.len();
    for (dst, &src) in tail.iter_mut().zip(source[..length].iter().rev()) {
        dst.write(src);
    }
}

#[inline]
pub fn transpose_rgb_four(rows: [&[u8]; 4]) -> [[u8; 12]; 4] {
    use std::simd::simd_swizzle;

    let [a, b, c, d] = rows.map(|row| Bytes::from_array(std::array::from_fn(|i| if i < 12 { row[i] } else { 0 })));
    let ab0 = simd_swizzle!(a, b, [0, 1, 2, 16, 17, 18, 3, 4, 5, 19, 20, 21, 0, 0, 0, 0]);
    let ab1 = simd_swizzle!(a, b, [6, 7, 8, 22, 23, 24, 9, 10, 11, 25, 26, 27, 0, 0, 0, 0]);
    let cd0 = simd_swizzle!(c, d, [0, 1, 2, 16, 17, 18, 3, 4, 5, 19, 20, 21, 0, 0, 0, 0]);
    let cd1 = simd_swizzle!(c, d, [6, 7, 8, 22, 23, 24, 9, 10, 11, 25, 26, 27, 0, 0, 0, 0]);
    [
        simd_swizzle!(ab0, cd0, [0, 1, 2, 3, 4, 5, 16, 17, 18, 19, 20, 21, 0, 0, 0, 0]),
        simd_swizzle!(ab0, cd0, [6, 7, 8, 9, 10, 11, 22, 23, 24, 25, 26, 27, 0, 0, 0, 0]),
        simd_swizzle!(ab1, cd1, [0, 1, 2, 3, 4, 5, 16, 17, 18, 19, 20, 21, 0, 0, 0, 0]),
        simd_swizzle!(ab1, cd1, [6, 7, 8, 9, 10, 11, 22, 23, 24, 25, 26, 27, 0, 0, 0, 0]),
    ]
    .map(|v| v.to_array()[..12].try_into().unwrap())
}

pub fn reduce_float_two(upper: &[u8], lower: &[u8], output: &mut [u8]) -> usize {
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

/// Reusable byte gather map for an axis-aligned nearest transform.
pub struct NearestL {
    columns: Vec<usize>,
    blocks: Vec<Option<(usize, Bytes)>>,
    width: usize,
}

impl NearestL {
    pub fn new(columns: &[Option<usize>], width: usize) -> Option<Self> {
        let columns: Vec<usize> = columns.iter().copied().collect::<Option<_>>()?;
        if columns.iter().any(|&x| x >= width) {
            return None;
        }
        let blocks = columns
            .as_chunks::<16>()
            .0
            .iter()
            .map(|columns| {
                let start = (*columns.iter().min().unwrap()).min(width.saturating_sub(32));
                (width >= 32 && columns.iter().all(|&x| x - start < 32))
                    .then(|| (start, Bytes::from_array(std::array::from_fn(|i| (columns[i] - start) as u8))))
            })
            .collect();
        Some(Self { columns, blocks, width })
    }

    pub fn is_vectorized(&self) -> bool {
        self.columns.len() >= 16
    }

    pub fn sample(&self, source: &[u8], output: &mut [u8]) {
        assert_eq!(source.len(), self.width);
        assert_eq!(output.len(), self.columns.len());
        let done = output.len() / 16 * 16;
        for (i, (block, dst)) in self.blocks.iter().zip(output[..done].as_chunks_mut::<16>().0).enumerate() {
            let values = if let Some((start, indices)) = block {
                let low = Bytes::from_slice(&source[*start..*start + 16]);
                let high = Bytes::from_slice(&source[*start + 16..*start + 32]);
                low.swizzle_dyn(*indices) | high.swizzle_dyn(*indices - Bytes::splat(16))
            } else {
                let i = i * 16;
                Bytes::gather_or_default(source, Simd::from_slice(&self.columns[i..i + 16]))
            };
            values.copy_to_slice(dst);
        }
        for (dst, &x) in output[done..].iter_mut().zip(&self.columns[done..]) {
            *dst = source[x];
        }
    }
}

pub fn reverse_rgb(source: &[u8], output: &mut [u8]) {
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
    assert!(tables.len() == 256 || tables.len() == C << 8);
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

pub fn colorize(source: &[u8], output: &mut [u8], tables: &[u8]) {
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
        store(&mut output[i * 3..], std::array::from_fn::<_, 3, _>(|c| Bytes::gather_or_default(&tables[c * 256..(c + 1) * 256], indices)));
    }
    end
}

/// Returns the byte count processed without overflowing fixed-point accumulators.
pub fn vertical(source: &[u8], output: &mut [u8], stride: usize, weights: &[i32]) -> usize {
    let magnitude: i64 = weights.iter().map(|&w| i64::from(w).abs()).sum();
    if magnitude * 255 + i64::from(RESAMPLE_ROUND) > i64::from(i32::MAX) {
        return 0;
    }
    assert!(weights.is_empty() || source.len() >= (weights.len() - 1).saturating_mul(stride).saturating_add(output.len()));
    let end = output.len() / 16 * 16;
    for x in (0..end).step_by(16) {
        let mut acc = Simd::<i32, 16>::splat(RESAMPLE_ROUND);
        for (i, &weight) in weights.iter().enumerate() {
            let values = Bytes::from_slice(&source[i * stride + x..i * stride + x + 16]);
            acc += values.cast::<i32>() * Simd::splat(weight);
        }
        (acc >> RESAMPLE_BITS as i32).simd_clamp(Simd::splat(0), Simd::splat(255)).cast::<u8>().copy_to_slice(&mut output[x..x + 16]);
    }
    end
}

pub fn horizontal_half<const C: usize>(source: &[u8], output: &mut [u8], weights: &[i32]) {
    let mut sums = [Simd::<i32, 16>::splat(RESAMPLE_ROUND); C];
    for (pair, weights) in weights.chunks(2).enumerate() {
        let tap = pair * 2;
        let first = load::<C>(&source[tap * C..]);
        let second = load::<C>(&source[(tap + 16) * C..]);
        for c in 0..C {
            let (even, odd) = first[c].deinterleave(second[c]);
            sums[c] += even.cast::<i32>() * Simd::splat(weights[0]);
            if let Some(&weight) = weights.get(1) {
                sums[c] += odd.cast::<i32>() * Simd::splat(weight);
            }
        }
    }
    store::<C>(output, sums.map(|sum| (sum >> RESAMPLE_BITS as i32).simd_clamp(Simd::splat(0), Simd::splat(255)).cast::<u8>()));
}

#[inline]
pub fn horizontal_four<const C: usize>(source: [&[[u8; C]]; 4], weights: &[i32]) -> [[u8; C]; 4] {
    let mut sums = [Simd::<i32, 4>::splat(RESAMPLE_ROUND); C];
    for (i, &weight) in weights.iter().enumerate() {
        for c in 0..C {
            let values = Simd::<i32, 4>::from_array(std::array::from_fn(|row| i32::from(source[row][i][c])));
            sums[c] += values * Simd::splat(weight);
        }
    }
    let values = sums.map(|sum| (sum >> RESAMPLE_BITS as i32).simd_clamp(Simd::splat(0), Simd::splat(255)).cast::<u8>().to_array());
    std::array::from_fn(|row| std::array::from_fn(|c| values[c][row]))
}

pub fn vertical_wide(source: &[f64], output: &mut [f64], stride: usize, weights: &[i32], maximum: f64) -> usize {
    let end = output.len() / 4 * 4;
    for x in (0..end).step_by(4) {
        let mut sum = Simd::<f64, 4>::splat(0.0);
        for (i, &weight) in weights.iter().enumerate() {
            let values = Simd::<f64, 4>::from_slice(&source[i * stride + x..]);
            sum += values * Simd::splat(f64::from(weight)) / Simd::splat(f64::from(crate::ops::RESAMPLE_SCALE));
        }
        sum.simd_clamp(Simd::splat(0.0), Simd::splat(maximum)).copy_to_slice(&mut output[x..x + 4]);
    }
    end
}

#[inline]
pub fn horizontal_float_four(source: [&[u8]; 4], weights: &[i32]) -> [f64; 4] {
    let mut sum = Simd::<f64, 4>::splat(0.0);
    for (i, &weight) in weights.iter().enumerate() {
        let values = Simd::from_array(source.map(|row| f64::from(f32::from_le_bytes(row[i * 4..i * 4 + 4].try_into().unwrap()))));
        sum += values * Simd::splat(f64::from(weight)) / Simd::splat(f64::from(crate::ops::RESAMPLE_SCALE));
    }
    sum.to_array()
}

pub fn vertical_float(source: &[f64], output: &mut [u8], stride: usize, weights: &[i32]) {
    let count = output.len() / 4;
    let end = count / 4 * 4;
    for x in (0..end).step_by(4) {
        let mut sum = Simd::<f64, 4>::splat(0.0);
        for (i, &weight) in weights.iter().enumerate() {
            sum += Simd::from_slice(&source[i * stride + x..]) * Simd::splat(f64::from(weight)) / Simd::splat(f64::from(crate::ops::RESAMPLE_SCALE));
        }
        for (c, v) in sum.cast::<f32>().to_array().iter().enumerate() {
            output[(x + c) * 4..(x + c + 1) * 4].copy_from_slice(&v.to_le_bytes());
        }
    }
    for x in end..count {
        let sum: f64 = weights.iter().enumerate().map(|(i, &w)| source[i * stride + x] * f64::from(w) / f64::from(crate::ops::RESAMPLE_SCALE)).sum();
        output[x * 4..x * 4 + 4].copy_from_slice(&(sum as f32).to_le_bytes());
    }
}

pub fn horizontal_wide_four(source: [&[u8]; 4], weights: &[i32]) -> [f64; 4] {
    let mut sum = Simd::<f64, 4>::splat(0.0);
    for (i, &weight) in weights.iter().enumerate() {
        let values = Simd::<u16, 4>::from_array(source.map(|row| u16::from_le_bytes([row[i * 2], row[i * 2 + 1]]))).cast::<f64>();
        sum += values * Simd::splat(f64::from(weight)) / Simd::splat(f64::from(crate::ops::RESAMPLE_SCALE));
    }
    sum.to_array()
}

#[inline]
pub fn horizontal_wide<const C: usize>(source: &[u8], weights: &[i32], maximum: f64) -> [f64; C] {
    let mut sum = Simd::<f64, 4>::splat(0.0);
    for (i, &weight) in weights.iter().enumerate() {
        let pixel = &source[i * C * 2..][..C * 2];
        let samples =
            Simd::<u16, 4>::from_array(std::array::from_fn(|c| if c < C { u16::from_le_bytes([pixel[c * 2], pixel[c * 2 + 1]]) } else { 0 }));
        let mut values = samples.cast::<f64>();
        if C == 4 {
            let alpha = values[3];
            values = values * Simd::splat(alpha) / Simd::splat(maximum);
            values.as_mut_array()[3] = alpha;
        }
        sum += values * Simd::splat(f64::from(weight)) / Simd::splat(f64::from(crate::ops::RESAMPLE_SCALE));
    }
    std::array::from_fn(|c| sum[c])
}

pub fn nearest_half<const C: usize>(source: &[u8], output: &mut [u8]) {
    assert_eq!(source.len(), output.len() * 2);
    let done = output.len() / C / 16 * 16 * C;
    for i in (0..done).step_by(16 * C) {
        let first = load::<C>(&source[i * 2..]);
        let second = load::<C>(&source[i * 2 + 16 * C..]);
        let values = std::array::from_fn::<_, C, _>(|c| first[c].deinterleave(second[c]).1);
        store(&mut output[i..], values);
    }
    for (dst, pair) in output[done..].as_chunks_mut::<C>().0.iter_mut().zip(source[done * 2..].as_chunks::<C>().0.as_chunks::<2>().0) {
        *dst = pair[1];
    }
}

pub fn nearest_double_rgb(source: &[u8], output: &mut [u8]) {
    assert_eq!(output.len(), source.len() * 2);
    let done = source.len() / 48 * 48;
    for i in (0..done).step_by(48) {
        let [r, g, b] = load::<3>(&source[i..]);
        let (r0, r1) = r.interleave(r);
        let (g0, g1) = g.interleave(g);
        let (b0, b1) = b.interleave(b);
        store(&mut output[i * 2..], [r0, g0, b0]);
        store(&mut output[i * 2 + 48..], [r1, g1, b1]);
    }
    for (src, dst) in source[done..].as_chunks::<3>().0.iter().zip(output[done * 2..].as_chunks_mut::<6>().0) {
        dst[..3].copy_from_slice(src);
        dst[3..].copy_from_slice(src);
    }
}

pub fn reduce_three_l(upper: &[u8], middle: &[u8], lower: &[u8], output: &mut [u8]) -> usize {
    assert_eq!(upper.len(), output.len() * 3);
    assert_eq!(middle.len(), upper.len());
    assert_eq!(lower.len(), upper.len());
    pixels::reduce_three_l([upper, middle, lower], output)
}

pub fn reduce_two_l(upper: &[u8], lower: &[u8], output: &mut [u8]) -> usize {
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
    fn color_sampling_matches_scalar_edges_rounding_and_guards() {
        fn check<const C: usize>() {
            let interpolate = |a: f64, b: f64, t: f64| {
                if cfg!(target_arch = "aarch64") { (b - a).mul_add(t, a) } else { a + (b - a) * t }
            };
            let cubic = |v: [f64; 4], t: f64| {
                let a = -v[0] + v[1] - v[2] + v[3];
                let b = 2.0 * v[0] - 2.0 * v[1] + v[2] - v[3];
                let c = -v[0] + v[2];
                ((a * t + b) * t + c) * t + v[1]
            };
            for (width, height) in [(1, 1), (1, 7), (7, 1), (5, 3)] {
                let backing: Vec<_> = (0..width * height * C + 1).map(|i| (i * 137 + i / 7 * 53) as u8).collect();
                let source = &backing[1..];
                for y in -1..height as i64 {
                    for x in -1..width as i64 {
                        for tx in [0.0, 0.1, 0.5, 0.9999999999999999] {
                            for ty in [0.0, 0.3, 0.5, 0.9999999999999999] {
                                for method in [2, 3] {
                                    let mut output = vec![173; C + 2];
                                    sample_color::<C>(source, &mut output[1..C + 1], (width, height), (x, y), (tx, ty), method);
                                    for (channel, &actual) in output[1..C + 1].iter().enumerate() {
                                        let get = |dx: i64, dy: i64| {
                                            let xx = (x + dx).clamp(0, width as i64 - 1) as usize;
                                            let yy = (y + dy).clamp(0, height as i64 - 1) as usize;
                                            f64::from(source[(yy * width + xx) * C + channel])
                                        };
                                        let expected = if method == 2 {
                                            interpolate(interpolate(get(0, 0), get(1, 0), tx), interpolate(get(0, 1), get(1, 1), tx), ty)
                                        } else {
                                            cubic([-1, 0, 1, 2].map(|dy| cubic([-1, 0, 1, 2].map(|dx| get(dx, dy)), tx)), ty)
                                        };
                                        assert_eq!(actual, expected.clamp(0.0, 255.0) as u8);
                                    }
                                    assert_eq!((output[0], output[C + 1]), (173, 173));
                                }
                            }
                        }
                    }
                }
            }
        }
        check::<2>();
        check::<3>();
        check::<4>();
    }

    #[test]
    fn constant_pixels_check_every_channel_lane_unaligned_tail_and_guard() {
        fn check<const C: usize>() {
            let pixel: [u8; C] = std::array::from_fn(|c| (c * 71 + 3) as u8);
            for count in [0, 1, 15, 16, 17, 31, 32, 33, 65537] {
                let mut source = vec![91];
                source.extend(std::iter::repeat_n(pixel, count).flatten());
                source.push(93);
                assert_eq!(constant_pixel::<C>(&source[1..source.len() - 1]), (count != 0).then_some(pixel));
                for i in (0..count.min(33)).chain((count > 33).then(|| count - 1)) {
                    for c in 0..C {
                        let index = 1 + i * C + c;
                        source[index] ^= 1;
                        assert!(constant_pixel::<C>(&source[1..source.len() - 1]).is_none() || count == 1);
                        source[index] ^= 1;
                    }
                }
                assert_eq!(source[0], 91);
                assert_eq!(source[source.len() - 1], 93);
            }
        }
        check::<1>();
        check::<2>();
        check::<3>();
        check::<4>();
    }

    #[test]
    fn box_half_preserves_pass_rounding_unaligned_tails_and_guards() {
        fn check<const C: usize>() {
            for count in 0..=65 {
                for offset in 1..=3 {
                    let upper: Vec<_> = (0..count * C * 2 + offset + 1).map(|i| (i * 37) as u8).collect();
                    let lower: Vec<_> = (0..upper.len()).map(|i| (i * 73 + i / 7) as u8).collect();
                    let mut output = vec![0xa5; count * C + 2];
                    super::box_half::<C>(
                        &upper[offset..offset + count * C * 2],
                        &lower[offset..offset + count * C * 2],
                        &mut output[1..count * C + 1],
                    );
                    for i in 0..count * C {
                        let start = offset + i / C * C * 2 + i % C;
                        let average = |row: &[u8]| (u16::from(row[start]) + u16::from(row[start + C])).div_ceil(2);
                        let expected = (average(&upper) + average(&lower)).div_ceil(2) as u8;
                        assert_eq!(output[i + 1], expected);
                    }
                    assert_eq!(output[0], 0xa5);
                    assert_eq!(output[count * C + 1], 0xa5);
                }
            }
        }
        check::<1>();
        check::<2>();
        check::<3>();
        check::<4>();
    }

    #[test]
    fn float_reduction_preserves_odd_edges_and_accumulation_order() {
        for width in 0_usize..=33 {
            let samples = [0.0_f32, -0.0, f32::MAX, -f32::MAX, 1.25, -2.75, f32::MIN_POSITIVE, f32::INFINITY, f32::NAN];
            let rows: [Vec<u8>; 2] = std::array::from_fn(|r| {
                std::iter::once(93).chain((0..width).flat_map(|i| samples[(i + r * 3) % samples.len()].to_le_bytes())).collect()
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
                    let sum: u32 = [&upper, &lower].iter().flat_map(|r| &r[1 + i * 2..1 + (i + 1) * 2]).map(|&v| u32::from(v)).sum();
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
                    let sum: u32 = [&upper, &middle, &lower].iter().flat_map(|r| &r[1 + i * 3..1 + (i + 1) * 3]).map(|&v| u32::from(v)).sum();
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
                let expected: Vec<_> = (0..count).flat_map(|i| (0..C).map(move |c| 1 + (i * 2 + 1) * C + c)).map(|i| source[i]).collect();
                assert_eq!(&output[1..count * C + 1], expected);
                assert_eq!(output[0], 199);
                assert_eq!(output[count * C + 1], 199);
            }
        }
        check::<1>();
        check::<2>();
        check::<3>();
        check::<4>();
        check::<6>();
        check::<8>();
    }

    #[test]
    fn horizontal_half_matches_scalar_with_unaligned_buffers_and_guards() {
        fn check<const C: usize>() {
            for weights in [
                &[][..],
                &[1 << 22][..],
                &[-(1 << 22)][..],
                &[2 << 22][..],
                &[-400_000, 2_000_000, 3_000_000, -405_696][..],
                &[200_000; 3][..],
                &[200_000; 11][..],
                &[200_000; 12][..],
            ] {
                for white in [false, true] {
                    let source: Vec<u8> = (0..1 + (31 + weights.len()) * C).map(|i| if white { 255 } else { (i * 137) as u8 }).collect();
                    let mut actual = vec![93; 16 * C + 2];
                    horizontal_half::<C>(&source[1..], &mut actual[1..16 * C + 1], weights);
                    for x in 0..16 {
                        for c in 0..C {
                            let sum = weights.iter().enumerate().fold(i64::from(RESAMPLE_ROUND), |sum, (tap, &weight)| {
                                sum + i64::from(source[1 + (x * 2 + tap) * C + c]) * i64::from(weight)
                            });
                            assert_eq!(actual[1 + x * C + c], (sum >> RESAMPLE_BITS).clamp(0, 255) as u8);
                        }
                    }
                    assert_eq!(actual[0], 93);
                    assert_eq!(actual[16 * C + 1], 93);
                }
            }
        }
        check::<1>();
        check::<2>();
        check::<3>();
        check::<4>();
    }

    #[test]
    fn horizontal_batches_match_scalar_signed_rounding_and_saturation() {
        fn check<const C: usize>() {
            for weights in [&[][..], &[1 << 22][..], &[-(1 << 22)][..], &[2 << 22][..], &[-400_000, 2_000_000, 3_000_000, -405_696][..]] {
                let data: [Vec<u8>; 4] = std::array::from_fn(|row| (0..1 + weights.len() * C).map(|i| (i * 137 + row * 37) as u8).collect());
                let source = data.each_ref().map(|row| row[1..].as_chunks::<C>().0);
                let actual = horizontal_four::<C>(source, weights);
                for row in 0..4 {
                    for c in 0..C {
                        let sum = weights
                            .iter()
                            .enumerate()
                            .fold(i64::from(RESAMPLE_ROUND), |sum, (i, &weight)| sum + i64::from(source[row][i][c]) * i64::from(weight));
                        assert_eq!(actual[row][c], (sum >> RESAMPLE_BITS).clamp(0, 255) as u8);
                    }
                }
            }
        }
        check::<1>();
        check::<2>();
        check::<3>();
        check::<4>();
    }

    #[test]
    fn vertical_wide_matches_scalar_order_at_unaligned_boundaries() {
        for count in [0, 1, 3, 4, 5, 7, 8, 9, 31, 32, 33] {
            for weights in [&[][..], &[1 << 22][..], &[-(1 << 22)][..], &[2 << 22][..], &[-400_000, 2_000_000, 3_000_000, -405_696][..]] {
                let stride = count + 3;
                let data: Vec<_> = (0..1 + stride * weights.len()).map(|i| (i * 137 % 65536) as f64 + 0.125).collect();
                let source = &data[1..];
                let mut output = vec![91.0; count + 2];
                let done = vertical_wide(source, &mut output[1..count + 1], stride, weights, 4095.0);
                for i in 0..done {
                    let expected = weights
                        .iter()
                        .enumerate()
                        .fold(0.0, |sum, (y, &weight)| sum + source[y * stride + i] * f64::from(weight) / f64::from(crate::ops::RESAMPLE_SCALE));
                    assert_eq!(output[i + 1], expected.clamp(0.0, 4095.0));
                }
                assert_eq!(output[0], 91.0);
                assert!(output[done + 1..].iter().all(|&v| v == 91.0));
            }
        }
    }

    #[test]
    fn wide_grayscale_batches_preserve_unaligned_samples_and_accumulation_order() {
        for depth in [10, 12, 16] {
            let maximum = (1_u32 << depth) - 1;
            for weights in [&[][..], &[1 << 22][..], &[-(1 << 22)][..], &[2 << 22][..], &[-400_000, 2_000_000, 3_000_000, -405_696][..]] {
                let sources = std::array::from_fn::<_, 4, _>(|row| {
                    let mut bytes = vec![91];
                    bytes.extend((0..weights.len()).flat_map(|i| {
                        let value = match (i + row) % 4 {
                            0 => 0,
                            1 => maximum,
                            _ => (i * 719 + row * 137) as u32 & maximum,
                        };
                        (value as u16).to_le_bytes()
                    }));
                    bytes.push(93);
                    bytes
                });
                let actual = horizontal_wide_four(sources.each_ref().map(|v| &v[1..v.len() - 1]), weights);
                for row in 0..4 {
                    let expected: f64 = weights
                        .iter()
                        .enumerate()
                        .map(|(i, &weight)| {
                            let value = u16::from_le_bytes([sources[row][i * 2 + 1], sources[row][i * 2 + 2]]);
                            f64::from(value) * f64::from(weight) / f64::from(crate::ops::RESAMPLE_SCALE)
                        })
                        .sum();
                    assert_eq!(actual[row], expected);
                    assert_eq!(sources[row][0], 91);
                    assert_eq!(sources[row][sources[row].len() - 1], 93);
                }
            }
        }
    }

    #[test]
    fn horizontal_wide_preserves_alpha_precision_and_scalar_order() {
        fn check<const C: usize>() {
            for depth in [10, 12, 16] {
                let maximum = (1_u32 << depth) - 1;
                for weights in [&[][..], &[1 << 22][..], &[-(1 << 22)][..], &[2 << 22][..], &[-400_000, 2_000_000, 3_000_000, -405_696][..]] {
                    for alpha in [0, 1, maximum / 2, maximum] {
                        let samples: Vec<_> = (0..weights.len() * C)
                            .map(|i| if C == 4 && i % C == 3 { alpha as u16 } else { (i as u32 * 719 % (maximum + 1)) as u16 })
                            .collect();
                        let mut bytes = vec![91];
                        bytes.extend(samples.iter().flat_map(|v| v.to_le_bytes()));
                        let actual = horizontal_wide::<C>(&bytes[1..], weights, f64::from(maximum));
                        for c in 0..C {
                            let expected = weights
                                .iter()
                                .enumerate()
                                .map(|(i, &weight)| {
                                    let value = f64::from(samples[i * C + c]);
                                    let value = if C == 4 && c < 3 { value * f64::from(samples[i * C + 3]) / f64::from(maximum) } else { value };
                                    value * f64::from(weight) / f64::from(crate::ops::RESAMPLE_SCALE)
                                })
                                .sum::<f64>();
                            assert_eq!(actual[c], expected);
                        }
                    }
                }
            }
        }
        check::<3>();
        check::<4>();
    }

    #[test]
    fn nearest_double_rgb_unaligned_tails_and_guards() {
        for count in [0, 1, 15, 16, 17, 31, 32, 33, 400] {
            for offset in 0..16 {
                let source: Vec<u8> = (0..count * 3 + offset).map(|i| (i * 37 + i / 11) as u8).collect();
                let source = &source[offset..];
                let mut output = vec![199; offset + count * 6 + 16];
                super::nearest_double_rgb(source, &mut output[offset..offset + count * 6]);
                let expected: Vec<_> = source.as_chunks::<3>().0.iter().flat_map(|p| [*p, *p]).flatten().collect();
                assert_eq!(&output[offset..offset + count * 6], expected);
                assert!(output[..offset].iter().all(|&v| v == 199));
                assert!(output[offset + count * 6..].iter().all(|&v| v == 199));
            }
        }
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
        assert_eq!(count, 16);
        for x in 0..count {
            let sum = weights.iter().enumerate().fold(1_i64 << 21, |sum, (i, &w)| sum + i64::from(source[i * 31 + x]) * i64::from(w));
            assert_eq!(actual[x], (sum >> 22).clamp(0, 255) as u8);
        }
    }

    #[test]
    fn vertical_preserves_tails_and_guards() {
        for width in [0, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33] {
            for weights in [&[][..], &[1 << 22][..], &[-(1 << 22)][..], &[2 << 22][..], &[-400_000, 2_000_000, 3_000_000, -405_696][..]] {
                let stride = width + 3;
                // Offset both buffers by one to exercise unaligned loads/stores.
                let source: Vec<u8> = (0..1 + weights.len() * stride).map(|i| (i * 67) as u8).collect();
                let mut actual = vec![123; width + 2];
                let count = vertical(&source[1..], &mut actual[1..1 + width], stride, weights);
                assert_eq!(count, width / 16 * 16);
                for x in 0..count {
                    let sum = weights.iter().enumerate().fold(1_i64 << 21, |sum, (i, &w)| sum + i64::from(source[1 + i * stride + x]) * i64::from(w));
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
        for width in [1, 15, 16, 17, 31, 32, 33, 64, 256] {
            for channels in [1, 2, 3, 4] {
                let bytes = width * channels;
                let source: Vec<u8> = (0..bytes + 2).map(|i| (i * 37) as u8).collect();
                for n in [0, 1, 15, 16, 17, 31, 32, 33, 127] {
                    for step in [0, 1, 2, 3, 5, 31] {
                        for reverse in [false, true] {
                            let columns: Vec<_> = (0..n * channels)
                                .map(|i| {
                                    let x = i / channels * step % width;
                                    Some((if reverse { width - 1 - x } else { x }) * channels + i % channels)
                                })
                                .collect();
                            let map = NearestL::new(&columns, bytes).unwrap();
                            let mut output = vec![123; n * channels + 2];
                            map.sample(&source[1..bytes + 1], &mut output[1..n * channels + 1]);
                            for (i, column) in columns.iter().enumerate() {
                                assert_eq!(output[i + 1], source[1 + column.unwrap()]);
                            }
                            assert_eq!(output[0], 123);
                            assert_eq!(output[n * channels + 1], 123);
                        }
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
        assert!(map.is_vectorized());
    }
}
