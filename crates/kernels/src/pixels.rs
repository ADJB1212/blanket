//! Portable kernels for packed byte pixels.
use std::simd::cmp::{SimdOrd, SimdPartialEq, SimdPartialOrd};
use std::simd::num::{SimdFloat, SimdInt, SimdUint};
use std::simd::{Select, Simd, ToBytes, simd_swizzle};

// Multiplication by ceil(2^24 / 9) implements division for a 3x3 box sum.
pub const BOX_3X3_RECIPROCAL: u32 = 1_864_135;
pub const BOX_3X3_RECIPROCAL_BITS: u32 = 24;

pub type Bytes = Simd<u8, 16>;

pub fn histogram_count(values: &[u64]) -> u128 {
    let mut low = Simd::<u64, 8>::splat(0);
    let mut high = Simd::<u64, 8>::splat(0);
    let (blocks, tail) = values.as_chunks::<8>();
    for block in blocks {
        let previous = low;
        low += Simd::from_array(*block);
        high += low.simd_lt(previous).select(Simd::splat(1), Simd::splat(0));
    }
    let total: u128 = low.to_array().into_iter().zip(high.to_array()).map(|(lo, hi)| u128::from(lo) + (u128::from(hi) << 64)).sum();
    total + tail.iter().map(|&v| u128::from(v)).sum::<u128>()
}

pub fn histogram_bounds(values: &[u64]) -> Option<(usize, usize)> {
    if values.first().is_some_and(|&v| v != 0) && values.last().is_some_and(|&v| v != 0) {
        return Some((0, values.len() - 1));
    }
    let mut first = 0;
    while first + 8 <= values.len() {
        let mask = Simd::<u64, 8>::from_slice(&values[first..]).simd_ne(Simd::splat(0)).to_bitmask();
        if mask != 0 {
            first += mask.trailing_zeros() as usize;
            break;
        }
        first += 8;
    }
    while first < values.len() && values[first] == 0 {
        first += 1;
    }
    if first == values.len() {
        return None;
    }
    let mut last = values.len();
    while last >= first + 8 {
        let mask = Simd::<u64, 8>::from_slice(&values[last - 8..]).simd_ne(Simd::splat(0)).to_bitmask();
        if mask != 0 {
            return Some((first, last - 8 + (63 - mask.leading_zeros()) as usize));
        }
        last -= 8;
    }
    while values[last - 1] == 0 {
        last -= 1;
    }
    Some((first, last - 1))
}

pub fn pack_i32_planes<const C: usize, const WIDE: bool>(planes: [&[i32]; C], maximum: i32, output: &mut [u8]) {
    assert!((1..=4).contains(&C) && (1..=65535).contains(&maximum));
    let count = planes[0].len();
    assert!(planes.iter().all(|p| p.len() == count));
    let channels = if WIDE && C == 2 { 4 } else { C };
    let stride = channels * if WIDE { 2 } else { 1 };
    assert_eq!(output.len(), count * stride);
    let scale = if WIDE { 65535_u32 } else { 255 };
    let end = count / 16 * 16;
    for i in (0..end).step_by(16) {
        let values: [Simd<u16, 16>; C] = std::array::from_fn(|c| {
            let v = Simd::<i32, 16>::from_slice(&planes[c][i..]).simd_clamp(Simd::splat(0), Simd::splat(maximum)).cast::<u32>();
            let numerator = v * Simd::splat(scale) + Simd::splat((maximum / 2) as u32);
            (numerator.cast::<f64>() / Simd::splat(f64::from(maximum))).cast::<u16>()
        });
        if WIDE && C == 2 {
            store_sample_bytes::<4>(&mut output[i * stride..], [values[0], values[0], values[0], values[1]]);
        } else if WIDE {
            store_sample_bytes::<C>(&mut output[i * stride..], values);
        } else {
            store::<C>(&mut output[i * stride..], values.map(|v| v.cast::<u8>()));
        }
    }
    for i in end..count {
        for c in 0..channels {
            let channel = if WIDE && C == 2 { usize::from(c == 3) } else { c };
            let value = ((i64::from(planes[channel][i].clamp(0, maximum)) * i64::from(scale) + i64::from(maximum / 2)) / i64::from(maximum)) as u16;
            if WIDE {
                output[i * stride + c * 2..i * stride + c * 2 + 2].copy_from_slice(&value.to_le_bytes());
            } else {
                output[i * stride + c] = value as u8;
            }
        }
    }
}

pub fn float_color_layout<const S: usize, const D: usize>(source: &[f32], output: &mut [f32]) {
    assert!(matches!((S, D), (3, 4) | (4, 3)));
    assert_eq!(source.len() % S, 0);
    assert_eq!(output.len(), source.len() / S * D);
    if S == 3 {
        for (src, dst) in source.as_chunks::<S>().0.iter().zip(output.as_chunks_mut::<D>().0) {
            dst[..3].copy_from_slice(&src[..3]);
            dst[3] = 1.0;
        }
        return;
    }
    let count = source.len() / S;
    let end = count / 16 * 16;
    for i in (0..end).step_by(16) {
        let src = &source[i * S..];
        let (a, b) = Simd::<f32, 16>::from_slice(src).deinterleave(Simd::from_slice(&src[16..]));
        let (c, d) = Simd::<f32, 16>::from_slice(&src[32..]).deinterleave(Simd::from_slice(&src[48..]));
        let (r, blue) = a.deinterleave(c);
        let (g, _) = b.deinterleave(d);
        store_rgb(&mut output[i * D..], [r, g, blue]);
    }
    for (src, dst) in source[end * S..].as_chunks::<S>().0.iter().zip(output[end * D..].as_chunks_mut::<D>().0) {
        dst[..3].copy_from_slice(&src[..3]);
        if D == 4 {
            dst[3] = 1.0;
        }
    }
}

pub fn alpha_scale<const C: usize, const DIVIDE: bool>(pixels: &mut [u8]) {
    assert!(C == 2 || C == 4);
    assert_eq!(pixels.len() % C, 0);
    let end = pixels.len() / (C * 16) * (C * 16);
    let max = Simd::<u32, 16>::splat(255);
    for offset in (0..end).step_by(C * 16) {
        let mut p = load::<C>(&pixels[offset..]);
        let alpha = p[C - 1].cast::<u32>();
        let zero = alpha.simd_eq(Simd::splat(0));
        let divisor = zero.select(Simd::splat(1), alpha);
        for channel in &mut p[..C - 1] {
            let value = channel.cast::<u32>();
            *channel = if DIVIDE {
                // Byte-sample quotient gaps exceed f32 rounding error.
                zero.select(value, ((value * max).cast::<f32>() / divisor.cast::<f32>()).cast::<u32>().simd_min(max)).cast::<u8>()
            } else {
                let value = value * alpha + Simd::splat(128);
                (((value >> 8) + value) >> 8).cast::<u8>()
            };
        }
        store::<C>(&mut pixels[offset..], p);
    }
    for p in pixels[end..].as_chunks_mut::<C>().0 {
        for c in 0..C - 1 {
            p[c] = if DIVIDE {
                if p[C - 1] == 0 { p[c] } else { (u32::from(p[c]) * 255 / u32::from(p[C - 1])).min(255) as u8 }
            } else {
                let value = u32::from(p[c]) * u32::from(p[C - 1]) + 128;
                (((value >> 8) + value) >> 8) as u8
            };
        }
    }
}

pub fn rgba_channels<const C: usize>(source: &[u8], output: &mut [u8], first: usize) {
    assert!(C == 1 || C == 3);
    assert!(first + C <= 4);
    assert_eq!(source.len() % 4, 0);
    assert_eq!(output.len(), source.len() / 4 * C);
    let end = source.len() / 64 * 16;
    for i in (0..end).step_by(16) {
        let channels = load::<4>(&source[i * 4..]);
        store::<C>(&mut output[i * C..], std::array::from_fn(|c| channels[first + c]));
    }
    for (src, dst) in source[end * 4..].as_chunks::<4>().0.iter().zip(output[end * C..].as_chunks_mut::<C>().0) {
        dst.copy_from_slice(&src[first..first + C]);
    }
}

pub fn paint_mask<const C: usize>(output: &mut [u8], mask: &[u8], ink: &[u8], blend: bool) {
    assert_eq!(output.len(), mask.len() * C);
    assert!(ink.len() >= C && (!blend || ink.len() >= 4));
    let end = mask.len() / 16 * 16;
    let max = Simd::<u32, 16>::splat(255);
    let round = Simd::splat(127);
    for i in (0..end).step_by(16) {
        let coverage = Bytes::from_slice(&mask[i..]).cast::<u32>();
        let alpha = if blend { (coverage * Simd::splat(u32::from(ink[3])) + round) / max } else { coverage };
        let mut channels = load::<C>(&output[i * C..]);
        let transparent = if C == 4 { channels[3].simd_eq(Bytes::splat(0)) } else { channels[0].simd_eq(channels[0]) };
        for c in 0..C {
            let weight = if C == 4 && c < 3 {
                transparent.cast::<i32>().select(max, coverage)
            } else if C == 4 {
                coverage
            } else {
                alpha
            };
            let value = ((Simd::splat(u32::from(ink[c])) * weight + channels[c].cast::<u32>() * (max - weight) + round) / max).cast::<u8>();
            channels[c] = coverage.simd_eq(Simd::splat(0)).cast::<i8>().select(channels[c], value);
        }
        store::<C>(&mut output[i * C..], channels);
    }
    for (dst, &coverage) in output[end * C..].as_chunks_mut::<C>().0.iter_mut().zip(&mask[end..]) {
        if coverage == 0 {
            continue;
        }
        let transparent = C == 4 && dst[3] == 0;
        let alpha = if blend { (u32::from(coverage) * u32::from(ink[3]) + 127) / 255 } else { u32::from(coverage) };
        for c in 0..C {
            let weight = if C == 4 && c < 3 && transparent {
                255
            } else if C == 4 {
                u32::from(coverage)
            } else {
                alpha
            };
            dst[c] = ((u32::from(ink[c]) * weight + u32::from(dst[c]) * (255 - weight) + 127) / 255) as u8;
        }
    }
}

pub fn extract_i32_channel<const C: usize, const WIDE: bool>(source: &[u8], output: &mut [i32], channel: usize) {
    assert!(channel < C);
    let stride = C * if WIDE { 2 } else { 1 };
    assert_eq!(source.len(), output.len() * stride);
    let end = output.len() / 16 * 16;
    for i in (0..end).step_by(16) {
        let values = if WIDE && C == 2 {
            let bytes = load::<4>(&source[i * stride..]);
            bytes[channel * 2].cast::<i32>() | (bytes[channel * 2 + 1].cast::<i32>() << Simd::splat(8))
        } else if WIDE {
            load_sample_bytes::<C>(&source[i * stride..])[channel].cast::<i32>()
        } else {
            load::<C>(&source[i * stride..])[channel].cast::<i32>()
        };
        values.copy_to_slice(&mut output[i..i + 16]);
    }
    for (src, dst) in source[end * stride..].chunks_exact(stride).zip(&mut output[end..]) {
        *dst = if WIDE { i32::from(u16::from_le_bytes([src[channel * 2], src[channel * 2 + 1]])) } else { i32::from(src[channel]) };
    }
}

pub fn expand_luma_alpha<const WIDE: bool, const BIG_ENDIAN: bool>(source: &[u8], output: &mut [u8]) {
    let stride = if WIDE { 4 } else { 2 };
    assert_eq!(source.len() % stride, 0);
    assert_eq!(output.len(), source.len() * 2);
    let count = source.len() / stride;
    let end = count / 16 * 16;
    for x in (0..end).step_by(16) {
        if WIDE {
            let p = load::<4>(&source[x * 4..]);
            let [l, h, a, b] = if BIG_ENDIAN { [p[1], p[0], p[3], p[2]] } else { p };
            store::<8>(&mut output[x * 8..], [l, h, l, h, l, h, a, b]);
        } else {
            let [l, a] = load::<2>(&source[x * 2..]);
            store::<4>(&mut output[x * 4..], [l, l, l, a]);
        }
    }
    for (src, dst) in source[end * stride..].chunks_exact(stride).zip(output[end * stride * 2..].chunks_exact_mut(stride * 2)) {
        if WIDE {
            let [l, h, a, b] = if BIG_ENDIAN { [src[1], src[0], src[3], src[2]] } else { [src[0], src[1], src[2], src[3]] };
            dst.copy_from_slice(&[l, h, l, h, l, h, a, b]);
        } else {
            dst.copy_from_slice(&[src[0], src[0], src[0], src[1]]);
        }
    }
}

pub fn restore_alpha<const COPY: bool>(output: &mut [u8], source: &[u8]) {
    assert_eq!(output.len() % 4, 0);
    if COPY {
        assert_eq!(source.len(), output.len());
    }
    let mask = Bytes::from_array([0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255]);
    let end = if output.len() < 128 { 0 } else { output.len() / 16 * 16 };
    for i in (0..end).step_by(16) {
        let rgb = Bytes::from_slice(&output[i..]);
        let alpha = if COPY { Bytes::from_slice(&source[i..]) & mask } else { mask };
        ((rgb & !mask) | alpha).copy_to_slice(&mut output[i..i + 16]);
    }
    for (i, pixel) in output[end..].as_chunks_mut::<4>().0.iter_mut().enumerate() {
        pixel[3] = if COPY { source[end + i * 4 + 3] } else { 255 };
    }
}

pub fn blend_solid_rgb(output: &mut [u8], color: [u8; 3], alpha: u8) {
    assert_eq!(output.len() % 3, 0);
    if output.len() < 96 {
        for pixel in output.as_chunks_mut::<3>().0 {
            for c in 0..3 {
                pixel[c] = ((u32::from(pixel[c]) * (255 - u32::from(alpha)) + u32::from(color[c]) * u32::from(alpha) + 127) / 255) as u8;
            }
        }
        return;
    }
    let a = Simd::<u16, 16>::splat(u16::from(alpha));
    let inverse = Simd::splat(255) - a;
    let colors: [_; 3] = std::array::from_fn(|block| {
        Simd::<u16, 16>::from_array(std::array::from_fn(|lane| u16::from(color[(block * 16 + lane) % 3]))) * a + Simd::splat(127)
    });
    let (groups, tail) = output.as_chunks_mut::<48>();
    for group in groups {
        for (block, color) in group.as_chunks_mut::<16>().0.iter_mut().zip(colors) {
            ((Bytes::from_array(*block).cast::<u16>() * inverse + color) / Simd::splat(255)).cast::<u8>().copy_to_slice(block);
        }
    }
    for pixel in tail.as_chunks_mut::<3>().0 {
        for c in 0..3 {
            pixel[c] = ((u32::from(pixel[c]) * (255 - u32::from(alpha)) + u32::from(color[c]) * u32::from(alpha) + 127) / 255) as u8;
        }
    }
}

pub fn swap_red_blue<const I: usize, const O: usize>(source: &[u8], output: &mut [u8]) {
    assert!(matches!(I, 3 | 4) && matches!(O, 3 | 4));
    assert_eq!(source.len() % I, 0);
    assert_eq!(output.len(), source.len() / I * O);
    let count = source.len() / I;
    let end = if I == 3 { 0 } else { count / 16 * 16 };
    for x in (0..end).step_by(16) {
        let src = load::<I>(&source[x * I..]);
        let mut dst = [Bytes::splat(0); O];
        dst[0] = src[2];
        dst[1] = src[1];
        dst[2] = src[0];
        if I == 4 && O == 4 {
            dst[3] = src[3];
        }
        store::<O>(&mut output[x * O..], dst);
    }
    for (src, dst) in source[end * I..].as_chunks::<I>().0.iter().zip(output[end * O..].as_chunks_mut::<O>().0) {
        dst[..3].copy_from_slice(&[src[2], src[1], src[0]]);
        if O == 4 {
            dst[3] = if I == 4 { src[3] } else { 0 };
        }
    }
}

#[cfg(test)]
mod channel_swap_tests {
    use super::*;

    #[test]
    fn histogram_vectors_preserve_overflow_extrema_and_tails() {
        for count in [0, 1, 7, 8, 9, 15, 16, 17, 65, 256, 65536] {
            for offset in 0..8 {
                let values: Vec<u64> = (0..count + offset).map(|i| [0, 1, u64::MAX, u64::MAX - 1][i % 4]).collect();
                let values = &values[offset..];
                assert_eq!(histogram_count(values), values.iter().map(|&v| u128::from(v)).sum::<u128>());
                let mut bins = vec![0; count + offset];
                assert_eq!(histogram_bounds(&bins[offset..]), None);
                for i in 0..count {
                    bins[offset + i] = u64::MAX;
                    let actual = histogram_bounds(&bins[offset..]);
                    assert_eq!(actual, Some((0, i)));
                    if count > 256 && i > 256 {
                        break;
                    }
                }
            }
        }
    }

    #[test]
    fn planar_normalization_matches_integer_division() {
        fn check<const C: usize, const W: bool>() {
            for depth in 1..=16 {
                let maximum = (1_i32 << depth) - 1;
                let scale = if W { 65535_i64 } else { 255 };
                let count = maximum as usize + 4;
                let values: Vec<i32> = (0..count).map(|i| i as i32 - 2).collect();
                let planes = std::array::from_fn(|_| values.as_slice());
                let channels = if W && C == 2 { 4 } else { C };
                let stride = channels * if W { 2 } else { 1 };
                let mut output = vec![0xa5; count * stride + 2];
                pack_i32_planes::<C, W>(planes, maximum, &mut output[1..count * stride + 1]);
                for (i, pixel) in output[1..count * stride + 1].chunks_exact(stride).enumerate() {
                    let expected = ((i64::from(values[i].clamp(0, maximum)) * scale + i64::from(maximum / 2)) / i64::from(maximum)) as u16;
                    if W {
                        assert!(pixel.as_chunks::<2>().0.iter().all(|&p| u16::from_le_bytes(p) == expected));
                    } else {
                        assert!(pixel.iter().all(|&p| u16::from(p) == expected));
                    }
                }
                assert_eq!((output[0], output[count * stride + 1]), (0xa5, 0xa5));
            }
        }
        check::<1, false>();
        check::<2, false>();
        check::<3, false>();
        check::<4, false>();
        check::<1, true>();
        check::<2, true>();
        check::<3, true>();
        check::<4, true>();
    }

    #[test]
    fn planar_normalization_handles_unaligned_boundaries_and_guards() {
        fn check<const C: usize, const W: bool>() {
            for count in 0..=65 {
                for offset in 0..16 {
                    let planes: [Vec<i32>; C] =
                        std::array::from_fn(|c| (0..offset + count).map(|i| [i32::MIN, -1, 0, 1, 255, 4095, 65535, i32::MAX][(i + c) % 8]).collect());
                    let channels = if W && C == 2 { 4 } else { C };
                    let stride = channels * if W { 2 } else { 1 };
                    let mut output = vec![0xa5; offset + count * stride + 17];
                    pack_i32_planes::<C, W>(std::array::from_fn(|c| &planes[c][offset..]), 4095, &mut output[offset..offset + count * stride]);
                    for i in 0..count {
                        for c in 0..channels {
                            let channel = if W && C == 2 { usize::from(c == 3) } else { c };
                            let expected =
                                ((i64::from(planes[channel][offset + i].clamp(0, 4095)) * if W { 65535 } else { 255 } + 2047) / 4095) as u16;
                            let index = offset + i * stride + c * if W { 2 } else { 1 };
                            let actual = if W { u16::from_le_bytes([output[index], output[index + 1]]) } else { u16::from(output[index]) };
                            assert_eq!(actual, expected);
                        }
                    }
                    assert!(output[..offset].iter().chain(&output[offset + count * stride..]).all(|&v| v == 0xa5));
                }
            }
        }
        check::<1, false>();
        check::<2, false>();
        check::<3, false>();
        check::<4, false>();
        check::<1, true>();
        check::<2, true>();
        check::<3, true>();
        check::<4, true>();
    }

    #[test]
    fn float_layout_preserves_nonfinite_bits_and_guards() {
        fn check<const S: usize, const D: usize>() {
            let values = [0.0, -0.0, f32::INFINITY, f32::NEG_INFINITY, f32::from_bits(0x7fc00123), f32::from_bits(1), 0.25];
            for count in 0..=65 {
                for offset in 0..16 {
                    let source: Vec<f32> = (0..offset + count * S).map(|i| values[i % values.len()]).collect();
                    let mut output = vec![123.0; offset + count * D + 17];
                    float_color_layout::<S, D>(&source[offset..], &mut output[offset..offset + count * D]);
                    for i in 0..count {
                        for c in 0..D {
                            assert_eq!(
                                output[offset + i * D + c].to_bits(),
                                if c < 3 { source[offset + i * S + c].to_bits() } else { 1.0_f32.to_bits() }
                            );
                        }
                    }
                    assert!(output[..offset].iter().chain(&output[offset + count * D..]).all(|&v| v == 123.0));
                }
            }
        }
        check::<3, 4>();
        check::<4, 3>();
    }

    #[test]
    fn alpha_scaling_is_exhaustive_and_preserves_guards() {
        fn check<const C: usize, const D: bool>() {
            for offset in 0..16 {
                let mut data = vec![0xa5; offset];
                for a in 0..=255_u8 {
                    for value in 0..=255_u8 {
                        data.extend((0..C).map(|c| if c == C - 1 { a } else { value }));
                    }
                }
                let end = data.len();
                data.extend_from_slice(&[0xa5; 19]);
                alpha_scale::<C, D>(&mut data[offset..end]);
                for (i, pixel) in data[offset..end].as_chunks::<C>().0.iter().enumerate() {
                    let a = (i / 256) as u32;
                    let v = (i % 256) as u32;
                    let expected = if D {
                        (v * 255).checked_div(a).map_or(v, |q| q.min(255))
                    } else {
                        let t = v * a + 128;
                        ((t >> 8) + t) >> 8
                    };
                    assert!(pixel[..C - 1].iter().all(|&sample| u32::from(sample) == expected));
                    assert_eq!(u32::from(pixel[C - 1]), a);
                }
                assert!(data[..offset].iter().chain(&data[end..]).all(|&v| v == 0xa5));
            }
        }
        check::<2, false>();
        check::<2, true>();
        check::<4, false>();
        check::<4, true>();
    }

    #[test]
    fn mask_and_pdf_channels_preserve_boundaries_and_alpha() {
        fn check<const C: usize>() {
            for count in 0..=65 {
                for offset in 0..16 {
                    let source: Vec<u8> = (0..count * C).map(|i| (i * 73) as u8).collect();
                    let mask: Vec<u8> = (0..count).map(|i| [0, 1, 127, 254, 255][i % 5]).collect();
                    for blend in [false, true] {
                        let ink = [21, 149, 231, 137];
                        let mut data = vec![0xa5; offset + source.len() + 17];
                        data[offset..offset + source.len()].copy_from_slice(&source);
                        paint_mask::<C>(&mut data[offset..offset + source.len()], &mask, &ink, blend);
                        for i in 0..count {
                            for c in 0..C {
                                let original = source[i * C + c];
                                let a = u32::from(mask[i]);
                                let weight = if C == 4 && c < 3 && source[i * C + 3] == 0 {
                                    255
                                } else if C == 4 || !blend {
                                    a
                                } else {
                                    (a * u32::from(ink[3]) + 127) / 255
                                };
                                let expected = if a == 0 {
                                    original
                                } else {
                                    ((u32::from(ink[c]) * weight + u32::from(original) * (255 - weight) + 127) / 255) as u8
                                };
                                assert_eq!(data[offset + i * C + c], expected);
                            }
                        }
                        assert!(data[..offset].iter().chain(&data[offset + source.len()..]).all(|&v| v == 0xa5));
                    }
                    if C == 1 || C == 3 {
                        let rgba: Vec<u8> = (0..offset + count * 4).map(|i| (i * 31) as u8).collect();
                        for first in 0..=4 - C {
                            let mut dst = vec![0xa5; offset + count * C + 17];
                            rgba_channels::<C>(&rgba[offset..], &mut dst[offset..offset + count * C], first);
                            for i in 0..count {
                                assert_eq!(&dst[offset + i * C..offset + (i + 1) * C], &rgba[offset + i * 4 + first..offset + i * 4 + first + C]);
                            }
                            assert!(dst[..offset].iter().chain(&dst[offset + count * C..]).all(|&v| v == 0xa5));
                        }
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
    fn planar_extraction_preserves_unaligned_samples_tails_and_guards() {
        fn check<const C: usize, const W: bool>() {
            let stride = C * if W { 2 } else { 1 };
            for count in 0..=65 {
                for offset in 0..16 {
                    let bytes: Vec<u8> = (0..offset + count * stride).map(|i| (i * 73) as u8).collect();
                    let source = &bytes[offset..];
                    for channel in 0..C {
                        let mut output = vec![-1; count + 2];
                        extract_i32_channel::<C, W>(source, &mut output[1..count + 1], channel);
                        for i in 0..count {
                            let index = i * stride + channel * if W { 2 } else { 1 };
                            assert_eq!(
                                output[i + 1],
                                if W { i32::from(u16::from_le_bytes([source[index], source[index + 1]])) } else { i32::from(source[index]) }
                            );
                        }
                        assert_eq!((output[0], output[count + 1]), (-1, -1));
                    }
                }
            }
        }
        check::<1, false>();
        check::<2, false>();
        check::<3, false>();
        check::<4, false>();
        check::<1, true>();
        check::<2, true>();
        check::<3, true>();
        check::<4, true>();
    }

    #[test]
    fn alpha_expansion_and_blending_preserve_samples_tails_and_guards() {
        for count in 0..=65 {
            for offset in 0..8 {
                let storage: Vec<u8> = (0..count * 4 + offset).map(|i| (i * 73) as u8).collect();
                let source = &storage[offset..];
                for wide in [false, true] {
                    for big in [false, true] {
                        let stride = if wide { 4 } else { 2 };
                        let src = &source[..count * stride];
                        let mut output = vec![123; offset + src.len() * 2 + 16];
                        let dst = &mut output[offset..offset + src.len() * 2];
                        match (wide, big) {
                            (false, false) => expand_luma_alpha::<false, false>(src, dst),
                            (false, true) => expand_luma_alpha::<false, true>(src, dst),
                            (true, false) => expand_luma_alpha::<true, false>(src, dst),
                            (true, true) => expand_luma_alpha::<true, true>(src, dst),
                        }
                        for (s, d) in src.chunks_exact(stride).zip(dst.chunks_exact(stride * 2)) {
                            if wide {
                                let p = if big { [s[1], s[0], s[3], s[2]] } else { [s[0], s[1], s[2], s[3]] };
                                assert_eq!(d, &[p[0], p[1], p[0], p[1], p[0], p[1], p[2], p[3]]);
                            } else {
                                assert_eq!(d, &[s[0], s[0], s[0], s[1]]);
                            }
                        }
                        assert!(output[..offset].iter().chain(&output[offset + src.len() * 2..]).all(|&v| v == 123));
                    }
                }
                for copy in [false, true] {
                    let mut output = vec![123; offset + source.len() + 16];
                    let dst = &mut output[offset..offset + source.len()];
                    if copy {
                        restore_alpha::<true>(dst, source);
                    } else {
                        restore_alpha::<false>(dst, source);
                    }
                    for (i, &v) in dst.iter().enumerate() {
                        assert_eq!(v, if i % 4 == 3 { if copy { source[i] } else { 255 } } else { 123 });
                    }
                    assert!(output[..offset].iter().chain(&output[offset + source.len()..]).all(|&v| v == 123));
                }
                for alpha in 0..=255 {
                    let mut output = vec![123; offset + count * 3 + 16];
                    let dst = &mut output[offset..offset + count * 3];
                    dst.copy_from_slice(&source[..count * 3]);
                    blend_solid_rgb(dst, [0, 127, 255], alpha);
                    for (i, &value) in dst.iter().enumerate() {
                        assert_eq!(
                            u32::from(value),
                            (u32::from(source[i]) * (255 - u32::from(alpha)) + [0, 127, 255][i % 3] * u32::from(alpha) + 127) / 255
                        );
                    }
                    assert!(output[..offset].iter().chain(&output[offset + count * 3..]).all(|&v| v == 123));
                }
            }
        }
    }

    fn check<const I: usize, const O: usize>() {
        for count in 0..=97 {
            for offset in 0..16 {
                let source: Vec<u8> = (0..count * I + offset).map(|i| (i * 73) as u8).collect();
                let source = &source[offset..];
                let mut output = vec![123; count * O + offset + 16];
                swap_red_blue::<I, O>(source, &mut output[offset..offset + count * O]);
                for x in 0..count {
                    assert_eq!(&output[offset + x * O..offset + x * O + 3], &[source[x * I + 2], source[x * I + 1], source[x * I]]);
                    if O == 4 {
                        assert_eq!(output[offset + x * O + 3], if I == 4 { source[x * I + 3] } else { 0 });
                    }
                }
                assert!(output[..offset].iter().all(|&v| v == 123));
                assert!(output[offset + count * O..].iter().all(|&v| v == 123));
            }
        }
    }

    #[test]
    fn swaps_preserve_unaligned_pixels_tails_and_guards() {
        check::<3, 3>();
        check::<4, 4>();
        check::<3, 4>();
        check::<4, 3>();
    }
}

#[inline]
pub fn load_u16<const BIG_ENDIAN: bool>(source: &[u8]) -> Simd<u16, 16> {
    let bytes = Simd::<u8, 32>::from_slice(source);
    if BIG_ENDIAN { Simd::<u16, 16>::from_be_bytes(bytes) } else { Simd::<u16, 16>::from_le_bytes(bytes) }
}

pub fn pack_bilevel<const INVERT: bool>(source: &[u8], output: &mut [u8]) {
    let wide_end = source.len() / 64 * 64;
    for i in (0..wide_end).step_by(64) {
        let nonzero = Simd::<u8, 64>::from_slice(&source[i..]).simd_ne(Simd::splat(0));
        let bits = if INVERT { !nonzero } else { nonzero }.to_bitmask();
        output[i / 8..i / 8 + 8].copy_from_slice(&bits.reverse_bits().to_be_bytes());
    }
    let end = source.len() / 16 * 16;
    for i in (wide_end..end).step_by(16) {
        let nonzero = Bytes::from_slice(&source[i..]).simd_ne(Bytes::splat(0));
        let bits = if INVERT { !nonzero } else { nonzero }.to_bitmask() as u16;
        output[i / 8] = (bits as u8).reverse_bits();
        output[i / 8 + 1] = ((bits >> 8) as u8).reverse_bits();
    }
    for (group, dst) in source[end..].chunks(8).zip(&mut output[end / 8..]) {
        *dst = group.iter().enumerate().fold(0, |bits, (bit, &value)| bits | (u8::from((value != 0) != INVERT) << (7 - bit)));
    }
}

#[inline]
pub fn load<const C: usize>(source: &[u8]) -> [Bytes; C] {
    assert!(source.len() >= 16 * C);
    let mut result = [Bytes::splat(0); C];
    match C {
        1 => result[0] = Bytes::from_slice(&source[..16]),
        2 => {
            let channels = <[Bytes; 2]>::from(Bytes::from_slice(&source[..16]).deinterleave(Bytes::from_slice(&source[16..32])));
            result.copy_from_slice(&channels);
        }
        3 => result.copy_from_slice(&load_rgb(source)),
        4 => {
            let (a, b) = Bytes::from_slice(&source[..16]).deinterleave(Bytes::from_slice(&source[16..32]));
            let (c, d) = Bytes::from_slice(&source[32..48]).deinterleave(Bytes::from_slice(&source[48..64]));
            let (r, blue) = a.deinterleave(c);
            let (g, alpha) = b.deinterleave(d);
            result.copy_from_slice(&[r, g, blue, alpha]);
        }
        6 => {
            let a = load_rgb(source);
            let b = load_rgb(&source[48..]);
            for c in 0..3 {
                (result[c], result[c + 3]) = a[c].deinterleave(b[c]);
            }
        }
        8 => {
            let a = load::<4>(source);
            let b = load::<4>(&source[64..]);
            for c in 0..4 {
                (result[c], result[c + 4]) = a[c].deinterleave(b[c]);
            }
        }
        _ => return std::array::from_fn(|c| Bytes::gather_or_default(source, Simd::from_array(std::array::from_fn(|i| i * C + c)))),
    }
    result
}

#[inline]
pub fn store<const C: usize>(output: &mut [u8], values: [Bytes; C]) {
    assert!(output.len() >= 16 * C);
    match C {
        1 => values[0].copy_to_slice(&mut output[..16]),
        2 => {
            let (a, b) = values[0].interleave(values[1]);
            a.copy_to_slice(&mut output[..16]);
            b.copy_to_slice(&mut output[16..32]);
        }
        3 => store_rgb(output, [values[0], values[1], values[2]]),
        4 => {
            let (a, c) = values[0].interleave(values[2]);
            let (b, d) = values[1].interleave(values[3]);
            let (ab, ba) = a.interleave(b);
            let (cd, dc) = c.interleave(d);
            for (chunk, value) in output[..64].as_chunks_mut::<16>().0.iter_mut().zip([ab, ba, cd, dc]) {
                value.copy_to_slice(chunk);
            }
        }
        6 => {
            let pairs = std::array::from_fn::<_, 3, _>(|c| values[c].interleave(values[c + 3]));
            store_rgb(&mut output[..48], pairs.map(|pair| pair.0));
            store_rgb(&mut output[48..], pairs.map(|pair| pair.1));
        }
        8 => {
            let pairs = std::array::from_fn::<_, 4, _>(|c| values[c].interleave(values[c + 4]));
            store::<4>(&mut output[..64], pairs.map(|pair| pair.0));
            store::<4>(&mut output[64..], pairs.map(|pair| pair.1));
        }
        _ => {
            for (c, value) in values.into_iter().enumerate() {
                value.scatter(output, Simd::from_array(std::array::from_fn(|i| i * C + c)));
            }
        }
    }
}

#[inline]
fn load_rgb<T: std::simd::SimdElement>(source: &[T]) -> [Simd<T, 16>; 3] {
    let a = Simd::<T, 16>::from_slice(&source[..16]);
    let b = Simd::<T, 16>::from_slice(&source[16..32]);
    let c = Simd::<T, 16>::from_slice(&source[32..48]);
    deinterleave_rgb(a, b, c)
}

#[inline]
fn deinterleave_rgb<T: std::simd::SimdElement>(a: Simd<T, 16>, b: Simd<T, 16>, c: Simd<T, 16>) -> [Simd<T, 16>; 3] {
    let t0 = simd_swizzle!(a, b, [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 0, 0, 0, 0, 0]);
    let t1 = simd_swizzle!(a, b, [1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 0, 0, 0, 0, 0]);
    let t2 = simd_swizzle!(a, b, [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 0, 0, 0, 0, 0, 0]);
    [
        simd_swizzle!(t0, c, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 17, 20, 23, 26, 29]),
        simd_swizzle!(t1, c, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 18, 21, 24, 27, 30]),
        simd_swizzle!(t2, c, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 19, 22, 25, 28, 31]),
    ]
}

#[inline]
fn store_rgb<T: std::simd::SimdElement>(output: &mut [T], values: [Simd<T, 16>; 3]) {
    for (dst, value) in output[..48].as_chunks_mut::<16>().0.iter_mut().zip(interleave_rgb(values)) {
        value.copy_to_slice(dst);
    }
}

#[inline]
fn interleave_rgb<T: std::simd::SimdElement>(values: [Simd<T, 16>; 3]) -> [Simd<T, 16>; 3] {
    let [a, b, c] = values;
    let t = simd_swizzle!(a, b, [0, 16, 0, 1, 17, 0, 2, 18, 0, 3, 19, 0, 4, 20, 0, 5]);
    let first = simd_swizzle!(t, c, [0, 1, 16, 3, 4, 17, 6, 7, 18, 9, 10, 19, 12, 13, 20, 15]);
    let t = simd_swizzle!(a, b, [21, 0, 6, 22, 0, 7, 23, 0, 8, 24, 0, 9, 25, 0, 10, 26]);
    let second = simd_swizzle!(t, c, [0, 21, 2, 3, 22, 5, 6, 23, 8, 9, 24, 11, 12, 25, 14, 15]);
    let t = simd_swizzle!(a, b, [0, 11, 27, 0, 12, 28, 0, 13, 29, 0, 14, 30, 0, 15, 31, 0]);
    let third = simd_swizzle!(t, c, [26, 1, 2, 27, 4, 5, 28, 7, 8, 29, 10, 11, 30, 13, 14, 31]);
    [first, second, third]
}

pub type Samples = Simd<u16, 16>;

#[inline]
pub fn load_sample_bytes<const C: usize>(source: &[u8]) -> [Samples; C] {
    assert!(source.len() >= 32 * C);
    let mut result = [Samples::splat(0); C];
    match C {
        1 => result[0] = load_u16::<false>(source),
        3 => result.copy_from_slice(&deinterleave_rgb(load_u16::<false>(source), load_u16::<false>(&source[32..]), load_u16::<false>(&source[64..]))),
        4 => {
            let (a, b) = load_u16::<false>(source).deinterleave(load_u16::<false>(&source[32..]));
            let (c, d) = load_u16::<false>(&source[64..]).deinterleave(load_u16::<false>(&source[96..]));
            let (r, blue) = a.deinterleave(c);
            let (g, alpha) = b.deinterleave(d);
            result.copy_from_slice(&[r, g, blue, alpha]);
        }
        _ => unreachable!("sample channels must be 1, 3, or 4"),
    }
    result
}

#[inline]
pub fn store_sample_bytes<const C: usize>(output: &mut [u8], values: [Samples; C]) {
    assert!(output.len() >= 32 * C);
    let mut packed = values;
    match C {
        1 => (),
        3 => packed.copy_from_slice(&interleave_rgb([values[0], values[1], values[2]])),
        4 => {
            let (a, c) = values[0].interleave(values[2]);
            let (b, d) = values[1].interleave(values[3]);
            let (ab, ba) = a.interleave(b);
            let (cd, dc) = c.interleave(d);
            packed.copy_from_slice(&[ab, ba, cd, dc]);
        }
        _ => unreachable!("sample channels must be 1, 3, or 4"),
    }
    for (dst, value) in output[..32 * C].as_chunks_mut::<32>().0.iter_mut().zip(packed) {
        value.to_le_bytes().copy_to_slice(dst);
    }
}

#[cfg(test)]
mod sample_tests {
    use super::*;

    #[test]
    fn merging_samples_preserves_channel_order_unaligned_blocks_tails_and_guards() {
        fn check<const C: usize>() {
            for count in 0..=65 {
                for offset in 0..4 {
                    let bands: [Vec<u16>; C] = std::array::from_fn(|c| (0..count + offset).map(|i| (i * 719 + c * 313 + 65503) as u16).collect());
                    let mut output = vec![[173; C]; count + 2];
                    let end = merge_samples(std::array::from_fn(|c| &bands[c][offset..]), &mut output[1..count + 1]);
                    assert_eq!(end, count / 16 * 16);
                    for i in 0..end {
                        assert_eq!(output[i + 1], std::array::from_fn(|c| bands[c][offset + i]));
                    }
                    assert_eq!(output[0], [173; C]);
                    assert!(output[end + 1..].iter().all(|pixel| *pixel == [173; C]));
                }
            }
        }
        check::<1>();
        check::<3>();
        check::<4>();
    }

    fn packed_roundtrip<const C: usize>() {
        let source: Vec<_> = (0..16 * C + 1).map(|i| (i * 719 + 65503) as u16).collect();
        let channels = load_samples::<C>(&source[1..]);
        for c in 0..C {
            assert_eq!(channels[c].to_array(), std::array::from_fn(|i| source[1 + i * C + c]));
        }
        let mut output = vec![173; 16 * C + 2];
        store_samples(&mut output[1..1 + 16 * C], channels);
        assert_eq!(&output[1..1 + 16 * C], &source[1..]);
        assert_eq!((output[0], output[16 * C + 1]), (173, 173));
    }

    #[test]
    fn sample_channels_preserve_unaligned_values_and_guards() {
        packed_roundtrip::<1>();
        packed_roundtrip::<3>();
        packed_roundtrip::<4>();
    }

    #[test]
    fn sample_byte_channels_preserve_endianness_unaligned_values_and_guards() {
        fn check<const C: usize>() {
            for offset in 0..4 {
                let source: Vec<u8> = (0..32 * C + offset).map(|i| (i * 137 + i / 7) as u8).collect();
                let channels = load_sample_bytes::<C>(&source[offset..]);
                for (c, channel) in channels.iter().enumerate() {
                    for (i, &value) in channel.as_array().iter().enumerate() {
                        let start = offset + (i * C + c) * 2;
                        assert_eq!(value, u16::from_le_bytes([source[start], source[start + 1]]));
                    }
                }
                let mut output = vec![173; offset + 32 * C + 3];
                store_sample_bytes(&mut output[offset..offset + 32 * C], channels);
                assert_eq!(&output[offset..offset + 32 * C], &source[offset..]);
                assert!(output[..offset].iter().all(|&value| value == 173));
                assert_eq!(&output[offset + 32 * C..], &[173; 3]);
            }
        }
        check::<1>();
        check::<3>();
        check::<4>();
    }
}

#[inline]
pub fn load_samples<const C: usize>(source: &[u16]) -> [Samples; C] {
    assert!(source.len() >= 16 * C);
    let mut result = [Samples::splat(0); C];
    match C {
        1 => result[0] = Samples::from_slice(source),
        3 => result.copy_from_slice(&load_rgb(source)),
        4 => {
            let (a, b) = Samples::from_slice(source).deinterleave(Samples::from_slice(&source[16..]));
            let (c, d) = Samples::from_slice(&source[32..]).deinterleave(Samples::from_slice(&source[48..]));
            let (r, blue) = a.deinterleave(c);
            let (g, alpha) = b.deinterleave(d);
            result.copy_from_slice(&[r, g, blue, alpha]);
        }
        _ => return std::array::from_fn(|c| Samples::gather_or_default(source, Simd::from_array(std::array::from_fn(|i| i * C + c)))),
    }
    result
}

#[inline]
pub fn store_samples<const C: usize>(output: &mut [u16], values: [Samples; C]) {
    assert!(output.len() >= 16 * C);
    match C {
        1 => values[0].copy_to_slice(&mut output[..16]),
        3 => store_rgb(output, [values[0], values[1], values[2]]),
        4 => {
            let (a, c) = values[0].interleave(values[2]);
            let (b, d) = values[1].interleave(values[3]);
            let (ab, ba) = a.interleave(b);
            let (cd, dc) = c.interleave(d);
            for (chunk, value) in output[..64].as_chunks_mut::<16>().0.iter_mut().zip([ab, ba, cd, dc]) {
                value.copy_to_slice(chunk);
            }
        }
        _ => {
            for (c, value) in values.into_iter().enumerate() {
                value.scatter(output, Simd::from_array(std::array::from_fn(|i| i * C + c)));
            }
        }
    }
}

#[inline]
pub fn store_gray_rgb(output: &mut [u8], gray: Bytes) {
    simd_swizzle!(gray, [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5]).copy_to_slice(&mut output[..16]);
    simd_swizzle!(gray, [5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10]).copy_to_slice(&mut output[16..32]);
    simd_swizzle!(gray, [10, 11, 11, 11, 12, 12, 12, 13, 13, 13, 14, 14, 14, 15, 15, 15]).copy_to_slice(&mut output[32..48]);
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
        ((sum * Simd::splat(BOX_3X3_RECIPROCAL)) >> BOX_3X3_RECIPROCAL_BITS).cast::<u8>().copy_to_slice(&mut dst[i..i + 16]);
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

pub fn merge_samples<const C: usize>(bands: [&[u16]; C], dst: &mut [[u16; C]]) -> usize {
    let end = dst.len() / 16 * 16;
    for i in (0..end).step_by(16) {
        store_samples(dst[i..i + 16].as_flattened_mut(), bands.map(|band| Samples::from_slice(&band[i..i + 16])));
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

pub fn paste_nontransparent_rgba(dst: &mut [u8], src: &[u8]) {
    let end = dst.len() / 64 * 64;
    for i in (0..end).step_by(64) {
        let source = load::<4>(&src[i..]);
        let output = load::<4>(&dst[i..]);
        let opaque = source[3].simd_ne(Bytes::splat(0));
        store(&mut dst[i..], std::array::from_fn::<_, 4, _>(|c| opaque.select(source[c], output[c])));
    }
    for (src, dst) in src[end..].as_chunks::<4>().0.iter().zip(dst[end..].as_chunks_mut::<4>().0) {
        if src[3] != 0 {
            *dst = *src;
        }
    }
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
        let rgb_a =
            if fill && C == 4 && M == 1 { (d[C - 1].simd_eq(Bytes::splat(0)) & a.simd_ne(Bytes::splat(0))).select(Bytes::splat(255), a) } else { a };
        store(&mut dst[i * C..], std::array::from_fn::<_, C, _>(|c| blend(d[c], s[c], if c == 3 { a } else { rgb_a })));
    }
    end
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nontransparent_paste_handles_unaligned_vectors_tails_and_guards() {
        for count in 0..=65 {
            for offset in 0..4 {
                for alpha in [Some(0), Some(255), None] {
                    let length = count * 4;
                    let mut src: Vec<_> = (0..offset + length).map(|i| (i * 37) as u8).collect();
                    for (i, pixel) in src[offset..].as_chunks_mut::<4>().0.iter_mut().enumerate() {
                        pixel[3] = alpha.unwrap_or([0, 1, 127, 255][i % 4]);
                    }
                    let mut dst: Vec<_> = (0..length + 2).map(|i| (i * 53) as u8).collect();
                    let mut expected = dst.clone();
                    for (src, dst) in src[offset..].as_chunks::<4>().0.iter().zip(expected[1..length + 1].as_chunks_mut::<4>().0) {
                        if src[3] != 0 {
                            *dst = *src;
                        }
                    }
                    paste_nontransparent_rgba(&mut dst[1..length + 1], &src[offset..]);
                    assert_eq!(dst, expected);
                }
            }
        }
    }

    #[test]
    fn bilevel_packing_preserves_each_bit_in_wide_masks() {
        for bit in 0..64 {
            let mut source = [0; 64];
            source[bit] = 173;
            let mut output = [0; 8];
            pack_bilevel::<false>(&source, &mut output);
            let mut expected = [0; 8];
            expected[bit / 8] = 128 >> (bit % 8);
            assert_eq!(output, expected);
            pack_bilevel::<true>(&source, &mut output);
            assert_eq!(output, expected.map(|value| !value));
        }
    }

    #[test]
    fn bilevel_packing_preserves_polarity_padding_unaligned_slices_and_guards() {
        fn check<const INVERT: bool>() {
            for count in 0..=257 {
                for offset in 0..16 {
                    for fill in [None, Some(0), Some(1), Some(255)] {
                        let source: Vec<_> = (0..offset + count).map(|i| fill.unwrap_or((i * 37) as u8)).collect();
                        let source = &source[offset..];
                        let length = count.div_ceil(8);
                        let mut output = vec![0xa5; length + 2];
                        pack_bilevel::<INVERT>(source, &mut output[1..length + 1]);
                        for (group, &byte) in source.chunks(8).zip(&output[1..length + 1]) {
                            let expected = group.iter().enumerate().fold(0, |bits, (i, &value)| bits | (u8::from((value != 0) != INVERT) << (7 - i)));
                            assert_eq!(byte, expected);
                        }
                        assert_eq!(output[0], 0xa5);
                        assert_eq!(output[length + 1], 0xa5);
                    }
                }
            }
        }
        check::<false>();
        check::<true>();
    }

    #[test]
    fn packed_layouts_match_scalar_at_every_alignment() {
        fn check<const C: usize>() {
            for offset in 0..16 {
                let source: Vec<u8> = (0..offset + 16 * C).map(|i| (i * 37 + i / 7) as u8).collect();
                let channels = load::<C>(&source[offset..]);
                for (c, channel) in channels.iter().enumerate() {
                    assert_eq!(channel.to_array(), std::array::from_fn(|i| source[offset + i * C + c]));
                }
                let mut output = vec![93; offset + 16 * C + 16];
                store(&mut output[offset..offset + 16 * C], channels);
                assert_eq!(&output[offset..offset + 16 * C], &source[offset..]);
                assert!(output[..offset].iter().chain(&output[offset + 16 * C..]).all(|&v| v == 93));
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
