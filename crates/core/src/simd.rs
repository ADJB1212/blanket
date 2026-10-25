use std::simd::cmp::{SimdOrd, SimdPartialOrd};
use std::simd::num::{SimdFloat, SimdInt, SimdUint};
use std::simd::{Simd, ToBytes};

use crate::color::{LUMA_BITS, LUMA_BLUE, LUMA_GREEN, LUMA_RED, LUMA_ROUND};
use crate::parallel::{CHUNK_PIXELS, chunks_mut};
use crate::pixels::{Bytes, load, load_u16, store};
use crate::raster::PixelMode;

pub(crate) fn samples_in_range(source: &[u8], bit_depth: u8) -> bool {
    let maximum = ((1_u32 << bit_depth) - 1) as u16;
    let (blocks, tail) = source.as_chunks::<32>();
    blocks.iter().all(|block| load_u16::<false>(block).simd_le(Simd::splat(maximum)).all())
        && tail.as_chunks::<2>().0.iter().all(|&bytes| u16::from_le_bytes(bytes) <= maximum)
}

pub fn samples_to_be16(source: &[u8], output: &mut [u8], depth: u8) {
    fn convert<const DEPTH: u16>(source: &[u8], output: &mut [u8]) {
        crate::parallel::chunks_mut_above(output, 128 * 1024, 4 * 1024 * 1024, |chunk, dst| {
            let src = &source[chunk * 128 * 1024..][..dst.len()];
            let (src_groups, src_tail) = src.as_chunks::<32>();
            let (dst_groups, dst_tail) = dst.as_chunks_mut::<32>();
            for (src, dst) in src_groups.iter().zip(dst_groups) {
                let samples = load_u16::<false>(src);
                let scaled = if DEPTH == 16 {
                    samples
                } else {
                    let shift = 16 - DEPTH;
                    let numerator = samples * Simd::splat((1 << shift) - 1) + Simd::splat(((1 << DEPTH) - 1) / 2);
                    // Exact division by 2^DEPTH - 1 for this reduced numerator.
                    let correction = (numerator + Simd::splat(1) + (numerator >> DEPTH)) >> DEPTH;
                    (samples << shift) + correction
                };
                scaled.to_be_bytes().copy_to_slice(dst);
            }
            let maximum = (1_u32 << DEPTH) - 1;
            for (src, dst) in src_tail.as_chunks::<2>().0.iter().zip(dst_tail.as_chunks_mut::<2>().0) {
                let sample = (u32::from(u16::from_le_bytes(*src)) * 65535 + maximum / 2) / maximum;
                *dst = (sample as u16).to_be_bytes();
            }
        });
    }
    match depth {
        10 => convert::<10>(source, output),
        12 => convert::<12>(source, output),
        16 => convert::<16>(source, output),
        _ => unreachable!("wide codecs require 10-, 12-, or 16-bit samples"),
    }
}

pub fn rescale_samples(source: &[u8], source_depth: u8, target_depth: u8) -> Vec<u8> {
    fn convert<const SOURCE_MAX: u32>(source: &[u8], target_depth: u8) -> Vec<u8> {
        let source_bytes = if SOURCE_MAX == u32::from(u8::MAX) { 1 } else { 2 };
        let target_bytes = if target_depth == 8 { 1 } else { 2 };
        let target_max = (1_u32 << target_depth) - 1;
        let mut output = vec![0; source.len() / source_bytes * target_bytes];
        chunks_mut(&mut output, CHUNK_PIXELS * target_bytes, |chunk, dst| {
            let src = &source[chunk * CHUNK_PIXELS * source_bytes..][..dst.len() / target_bytes * source_bytes];
            rescale_samples_chunk::<SOURCE_MAX>(src, dst, target_max, target_bytes);
        });
        output
    }
    match source_depth {
        8 => convert::<255>(source, target_depth),
        10 => convert::<1023>(source, target_depth),
        12 => convert::<4095>(source, target_depth),
        16 => convert::<65535>(source, target_depth),
        _ => unreachable!("validated sample depth"),
    }
}

fn rescale_samples_chunk<const SOURCE_MAX: u32>(src: &[u8], dst: &mut [u8], target_max: u32, target_bytes: usize) {
    let source_bytes = if SOURCE_MAX == u32::from(u8::MAX) { 1 } else { 2 };
    let end = dst.len() / target_bytes / 16 * 16;
    for i in (0..end).step_by(16) {
        let input = &src[i * source_bytes..];
        let values = if source_bytes == 1 { Bytes::from_slice(input).cast::<u32>() } else { load_u16::<false>(input).cast::<u32>() };
        let values = (values * Simd::splat(target_max) + Simd::splat(SOURCE_MAX / 2)) / Simd::splat(SOURCE_MAX);
        if target_bytes == 1 {
            values.cast::<u8>().copy_to_slice(&mut dst[i..i + 16]);
        } else {
            values.cast::<u16>().to_le_bytes().copy_to_slice(&mut dst[i * 2..(i + 16) * 2]);
        }
    }
    for (input, output) in src[end * source_bytes..].chunks_exact(source_bytes).zip(dst[end * target_bytes..].chunks_exact_mut(target_bytes)) {
        let value = if source_bytes == 1 { u32::from(input[0]) } else { u32::from(u16::from_le_bytes(input.try_into().unwrap())) };
        let value = ((value * target_max + SOURCE_MAX / 2) / SOURCE_MAX) as u16;
        if target_bytes == 1 {
            output[0] = value as u8;
        } else {
            output.copy_from_slice(&value.to_le_bytes());
        }
    }
}

pub fn gray_to_integer(source: &[u8], mode: PixelMode) -> Vec<u8> {
    let stride = mode.sample_bytes();
    let mut output = vec![0; source.len() * stride];
    chunks_mut(&mut output, CHUNK_PIXELS * stride, |chunk, dst| {
        let src = &source[chunk * CHUNK_PIXELS..][..dst.len() / stride];
        gray_to_integer_chunk(src, dst, mode);
    });
    output
}

fn gray_to_integer_chunk(src: &[u8], dst: &mut [u8], mode: PixelMode) {
    let stride = mode.sample_bytes();
    let end = src.len() / 16 * 16;
    for i in (0..end).step_by(16) {
        let values = Bytes::from_slice(&src[i..]);
        if mode == PixelMode::I {
            values.cast::<i32>().to_le_bytes().copy_to_slice(&mut dst[i * 4..(i + 16) * 4]);
        } else {
            let values = values.cast::<u16>();
            let bytes = if mode == PixelMode::I16B { values.to_be_bytes() } else { values.to_le_bytes() };
            bytes.copy_to_slice(&mut dst[i * 2..(i + 16) * 2]);
        }
    }
    for (&value, pixel) in src[end..].iter().zip(dst[end * stride..].chunks_exact_mut(stride)) {
        if mode == PixelMode::I {
            pixel.copy_from_slice(&i32::from(value).to_le_bytes());
        } else {
            let value = u16::from(value);
            pixel.copy_from_slice(&if mode == PixelMode::I16B { value.to_be_bytes() } else { value.to_le_bytes() });
        }
    }
}

pub fn integer_to_float(source: &[u8], mode: PixelMode) -> Vec<u8> {
    let stride = mode.sample_bytes();
    let mut output = vec![0; source.len() / stride * 4];
    chunks_mut(&mut output, CHUNK_PIXELS * 4, |chunk, dst| {
        let src = &source[chunk * CHUNK_PIXELS * stride..];
        integer_to_float_chunk(src, dst, mode);
    });
    output
}

fn integer_to_float_chunk(src: &[u8], dst: &mut [u8], mode: PixelMode) {
    let stride = mode.sample_bytes();
    let end = dst.len() / 4 / 16 * 16;
    for i in (0..end).step_by(16) {
        let input = &src[i * stride..];
        let values = match mode {
            PixelMode::I => Simd::<i32, 16>::from_le_bytes(Simd::from_slice(input)).cast::<f32>(),
            PixelMode::I16B => load_u16::<true>(input).cast::<f32>(),
            _ => load_u16::<false>(input).cast::<f32>(),
        };
        values.to_le_bytes().copy_to_slice(&mut dst[i * 4..(i + 16) * 4]);
    }
    for (bytes, pixel) in src[end * stride..].chunks_exact(stride).zip(dst[end * 4..].as_chunks_mut::<4>().0) {
        let value = match mode {
            PixelMode::I => i32::from_le_bytes(bytes.try_into().unwrap()),
            PixelMode::I16B => i32::from(u16::from_be_bytes(bytes.try_into().unwrap())),
            _ => i32::from(u16::from_le_bytes(bytes.try_into().unwrap())),
        };
        *pixel = (value as f32).to_le_bytes();
    }
}

pub fn gray_to_float(source: &[u8]) -> Vec<u8> {
    let (blocks, tail) = source.as_chunks::<16>();
    let mut output = Vec::with_capacity(blocks.len() + usize::from(!tail.is_empty()));
    output.extend(blocks.iter().map(|block| {
        let mut pixels = [0; 64];
        gray_to_float_chunk(block, &mut pixels);
        pixels
    }));
    let mut output = output.into_flattened();
    output.extend(tail.iter().flat_map(|&value| f32::from(value).to_le_bytes()));
    output
}

fn gray_to_float_chunk(src: &[u8], dst: &mut [u8]) {
    let end = src.len() / 16 * 16;
    for i in (0..end).step_by(16) {
        Bytes::from_slice(&src[i..]).cast::<f32>().to_le_bytes().copy_to_slice(&mut dst[i * 4..(i + 16) * 4]);
    }
    for (&value, pixel) in src[end..].iter().zip(dst[end * 4..].as_chunks_mut::<4>().0) {
        *pixel = f32::from(value).to_le_bytes();
    }
}

pub fn float_to_u16(source: &[u8], big_endian: bool) -> Vec<u8> {
    let mut output = vec![0; source.len() / 4 * 2];
    chunks_mut(&mut output, CHUNK_PIXELS * 2, |chunk, dst| {
        let src = &source[chunk * CHUNK_PIXELS * 4..][..dst.len() * 2];
        float_to_u16_chunk(src, dst, big_endian);
    });
    output
}

fn float_to_u16_chunk(src: &[u8], dst: &mut [u8], big_endian: bool) {
    let end = dst.len() / 2 / 16 * 16;
    for i in (0..end).step_by(16) {
        let values = Simd::<f32, 16>::from_le_bytes(Simd::from_slice(&src[i * 4..])).cast::<u16>();
        let bytes = if big_endian { values.to_be_bytes() } else { values.to_le_bytes() };
        bytes.copy_to_slice(&mut dst[i * 2..(i + 16) * 2]);
    }
    for (bytes, pixel) in src[end * 4..].as_chunks::<4>().0.iter().zip(dst[end * 2..].as_chunks_mut::<2>().0) {
        let value = f32::from_le_bytes(*bytes) as u16;
        *pixel = if big_endian { value.to_be_bytes() } else { value.to_le_bytes() };
    }
}

pub fn float_to_l(source: &[u8]) -> Vec<u8> {
    let mut output = vec![0; source.len() / 4];
    chunks_mut(&mut output, 64 << 10, |chunk, dst| {
        float_to_l_chunk(&source[chunk * (64 << 10) * 4..][..dst.len() * 4], dst);
    });
    output
}

#[inline]
fn float_to_l_chunk(source: &[u8], dst: &mut [u8]) {
    let src = source.as_chunks::<4>().0;
    let end = dst.len() / 16 * 16;
    for i in (0..end).step_by(16) {
        let values = Simd::<f32, 16>::from_array(std::array::from_fn(|lane| f32::from_le_bytes(src[i + lane])));
        #[cfg(target_arch = "aarch64")]
        let values = values.cast::<u8>();
        #[cfg(not(target_arch = "aarch64"))]
        let values = narrow_integer_l(values.cast::<i32>());
        values.copy_to_slice(&mut dst[i..i + 16]);
    }
    for (dst, src) in dst[end..].iter_mut().zip(&src[end..]) {
        *dst = f32::from_le_bytes(*src) as u8;
    }
}

pub fn float_to_integer(source: &[u8]) -> Vec<u8> {
    let mut output = vec![0; source.len()];
    chunks_mut(&mut output, CHUNK_PIXELS << 2, |chunk, dst| {
        let src = &source[chunk * CHUNK_PIXELS * 4..][..dst.len()];
        let mut offset = 0;
        while offset + 16 <= dst.len() {
            let values =
                Simd::<f32, 4>::from_array(std::array::from_fn(|i| f32::from_le_bytes(src[offset + i * 4..offset + i * 4 + 4].try_into().unwrap())));
            for (dst, value) in dst[offset..offset + 16].as_chunks_mut::<4>().0.iter_mut().zip(values.cast::<i32>().to_array()) {
                *dst = value.to_le_bytes();
            }
            offset += 16;
        }
        for (src, dst) in src[offset..].as_chunks::<4>().0.iter().zip(dst[offset..].as_chunks_mut::<4>().0) {
            *dst = (f32::from_le_bytes(*src) as i32).to_le_bytes();
        }
    });
    output
}

fn narrow_integer_l<const N: usize>(values: Simd<i32, N>) -> Simd<u8, N> {
    // Staged narrowing lets LLVM use saturating SIMD packs.
    #[cfg(not(target_arch = "aarch64"))]
    return values.simd_clamp(Simd::splat(-32768), Simd::splat(32767)).cast::<i16>().simd_clamp(Simd::splat(0), Simd::splat(255)).cast::<u8>();
    #[cfg(target_arch = "aarch64")]
    values.simd_clamp(Simd::splat(0), Simd::splat(65535)).cast::<u16>().simd_min(Simd::splat(255)).cast::<u8>()
}

pub fn integer_to_l(source: &[u8], mode: PixelMode) -> Vec<u8> {
    if mode == PixelMode::I && source.len() / 4 < 2 << 20 {
        let (blocks, tail) = source.as_chunks::<256>();
        let mut output = Vec::with_capacity(blocks.len() + usize::from(!tail.is_empty()));
        output.extend(blocks.iter().map(|block| {
            narrow_integer_l(Simd::<i32, 64>::from_array(std::array::from_fn(|lane| {
                i32::from_le_bytes(block[lane * 4..lane * 4 + 4].try_into().unwrap())
            })))
            .to_array()
        }));
        let mut output = output.into_flattened();
        output.extend(tail.as_chunks::<4>().0.iter().map(|&v| i32::from_le_bytes(v).clamp(0, 255) as u8));
        return output;
    }
    let stride = mode.sample_bytes();
    let mut output = vec![0; source.len() / stride];
    crate::parallel::chunks_mut_above(&mut output, CHUNK_PIXELS, 2 << 20, |chunk, dst| {
        let src = &source[chunk * CHUNK_PIXELS * stride..(chunk * CHUNK_PIXELS + dst.len()) * stride];
        match mode {
            PixelMode::I => {
                let src = src.as_chunks::<4>().0;
                let end = dst.len() / 16 * 16;
                for i in (0..end).step_by(16) {
                    let values = Simd::<i32, 16>::from_array(std::array::from_fn(|lane| i32::from_le_bytes(src[i + lane])));
                    narrow_integer_l(values).copy_to_slice(&mut dst[i..i + 16]);
                }
                for (pixel, bytes) in dst[end..].iter_mut().zip(&src[end..]) {
                    *pixel = i32::from_le_bytes(*bytes).clamp(0, 255) as u8;
                }
            }
            PixelMode::I16B => {
                for (pixel, bytes) in dst.iter_mut().zip(src.as_chunks::<2>().0) {
                    *pixel = u16::from_be_bytes(*bytes).min(255) as u8;
                }
            }
            _ => {
                for (pixel, bytes) in dst.iter_mut().zip(src.as_chunks::<2>().0) {
                    *pixel = u16::from_le_bytes(*bytes).min(255) as u8;
                }
            }
        }
    });
    output
}

pub fn integer_to_color(source: &[u8], mode: PixelMode, channels: usize) -> Vec<u8> {
    match channels {
        3 => integer_to_color_pixels::<3>(source, mode),
        4 => integer_to_color_pixels::<4>(source, mode),
        _ => unreachable!("RGB or RGBA destination"),
    }
}

fn integer_to_color_pixels<const C: usize>(source: &[u8], mode: PixelMode) -> Vec<u8> {
    let stride = mode.sample_bytes();
    if C == 3 && mode == PixelMode::I && source.len() / 4 * 3 < 2 << 20 {
        let (blocks, tail) = source.as_chunks::<64>();
        let mut output = Vec::with_capacity(blocks.len() + usize::from(!tail.is_empty()));
        output.extend(blocks.iter().map(|block| {
            let values = Simd::<i32, 16>::from_array(std::array::from_fn(|i| i32::from_le_bytes(block[i * 4..i * 4 + 4].try_into().unwrap())));
            let mut pixels = [0; 48];
            crate::pixels::store_gray_rgb(&mut pixels, narrow_integer_l(values));
            pixels
        }));
        let mut output = output.into_flattened();
        output.extend(tail.as_chunks::<4>().0.iter().flat_map(|&v| [i32::from_le_bytes(v).clamp(0, 255) as u8; 3]));
        return output;
    }
    if C == 4 && source.len() / stride * C < 4 << 20 {
        return match mode {
            PixelMode::I => integer_to_color_serial::<C, 4>(source.as_chunks::<4>().0, i32::from_le_bytes),
            PixelMode::I16B => integer_to_color_serial::<C, 2>(source.as_chunks::<2>().0, |v| i32::from(u16::from_be_bytes(v))),
            _ => integer_to_color_serial::<C, 2>(source.as_chunks::<2>().0, |v| i32::from(u16::from_le_bytes(v))),
        };
    }
    let mut output = vec![0; source.len() / stride * C];
    const CHUNK: usize = 64 << 10;
    crate::parallel::chunks_mut_above(&mut output, CHUNK * C, 256 << 10, |chunk, dst| {
        let start = chunk * CHUNK * stride;
        let src = &source[start..start + dst.len() / C * stride];
        integer_to_color_chunk::<C>(src, dst, mode);
    });
    output
}

#[inline]
fn integer_to_color_serial<const C: usize, const S: usize>(source: &[[u8; S]], sample: impl Fn([u8; S]) -> i32) -> Vec<u8> {
    source
        .iter()
        .map(|&input| {
            let value = sample(input).clamp(0, 255) as u8;
            std::array::from_fn(|c| if c == 3 { 255 } else { value })
        })
        .collect::<Vec<[u8; C]>>()
        .into_flattened()
}

#[inline]
fn integer_to_color_chunk<const C: usize>(src: &[u8], dst: &mut [u8], mode: PixelMode) {
    match mode {
        PixelMode::I => integer_to_color_samples::<C, 4>(src.as_chunks::<4>().0, dst, i32::from_le_bytes),
        PixelMode::I16B => integer_to_color_samples::<C, 2>(src.as_chunks::<2>().0, dst, |v| i32::from(u16::from_be_bytes(v))),
        _ => integer_to_color_samples::<C, 2>(src.as_chunks::<2>().0, dst, |v| i32::from(u16::from_le_bytes(v))),
    }
}

#[inline]
fn integer_to_color_samples<const C: usize, const S: usize>(src: &[[u8; S]], dst: &mut [u8], sample: impl Fn([u8; S]) -> i32) {
    let end = dst.len() / C / 16 * 16;
    for i in (0..end).step_by(16) {
        let input: &[[u8; S]; 16] = src[i..i + 16].try_into().unwrap();
        let values = Simd::<i32, 16>::from_array(input.map(&sample)).simd_clamp(Simd::splat(0), Simd::splat(255)).cast::<u8>();
        if C == 3 {
            crate::pixels::store_gray_rgb(&mut dst[i * C..], values);
        } else {
            store::<C>(&mut dst[i * C..], std::array::from_fn(|c| if c == 3 { Bytes::splat(255) } else { values }));
        }
    }
    for (pixel, &input) in dst[end * C..].as_chunks_mut::<C>().0.iter_mut().zip(&src[end..]) {
        let value = sample(input).clamp(0, 255) as u8;
        *pixel = std::array::from_fn(|c| if c == 3 { 255 } else { value });
    }
}

pub fn convert_integer(source: &[u8], from: PixelMode, to: PixelMode) -> Vec<u8> {
    if to == PixelMode::I && from != PixelMode::I {
        return if from == PixelMode::I16B { widen_integer::<true>(source) } else { widen_integer::<false>(source) };
    }
    let source_bytes = from.sample_bytes();
    let target_bytes = to.sample_bytes();
    let mut output = vec![0; source.len() / source_bytes * target_bytes];
    chunks_mut(&mut output, CHUNK_PIXELS * target_bytes, |chunk, dst| {
        let src = &source[chunk * CHUNK_PIXELS * source_bytes..(chunk * CHUNK_PIXELS + dst.len() / target_bytes) * source_bytes];
        let offset = dst.len() / target_bytes / 8 * 8;
        for i in (0..offset).step_by(8) {
            let values = Simd::<i32, 8>::from_array(std::array::from_fn(|lane| {
                let input = &src[(i + lane) * source_bytes..(i + lane + 1) * source_bytes];
                match from {
                    PixelMode::I => i32::from_le_bytes(input.try_into().unwrap()),
                    PixelMode::I16B => i32::from(u16::from_be_bytes(input.try_into().unwrap())),
                    _ => i32::from(u16::from_le_bytes(input.try_into().unwrap())),
                }
            }));
            if to == PixelMode::I {
                for (dst, value) in dst[i * 4..(i + 8) * 4].as_chunks_mut::<4>().0.iter_mut().zip(values.to_array()) {
                    *dst = value.to_le_bytes();
                }
            } else {
                let values = values.simd_clamp(Simd::splat(0), Simd::splat(65535)).cast::<u16>();
                for (dst, value) in dst[i * 2..(i + 8) * 2].as_chunks_mut::<2>().0.iter_mut().zip(values.to_array()) {
                    *dst = if to == PixelMode::I16B { value.to_be_bytes() } else { value.to_le_bytes() };
                }
            }
        }
        for (input, output) in
            src[offset * source_bytes..].chunks_exact(source_bytes).zip(dst[offset * target_bytes..].chunks_exact_mut(target_bytes))
        {
            let value = match from {
                PixelMode::I => i64::from(i32::from_le_bytes(input.try_into().unwrap())),
                PixelMode::I16B => i64::from(u16::from_be_bytes(input.try_into().unwrap())),
                _ => i64::from(u16::from_le_bytes(input.try_into().unwrap())),
            };
            if to == PixelMode::I {
                output.copy_from_slice(&(value as i32).to_le_bytes());
            } else {
                let value = value.clamp(0, 65535) as u16;
                let bytes = if to == PixelMode::I16B { value.to_be_bytes() } else { value.to_le_bytes() };
                output.copy_from_slice(&bytes);
            }
        }
    });
    output
}

fn widen_integer<const BIG_ENDIAN: bool>(source: &[u8]) -> Vec<u8> {
    let sample = |bytes| {
        if BIG_ENDIAN { u16::from_be_bytes(bytes) } else { u16::from_le_bytes(bytes) }
    };
    let (blocks, tail) = source.as_chunks::<32>();
    let mut output = Vec::with_capacity(blocks.len() + usize::from(!tail.is_empty()));
    output.extend(blocks.iter().map(|block| {
        let values = Simd::<u16, 16>::from_array(std::array::from_fn(|i| sample([block[i * 2], block[i * 2 + 1]])));
        let bytes: [u8; 64] = values.cast::<u32>().to_array().map(u32::to_le_bytes).as_flattened().try_into().unwrap();
        bytes
    }));
    let mut output = output.into_flattened();
    output.extend(tail.as_chunks::<2>().0.iter().flat_map(|&bytes| u32::from(sample(bytes)).to_le_bytes()));
    output
}

pub fn convert(source: &[u8], from: PixelMode, to: PixelMode) -> Vec<u8> {
    debug_assert_ne!(from, to, "caller should short-circuit identity conversion");

    match (from, to) {
        (PixelMode::Rgba, PixelMode::Rgb) => convert_layout::<4, 3>(source),
        (PixelMode::Rgb, PixelMode::Rgba) => convert_layout::<3, 4>(source),
        (PixelMode::L, PixelMode::Rgb) => convert_layout::<1, 3>(source),
        (PixelMode::L, PixelMode::Rgba) => convert_layout::<1, 4>(source),
        (PixelMode::Rgb, PixelMode::L) => color_to_gray::<3>(source),
        (PixelMode::Rgba, PixelMode::L) => color_to_gray::<4>(source),
        _ => unreachable!("all mode pairs are covered"),
    }
}

fn map_blocks<const S: usize, const D: usize>(
    source: &[u8], parallel_bytes: usize, block: impl Fn([Bytes; S]) -> [Bytes; D] + Sync, pixel: impl Fn(&[u8; S]) -> [u8; D] + Sync,
) -> Vec<u8> {
    let mut output = vec![0; source.len() / S * D];
    let chunk_pixels = 64 << 10;
    crate::parallel::chunks_mut_above(&mut output, chunk_pixels * D, parallel_bytes, |i, dst| {
        let src = &source[i * chunk_pixels * S..(i * chunk_pixels + dst.len() / D) * S];
        let end = src.len() / S / 16 * 16;
        for i in (0..end).step_by(16) {
            store(&mut dst[i * D..], block(load::<S>(&src[i * S..])));
        }
        for (src, dst) in src[end * S..].as_chunks::<S>().0.iter().zip(dst[end * D..].as_chunks_mut::<D>().0) {
            *dst = pixel(src);
        }
    });
    output
}

pub fn gray_to_hsv<const C: usize>(source: &[u8]) -> Vec<u8> {
    map_blocks::<C, 3>(source, 5 << 20, |p| [Bytes::splat(0), Bytes::splat(0), p[0]], |p| [0, 0, p[0]])
}

/// Pillow's L to CMYK stores inverted gray as K.
pub fn gray_to_cmyk(source: &[u8]) -> Vec<u8> {
    let (blocks, tail) = source.as_chunks::<16>();
    let mut output = Vec::with_capacity(blocks.len() + usize::from(!tail.is_empty()));
    output.extend(blocks.iter().map(|block| ((!Bytes::from_array(*block)).cast::<u32>() << 24).to_array().map(u32::to_le_bytes)));
    let mut output = output.into_flattened();
    output.extend(tail.iter().map(|&v| [0, 0, 0, 255 - v]));
    output.into_flattened()
}

pub fn gray_to_ycbcr(source: &[u8]) -> Vec<u8> {
    map_blocks::<1, 3>(source, 4 << 20, |p| [p[0], Bytes::splat(128), Bytes::splat(128)], |p| [p[0], 128, 128])
}

pub fn color_to_cmyk<const S: usize>(source: &[u8]) -> Vec<u8> {
    if S == 4 {
        // Packed little-endian RGB occupies the low 24 bits; CMYK clears K.
        const RGB_MASK: u32 = 0x00ff_ffff;
        let (blocks, tail) = source.as_chunks::<64>();
        let mut output = Vec::with_capacity(blocks.len() + usize::from(!tail.is_empty()));
        output.extend(blocks.iter().map(|block| {
            let values = Simd::<u32, 16>::from_le_bytes(Simd::from_array(*block));
            ((!values) & Simd::splat(RGB_MASK)).to_le_bytes().to_array()
        }));
        let mut output = output.into_flattened();
        output.extend(tail.as_chunks::<4>().0.iter().flat_map(|p| [255 - p[0], 255 - p[1], 255 - p[2], 0]));
        return output;
    }
    map_blocks::<S, 4>(source, 4 << 20, |p| [!p[0], !p[1], !p[2], Bytes::splat(0)], |p| [255 - p[0], 255 - p[1], 255 - p[2], 0])
}

pub fn extract_channel<const S: usize>(source: &[u8], channel: usize) -> Vec<u8> {
    fn extract<const S: usize, const BAND: usize>(source: &[u8]) -> Vec<u8> {
        if S == 2 || S == 3 || source.len() / S < 4 << 20 {
            let count = source.len() / S / 16;
            let blocks = &source[..count * S * 16];
            let tail = &source[count * S * 16..];
            let mut output = Vec::with_capacity(count + usize::from(!tail.is_empty()));
            output.extend(blocks.chunks_exact(S * 16).map(|block| load::<S>(block)[BAND].to_array()));
            let mut output = output.into_flattened();
            output.extend(tail.as_chunks::<S>().0.iter().map(|p| p[BAND]));
            return output;
        }
        let mut output = vec![0; source.len() / S];
        const CHUNK: usize = 64 << 10;
        crate::parallel::chunks_mut_above(&mut output, CHUNK, 4 << 20, |i, dst| {
            let src = &source[i * CHUNK * S..(i * CHUNK + dst.len()) * S];
            let end = dst.len() / 16 * 16;
            for offset in (0..end).step_by(16) {
                load::<S>(&src[offset * S..])[BAND].copy_to_slice(&mut dst[offset..offset + 16]);
            }
            for (pixel, value) in dst[end..].iter_mut().zip(src[end * S..].as_chunks::<S>().0) {
                *pixel = value[BAND];
            }
        });
        output
    }
    match channel {
        0 => extract::<S, 0>(source),
        1 => extract::<S, 1>(source),
        2 => extract::<S, 2>(source),
        3 => extract::<S, 3>(source),
        _ => unreachable!("validated channel index"),
    }
}

pub fn convert_la(source: &[u8], to: PixelMode) -> Vec<u8> {
    if to == PixelMode::L {
        return extract_channel::<2>(source, 0);
    }
    if to == PixelMode::Rgb && source.len() / 2 * 3 < 2 << 20 {
        return gray_to_rgb::<2>(source);
    }
    if to == PixelMode::Rgba && source.len() < 2 << 20 {
        let (blocks, tail) = source.as_chunks::<32>();
        let mut output = Vec::with_capacity(blocks.len() + usize::from(!tail.is_empty()));
        output.extend(blocks.iter().map(|block| {
            std::simd::simd_swizzle!(
                Simd::<u8, 32>::from_array(*block),
                [
                    0, 0, 0, 1, 2, 2, 2, 3, 4, 4, 4, 5, 6, 6, 6, 7, 8, 8, 8, 9, 10, 10, 10, 11, 12, 12, 12, 13, 14, 14, 14, 15, 16, 16, 16, 17, 18,
                    18, 18, 19, 20, 20, 20, 21, 22, 22, 22, 23, 24, 24, 24, 25, 26, 26, 26, 27, 28, 28, 28, 29, 30, 30, 30, 31
                ]
            )
            .to_array()
        }));
        let mut output = output.into_flattened();
        output.extend(tail.as_chunks::<2>().0.iter().flat_map(|p| [p[0], p[0], p[0], p[1]]));
        return output;
    }
    let channels = to.channels();
    let mut output = vec![0; source.len() / 2 * channels];
    crate::parallel::chunks_mut_above(&mut output, CHUNK_PIXELS * channels, 2 << 20, |chunk, dst| {
        let start = chunk * CHUNK_PIXELS * 2;
        let src = &source[start..start + dst.len() / channels * 2];
        let offset = src.len() / 2 / 16 * 16;
        for i in (0..offset).step_by(16) {
            let [l, a] = load::<2>(&src[i * 2..]);
            let dst = &mut dst[i * channels..];
            match to {
                PixelMode::L => l.copy_to_slice(&mut dst[..16]),
                PixelMode::Rgb => store(dst, [l; 3]),
                PixelMode::Rgba => store(dst, [l, l, l, a]),
                _ => unreachable!(),
            }
        }
        for (pair, pixel) in src[offset * 2..].as_chunks::<2>().0.iter().zip(dst[offset * channels..].chunks_exact_mut(channels)) {
            match to {
                PixelMode::L => pixel[0] = pair[0],
                PixelMode::Rgb => pixel.copy_from_slice(&[pair[0]; 3]),
                PixelMode::Rgba => pixel.copy_from_slice(&[pair[0], pair[0], pair[0], pair[1]]),
                _ => unreachable!(),
            }
        }
    });
    output
}

pub fn extract_two_channel(source: &[u8], channel: usize) -> Vec<u8> {
    extract_channel::<2>(source, channel)
}

fn convert_layout<const S: usize, const C: usize>(source: &[u8]) -> Vec<u8> {
    if S == 1 && C == 3 && source.len() * 3 < 2 << 20 {
        return gray_to_rgb::<1>(source);
    }
    if C == 4 {
        if S == 3 {
            let (blocks, tail) = source.as_chunks::<48>();
            let mut output = Vec::with_capacity(blocks.len() + usize::from(!tail.is_empty()));
            output.extend(blocks.iter().map(|block| {
                let mut pixels = [0; 64];
                let [r, g, b] = load::<3>(block);
                store(&mut pixels, [r, g, b, Bytes::splat(255)]);
                pixels
            }));
            let mut output = output.into_flattened();
            output.extend(tail.as_chunks::<3>().0.iter().flat_map(|p| [p[0], p[1], p[2], 255]));
            return output;
        }
        return source.iter().map(|&v| ((u32::from(v) * 0x0001_0101) | 0xff00_0000).to_le_bytes()).collect::<Vec<_>>().into_flattened();
    }
    map_blocks::<S, C>(
        source,
        2 << 20,
        |p| std::array::from_fn(|c| if c == 3 { Bytes::splat(255) } else { p[if S == 1 { 0 } else { c }] }),
        |p| std::array::from_fn(|c| if c == 3 { 255 } else { p[if S == 1 { 0 } else { c }] }),
    )
}

fn gray_to_rgb<const S: usize>(source: &[u8]) -> Vec<u8> {
    let pixels = source.as_chunks::<S>().0;
    let (blocks, tail) = pixels.as_chunks::<16>();
    let mut output = Vec::with_capacity(blocks.len() + usize::from(!tail.is_empty()));
    output.extend(blocks.iter().map(|block| {
        let gray = load::<S>(block.as_flattened())[0];
        let mut pixels = [0; 48];
        crate::pixels::store_gray_rgb(&mut pixels, gray);
        pixels
    }));
    let mut output = output.into_flattened();
    output.extend(tail.iter().flat_map(|p| [p[0]; 3]));
    output
}

fn color_to_gray<const SOURCE_CHANNELS: usize>(source: &[u8]) -> Vec<u8> {
    map_blocks::<SOURCE_CHANNELS, 1>(
        source,
        2 << 20,
        |p| {
            let sum = p[0].cast::<u32>() * Simd::splat(LUMA_RED)
                + p[1].cast::<u32>() * Simd::splat(LUMA_GREEN)
                + p[2].cast::<u32>() * Simd::splat(LUMA_BLUE)
                + Simd::splat(LUMA_ROUND);
            [(sum >> LUMA_BITS).cast::<u8>()]
        },
        |p| [pillow_luma(p[0], p[1], p[2])],
    )
}

#[inline(always)]
pub fn pillow_luma(r: u8, g: u8, b: u8) -> u8 {
    ((u32::from(r) * LUMA_RED + u32::from(g) * LUMA_GREEN + u32::from(b) * LUMA_BLUE + LUMA_ROUND) >> LUMA_BITS) as u8
}

/// Fixed divisors let LLVM vectorize exact wide-sample normalization. Preserve
/// the existing rounded full-range mapping, including 16-bit identity input.
pub fn extract_sample_channel(source: &[u8], channels: usize, channel: usize) -> Vec<u8> {
    let mut pixels = Vec::with_capacity(source.len() / channels);
    pixels.extend(source.chunks_exact(channels * 2).flat_map(|pixel| [pixel[channel * 2], pixel[channel * 2 + 1]]));
    pixels
}

pub fn normalize_u16(source: &[u8], depth: u8) -> Vec<u16> {
    use rayon::prelude::*;

    use crate::parallel::should_parallel;

    const CHUNK_PIXELS: usize = 64 << 10;
    const MEMORY_PARALLEL_BYTES: usize = 4 << 20;

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wide_conversion_preserves_rounding_endianness_and_partition_boundaries() {
        for depth in [10, 12, 16] {
            let maximum = (1_u32 << depth) - 1;
            for count in [0, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 65535, 65536, 65537, 2_097_153] {
                let samples: Vec<_> = (0..count).map(|i| (i as u32 * 719 % (maximum + 1)) as u16).collect();
                let mut input = vec![0x5a];
                input.extend(samples.iter().flat_map(|v| v.to_le_bytes()));
                let mut output = vec![0xa5; count * 2 + 2];
                samples_to_be16(&input[1..], &mut output[1..count * 2 + 1], depth);
                for (&sample, bytes) in samples.iter().zip(output[1..count * 2 + 1].as_chunks::<2>().0) {
                    assert_eq!(u32::from(u16::from_be_bytes(*bytes)), (u32::from(sample) * 65535 + maximum / 2) / maximum);
                }
                assert_eq!(output[0], 0xa5);
                assert_eq!(output[count * 2 + 1], 0xa5);
            }
        }
    }

    #[test]
    fn sample_validation_checks_every_lane_unaligned_slice_and_tail() {
        for depth in [10, 12, 16] {
            let maximum = ((1_u32 << depth) - 1) as u16;
            for count in [0, 1, 15, 16, 17, 31, 32, 33, 65] {
                for offset in 0..16 {
                    let mut source = vec![93; offset];
                    source.extend((0..count).flat_map(|i| ((i * 37) as u16 & maximum).to_le_bytes()));
                    assert!(samples_in_range(&source[offset..], depth));
                    if depth == 16 {
                        continue;
                    }
                    for index in 0..count {
                        let start = offset + index * 2;
                        let original = [source[start], source[start + 1]];
                        source[start..start + 2].copy_from_slice(&(maximum + 1).to_le_bytes());
                        assert!(!samples_in_range(&source[offset..], depth));
                        source[start..start + 2].copy_from_slice(&original);
                    }
                }
            }
        }
    }

    #[test]
    fn depth_rescaling_matches_scalar_for_full_ranges_unaligned_slices_and_guards() {
        fn check<const SOURCE_MAX: u32>() {
            let source_bytes = if SOURCE_MAX == 255 { 1 } else { 2 };
            for count in [0, 1, 15, 16, 17, 31, 32, 33, SOURCE_MAX as usize + 1] {
                for offset in 0..16 {
                    let values: Vec<_> = (0..count).map(|i| (i as u32 % (SOURCE_MAX + 1)) as u16).collect();
                    let mut source = vec![93; offset];
                    for &value in &values {
                        source.extend_from_slice(&value.to_le_bytes()[..source_bytes]);
                    }
                    for target_depth in [8, 10, 12, 16] {
                        let target_bytes = if target_depth == 8 { 1 } else { 2 };
                        let target_max = (1 << target_depth) - 1;
                        let mut output = vec![93; offset + count * target_bytes + 16];
                        rescale_samples_chunk::<SOURCE_MAX>(
                            &source[offset..],
                            &mut output[offset..offset + count * target_bytes],
                            target_max,
                            target_bytes,
                        );
                        let expected: Vec<_> = values
                            .iter()
                            .flat_map(|&value| {
                                let value = ((u32::from(value) * target_max + SOURCE_MAX / 2) / SOURCE_MAX) as u16;
                                value.to_le_bytes().into_iter().take(target_bytes)
                            })
                            .collect();
                        assert_eq!(&output[offset..offset + count * target_bytes], expected);
                        assert!(output[..offset].iter().chain(&output[offset + count * target_bytes..]).all(|&v| v == 93));
                    }
                }
            }
        }
        check::<255>();
        check::<1023>();
        check::<4095>();
        check::<65535>();
    }

    #[test]
    fn gray_integer_widening_preserves_byte_order_unaligned_tails_and_guards() {
        for count in [0, 1, 15, 16, 17, 31, 32, 33, 257] {
            for offset in 0..16 {
                let source: Vec<_> = (0..offset + count).map(|i| (i * 37) as u8).collect();
                for mode in [PixelMode::I, PixelMode::I16, PixelMode::I16L, PixelMode::I16B] {
                    let stride = mode.sample_bytes();
                    let mut output = vec![93; offset + count * stride + 16];
                    gray_to_integer_chunk(&source[offset..], &mut output[offset..offset + count * stride], mode);
                    let expected: Vec<_> = source[offset..]
                        .iter()
                        .flat_map(|&value| {
                            if mode == PixelMode::I {
                                i32::from(value).to_le_bytes().to_vec()
                            } else if mode == PixelMode::I16B {
                                u16::from(value).to_be_bytes().to_vec()
                            } else {
                                u16::from(value).to_le_bytes().to_vec()
                            }
                        })
                        .collect();
                    assert_eq!(&output[offset..offset + count * stride], expected);
                    assert!(output[..offset].iter().chain(&output[offset + count * stride..]).all(|&v| v == 93));
                }
            }
        }
    }

    #[test]
    fn gray_integer_widening_preserves_parallel_tail() {
        let source: Vec<_> = (0..(CHUNK_PIXELS * 16 + 17)).map(|i| (i * 37) as u8).collect();
        for mode in [PixelMode::I, PixelMode::I16, PixelMode::I16L, PixelMode::I16B] {
            let output = gray_to_integer(&source, mode);
            assert_eq!(output.len(), source.len() * mode.sample_bytes());
            for (&value, pixel) in source.iter().zip(output.chunks_exact(mode.sample_bytes())) {
                let result = match mode {
                    PixelMode::I => i32::from_le_bytes(pixel.try_into().unwrap()),
                    PixelMode::I16B => i32::from(u16::from_be_bytes(pixel.try_into().unwrap())),
                    _ => i32::from(u16::from_le_bytes(pixel.try_into().unwrap())),
                };
                assert_eq!(result, i32::from(value));
            }
        }
    }

    #[test]
    fn expanding_and_extracting_unaligned_pixels_preserves_channels_and_tails() {
        for count in [0, 1, 3, 4, 5, 15, 16, 17, 31, 32, 33, 257] {
            for offset in 0..16 {
                let source: Vec<u8> = (0..offset + count * 4).map(|i| (i * 37 + 113) as u8).collect();
                let gray = &source[offset..offset + count];
                let rgb = &source[offset..offset + count * 3];
                let rgba = &source[offset..];
                assert_eq!(convert(rgb, PixelMode::Rgb, PixelMode::L), scalar_luminance_rgb(rgb));
                assert_eq!(convert(rgba, PixelMode::Rgba, PixelMode::L), scalar_luminance_rgba(rgba));
                for band in 0..3 {
                    let expected: Vec<_> = rgb.as_chunks::<3>().0.iter().map(|p| p[band]).collect();
                    assert_eq!(extract_channel::<3>(rgb, band), expected);
                }
                for band in 0..4 {
                    let expected: Vec<_> = rgba.as_chunks::<4>().0.iter().map(|p| p[band]).collect();
                    assert_eq!(extract_channel::<4>(rgba, band), expected);
                }
                let expected: Vec<_> = gray.iter().flat_map(|&v| [v, v, v, 255]).collect();
                assert_eq!(convert(gray, PixelMode::L, PixelMode::Rgba), expected);
                let expected: Vec<_> = gray.iter().flat_map(|&v| [v; 3]).collect();
                assert_eq!(convert(gray, PixelMode::L, PixelMode::Rgb), expected);
                let expected: Vec<_> = rgb.as_chunks::<3>().0.iter().flat_map(|p| [p[0], p[1], p[2], 255]).collect();
                assert_eq!(convert(rgb, PixelMode::Rgb, PixelMode::Rgba), expected);
                let expected: Vec<_> = gray.iter().flat_map(|&v| [0, 0, 0, 255 - v]).collect();
                assert_eq!(gray_to_cmyk(gray), expected);
                let expected: Vec<_> = rgba.as_chunks::<4>().0.iter().flat_map(|p| [255 - p[0], 255 - p[1], 255 - p[2], 0]).collect();
                assert_eq!(color_to_cmyk::<4>(rgba), expected);
                let pairs = &source[offset..offset + count * 2];
                let expected: Vec<_> = pairs.as_chunks::<2>().0.iter().flat_map(|p| [p[0], p[0], p[0], p[1]]).collect();
                assert_eq!(convert_la(pairs, PixelMode::Rgba), expected);
                let expected: Vec<_> = pairs.as_chunks::<2>().0.iter().flat_map(|p| [p[0]; 3]).collect();
                assert_eq!(convert_la(pairs, PixelMode::Rgb), expected);
                for band in 0..2 {
                    let expected: Vec<_> = pairs.as_chunks::<2>().0.iter().map(|p| p[band]).collect();
                    assert_eq!(extract_two_channel(pairs, band), expected);
                    if band == 0 {
                        assert_eq!(convert_la(pairs, PixelMode::L), expected);
                    }
                }
            }
        }
    }

    #[test]
    fn extracting_rgba_channels_preserves_parallel_tail() {
        let count = (4 << 20) + 17;
        let source: Vec<_> = (0..count * 4 + 1).map(|i| (i * 37 + i / 11) as u8).collect();
        for band in 0..4 {
            let output = extract_channel::<4>(&source[1..], band);
            assert_eq!(output.len(), count);
            for (pixel, &value) in source[1..].as_chunks::<4>().0.iter().zip(&output) {
                assert_eq!(value, pixel[band]);
            }
        }
    }

    #[test]
    fn numeric_conversion_kernels_preserve_unaligned_tails_and_guards() {
        let floats = [f32::NAN, f32::NEG_INFINITY, f32::INFINITY, -0.0, 0.0, -1.5, 1.75, 254.9, 255.1];
        let integers = [i32::MIN, i32::MAX, -1, 0, 1, 127, 255, 256, 65535];
        for count in [0, 1, 3, 4, 5, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129] {
            for offset in 0..16 {
                let source: Vec<_> = std::iter::repeat_n(93, offset).chain((0..count).flat_map(|i| floats[i % floats.len()].to_le_bytes())).collect();
                let mut output = vec![93; offset + count + 16];
                float_to_l_chunk(&source[offset..], &mut output[offset..offset + count]);
                assert!(output[..offset].iter().all(|&v| v == 93));
                assert!(output[offset + count..].iter().all(|&v| v == 93));
                for i in 0..count {
                    assert_eq!(output[offset + i], floats[i % floats.len()].clamp(0.0, 255.0) as u8);
                }
                for mode in [PixelMode::I, PixelMode::I16, PixelMode::I16L, PixelMode::I16B] {
                    let source: Vec<_> = std::iter::repeat_n(93, offset)
                        .chain((0..count).flat_map(|i| {
                            let v = integers[i % integers.len()];
                            match mode {
                                PixelMode::I => v.to_le_bytes().to_vec(),
                                PixelMode::I16B => (v as u16).to_be_bytes().to_vec(),
                                _ => (v as u16).to_le_bytes().to_vec(),
                            }
                        }))
                        .collect();
                    let gray = integer_to_l(&source[offset..], mode);
                    let expected: Vec<_> = (0..count)
                        .map(|i| {
                            let value = integers[i % integers.len()];
                            if mode == PixelMode::I { value.clamp(0, 255) as u8 } else { (value as u16).min(255) as u8 }
                        })
                        .collect();
                    assert_eq!(gray, expected);
                    let mut output = vec![93; offset + count * 4 + 16];
                    integer_to_float_chunk(&source[offset..], &mut output[offset..offset + count * 4], mode);
                    let expected: Vec<_> = (0..count)
                        .flat_map(|i| {
                            let value = integers[i % integers.len()];
                            let value = if mode == PixelMode::I { value as f32 } else { f32::from(value as u16) };
                            value.to_le_bytes()
                        })
                        .collect();
                    assert_eq!(&output[offset..offset + count * 4], expected);
                    assert!(output[..offset].iter().chain(&output[offset + count * 4..]).all(|&v| v == 93));
                    if mode != PixelMode::I {
                        let expected: Vec<_> = (0..count).flat_map(|i| u32::from(integers[i % integers.len()] as u16).to_le_bytes()).collect();
                        assert_eq!(convert_integer(&source[offset..], mode, PixelMode::I), expected);
                    }
                    for channels in [3, 4] {
                        let mut output = vec![93; offset + count * channels + 16];
                        let dst = &mut output[offset..offset + count * channels];
                        if channels == 3 {
                            integer_to_color_chunk::<3>(&source[offset..], dst, mode);
                        } else {
                            integer_to_color_chunk::<4>(&source[offset..], dst, mode);
                        }
                        assert!(output[..offset].iter().all(|&v| v == 93));
                        assert!(output[offset + count * channels..].iter().all(|&v| v == 93));
                        for i in 0..count {
                            assert_eq!(&output[offset + i * channels..offset + i * channels + 3], &[gray[i]; 3]);
                            if channels == 4 {
                                assert_eq!(output[offset + i * channels + 3], 255);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn float_and_gray_widening_preserves_saturation_alignment_and_guards() {
        let floats = [
            f32::NAN,
            f32::from_bits(0xffc0_0001),
            f32::INFINITY,
            f32::NEG_INFINITY,
            -0.0,
            0.0,
            -1.75,
            1.75,
            255.9,
            256.25,
            65534.9,
            65535.0,
            65536.0,
            f32::MAX,
            f32::MIN,
        ];
        for count in [0, 1, 15, 16, 17, 31, 32, 33, 257] {
            for offset in 0..16 {
                let source: Vec<_> = std::iter::repeat_n(93, offset).chain((0..count).flat_map(|i| floats[i % floats.len()].to_le_bytes())).collect();
                for big_endian in [false, true] {
                    let mut output = vec![93; offset + count * 2 + 16];
                    float_to_u16_chunk(&source[offset..], &mut output[offset..offset + count * 2], big_endian);
                    let expected: Vec<_> = (0..count)
                        .flat_map(|i| {
                            let value = floats[i % floats.len()] as u16;
                            if big_endian { value.to_be_bytes() } else { value.to_le_bytes() }
                        })
                        .collect();
                    assert_eq!(&output[offset..offset + count * 2], expected);
                    assert!(output[..offset].iter().chain(&output[offset + count * 2..]).all(|&v| v == 93));
                }
                let source: Vec<_> = (0..offset + count).map(|i| (i * 37) as u8).collect();
                let mut output = vec![93; offset + count * 4 + 16];
                gray_to_float_chunk(&source[offset..], &mut output[offset..offset + count * 4]);
                let expected: Vec<_> = source[offset..].iter().flat_map(|&value| f32::from(value).to_le_bytes()).collect();
                assert_eq!(&output[offset..offset + count * 4], expected);
                assert!(output[..offset].iter().chain(&output[offset + count * 4..]).all(|&v| v == 93));
            }
        }
    }

    #[test]
    fn integer_narrowing_preserves_parallel_tail() {
        let values = [i32::MIN, -1, 0, 1, 127, 255, 256, 65535, 65536, i32::MAX];
        let count = (2 << 20) + 17;
        let source: Vec<_> = (0..count).flat_map(|i| values[i % values.len()].to_le_bytes()).collect();
        let output = integer_to_l(&source, PixelMode::I);
        for (i, &value) in output.iter().enumerate() {
            assert_eq!(value, values[i % values.len()].clamp(0, 255) as u8);
        }
    }

    #[test]
    fn float_cast_matches_scalar_at_vector_boundaries() {
        let values = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.0, 0.0, 1.75, -1.75, 2147483648.0, -2147483648.0, 2147483520.0];
        for count in [0, 1, 3, 4, 5, 15, 16, 17, 33] {
            let source: Vec<_> = std::iter::once(93).chain((0..count).flat_map(|i| values[i % values.len()].to_le_bytes())).collect();
            let expected: Vec<_> = (0..count).flat_map(|i| (values[i % values.len()] as i32).to_le_bytes()).collect();
            assert_eq!(float_to_integer(&source[1..]), expected);
        }
    }

    #[test]
    fn rgba_to_rgb_basic() {
        let rgba: Vec<u8> = (0..256).flat_map(|i| [i as u8, (i * 2) as u8, (i * 3) as u8, 0xAA]).collect();
        let result = convert(&rgba, PixelMode::Rgba, PixelMode::Rgb);
        for i in 0..256 {
            assert_eq!(result[i * 3], rgba[i * 4], "R mismatch at pixel {i}");
            assert_eq!(result[i * 3 + 1], rgba[i * 4 + 1], "G mismatch at pixel {i}");
            assert_eq!(result[i * 3 + 2], rgba[i * 4 + 2], "B mismatch at pixel {i}");
        }
    }

    #[test]
    fn integer_to_color_matches_gray_expansion() {
        for mode in [PixelMode::I, PixelMode::I16, PixelMode::I16L, PixelMode::I16B] {
            let values = [-10_i32, 0, 1, 127, 255, 256, 65535, 100000];
            for count in [0, 1, 15, 16, 17, 257, (2 << 20) / 3 - 1, (2 << 20) / 3 + 1] {
                let source: Vec<u8> = values
                    .iter()
                    .cycle()
                    .take(count)
                    .flat_map(|&value| match mode {
                        PixelMode::I => value.to_le_bytes().to_vec(),
                        PixelMode::I16B => (value as u16).to_be_bytes().to_vec(),
                        _ => (value as u16).to_le_bytes().to_vec(),
                    })
                    .collect();
                let mut unaligned = vec![0xa5];
                unaligned.extend_from_slice(&source);
                let source = &unaligned[1..];
                let gray = integer_to_l(source, mode);
                for destination in [PixelMode::Rgb, PixelMode::Rgba] {
                    assert_eq!(integer_to_color(source, mode, destination.channels()), convert(&gray, PixelMode::L, destination));
                }
            }
        }
    }

    #[test]
    fn rgb_to_rgba_basic() {
        let rgb: Vec<u8> = (0..256).flat_map(|i| [i as u8, (i * 2) as u8, (i * 3) as u8]).collect();
        let result = convert(&rgb, PixelMode::Rgb, PixelMode::Rgba);
        for i in 0..256 {
            assert_eq!(result[i * 4], rgb[i * 3], "R mismatch at pixel {i}");
            assert_eq!(result[i * 4 + 1], rgb[i * 3 + 1], "G mismatch at pixel {i}");
            assert_eq!(result[i * 4 + 2], rgb[i * 3 + 2], "B mismatch at pixel {i}");
            assert_eq!(result[i * 4 + 3], 255, "A mismatch at pixel {i}");
        }
    }

    #[test]
    fn l_to_rgb_basic() {
        let luma: Vec<u8> = (0..=255).collect();
        let result = convert(&luma, PixelMode::L, PixelMode::Rgb);
        for (i, &l) in luma.iter().enumerate() {
            assert_eq!(result[i * 3], l);
            assert_eq!(result[i * 3 + 1], l);
            assert_eq!(result[i * 3 + 2], l);
        }
    }

    #[test]
    fn l_to_rgba_basic() {
        let luma: Vec<u8> = (0..=255).collect();
        let result = convert(&luma, PixelMode::L, PixelMode::Rgba);
        for (i, &l) in luma.iter().enumerate() {
            assert_eq!(result[i * 4], l);
            assert_eq!(result[i * 4 + 1], l);
            assert_eq!(result[i * 4 + 2], l);
            assert_eq!(result[i * 4 + 3], 255);
        }
    }

    #[test]
    fn rgb_to_l_matches_pillow_luminance() {
        let rgb: Vec<u8> = (0u8..=u8::MAX).flat_map(|value| [value, value.wrapping_mul(37), value.wrapping_add(113)]).collect();
        let result = convert(&rgb, PixelMode::Rgb, PixelMode::L);
        let expected = scalar_luminance_rgb(&rgb);
        assert_eq!(result, expected);
    }

    #[test]
    fn rgba_to_l_matches_pillow_luminance() {
        let rgba: Vec<u8> =
            (0u8..=u8::MAX).flat_map(|value| [value, value.wrapping_mul(37), value.wrapping_add(113), value.wrapping_mul(19)]).collect();
        let result = convert(&rgba, PixelMode::Rgba, PixelMode::L);
        let expected = scalar_luminance_rgba(&rgba);
        assert_eq!(result, expected);
    }

    #[test]
    fn handles_non_simd_aligned_sizes() {
        let rgba: Vec<u8> = (0..68).collect();
        let result = convert(&rgba, PixelMode::Rgba, PixelMode::Rgb);
        assert_eq!(result.len(), 51);
        for i in 0..17 {
            assert_eq!(result[i * 3], rgba[i * 4]);
            assert_eq!(result[i * 3 + 1], rgba[i * 4 + 1]);
            assert_eq!(result[i * 3 + 2], rgba[i * 4 + 2]);
        }
    }

    #[test]
    fn handles_empty_input() {
        assert!(convert(&[], PixelMode::Rgba, PixelMode::Rgb).is_empty());
        assert!(convert(&[], PixelMode::Rgb, PixelMode::L).is_empty());
        assert!(convert(&[], PixelMode::L, PixelMode::Rgba).is_empty());
    }

    #[test]
    fn handles_single_pixel() {
        assert_eq!(convert(&[10, 20, 30, 40], PixelMode::Rgba, PixelMode::Rgb), [10, 20, 30]);
        assert_eq!(convert(&[10, 20, 30], PixelMode::Rgb, PixelMode::Rgba), [10, 20, 30, 255]);
        assert_eq!(convert(&[42], PixelMode::L, PixelMode::Rgb), [42, 42, 42]);
    }

    fn scalar_luminance_rgb(source: &[u8]) -> Vec<u8> {
        source
            .as_chunks::<3>()
            .0
            .iter()
            .map(|px| ((px[0] as u32 * 19_595 + px[1] as u32 * 38_470 + px[2] as u32 * 7_471 + 0x8000) >> 16) as u8)
            .collect()
    }

    fn scalar_luminance_rgba(source: &[u8]) -> Vec<u8> {
        source
            .as_chunks::<4>()
            .0
            .iter()
            .map(|px| ((px[0] as u32 * 19_595 + px[1] as u32 * 38_470 + px[2] as u32 * 7_471 + 0x8000) >> 16) as u8)
            .collect()
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
}
