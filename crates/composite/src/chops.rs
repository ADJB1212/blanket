//! Channel arithmetic and wraparound translation for `ImageChops`.

use std::mem::MaybeUninit;
use std::simd::cmp::{SimdOrd, SimdPartialOrd};
use std::simd::num::{SimdFloat, SimdInt, SimdUint};
use std::simd::{Select, Simd, ToBytes};

use blanket_core::parallel::{CHUNK_PIXELS, chunks_mut_above};
use blanket_core::pixels::{Bytes, Samples};
use blanket_core::raster::{Image, PixelMode};
use pyo3::exceptions::{PyMemoryError, PyValueError};
use pyo3::prelude::*;

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(chops_binary, module)?)?;
    module.add_function(wrap_pyfunction!(chops_invert, module)?)?;
    module.add_function(wrap_pyfunction!(chops_offset, module)?)?;
    Ok(())
}

fn reserved_buffer(len: usize) -> PyResult<Vec<u8>> {
    let mut pixels = Vec::new();
    pixels.try_reserve_exact(len).map_err(|_| PyMemoryError::new_err("cannot allocate image"))?;
    Ok(pixels)
}

#[derive(Clone, Copy)]
enum Operation {
    Difference,
    Multiply,
    Screen,
    Lighter,
    Darker,
    Add,
    Subtract,
    AddModulo,
    SubtractModulo,
    SoftLight,
    HardLight,
    Overlay,
    And,
    Or,
    Xor,
}

impl Operation {
    fn sample_vector<const M: u64>(self, a: Samples, b: Samples, scale: f32, offset: i32) -> Samples {
        let x = a.cast::<u64>();
        let y = b.cast::<u64>();
        let m = Simd::splat(M);
        match self {
            Self::Difference => a.simd_max(b) - a.simd_min(b),
            Self::Lighter => a.simd_max(b),
            Self::Darker => a.simd_min(b),
            Self::Multiply => (x * y / m).cast(),
            Self::Screen => (m - (m - x) * (m - y) / m).cast(),
            Self::Add | Self::Subtract => {
                let a = a.cast::<i32>();
                let b = b.cast::<i32>();
                let sum = if matches!(self, Self::Add) { a + b } else { a - b };
                let value = sum.cast::<f32>() / Simd::splat(scale) + Simd::splat(offset as f32);
                value.is_finite().select(value.simd_clamp(Simd::splat(0.0), Simd::splat(M as f32)), Simd::splat(0.0)).cast()
            }
            Self::AddModulo => (a + b) & Samples::splat(M as u16),
            Self::SubtractModulo => (a - b) & Samples::splat(M as u16),
            Self::SoftLight => (((m - x) * x * y / Simd::splat((M + 1).pow(2))) + x * (m - (m - x) * (m - y) / m) / m).cast(),
            Self::HardLight | Self::Overlay => {
                let selector = if matches!(self, Self::HardLight) { y } else { x };
                let low = selector.simd_lt(Simd::splat(M.div_ceil(2)));
                let value = low.select(x, m - x) * low.select(y, m - y) / Simd::splat(M / 2);
                low.select(value, m - value).cast()
            }
            Self::And => a & b,
            Self::Or => a | b,
            Self::Xor => a ^ b,
        }
    }

    fn apply_sample(self, a: u16, b: u16, maximum: u16, scale: f32, offset: i32) -> u16 {
        let (x, y, m) = (u64::from(a), u64::from(b), u64::from(maximum));
        match self {
            Self::Difference => a.abs_diff(b),
            Self::Multiply => (x * y / m) as u16,
            Self::Screen => (m - (m - x) * (m - y) / m) as u16,
            Self::Lighter => a.max(b),
            Self::Darker => a.min(b),
            Self::Add | Self::Subtract => {
                let value = if matches!(self, Self::Add) { i32::from(a) + i32::from(b) } else { i32::from(a) - i32::from(b) };
                let value = value as f32 / scale + offset as f32;
                if value.is_finite() { value.clamp(0.0, f32::from(maximum)) as u16 } else { 0 }
            }
            Self::AddModulo => a.wrapping_add(b) & maximum,
            Self::SubtractModulo => a.wrapping_sub(b) & maximum,
            Self::SoftLight => (((m - x) * x * y / (m + 1).pow(2)) + x * (m - (m - x) * (m - y) / m) / m) as u16,
            Self::HardLight | Self::Overlay => {
                let selector = if matches!(self, Self::HardLight) { y } else { x };
                if selector < m.div_ceil(2) { (x * y / (m / 2)) as u16 } else { (m - (m - x) * (m - y) / (m / 2)) as u16 }
            }
            Self::And => a & b,
            Self::Or => a | b,
            Self::Xor => a ^ b,
        }
    }

    fn clip_scaled(value: f32) -> u8 {
        let temp = if !value.is_finite() || value > i32::MAX as f32 || value < i32::MIN as f32 { i32::MIN } else { value as i32 };
        if temp <= 0 {
            0
        } else if temp >= 255 {
            255
        } else {
            temp as u8
        }
    }

    fn parse(name: &str) -> PyResult<Self> {
        Ok(match name {
            "difference" => Self::Difference,
            "multiply" => Self::Multiply,
            "screen" => Self::Screen,
            "lighter" => Self::Lighter,
            "darker" => Self::Darker,
            "add" => Self::Add,
            "subtract" => Self::Subtract,
            "add_modulo" => Self::AddModulo,
            "subtract_modulo" => Self::SubtractModulo,
            "soft_light" => Self::SoftLight,
            "hard_light" => Self::HardLight,
            "overlay" => Self::Overlay,
            "and" => Self::And,
            "or" => Self::Or,
            "xor" => Self::Xor,
            _ => return Err(PyValueError::new_err("unknown channel operation")),
        })
    }

    fn apply(self, a: u8, b: u8, scale: f32, offset: i32) -> u8 {
        let x = u32::from(a);
        let y = u32::from(b);
        match self {
            Self::Difference => a.abs_diff(b),
            Self::Multiply => (x * y / 255) as u8,
            Self::Screen => (255 - (255 - x) * (255 - y) / 255) as u8,
            Self::Lighter => a.max(b),
            Self::Darker => a.min(b),
            Self::Add => Self::clip_scaled((x + y) as f32 / scale + offset as f32),
            Self::Subtract => Self::clip_scaled((i32::from(a) - i32::from(b)) as f32 / scale + offset as f32),
            Self::AddModulo => a.wrapping_add(b),
            Self::SubtractModulo => a.wrapping_sub(b),
            Self::SoftLight => (((255 - x) * x * y / 65536) + x * (255 - (255 - x) * (255 - y) / 255) / 255) as u8,
            Self::HardLight | Self::Overlay => {
                let selector = if matches!(self, Self::HardLight) { y } else { x };
                if selector < 128 { (x * y / 127) as u8 } else { (255 - (255 - x) * (255 - y) / 127) as u8 }
            }
            Self::And => a & b,
            Self::Or => a | b,
            Self::Xor => a ^ b,
        }
    }
}

fn hard_light_chunk<const OVERLAY: bool>(a: &[u8], b: &[u8], output: &mut [MaybeUninit<u8>]) {
    let full = Simd::splat(255);
    let done = output.len() / 16 * 16;
    for i in (0..done).step_by(16) {
        let x = Bytes::from_slice(&a[i..]).cast::<u16>();
        let y = Bytes::from_slice(&b[i..]).cast::<u16>();
        let selector = if OVERLAY { x } else { y };
        let low = selector.simd_lt(Simd::splat(128));
        let value = low.select(x, full - x) * low.select(y, full - y) / Simd::splat(127);
        let values = low.select(value, full - value).cast::<u8>().to_array();
        for (dst, value) in output[i..i + 16].iter_mut().zip(values) {
            dst.write(value);
        }
    }
    let operation = if OVERLAY { Operation::Overlay } else { Operation::HardLight };
    for ((dst, &a), &b) in output[done..].iter_mut().zip(&a[done..]).zip(&b[done..]) {
        dst.write(operation.apply(a, b, 1.0, 0));
    }
}

fn bitwise_chunk<const OP: u8>(a: &[u8], b: &[u8], output: &mut [u8]) {
    let (words, tail) = output.as_chunks_mut::<8>();
    for ((dst, left), right) in words.iter_mut().zip(a.as_chunks::<8>().0).zip(b.as_chunks::<8>().0) {
        let left = u64::from_ne_bytes(*left);
        let right = u64::from_ne_bytes(*right);
        *dst = match OP {
            0 => left & right,
            1 => left | right,
            _ => left ^ right,
        }
        .to_ne_bytes();
    }
    for ((dst, &left), &right) in tail.iter_mut().zip(&a[words.len() * 8..]).zip(&b[words.len() * 8..]) {
        *dst = match OP {
            0 => left & right,
            1 => left | right,
            _ => left ^ right,
        };
    }
}

#[pyfunction]
#[pyo3(signature = (first, second, operation, scale=1.0, offset=0))]
pub fn chops_binary(py: Python<'_>, first: &Image, second: &Image, operation: &str, scale: f32, offset: i32) -> PyResult<Image> {
    let operation = Operation::parse(operation)?;
    if blanket_core::samples::supported(first) || blanket_core::samples::supported(second) {
        return py.detach(|| binary_samples(first, second, operation, scale, offset));
    }
    let a = first.pixel_data()?;
    let b = second.pixel_data()?;
    if first.mode != second.mode {
        return Err(PyValueError::new_err("images do not match"));
    }
    // Like Pillow, differently sized inputs use their top-left intersection.
    let width = first.width.min(second.width);
    let height = first.height.min(second.height);
    let channels = first.mode.channels();
    let row = width as usize * channels;
    let length = row * height as usize;
    if matches!(operation, Operation::And | Operation::Or | Operation::Xor) && first.width == second.width {
        let mut pixels = reserved_buffer(length)?;
        pixels.resize(length, 0);
        py.detach(|| {
            chunks_mut_above(&mut pixels, 256 * 1024, 512 * 1024, |chunk, dst| {
                let start = chunk * 256 * 1024;
                let a = &a[start..start + dst.len()];
                let b = &b[start..start + dst.len()];
                match operation {
                    Operation::And => bitwise_chunk::<0>(a, b, dst),
                    Operation::Or => bitwise_chunk::<1>(a, b, dst),
                    Operation::Xor => bitwise_chunk::<2>(a, b, dst),
                    _ => unreachable!(),
                }
            });
        });
        return Image::from_pixels(width, height, first.mode, pixels, None);
    }
    let mut pixels = reserved_buffer(length)?;
    let rows_per_chunk = (CHUNK_PIXELS / (width as usize).max(1)).max(1);
    let threshold = match operation {
        Operation::AddModulo | Operation::SubtractModulo | Operation::Lighter | Operation::Darker | Operation::Difference => 16 * 1024 * 1024,
        _ => blanket_core::parallel::MIN_PARALLEL_BYTES,
    };
    py.detach(|| {
        let output = &mut pixels.spare_capacity_mut()[..length];
        chunks_mut_above(output, row.max(1) * rows_per_chunk, threshold, |chunk, dst| {
            for (i, output_row) in dst.chunks_exact_mut(row).enumerate() {
                let y = chunk * rows_per_chunk + i;
                let a_start = y * first.width as usize * channels;
                let b_start = y * second.width as usize * channels;
                // Dispatch outside the pixel loop so each simple operation
                // can be vectorized without a per-sample enum branch.
                macro_rules! apply {
                    ($f:expr) => {
                        for ((value, &a), &b) in output_row.iter_mut().zip(&a[a_start..a_start + row]).zip(&b[b_start..b_start + row]) {
                            value.write($f(a, b));
                        }
                    };
                }
                match operation {
                    Operation::AddModulo => apply!(u8::wrapping_add),
                    Operation::SubtractModulo => apply!(u8::wrapping_sub),
                    Operation::Lighter => apply!(u8::max),
                    Operation::Darker => apply!(u8::min),
                    Operation::Difference => apply!(u8::abs_diff),
                    Operation::And => apply!(std::ops::BitAnd::bitand),
                    Operation::Or => apply!(std::ops::BitOr::bitor),
                    Operation::Xor => apply!(std::ops::BitXor::bitxor),
                    Operation::Add => apply!(|a, b| Operation::Add.apply(a, b, scale, offset)),
                    Operation::Subtract => apply!(|a, b| Operation::Subtract.apply(a, b, scale, offset)),
                    Operation::Multiply => apply!(|a, b| Operation::Multiply.apply(a, b, scale, offset)),
                    Operation::Screen => apply!(|a, b| Operation::Screen.apply(a, b, scale, offset)),
                    Operation::SoftLight => apply!(|a, b| Operation::SoftLight.apply(a, b, scale, offset)),
                    Operation::HardLight => {
                        hard_light_chunk::<false>(&a[a_start..a_start + row], &b[b_start..b_start + row], output_row);
                    }
                    Operation::Overlay => {
                        hard_light_chunk::<true>(&a[a_start..a_start + row], &b[b_start..b_start + row], output_row);
                    }
                }
            }
        });
    });
    // Every output sample is initialized by the disjoint row partitions.
    unsafe { pixels.set_len(length) };
    let mut result = Image::from_pixels(width, height, first.mode, pixels, None)?;
    // Channel arithmetic creates an empty output palette, as Pillow does.
    result.palette = first.palette.as_ref().map(|_| (PixelMode::Rgb, Vec::new()));
    Ok(result)
}

fn sample_row<const M: u64>(a: &[u16], b: &[u16], output: &mut [u8], operation: Operation, parameters: (f32, i32), big_endian: bool) {
    let (scale, offset) = parameters;
    let end = a.len() / 16 * 16;
    for i in (0..end).step_by(16) {
        let value = operation.sample_vector::<M>(Samples::from_slice(&a[i..]), Samples::from_slice(&b[i..]), scale, offset);
        let bytes = if big_endian { value.to_be_bytes() } else { value.to_le_bytes() };
        bytes.copy_to_slice(&mut output[i * 2..i * 2 + 32]);
    }
    for ((dst, &a), &b) in output[end * 2..].as_chunks_mut::<2>().0.iter_mut().zip(&a[end..]).zip(&b[end..]) {
        let value = operation.apply_sample(a, b, M as u16, scale, offset);
        *dst = if big_endian { value.to_be_bytes() } else { value.to_le_bytes() };
    }
}

fn binary_samples(first: &Image, second: &Image, operation: Operation, scale: f32, offset: i32) -> PyResult<Image> {
    if first.mode != second.mode || first.bit_depth != second.bit_depth {
        return Err(PyValueError::new_err("images do not match"));
    }
    let a = blanket_core::samples::view(first)?;
    let b = blanket_core::samples::view(second)?;
    let (width, height) = (first.width.min(second.width), first.height.min(second.height));
    let channels = first.mode.channels();
    let row = width as usize * channels;
    let maximum = blanket_core::samples::maximum(first);
    let mut pixels = blanket_core::raster::alloc_pixels(row * height as usize * 2)?;
    if row != 0 {
        chunks_mut_above(&mut pixels, row * 2 * 16, 256 * 1024, |band, rows| {
            for (i, dst) in rows.chunks_exact_mut(row * 2).enumerate() {
                let y = band * 16 + i;
                let a = &a[y * first.width as usize * channels..][..row];
                let b = &b[y * second.width as usize * channels..][..row];
                match maximum {
                    1023 => sample_row::<1023>(a, b, dst, operation, (scale, offset), first.mode == PixelMode::I16B),
                    4095 => sample_row::<4095>(a, b, dst, operation, (scale, offset), first.mode == PixelMode::I16B),
                    65535 => sample_row::<65535>(a, b, dst, operation, (scale, offset), first.mode == PixelMode::I16B),
                    _ => unreachable!(),
                }
            }
        });
    }
    if first.mode.is_wide_scalar() {
        Image::from_integer_bytes(width, height, first.mode, pixels)
    } else {
        Image::from_sample_bytes(width, height, first.mode, pixels, first.bit_depth, None)
    }
}

#[pyfunction]
pub fn chops_invert(py: Python<'_>, image: &Image) -> PyResult<Image> {
    if blanket_core::samples::supported(image) {
        return py.detach(|| {
            let source = blanket_core::samples::view(image)?;
            let maximum = blanket_core::samples::maximum(image);
            let mut pixels = blanket_core::raster::alloc_pixels(source.len() * 2)?;
            chunks_mut_above(&mut pixels, CHUNK_PIXELS * 2, 256 * 1024, |chunk, dst| {
                let source = &source[chunk * CHUNK_PIXELS..][..dst.len() / 2];
                let end = source.len() / 16 * 16;
                for i in (0..end).step_by(16) {
                    let value = Samples::splat(maximum) - Samples::from_slice(&source[i..]);
                    let bytes = if image.mode == PixelMode::I16B { value.to_be_bytes() } else { value.to_le_bytes() };
                    bytes.copy_to_slice(&mut dst[i * 2..i * 2 + 32]);
                }
                for (dst, &sample) in dst[end * 2..].as_chunks_mut::<2>().0.iter_mut().zip(&source[end..]) {
                    let value = maximum - sample;
                    *dst = if image.mode == PixelMode::I16B { value.to_be_bytes() } else { value.to_le_bytes() };
                }
            });
            blanket_core::samples::output_bytes(image, pixels)
        });
    }
    let source = image.pixel_data()?;
    let mut pixels = reserved_buffer(source.len())?;
    py.detach(|| {
        chunks_mut_above(&mut pixels.spare_capacity_mut()[..source.len()], CHUNK_PIXELS, 16 * 1024 * 1024, |chunk, dst| {
            for (value, &sample) in dst.iter_mut().zip(&source[chunk * CHUNK_PIXELS..]) {
                value.write(255 - sample);
            }
        });
    });
    // The loop initializes exactly source.len() bytes, including the tail.
    unsafe { pixels.set_len(source.len()) };
    let mut result = Image::from_pixels(image.width, image.height, image.mode, pixels, None)?;
    result.palette = image.palette.as_ref().map(|_| (PixelMode::Rgb, Vec::new()));
    Ok(result)
}

#[pyfunction]
fn chops_offset(py: Python<'_>, image: &Image, xoffset: i64, yoffset: i64) -> PyResult<Image> {
    let source = image.raw_data()?;
    let mut pixels = reserved_buffer(source.len())?;
    if image.width != 0 && image.height != 0 {
        let bytes_per_pixel = image.raw_bytes_per_pixel();
        let row = image.width as usize * bytes_per_pixel;
        let head = xoffset.rem_euclid(i64::from(image.width)) as usize * bytes_per_pixel;
        let shift_y = yoffset.rem_euclid(i64::from(image.height)) as usize;
        let height = image.height as usize;
        py.detach(|| {
            let tail = row - head;
            if head.min(tail) <= row / 8 && source.len() < 4 << 20 {
                // Rotate the flat buffer, then repair the narrow wrap at each row.
                let shift = (shift_y * row + if head <= tail { head } else { source.len() - tail }) % source.len();
                pixels.extend_from_slice(&source[source.len() - shift..]);
                pixels.extend_from_slice(&source[..source.len() - shift]);
                let mut sy = height - shift_y;
                if head != 0 {
                    for y in 0..height {
                        if sy == height {
                            sy = 0;
                        }
                        let (src, target, count) = if head <= tail { (sy * row + tail, y * row, head) } else { (sy * row, y * row + head, tail) };
                        pixels[target..target + count].copy_from_slice(&source[src..src + count]);
                        sy += 1;
                    }
                }
            } else {
                chunks_mut_above(&mut pixels.spare_capacity_mut()[..source.len()], row * 32, 4 << 20, |band, rows| {
                    let mut sy = (band * 32 + height - shift_y) % height;
                    for dst in rows.chunks_exact_mut(row) {
                        let src = &source[sy * row..(sy + 1) * row];
                        dst[..head].write_copy_of_slice(&src[tail..]);
                        dst[head..].write_copy_of_slice(&src[..tail]);
                        sy += 1;
                        if sy == height {
                            sy = 0;
                        }
                    }
                });
                // SAFETY: every reserved byte was initialized by the row copies.
                unsafe { pixels.set_len(source.len()) };
            }
        });
    }
    Ok(Image {
        width: image.width,
        height: image.height,
        mode: image.mode,
        pixels: Some(pixels),
        bit_depth: image.bit_depth,
        format: None,
        palette: image.palette.clone(),
        shared_pixels: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_vectors_match_scalar_with_tails_and_guards() {
        fn check<const M: u64>() {
            for operation in [
                Operation::Difference,
                Operation::Multiply,
                Operation::Screen,
                Operation::Lighter,
                Operation::Darker,
                Operation::Add,
                Operation::Subtract,
                Operation::AddModulo,
                Operation::SubtractModulo,
                Operation::SoftLight,
                Operation::HardLight,
                Operation::Overlay,
                Operation::And,
                Operation::Or,
                Operation::Xor,
            ] {
                for length in [0, 1, 15, 16, 17, 31, 32, 33, 257] {
                    let a: Vec<u16> = (0..length + 1).map(|i| ([0, M, M / 2, M / 2 + 1, (i * 7919) as u64 & M][i % 5]) as u16).collect();
                    let b: Vec<u16> = (0..length + 1).map(|i| ((i * 171 + 65535) as u64 & M) as u16).collect();
                    for parameters in [(1.0, 0), (2.3, -17), (0.0, 1), (f32::NAN, 0), (f32::INFINITY, 12), (-1.0, i32::MAX)] {
                        for big in [false, true] {
                            let mut output = vec![173; length * 2 + 2];
                            sample_row::<M>(&a[1..], &b[1..], &mut output[1..length * 2 + 1], operation, parameters, big);
                            for (i, bytes) in output[1..length * 2 + 1].as_chunks::<2>().0.iter().enumerate() {
                                let actual = if big { u16::from_be_bytes(*bytes) } else { u16::from_le_bytes(*bytes) };
                                assert_eq!(actual, operation.apply_sample(a[i + 1], b[i + 1], M as u16, parameters.0, parameters.1));
                            }
                            assert_eq!((output[0], output[length * 2 + 1]), (173, 173));
                        }
                    }
                }
            }
        }
        check::<1023>();
        check::<4095>();
        check::<65535>();
    }

    #[test]
    fn sample_arithmetic_matches_byte_formulas() {
        for operation in [
            Operation::Difference,
            Operation::Multiply,
            Operation::Screen,
            Operation::Lighter,
            Operation::Darker,
            Operation::Add,
            Operation::Subtract,
            Operation::AddModulo,
            Operation::SubtractModulo,
            Operation::SoftLight,
            Operation::HardLight,
            Operation::Overlay,
            Operation::And,
            Operation::Or,
            Operation::Xor,
        ] {
            for a in 0..=255_u16 {
                for b in 0..=255_u16 {
                    assert_eq!(operation.apply_sample(a, b, 255, 1.0, 0), u16::from(operation.apply(a as u8, b as u8, 1.0, 0)));
                }
            }
        }
    }

    #[test]
    fn bitwise_chunks_match_scalar_with_unaligned_buffers_tails_and_guards() {
        for offset in 0..8 {
            for len in 0..=33 {
                let a: Vec<u8> = (0..offset + len).map(|i| (i * 37) as u8).collect();
                let b: Vec<u8> = (0..offset + len + 1).map(|i| (i * 53) as u8).collect();
                for operation in [Operation::And, Operation::Or, Operation::Xor] {
                    let mut expected = vec![0xa5; offset + len + 8];
                    for i in 0..len {
                        expected[offset + i] = operation.apply(a[offset + i], b[offset + i + 1], 1.0, 0);
                    }
                    let mut output = vec![0xa5; expected.len()];
                    let dst = &mut output[offset..offset + len];
                    match operation {
                        Operation::And => bitwise_chunk::<0>(&a[offset..], &b[offset + 1..], dst),
                        Operation::Or => bitwise_chunk::<1>(&a[offset..], &b[offset + 1..], dst),
                        Operation::Xor => bitwise_chunk::<2>(&a[offset..], &b[offset + 1..], dst),
                        _ => unreachable!(),
                    }
                    assert_eq!(output, expected, "offset={offset}, len={len}");
                }
            }
        }
    }

    fn blend_chunk(operation: Operation, a: &[u8], b: &[u8], output: &mut [MaybeUninit<u8>]) {
        match operation {
            Operation::HardLight => hard_light_chunk::<false>(a, b, output),
            Operation::Overlay => hard_light_chunk::<true>(a, b, output),
            _ => unreachable!(),
        }
    }

    #[test]
    fn blends_match_scalar_for_every_channel_pair() {
        let a: Vec<u8> = (0..=255).flat_map(|a| [a; 256]).collect();
        let b: Vec<u8> = (0..=255).cycle().take(a.len()).collect();
        let mut output = vec![MaybeUninit::uninit(); a.len()];
        for operation in [Operation::HardLight, Operation::Overlay] {
            blend_chunk(operation, &a, &b, &mut output);
            for ((value, &a), &b) in output.iter().zip(&a).zip(&b) {
                // SAFETY: blend_chunk initializes every output sample.
                assert_eq!(unsafe { value.assume_init() }, operation.apply(a, b, 1.0, 0));
            }
        }
    }

    #[test]
    fn blends_match_scalar_with_unaligned_buffers_tails_and_guards() {
        for offset in 0..16 {
            for len in 0..=65 {
                let a: Vec<u8> = (0..offset + len).map(|i| (i * 37 + i / 11) as u8).collect();
                let b: Vec<u8> = (0..offset + len + 1).map(|i| (i * 53 + i / 7) as u8).collect();
                for operation in [Operation::HardLight, Operation::Overlay] {
                    let mut expected = vec![0xa5; offset + len + 16];
                    for i in 0..len {
                        expected[offset + i] = operation.apply(a[offset + i], b[offset + i + 1], 1.0, 0);
                    }
                    let mut output = vec![MaybeUninit::new(0xa5); expected.len()];
                    blend_chunk(operation, &a[offset..], &b[offset + 1..], &mut output[offset..offset + len]);
                    // SAFETY: guards are initialized above and blend_chunk writes the entire output slice.
                    let output: Vec<u8> = output.into_iter().map(|value| unsafe { value.assume_init() }).collect();
                    assert_eq!(output, expected, "offset={offset}, len={len}");
                }
            }
        }
    }
}
