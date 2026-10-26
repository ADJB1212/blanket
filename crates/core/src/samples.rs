//! Unsigned high-depth sample access shared by processing kernels.

use std::borrow::Cow;
use std::simd::{Simd, ToBytes};

use pyo3::prelude::*;

use crate::raster::{Image, PixelMode};

pub fn supported(image: &Image) -> bool {
    matches!(image.bit_depth, 10 | 12 | 16)
        && matches!(image.mode, PixelMode::L | PixelMode::Rgb | PixelMode::Rgba | PixelMode::I16 | PixelMode::I16L | PixelMode::I16B)
}

pub fn maximum(image: &Image) -> u16 {
    ((1_u32 << image.bit_depth) - 1) as u16
}

pub fn read(image: &Image) -> PyResult<Vec<u16>> {
    decode(image.raw_data()?, image.mode == PixelMode::I16B)
}

pub fn view(image: &Image) -> PyResult<Cow<'_, [u16]>> {
    view_bytes(image.raw_data()?, image.mode == PixelMode::I16B)
}

fn view_bytes(source: &[u8], big_endian: bool) -> PyResult<Cow<'_, [u16]>> {
    if big_endian == cfg!(target_endian = "big") {
        // All u16 bit patterns are valid; align_to checks alignment and bounds.
        let (prefix, samples, suffix) = unsafe { source.align_to::<u16>() };
        if prefix.is_empty() && suffix.is_empty() {
            return Ok(Cow::Borrowed(samples));
        }
    }
    decode(source, big_endian).map(Cow::Owned)
}

fn decode(source: &[u8], big_endian: bool) -> PyResult<Vec<u16>> {
    let len = source.len() / 2;
    let mut values = Vec::new();
    values.try_reserve_exact(len).map_err(|_| pyo3::exceptions::PyMemoryError::new_err("cannot allocate image"))?;
    let chunk = 128 * 1024;
    crate::parallel::chunks_mut_above(&mut values.spare_capacity_mut()[..len], chunk, 4 << 20, |index, output| {
        let source = &source[index * chunk * 2..][..output.len() * 2];
        let end = output.len() / 16 * 16;
        for i in (0..end).step_by(16) {
            let block =
                if big_endian { crate::pixels::load_u16::<true>(&source[i * 2..]) } else { crate::pixels::load_u16::<false>(&source[i * 2..]) };
            output[i..i + 16].write_copy_of_slice(&block.to_array());
        }
        for (dst, &bytes) in output[end..].iter_mut().zip(source[end * 2..].as_chunks::<2>().0) {
            dst.write(if big_endian { u16::from_be_bytes(bytes) } else { u16::from_le_bytes(bytes) });
        }
    });
    // Every sample, including the final partial partition, was initialized above.
    unsafe { values.set_len(len) };
    Ok(values)
}

pub(crate) fn encode(values: &[u16], big_endian: bool) -> PyResult<Vec<u8>> {
    let len = values.len().checked_mul(2).ok_or_else(|| pyo3::exceptions::PyMemoryError::new_err("cannot allocate image"))?;
    let mut pixels = Vec::new();
    pixels.try_reserve_exact(len).map_err(|_| pyo3::exceptions::PyMemoryError::new_err("cannot allocate image"))?;
    let chunk = 256 * 1024;
    crate::parallel::chunks_mut_above(&mut pixels.spare_capacity_mut()[..len], chunk, 4 << 20, |index, output| {
        let source = &values[index * chunk / 2..][..output.len() / 2];
        let end = source.len() / 16 * 16;
        for i in (0..end).step_by(16) {
            let value = Simd::<u16, 16>::from_slice(&source[i..]);
            let bytes = if big_endian { value.to_be_bytes() } else { value.to_le_bytes() };
            output[i * 2..i * 2 + 32].write_copy_of_slice(&bytes.to_array());
        }
        for (dst, &value) in output[end * 2..].as_chunks_mut::<2>().0.iter_mut().zip(&source[end..]) {
            dst.write_copy_of_slice(&if big_endian { value.to_be_bytes() } else { value.to_le_bytes() });
        }
    });
    // Every byte, including the final partial partition, was initialized above.
    unsafe { pixels.set_len(len) };
    Ok(pixels)
}

pub fn output(image: &Image, values: &[u16]) -> PyResult<Image> {
    let pixels = encode(values, image.mode == PixelMode::I16B)?;
    if image.mode.is_wide_scalar() {
        Image::from_integer_bytes(image.width, image.height, image.mode, pixels)
    } else {
        Image::from_sample_bytes(image.width, image.height, image.mode, pixels, image.bit_depth, None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn views_borrow_only_aligned_native_endian_samples() {
        for count in [0, 1, 15, 16, 17, 131073, 2097169] {
            let expected: Vec<u16> = (0..count).map(|i| (i * 719 + 65503) as u16).collect();
            for big_endian in [false, true] {
                for unaligned in [false, true] {
                    let mut storage = vec![173_u8; count * 2 + 4];
                    let start = storage.as_ptr().align_offset(std::mem::align_of::<u16>()) + usize::from(unaligned);
                    for (dst, &value) in storage[start..start + count * 2].as_chunks_mut::<2>().0.iter_mut().zip(&expected) {
                        *dst = if big_endian { value.to_be_bytes() } else { value.to_le_bytes() };
                    }
                    let source = &storage[start..start + count * 2];
                    let actual = view_bytes(source, big_endian).unwrap();
                    assert_eq!(&*actual, expected);
                    if count != 0 {
                        let borrowed = !unaligned && big_endian == cfg!(target_endian = "big");
                        assert_eq!(matches!(actual, Cow::Borrowed(_)), borrowed);
                        if borrowed {
                            assert_eq!(actual.as_ptr().cast::<u8>(), source.as_ptr());
                        }
                    }
                    assert_eq!(storage[start + count * 2], 173);
                }
            }
        }
    }

    #[test]
    fn encoding_preserves_endianness_unaligned_samples_and_tails() {
        for count in [0, 1, 15, 16, 17, 31, 32, 33, 65, 1025, 131073, 2097169] {
            let source: Vec<u16> = (0..count + 1).map(|i| (i * 719 + 65503) as u16).collect();
            for big_endian in [false, true] {
                let encoded = encode(&source[1..], big_endian).unwrap();
                let expected: Vec<u8> =
                    source[1..].iter().flat_map(|value| if big_endian { value.to_be_bytes() } else { value.to_le_bytes() }).collect();
                assert_eq!(encoded, expected);
                let mode = if big_endian { PixelMode::I16B } else { PixelMode::I16L };
                let image = Image::from_integer_bytes(count as u32, 1, mode, encoded).unwrap();
                assert_eq!(read(&image).unwrap(), source[1..]);
            }
        }
    }
}
