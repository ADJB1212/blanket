use blanket_convert_kernels::float::{AlphaOperation, alpha, decode_values, quantize_into, transfer_colors};
pub use blanket_convert_kernels::float::{composite, quantize, transfer};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::{Image, PixelMode};

pub fn convert(image: &Image, mode: PixelMode, depth: Option<u8>) -> PyResult<Image> {
    let color = |m| matches!(m, PixelMode::Rgb | PixelMode::Rgba | PixelMode::RgbF | PixelMode::RgbaF);
    if !color(image.mode) || !color(mode) {
        return Err(PyValueError::new_err("float color conversion requires RGB or RGBA modes"));
    }
    let float = mode.is_float();
    let depth = depth.unwrap_or(if float { 32 } else { 8 });
    if (float && depth != 32) || (!float && !matches!(depth, 8 | 10 | 12 | 16)) {
        return Err(PyValueError::new_err("invalid color sample depth"));
    }
    if float && image.mode.is_float() {
        let source = image.raw_data()?;
        let source_channels = image.mode.channels();
        let channels = mode.channels();
        let pixels = if source_channels == channels {
            source.to_vec()
        } else {
            let mut pixels = crate::buffers::buffer(image.size(), channels * 4)?;
            crate::parallel::chunks_mut(&mut pixels, crate::parallel::CHUNK_PIXELS * channels * 4, |chunk, dst| {
                let start = chunk * crate::parallel::CHUNK_PIXELS * source_channels * 4;
                for (src, dst) in source[start..].chunks_exact(source_channels * 4).zip(dst.chunks_exact_mut(channels * 4)) {
                    dst[..12].copy_from_slice(&src[..12]);
                    if channels == 4 {
                        dst[12..16].copy_from_slice(&1.0_f32.to_le_bytes());
                    }
                }
            });
            pixels
        };
        return Image::from_color_float_bytes(image.width, image.height, mode, pixels);
    }
    let pixels = match (image.mode.channels(), mode.channels()) {
        (3, 3) => convert_color_bytes::<3, 3>(image, mode, depth)?,
        (3, 4) => convert_color_bytes::<3, 4>(image, mode, depth)?,
        (4, 3) => convert_color_bytes::<4, 3>(image, mode, depth)?,
        (4, 4) => convert_color_bytes::<4, 4>(image, mode, depth)?,
        _ => unreachable!(),
    };
    if float {
        Image::from_color_float_bytes(image.width, image.height, mode, pixels)
    } else if depth == 8 {
        Image::from_pixels(image.width, image.height, mode, pixels, None)
    } else {
        Image::from_sample_bytes(image.width, image.height, mode, pixels, depth, None)
    }
}

fn convert_color_bytes<const S: usize, const D: usize>(image: &Image, mode: PixelMode, depth: u8) -> PyResult<Vec<u8>> {
    let source = image.raw_data()?;
    let source_bytes = if image.mode.is_float() {
        4
    } else if image.bit_depth == 8 {
        1
    } else {
        2
    };
    let target_bytes = if mode.is_float() {
        4
    } else if depth == 8 {
        1
    } else {
        2
    };
    let mut output = crate::buffers::buffer(image.size(), D * target_bytes)?;
    crate::parallel::chunks_mut(&mut output, crate::parallel::CHUNK_PIXELS * D * target_bytes, |chunk, dst| {
        let mut values = [0.0; 1024];
        let mut mapped = [0.0; 1024];
        let start = chunk * crate::parallel::CHUNK_PIXELS * S * source_bytes;
        for (block, dst) in dst.chunks_mut(256 * D * target_bytes).enumerate() {
            let count = dst.len() / (D * target_bytes);
            let src = &source[start + block * 256 * S * source_bytes..][..count * S * source_bytes];
            let values = &mut values[..count * S];
            decode_values(src, image.mode.is_float(), image.bit_depth, values);
            let values = if S == D {
                &*values
            } else {
                let mapped = &mut mapped[..count * D];
                crate::pixels::float_color_layout::<S, D>(values, mapped);
                mapped
            };
            if mode.is_float() {
                for (value, dst) in values.iter().zip(dst.as_chunks_mut::<4>().0) {
                    *dst = value.to_le_bytes();
                }
            } else {
                quantize_into(values, dst, depth);
            }
        }
    });
    Ok(output)
}

fn convert_values(image: &Image, mode: PixelMode, depth: u8, source: Vec<f32>) -> PyResult<Image> {
    if image.mode.channels() != mode.channels() {
        let pixels = if mode.channels() == 4 {
            mapped_color_bytes::<3, 4>(image.size(), &source, mode.is_float(), depth)?
        } else {
            mapped_color_bytes::<4, 3>(image.size(), &source, mode.is_float(), depth)?
        };
        return if mode.is_float() {
            Image::from_color_float_bytes(image.width, image.height, mode, pixels)
        } else if depth == 8 {
            Image::from_pixels(image.width, image.height, mode, pixels, None)
        } else {
            Image::from_sample_bytes(image.width, image.height, mode, pixels, depth, None)
        };
    }
    color_output(image.width, image.height, mode, depth, &source)
}

fn mapped_color_bytes<const S: usize, const D: usize>(size: (u32, u32), source: &[f32], floating: bool, depth: u8) -> PyResult<Vec<u8>> {
    let bytes = if floating {
        4
    } else if depth == 8 {
        1
    } else {
        2
    };
    let mut output = crate::buffers::buffer(size, D * bytes)?;
    crate::parallel::chunks_mut(&mut output, crate::parallel::CHUNK_PIXELS * D * bytes, |chunk, dst| {
        let mut mapped = [0.0; 64];
        let start = chunk * crate::parallel::CHUNK_PIXELS * S;
        for (block, dst) in dst.chunks_mut(16 * D * bytes).enumerate() {
            let count = dst.len() / (D * bytes);
            let source = &source[start + block * 16 * S..][..count * S];
            let mapped = &mut mapped[..count * D];
            crate::pixels::float_color_layout::<S, D>(source, mapped);
            if floating {
                for (sample, dst) in mapped.iter().zip(dst.as_chunks_mut::<4>().0) {
                    *dst = sample.to_le_bytes();
                }
            } else {
                quantize_into(mapped, dst, depth);
            }
        }
    });
    Ok(output)
}

fn color_output(width: u32, height: u32, mode: PixelMode, depth: u8, values: &[f32]) -> PyResult<Image> {
    if mode.is_float() {
        let bytes = values.iter().map(|value| value.to_le_bytes()).collect::<Vec<_>>().into_flattened();
        return Image::from_color_float_bytes(width, height, mode, bytes);
    }
    let bytes = quantize(values, depth);
    if depth == 8 { Image::from_pixels(width, height, mode, bytes, None) } else { Image::from_sample_bytes(width, height, mode, bytes, depth, None) }
}

pub fn values(image: &Image) -> PyResult<Vec<f32>> {
    let data = image.raw_data()?;
    if !image.mode.is_float() && !matches!(image.mode, PixelMode::Rgb | PixelMode::Rgba) {
        return Err(PyValueError::new_err("requires RGB or RGBA color"));
    }
    let bytes = if image.mode.is_float() {
        4
    } else if image.bit_depth == 8 {
        1
    } else {
        2
    };
    let mut result = vec![0.0; data.len() / bytes];
    decode_values(data, image.mode.is_float(), image.bit_depth, &mut result);
    Ok(result)
}

pub fn prepare(image: &Image, srgb: bool, premultiply: bool) -> PyResult<Image> {
    if !matches!(image.mode, PixelMode::Rgb | PixelMode::Rgba | PixelMode::RgbF | PixelMode::RgbaF) {
        return Err(PyValueError::new_err("linear-light processing requires RGB or RGBA color"));
    }
    let channels = image.mode.channels();
    let source = image.raw_data()?;
    let sample_bytes = if image.mode.is_float() {
        4
    } else if image.bit_depth == 8 {
        1
    } else {
        2
    };
    let mut pixels = crate::buffers::buffer(image.size(), channels * 4)?;
    crate::parallel::chunks_mut(&mut pixels, crate::parallel::CHUNK_PIXELS * channels * 4, |chunk, dst| {
        let mut values = [0.0; 1024];
        let start = chunk * crate::parallel::CHUNK_PIXELS * channels * sample_bytes;
        for (block, dst) in dst.chunks_mut(256 * channels * 4).enumerate() {
            let count = dst.len() / 4;
            let values = &mut values[..count];
            let src = &source[start + block * 256 * channels * sample_bytes..][..count * sample_bytes];
            decode_values(src, image.mode.is_float(), image.bit_depth, values);
            if srgb {
                transfer_colors(values, channels, true);
            }
            if channels == 4 {
                alpha(values, if premultiply { AlphaOperation::Premultiply } else { AlphaOperation::ClampInput });
            }
            for (value, dst) in values.iter().zip(dst.as_chunks_mut::<4>().0) {
                *dst = value.to_le_bytes();
            }
        }
    });
    Image::from_color_float_bytes(image.width, image.height, if image.mode.channels() == 4 { PixelMode::RgbaF } else { PixelMode::RgbF }, pixels)
}

pub fn prepare_values(image: &Image, srgb: bool, premultiply: bool) -> PyResult<Vec<f32>> {
    if !matches!(image.mode, PixelMode::Rgb | PixelMode::Rgba | PixelMode::RgbF | PixelMode::RgbaF) {
        return Err(PyValueError::new_err("linear-light processing requires RGB or RGBA color"));
    }
    let mut values = values(image)?;
    let channels = image.mode.channels();
    if srgb {
        transfer_colors(&mut values, channels, true);
    }
    if channels == 4 {
        alpha(&mut values, if premultiply { AlphaOperation::Premultiply } else { AlphaOperation::ClampInput });
    }
    Ok(values)
}

pub fn composite_rgba(background: &Image, foreground: &Image, srgb: bool) -> PyResult<Image> {
    if background.size() != foreground.size()
        || background.mode != foreground.mode
        || background.bit_depth != foreground.bit_depth
        || !matches!(background.mode, PixelMode::Rgba | PixelMode::RgbaF)
    {
        return Err(PyValueError::new_err("compositing requires matching RGBA images"));
    }
    let back = background.raw_data()?;
    let front = foreground.raw_data()?;
    let depth = background.bit_depth;
    let floating = background.mode.is_float();
    let pixel_bytes = background.raw_bytes_per_pixel();
    let mut pixels = crate::buffers::buffer(background.size(), pixel_bytes)?;
    let chunk_bytes = crate::parallel::CHUNK_PIXELS * pixel_bytes;
    let block_bytes = 256 * pixel_bytes;
    crate::parallel::chunks_mut(&mut pixels, chunk_bytes, |chunk, dst| {
        let mut back_values = [0.0; 1024];
        let mut front_values = [0.0; 1024];
        let start = chunk * chunk_bytes;
        for (block, dst) in dst.chunks_mut(block_bytes).enumerate() {
            let start = start + block * block_bytes;
            let range = start..start + dst.len();
            let count = dst.len() / pixel_bytes * 4;
            let back_values = &mut back_values[..count];
            let front_values = &mut front_values[..count];
            decode_values(&back[range.clone()], floating, depth, back_values);
            decode_values(&front[range], floating, depth, front_values);
            for values in [&mut *back_values, &mut *front_values] {
                if srgb {
                    transfer_colors(values, 4, true);
                }
                alpha(values, AlphaOperation::Premultiply);
            }
            composite(back_values, front_values);
            alpha(back_values, AlphaOperation::Unpremultiply);
            if srgb {
                transfer_colors(back_values, 4, false);
            }
            if floating {
                for (value, dst) in back_values.iter().zip(dst.as_chunks_mut::<4>().0) {
                    *dst = value.to_le_bytes();
                }
            } else {
                quantize_into(back_values, dst, depth);
            }
        }
    });
    Image::from_raw_bytes(background.width, background.height, background.mode, pixels, depth)
}

pub fn finish(image: &Image, original: &Image, srgb: bool, premultiply: bool) -> PyResult<Image> {
    if !image.mode.is_float() && !matches!(image.mode, PixelMode::Rgb | PixelMode::Rgba) {
        return Err(PyValueError::new_err("requires RGB or RGBA color"));
    }
    if image.mode.channels() != original.mode.channels() {
        return finish_values(image, original, srgb, premultiply, values(image)?);
    }
    let channels = image.mode.channels();
    let source = image.raw_data()?;
    let sample_bytes = if image.mode.is_float() {
        4
    } else if image.bit_depth == 8 {
        1
    } else {
        2
    };
    let target_bytes = if original.mode.is_float() {
        4
    } else if original.bit_depth == 8 {
        1
    } else {
        2
    };
    let mut pixels = crate::buffers::buffer(image.size(), channels * target_bytes)?;
    crate::parallel::chunks_mut(&mut pixels, crate::parallel::CHUNK_PIXELS * channels * target_bytes, |chunk, dst| {
        let mut values = [0.0; 1024];
        let start = chunk * crate::parallel::CHUNK_PIXELS * channels * sample_bytes;
        for (block, dst) in dst.chunks_mut(256 * channels * target_bytes).enumerate() {
            let count = dst.len() / target_bytes;
            let values = &mut values[..count];
            let src = &source[start + block * 256 * channels * sample_bytes..][..count * sample_bytes];
            decode_values(src, image.mode.is_float(), image.bit_depth, values);
            if channels == 4 {
                alpha(values, if premultiply { AlphaOperation::Unpremultiply } else { AlphaOperation::ClampOutput });
            }
            if srgb {
                transfer_colors(values, channels, false);
            }
            if original.mode.is_float() {
                for (value, dst) in values.iter().zip(dst.as_chunks_mut::<4>().0) {
                    *dst = value.to_le_bytes();
                }
            } else {
                quantize_into(values, dst, original.bit_depth);
            }
        }
    });
    if original.mode.is_float() {
        Image::from_color_float_bytes(image.width, image.height, original.mode, pixels)
    } else if original.bit_depth == 8 {
        Image::from_pixels(image.width, image.height, original.mode, pixels, None)
    } else {
        Image::from_sample_bytes(image.width, image.height, original.mode, pixels, original.bit_depth, None)
    }
}

pub fn finish_values(image: &Image, original: &Image, srgb: bool, premultiply: bool, mut values: Vec<f32>) -> PyResult<Image> {
    if image.mode.channels() == 4 {
        alpha(&mut values, if premultiply { AlphaOperation::Unpremultiply } else { AlphaOperation::ClampOutput });
    }
    if srgb {
        transfer_colors(&mut values, image.mode.channels(), false);
    }
    convert_values(image, original.mode, original.bit_depth, values)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_composite_matches_full_image_intermediates() {
        let width = 257;
        for (mode, depth) in [(PixelMode::Rgba, 8), (PixelMode::Rgba, 10), (PixelMode::Rgba, 12), (PixelMode::Rgba, 16), (PixelMode::RgbaF, 32)] {
            let build = |offset| {
                let values: Vec<f32> = (0..width * 4).map(|i| ((i * 13 + offset) % 313) as f32 / 255.0).collect();
                color_output(width, 1, mode, depth, &values).unwrap()
            };
            let back = build(7);
            let front = build(17);
            for srgb in [false, true] {
                let mut values = prepare_values(&back, srgb, true).unwrap();
                composite(&mut values, &prepare_values(&front, srgb, true).unwrap());
                let expected = finish_values(&back, &back, srgb, true, values).unwrap();
                let actual = composite_rgba(&back, &front, srgb).unwrap();
                assert_eq!(actual.raw_data().unwrap(), expected.raw_data().unwrap());
            }
        }
    }

    #[test]
    fn bounded_preparation_and_finish_match_full_image_intermediates() {
        for count in [0, 1, 7, 8, 15, 16, 17, 255, 256, 257, 16385, 65537] {
            for mode in [PixelMode::Rgb, PixelMode::Rgba] {
                let channels = mode.channels();
                let special = [-0.0_f32, 0.0, -2.0, 0.25, 0.5, 1.0, 2.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY];
                let bytes = (0..count * channels).flat_map(|i| special[i % special.len()].to_le_bytes()).collect();
                let floating =
                    Image::from_color_float_bytes(count as u32, 1, if channels == 3 { PixelMode::RgbF } else { PixelMode::RgbaF }, bytes).unwrap();
                for depth in [8, 10, 12, 16, 32] {
                    let image = if depth == 32 { floating.clone() } else { convert(&floating, mode, Some(depth)).unwrap() };
                    for srgb in [false, true] {
                        for premultiply in [false, true] {
                            let expected: Vec<_> =
                                prepare_values(&image, srgb, premultiply).unwrap().into_iter().flat_map(f32::to_le_bytes).collect();
                            let actual = prepare(&image, srgb, premultiply).unwrap();
                            assert_eq!(actual.raw_data().unwrap(), expected, "prepare count={count}, mode={mode:?}, depth={depth}");
                            let expected = finish_values(&floating, &image, srgb, premultiply, values(&floating).unwrap()).unwrap();
                            let actual = finish(&floating, &image, srgb, premultiply).unwrap();
                            assert_eq!(
                                actual.raw_data().unwrap(),
                                expected.raw_data().unwrap(),
                                "finish count={count}, mode={mode:?}, depth={depth}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn bounded_conversion_matches_full_image_intermediates() {
        for count in [0, 1, 15, 16, 17, 255, 256, 257, 65537] {
            for source_mode in [PixelMode::Rgb, PixelMode::Rgba, PixelMode::RgbF, PixelMode::RgbaF] {
                for source_depth in if source_mode.is_float() { &[32][..] } else { &[8, 10, 12, 16][..] } {
                    let source = if source_mode.is_float() {
                        let special = [-0.0_f32, 0.0, -2.0, 0.25, 0.5, 1.0, 2.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY];
                        let bytes = (0..count * source_mode.channels()).flat_map(|i| special[i % special.len()].to_le_bytes()).collect();
                        Image::from_color_float_bytes(count as u32, 1, source_mode, bytes).unwrap()
                    } else {
                        let maximum = (1_u32 << source_depth) - 1;
                        let samples: Vec<u16> = (0..count * source_mode.channels()).map(|i| ((i as u32 * 719 + 257) & maximum) as u16).collect();
                        if *source_depth == 8 {
                            Image::from_pixels(count as u32, 1, source_mode, samples.into_iter().map(|v| v as u8).collect(), None).unwrap()
                        } else {
                            Image::from_samples(count as u32, 1, source_mode, samples, *source_depth, None).unwrap()
                        }
                    };
                    for target_mode in [PixelMode::Rgb, PixelMode::Rgba, PixelMode::RgbF, PixelMode::RgbaF] {
                        for target_depth in if target_mode.is_float() { &[32][..] } else { &[8, 10, 12, 16][..] } {
                            if !source_mode.is_float() && !target_mode.is_float() {
                                continue;
                            }
                            let reference = convert_values(&source, target_mode, *target_depth, values(&source).unwrap()).unwrap();
                            let actual = convert(&source, target_mode, Some(*target_depth)).unwrap();
                            assert_eq!(actual.raw_data().unwrap(), reference.raw_data().unwrap());
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn mapped_color_output_matches_materialized_layout() {
        fn check<const S: usize, const D: usize>() {
            let values = [-0.0, 0.0, -2.0, 0.25, 0.5, 1.0, 2.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY];
            for count in [0, 1, 3, 4, 5, 15, 16, 17, 16385, 65537] {
                let storage: Vec<_> = (0..count * S + 1).map(|i| values[i % values.len()]).collect();
                let source = &storage[1..];
                let mut expected = vec![0.0; count * D];
                crate::pixels::float_color_layout::<S, D>(source, &mut expected);
                for depth in [8, 10, 12, 16, 32] {
                    let actual = mapped_color_bytes::<S, D>((count as u32, 1), source, depth == 32, depth).unwrap();
                    let reference: Vec<u8> =
                        if depth == 32 { expected.iter().flat_map(|v| v.to_le_bytes()).collect() } else { quantize(&expected, depth) };
                    assert_eq!(actual, reference, "S={S}, D={D}, count={count}, depth={depth}");
                }
            }
        }
        check::<3, 4>();
        check::<4, 3>();
    }

    #[test]
    fn finish_matches_materialized_float_conversion() {
        let samples = [-0.0, 0.0, -2.0, 0.0031308, 0.04045, 0.5, 1.0, 4.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY];
        for width in [0, 1, 3, 4, 5, 17, 33] {
            for mode in [PixelMode::RgbF, PixelMode::RgbaF] {
                let data: Vec<_> = (0..width as usize * mode.channels()).map(|i| samples[i % samples.len()]).collect();
                let image = color_output(width, 1, mode, 32, &data).unwrap();
                for target in [PixelMode::Rgb, PixelMode::Rgba, PixelMode::RgbF, PixelMode::RgbaF] {
                    for depth in if target.is_float() { &[32][..] } else { &[8, 10, 12, 16][..] } {
                        let original = convert(&image, target, Some(*depth)).unwrap();
                        for srgb in [false, true] {
                            for premultiply in [false, true] {
                                let mut expected = values(&image).unwrap();
                                if mode.channels() == 4 {
                                    alpha(&mut expected, if premultiply { AlphaOperation::Unpremultiply } else { AlphaOperation::ClampOutput });
                                }
                                if srgb {
                                    transfer_colors(&mut expected, mode.channels(), false);
                                }
                                let intermediate = color_output(width, 1, mode, 32, &expected).unwrap();
                                let expected = convert(&intermediate, target, Some(*depth)).unwrap();
                                let actual = finish(&image, &original, srgb, premultiply).unwrap();
                                assert_eq!(actual.raw_data().unwrap(), expected.raw_data().unwrap());
                                assert_eq!((actual.mode, actual.bit_depth), (target, *depth));
                            }
                        }
                    }
                }
            }
        }
    }

    fn equal(actual: &[f32], expected: &[f32]) {
        for (a, b) in actual.iter().zip(expected) {
            assert!(a.to_bits() == b.to_bits() || a.is_nan() && b.is_nan(), "{a:?} != {b:?}");
        }
    }

    #[test]
    fn normalized_samples_match_scalar() {
        for depth in [8, 10, 12, 16] {
            for width in 0..35 {
                let maximum = (1_u32 << depth) - 1;
                let samples: Vec<_> = (0..width * 3).map(|i| ((i * 7919) & maximum) as u16).collect();
                let image = if depth == 8 {
                    Image::from_pixels(width, 1, PixelMode::Rgb, samples.iter().map(|&v| v as u8).collect(), None)
                } else {
                    Image::from_samples(width, 1, PixelMode::Rgb, samples.clone(), depth, None)
                }
                .unwrap();
                equal(&values(&image).unwrap(), &samples.iter().map(|&v| v as f32 / maximum as f32).collect::<Vec<_>>());
            }
        }
    }
}
