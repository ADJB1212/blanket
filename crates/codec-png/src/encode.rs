use std::io::Write;
use std::simd::Select;
use std::simd::cmp::SimdPartialOrd;
use std::simd::num::{SimdInt, SimdUint};

use blanket_codec_common::codec_error;
use blanket_core::parallel::chunks_mut_above;
use blanket_core::pixels::Bytes;
use blanket_core::{Image, PixelMode};
use pyo3::prelude::*;

fn paeth(a: Bytes, b: Bytes, c: Bytes) -> Bytes {
    let (a, b, c) = (a.cast::<i16>(), b.cast::<i16>(), c.cast::<i16>());
    let pa = (b - c).abs();
    let pb = (a - c).abs();
    let pc = (a + b - c - c).abs();
    pa.simd_le(pb).select(pa.simd_le(pc).select(a, c), pb.simd_le(pc).select(b, c)).cast()
}

fn residuals(current: Bytes, a: Bytes, b: Bytes, c: Bytes) -> [Bytes; 4] {
    [current - b, current - a, current - ((a & b) + ((a ^ b) >> 1)), current - paeth(a, b, c)]
}

fn scalar(current: u8, a: u8, b: u8, c: u8) -> [u8; 4] {
    let p = i16::from(a) + i16::from(b) - i16::from(c);
    let pa = (p - i16::from(a)).abs();
    let pb = (p - i16::from(b)).abs();
    let pc = (p - i16::from(c)).abs();
    let predictor = if pa <= pb && pa <= pc {
        a
    } else if pb <= pc {
        b
    } else {
        c
    };
    [
        current.wrapping_sub(b),
        current.wrapping_sub(a),
        current.wrapping_sub(((u16::from(a) + u16::from(b)) / 2) as u8),
        current.wrapping_sub(predictor),
    ]
}

fn filter(current: &[u8], previous: &[u8], channels: usize, output: &mut [u8]) {
    if current == previous {
        output[0] = 2;
        output[1..].fill(0);
        return;
    }
    let mut costs = [0_u64; 4];
    let end = channels + (current.len() - channels) / 16 * 16;
    for i in 0..channels {
        for (cost, value) in costs.iter_mut().zip(scalar(current[i], 0, previous[i], 0)) {
            *cost += u64::from((value as i8).unsigned_abs());
        }
    }
    for i in (channels..end).step_by(16) {
        let values = residuals(
            Bytes::from_slice(&current[i..]),
            Bytes::from_slice(&current[i - channels..]),
            Bytes::from_slice(&previous[i..]),
            Bytes::from_slice(&previous[i - channels..]),
        );
        for (cost, value) in costs.iter_mut().zip(values) {
            *cost += u64::from(value.cast::<i8>().abs().cast::<u8>().cast::<u16>().reduce_sum());
        }
    }
    for i in end..current.len() {
        for (cost, value) in costs.iter_mut().zip(scalar(current[i], current[i - channels], previous[i], previous[i - channels])) {
            *cost += u64::from((value as i8).unsigned_abs());
        }
    }
    let mut choice = 0;
    for i in 1..4 {
        if costs[choice] == 0 {
            break;
        }
        if costs[i] <= costs[choice] {
            choice = i;
        }
    }
    output[0] = [2, 1, 3, 4][choice];
    let dst = &mut output[1..];
    for i in 0..channels {
        dst[i] = scalar(current[i], 0, previous[i], 0)[choice];
    }
    for i in (channels..end).step_by(16) {
        let values = residuals(
            Bytes::from_slice(&current[i..]),
            Bytes::from_slice(&current[i - channels..]),
            Bytes::from_slice(&previous[i..]),
            Bytes::from_slice(&previous[i - channels..]),
        );
        values[choice].copy_to_slice(&mut dst[i..i + 16]);
    }
    for i in end..current.len() {
        dst[i] = scalar(current[i], current[i - channels], previous[i], previous[i - channels])[choice];
    }
}

pub(super) fn encode(image: &Image, pixels: &[u8], level: u8) -> PyResult<Vec<u8>> {
    let channels = image.mode.channels();
    let stride = image.width as usize * channels;
    if stride == 0 || image.height == 0 {
        return Err(codec_error("cannot encode empty PNG"));
    }
    let row_bytes = stride + 1;
    let rows = (4 * 1024 * 1024 / row_bytes).max(1).min(image.height as usize);
    let mut filtered = vec![0; row_bytes * rows];
    let zero = vec![0; stride];
    let mut compressor = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::new(u32::from(level)));
    for start in (0..image.height as usize).step_by(rows) {
        let count = rows.min(image.height as usize - start);
        let batch = &mut filtered[..row_bytes * count];
        chunks_mut_above(batch, row_bytes * 16, 256 * 1024, |chunk, output| {
            for (row, dst) in output.chunks_exact_mut(row_bytes).enumerate() {
                let y = start + chunk * 16 + row;
                let source = &pixels[y * stride..(y + 1) * stride];
                let previous = if y == 0 { &zero } else { &pixels[(y - 1) * stride..y * stride] };
                filter(source, previous, channels, dst);
            }
        });
        for row in batch.chunks_exact(row_bytes) {
            compressor.write_all(&row[..1]).map_err(codec_error)?;
            compressor.write_all(&row[1..]).map_err(codec_error)?;
        }
    }
    let compressed = compressor.finish().map_err(codec_error)?;
    let mut output = Vec::with_capacity(compressed.len() + 128);
    {
        let mut encoder = png::Encoder::new(&mut output, image.width, image.height);
        encoder.set_color(match image.mode {
            PixelMode::One | PixelMode::L => png::ColorType::Grayscale,
            PixelMode::La => png::ColorType::GrayscaleAlpha,
            PixelMode::Rgb => png::ColorType::Rgb,
            PixelMode::Rgba => png::ColorType::Rgba,
            _ => unreachable!("validated PNG mode"),
        });
        encoder.set_depth(png::BitDepth::Eight);
        encoder.validate_sequence(false);
        let mut writer = encoder.write_header().map_err(codec_error)?;
        for part in compressed.chunks(1024 * 1024) {
            writer.write_chunk(png::chunk::IDAT, part).map_err(codec_error)?;
        }
        writer.finish().map_err(codec_error)?;
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adaptive_rows_preserve_filter_ties_tails_and_guards() {
        for channels in 1..=4 {
            for width in [1, 2, 15, 16, 17, 31, 32, 33, 257] {
                let len = width * channels;
                for pattern in 0..4 {
                    let current: Vec<_> = (0..len + 1).map(|i| if pattern == 0 { 128 } else { (i * 137 + i / 7) as u8 }).collect();
                    let previous: Vec<_> = (0..len + 1)
                        .map(|i| match pattern {
                            0 => 0,
                            1 => current[i],
                            2 => (i * 53) as u8,
                            _ => 0,
                        })
                        .collect();
                    let (current, previous) = (&current[1..], &previous[1..]);
                    let mut candidates = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
                    for i in 0..len {
                        let a = if i < channels { 0 } else { current[i - channels] };
                        let c = if i < channels { 0 } else { previous[i - channels] };
                        for (dst, value) in candidates.iter_mut().zip(scalar(current[i], a, previous[i], c)) {
                            dst.push(value);
                        }
                    }
                    let mut best = 0;
                    let cost = |bytes: &[u8]| bytes.iter().map(|&v| u64::from((v as i8).unsigned_abs())).sum::<u64>();
                    for i in 1..4 {
                        if cost(&candidates[best]) == 0 {
                            break;
                        }
                        if cost(&candidates[i]) <= cost(&candidates[best]) {
                            best = i;
                        }
                    }
                    let mut output = vec![91; len + 3];
                    filter(current, previous, channels, &mut output[1..len + 2]);
                    assert_eq!(output[0], 91);
                    assert_eq!(output[len + 2], 91);
                    assert_eq!(output[1], [2, 1, 3, 4][best]);
                    assert_eq!(&output[2..len + 2], candidates[best]);
                }
            }
        }
    }

    #[test]
    fn batched_filtering_preserves_pixels_and_compression_ratio() {
        use image::ImageEncoder;
        use image::codecs::png::{CompressionType, FilterType, PngEncoder};

        for mode in [PixelMode::L, PixelMode::La, PixelMode::Rgb, PixelMode::Rgba] {
            for size in [(17, 9), (1025, 1025)] {
                let image = blanket_bench_support::fixture(size.0, size.1, mode, 8, 12345, false);
                let pixels = image.pixel_data().unwrap();
                let actual = encode(&image, pixels, 1).unwrap();
                let mut reference = Vec::new();
                PngEncoder::new_with_quality(&mut reference, CompressionType::Level(1), FilterType::Adaptive)
                    .write_image(pixels, size.0, size.1, blanket_codec_image::color_type(mode))
                    .unwrap();
                assert!(actual.len() <= reference.len() + 64, "{mode:?}/{size:?}: {} > {}", actual.len(), reference.len());
                assert_eq!(crate::decode_png(&actual).unwrap().pixel_data().unwrap(), pixels);
            }
        }
    }
}
