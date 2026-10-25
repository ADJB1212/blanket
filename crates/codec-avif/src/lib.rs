use blanket_codec_common::{SaveOptions, codec_error, validate_dimensions};
use blanket_codec_image::{color_type, decode_rust_image};
use blanket_core::{Image, PixelMode};
use image::{ExtendedColorType, ImageEncoder, ImageFormat as RustFormat};
use pyo3::prelude::*;

pub fn decode_avif(data: &[u8]) -> Result<Image, String> {
    let context = mp4parse::read_avif(&mut std::io::Cursor::new(data), mp4parse::ParseStrictness::Normal).map_err(|e| e.to_string())?;
    validate_av1(context.primary_item_coded_data().ok_or("missing AVIF primary image")?)?;
    if let Some(alpha) = context.alpha_item_coded_data() {
        validate_av1(alpha)?;
    }
    drop(context);
    decode_rust_image(data, RustFormat::Avif, "AVIF")
}

fn validate_av1(mut data: &[u8]) -> Result<(), String> {
    let mut sequence = false;
    while let Some(&header) = data.first() {
        if header & 0x81 != 0 {
            return Err("invalid AV1 OBU header".into());
        }
        let mut start = 1 + usize::from(header & 4 != 0);
        let length = if header & 2 != 0 {
            let mut length = 0_u64;
            let mut complete = false;
            for shift in (0..56).step_by(7) {
                let byte = *data.get(start).ok_or("truncated AV1 OBU size")?;
                start += 1;
                length |= u64::from(byte & 127) << shift;
                if byte & 128 == 0 {
                    complete = true;
                    break;
                }
            }
            if !complete {
                return Err("invalid AV1 OBU size".into());
            }
            usize::try_from(length).map_err(|_| "AV1 OBU size overflow")?
        } else {
            data.len().checked_sub(start).ok_or("truncated AV1 OBU header")?
        };
        let end = start.checked_add(length).filter(|&end| end <= data.len()).ok_or("truncated AV1 OBU")?;
        if (header >> 3) & 15 == 1 {
            let (width, height) = sequence_dimensions(&data[start..end])?;
            validate_dimensions(width, height)?;
            sequence = true;
        }
        data = &data[end..];
    }
    if !sequence {
        return Err("missing AV1 sequence header".into());
    }
    Ok(())
}

fn sequence_dimensions(data: &[u8]) -> Result<(u32, u32), String> {
    let mut bits = bitreader::BitReader::new(data);
    let mut read = |count| bits.read_u32(count).map_err(|_| "truncated AV1 sequence header".to_owned());
    if read(3)? > 2 {
        return Err("invalid AV1 profile".into());
    }
    let still_picture = read(1)? != 0;
    let reduced = read(1)? != 0;
    if reduced {
        if !still_picture {
            return Err("invalid reduced AV1 sequence header".into());
        }
        read(5)?;
    } else {
        let mut buffer_delay_bits = 0;
        if read(1)? != 0 {
            read(32)?;
            read(32)?;
            if read(1)? != 0 {
                let mut leading_zeros = 0;
                while read(1)? == 0 {
                    leading_zeros += 1;
                    if leading_zeros == 32 {
                        return Err("invalid AV1 picture interval".into());
                    }
                }
                read(leading_zeros)?;
            }
            if read(1)? != 0 {
                buffer_delay_bits = read(5)? as u8 + 1;
                read(32)?;
                read(5)?;
                read(5)?;
            }
        }
        let display_delay_present = read(1)? != 0;
        let operating_points = read(5)? + 1;
        for _ in 0..operating_points {
            read(12)?;
            if read(5)? > 7 {
                read(1)?;
            }
            if buffer_delay_bits != 0 && read(1)? != 0 {
                read(buffer_delay_bits)?;
                read(buffer_delay_bits)?;
                read(1)?;
            }
            if display_delay_present && read(1)? != 0 {
                read(4)?;
            }
        }
    }
    let width_bits = read(4)? as u8 + 1;
    let height_bits = read(4)? as u8 + 1;
    Ok((read(width_bits)? + 1, read(height_bits)? + 1))
}

pub fn encode_avif(image: &Image, pixels: &[u8], options: SaveOptions) -> PyResult<Vec<u8>> {
    let rgb;
    let (pixels, color) = if image.mode == PixelMode::L {
        rgb = pixels.iter().flat_map(|v| [*v; 3]).collect::<Vec<_>>();
        (rgb.as_slice(), ExtendedColorType::Rgb8)
    } else {
        (pixels, color_type(image.mode))
    };
    let mut output = Vec::new();
    image::codecs::avif::AvifEncoder::new_with_speed_quality(&mut output, 11 - options.effort, options.quality)
        .write_image(pixels, image.width, image.height, color)
        .map_err(codec_error)?;
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn packed(fields: &[(u32, u8)]) -> Vec<u8> {
        let mut bits = Vec::new();
        for &(value, count) in fields {
            bits.extend((0..count).rev().map(|shift| ((value >> shift) & 1) as u8));
        }
        bits.chunks(8).map(|chunk| chunk.iter().enumerate().fold(0, |byte, (i, &bit)| byte | (bit << (7 - i)))).collect()
    }

    fn sequence(width: u32, height: u32) -> Vec<u8> {
        let data = packed(&[(0, 3), (1, 1), (1, 1), (0, 5), (15, 4), (15, 4), (width - 1, 16), (height - 1, 16), (0, 7), (1, 1), (0, 3), (1, 1)]);
        [vec![0x0a, data.len() as u8], data].concat()
    }

    #[test]
    fn reads_sequence_dimensions_from_unaligned_and_truncated_slices() {
        let header = sequence(12345, 6789);
        for offset in 0..32 {
            let mut data = vec![0; offset];
            data.extend_from_slice(&header[2..]);
            assert_eq!(sequence_dimensions(&data[offset..]).unwrap(), (12345, 6789));
            for length in 0..7 {
                assert!(sequence_dimensions(&data[offset..offset + length]).is_err());
            }
        }
    }

    #[test]
    fn reads_full_sequence_headers_with_timing_and_operating_points() {
        for timing in [false, true] {
            for decoder_model in [false, true] {
                for display_delay in [false, true] {
                    for points in [1, 32] {
                        let mut fields = vec![(2, 3), (1, 1), (0, 1), (u32::from(timing), 1)];
                        if timing {
                            fields.extend([(1, 32), (60, 32), (1, 1), (1, 3), (2, 2), (u32::from(decoder_model), 1)]);
                            if decoder_model {
                                fields.extend([(31, 5), (1, 32), (31, 5), (31, 5)]);
                            }
                        }
                        fields.extend([(u32::from(display_delay), 1), (points - 1, 5)]);
                        for point in 0..points {
                            fields.extend([(0, 12), (if point % 2 == 0 { 0 } else { 8 }, 5)]);
                            if point % 2 != 0 {
                                fields.push((1, 1));
                            }
                            if timing && decoder_model {
                                fields.push((point % 2, 1));
                                if point % 2 != 0 {
                                    fields.extend([(u32::MAX, 32), (u32::MAX, 32), (1, 1)]);
                                }
                            }
                            if display_delay {
                                fields.push((point % 2, 1));
                                if point % 2 != 0 {
                                    fields.push((15, 4));
                                }
                            }
                        }
                        fields.extend([(15, 4), (15, 4), (12344, 16), (6788, 16)]);
                        assert_eq!(sequence_dimensions(&packed(&fields)).unwrap(), (12345, 6789));
                    }
                }
            }
        }
    }

    #[test]
    fn rejects_invalid_profiles_and_timing_headers() {
        assert!(sequence_dimensions(&packed(&[(3, 3)])).unwrap_err().contains("invalid AV1 profile"));
        assert!(sequence_dimensions(&packed(&[(0, 3), (0, 1), (1, 1)])).unwrap_err().contains("invalid reduced"));
        let invalid_interval = packed(&[(0, 3), (1, 1), (0, 1), (1, 1), (1, 32), (60, 32), (1, 1), (0, 32)]);
        assert!(sequence_dimensions(&invalid_interval).unwrap_err().contains("invalid AV1 picture interval"));
    }

    #[test]
    fn bounds_all_sequence_headers_before_decoding() {
        assert!(validate_av1(&sequence(19, 13)).is_ok());
        assert!(validate_av1(&sequence(65536, 65536)).unwrap_err().contains("exceeds Blanket limit"));
        let mut multiple = sequence(65536, 65536);
        multiple.extend(sequence(19, 13));
        assert!(validate_av1(&multiple).unwrap_err().contains("exceeds Blanket limit"));
        let mut multiple = sequence(19, 13);
        multiple.extend(sequence(65536, 65536));
        assert!(validate_av1(&multiple).unwrap_err().contains("exceeds Blanket limit"));
    }

    #[test]
    fn rejects_oversized_primary_and_alpha_bitstreams_in_small_containers() {
        let source = Image::from_pixels(2, 2, PixelMode::Rgba, [30, 80, 120, 140].repeat(4), None).unwrap();
        let encoded = encode_avif(&source, source.pixel_data().unwrap(), SaveOptions { effort: 1, ..Default::default() }).unwrap();
        let context = mp4parse::read_avif(&mut std::io::Cursor::new(&encoded), mp4parse::ParseStrictness::Normal).unwrap();
        for stream in [context.primary_item_coded_data().unwrap(), context.alpha_item_coded_data().unwrap()] {
            let offset = encoded.windows(stream.len()).position(|window| window == stream).unwrap();
            let oversized = sequence(65536, 65536);
            assert!(oversized.len() <= stream.len());
            let mut data = encoded.clone();
            data[offset..offset + oversized.len()].copy_from_slice(&oversized);
            assert!(decode_avif(&data).err().unwrap().contains("exceeds Blanket limit"));
        }
    }

    #[test]
    fn rejects_malformed_obus() {
        for data in
            [vec![], vec![0x80], vec![0x0e], vec![0x0a], vec![0x0a, 255, 255, 255, 255, 255, 255, 255, 255], vec![0x0a, 4, 0], vec![0x0a, 1, 0]]
        {
            assert!(validate_av1(&data).is_err(), "{data:?}");
        }
    }
}
