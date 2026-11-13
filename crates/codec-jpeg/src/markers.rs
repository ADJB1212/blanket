use std::borrow::Cow;

use memchr::memchr;
use pyo3::exceptions::PyOSError;
use pyo3::prelude::*;

pub(super) fn separate(data: Vec<u8>, streamtype: i32, coding: super::JpegCoding) -> PyResult<Vec<u8>> {
    if !matches!(streamtype, 1 | 2) {
        return Ok(data);
    }
    if !data.starts_with(&[0xff, 0xd8]) {
        return Err(PyOSError::new_err("invalid encoded JPEG signature"));
    }
    let mut output = vec![0xff, 0xd8];
    let mut position = 2;
    while position < data.len() {
        let start = position;
        if data.get(position) != Some(&0xff) {
            return Err(PyOSError::new_err("invalid encoded JPEG marker"));
        }
        while data.get(position) == Some(&0xff) {
            position += 1;
        }
        let marker = *data.get(position).ok_or_else(|| PyOSError::new_err("truncated encoded JPEG"))?;
        position += 1;
        if marker == 0xd9 {
            break;
        }
        let length = data.get(position..position + 2).ok_or_else(|| PyOSError::new_err("truncated encoded JPEG"))?;
        let length = usize::from(u16::from_be_bytes([length[0], length[1]]));
        if length < 2 || position + length > data.len() {
            return Err(PyOSError::new_err("invalid encoded JPEG segment"));
        }
        position += length;
        if marker == 0xda {
            while position + 1 < data.len() {
                let Some(offset) = memchr(0xff, &data[position..data.len() - 1]) else {
                    position = data.len() - 1;
                    break;
                };
                position += offset;
                if data[position + 1] != 0 && !(0xd0..=0xd7).contains(&data[position + 1]) {
                    break;
                }
                position += 2;
            }
        }
        let table = matches!(marker, 0xc4 | 0xcc | 0xdb);
        if table == (streamtype == 1) || (streamtype == 2 && (coding.optimize || coding.progressive) && marker == 0xc4) {
            output.extend_from_slice(&data[start..position]);
        }
    }
    output.extend_from_slice(&[0xff, 0xd9]);
    Ok(output)
}

pub(super) fn combine(data: &[u8]) -> Cow<'_, [u8]> {
    if !data.starts_with(&[0xff, 0xd8]) {
        return Cow::Borrowed(data);
    }
    let mut position = 2;
    let mut tables = false;
    while position + 1 < data.len() {
        let start = position;
        if data[position] != 0xff {
            break;
        }
        while data.get(position) == Some(&0xff) {
            position += 1;
        }
        let Some(&marker) = data.get(position) else {
            break;
        };
        position += 1;
        if marker == 0xd9 && tables && data.get(position..position + 2) == Some(&[0xff, 0xd8]) {
            let mut output = Vec::with_capacity(data.len() - 4);
            output.extend_from_slice(&data[..start]);
            output.extend_from_slice(&data[position + 2..]);
            return Cow::Owned(output);
        }
        if matches!(marker, 0xc0..=0xc3 | 0xc5..=0xcb | 0xcd..=0xcf | 0xd8..=0xda) {
            break;
        }
        tables |= matches!(marker, 0xc4 | 0xcc | 0xdb);
        let Some(length) = data.get(position..position + 2) else {
            break;
        };
        let length = usize::from(u16::from_be_bytes([length[0], length[1]]));
        if length < 2 || length > data.len() - position {
            break;
        }
        position += length;
    }
    Cow::Borrowed(data)
}

#[cfg(test)]
mod tests {
    use blanket_codec_common::SaveOptions;
    use blanket_core::{Image, PixelMode};

    use super::*;
    use crate::{JpegCoding, decode_jpeg, encode_jpeg_variant};

    #[test]
    fn abbreviated_streams_preserve_progressive_and_baseline_pixels() {
        let pixels = (0..33 * 19 * 3).map(|i| (i * 137 + i / 7) as u8).collect();
        let image = Image::from_pixels(33, 19, PixelMode::Rgb, pixels, None).unwrap();
        for coding in [JpegCoding::BASELINE, JpegCoding { progressive: true, optimize: true }] {
            let full = encode_jpeg_variant(&image, SaveOptions { .. }, coding).unwrap();
            let tables = separate(full.clone(), 1, coding).unwrap();
            let body = separate(full.clone(), 2, coding).unwrap();
            let expected = decode_jpeg(&full).unwrap();
            assert!(decode_jpeg(&body).is_err());
            let combined = [tables.as_slice(), body.as_slice()].concat();
            let actual = decode_jpeg(&combined).unwrap();
            assert_eq!(actual.pixel_data().unwrap(), expected.pixel_data().unwrap());
        }
    }

    #[test]
    fn malformed_and_incomplete_table_headers_are_not_joined() {
        let table = [0xff, 0xd8, 0xff, 0xdb, 0, 3, 0, 0xff, 0xd9];
        for end in 0..table.len() {
            let data = [table[..end].as_ref(), &[0xff, 0xd8]].concat();
            assert!(matches!(combine(&data), Cow::Borrowed(_)));
        }
        for length in [0u16, 1, 65535] {
            let mut data = table.to_vec();
            data[4..6].copy_from_slice(&length.to_be_bytes());
            data.extend_from_slice(&[0xff, 0xd8]);
            assert!(matches!(combine(&data), Cow::Borrowed(_)));
        }
        let frame = [0xff, 0xd8, 0xff, 0xc0, 0, 3, 0, 0xff, 0xd9, 0xff, 0xd8];
        assert!(matches!(combine(&frame), Cow::Borrowed(_)));
    }
}
