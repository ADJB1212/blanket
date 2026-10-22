use blanket_core::{Image, PixelMode};
use rustc_hash::FxHashMap;

use super::{bytes, dimensions, image};

fn samples(w: usize, h: usize, data: &[u8], bits: i32, big: bool, flip: bool, format: &str) -> Result<Image, String> {
    let size = bits.unsigned_abs() as usize / 8;
    if !matches!(bits, 8 | 16 | 32 | -32 | -64) {
        return Err("unsupported sample type".into());
    }
    let source = bytes(data, 0, w * h * size)?;
    let mut pixels = Vec::with_capacity(w * h * size);
    for y in 0..h {
        let sy = if flip { h - 1 - y } else { y };
        for v in source[sy * w * size..(sy + 1) * w * size].chunks_exact(size) {
            match bits {
                8 => pixels.push(v[0]),
                16 => {
                    let v = v.try_into().unwrap();
                    pixels.extend_from_slice(&if big { u16::from_be_bytes(v) } else { u16::from_le_bytes(v) }.to_le_bytes());
                }
                32 | -32 => {
                    let v = v.try_into().unwrap();
                    pixels.extend_from_slice(&if big { u32::from_be_bytes(v) } else { u32::from_le_bytes(v) }.to_le_bytes());
                }
                -64 => {
                    let v = v.try_into().unwrap();
                    let v = if big { f64::from_be_bytes(v) } else { f64::from_le_bytes(v) };
                    pixels.extend_from_slice(&(v as f32).to_le_bytes());
                }
                _ => unreachable!(),
            }
        }
    }
    let mut im = match bits {
        8 => return image(w, h, PixelMode::L, pixels, format),
        16 => Image::from_integer_bytes(w as u32, h as u32, PixelMode::I16, pixels),
        32 => Image::from_integer_bytes(w as u32, h as u32, PixelMode::I, pixels),
        _ => Image::from_float_bytes(w as u32, h as u32, pixels),
    }
    .map_err(|e| e.to_string())?;
    im.format = Some(format.into());
    Ok(im)
}

pub(super) fn decode_fits(data: &[u8]) -> Result<Image, String> {
    let mut headers = FxHashMap::default();
    let mut pos = 0;
    loop {
        let card = bytes(data, pos, 80)?;
        pos += 80;
        let key = std::str::from_utf8(&card[..8]).map_err(|_| "invalid FITS header")?.trim();
        if key == "END" {
            break;
        }
        if card[8] == b'=' {
            let value = std::str::from_utf8(&card[10..])
                .map_err(|_| "invalid FITS value")?
                .split('/')
                .next()
                .unwrap()
                .trim();
            headers.insert(key, value);
        }
    }
    if headers.get("SIMPLE") != Some(&"T") {
        return Err("invalid FITS primary header".into());
    }
    let number = |key: &str| -> Result<i32, String> {
        headers
            .get(key)
            .ok_or_else(|| format!("missing FITS {key}"))?
            .parse()
            .map_err(|_| format!("invalid FITS {key}"))
    };
    let axes = number("NAXIS")?;
    if axes < 1 {
        return Err("FITS primary HDU has no image".into());
    }
    let width = number("NAXIS1")?;
    let height = if axes == 1 { 1 } else { number("NAXIS2")? };
    let (w, h) = dimensions(
        width.try_into().map_err(|_| "invalid FITS width")?,
        height.try_into().map_err(|_| "invalid FITS height")?,
    )?;
    let start = pos.next_multiple_of(2880);
    let source = bytes(data, start, data.len().saturating_sub(start))?;
    samples(w, h, source, number("BITPIX")?, true, true, "FITS")
}

pub(super) fn spider_header(data: &[u8]) -> Option<(bool, usize, u32, u32)> {
    if data.len() < 108 {
        return None;
    }
    for big in [true, false] {
        let f = |i: usize| {
            let v = data[(i - 1) * 4..i * 4].try_into().unwrap();
            if big { f32::from_be_bytes(v) } else { f32::from_le_bytes(v) }
        };
        if [1, 2, 5, 12, 13, 22, 23, 24, 27]
            .iter()
            .any(|&i| !f(i).is_finite() || f(i).fract() != 0.0 || f(i) < 0.0)
        {
            continue;
        }
        if f(5) != 1.0 || f(1) != 1.0 || f(12) < 1.0 || f(2) < 1.0 || f(22) < 108.0 || f(23) != f(12) * 4.0 || f(22) != f(13) * f(23) {
            continue;
        }
        return Some((big, f(22) as usize, f(12) as u32, f(2) as u32));
    }
    None
}

pub(super) fn decode_spider(data: &[u8]) -> Result<Image, String> {
    let (big, header, w, h) = spider_header(data).ok_or("invalid SPIDER header")?;
    let (w, h) = dimensions(w, h)?;
    let v = bytes(data, 92, 4)?.try_into().unwrap();
    let stack = if big { f32::from_be_bytes(v) } else { f32::from_le_bytes(v) };
    let start = header
        .checked_mul(if stack > 0.0 { 2 } else { 1 })
        .ok_or("invalid SPIDER header length")?;
    samples(w, h, bytes(data, start, w * h * 4)?, -32, big, false, "SPIDER")
}
