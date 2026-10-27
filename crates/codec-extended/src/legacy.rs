mod scientific;
mod x11;

use std::borrow::Cow;

use blanket_codec_common::validate_dimensions;
use blanket_core::{Image, PixelMode};

fn bytes(data: &[u8], offset: usize, length: usize) -> Result<&[u8], String> {
    data.get(offset..offset.checked_add(length).ok_or("invalid image offset")?).ok_or_else(|| "truncated image data".into())
}

fn word(data: &[u8], offset: usize, big: bool) -> Result<u16, String> {
    let v = bytes(data, offset, 2)?.try_into().unwrap();
    Ok(if big { u16::from_be_bytes(v) } else { u16::from_le_bytes(v) })
}

fn long(data: &[u8], offset: usize, big: bool) -> Result<u32, String> {
    let v = bytes(data, offset, 4)?.try_into().unwrap();
    Ok(if big { u32::from_be_bytes(v) } else { u32::from_le_bytes(v) })
}

fn dimensions(width: u32, height: u32) -> Result<(usize, usize), String> {
    validate_dimensions(width, height)?;
    if width == 0 || height == 0 {
        return Err("empty image".into());
    }
    Ok((width as usize, height as usize))
}

fn image(width: usize, height: usize, mode: PixelMode, pixels: Vec<u8>, format: &str) -> Result<Image, String> {
    Image::from_pixels(width as u32, height as u32, mode, pixels, Some(format.into())).map_err(|e| e.to_string())
}

fn indexed(width: usize, height: usize, pixels: Vec<u8>, palette: Vec<u8>, format: &str) -> Result<Image, String> {
    let mut result = image(width, height, PixelMode::L, pixels, format)?;
    result.palette = Some((PixelMode::Rgb, palette));
    Ok(result)
}

pub fn detect(data: &[u8]) -> Option<&'static str> {
    if data.starts_with(b"\x0a") && data.get(1).is_some_and(|v| matches!(v, 0 | 2 | 3 | 5)) {
        Some("PCX")
    } else if data.starts_with(b"\x01\xda") {
        Some("SGI")
    } else if data.starts_with(b"DanM") || data.starts_with(b"LinS") {
        Some("MSP")
    } else if data.starts_with(b"\x59\xa6\x6a\x95") {
        Some("SUN")
    } else if data.starts_with(b"\x80\xe8\0\0") {
        Some("PIXAR")
    } else if data.starts_with(b"SIMPLE  =") {
        Some("FITS")
    } else if data.starts_with(b"/* XPM */") {
        Some("XPM")
    } else if data.get(..4096.min(data.len())).is_some_and(|v| v.windows(6).any(|w| w == b"_width") && v.windows(7).any(|w| w == b"_height")) {
        Some("XBM")
    } else if scientific::spider_header(data).is_some() {
        Some("SPIDER")
    } else {
        None
    }
}

pub fn decode(data: &[u8], format: &str) -> Result<Image, String> {
    match format {
        "PCX" => decode_pcx(data),
        "SGI" => decode_sgi(data),
        "MSP" => decode_msp(data),
        "SUN" => decode_sun(data),
        "XBM" => x11::decode_xbm(data),
        "XPM" => x11::decode_xpm(data),
        "FITS" => scientific::decode_fits(data),
        "SPIDER" => scientific::decode_spider(data),
        "PIXAR" => {
            let (w, h) = dimensions(u32::from(word(data, 418, false)?), u32::from(word(data, 416, false)?))?;
            if word(data, 424, false)? != 14 || word(data, 426, false)? != 2 {
                return Err("unsupported Pixar mode".into());
            }
            image(w, h, PixelMode::Rgb, bytes(data, 1024, w * h * 3)?.to_vec(), format)
        }
        _ => Err(format!("{format} is a write-only format")),
    }
}

pub fn encode(im: &Image, pixels: &[u8], format: &str) -> Result<Vec<u8>, String> {
    dimensions(im.width, im.height)?;
    match format {
        "PCX" => encode_pcx(im, pixels),
        "SGI" => encode_sgi(im, pixels),
        "XBM" => x11::encode_xbm(im, pixels),
        "XPM" => x11::encode_xpm(im, pixels),
        "MSP" | "PALM" => encode_bitmap(im, pixels, format),
        _ => Err(format!("{format} is a read-only format")),
    }
}

fn decode_pcx(data: &[u8]) -> Result<Image, String> {
    bytes(data, 0, 128)?;
    let w = i32::from(word(data, 8, false)?) - i32::from(word(data, 4, false)?) + 1;
    let h = i32::from(word(data, 10, false)?) - i32::from(word(data, 6, false)?) + 1;
    let (w, h) = dimensions(w.try_into().map_err(|_| "invalid PCX width")?, h.try_into().map_err(|_| "invalid PCX height")?)?;
    let bits = data[3];
    let planes = usize::from(data[65]);
    if !matches!((bits, planes), (1, 1 | 2 | 4) | (8, 1 | 3)) || data[2] > 1 {
        return Err("unsupported PCX layout".into());
    }
    let stride = usize::from(word(data, 66, false)?);
    if stride < (w * usize::from(bits)).div_ceil(8) {
        return Err("invalid PCX stride".into());
    }
    let mut pixels = if bits == 1 && planes == 1 { Vec::new() } else { vec![0; w * h * if planes == 3 { 3 } else { 1 }] };
    let mut pos = 128;
    let mut row = vec![0; stride * planes];
    let mut packed = Vec::with_capacity(if bits == 1 && planes == 1 { w.div_ceil(8) * h } else { 0 });
    for y in 0..h {
        let mut written = 0;
        if data[2] == 0 {
            let source = bytes(data, pos, row.len())?;
            row.copy_from_slice(source);
            pos += row.len();
            written = row.len();
        }
        while written < row.len() {
            let source = data.get(pos..).ok_or("truncated image data")?;
            let length = crate::pixels::pcx_literals(&source[..source.len().min(row.len() - written)]);
            row[written..written + length].copy_from_slice(&source[..length]);
            written += length;
            pos += length;
            if written == row.len() {
                break;
            }
            let run = bytes(data, pos, 2)?;
            let count = usize::from(run[0] & 63);
            if count == 0 || written + count > row.len() {
                return Err("invalid PCX run".into());
            }
            row[written..written + count].fill(run[1]);
            written += count;
            pos += 2;
        }
        if bits == 1 && planes == 1 {
            packed.extend_from_slice(&row[..w.div_ceil(8)]);
        } else if bits == 1 {
            for x in 0..w {
                let value = (0..planes).fold(0, |v, c| v | ((row[c * stride + x / 8] >> (7 - x % 8)) & 1) << c);
                pixels[y * w + x] = value;
            }
        } else if planes == 3 {
            let output = pixels[y * w * 3..(y + 1) * w * 3].as_chunks_mut::<3>().0;
            let bands = [&row[..w], &row[stride..stride + w], &row[2 * stride..2 * stride + w]];
            let done = blanket_core::pixels::merge(bands, output);
            for (x, pixel) in output.iter_mut().enumerate().skip(done) {
                for c in 0..3 {
                    pixel[c] = bands[c][x];
                }
            }
        } else {
            pixels[y * w..(y + 1) * w].copy_from_slice(&row[..w]);
        }
    }
    if bits == 1 && planes == 1 {
        pixels = blanket_core::raster::unpack_bilevel(&packed, w, h);
    }
    if bits == 1 && planes > 1 {
        return indexed(w, h, pixels, data[16..64].to_vec(), "PCX");
    }
    if bits == 8 && planes == 1 && data.len() >= 769 && data[data.len() - 769] == 12 {
        let palette = &data[data.len() - 768..];
        if palette.as_chunks::<3>().0.iter().enumerate().any(|(i, v)| *v != [i as u8; 3]) {
            return indexed(w, h, pixels, palette.to_vec(), "PCX");
        }
    }
    image(
        w,
        h,
        if bits == 1 {
            PixelMode::One
        } else if planes == 3 {
            PixelMode::Rgb
        } else {
            PixelMode::L
        },
        pixels,
        "PCX",
    )
}

fn encode_pcx(im: &Image, pixels: &[u8]) -> Result<Vec<u8>, String> {
    let (w, h) = (im.width as usize, im.height as usize);
    if w > 65535 || h > 65535 || !matches!(im.mode, PixelMode::One | PixelMode::L | PixelMode::Rgb) {
        return Err("unsupported PCX mode or dimensions".into());
    }
    let bits = if im.mode == PixelMode::One { 1 } else { 8 };
    let planes = if im.mode == PixelMode::Rgb { 3 } else { 1 };
    let stride = (w * bits).div_ceil(8).next_multiple_of(2);
    let stride16 = u16::try_from(stride).map_err(|_| "PCX row is too wide")?;
    let mut out = vec![0; 128];
    out[..4].copy_from_slice(&[10, if bits == 1 { 2 } else { 5 }, 1, bits as u8]);
    out[8..10].copy_from_slice(&((w - 1) as u16).to_le_bytes());
    out[10..12].copy_from_slice(&((h - 1) as u16).to_le_bytes());
    out[65] = planes as u8;
    out[66..68].copy_from_slice(&stride16.to_le_bytes());
    let mut row = vec![0; stride];
    for y in 0..h {
        for c in 0..planes {
            let source = &pixels[y * w * planes..(y + 1) * w * planes];
            if bits == 1 {
                blanket_core::pixels::pack_bilevel::<false>(source, &mut row);
            } else if planes == 1 {
                row[..w].copy_from_slice(source);
            } else {
                for (dst, pixel) in row[..w].iter_mut().zip(source.as_chunks::<3>().0) {
                    *dst = pixel[c];
                }
            }
            let mut x = 0;
            while x < row.len() {
                let v = row[x];
                let mut count = 1;
                while count < 63 && x + count < row.len() && row[x + count] == v {
                    count += 1;
                }
                if count > 1 || v >= 192 {
                    out.extend_from_slice(&[192 | count as u8, v]);
                } else {
                    out.push(v);
                }
                x += count;
            }
        }
    }
    if bits == 8 && planes == 1 {
        out.push(12);
        if let Some((_, palette)) = &im.palette {
            let start = out.len();
            out.extend_from_slice(palette);
            out.resize(start + 768, 0);
        } else {
            for i in 0..=255 {
                out.extend_from_slice(&[i; 3]);
            }
        }
    }
    Ok(out)
}

fn decode_sgi(data: &[u8]) -> Result<Image, String> {
    bytes(data, 0, 512)?;
    let (w, h) = dimensions(u32::from(word(data, 6, true)?), u32::from(word(data, 8, true)?))?;
    let channels = usize::from(word(data, 10, true)?);
    let bpc = usize::from(data[3]);
    if !matches!(channels, 1 | 3 | 4) || !matches!(bpc, 1 | 2) || data[2] > 1 || long(data, 104, true)? != 0 {
        return Err("unsupported SGI layout".into());
    }
    let mode = match channels {
        1 => PixelMode::L,
        3 => PixelMode::Rgb,
        _ => PixelMode::Rgba,
    };
    if data[2] == 0 && bpc == 1 {
        let source = bytes(data, 512, w * h * channels)?;
        let mut pixels = vec![0; source.len()];
        for (y, row) in pixels.chunks_exact_mut(w * channels).enumerate() {
            let start = (h - 1 - y) * w;
            match channels {
                1 => row.copy_from_slice(&source[start..start + w]),
                3 => merge_sgi_row::<3>(source, row, start, w * h),
                4 => merge_sgi_row::<4>(source, row, start, w * h),
                _ => unreachable!(),
            }
        }
        return image(w, h, mode, pixels, "SGI");
    }
    let mut pixels = vec![0; w * h * channels * bpc];
    for c in 0..channels {
        for y in 0..h {
            let row = if data[2] == 0 {
                bytes(data, 512 + (c * h + y) * w * bpc, w * bpc)?.to_vec()
            } else {
                let offset = long(data, 512 + (c * h + y) * 4, true)? as usize;
                let length = long(data, 512 + h * channels * 4 + (c * h + y) * 4, true)? as usize;
                let source = bytes(data, offset, length)?;
                let mut row = Vec::with_capacity(w * bpc);
                let mut pos = 0;
                loop {
                    let control = if bpc == 1 { u16::from(bytes(source, pos, 1)?[0]) } else { word(source, pos, true)? };
                    pos += bpc;
                    let count = usize::from(control & 127);
                    if count == 0 {
                        break;
                    }
                    if row.len() + count * bpc > w * bpc {
                        return Err("invalid SGI run".into());
                    }
                    if control & 128 != 0 {
                        row.extend_from_slice(bytes(source, pos, count * bpc)?);
                        pos += count * bpc;
                    } else {
                        let value = bytes(source, pos, bpc)?;
                        pos += bpc;
                        for _ in 0..count {
                            row.extend_from_slice(value);
                        }
                    }
                }
                if row.len() != w * bpc {
                    return Err("truncated SGI row".into());
                }
                row
            };
            for x in 0..w {
                let dst = ((h - 1 - y) * w * channels + x * channels + c) * bpc;
                if bpc == 1 {
                    pixels[dst] = row[x];
                } else {
                    pixels[dst..dst + 2].copy_from_slice(&[row[x * 2 + 1], row[x * 2]]);
                }
            }
        }
    }
    if bpc == 2 {
        Image::from_sample_bytes(w as u32, h as u32, mode, pixels, 16, Some("SGI".into())).map_err(|e| e.to_string())
    } else {
        image(w, h, mode, pixels, "SGI")
    }
}

fn merge_sgi_row<const C: usize>(source: &[u8], output: &mut [u8], start: usize, plane: usize) {
    let output = output.as_chunks_mut::<C>().0;
    let bands = std::array::from_fn(|c| &source[c * plane + start..][..output.len()]);
    let done = blanket_core::pixels::merge(bands, output);
    for (x, pixel) in output.iter_mut().enumerate().skip(done) {
        for c in 0..C {
            pixel[c] = bands[c][x];
        }
    }
}

fn encode_sgi(im: &Image, pixels: &[u8]) -> Result<Vec<u8>, String> {
    if !matches!(im.mode, PixelMode::L | PixelMode::Rgb | PixelMode::Rgba) || im.width > 65535 || im.height > 65535 {
        return Err("unsupported SGI mode or dimensions".into());
    }
    let (w, h, c) = (im.width as usize, im.height as usize, im.mode.channels());
    let mut out = vec![0; 512 + pixels.len()];
    out[..4].copy_from_slice(&[1, 218, 0, 1]);
    for (offset, value) in [(4, if c == 1 { 2 } else { 3 }), (6, w), (8, h), (10, c)] {
        out[offset..offset + 2].copy_from_slice(&(value as u16).to_be_bytes());
    }
    out[16..20].copy_from_slice(&255_u32.to_be_bytes());
    for band in 0..c {
        for (y, row) in out[512 + band * w * h..512 + (band + 1) * w * h].chunks_exact_mut(w).enumerate() {
            let source = &pixels[(h - 1 - y) * w * c..][..w * c];
            match c {
                1 => row.copy_from_slice(source),
                3 => crate::pixels::extract_channel::<3>(source, row, band),
                4 => crate::pixels::extract_channel::<4>(source, row, band),
                _ => unreachable!(),
            }
        }
    }
    Ok(out)
}

fn decode_msp(data: &[u8]) -> Result<Image, String> {
    bytes(data, 0, 32)?;
    if (0..32).step_by(2).try_fold(0, |v, i| word(data, i, false).map(|w| v ^ w))? != 0 {
        return Err("invalid MSP checksum".into());
    }
    let (w, h) = dimensions(u32::from(word(data, 4, false)?), u32::from(word(data, 6, false)?))?;
    let stride = w.div_ceil(8);
    let mut packed = Vec::new();
    if data.starts_with(b"DanM") {
        packed.extend_from_slice(bytes(data, 32, stride * h)?);
    } else {
        let mut pos = 32 + h * 2;
        bytes(data, 32, h * 2)?;
        for y in 0..h {
            let length = usize::from(word(data, 32 + y * 2, false)?);
            if length == 0 {
                packed.resize(packed.len() + stride, 255);
                continue;
            }
            let row = bytes(data, pos, length)?;
            pos += length;
            let mut p = 0;
            let end = packed.len() + stride;
            while p < row.len() {
                let n = row[p];
                p += 1;
                if n == 0 {
                    let run = bytes(row, p, 2)?;
                    p += 2;
                    if packed.len() + usize::from(run[0]) > end {
                        return Err("invalid MSP run".into());
                    }
                    packed.resize(packed.len() + usize::from(run[0]), run[1]);
                } else {
                    let n = usize::from(n);
                    if packed.len() + n > end {
                        return Err("invalid MSP run".into());
                    }
                    packed.extend_from_slice(bytes(row, p, n)?);
                    p += n;
                }
            }
            if packed.len() != end {
                return Err("truncated MSP row".into());
            }
        }
    }
    let pixels = blanket_core::raster::unpack_bilevel(&packed, w, h);
    image(w, h, PixelMode::One, pixels, "MSP")
}

fn encode_bitmap(im: &Image, pixels: &[u8], format: &str) -> Result<Vec<u8>, String> {
    let (w, h) = (im.width as usize, im.height as usize);
    if w > 65535 || h > 65535 || (im.mode != PixelMode::One && !(format == "PALM" && im.mode == PixelMode::L)) {
        return Err(format!("unsupported {format} mode or dimensions"));
    }
    let mut out;
    let stride;
    if format == "MSP" {
        let mut header = [0_u16; 16];
        header[..10].copy_from_slice(&[0x6144, 0x4d6e, w as u16, h as u16, 1, 1, 1, 1, w as u16, h as u16]);
        header[12] = header.iter().fold(0, |v, w| v ^ w);
        out = header.into_iter().flat_map(u16::to_le_bytes).collect::<Vec<_>>();
        stride = w.div_ceil(8);
    } else {
        let bpp = if im.mode == PixelMode::One { 1 } else { 8 };
        stride = (w * bpp).div_ceil(16) * 2;
        let stride16 = u16::try_from(stride).map_err(|_| "PALM row is too wide")?;
        out = vec![0; 16];
        for (offset, value) in [(0, w as u16), (2, h as u16), (4, stride16), (6, if bpp == 8 { 0x4000 } else { 0 })] {
            out[offset..offset + 2].copy_from_slice(&value.to_be_bytes());
        }
        out[8] = bpp as u8;
        out[9] = u8::from(bpp == 8);
        out[13] = 255;
        if bpp == 8 {
            out.extend_from_slice(&256_u16.to_be_bytes());
            for i in 0..256 {
                out.push(i as u8);
                if let Some((_, p)) = &im.palette {
                    out.extend_from_slice(p.get(i * 3..i * 3 + 3).unwrap_or(&[0; 3]));
                } else {
                    out.extend_from_slice(&[i as u8; 3]);
                }
            }
        }
    }
    let start = out.len();
    out.resize(start + stride * h, 0);
    for (source, row) in pixels.chunks_exact(w).zip(out[start..].chunks_exact_mut(stride)) {
        if im.mode == PixelMode::One {
            if format == "PALM" {
                blanket_core::pixels::pack_bilevel::<true>(source, row);
            } else {
                blanket_core::pixels::pack_bilevel::<false>(source, row);
            }
        } else {
            row[..w].copy_from_slice(source);
        }
    }
    Ok(out)
}

fn decode_sun(data: &[u8]) -> Result<Image, String> {
    let (w, h) = dimensions(long(data, 4, true)?, long(data, 8, true)?)?;
    let depth = long(data, 12, true)? as usize;
    let kind = long(data, 20, true)?;
    let map = long(data, 24, true)?;
    let maplen = long(data, 28, true)? as usize;
    if !matches!(depth, 1 | 4 | 8 | 24 | 32) || kind > 5 || maplen > 768 || (maplen != 0 && (map != 1 || !maplen.is_multiple_of(3))) {
        return Err("unsupported Sun raster layout".into());
    }
    let palette = bytes(data, 32, maplen)?;
    let stride = (w * depth).div_ceil(16) * 2;
    let source = bytes(data, 32 + maplen, data.len().saturating_sub(32 + maplen))?;
    let raw = if kind == 2 {
        let mut raw = Vec::with_capacity(stride * h);
        let mut pos = 0;
        while raw.len() < stride * h {
            let v = bytes(source, pos, 1)?[0];
            pos += 1;
            let (count, value) = if v == 128 {
                let n = bytes(source, pos, 1)?[0];
                pos += 1;
                if n == 0 {
                    (1, 128)
                } else {
                    let v = bytes(source, pos, 1)?[0];
                    pos += 1;
                    (usize::from(n) + 1, v)
                }
            } else {
                (1, v)
            };
            if raw.len() + count > stride * h {
                return Err("invalid Sun raster run".into());
            }
            raw.resize(raw.len() + count, value);
        }
        Cow::Owned(raw)
    } else {
        Cow::Borrowed(bytes(source, 0, stride * h)?)
    };
    let channels = if depth >= 24 { 3 } else { 1 };
    if depth == 1 {
        let row_bytes = w.div_ceil(8);
        let packed = if row_bytes == stride {
            Cow::Borrowed(raw.as_ref())
        } else {
            Cow::Owned(raw.chunks_exact(stride).flat_map(|row| row[..row_bytes].iter().copied()).collect::<Vec<_>>())
        };
        return image(w, h, PixelMode::One, blanket_core::raster::unpack_bilevel_inverted(&packed, w, h), "SUN");
    }
    let mut pixels = vec![0; w * h * channels];
    for (row, output) in raw.chunks_exact(stride).zip(pixels.chunks_exact_mut(w * channels)) {
        match depth {
            8 => output.copy_from_slice(&row[..w]),
            24 if kind == 3 => output.copy_from_slice(&row[..w * 3]),
            24 => crate::pixels::swap_red_blue::<3>(&row[..w * 3], output),
            _ => {
                for (x, pixel) in output.chunks_exact_mut(channels).enumerate() {
                    match depth {
                        4 => {
                            let v = (row[x / 2] >> (if x % 2 == 0 { 4 } else { 0 })) & 15;
                            pixel[0] = if maplen == 0 { v * 17 } else { v };
                        }
                        _ => {
                            let p = &row[x * (depth / 8)..];
                            if kind == 3 {
                                pixel.copy_from_slice(&p[..3]);
                            } else {
                                pixel.copy_from_slice(&[p[2], p[1], p[0]]);
                            }
                        }
                    }
                }
            }
        };
    }
    if maplen > 0 && matches!(depth, 4 | 8) {
        let n = maplen / 3;
        let p = (0..n).flat_map(|i| [palette[i], palette[n + i], palette[2 * n + i]]).collect();
        indexed(w, h, pixels, p, "SUN")
    } else {
        image(
            w,
            h,
            if depth == 1 {
                PixelMode::One
            } else if depth >= 24 {
                PixelMode::Rgb
            } else {
                PixelMode::L
            },
            pixels,
            "SUN",
        )
    }
}
