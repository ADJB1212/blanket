use blanket_core::{Image, PixelMode};
use rustc_hash::FxHashMap;

use super::{dimensions, image};

pub(super) fn decode_xbm(data: &[u8]) -> Result<Image, String> {
    let text = std::str::from_utf8(data).map_err(|_| "invalid XBM text")?;
    let define = |suffix: &str| -> Result<u32, String> {
        text.lines()
            .find_map(|line| {
                let mut words = line.split_whitespace();
                (words.next() == Some("#define") && words.next().is_some_and(|v| v.ends_with(suffix))).then(|| words.next()?.parse().ok()).flatten()
            })
            .ok_or_else(|| format!("missing XBM {suffix}"))
    };
    let (w, h) = dimensions(define("_width")?, define("_height")?)?;
    let (header, body) = text.split_once('{').ok_or("missing XBM pixels")?;
    let body = body.split_once('}').ok_or("truncated XBM pixels")?.0;
    let short = header.split_whitespace().any(|v| v == "short");
    let stride = if short { w.div_ceil(16) * 2 } else { w.div_ceil(8) };
    let mut raw = Vec::new();
    for token in body.split(',').map(str::trim).filter(|v| !v.is_empty()) {
        let value =
            if let Some(v) = token.strip_prefix("0x").or_else(|| token.strip_prefix("0X")) { u16::from_str_radix(v, 16) } else { token.parse() }
                .map_err(|_| "invalid XBM pixel")?;
        if short {
            raw.extend_from_slice(&value.to_le_bytes());
        } else {
            raw.push(u8::try_from(value).map_err(|_| "invalid XBM byte")?);
        }
    }
    if raw.len() < stride * h {
        return Err("truncated XBM pixels".into());
    }
    let row_bytes = w.div_ceil(8);
    for y in 0..h {
        for x in 0..row_bytes {
            raw[y * row_bytes + x] = raw[y * stride + x].reverse_bits();
        }
    }
    let pixels = blanket_core::raster::unpack_bilevel(&raw, w, h);
    image(w, h, PixelMode::One, pixels, "XBM")
}

pub(super) fn encode_xbm(im: &Image, pixels: &[u8]) -> Result<Vec<u8>, String> {
    if im.mode != PixelMode::One {
        return Err("XBM requires mode 1".into());
    }
    let mut out = format!("#define im_width {}\n#define im_height {}\nstatic char im_bits[] = {{\n", im.width, im.height).into_bytes();
    let packed = blanket_core::raster::pack_bilevel(pixels, im.width as usize, im.height as usize);
    out.reserve(packed.len() * 5 + im.height as usize + 3);
    let hex = b"0123456789abcdef";
    for row in packed.chunks_exact((im.width as usize).div_ceil(8)) {
        for &v in row {
            let v = v.reverse_bits();
            out.extend_from_slice(&[b'0', b'x', hex[(v >> 4) as usize], hex[(v & 15) as usize], b',']);
        }
        out.push(b'\n');
    }
    out.extend_from_slice(b"};\n");
    Ok(out)
}

fn strings(data: &[u8]) -> Result<Vec<Vec<u8>>, String> {
    let mut result = Vec::new();
    let mut pos = 0;
    while pos < data.len() {
        if data[pos..].starts_with(b"/*") {
            let n = data[pos + 2..].windows(2).position(|v| v == b"*/").ok_or("unterminated XPM comment")?;
            pos += n + 4;
            continue;
        }
        if data[pos] != b'"' {
            pos += 1;
            continue;
        }
        pos += 1;
        let mut value = Vec::new();
        while pos < data.len() && data[pos] != b'"' {
            if data[pos] == b'\\' {
                pos += 1;
                if pos == data.len() {
                    return Err("truncated XPM escape".into());
                }
            }
            value.push(data[pos]);
            pos += 1;
        }
        if pos == data.len() {
            return Err("unterminated XPM string".into());
        }
        pos += 1;
        result.push(value);
    }
    Ok(result)
}

fn color(value: &str) -> Result<[u8; 4], String> {
    if value.eq_ignore_ascii_case("none") {
        return Ok([0; 4]);
    }
    if let Some(hex) = value.strip_prefix('#') {
        if !hex.is_ascii() || !matches!(hex.len(), 3 | 6 | 9 | 12) {
            return Err("invalid XPM color".into());
        }
        let n = hex.len() / 3;
        let maximum = (1_u32 << (n * 4)) - 1;
        let mut rgb = [0, 0, 0, 255];
        for (i, v) in rgb[..3].iter_mut().enumerate() {
            *v = ((u32::from_str_radix(&hex[i * n..(i + 1) * n], 16).map_err(|_| "invalid XPM color")? * 255 + maximum / 2) / maximum) as u8;
        }
        return Ok(rgb);
    }
    let rgb = match value.to_ascii_lowercase().as_str() {
        "black" => [0, 0, 0],
        "white" => [255; 3],
        "red" => [255, 0, 0],
        "green" => [0, 255, 0],
        "blue" => [0, 0, 255],
        "yellow" => [255, 255, 0],
        "cyan" => [0, 255, 255],
        "magenta" => [255, 0, 255],
        "gray" | "grey" => [190; 3],
        _ => return Err(format!("unsupported XPM color {value:?}; use hexadecimal colors")),
    };
    Ok([rgb[0], rgb[1], rgb[2], 255])
}

pub(super) fn decode_xpm(data: &[u8]) -> Result<Image, String> {
    let values = strings(data)?;
    let header = std::str::from_utf8(values.first().ok_or("missing XPM header")?).map_err(|_| "invalid XPM header")?;
    let nums = header.split_whitespace().take(4).map(str::parse::<u32>).collect::<Result<Vec<_>, _>>().map_err(|_| "invalid XPM header")?;
    if nums.len() != 4 {
        return Err("invalid XPM header".into());
    }
    let (w, h) = dimensions(nums[0], nums[1])?;
    let n = nums[2] as usize;
    let cpp = nums[3] as usize;
    if n == 0 || cpp == 0 || cpp > 64 || n > values.len().saturating_sub(1) || h > values.len().saturating_sub(1 + n) {
        return Err("invalid XPM palette or rows".into());
    }
    let mut palette = FxHashMap::default();
    let mut alpha = false;
    for entry in &values[1..1 + n] {
        let key = entry.get(..cpp).ok_or("truncated XPM key")?;
        let desc = std::str::from_utf8(&entry[cpp..]).map_err(|_| "invalid XPM color")?;
        let words = desc.split_whitespace().collect::<Vec<_>>();
        let start = words.iter().position(|&v| v == "c").ok_or("missing XPM color")? + 1;
        let end = words[start..].iter().position(|v| matches!(*v, "s" | "m" | "g" | "g4")).map_or(words.len(), |i| start + i);
        let rgba = color(&words[start..end].join(" "))?;
        alpha |= rgba[3] != 255;
        if palette.insert(key, rgba).is_some() {
            return Err("duplicate XPM key".into());
        }
    }
    let mut pixels = Vec::with_capacity(w * h * if alpha { 4 } else { 3 });
    for row in &values[1 + n..1 + n + h] {
        if row.len() != w * cpp {
            return Err("invalid XPM row length".into());
        }
        for key in row.chunks_exact(cpp) {
            let rgb = palette.get(key).ok_or("unknown XPM color")?;
            pixels.extend_from_slice(&rgb[..if alpha { 4 } else { 3 }]);
        }
    }
    image(w, h, if alpha { PixelMode::Rgba } else { PixelMode::Rgb }, pixels, "XPM")
}

pub(super) fn encode_xpm(im: &Image, pixels: &[u8]) -> Result<Vec<u8>, String> {
    if !matches!(im.mode, PixelMode::L | PixelMode::Rgb | PixelMode::Rgba) {
        return Err("XPM requires L, RGB, or RGBA".into());
    }
    let mut palette = FxHashMap::default();
    let mut colors = Vec::new();
    let mut indices = Vec::with_capacity(im.width as usize * im.height as usize);
    for p in pixels.chunks_exact(im.mode.channels()) {
        let rgba = match im.mode {
            PixelMode::L => [p[0], p[0], p[0], 255],
            PixelMode::Rgb => [p[0], p[1], p[2], 255],
            _ => [p[0], p[1], p[2], p[3]],
        };
        if !matches!(rgba[3], 0 | 255) {
            return Err("XPM supports binary transparency only".into());
        }
        let index = *palette.entry(rgba).or_insert_with(|| {
            colors.push(rgba);
            colors.len() - 1
        });
        indices.push(index);
    }
    let cpp = format!("{:x}", colors.len() - 1).len();
    let mut out = format!("/* XPM */\nstatic char *im[] = {{\n\"{} {} {} {cpp}\",\n", im.width, im.height, colors.len());
    for (i, p) in colors.iter().enumerate() {
        let c = if p[3] == 0 { "None".into() } else { format!("#{:02x}{:02x}{:02x}", p[0], p[1], p[2]) };
        out.push_str(&format!("\"{i:0cpp$x} c {c}\",\n"));
    }
    for row in indices.chunks_exact(im.width as usize) {
        out.push('"');
        for i in row {
            out.push_str(&format!("{i:0cpp$x}"));
        }
        out.push_str("\",\n");
    }
    out.push_str("};\n");
    Ok(out.into_bytes())
}
