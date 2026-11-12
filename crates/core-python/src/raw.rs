use std::borrow::Cow;

use blanket_core::PixelMode;
use pyo3::exceptions::{PyMemoryError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

pub(crate) fn decode<'a>(data: &'a [u8], decoder: &str) -> PyResult<Cow<'a, [u8]>> {
    if decoder == "raw" {
        return Ok(Cow::Borrowed(data));
    }
    let mut output = Vec::new();
    match decoder {
        "hex" => {
            let mut high = None;
            for &byte in data.iter().filter(|byte| !byte.is_ascii_whitespace()) {
                let value = (byte as char).to_digit(16).ok_or_else(|| PyValueError::new_err("invalid hex image data"))? as u8;
                if let Some(first) = high.take() {
                    output.push(first * 16 + value);
                } else {
                    high = Some(value);
                }
            }
            if high.is_some() {
                return Err(PyValueError::new_err("not enough image data"));
            }
        }
        "packbits" => {
            let mut position = 0;
            while position < data.len() {
                let count = data[position] as i8;
                position += 1;
                if count >= 0 {
                    let end = position + count as usize + 1;
                    let literal = data.get(position..end).ok_or_else(|| PyValueError::new_err("not enough image data"))?;
                    output.try_reserve(literal.len()).map_err(|error| PyMemoryError::new_err(error.to_string()))?;
                    output.extend_from_slice(literal);
                    position = end;
                } else if count != -128 {
                    let byte = *data.get(position).ok_or_else(|| PyValueError::new_err("not enough image data"))?;
                    position += 1;
                    let count = (1 - i16::from(count)) as usize;
                    output.try_reserve(count).map_err(|error| PyMemoryError::new_err(error.to_string()))?;
                    output.resize(output.len() + count, byte);
                }
            }
        }
        _ => return Err(PyValueError::new_err("unsupported raw decoder")),
    }
    Ok(Cow::Owned(output))
}

struct Layout {
    image_row: usize,
    raw_row: usize,
    stride: usize,
    bgr_channels: Option<usize>,
}

impl Layout {
    fn new(mode: PixelMode, width: usize, depth: u8, rawmode: Option<&str>, stride: usize, orientation: i8) -> PyResult<Self> {
        if !matches!(orientation, -1 | 1) {
            return Err(PyValueError::new_err("orientation must be 1 or -1"));
        }
        let rawmode = rawmode.unwrap_or(mode.as_str());
        let bgr_channels = match (mode, rawmode, depth) {
            (PixelMode::Rgb, "BGR", 8) => Some(3),
            (PixelMode::Rgb, "BGRX", 8) | (PixelMode::Rgba, "BGRA", 8) => Some(4),
            _ if rawmode == mode.as_str() || mode == PixelMode::L && rawmode == "P" => None,
            _ => return Err(PyValueError::new_err(format!("unsupported raw mode {rawmode} for {} at {depth} bits", mode.as_str()))),
        };
        let sample_bytes = if mode.is_float() {
            4
        } else if mode.is_integer() {
            mode.sample_bytes()
        } else if depth > 8 {
            2
        } else {
            1
        };
        let image_row = if mode == PixelMode::One {
            width.div_ceil(8)
        } else {
            width.checked_mul(mode.channels() * sample_bytes).ok_or_else(|| PyValueError::new_err("raw row size overflow"))?
        };
        let raw_row = match bgr_channels {
            Some(channels) => width.checked_mul(channels).ok_or_else(|| PyValueError::new_err("raw row size overflow"))?,
            None => image_row,
        };
        let stride = if stride == 0 { raw_row } else { stride };
        if stride < raw_row {
            return Err(PyValueError::new_err("stride must be at least the packed row size"));
        }
        Ok(Self { image_row, raw_row, stride, bgr_channels })
    }

    fn is_identity(&self, orientation: i8) -> bool {
        self.bgr_channels.is_none() && self.stride == self.image_row && orientation == 1
    }
}

struct Rows {
    input_row: usize,
    input_stride: usize,
    output_row: usize,
    output_stride: usize,
    height: usize,
    length: usize,
    reverse: bool,
    swap_channels: Option<(usize, usize)>,
}

impl Rows {
    fn new(layout: Layout, mode: PixelMode, height: usize, orientation: i8, encode: bool, input_len: usize) -> PyResult<Self> {
        let (input_row, input_stride, output_row, output_stride) = if encode {
            (layout.image_row, layout.image_row, layout.raw_row, layout.stride)
        } else {
            (layout.raw_row, layout.stride, layout.image_row, layout.image_row)
        };
        let required = if height == 0 { Some(0) } else { (height - 1).checked_mul(input_stride).and_then(|v| v.checked_add(input_row)) }
            .ok_or_else(|| PyValueError::new_err("raw buffer size overflow"))?;
        if input_len < required {
            return Err(PyValueError::new_err("not enough image data"));
        }
        let length = output_stride.checked_mul(height).ok_or_else(|| PyValueError::new_err("raw buffer size overflow"))?;
        let swap_channels = layout.bgr_channels.map(|c| if encode { (mode.channels(), c) } else { (c, mode.channels()) });
        Ok(Self { input_row, input_stride, output_row, output_stride, height, length, reverse: orientation == -1, swap_channels })
    }

    fn source_row<'a>(&self, data: &'a [u8], row: usize) -> &'a [u8] {
        let row = if self.reverse { self.height - row - 1 } else { row };
        let offset = row * self.input_stride;
        &data[offset..offset + self.input_row]
    }

    fn write(&self, data: &[u8], output: &mut [u8]) {
        if output.is_empty() {
            return;
        }
        blanket_kernels::parallel::chunks_mut_above(output, self.output_stride.saturating_mul(32), 4 * 1024 * 1024, |band, rows| {
            for (i, target) in rows.chunks_exact_mut(self.output_stride).enumerate() {
                let source = self.source_row(data, band * 32 + i);
                let target = &mut target[..self.output_row];
                match self.swap_channels {
                    Some((3, 3)) => blanket_kernels::pixels::swap_red_blue::<3, 3>(source, target),
                    Some((3, 4)) => blanket_kernels::pixels::swap_red_blue::<3, 4>(source, target),
                    Some((4, 3)) => blanket_kernels::pixels::swap_red_blue::<4, 3>(source, target),
                    Some((4, 4)) => blanket_kernels::pixels::swap_red_blue::<4, 4>(source, target),
                    Some(_) => unreachable!(),
                    None => target.copy_from_slice(source),
                }
            }
        });
    }
}

pub(crate) fn transform<'a>(
    data: &'a [u8], mode: PixelMode, size: (u32, u32), depth: u8, rawmode: Option<&str>, stride: usize, orientation: i8, encode: bool,
) -> PyResult<Cow<'a, [u8]>> {
    let layout = match Layout::new(mode, size.0 as usize, depth, rawmode, stride, orientation) {
        Ok(layout) => layout,
        Err(error) => return extended_transform(data, mode, size, depth, rawmode, stride, orientation, encode).unwrap_or(Err(error)),
    };
    if layout.is_identity(orientation) {
        return Ok(Cow::Borrowed(data));
    }
    let rows = Rows::new(layout, mode, size.1 as usize, orientation, encode, data.len())?;
    let mut output = blanket_core::buffers::reserved(rows.length)?;
    if rows.swap_channels.is_none() && rows.output_stride == rows.output_row {
        blanket_core::buffers::copy_rows(&mut output, rows.output_stride, rows.height, |row| rows.source_row(data, row));
    } else {
        output.resize(rows.length, 0);
        rows.write(data, &mut output);
    }
    Ok(Cow::Owned(output))
}

pub(crate) fn encode<'py>(
    py: Python<'py>, data: &[u8], mode: PixelMode, size: (u32, u32), depth: u8, rawmode: Option<&str>, stride: usize, orientation: i8,
) -> PyResult<Bound<'py, PyBytes>> {
    let layout = match Layout::new(mode, size.0 as usize, depth, rawmode, stride, orientation) {
        Ok(layout) => layout,
        Err(error) => {
            let output = py.detach(|| extended_transform(data, mode, size, depth, rawmode, stride, orientation, true).unwrap_or(Err(error)))?;
            return Ok(PyBytes::new(py, &output));
        }
    };
    if layout.is_identity(orientation) {
        return Ok(PyBytes::new(py, data));
    }
    let rows = Rows::new(layout, mode, size.1 as usize, orientation, true, data.len())?;
    PyBytes::new_with(py, rows.length, |output| {
        py.detach(|| rows.write(data, output));
        Ok(())
    })
}

enum RawKind {
    Bits { bits: usize, invert: bool, reverse: bool, planes: bool, scale: bool },
    Numeric { bytes: usize, signed: bool, float: bool, big: bool, packed: bool },
    Channels { map: Vec<Option<usize>>, planar: bool, invert: bool, reverse: bool, bytes: usize, big: bool, premultiplied: bool },
    PackedColor { bits: usize, bgr: bool },
    LuminanceAlpha { bytes: usize, big: bool },
    PhotoYcc { alpha: bool },
    Cmyk,
}

fn bands(mode: PixelMode) -> &'static [&'static str] {
    match mode {
        PixelMode::One | PixelMode::L => &["L"],
        PixelMode::La => &["L", "A"],
        PixelMode::PremulLa => &["L", "a"],
        PixelMode::Pa => &["P", "A"],
        PixelMode::Rgb => &["R", "G", "B"],
        PixelMode::Rgba => &["R", "G", "B", "A"],
        PixelMode::PremulRgba => &["R", "G", "B", "a"],
        PixelMode::Rgbx => &["R", "G", "B", "X"],
        PixelMode::Cmyk => &["C", "M", "Y", "K"],
        PixelMode::YCbCr => &["Y", "Cb", "Cr"],
        PixelMode::Lab => &["L", "A", "B"],
        PixelMode::Hsv => &["H", "S", "V"],
        _ => &[],
    }
}

fn raw_kind(mode: PixelMode, rawmode: &str) -> Option<RawKind> {
    if !rawmode.is_ascii() {
        return None;
    }
    let (base, suffix) = rawmode.split_once(';').unwrap_or((rawmode, ""));
    if matches!(mode, PixelMode::Rgb | PixelMode::Rgbx | PixelMode::Rgba) && suffix == "P" && matches!(base, "YCC" | "YCCA") {
        return Some(RawKind::PhotoYcc { alpha: base == "YCCA" });
    }
    if mode == PixelMode::Rgb && rawmode == "CMYK" {
        return Some(RawKind::Cmyk);
    }
    if matches!(mode, PixelMode::Rgb | PixelMode::Rgba | PixelMode::Rgbx)
        && matches!(base, "RGB" | "RGBA" | "BGR" | "BGRA")
        && matches!(suffix, "15" | "16" | "4B")
    {
        return Some(RawKind::PackedColor { bits: if suffix == "4B" { 4 } else { suffix.parse().ok()? }, bgr: base.starts_with('B') });
    }
    if mode == PixelMode::Rgba && base == "LA" && matches!(suffix, "" | "16B" | "16L") {
        return Some(RawKind::LuminanceAlpha { bytes: if suffix.is_empty() { 1 } else { 2 }, big: suffix.ends_with('B') });
    }
    if mode.is_integer() || mode == PixelMode::F {
        if !matches!(base, "I" | "F") {
            return None;
        }
        let bits = suffix.chars().take_while(char::is_ascii_digit).collect::<String>().parse::<usize>().ok()?;
        if !matches!(bits, 8 | 12 | 16 | 32 | 64) {
            return None;
        }
        let qualifier = suffix.trim_start_matches(|value: char| value.is_ascii_digit());
        if !matches!(qualifier, "" | "B" | "L" | "N" | "S" | "BS" | "LS" | "NS" | "F" | "BF" | "LF" | "NF")
            || qualifier.contains('F') && !matches!(bits, 32 | 64)
        {
            return None;
        }
        return Some(RawKind::Numeric {
            bytes: bits.div_ceil(8),
            signed: suffix.contains('S'),
            float: suffix.contains('F'),
            big: suffix.contains('B') || suffix.contains('N') && cfg!(target_endian = "big"),
            packed: bits == 12,
        });
    }
    if matches!(mode, PixelMode::One | PixelMode::L) && matches!(base, "1" | "L" | "P") {
        let bits = if base == "1" && !suffix.contains('8') {
            1
        } else {
            suffix.chars().take_while(char::is_ascii_digit).collect::<String>().parse::<usize>().unwrap_or(8)
        };
        if matches!(bits, 1 | 2 | 4 | 8) {
            return Some(RawKind::Bits {
                bits,
                invert: suffix.contains('I'),
                reverse: suffix.contains('R'),
                planes: base == "P" && suffix.ends_with('L') && bits < 8,
                scale: base == "L",
            });
        }
    }
    let source_bands = bands(mode);
    if source_bands.is_empty() || mode == PixelMode::One {
        return None;
    }
    let mut rest = base;
    let mut map = Vec::new();
    while !rest.is_empty() {
        let token = if rest.starts_with("Cb") || rest.starts_with("Cr") { &rest[..2] } else { &rest[..1] };
        let mut channel = source_bands.iter().position(|&band| band == token || band == "A" && token == "a");
        if token == "X" && (map.contains(&channel) || matches!(rawmode, "BGRX" | "XRGB" | "XBGR")) {
            channel = None;
        }
        if channel.is_none() && !matches!(token, "X" | "K") {
            return None;
        }
        map.push(channel);
        rest = &rest[token.len()..];
    }
    let bytes = if suffix.starts_with("16") {
        2
    } else if matches!(suffix, "" | "L" | "I" | "R") {
        1
    } else {
        return None;
    };
    Some(RawKind::Channels {
        map,
        planar: suffix == "L",
        invert: suffix == "I",
        reverse: suffix == "R",
        bytes,
        big: suffix.contains('B') || suffix.contains('N') && cfg!(target_endian = "big"),
        premultiplied: base.contains('a') && mode == PixelMode::Rgba,
    })
}

fn extended_transform<'a>(
    data: &'a [u8], mode: PixelMode, size: (u32, u32), depth: u8, rawmode: Option<&str>, stride: usize, orientation: i8, encode: bool,
) -> Option<PyResult<Cow<'a, [u8]>>> {
    if !matches!(orientation, -1 | 1) || depth > 8 && !mode.is_wide_scalar() {
        return None;
    }
    let rawmode = rawmode?;
    let kind = raw_kind(mode, rawmode)?;
    Some((|| {
        if encode && matches!(kind, RawKind::Numeric { packed: true, .. } | RawKind::Bits { planes: true, .. }) {
            return Err(PyValueError::new_err("unsupported packed raw encoder"));
        }
        let (width, height) = (size.0 as usize, size.1 as usize);
        let channels = mode.channels();
        let sample_bytes = if mode.is_wide_scalar() { mode.sample_bytes() } else { 1 };
        let image_row = if mode == PixelMode::One {
            width.div_ceil(8)
        } else {
            width.checked_mul(channels * sample_bytes).ok_or_else(|| PyValueError::new_err("raw row size overflow"))?
        };
        let raw_row = match &kind {
            RawKind::Bits { bits, planes, .. } => {
                if *planes {
                    width.div_ceil(8) * bits
                } else {
                    width.checked_mul(*bits).ok_or_else(|| PyValueError::new_err("raw row size overflow"))?.div_ceil(8)
                }
            }
            RawKind::Numeric { bytes, packed, .. } => {
                if *packed {
                    (width * 12).div_ceil(8)
                } else {
                    width * bytes
                }
            }
            RawKind::Channels { map, bytes, .. } => width * map.len() * bytes,
            RawKind::PackedColor { .. } => width * 2,
            RawKind::LuminanceAlpha { bytes, .. } => width * bytes * 2,
            RawKind::PhotoYcc { alpha } => width * if *alpha { 4 } else { 3 },
            RawKind::Cmyk => width * 4,
        };
        let stride = if stride == 0 { raw_row } else { stride };
        if stride < raw_row {
            return Err(PyValueError::new_err("stride must be at least the packed row size"));
        }
        let (input_row, input_stride, output_stride) = if encode { (image_row, image_row, stride) } else { (raw_row, stride, image_row) };
        let required = if height == 0 { Some(0) } else { (height - 1).checked_mul(input_stride).and_then(|v| v.checked_add(input_row)) }
            .ok_or_else(|| PyValueError::new_err("raw buffer size overflow"))?;
        if data.len() < required {
            return Err(PyValueError::new_err("not enough image data"));
        }
        let length = output_stride.checked_mul(height).ok_or_else(|| PyValueError::new_err("raw buffer size overflow"))?;
        let mut output = blanket_core::raster::alloc_pixels(length)?;
        for y in 0..height {
            let source_y = if orientation == -1 { height - y - 1 } else { y };
            let source = &data[source_y * input_stride..source_y * input_stride + input_row];
            let target = &mut output[y * output_stride..(y + 1) * output_stride];
            match &kind {
                RawKind::PhotoYcc { alpha } => {
                    if encode {
                        return Err(PyValueError::new_err("unsupported PhotoYCC encoder"));
                    }
                    for (input, pixel) in source.chunks_exact(if *alpha { 4 } else { 3 }).zip(target[..image_row].chunks_exact_mut(channels)) {
                        let mut ycc = [input[0], input[1], input[2]];
                        if *alpha {
                            for v in &mut ycc {
                                *v = (u16::from(*v) * 255).checked_div(u16::from(input[3])).unwrap_or(0) as u8;
                            }
                        }
                        let l = (f64::from(ycc[0]) * 1.3584 + 0.5) as i32;
                        let cb = f64::from(ycc[1]) - 156.0;
                        let cr = f64::from(ycc[2]) - 137.0;
                        pixel[0] = (l + (cr * 1.8215 + 0.5) as i32).clamp(0, 255) as u8;
                        pixel[1] = (l + (cr * -0.9271 + 0.5) as i32 + (cb * -0.4303 + 0.5) as i32).clamp(0, 255) as u8;
                        pixel[2] = (l + (cb * 2.2179 + 0.5) as i32).clamp(0, 255) as u8;
                        if channels == 4 {
                            pixel[3] = if *alpha { input[3] } else { 255 };
                        }
                    }
                }
                RawKind::Cmyk => {
                    if encode {
                        return Err(PyValueError::new_err("unsupported CMYK raw encoder"));
                    }
                    for (input, pixel) in source.as_chunks::<4>().0.iter().zip(target[..image_row].as_chunks_mut::<3>().0) {
                        for c in 0..3 {
                            let product = u32::from(input[c]) * u32::from(255 - input[3]) + 128;
                            pixel[c] = (255 - u32::from(input[3]) - ((product + (product >> 8)) >> 8)) as u8;
                        }
                    }
                }
                RawKind::PackedColor { bits, bgr } => {
                    if encode {
                        return Err(PyValueError::new_err("unsupported packed color encoder"));
                    }
                    for (input, pixel) in source.as_chunks::<2>().0.iter().zip(target[..image_row].chunks_exact_mut(channels)) {
                        let value = u16::from_le_bytes(*input);
                        let (r, g, b, a) = if *bits == 4 {
                            ((value & 15) * 17, ((value >> 4) & 15) * 17, ((value >> 8) & 15) * 17, (value >> 12) * 17)
                        } else if *bits == 15 {
                            ((value & 31) * 255 / 31, ((value >> 5) & 31) * 255 / 31, ((value >> 10) & 31) * 255 / 31, (value >> 15) * 255)
                        } else {
                            ((value & 31) * 255 / 31, ((value >> 5) & 63) * 255 / 63, ((value >> 11) & 31) * 255 / 31, 255)
                        };
                        let rgb = if *bgr { [b as u8, g as u8, r as u8] } else { [r as u8, g as u8, b as u8] };
                        pixel[..3].copy_from_slice(&rgb);
                        if channels == 4 {
                            pixel[3] = if mode == PixelMode::Rgbx { 255 } else { a as u8 };
                        }
                    }
                }
                RawKind::LuminanceAlpha { bytes, big } => {
                    if encode {
                        return Err(PyValueError::new_err("unsupported luminance alpha encoder"));
                    }
                    for (input, pixel) in source.chunks_exact(bytes * 2).zip(target[..image_row].as_chunks_mut::<4>().0) {
                        let offset = if *bytes == 2 && !big { 1 } else { 0 };
                        pixel[..3].fill(input[offset]);
                        pixel[3] = input[bytes + offset];
                    }
                }
                RawKind::Bits { bits, invert, reverse, planes, scale } => {
                    let mask = (1u16 << bits) - 1;
                    for x in 0..width {
                        if encode {
                            let mut value = if mode == PixelMode::One {
                                ((source[x / 8] >> (7 - x % 8)) & 1) * if *bits == 8 { 255 } else { 1 }
                            } else {
                                source[x]
                            };
                            if *bits == 1 {
                                value = u8::from(value != 0);
                            }
                            if *invert {
                                value = !value;
                            }
                            let offset = x * bits;
                            target[offset / 8] |= ((u16::from(value) & mask) as u8) << (8 - bits - offset % 8);
                        } else {
                            let mut value = 0;
                            if *planes {
                                for plane in 0..*bits {
                                    value |= ((source[plane * width.div_ceil(8) + x / 8] >> (7 - x % 8)) & 1) << plane;
                                }
                            } else {
                                let offset = x * bits;
                                let byte = if *reverse { source[offset / 8].reverse_bits() } else { source[offset / 8] };
                                value = (byte >> (8 - bits - offset % 8)) & mask as u8;
                            }
                            if *invert {
                                value ^= mask as u8;
                            }
                            if mode == PixelMode::One {
                                if value != 0 {
                                    target[x / 8] |= 1 << (7 - x % 8);
                                }
                            } else {
                                target[x] = if *scale { (u16::from(value) * 255 / mask) as u8 } else { value };
                            }
                        }
                    }
                    if encode && *reverse {
                        for byte in &mut target[..raw_row] {
                            *byte = byte.reverse_bits();
                        }
                    }
                }
                RawKind::Numeric { bytes, signed, float, big, packed } => {
                    for x in 0..width {
                        if encode {
                            let pixel = &source[x * sample_bytes..(x + 1) * sample_bytes];
                            let value = if mode == PixelMode::F {
                                f32::from_le_bytes(pixel.try_into().unwrap()) as f64
                            } else {
                                blanket_core::raster::integer_value(mode, pixel) as f64
                            };
                            let value = if *float {
                                if *bytes == 4 { (value as f32).to_bits() as u64 } else { value.to_bits() }
                            } else {
                                value as i64 as u64
                            };
                            let packed = if *big { value.to_be_bytes() } else { value.to_le_bytes() };
                            target[x * bytes..(x + 1) * bytes].copy_from_slice(if *big { &packed[8 - bytes..] } else { &packed[..*bytes] });
                        } else {
                            let value = if *packed {
                                let offset = x * 12;
                                (0..12).fold(0u64, |v, bit| (v << 1) | u64::from((source[(offset + bit) / 8] >> (7 - (offset + bit) % 8)) & 1))
                            } else {
                                let pixel = &source[x * bytes..(x + 1) * bytes];
                                if *big {
                                    pixel.iter().fold(0u64, |v, &b| (v << 8) | u64::from(b))
                                } else {
                                    pixel.iter().rev().fold(0u64, |v, &b| (v << 8) | u64::from(b))
                                }
                            };
                            let value = if *float {
                                if *bytes == 4 { f32::from_bits(value as u32) as f64 } else { f64::from_bits(value) }
                            } else if *signed {
                                ((value << (64 - bytes * 8)) as i64 >> (64 - bytes * 8)) as f64
                            } else {
                                value as f64
                            };
                            let pixel = &mut target[x * sample_bytes..(x + 1) * sample_bytes];
                            if mode == PixelMode::F {
                                pixel.copy_from_slice(&(value as f32).to_le_bytes());
                            } else {
                                pixel.copy_from_slice(&blanket_core::raster::integer_bytes(mode, value as i64)[..sample_bytes]);
                            }
                        }
                    }
                }
                RawKind::Channels { map, planar, invert, reverse, bytes, big, premultiplied } => {
                    if !encode && mode == PixelMode::Lab && rawmode != "LAB" {
                        for pixel in target[..image_row].as_chunks_mut::<3>().0 {
                            pixel[1] = 128;
                            pixel[2] = 128;
                        }
                    }
                    if !encode && map.len() >= 3 && matches!(mode, PixelMode::Rgba | PixelMode::Rgbx) {
                        for pixel in target[..image_row].chunks_exact_mut(channels) {
                            pixel[3] = 255;
                        }
                    }
                    for x in 0..width {
                        for (slot, &channel) in map.iter().enumerate() {
                            let offset = if *planar { (slot * width + x) * bytes } else { (x * map.len() + slot) * bytes };
                            let Some(channel) = channel else {
                                if encode && slot >= 3 && (rawmode.starts_with("RGBX") || rawmode.starts_with("YCbCr")) {
                                    target[offset] = 255;
                                }
                                continue;
                            };
                            if encode {
                                let mut value = source[x * channels + channel];
                                if mode == PixelMode::Rgbx && rawmode != "RGBX;L" && rawmode != "X" && channel == 3 {
                                    value = 0;
                                }
                                if mode == PixelMode::Lab && rawmode != "LAB" && channel > 0 {
                                    value ^= 128;
                                }
                                if *invert {
                                    value = !value;
                                }
                                if *reverse {
                                    value = value.reverse_bits();
                                }
                                if *premultiplied && channel < 3 {
                                    let product = u16::from(value) * u16::from(source[x * channels + 3]) + 128;
                                    value = ((product + (product >> 8)) >> 8) as u8;
                                }
                                target[offset + if *bytes == 2 && !big { 1 } else { 0 }] = value;
                            } else {
                                let mut value = source[offset + if *bytes == 2 && !big { 1 } else { 0 }];
                                if *invert && !(mode == PixelMode::Rgba && channel == 3) {
                                    value = !value;
                                }
                                if *reverse {
                                    value = value.reverse_bits();
                                }
                                if mode == PixelMode::Lab && rawmode != "LAB" && channel > 0 {
                                    value ^= 128;
                                }
                                target[x * channels + channel] = value;
                            }
                        }
                        if !encode && *premultiplied {
                            let pixel = &mut target[x * channels..(x + 1) * channels];
                            let alpha = u16::from(pixel[3]);
                            for channel in &mut pixel[..3] {
                                *channel = (u16::from(*channel) * 255).checked_div(alpha).unwrap_or(0).min(255) as u8;
                            }
                        }
                    }
                }
            }
        }
        Ok(Cow::Owned(output))
    })())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn padded_unaligned_bgr_rows_ignore_guards_and_final_padding() {
        let data = [99, 1, 2, 3, 4, 5, 6, 88, 88, 7, 8, 9, 10, 11, 12, 99];
        let pixels = transform(&data[1..15], PixelMode::Rgb, (2, 2), 8, Some("BGR"), 8, -1, false).unwrap();
        assert_eq!(pixels.as_ref(), &[9, 8, 7, 12, 11, 10, 3, 2, 1, 6, 5, 4]);
        let encoded = transform(&pixels, PixelMode::Rgb, (2, 2), 8, Some("BGR"), 8, -1, true).unwrap();
        assert_eq!(encoded.as_ref(), &[1, 2, 3, 4, 5, 6, 0, 0, 7, 8, 9, 10, 11, 12, 0, 0]);
        assert_eq!(data[0], 99);
        assert_eq!(data[15], 99);
    }

    #[test]
    fn raw_layout_rejects_truncation_and_overflow_before_allocation() {
        assert!(transform(&[0; 13], PixelMode::Rgb, (2, 2), 8, Some("BGR"), 8, -1, false).is_err());
        assert!(transform(&[0; 12], PixelMode::Rgb, (2, 2), 8, None, usize::MAX, 1, true).is_err());
        assert!(transform(&[0; 12], PixelMode::Rgb, (2, 2), 8, None, 5, 1, false).is_err());
        assert!(transform(&[0; 12], PixelMode::Rgb, (2, 2), 8, None, 6, 0, false).is_err());
    }
}
