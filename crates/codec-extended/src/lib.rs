#![feature(portable_simd)]

pub mod legacy;
mod pixels;

use std::io::Write;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use blanket_codec_common::{codec_error, validate_dimensions};
use blanket_codec_image::{color_type, decode_rust_image, decode_with_metadata};
use blanket_core::{Image, PixelMode};
use image::{ColorType, ImageDecoder, ImageEncoder, ImageFormat};
use pyo3::exceptions::PyOSError;
use pyo3::prelude::*;

pub fn decode_pnm(data: &[u8]) -> Result<Image, String> {
    let decoder = image::codecs::pnm::PnmDecoder::new(data).map_err(|e| e.to_string())?;
    let (width, height) = decoder.dimensions();
    validate_dimensions(width, height)?;
    let color = decoder.color_type();
    if data.starts_with(b"P4") {
        let (source, _) = decoder.into_inner();
        let length = (width as usize).div_ceil(8) * height as usize;
        let source = source.get(..length).ok_or("truncated PBM pixels")?;
        let pixels = blanket_core::raster::unpack_bilevel_inverted(source, width as usize, height as usize);
        return Image::from_pixels(width, height, PixelMode::One, pixels, Some("PPM".into())).map_err(|e| e.to_string());
    }
    if matches!(data.get(1), Some(b'5'..=b'7')) && decoder.header().maximal_sample() == 65535 {
        let mode = match color {
            ColorType::L16 => Some(PixelMode::L),
            ColorType::Rgb16 => Some(PixelMode::Rgb),
            ColorType::Rgba16 => Some(PixelMode::Rgba),
            _ => None,
        };
        if let Some(mode) = mode {
            let (source, _) = decoder.into_inner();
            let length = width as usize * height as usize * mode.channels() * 2;
            let source = source.get(..length).ok_or("truncated Netpbm pixels")?;
            let mut pixels = vec![0; length];
            blanket_core::simd::samples_to_be16(source, &mut pixels, 16);
            return Image::from_sample_bytes(width, height, mode, pixels, 16, Some("PPM".into())).map_err(|e| e.to_string());
        }
    }
    decode_with_metadata(data, ImageFormat::Pnm, "PPM", None, matches!(data.get(1), Some(b'1' | b'4')))
}

pub fn decode_tga(data: &[u8]) -> Result<Image, String> {
    if data.len() >= 18 && data[1] == 0 && data[2] == 2 && matches!(data[16], 24 | 32) && data[17] & 0xd0 == 0 {
        let width = u32::from(u16::from_le_bytes([data[12], data[13]]));
        let height = u32::from(u16::from_le_bytes([data[14], data[15]]));
        validate_dimensions(width, height)?;
        if width > 0 && height > 0 {
            let channels = usize::from(data[16] / 8);
            let start = 18 + usize::from(data[0]);
            let length = width as usize * height as usize * channels;
            let source = data.get(start..start + length).ok_or("truncated TGA pixels")?;
            let mut pixels = vec![0; length];
            let row_bytes = width as usize * channels;
            for (y, row) in pixels.chunks_exact_mut(row_bytes).enumerate() {
                let src_y = if data[17] & 0x20 == 0 { height as usize - 1 - y } else { y };
                let src = &source[src_y * row_bytes..(src_y + 1) * row_bytes];
                if channels == 3 {
                    pixels::swap_red_blue::<3>(src, row);
                } else {
                    pixels::swap_red_blue::<4>(src, row);
                }
            }
            return Image::from_pixels(
                width,
                height,
                if channels == 3 { PixelMode::Rgb } else { PixelMode::Rgba },
                pixels,
                Some("TGA".into()),
            )
            .map_err(|e| e.to_string());
        }
    }
    decode_rust_image(data, ImageFormat::Tga, "TGA")
}

fn encode_tga(image: &Image, pixels: &[u8]) -> PyResult<Vec<u8>> {
    let width = u16::try_from(image.width).map_err(|_| PyOSError::new_err("TGA width exceeds 65535"))?;
    let height = u16::try_from(image.height).map_err(|_| PyOSError::new_err("TGA height exceeds 65535"))?;
    let channels = image.mode.channels();
    let mut output = vec![0; 18 + pixels.len()];
    output[2] = if channels <= 2 { 3 } else { 2 };
    output[12..14].copy_from_slice(&width.to_le_bytes());
    output[14..16].copy_from_slice(&height.to_le_bytes());
    output[16] = (channels * 8) as u8;
    output[17] = if matches!(image.mode, PixelMode::La | PixelMode::Rgba) {
        0x28
    } else {
        0x20
    };
    match channels {
        3 => pixels::swap_red_blue::<3>(pixels, &mut output[18..]),
        4 => pixels::swap_red_blue::<4>(pixels, &mut output[18..]),
        _ => output[18..].copy_from_slice(pixels),
    }
    Ok(output)
}

pub fn decode_dds(data: &[u8]) -> Result<Image, String> {
    decode_rust_image(data, ImageFormat::Dds, "DDS")
}

pub fn decode_qoi(data: &[u8]) -> Result<Image, String> {
    decode_rust_image(data, ImageFormat::Qoi, "QOI")
}

pub fn encode_wide_pnm(image: &Image, pam: bool) -> PyResult<Vec<u8>> {
    if image.width == 0 || image.height == 0 {
        return Err(PyOSError::new_err("cannot encode an empty image"));
    }
    let (magic, tuple) = match image.mode {
        PixelMode::L => ("P5", "GRAYSCALE"),
        PixelMode::Rgb => ("P6", "RGB"),
        PixelMode::Rgba if pam => ("P7", "RGB_ALPHA"),
        _ => return Err(PyOSError::new_err("PPM supports modes 1, L, and RGB; use PAM for alpha")),
    };
    let samples = image.raw_data()?;
    let header = if pam {
        format!(
            "P7\nWIDTH {}\nHEIGHT {}\nDEPTH {}\nMAXVAL 65535\nTUPLTYPE {tuple}\nENDHDR\n",
            image.width,
            image.height,
            image.mode.channels()
        )
    } else {
        format!("{magic}\n{} {}\n65535\n", image.width, image.height)
    };
    let mut output = header.into_bytes();
    let start = output.len();
    output.resize(start + samples.len(), 0);
    blanket_core::simd::samples_to_be16(samples, &mut output[start..], image.bit_depth);
    Ok(output)
}

pub fn is_tga(data: &[u8]) -> bool {
    data.len() >= 18
        && data[1] <= 1
        && matches!(data[2], 1..=3 | 9..=11)
        && u16::from_le_bytes([data[12], data[13]]) != 0
        && u16::from_le_bytes([data[14], data[15]]) != 0
        && match data[2] {
            1 | 9 => data[1] == 1 && matches!(data[16], 8 | 16) && matches!(data[7], 15 | 16 | 24 | 32),
            2 | 10 => data[1] == 0 && matches!(data[16], 16 | 24 | 32),
            _ => data[1] == 0 && matches!(data[16], 8 | 16),
        }
}

pub fn encode_extended(image: &Image, pixels: &[u8], format: &str) -> PyResult<Vec<u8>> {
    if image.width == 0 || image.height == 0 {
        return Err(PyOSError::new_err("cannot encode an empty image"));
    }
    let mut output = Vec::new();
    let color = color_type(image.mode);
    match format {
        "TGA" => return encode_tga(image, pixels),
        "QOI" => image::codecs::qoi::QoiEncoder::new(&mut output)
            .write_image(pixels, image.width, image.height, color)
            .map_err(codec_error)?,
        "PPM" | "PAM" => {
            if format == "PPM" && image.mode == PixelMode::One {
                output.extend_from_slice(format!("P4\n{} {}\n", image.width, image.height).as_bytes());
                let row_bytes = (image.width as usize).div_ceil(8);
                let start = output.len();
                output.resize(start + row_bytes * image.height as usize, 0);
                for (row, dst) in pixels.chunks_exact(image.width as usize).zip(output[start..].chunks_exact_mut(row_bytes)) {
                    blanket_core::pixels::pack_bilevel::<true>(row, dst);
                }
            } else {
                use image::codecs::pnm::{PnmEncoder, PnmSubtype, SampleEncoding};
                let subtype = match (format, image.mode) {
                    ("PAM", _) => PnmSubtype::ArbitraryMap,
                    (_, PixelMode::L) => PnmSubtype::Graymap(SampleEncoding::Binary),
                    (_, PixelMode::Rgb) => PnmSubtype::Pixmap(SampleEncoding::Binary),
                    _ => return Err(PyOSError::new_err("PPM supports modes 1, L, and RGB; use PAM for alpha")),
                };
                PnmEncoder::new(&mut output)
                    .with_subtype(subtype)
                    .write_image(pixels, image.width, image.height, color)
                    .map_err(codec_error)?;
            }
        }
        _ => return Err(PyOSError::new_err(format!("{format} is a read-only format"))),
    }
    Ok(output)
}

pub fn ghostscript_available() -> bool {
    Command::new("gs")
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

pub fn decode_eps(data: &[u8]) -> Result<Image, String> {
    let data = if data.starts_with(b"\xc5\xd0\xd3\xc6") {
        let header = data.get(..12).ok_or("truncated EPS preview header")?;
        let offset = u32::from_le_bytes(header[4..8].try_into().unwrap()) as usize;
        let length = u32::from_le_bytes(header[8..12].try_into().unwrap()) as usize;
        data.get(offset..offset.checked_add(length).ok_or("EPS offset overflow")?)
            .ok_or("truncated EPS payload")?
    } else {
        data
    };
    let bbox = data
        .split(|&byte| byte == b'\n')
        .find_map(|line| {
            let line = line.strip_prefix(b"%%BoundingBox:")?;
            let values = std::str::from_utf8(line)
                .ok()?
                .split_whitespace()
                .map(str::parse::<f64>)
                .collect::<Result<Vec<_>, _>>()
                .ok()?;
            (values.len() == 4 && values.iter().all(|v| v.is_finite())).then_some(values)
        })
        .ok_or("EPS requires a numeric BoundingBox")?;
    let width = (bbox[2] - bbox[0]).ceil();
    let height = (bbox[3] - bbox[1]).ceil();
    if width < 1.0 || height < 1.0 || width > u32::MAX as f64 || height > u32::MAX as f64 {
        return Err("invalid EPS BoundingBox".into());
    }
    validate_dimensions(width as u32, height as u32)?;
    let directory = tempfile::tempdir().map_err(|e| e.to_string())?;
    let input_path = directory.path().join("input.eps");
    let output_path = directory.path().join("output.ppm");
    std::fs::File::create(&input_path)
        .and_then(|mut file| file.write_all(data))
        .map_err(|e| e.to_string())?;
    let mut child = Command::new("gs")
        .args([
            "-q",
            "-dSAFER",
            "-dBATCH",
            "-dNOPAUSE",
            "-dFIXEDMEDIA",
            "-dEPSFitPage",
            "-sDEVICE=ppmraw",
            "-r72",
        ])
        .arg(format!("-g{}x{}", width as u32, height as u32))
        .arg(format!("-sOutputFile={}", output_path.display()))
        .arg("-f")
        .arg(input_path)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| format!("EPS reading requires Ghostscript (gs): {e}"))?;
    let started = Instant::now();
    loop {
        match child.try_wait().map_err(|e| e.to_string())? {
            Some(status) if status.success() => break,
            Some(_) => return Err("Ghostscript could not rasterize EPS".into()),
            None if started.elapsed() > Duration::from_secs(30) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err("Ghostscript EPS rasterization timed out".into());
            }
            None => std::thread::sleep(Duration::from_millis(10)),
        }
    }
    let mut image = decode_pnm(&std::fs::read(output_path).map_err(|e| e.to_string())?)?;
    image.format = Some("EPS".into());
    Ok(image)
}
