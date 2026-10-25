use std::borrow::Borrow;

use blanket_codec_common::codec_error;
use blanket_codec_image::{color_type, decode_rust_image};
use blanket_core::{Image, PixelMode};
use image::{ExtendedColorType, ImageFormat as RustFormat};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub fn decode_sequence(data: &[u8]) -> Result<blanket_codec_common::Sequence, String> {
    use blanket_codec_common::{Frame, Sequence, validate_frame_budget};
    let mut options = gif::DecodeOptions::new();
    options.set_color_output(gif::ColorOutput::RGBA);
    options.check_frame_consistency(true);
    options.set_memory_limit(gif::MemoryLimit::Bytes(std::num::NonZeroU64::new(blanket_codec_common::MAX_IMAGE_PIXELS as u64 * 4).unwrap()));
    let mut decoder = options.read_info(std::io::Cursor::new(data)).map_err(|e| e.to_string())?;
    let (width, height) = (u32::from(decoder.width()), u32::from(decoder.height()));
    blanket_codec_common::validate_dimensions(width, height)?;
    let background =
        decoder.bg_color().and_then(|i| decoder.global_palette()?.get(i * 3..i * 3 + 3)).map(|p| [p[0], p[1], p[2], 255]).unwrap_or([0; 4]);
    let palette = decoder.global_palette().map(<[u8]>::to_vec);
    let mut canvas = blanket_core::raster::alloc_pixels(width as usize * height as usize * 4).map_err(|e| e.to_string())?;
    let mut sequence = Sequence::default();
    let mut total = 0;
    let mut previous = Vec::new();
    while let Some(frame) = decoder.read_next_frame().map_err(|e| e.to_string())? {
        validate_frame_budget(&mut total, width, height)?;
        let (x, y, w, h) = (u32::from(frame.left), u32::from(frame.top), u32::from(frame.width), u32::from(frame.height));
        if x + w > width || y + h > height {
            return Err("GIF frame outside canvas".into());
        }
        let clear = if let Some(index) = frame.transparent {
            frame
                .palette
                .as_deref()
                .or(palette.as_deref())
                .and_then(|p| p.get(usize::from(index) * 3..usize::from(index) * 3 + 3))
                .map_or([0; 4], |p| [p[0], p[1], p[2], 0])
        } else {
            background
        };
        if sequence.frames.is_empty() {
            for pixel in canvas.as_chunks_mut::<4>().0 {
                *pixel = clear;
            }
        }
        if frame.dispose == gif::DisposalMethod::Previous && w != 0 {
            previous.clear();
            for row in y..y + h {
                let start = (row as usize * width as usize + x as usize) * 4;
                previous.extend_from_slice(&canvas[start..start + w as usize * 4]);
            }
        }
        for row in 0..h as usize {
            let src = row * w as usize * 4;
            let dst = ((y as usize + row) * width as usize + x as usize) * 4;
            blanket_core::pixels::paste_nontransparent_rgba(&mut canvas[dst..dst + w as usize * 4], &frame.buffer[src..src + w as usize * 4]);
        }
        sequence.frames.push(Frame {
            image: Image::from_pixels(width, height, PixelMode::Rgba, canvas.clone(), Some("GIF".into())).map_err(|e| e.to_string())?,
            duration: Some(f64::from(frame.delay) * 10.0),
            disposal: frame.dispose as u8,
            blend: 0,
        });
        if frame.dispose == gif::DisposalMethod::Previous && w != 0 {
            if previous.len() == canvas.len() {
                std::mem::swap(&mut canvas, &mut previous);
            } else {
                for (row, saved) in (y..y + h).zip(previous.chunks_exact(w as usize * 4)) {
                    let start = (row as usize * width as usize + x as usize) * 4;
                    canvas[start..start + w as usize * 4].copy_from_slice(saved);
                }
            }
        } else if frame.dispose == gif::DisposalMethod::Background {
            for row in y..y + h {
                for col in x..x + w {
                    let dst = (row as usize * width as usize + col as usize) * 4;
                    canvas[dst..dst + 4].copy_from_slice(&clear);
                }
            }
        }
    }
    sequence.loop_count = match decoder.repeat() {
        gif::Repeat::Infinite => Some(0),
        gif::Repeat::Finite(0) => None,
        gif::Repeat::Finite(n) => Some(u32::from(n)),
    };
    if sequence.frames.is_empty() {
        return Err("GIF contains no frames".into());
    }
    if sequence.frames.len() == 1 {
        sequence.frames[0].image = decode_gif(data)?;
    }
    Ok(sequence)
}

pub fn encode_sequence(sequence: &blanket_codec_common::Sequence<impl Borrow<Image>>) -> PyResult<Vec<u8>> {
    let first = sequence.frames[0].image.borrow();
    if first.width == 0 || first.height == 0 || first.width > 65535 || first.height > 65535 {
        return Err(PyValueError::new_err("GIF dimensions must be between 1 and 65535"));
    }
    let mut output = Vec::new();
    {
        let mut encoder = gif::Encoder::new(&mut output, first.width as u16, first.height as u16, &[]).map_err(codec_error)?;
        if let Some(count) = sequence.loop_count {
            encoder.set_repeat(if count == 0 { gif::Repeat::Infinite } else { gif::Repeat::Finite(count as u16) }).map_err(codec_error)?;
        }
        let mut pixels = Vec::new();
        for frame in &sequence.frames {
            pixels.clear();
            pixels.extend_from_slice(frame.image.borrow().pixel_data()?);
            let mut encoded = gif::Frame::from_rgba_speed(first.width as u16, first.height as u16, &mut pixels, 10);
            encoded.delay = (frame.duration.unwrap_or(0.0) / 10.0) as u16;
            encoded.dispose = gif::DisposalMethod::from_u8(frame.disposal).ok_or_else(|| PyValueError::new_err("invalid GIF disposal"))?;
            encoder.write_frame(&encoded).map_err(codec_error)?;
        }
    }
    Ok(output)
}

pub fn decode_gif(data: &[u8]) -> Result<Image, String> {
    decode_rust_image(data, RustFormat::Gif, "GIF")
}

pub fn encode_gif(image: &Image, pixels: &[u8]) -> PyResult<Vec<u8>> {
    if image.width == 0 || image.height == 0 {
        return Err(PyValueError::new_err("cannot encode an empty image"));
    }
    if image.width > 65535 || image.height > 65535 {
        return Err(PyValueError::new_err("GIF dimensions must be between 1 and 65535"));
    }
    let mut output = Vec::new();
    let rgb;
    let (pixels, color) = if image.mode == PixelMode::L {
        rgb = pixels.iter().flat_map(|v| [*v; 3]).collect::<Vec<_>>();
        (rgb.as_slice(), ExtendedColorType::Rgb8)
    } else {
        (pixels, color_type(image.mode))
    };
    image::codecs::gif::GifEncoder::new(&mut output).encode(pixels, image.width, image.height, color).map_err(codec_error)?;

    Ok(output)
}
