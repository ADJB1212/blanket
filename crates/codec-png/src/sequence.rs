use std::io::Cursor;

use blanket_codec_common::{Frame, SaveOptions, Sequence, codec_error, validate_frame_budget};
use blanket_core::{Image, PixelMode};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub fn decode_sequence(data: &[u8]) -> Result<Sequence, String> {
    let mut decoder = png::Decoder::new(Cursor::new(data));
    decoder.set_transformations(png::Transformations::EXPAND);
    let mut reader = decoder.read_info().map_err(|e| e.to_string())?;
    let Some(animation) = reader.info().animation_control else {
        return super::decode_png(data).map(Sequence::single);
    };
    let (width, height) = (reader.info().width, reader.info().height);
    let default_image = reader.info().frame_control.is_none();
    let count = animation
        .num_frames
        .checked_add(u32::from(default_image))
        .ok_or("APNG frame count overflow")?;
    if count == 0 {
        return Err("APNG contains no frames".into());
    }
    let mut total = (width as usize)
        .checked_mul(height as usize)
        .and_then(|v| v.checked_mul(count as usize - 1))
        .ok_or("APNG pixel count overflow")?;
    validate_frame_budget(&mut total, width, height)?;
    let mut sequence = Sequence {
        loop_count: Some(animation.num_plays),
        default_image,
        ..Sequence::default()
    };
    let mut buffer = blanket_core::raster::alloc_pixels(reader.output_buffer_size().ok_or("PNG buffer overflow")?).map_err(|e| e.to_string())?;
    let mut canvas = Vec::new();
    let length = width as usize * height as usize;
    canvas.try_reserve_exact(length).map_err(|_| "cannot allocate APNG canvas")?;
    canvas.resize(length, [0_u16; 4]);
    for frame_index in 0..count {
        let output = reader.next_frame(&mut buffer).map_err(|e| e.to_string())?;
        let control = reader.info().frame_control;
        let (x, y) = control.map_or((0, 0), |c| (c.x_offset, c.y_offset));
        if x.checked_add(output.width).is_none_or(|v| v > width) || y.checked_add(output.height).is_none_or(|v| v > height) {
            return Err("APNG frame outside canvas".into());
        }
        let previous = control.filter(|c| c.dispose_op == png::DisposeOp::Previous).map(|_| canvas.clone());
        let channels = output.color_type.samples();
        let wide = output.bit_depth == png::BitDepth::Sixteen;
        let max = if wide { 65535_u64 } else { 255 };
        let bytes = if wide { 2 } else { 1 };
        for row in 0..output.height as usize {
            for col in 0..output.width as usize {
                let start = (row * output.width as usize + col) * channels * bytes;
                let sample = |c: usize| -> u16 {
                    if wide {
                        u16::from_be_bytes([buffer[start + c * 2], buffer[start + c * 2 + 1]])
                    } else {
                        u16::from(buffer[start + c])
                    }
                };
                let src = match output.color_type {
                    png::ColorType::Grayscale => [sample(0), sample(0), sample(0), max as u16],
                    png::ColorType::GrayscaleAlpha => [sample(0), sample(0), sample(0), sample(1)],
                    png::ColorType::Rgb => [sample(0), sample(1), sample(2), max as u16],
                    png::ColorType::Rgba => [sample(0), sample(1), sample(2), sample(3)],
                    png::ColorType::Indexed => return Err("unexpanded APNG palette".into()),
                };
                let dst = &mut canvas[(y as usize + row) * width as usize + x as usize + col];
                if control.is_some_and(|c| c.blend_op == png::BlendOp::Over) {
                    let sa = u64::from(src[3]);
                    let da = u64::from(dst[3]);
                    let alpha = sa * max + da * (max - sa);
                    for c in 0..3 {
                        dst[c] = (u64::from(src[c]) * sa * max + u64::from(dst[c]) * da * (max - sa) + alpha / 2)
                            .checked_div(alpha)
                            .unwrap_or(0) as u16;
                    }
                    dst[3] = ((alpha + max / 2) / max) as u16;
                } else {
                    *dst = src;
                }
            }
        }
        let image = if wide {
            Image::from_samples(
                width,
                height,
                PixelMode::Rgba,
                canvas.iter().flatten().copied().collect(),
                16,
                Some("PNG".into()),
            )
        } else {
            Image::from_pixels(
                width,
                height,
                PixelMode::Rgba,
                canvas.iter().flatten().map(|&v| v as u8).collect(),
                Some("PNG".into()),
            )
        }
        .map_err(|e| e.to_string())?;
        sequence.frames.push(Frame {
            image,
            duration: control.map(|c| f64::from(c.delay_num) * 1000.0 / f64::from(if c.delay_den == 0 { 100 } else { c.delay_den })),
            disposal: control.map_or(0, |c| c.dispose_op as u8),
            blend: control.map_or(0, |c| c.blend_op as u8),
        });
        if default_image && frame_index == 0 {
            canvas.fill([0; 4]);
        } else if let Some(previous) = previous {
            canvas = previous;
        } else if control.is_some_and(|c| c.dispose_op == png::DisposeOp::Background) {
            for row in y..y + output.height {
                for col in x..x + output.width {
                    canvas[row as usize * width as usize + col as usize] = [0; 4];
                }
            }
        }
    }
    reader.finish().map_err(|e| e.to_string())?;
    Ok(sequence)
}

pub fn encode_sequence(sequence: &Sequence, options: SaveOptions) -> PyResult<Vec<u8>> {
    let first = &sequence.frames[0].image;
    let mut output = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut output, first.width, first.height);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(if first.bit_depth > 8 {
            png::BitDepth::Sixteen
        } else {
            png::BitDepth::Eight
        });
        encoder
            .set_animated(
                u32::try_from(sequence.frames.len() - usize::from(sequence.default_image)).map_err(codec_error)?,
                sequence.loop_count.unwrap_or(0),
            )
            .map_err(codec_error)?;
        encoder.set_sep_def_img(sequence.default_image).map_err(codec_error)?;
        encoder.set_deflate_compression(if options.compress_level == 0 {
            png::DeflateCompression::NoCompression
        } else {
            png::DeflateCompression::Level(options.compress_level)
        });
        let mut writer = encoder.write_header().map_err(codec_error)?;
        for (i, frame) in sequence.frames.iter().enumerate() {
            if !(sequence.default_image && i == 0) {
                let duration = frame.duration.unwrap_or(0.0);
                let (num, den) = if duration <= 65535.0 {
                    (duration.round() as u16, 1000)
                } else {
                    ((duration / 1000.0).round() as u16, 1)
                };
                writer.set_frame_delay(num, den).map_err(codec_error)?;
                writer
                    .set_dispose_op(png::DisposeOp::from_u8(frame.disposal).ok_or_else(|| PyValueError::new_err("invalid APNG disposal"))?)
                    .map_err(codec_error)?;
                writer
                    .set_blend_op(png::BlendOp::from_u8(frame.blend).ok_or_else(|| PyValueError::new_err("invalid APNG blend"))?)
                    .map_err(codec_error)?;
            }
            if first.bit_depth > 8 {
                let max = (1_u32 << frame.image.bit_depth) - 1;
                let pixels: Vec<u8> = frame
                    .image
                    .raw_data()?
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .flat_map(|v| (((u32::from(u16::from_le_bytes(*v)) * 65535 + max / 2) / max) as u16).to_be_bytes())
                    .collect();
                writer.write_image_data(&pixels).map_err(codec_error)?;
            } else {
                writer.write_image_data(frame.image.pixel_data()?).map_err(codec_error)?;
            }
        }
        writer.finish().map_err(codec_error)?;
    }
    Ok(output)
}
