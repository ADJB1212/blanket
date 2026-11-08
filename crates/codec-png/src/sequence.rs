use std::borrow::Borrow;
use std::io::Cursor;

use blanket_codec_common::{Frame, SaveOptions, Sequence, codec_error, validate_frame_budget};
use blanket_core::{Image, PixelMode};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

trait CanvasSample: Copy + Into<u64> {
    const MAX_SAMPLE: u64;
    fn read(bytes: &[u8]) -> Self;
    fn value(value: u64) -> Self;
    fn pack(canvas: &[[Self; 4]], mode: PixelMode) -> Vec<u8>;
    fn over(src: [Self; 4], dst: &mut [Self; 4]);
    fn copy_rgba(source: &[u8], canvas: &mut [[Self; 4]]);
}

impl CanvasSample for u8 {
    const MAX_SAMPLE: u64 = 255;
    fn read(bytes: &[u8]) -> Self {
        bytes[0]
    }
    fn value(value: u64) -> Self {
        value as u8
    }
    fn pack(canvas: &[[Self; 4]], mode: PixelMode) -> Vec<u8> {
        match mode {
            PixelMode::One | PixelMode::L => canvas.iter().map(|pixel| pixel[0]).collect(),
            PixelMode::La => canvas.iter().map(|pixel| [pixel[0], pixel[3]]).collect::<Vec<_>>().into_flattened(),
            PixelMode::Rgb => canvas.iter().map(|pixel| [pixel[0], pixel[1], pixel[2]]).collect::<Vec<_>>().into_flattened(),
            PixelMode::Rgba => canvas.as_flattened().to_vec(),
            _ => unreachable!("validated APNG output mode"),
        }
    }
    fn copy_rgba(source: &[u8], canvas: &mut [[Self; 4]]) {
        canvas.as_flattened_mut().copy_from_slice(source);
    }
    fn over(src: [Self; 4], dst: &mut [Self; 4]) {
        let sa = u32::from(src[3]);
        for c in 0..4 {
            dst[c] = ((u32::from(src[c]) * sa + u32::from(dst[c]) * (255 - sa) + 127) / 255) as u8;
        }
    }
}

impl CanvasSample for u16 {
    const MAX_SAMPLE: u64 = 65535;
    fn read(bytes: &[u8]) -> Self {
        u16::from_be_bytes([bytes[0], bytes[1]])
    }
    fn value(value: u64) -> Self {
        value as u16
    }
    fn pack(canvas: &[[Self; 4]], _mode: PixelMode) -> Vec<u8> {
        canvas.iter().map(|pixel| pixel.map(u16::to_le_bytes)).collect::<Vec<_>>().into_flattened().into_flattened()
    }
    fn copy_rgba(source: &[u8], canvas: &mut [[Self; 4]]) {
        for (src, dst) in source.as_chunks::<8>().0.iter().zip(canvas) {
            *dst = std::array::from_fn(|c| u16::from_be_bytes([src[c * 2], src[c * 2 + 1]]));
        }
    }
    fn over(src: [Self; 4], dst: &mut [Self; 4]) {
        let sa = u64::from(src[3]);
        let da = u64::from(dst[3]);
        let alpha = sa * Self::MAX_SAMPLE + da * (Self::MAX_SAMPLE - sa);
        for c in 0..3 {
            dst[c] = (u64::from(src[c]) * sa * Self::MAX_SAMPLE + u64::from(dst[c]) * da * (Self::MAX_SAMPLE - sa) + alpha / 2)
                .checked_div(alpha)
                .unwrap_or(0) as u16;
        }
        dst[3] = ((alpha + Self::MAX_SAMPLE / 2) / Self::MAX_SAMPLE) as u16;
    }
}

pub fn decode_sequence(data: &[u8]) -> Result<Sequence, String> {
    decode_frames(data, None)
}

pub fn decode_frame(data: &[u8], target: usize) -> Result<Sequence, String> {
    decode_frames(data, Some(target))
}

pub fn index_sequence(data: &[u8]) -> Result<blanket_codec_common::SequenceInfo, String> {
    use blanket_codec_common::SequenceInfo;
    let mut reader = super::decoder(data).read_info().map_err(|e| e.to_string())?;
    let Some(animation) = reader.info().animation_control else {
        validate_frame_budget(&mut 0, reader.info().width, reader.info().height)?;
        return Ok(SequenceInfo { count: 1, ..SequenceInfo::default() });
    };
    let default_image = reader.info().frame_control.is_none();
    let count = animation.num_frames.checked_add(u32::from(default_image)).ok_or("APNG frame count overflow")?;
    let mut pixels = (reader.info().width as usize)
        .checked_mul(reader.info().height as usize)
        .and_then(|pixels| pixels.checked_mul(count.saturating_sub(1) as usize))
        .ok_or("APNG pixel count overflow")?;
    validate_frame_budget(&mut pixels, reader.info().width, reader.info().height)?;
    for _ in 1..count {
        reader.next_frame_info().map_err(|e| e.to_string())?;
    }
    if count == 0 {
        return Err("APNG contains no frames".into());
    }
    Ok(SequenceInfo { count: count as usize, loop_count: Some(animation.num_plays), default_image })
}

fn decode_frames(data: &[u8], target: Option<usize>) -> Result<Sequence, String> {
    let mut decoder = super::decoder(data);
    decoder.set_transformations(png::Transformations::EXPAND);
    let reader = decoder.read_info().map_err(|e| e.to_string())?;
    let Some(animation) = reader.info().animation_control else {
        if reader.info().color_type == png::ColorType::Indexed
            || reader.info().color_type == png::ColorType::Grayscale && reader.info().bit_depth == png::BitDepth::One
        {
            return super::decode_png(data).map(Sequence::single).map_err(String::from);
        }
        return super::decode_png_reader(reader, false).map(Sequence::single).map_err(String::from);
    };
    if reader.info().color_type == png::ColorType::Indexed {
        let reader = super::decoder(data).read_info().map_err(|error| error.to_string())?;
        return decode_animation::<u8>(reader, animation, target);
    }
    if reader.info().bit_depth == png::BitDepth::Sixteen {
        decode_animation::<u16>(reader, animation, target)
    } else {
        decode_animation::<u8>(reader, animation, target)
    }
}

fn decode_animation<T: CanvasSample>(
    mut reader: png::Reader<Cursor<&[u8]>>, animation: png::AnimationControl, target: Option<usize>,
) -> Result<Sequence, String> {
    let (width, height) = (reader.info().width, reader.info().height);
    let color_type = reader.info().color_type;
    let bit_depth = reader.info().bit_depth;
    let palette = reader.info().palette.as_deref().map(<[u8]>::to_vec);
    let alpha = reader.info().trns.as_deref().map(<[u8]>::to_vec);
    let default_image = reader.info().frame_control.is_none();
    let count = animation.num_frames.checked_add(u32::from(default_image)).ok_or("APNG frame count overflow")?;
    if count == 0 {
        return Err("APNG contains no frames".into());
    }
    let mut total =
        (width as usize).checked_mul(height as usize).and_then(|v| v.checked_mul(count as usize - 1)).ok_or("APNG pixel count overflow")?;
    validate_frame_budget(&mut total, width, height)?;
    let mut sequence = Sequence { loop_count: Some(animation.num_plays), default_image, ..Sequence::default() };
    let mut buffer = blanket_core::raster::alloc_pixels(reader.output_buffer_size().ok_or("PNG buffer overflow")?).map_err(|e| e.to_string())?;
    let mut canvas = Vec::new();
    let length = width as usize * height as usize;
    canvas.try_reserve_exact(length).map_err(|_| "cannot allocate APNG canvas")?;
    let zero = T::value(0);
    canvas.resize(length, [zero; 4]);
    let mut previous = Vec::new();
    for frame_index in 0..count {
        let output = reader.next_frame(&mut buffer).map_err(|e| e.to_string())?;
        let control = reader.info().frame_control;
        let (x, y) = control.map_or((0, 0), |c| (c.x_offset, c.y_offset));
        if x.checked_add(output.width).is_none_or(|v| v > width) || y.checked_add(output.height).is_none_or(|v| v > height) {
            return Err("APNG frame outside canvas".into());
        }
        let terminal = frame_index + 1 == count || target == Some(frame_index as usize);
        let restore_previous = control.is_some_and(|c| c.dispose_op == png::DisposeOp::Previous) && !terminal;
        if terminal {
            drop(std::mem::take(&mut previous));
        }
        if restore_previous {
            previous.clear();
            for row in y..y + output.height {
                let start = row as usize * width as usize + x as usize;
                previous.extend_from_slice(&canvas[start..start + output.width as usize]);
            }
        }
        let channels = output.color_type.samples();
        let max = T::MAX_SAMPLE;
        let bytes = size_of::<T>();
        let over = control.is_some_and(|c| c.blend_op == png::BlendOp::Over) && frame_index != u32::from(default_image);
        for row in 0..output.height as usize {
            if !over && output.color_type == png::ColorType::Rgba {
                let start = row * output.width as usize * 4 * bytes;
                let dst = (y as usize + row) * width as usize + x as usize;
                T::copy_rgba(&buffer[start..start + output.width as usize * 4 * bytes], &mut canvas[dst..dst + output.width as usize]);
                continue;
            }
            for col in 0..output.width as usize {
                let start = (row * output.width as usize + col) * channels * bytes;
                let sample = |c: usize| T::read(&buffer[start + c * bytes..]);
                let src = match output.color_type {
                    png::ColorType::Grayscale => [sample(0), sample(0), sample(0), T::value(max)],
                    png::ColorType::GrayscaleAlpha => [sample(0), sample(0), sample(0), sample(1)],
                    png::ColorType::Rgb => [sample(0), sample(1), sample(2), T::value(max)],
                    png::ColorType::Rgba => [sample(0), sample(1), sample(2), sample(3)],
                    png::ColorType::Indexed => {
                        let bits = output.bit_depth as usize;
                        let index = (buffer[row * output.line_size + col * bits / 8] >> (8 - bits - col * bits % 8)) & ((1u16 << bits) - 1) as u8;
                        let value = T::value(u64::from(index));
                        let alpha = alpha.as_deref().and_then(|values| values.get(usize::from(index))).copied().unwrap_or(255);
                        [value, value, value, T::value(u64::from(alpha))]
                    }
                };
                let dst = &mut canvas[(y as usize + row) * width as usize + x as usize + col];
                if !over || src[3].into() == max || (bytes == 2 && dst[3].into() == 0 && src[3].into() != 0) {
                    *dst = src;
                } else if src[3].into() == 0 {
                    if bytes == 2 && dst[3].into() == 0 {
                        *dst = [zero; 4];
                    }
                } else {
                    T::over(src, dst);
                }
            }
        }
        if target.is_none_or(|target| target == frame_index as usize) {
            let mode = if bytes == 2 {
                PixelMode::Rgba
            } else {
                match color_type {
                    png::ColorType::Grayscale if bit_depth == png::BitDepth::One => PixelMode::One,
                    png::ColorType::Grayscale | png::ColorType::Indexed => PixelMode::L,
                    png::ColorType::GrayscaleAlpha => PixelMode::La,
                    png::ColorType::Rgb => PixelMode::Rgb,
                    png::ColorType::Rgba => PixelMode::Rgba,
                }
            };
            let pixels = T::pack(&canvas, mode);
            let mut image = if bytes == 2 {
                Image::from_sample_bytes(width, height, PixelMode::Rgba, pixels, 16, Some("PNG".into()))
            } else {
                Image::from_pixels(width, height, mode, pixels, Some("PNG".into()))
            }
            .map_err(|e| e.to_string())?;
            if color_type == png::ColorType::Indexed {
                image.palette = Some((PixelMode::Rgb, palette.clone().ok_or("missing APNG palette")?));
            }
            sequence.frames.push(Frame {
                image,
                duration: control.map(|c| f64::from(c.delay_num) / f64::from(if c.delay_den == 0 { 100 } else { c.delay_den }) * 1000.0),
                disposal: control.map_or(0, |c| c.dispose_op as u8),
                blend: control.map_or(0, |c| c.blend_op as u8),
            });
        }
        if target == Some(frame_index as usize) {
            if frame_index + 1 == count {
                reader.finish().map_err(|e| e.to_string())?;
            }
            return Ok(sequence);
        }
        if default_image && frame_index == 0 {
            canvas.fill([zero; 4]);
        } else if restore_previous {
            if previous.len() == canvas.len() {
                std::mem::swap(&mut canvas, &mut previous);
            } else {
                for (row, saved) in (y..y + output.height).zip(previous.chunks_exact(output.width as usize)) {
                    let start = row as usize * width as usize + x as usize;
                    canvas[start..start + output.width as usize].copy_from_slice(saved);
                }
            }
        } else if control.is_some_and(|c| c.dispose_op == png::DisposeOp::Background) {
            for row in y..y + output.height {
                let start = row as usize * width as usize + x as usize;
                canvas[start..start + output.width as usize].fill([zero; 4]);
            }
        }
    }
    reader.finish().map_err(|e| e.to_string())?;
    Ok(sequence)
}

fn frame_delay(duration: f64) -> PyResult<(u16, u16)> {
    if !duration.is_finite() || !(0.0..=65_535_000.0).contains(&duration) {
        return Err(PyValueError::new_err("cannot write duration"));
    }
    let value = duration / 1000.0;
    let mut remaining = value;
    let (mut p0, mut q0, mut p1, mut q1) = (0u64, 1u64, 1u64, 0u64);
    loop {
        let a = remaining.floor() as u64;
        if q1 != 0 && a > (65535 - q0) / q1 {
            break;
        }
        let (p2, q2) = (p0 + a * p1, q0 + a * q1);
        (p0, q0, p1, q1) = (p1, q1, p2, q2);
        let fraction = remaining - a as f64;
        if fraction == 0.0 {
            break;
        }
        remaining = 1.0 / fraction;
    }
    if q1 == 0 {
        return Err(PyValueError::new_err("cannot write duration"));
    }
    let k = (65535 - q0) / q1;
    let (p2, q2) = (p0 + k * p1, q0 + k * q1);
    if q2 > 0 && (p2 as f64 / q2 as f64 - value).abs() < (p1 as f64 / q1 as f64 - value).abs() {
        (p1, q1) = (p2, q2);
    }
    let numerator = u16::try_from(p1).map_err(|_| PyValueError::new_err("cannot write duration"))?;
    Ok((numerator, q1 as u16))
}

pub fn encode_sequence(sequence: &Sequence<impl Borrow<Image>>, options: SaveOptions) -> PyResult<Vec<u8>> {
    if sequence.frames.is_empty() {
        return Err(PyValueError::new_err("empty sequence"));
    }
    let size = (
        sequence.frames.iter().map(|frame| frame.image.borrow().width).max().unwrap(),
        sequence.frames.iter().map(|frame| frame.image.borrow().height).max().unwrap(),
    );
    encode_sequence_regions(sequence, options, size, &vec![(0, 0); sequence.frames.len()])
}

pub fn encode_sequence_regions(
    sequence: &Sequence<impl Borrow<Image>>, options: SaveOptions, size: (u32, u32), offsets: &[(u32, u32)],
) -> PyResult<Vec<u8>> {
    let first = sequence.frames.first().ok_or_else(|| PyValueError::new_err("empty sequence"))?.image.borrow();
    if offsets.len() != sequence.frames.len() || (sequence.default_image && sequence.frames.len() < 2) {
        return Err(PyValueError::new_err("invalid animation frames or offsets"));
    }
    for (frame, &(x, y)) in sequence.frames.iter().zip(offsets) {
        let image = frame.image.borrow();
        if !matches!(image.mode, PixelMode::One | PixelMode::L | PixelMode::La | PixelMode::Rgb | PixelMode::Rgba)
            || image.mode != first.mode
            || image.bit_depth != first.bit_depth
            || x.checked_add(image.width).is_none_or(|right| right > size.0)
            || y.checked_add(image.height).is_none_or(|bottom| bottom > size.1)
            || image.palette.as_ref().is_some_and(|(mode, _)| !matches!(mode, PixelMode::Rgb | PixelMode::Rgba))
        {
            return Err(PyValueError::new_err("invalid animation frame mode, depth, or bounds"));
        }
    }
    let mut output = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut output, size.0, size.1);
        encoder.set_color(if first.palette.is_some() {
            png::ColorType::Indexed
        } else {
            match first.mode {
                PixelMode::One | PixelMode::L => png::ColorType::Grayscale,
                PixelMode::La => png::ColorType::GrayscaleAlpha,
                PixelMode::Rgb => png::ColorType::Rgb,
                _ => png::ColorType::Rgba,
            }
        });
        if let Some((mode, entries)) = &first.palette {
            let mut rgb = Vec::new();
            let mut alpha = Vec::new();
            for entry in entries.chunks_exact(mode.channels()) {
                rgb.extend_from_slice(&entry[..3]);
                if *mode == PixelMode::Rgba {
                    alpha.push(entry[3]);
                }
            }
            let required = sequence
                .frames
                .iter()
                .map(|frame| frame.image.borrow().pixel_data().map(|pixels| pixels.iter().copied().max().unwrap_or(0)))
                .collect::<PyResult<Vec<_>>>()?
                .into_iter()
                .max()
                .unwrap_or(0);
            rgb.resize(rgb.len().max((usize::from(required) + 1) * 3), 0);
            encoder.set_palette(rgb);
            if !alpha.is_empty() {
                encoder.set_trns(alpha);
            }
        }
        encoder.set_depth(if first.bit_depth > 8 {
            png::BitDepth::Sixteen
        } else if first.mode == PixelMode::One {
            png::BitDepth::One
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
        let mut pixels = Vec::new();
        for (i, frame) in sequence.frames.iter().enumerate() {
            let image = frame.image.borrow();
            writer.reset_frame_position().map_err(codec_error)?;
            writer.set_frame_dimension(image.width, image.height).map_err(codec_error)?;
            writer.set_frame_position(offsets[i].0, offsets[i].1).map_err(codec_error)?;
            if !(sequence.default_image && i == 0) {
                let duration = frame.duration.unwrap_or(0.0);
                let (num, den) = frame_delay(duration)?;
                writer.set_frame_delay(num, den).map_err(codec_error)?;
                writer
                    .set_dispose_op(png::DisposeOp::from_u8(frame.disposal).ok_or_else(|| PyValueError::new_err("invalid APNG disposal"))?)
                    .map_err(codec_error)?;
                writer
                    .set_blend_op(png::BlendOp::from_u8(frame.blend).ok_or_else(|| PyValueError::new_err("invalid APNG blend"))?)
                    .map_err(codec_error)?;
            }
            if first.bit_depth > 8 {
                let raw = image.raw_data()?;
                pixels.resize(raw.len(), 0);
                blanket_core::simd::samples_to_be16(raw, &mut pixels, image.bit_depth);
                writer.write_image_data(&pixels).map_err(codec_error)?;
            } else if image.mode == PixelMode::One {
                let packed = blanket_core::raster::pack_bilevel(image.pixel_data()?, image.width as usize, image.height as usize);
                writer.write_image_data(&packed).map_err(codec_error)?;
            } else {
                writer.write_image_data(image.pixel_data()?).map_err(codec_error)?;
            }
        }
        writer.finish().map_err(codec_error)?;
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_public_encoder_inputs_return_errors() {
        let sequence: Sequence<Image> = Sequence { frames: vec![], loop_count: None, default_image: false };
        assert!(encode_sequence(&sequence, SaveOptions::default()).is_err());
        assert!(encode_sequence_regions(&sequence, SaveOptions::default(), (1, 1), &[]).is_err());
        let image = Image::from_pixels(1, 1, PixelMode::Rgb, vec![0; 3], None).unwrap();
        assert!(super::super::encode_palette_png(&image, 1).is_err());
        assert!(super::super::encode_png(&image, &[], 1).is_err());
        let sequence = Sequence::single(image);
        assert!(encode_sequence_regions(&sequence, SaveOptions::default(), (1, 1), &[]).is_err());
    }

    #[test]
    fn canvas_packing_selects_output_channels_without_rgba_intermediate() {
        for count in [0, 1, 15, 16, 17, 65537] {
            let storage: Vec<[u8; 4]> = (0..count + 1).map(|i| std::array::from_fn(|c| (i * 71 + c * 37) as u8)).collect();
            let canvas = &storage[1..];
            for (mode, channels) in [
                (PixelMode::One, &[0][..]),
                (PixelMode::L, &[0]),
                (PixelMode::La, &[0, 3]),
                (PixelMode::Rgb, &[0, 1, 2]),
                (PixelMode::Rgba, &[0, 1, 2, 3]),
            ] {
                let expected: Vec<u8> = canvas.iter().flat_map(|pixel| channels.iter().map(move |&c| pixel[c])).collect();
                assert_eq!(<u8 as CanvasSample>::pack(canvas, mode), expected);
            }
        }
    }

    #[test]
    fn animated_sources_preserve_modes_and_sample_widths() {
        for wide in [false, true] {
            let samples = if wide { [0x1234_u16, 0x4567, 0x89ab, 0xcdef] } else { [17, 81, 157, 239] };
            let max = if wide { 65535 } else { 255 };
            for color in [png::ColorType::Grayscale, png::ColorType::GrayscaleAlpha, png::ColorType::Rgb, png::ColorType::Rgba] {
                let source: Vec<u16> = samples[..color.samples()].repeat(17 * 3);
                let pixels: Vec<u8> =
                    if wide { source.iter().flat_map(|v| v.to_be_bytes()).collect() } else { source.iter().map(|&v| v as u8).collect() };
                let mut encoded = Vec::new();
                let mut encoder = png::Encoder::new(&mut encoded, 17, 3);
                encoder.set_color(color);
                encoder.set_depth(if wide { png::BitDepth::Sixteen } else { png::BitDepth::Eight });
                encoder.set_animated(2, 0).unwrap();
                let mut writer = encoder.write_header().unwrap();
                writer.write_image_data(&pixels).unwrap();
                writer.set_dispose_op(png::DisposeOp::Previous).unwrap();
                writer.write_image_data(&pixels).unwrap();
                writer.finish().unwrap();
                let expected = match color {
                    png::ColorType::Grayscale => [samples[0], samples[0], samples[0], max],
                    png::ColorType::GrayscaleAlpha => [samples[0], samples[0], samples[0], samples[1]],
                    png::ColorType::Rgb => [samples[0], samples[1], samples[2], max],
                    _ => samples,
                }
                .repeat(17 * 3);
                let expected: Vec<u8> = if wide { expected.iter().flat_map(|v| v.to_le_bytes()).collect() } else { pixels };
                let sequence = decode_sequence(&encoded).unwrap();
                for frame in sequence.frames {
                    let mode = if wide {
                        PixelMode::Rgba
                    } else {
                        match color {
                            png::ColorType::Grayscale => PixelMode::L,
                            png::ColorType::GrayscaleAlpha => PixelMode::La,
                            png::ColorType::Rgb => PixelMode::Rgb,
                            _ => PixelMode::Rgba,
                        }
                    };
                    assert_eq!(frame.image.mode, mode);
                    assert_eq!(frame.image.raw_data().unwrap(), expected);
                }
            }
        }
    }

    #[test]
    fn partial_disposal_restores_changing_regions_at_both_depths() {
        for wide in [false, true] {
            let max = if wide { 65535_u16 } else { 255 };
            let red = [max, 0, 0, max];
            let green = [0, max, 0, max];
            let blue = [0, 0, max, max];
            let frames = [
                (0, 0, 19, 5, png::DisposeOp::None, red),
                (2, 1, 17, 3, png::DisposeOp::Previous, green),
                (7, 2, 1, 1, png::DisposeOp::Previous, blue),
                (1, 0, 16, 2, png::DisposeOp::Background, green),
                (18, 4, 1, 1, png::DisposeOp::None, blue),
            ];
            let pack = |pixels: &[[u16; 4]], big_endian: bool| -> Vec<u8> {
                let samples = pixels.iter().flatten().copied();
                if !wide {
                    samples.map(|v| v as u8).collect()
                } else if big_endian {
                    samples.flat_map(u16::to_be_bytes).collect()
                } else {
                    samples.flat_map(u16::to_le_bytes).collect()
                }
            };
            let mut data = Vec::new();
            let mut encoder = png::Encoder::new(&mut data, 19, 5);
            encoder.set_color(png::ColorType::Rgba);
            encoder.set_depth(if wide { png::BitDepth::Sixteen } else { png::BitDepth::Eight });
            encoder.set_animated(frames.len() as u32, 0).unwrap();
            let mut writer = encoder.write_header().unwrap();
            for &(x, y, w, h, disposal, color) in &frames {
                writer.set_frame_position(0, 0).unwrap();
                writer.set_frame_dimension(w, h).unwrap();
                writer.set_frame_position(x, y).unwrap();
                writer.set_dispose_op(disposal).unwrap();
                writer.write_image_data(&pack(&vec![color; w as usize * h as usize], true)).unwrap();
            }
            writer.finish().unwrap();
            let sequence = decode_sequence(&data).unwrap();
            assert_eq!(sequence.frames.len(), frames.len());
            for (i, &(x, y, w, h, _, color)) in frames.iter().enumerate() {
                let mut expected = vec![red; 19 * 5];
                if i == 4 {
                    for row in 0..2 {
                        expected[row * 19 + 1..row * 19 + 17].fill([0; 4]);
                    }
                }
                for row in y..y + h {
                    expected[row as usize * 19 + x as usize..row as usize * 19 + (x + w) as usize].fill(color);
                }
                let image = &sequence.frames[i].image;
                assert_eq!(image.bit_depth, if wide { 16 } else { 8 });
                assert_eq!(image.raw_data().unwrap(), pack(&expected, false));
            }
        }
    }
}
