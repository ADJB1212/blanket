use std::cell::RefCell;
use std::io::Cursor;

use blanket_codec_common::{DecodeError, ImageHeader, codec_error, validate_dimensions};
use blanket_core::{Image, PixelMode};
use exr::block::reader::Reader;
use exr::meta::attribute::SampleType;
use exr::meta::header::Header;
use exr::prelude::{Encoding, ReadChannels, ReadLayers, SpecificChannels, Vec2, WritableImage, f16};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub const SIGNATURE: &[u8; 4] = b"\x76\x2f\x31\x01";

fn image_layout(headers: &[Header]) -> Result<(u32, u32, PixelMode), String> {
    let [header] = headers else {
        return Err("only single-part OpenEXR images are supported".into());
    };
    if header.deep {
        return Err("deep OpenEXR images are not supported".into());
    }
    let channel = |name| header.channels.list.iter().find(|channel| channel.name.eq(name));
    let alpha = channel("A");
    for name in ["R", "G", "B"].into_iter().chain(alpha.map(|_| "A")) {
        let channel = channel(name).ok_or("OpenEXR requires R, G and B channels; other layouts are unsupported")?;
        if channel.sample_type == SampleType::U32 || channel.sampling != Vec2(1, 1) {
            return Err("OpenEXR requires full-resolution HALF or FLOAT color and alpha channels; other layouts are unsupported".into());
        }
    }
    let width = u32::try_from(header.layer_size.width()).map_err(|error| error.to_string())?;
    let height = u32::try_from(header.layer_size.height()).map_err(|error| error.to_string())?;
    if width == 0 || height == 0 {
        return Err("empty OpenEXR data window".into());
    }
    let mode = if alpha.is_some() { PixelMode::RgbaF } else { PixelMode::RgbF };
    Ok((width, height, mode))
}

pub fn inspect(data: &[u8]) -> Result<ImageHeader, String> {
    let reader = Reader::read_from_buffered(Cursor::new(data), false).map_err(|error| error.to_string())?;
    let (width, height, mode) = image_layout(reader.headers())?;
    ImageHeader::new(width, height, mode.as_str(), 32)
}

pub fn decode(data: &[u8]) -> Result<Image, DecodeError> {
    let reader = Reader::read_from_buffered(Cursor::new(data), true).map_err(DecodeError::with_diagnostic_source)?;
    let (width, height, mode) = image_layout(reader.headers())?;
    validate_dimensions(width, height)?;
    let stride = mode.channels() * size_of::<f32>();
    let row_width = width as usize;
    let length = (row_width * height as usize).checked_mul(stride).ok_or("OpenEXR dimensions overflow addressable memory")?;
    let mut pixels = Vec::new();
    pixels.try_reserve_exact(length).map_err(|_| "OpenEXR raster exceeds memory limit")?;
    pixels.resize(length, 0);
    let pixels = RefCell::new(pixels);
    let decoded = exr::prelude::read()
        .no_deep_data()
        .largest_resolution_level()
        .rgba_channels(
            |_, _| pixels.take(),
            move |pixels: &mut Vec<u8>, position, (mut r, mut g, mut b, alpha): (f32, f32, f32, f32)| {
                // Zero alpha can carry additive color in OpenEXR.
                if alpha.is_finite() && alpha != 0.0 && alpha != 1.0 {
                    r /= alpha;
                    g /= alpha;
                    b /= alpha;
                }
                let offset = (position.y() * row_width + position.x()) * stride;
                for (dest, sample) in pixels[offset..offset + stride].as_chunks_mut::<4>().0.iter_mut().zip([r, g, b, alpha]) {
                    *dest = sample.to_le_bytes();
                }
            },
        )
        .first_valid_layer()
        .all_attributes()
        .pedantic()
        .non_parallel()
        .from_chunks(reader)
        .map_err(DecodeError::with_diagnostic_source)?;
    let mut image = Image::from_color_float_bytes(width, height, mode, decoded.layer_data.channel_data.pixels).map_err(DecodeError::with_source)?;
    image.format = Some("EXR".into());
    Ok(image)
}

pub fn encode(image: &Image, sample_type: &str) -> PyResult<Vec<u8>> {
    let mode = image.pixel_mode();
    if !matches!(mode, PixelMode::RgbF | PixelMode::RgbaF) {
        return Err(PyValueError::new_err(format!("cannot write mode {} as EXR; convert to RGB;F or RGBA;F first", mode.as_str())));
    }
    if image.width() == 0 || image.height() == 0 {
        return Err(PyValueError::new_err("cannot write empty image as EXR"));
    }
    let half = match sample_type {
        "half" => true,
        "float" => false,
        _ => return Err(PyValueError::new_err("exr_sample_type must be half or float")),
    };
    let size = (image.width() as usize, image.height() as usize);
    let samples = image.raw_data()?.as_chunks::<4>().0;
    let mut output = Cursor::new(Vec::new());
    let encoding = Encoding::SMALL_LOSSLESS;
    let rgb = |Vec2(x, y)| {
        let offset = (y * size.0 + x) * 3;
        let [r, g, b] = samples[offset..offset + 3].try_into().expect("validated RGB raster");
        (f32::from_le_bytes(r), f32::from_le_bytes(g), f32::from_le_bytes(b))
    };
    let rgba = |Vec2(x, y)| {
        let offset = (y * size.0 + x) * 4;
        let packed: [[u8; 4]; 4] = samples[offset..offset + 4].try_into().expect("validated RGBA raster");
        let [mut r, mut g, mut b, alpha] = packed.map(f32::from_le_bytes);
        // Associate color with the alpha value that will actually be stored.
        let alpha = if half { f16::from_f32(alpha).to_f32() } else { alpha };
        if alpha.is_finite() && alpha != 0.0 && alpha != 1.0 {
            r *= alpha;
            g *= alpha;
            b *= alpha;
        }
        (r, g, b, alpha)
    };
    let result = match (mode, half) {
        (PixelMode::RgbF, false) => {
            exr::image::Image::from_encoded_channels(size, encoding, SpecificChannels::rgb(rgb)).write().non_parallel().to_buffered(&mut output)
        }
        (PixelMode::RgbF, true) => {
            let channels = SpecificChannels::rgb(|position| {
                let (r, g, b) = rgb(position);
                (f16::from_f32(r), f16::from_f32(g), f16::from_f32(b))
            });
            exr::image::Image::from_encoded_channels(size, encoding, channels).write().non_parallel().to_buffered(&mut output)
        }
        (PixelMode::RgbaF, false) => {
            exr::image::Image::from_encoded_channels(size, encoding, SpecificChannels::rgba(rgba)).write().non_parallel().to_buffered(&mut output)
        }
        (PixelMode::RgbaF, true) => {
            let channels = SpecificChannels::rgba(|position| {
                let (r, g, b, a) = rgba(position);
                (f16::from_f32(r), f16::from_f32(g), f16::from_f32(b), f16::from_f32(a))
            });
            exr::image::Image::from_encoded_channels(size, encoding, channels).write().non_parallel().to_buffered(&mut output)
        }
        _ => unreachable!("validated EXR mode"),
    };
    result.map_err(codec_error)?;
    Ok(output.into_inner())
}
