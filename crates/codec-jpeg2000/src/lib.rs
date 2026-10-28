#![feature(default_field_values)]

use std::ffi::{CStr, c_void};
use std::ptr;

use blanket_codec_common::{codec_error, validate_dimensions};
use blanket_core::{Image, PixelMode};
use openjpeg_sys::*;
use pyo3::exceptions::PyOSError;
use pyo3::prelude::*;

struct Handles {
    codec: *mut opj_codec_t,
    stream: *mut opj_stream_t,
    image: *mut opj_image_t,
}

impl Drop for Handles {
    fn drop(&mut self) {
        // SAFETY: Each non-null handle is owned by this struct and freed once.
        unsafe {
            if !self.codec.is_null() {
                opj_destroy_codec(self.codec);
            }
            if !self.stream.is_null() {
                opj_stream_destroy(self.stream);
            }
            if !self.image.is_null() {
                opj_image_destroy(self.image);
            }
        }
    }
}

struct Buffer<'a> {
    input: &'a [u8] = &[],
    output: Vec<u8> = Vec::new(),
    position: usize = 0,
    writing: bool = false,
}

unsafe extern "C" fn read(buffer: *mut c_void, count: usize, context: *mut c_void) -> usize {
    // SAFETY: OpenJPEG supplies a writable count-byte buffer and the live Buffer context.
    unsafe {
        let state = &mut *context.cast::<Buffer<'_>>();
        let count = count.min(state.input.len().saturating_sub(state.position));
        if count == 0 {
            return usize::MAX;
        }
        ptr::copy_nonoverlapping(state.input.as_ptr().add(state.position), buffer.cast(), count);
        state.position += count;
        count
    }
}

unsafe extern "C" fn write(buffer: *mut c_void, count: usize, context: *mut c_void) -> usize {
    // SAFETY: OpenJPEG supplies count readable bytes and the live Buffer context.
    unsafe {
        let state = &mut *context.cast::<Buffer<'_>>();
        let Some(end) = state.position.checked_add(count).filter(|&end| end <= i32::MAX as usize) else {
            return usize::MAX;
        };
        if end > state.output.len() {
            if state.output.try_reserve(end - state.output.len()).is_err() {
                return usize::MAX;
            }
            state.output.resize(end, 0);
        }
        ptr::copy_nonoverlapping(buffer.cast(), state.output.as_mut_ptr().add(state.position), count);
        state.position = end;
        count
    }
}

unsafe extern "C" fn seek(position: i64, context: *mut c_void) -> i32 {
    // SAFETY: The context is the Buffer registered on this synchronous stream.
    let state = unsafe { &mut *context.cast::<Buffer<'_>>() };
    let limit = if state.writing { i32::MAX as usize } else { state.input.len() };
    match usize::try_from(position) {
        Ok(position) if position <= limit => {
            state.position = position;
            1
        }
        _ => 0,
    }
}

unsafe extern "C" fn skip(offset: i64, context: *mut c_void) -> i64 {
    // SAFETY: The context is live; seek validates the resulting position.
    unsafe {
        let position = (*context.cast::<Buffer<'_>>()).position as i64;
        match position.checked_add(offset) {
            Some(next) if seek(next, context) != 0 => offset,
            _ => -1,
        }
    }
}

impl Handles {
    fn configure_threads(&self, parallel: bool) {
        if parallel && std::env::var_os("OPJ_NUM_THREADS").is_none() {
            let threads = std::thread::available_parallelism().map_or(1, |count| count.get().min(8));
            if threads > 1 {
                // SAFETY: The codec is live and has not created a tile coder yet.
                // OpenJPEG falls back to serial execution if worker creation fails.
                unsafe {
                    opj_codec_set_threads(self.codec, threads as i32);
                }
            }
        }
    }

    fn new(format: CODEC_FORMAT, buffer: &mut Buffer<'_>) -> Result<Self, String> {
        // SAFETY: The buffer outlives all synchronous codec calls. No callback owns it.
        unsafe {
            let handles = Self {
                codec: if buffer.writing { opj_create_compress(format) } else { opj_create_decompress(format) },
                stream: opj_stream_default_create(i32::from(!buffer.writing)),
                image: ptr::null_mut(),
            };
            if handles.codec.is_null() || handles.stream.is_null() {
                return Err("could not allocate OpenJPEG handles".into());
            }
            opj_stream_set_user_data(handles.stream, ptr::from_mut(buffer).cast(), None);
            opj_stream_set_user_data_length(handles.stream, buffer.input.len() as u64);
            opj_stream_set_read_function(handles.stream, Some(read));
            opj_stream_set_write_function(handles.stream, Some(write));
            opj_stream_set_seek_function(handles.stream, Some(seek));
            opj_stream_set_skip_function(handles.stream, Some(skip));
            Ok(handles)
        }
    }
}

pub fn backend_version() -> String {
    // SAFETY: OpenJPEG returns a static, nul-terminated version string.
    unsafe { format!("OpenJPEG {}", CStr::from_ptr(opj_version()).to_string_lossy()) }
}

pub fn inspect(data: &[u8]) -> Result<blanket_codec_common::ImageHeader, String> {
    let format = if data.starts_with(&[0xff, 0x4f]) { CODEC_FORMAT::OPJ_CODEC_J2K } else { CODEC_FORMAT::OPJ_CODEC_JP2 };
    let mut buffer = Buffer { input: data, .. };
    let mut handles = Handles::new(format, &mut buffer)?;
    // SAFETY: The handles own the initialized decoder and header. No pixel decode is requested.
    unsafe {
        let mut parameters = std::mem::zeroed();
        opj_set_default_decoder_parameters(&raw mut parameters);
        if opj_setup_decoder(handles.codec, &raw mut parameters) == 0
            || opj_read_header(handles.stream, handles.codec, &raw mut handles.image) == 0
            || handles.image.is_null()
        {
            return Err("invalid JPEG 2000 header".into());
        }
        let header = &*handles.image;
        if header.comps.is_null() || !(1..=4).contains(&header.numcomps) {
            return Err("unsupported JPEG 2000 component layout".into());
        }
        let components = std::slice::from_raw_parts(header.comps, header.numcomps as usize);
        let depth = components[0].prec;
        if !(1..=16).contains(&depth) || components.iter().any(|c| c.prec != depth || c.sgnd != 0) {
            return Err("unsupported JPEG 2000 sample precision".into());
        }
        let mode = match header.numcomps {
            1 => "L",
            2 => "LA",
            3 => "RGB",
            _ => "RGBA",
        };
        blanket_codec_common::ImageHeader::new(
            header.x1.checked_sub(header.x0).ok_or("invalid JPEG 2000 width")?,
            header.y1.checked_sub(header.y0).ok_or("invalid JPEG 2000 height")?,
            mode,
            depth as u8,
        )
    }
}

pub fn decode_jpeg2000(data: &[u8]) -> Result<Image, String> {
    let format = if data.starts_with(&[0xff, 0x4f]) { CODEC_FORMAT::OPJ_CODEC_J2K } else { CODEC_FORMAT::OPJ_CODEC_JP2 };
    let mut buffer = Buffer { input: data, .. };
    let mut handles = Handles::new(format, &mut buffer)?;
    // SAFETY: OpenJPEG initializes parameters and owns all image/component allocations.
    // Header dimensions and component layouts are validated before reading sample buffers.
    unsafe {
        let mut parameters = std::mem::zeroed();
        opj_set_default_decoder_parameters(&raw mut parameters);
        if opj_setup_decoder(handles.codec, &raw mut parameters) == 0 {
            return Err("invalid JPEG 2000 decoder parameters".into());
        }
        handles.configure_threads(data.len() >= 32 * 1024);
        if opj_read_header(handles.stream, handles.codec, &raw mut handles.image) == 0 || handles.image.is_null() {
            return Err("invalid JPEG 2000 header".into());
        }
        let header = &*handles.image;
        let width = header.x1.checked_sub(header.x0).ok_or("invalid JPEG 2000 width")?;
        let height = header.y1.checked_sub(header.y0).ok_or("invalid JPEG 2000 height")?;
        validate_dimensions(width, height)?;
        if width == 0 || height == 0 || !(1..=4).contains(&header.numcomps) || header.comps.is_null() {
            return Err("unsupported JPEG 2000 component layout".into());
        }
        if !matches!(
            header.color_space,
            COLOR_SPACE::OPJ_CLRSPC_UNKNOWN | COLOR_SPACE::OPJ_CLRSPC_UNSPECIFIED | COLOR_SPACE::OPJ_CLRSPC_SRGB | COLOR_SPACE::OPJ_CLRSPC_GRAY
        ) {
            return Err("JPEG 2000 requires grayscale or RGB color space".into());
        }
        let channels = header.numcomps as usize;
        let components = std::slice::from_raw_parts(header.comps, channels);
        let depth = components[0].prec;
        if !(1..=16).contains(&depth)
            || components.iter().any(|c| c.prec != depth || c.sgnd != 0 || c.w != width || c.h != height || c.dx != 1 || c.dy != 1)
        {
            return Err("unsupported JPEG 2000 sample precision or subsampling".into());
        }
        if opj_decode(handles.codec, handles.stream, handles.image) == 0 || opj_end_decompress(handles.codec, handles.stream) == 0 {
            return Err("could not decode JPEG 2000 pixels".into());
        }
        let header = &*handles.image;
        if header.numcomps as usize != channels || header.comps.is_null() {
            return Err("unsupported JPEG 2000 channel mapping".into());
        }
        let components = std::slice::from_raw_parts(header.comps, channels);
        if components.iter().any(|c| c.data.is_null() || c.w != width || c.h != height) {
            return Err("invalid JPEG 2000 component data".into());
        }
        let count = width as usize * height as usize;
        let planes: Vec<_> = components.iter().map(|c| std::slice::from_raw_parts(c.data, count)).collect();
        let mode = match channels {
            1 => PixelMode::L,
            2 => PixelMode::La,
            3 => PixelMode::Rgb,
            _ => PixelMode::Rgba,
        };
        let maximum = (1_i32 << depth) - 1;
        let pack = |wide: bool, output: &mut [u8]| {
            macro_rules! convert {
                ($wide:literal) => {
                    match channels {
                        1 => blanket_core::pixels::pack_i32_planes::<1, $wide>([planes[0]], maximum, output),
                        2 => blanket_core::pixels::pack_i32_planes::<2, $wide>([planes[0], planes[1]], maximum, output),
                        3 => blanket_core::pixels::pack_i32_planes::<3, $wide>([planes[0], planes[1], planes[2]], maximum, output),
                        4 => blanket_core::pixels::pack_i32_planes::<4, $wide>([planes[0], planes[1], planes[2], planes[3]], maximum, output),
                        _ => unreachable!(),
                    }
                };
            }
            if wide {
                convert!(true);
            } else {
                convert!(false);
            }
        };
        if depth <= 8 {
            let mut pixels = vec![0; count * channels];
            pack(false, &mut pixels);
            Image::from_pixels(width, height, mode, pixels, Some("JPEG2000".into())).map_err(|e| e.to_string())
        } else {
            let mut samples = vec![0; count * if channels == 2 { 4 } else { channels } * 2];
            pack(true, &mut samples);
            Image::from_sample_bytes(width, height, if channels == 2 { PixelMode::Rgba } else { mode }, samples, 16, Some("JPEG2000".into()))
                .map_err(|e| e.to_string())
        }
    }
}

pub fn encode_jpeg2000(image: &Image) -> PyResult<Vec<u8>> {
    if image.width == 0
        || image.height == 0
        || !matches!(image.mode, PixelMode::One | PixelMode::L | PixelMode::La | PixelMode::Rgb | PixelMode::Rgba)
    {
        return Err(PyOSError::new_err("JPEG 2000 requires a nonempty grayscale or RGB image"));
    }
    let mut buffer = Buffer { writing: true, .. };
    let mut handles = Handles::new(CODEC_FORMAT::OPJ_CODEC_JP2, &mut buffer).map_err(codec_error)?;
    let channels = image.mode.channels();
    let depth = u32::from(image.bit_depth);
    // SAFETY: OpenJPEG initializes all C structures, owns planar allocations, and
    // receives validated image dimensions. Handles frees resources on every exit.
    unsafe {
        let mut components =
            vec![opj_image_cmptparm_t { dx: 1, dy: 1, w: image.width, h: image.height, x0: 0, y0: 0, prec: depth, bpp: depth, sgnd: 0 }; channels];
        handles.image = opj_image_create(
            channels as u32,
            components.as_mut_ptr(),
            if channels <= 2 { COLOR_SPACE::OPJ_CLRSPC_GRAY } else { COLOR_SPACE::OPJ_CLRSPC_SRGB },
        );
        if handles.image.is_null() {
            return Err(PyOSError::new_err("could not allocate JPEG 2000 image"));
        }
        (*handles.image).x1 = image.width;
        (*handles.image).y1 = image.height;
        let planes = std::slice::from_raw_parts_mut((*handles.image).comps, channels);
        let pixels = image.raw_data()?;
        for (channel, plane) in planes.iter_mut().enumerate() {
            if plane.data.is_null() {
                return Err(PyOSError::new_err("could not allocate JPEG 2000 samples"));
            }
            plane.alpha = u16::from(channel == channels - 1 && matches!(channels, 2 | 4));
            let output = std::slice::from_raw_parts_mut(plane.data, image.width as usize * image.height as usize);
            macro_rules! extract {
                ($wide:literal) => {
                    match channels {
                        1 => blanket_core::pixels::extract_i32_channel::<1, $wide>(pixels, output, channel),
                        2 => blanket_core::pixels::extract_i32_channel::<2, $wide>(pixels, output, channel),
                        3 => blanket_core::pixels::extract_i32_channel::<3, $wide>(pixels, output, channel),
                        4 => blanket_core::pixels::extract_i32_channel::<4, $wide>(pixels, output, channel),
                        _ => unreachable!(),
                    }
                };
            }
            if depth > 8 {
                extract!(true);
            } else {
                extract!(false);
            }
        }
        let mut parameters = std::mem::zeroed();
        opj_set_default_encoder_parameters(&raw mut parameters);
        parameters.tcp_numlayers = 1;
        parameters.cp_disto_alloc = 1;
        parameters.tcp_rates[0] = 0.0;
        parameters.numresolution = (image.width.min(image.height).ilog2() + 1).min(6) as i32;
        if opj_setup_encoder(handles.codec, &raw mut parameters, handles.image) == 0 {
            return Err(PyOSError::new_err("could not configure JPEG 2000 encoder"));
        }
        handles.configure_threads(u64::from(image.width) * u64::from(image.height) >= 256 * 1024);
        if opj_start_compress(handles.codec, handles.image, handles.stream) == 0
            || opj_encode(handles.codec, handles.stream) == 0
            || opj_end_compress(handles.codec, handles.stream) == 0
        {
            return Err(PyOSError::new_err("could not encode JPEG 2000 image"));
        }
    }
    drop(handles);
    Ok(buffer.output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn input_callbacks_bound_reads_and_preserve_failed_seek_position() {
        let mut buffer = Buffer { input: &[1, 2, 3], .. };
        let context = ptr::from_mut(&mut buffer).cast();
        let mut output = [0xcc_u8; 8];
        // SAFETY: All buffers and the context remain live and match the requested sizes.
        unsafe {
            assert_eq!(read(output.as_mut_ptr().add(1).cast(), 6, context), 3);
            assert_eq!(output, [0xcc, 1, 2, 3, 0xcc, 0xcc, 0xcc, 0xcc]);
            assert_eq!(read(output.as_mut_ptr().cast(), 1, context), usize::MAX);
            assert_eq!(seek(-1, context), 0);
            assert_eq!(seek(4, context), 0);
            assert_eq!(buffer.position, 3);
            assert_eq!(skip(-2, context), -2);
            assert_eq!(read(output.as_mut_ptr().cast(), 1, context), 1);
            assert_eq!(output[0], 2);
        }
    }

    #[test]
    fn output_callbacks_support_header_rewrites_and_zero_filled_holes() {
        let mut buffer = Buffer { writing: true, .. };
        let context = ptr::from_mut(&mut buffer).cast();
        let mut input = [1_u8, 2, 3];
        // SAFETY: The callback only reads the provided input and writes its live context.
        unsafe {
            assert_eq!(write(input.as_mut_ptr().cast(), input.len(), context), 3);
            assert_eq!(seek(1, context), 1);
            assert_eq!(write(input.as_mut_ptr().cast(), 1, context), 1);
            assert_eq!(seek(5, context), 1);
            assert_eq!(write(input.as_mut_ptr().cast(), 1, context), 1);
            assert_eq!(buffer.output, [1, 1, 3, 0, 0, 1]);
            assert_eq!(seek(i64::MAX, context), 0);
            assert_eq!(buffer.position, 6);
        }
    }
}
