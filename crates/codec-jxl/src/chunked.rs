use std::ffi::c_void;

use blanket_codec_common::codec_error;
use jpegxl_sys::common::types::{JxlBool, JxlPixelFormat};
use jpegxl_sys::encoder::encode::*;
use pyo3::prelude::*;

// jpegxl-sys 0.13 makes these C struct fields private. Keep the public C ABI
// layouts here until the bindings expose constructors for the callbacks.
#[repr(C)]
struct InputSource {
    opaque: *mut c_void,
    color_format: extern "C-unwind" fn(*mut c_void, *mut JxlPixelFormat),
    color_data: extern "C-unwind" fn(*mut c_void, usize, usize, usize, usize, *mut usize) -> *const c_void,
    extra_format: extern "C-unwind" fn(*mut c_void, usize, *mut JxlPixelFormat),
    extra_data: extern "C-unwind" fn(*mut c_void, usize, usize, usize, usize, usize, *mut usize) -> *const c_void,
    release: extern "C-unwind" fn(*mut c_void, *const c_void),
}

#[repr(C)]
struct OutputProcessor {
    opaque: *mut c_void,
    get: extern "C-unwind" fn(*mut c_void, *mut usize) -> *mut c_void,
    release: extern "C-unwind" fn(*mut c_void, usize),
    seek: Option<extern "C-unwind" fn(*mut c_void, u64)>,
    finalized: extern "C-unwind" fn(*mut c_void, u64),
}

struct Input {
    pixels: *const c_void,
    format: JxlPixelFormat,
    stride: usize,
    bytes_per_pixel: usize,
}

extern "C-unwind" fn color_format(opaque: *mut c_void, format: *mut JxlPixelFormat) {
    // SAFETY: libjxl supplies a writable format and our live Input context.
    unsafe { *format = (*opaque.cast::<Input>()).format };
}

extern "C-unwind" fn color_data(opaque: *mut c_void, x: usize, y: usize, _: usize, _: usize, stride: *mut usize) -> *const c_void {
    // SAFETY: libjxl requests rectangles within the validated input image.
    // The immutable source remains alive until synchronous encoding finishes.
    unsafe {
        let input = &*opaque.cast::<Input>();
        *stride = input.stride;
        input.pixels.cast::<u8>().add(y * input.stride + x * input.bytes_per_pixel).cast()
    }
}

extern "C-unwind" fn extra_format(_: *mut c_void, _: usize, _: *mut JxlPixelFormat) {}
extern "C-unwind" fn extra_data(_: *mut c_void, _: usize, _: usize, _: usize, _: usize, _: usize, _: *mut usize) -> *const c_void {
    // Alpha is interleaved in color_data; there are no other extra channels.
    std::ptr::null()
}
extern "C-unwind" fn release_input(_: *mut c_void, _: *const c_void) {}

#[derive(Default)]
struct Output {
    bytes: Vec<u8>,
    position: usize,
    failed: bool,
}

extern "C-unwind" fn get_output(opaque: *mut c_void, size: *mut usize) -> *mut c_void {
    // SAFETY: output callbacks are serialized and the context outlives encoding.
    unsafe {
        let output = &mut *opaque.cast::<Output>();
        let requested = (*size).max(64 * 1024);
        let end = output.position.checked_add(requested);
        if output.failed || end.is_none_or(|end| output.bytes.try_reserve(end.saturating_sub(output.bytes.len())).is_err()) {
            output.failed = true;
            *size = 0;
            return std::ptr::null_mut();
        }
        *size = output.bytes.capacity() - output.position;
        output.bytes.as_mut_ptr().add(output.position).cast()
    }
}

extern "C-unwind" fn release_output(opaque: *mut c_void, written: usize) {
    // SAFETY: libjxl initialized exactly `written` bytes in the acquired buffer.
    unsafe {
        let output = &mut *opaque.cast::<Output>();
        output.position += written;
        output.bytes.set_len(output.bytes.len().max(output.position));
    }
}

extern "C-unwind" fn finalized(_: *mut c_void, _: u64) {}

pub(super) fn encode(
    encoder: &super::encode::Encoder, settings: *const JxlEncoderFrameSettings, pixels: *const c_void, length: usize, format: JxlPixelFormat,
    width: u32,
) -> PyResult<Vec<u8>> {
    let bytes_per_pixel = format.num_channels as usize * if format.data_type == jpegxl_sys::common::types::JxlDataType::Uint16 { 2 } else { 1 };
    let stride = width as usize * bytes_per_pixel;
    if stride == 0 || !length.is_multiple_of(stride) {
        return Err(codec_error("invalid JPEG XL input buffer"));
    }
    let mut input = Input { pixels, format, stride, bytes_per_pixel };
    let mut output = Output::default();
    let source = InputSource { opaque: (&raw mut input).cast(), color_format, color_data, extra_format, extra_data, release: release_input };
    let sink = OutputProcessor { opaque: (&raw mut output).cast(), get: get_output, release: release_output, seek: None, finalized };
    // SAFETY: both pairs are repr(C) with identical field types and order from
    // jxl/encode.h. transmute also checks their sizes at compile time.
    let source = unsafe { std::mem::transmute::<InputSource, JxlChunkedFrameInputSource>(source) };
    let sink = unsafe { std::mem::transmute::<OutputProcessor, JxlEncoderOutputProcessor>(sink) };
    // SAFETY: input samples and both callback contexts live through synchronous
    // AddChunkedFrame. Reset clears all references before the contexts drop,
    // including on failure; the outer Session also resets the cached encoder.
    let status = unsafe {
        let status = JxlEncoderSetOutputProcessor(encoder.0, sink);
        let status = if status == JxlEncoderStatus::Success { JxlEncoderAddChunkedFrame(settings, JxlBool::True, source) } else { status };
        JxlEncoderReset(encoder.0);
        status
    };
    if output.failed || status != JxlEncoderStatus::Success {
        return Err(codec_error("JPEG XL chunked encoding failed"));
    }
    if output.bytes.capacity() > output.bytes.len().saturating_mul(4).max(4096) {
        output.bytes.shrink_to_fit();
    }
    Ok(output.bytes)
}
