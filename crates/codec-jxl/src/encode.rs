use std::cell::RefCell;

use blanket_codec_common::{SaveOptions, codec_error};
use blanket_core::Image;
use jpegxl_rs::parallel::ParallelRunner;
use jpegxl_sys::encoder::encode::{JxlEncoder, JxlEncoderCreate, JxlEncoderDestroy, JxlEncoderReset};
use pyo3::prelude::*;

use super::JxlSamples;

pub(super) struct Encoder(pub(super) *mut JxlEncoder);

impl Drop for Encoder {
    fn drop(&mut self) {
        // SAFETY: this thread exclusively owns a non-null encoder.
        unsafe { JxlEncoderDestroy(self.0) };
    }
}

struct Session<'a>(&'a Encoder);

impl Drop for Session<'_> {
    fn drop(&mut self) {
        // SAFETY: reset releases all input, output, profile, and runner references
        // before their borrows end, including when encoding fails.
        unsafe { JxlEncoderReset(self.0.0) };
    }
}

thread_local! {
    static ENCODER: RefCell<Option<Encoder>> = const { RefCell::new(None) };
}

pub(super) fn encode(image: &Image, pixels: JxlSamples<'_>, options: SaveOptions, runner: Option<&dyn ParallelRunner>) -> PyResult<Vec<u8>> {
    ENCODER.with(|cache| {
        let mut cache = cache.borrow_mut();
        if cache.is_none() {
            // SAFETY: null selects libjxl's default allocator.
            let raw = unsafe { JxlEncoderCreate(std::ptr::null()) };
            if raw.is_null() {
                return Err(codec_error("cannot allocate JPEG XL encoder"));
            }
            *cache = Some(Encoder(raw));
        }
        let session = Session(cache.as_ref().expect("encoder initialized above"));
        super::encode_native(session.0, image, pixels, options, runner)
    })
}
