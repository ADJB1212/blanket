//! Versioned C descriptor with capsule-owned copy-on-write image storage.

use std::ffi::{CString, c_char};

use blanket_core::Image;
use pyo3::prelude::*;
use pyo3::types::PyCapsule;

#[repr(C)]
#[derive(Debug)]
struct ImageDescriptor {
    abi_version: u32,
    width: u32,
    height: u32,
    bit_depth: u32,
    channels: u32,
    data: *const u8,
    len: usize,
    row_stride: usize,
    mode: *const c_char,
    palette: *const u8,
    palette_len: usize,
    palette_channels: u32,
}

#[repr(C)]
#[derive(Debug)]
struct CapsuleImage {
    descriptor: ImageDescriptor,
    image: Image,
    mode: CString,
}

// SAFETY: All pointers refer to immutable allocations owned by this holder.
// Image writes detach through copy-on-write; consumers must retain the capsule.
unsafe impl Send for CapsuleImage {}

#[pyfunction(signature = (image, copy=false))]
pub(crate) fn image_capsule<'py>(py: Python<'py>, image: &Image, copy: bool) -> PyResult<Bound<'py, PyCapsule>> {
    let mut owner = if copy {
        Image::from_raw_bytes(image.width(), image.height(), image.pixel_mode(), blanket_core::buffers::copy(image.raw_data()?)?, image.bit_depth())?
    } else {
        image.clone()
    };
    owner.copy_palette_from(image);
    let data = owner.raw_data()?;
    let mode = CString::new(owner.pixel_mode().as_str()).expect("image modes contain no NUL");
    let (palette, palette_len, palette_channels) =
        owner.palette().as_ref().map_or((std::ptr::null(), 0, 0), |(mode, data)| (data.as_ptr(), data.len(), mode.channels() as u32));
    let descriptor = ImageDescriptor {
        abi_version: 1,
        width: owner.width(),
        height: owner.height(),
        bit_depth: u32::from(owner.bit_depth()),
        channels: owner.pixel_mode().channels() as u32,
        data: data.as_ptr(),
        len: data.len(),
        row_stride: owner.width() as usize * owner.raw_bytes_per_pixel(),
        mode: mode.as_ptr(),
        palette,
        palette_len,
        palette_channels,
    };
    PyCapsule::new_with_value(py, CapsuleImage { descriptor, image: owner, mode }, c"blanket.image.v1")
}
