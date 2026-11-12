//! Shared image storage, sample validation, and basic pixel access.

mod methods;
mod packing;
mod pixels;
mod queries;
mod storage;
mod validation;

pub use blanket_modes::PixelMode;
pub use packing::{lab_raw_bytes, lab_raw_bytes_mut, pack_bilevel, pack_bilevel_into, unpack_bilevel, unpack_bilevel_inverted};
pub use pixels::{integer_bytes, integer_value};
pub use storage::{Image, SharedPixels, alloc_pixels};

#[cfg(test)]
mod tests;
