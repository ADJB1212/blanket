#![feature(portable_simd)]
pub mod font;
mod font_simd;
pub use font::register;
