#![feature(portable_simd)]
mod geometry;
pub mod hash;
pub mod metrics;
pub use blanket_math::math;
pub mod ops;
pub mod ops_simd;
pub mod smart_crop;
