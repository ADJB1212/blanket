//! CUDA kernels compiled at build time and embedded in the extension.

#[cfg(test)]
mod tests;

#[cfg(feature = "precompiled")]
pub const FATBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/blanket.fatbin"));
