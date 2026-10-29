pub const SHADERS: &str = include_str!("kernels.metal");
#[cfg(target_os = "macos")]
mod native;
#[cfg(target_os = "macos")]
pub use native::{Context, Image, Plan};
