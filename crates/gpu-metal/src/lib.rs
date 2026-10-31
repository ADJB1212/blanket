// Shared helpers precede the scalar entry points used by native Metal and MPS.
pub const SHADERS: &str = concat!(
    include_str!("shaders/common.metal"),
    "\n",
    include_str!("shaders/color.metal"),
    "\n",
    include_str!("shaders/geometry.metal"),
    "\n",
    include_str!("shaders/filters.metal"),
    "\n",
    include_str!("shaders/reductions.metal"),
    "\n",
    include_str!("shaders/arithmetic.metal"),
    "\n",
    include_str!("shaders/kernels.metal"),
);
pub const TILE_SHADERS: &str = include_str!("shaders/tiles.metal");

// Native entry points reuse the shared helpers and operation specialization.
#[cfg(target_os = "macos")]
const NATIVE_SHADERS: &str = concat!(
    include_str!("shaders/packed.metal"),
    "\n",
    include_str!("shaders/pixels.metal"),
    "\n",
    include_str!("shaders/tiles.metal"),
    "\n",
    include_str!("shaders/blur.metal"),
    "\n",
    include_str!("shaders/statistics.metal"),
    "\n",
    include_str!("shaders/paste.metal"),
);

#[cfg(target_os = "macos")]
mod native;
#[cfg(target_os = "macos")]
pub use native::{Context, Image, Plan};
