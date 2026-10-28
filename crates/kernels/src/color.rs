// Pillow-compatible BT.601 luminance coefficients, scaled by 2^16.
pub const LUMA_RED: u32 = 19_595;
pub const LUMA_GREEN: u32 = 38_470;
pub const LUMA_BLUE: u32 = 7_471;
// Half a fixed-point unit rounds luminance before the fractional bits are removed.
pub const LUMA_BITS: u32 = 16;
pub const LUMA_ROUND: u32 = 1 << (LUMA_BITS - 1);
