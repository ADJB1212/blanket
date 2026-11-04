use std::simd::Simd;

// Pillow-compatible BT.601 luminance coefficients, scaled by 2^16.
pub const LUMA_RED: u32 = 19_595;
pub const LUMA_GREEN: u32 = 38_470;
pub const LUMA_BLUE: u32 = 7_471;
// Half a fixed-point unit rounds luminance before the fractional bits are removed.
pub const LUMA_BITS: u32 = 16;
pub const LUMA_ROUND: u32 = 1 << (LUMA_BITS - 1);
#[inline]
pub fn luminance(r: u16, g: u16, b: u16) -> u16 {
    ((u32::from(r) * LUMA_RED + u32::from(g) * LUMA_GREEN + u32::from(b) * LUMA_BLUE + LUMA_ROUND) >> LUMA_BITS) as u16
}

/// Channels must contain samples no greater than u16::MAX.
#[inline]
pub fn luminance_simd(r: Simd<u32, 16>, g: Simd<u32, 16>, b: Simd<u32, 16>) -> Simd<u32, 16> {
    (r * Simd::splat(LUMA_RED) + g * Simd::splat(LUMA_GREEN) + b * Simd::splat(LUMA_BLUE) + Simd::splat(LUMA_ROUND)) >> LUMA_BITS
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn luminance_preserves_fixed_point_rounding_at_all_sample_depths() {
        for maximum in [255_u16, 1023, 4095, 65535] {
            let values = [0, 1, maximum / 2, maximum - 1, maximum];
            for r in values {
                for g in values {
                    let blue = std::array::from_fn(|i| u32::from(values[i % values.len()]));
                    let actual = luminance_simd(Simd::splat(u32::from(r)), Simd::splat(u32::from(g)), Simd::from_array(blue)).to_array();
                    for (b, actual) in blue.into_iter().zip(actual) {
                        let expected = ((u64::from(r) * 19595 + u64::from(g) * 38470 + u64::from(b) * 7471 + 32768) >> 16) as u16;
                        assert_eq!(luminance(r, g, b as u16), expected);
                        assert_eq!(actual, u32::from(expected));
                    }
                }
            }
        }
    }
}
