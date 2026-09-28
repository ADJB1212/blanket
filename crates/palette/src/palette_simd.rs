//! Fixed-size palette kernels.
use std::simd::Simd;

pub(super) fn linear_lut(white: u8) -> [u16; 256] {
    let mut output = [0; 256];
    let mut indices = Simd::<u16, 8>::from_array([0, 1, 2, 3, 4, 5, 6, 7]);
    for block in output.as_chunks_mut::<8>().0 {
        let product = indices * Simd::splat(u16::from(white));
        ((product + Simd::splat(1) + (product >> 8)) >> 8).copy_to_slice(block);
        indices += Simd::splat(8);
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_byte_lut_matches_integer_division() {
        for white in 0..=255 {
            for (index, value) in linear_lut(white).into_iter().enumerate() {
                assert_eq!(value, (index as u16 * u16::from(white)) / 255);
            }
        }
    }
}
