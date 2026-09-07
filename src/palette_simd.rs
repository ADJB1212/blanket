//! Fixed-size palette kernels. At 256 entries, Rayon scheduling costs more
//! than these kernels; keep work local to the calling thread.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub(super) fn linear_lut(white: u8) -> [u16; 256] {
    let mut output = [0; 256];
    #[cfg(target_arch = "aarch64")]
    // SAFETY: NEON is baseline on AArch64; output contains 256 elements.
    unsafe {
        linear_neon(white, &mut output);
        return output;
    }
    #[cfg(target_arch = "x86_64")]
    // SAFETY: SSE2 is baseline on x86_64.
    unsafe {
        linear_sse2(white, &mut output);
        return output;
    }
    #[cfg(target_arch = "x86")]
    if is_x86_feature_detected!("sse2") {
        // SAFETY: checked SSE2 above.
        unsafe {
            linear_sse2(white, &mut output);
        }
        return output;
    }
    #[allow(unreachable_code)]
    {
        for (i, value) in output.iter_mut().enumerate() {
            *value = (i as u16 * u16::from(white)) / 255;
        }
        output
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn linear_neon(white: u8, output: &mut [u16; 256]) {
    unsafe {
        let mut indices = vld1q_u16([0, 1, 2, 3, 4, 5, 6, 7].as_ptr());
        for block in output.as_chunks_mut::<8>().0 {
            let product = vmulq_u16(indices, vdupq_n_u16(u16::from(white)));
            // Exact division by 255 for products in 0..=65025.
            let quotient = vshrq_n_u16::<8>(vaddq_u16(vaddq_u16(product, vdupq_n_u16(1)), vshrq_n_u16::<8>(product)));
            vst1q_u16(block.as_mut_ptr(), quotient);
            indices = vaddq_u16(indices, vdupq_n_u16(8));
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn linear_sse2(white: u8, output: &mut [u16; 256]) {
    unsafe {
        let mut indices = _mm_setr_epi16(0, 1, 2, 3, 4, 5, 6, 7);
        for block in output.as_chunks_mut::<8>().0 {
            let product = _mm_mullo_epi16(indices, _mm_set1_epi16(i16::from(white)));
            let quotient = _mm_srli_epi16::<8>(_mm_add_epi16(_mm_add_epi16(product, _mm_set1_epi16(1)), _mm_srli_epi16::<8>(product)));
            _mm_storeu_si128(block.as_mut_ptr().cast(), quotient);
            indices = _mm_add_epi16(indices, _mm_set1_epi16(8));
        }
    }
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
