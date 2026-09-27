//! Four-lane ImageMath kernels with scalar fallbacks for unsupported operations.

use super::Operation;

#[inline]
pub(super) fn apply(output: &mut [[u8; 4]], a: &[[u8; 4]], b: Option<&[[u8; 4]]>, op: Operation, float: bool) -> usize {
    assert!(a.len() >= output.len() && b.is_none_or(|b| b.len() >= output.len()));
    if matches!(op, Operation::Mod | Operation::Pow | Operation::Lshift | Operation::Rshift) || (!float && matches!(op, Operation::Div)) {
        return 0;
    }
    #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
    // SAFETY: NEON is baseline, and input lengths were checked above.
    unsafe {
        return neon::apply(output, a, b, op, float);
    }
    #[cfg(all(target_arch = "x86_64", target_endian = "little"))]
    // SAFETY: SSE2 is baseline, and input lengths were checked above.
    unsafe {
        return sse2::apply(output, a, b, op, float);
    }
    #[cfg(target_arch = "x86")]
    if is_x86_feature_detected!("sse2") {
        // SAFETY: SSE2 and input lengths were checked above.
        unsafe {
            return sse2::apply(output, a, b, op, float);
        }
    }
    #[allow(unreachable_code)]
    0
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::Operation;
    use std::arch::aarch64::*;

    #[target_feature(enable = "neon")]
    unsafe fn integer(op: Operation, a: int32x4_t, b: int32x4_t) -> int32x4_t {
        match op {
            Operation::Abs => vabsq_s32(a),
            Operation::Neg => vnegq_s32(a),
            Operation::Invert => vmvnq_s32(a),
            Operation::Add => vaddq_s32(a, b),
            Operation::Sub => vsubq_s32(a, b),
            Operation::Mul => vmulq_s32(a, b),
            Operation::And => vandq_s32(a, b),
            Operation::Or => vorrq_s32(a, b),
            Operation::Xor => veorq_s32(a, b),
            Operation::Min => vminq_s32(a, b),
            Operation::Max => vmaxq_s32(a, b),
            Operation::Eq => vreinterpretq_s32_u32(vandq_u32(vceqq_s32(a, b), vdupq_n_u32(1))),
            Operation::Ne => vreinterpretq_s32_u32(vandq_u32(vmvnq_u32(vceqq_s32(a, b)), vdupq_n_u32(1))),
            Operation::Lt => vreinterpretq_s32_u32(vandq_u32(vcltq_s32(a, b), vdupq_n_u32(1))),
            Operation::Le => vreinterpretq_s32_u32(vandq_u32(vcleq_s32(a, b), vdupq_n_u32(1))),
            Operation::Gt => vreinterpretq_s32_u32(vandq_u32(vcgtq_s32(a, b), vdupq_n_u32(1))),
            Operation::Ge => vreinterpretq_s32_u32(vandq_u32(vcgeq_s32(a, b), vdupq_n_u32(1))),
            _ => unreachable!("operation uses scalar fallback"),
        }
    }

    #[target_feature(enable = "neon")]
    unsafe fn float(op: Operation, a: float32x4_t, b: float32x4_t) -> float32x4_t {
        match op {
            Operation::Abs => vabsq_f32(a),
            Operation::Neg => vnegq_f32(a),
            Operation::Add => vaddq_f32(a, b),
            Operation::Sub => vsubq_f32(a, b),
            Operation::Mul => vmulq_f32(a, b),
            Operation::Div => vbslq_f32(vceqq_f32(b, vdupq_n_f32(0.0)), vdupq_n_f32(0.0), vdivq_f32(a, b)),
            Operation::Min => vbslq_f32(vcltq_f32(a, b), a, b),
            Operation::Max => vbslq_f32(vcgtq_f32(a, b), a, b),
            Operation::Eq => vbslq_f32(vceqq_f32(a, b), vdupq_n_f32(1.0), vdupq_n_f32(0.0)),
            Operation::Ne => vbslq_f32(vmvnq_u32(vceqq_f32(a, b)), vdupq_n_f32(1.0), vdupq_n_f32(0.0)),
            Operation::Lt => vbslq_f32(vcltq_f32(a, b), vdupq_n_f32(1.0), vdupq_n_f32(0.0)),
            Operation::Le => vbslq_f32(vcleq_f32(a, b), vdupq_n_f32(1.0), vdupq_n_f32(0.0)),
            Operation::Gt => vbslq_f32(vcgtq_f32(a, b), vdupq_n_f32(1.0), vdupq_n_f32(0.0)),
            Operation::Ge => vbslq_f32(vcgeq_f32(a, b), vdupq_n_f32(1.0), vdupq_n_f32(0.0)),
            _ => unreachable!("operation uses scalar fallback"),
        }
    }

    #[target_feature(enable = "neon")]
    pub(super) unsafe fn apply(output: &mut [[u8; 4]], a: &[[u8; 4]], b: Option<&[[u8; 4]]>, op: Operation, float_mode: bool) -> usize {
        let len = output.len() / 4 * 4;
        // SAFETY: Each load/store covers four pixels within the validated slices.
        unsafe {
            for i in (0..len).step_by(4) {
                if float_mode {
                    let left = vld1q_f32(a.as_ptr().add(i).cast());
                    let right = b.map_or_else(|| vdupq_n_f32(0.0), |b| vld1q_f32(b.as_ptr().add(i).cast()));
                    vst1q_f32(output.as_mut_ptr().add(i).cast(), float(op, left, right));
                } else {
                    let left = vld1q_s32(a.as_ptr().add(i).cast());
                    let right = b.map_or_else(|| vdupq_n_s32(0), |b| vld1q_s32(b.as_ptr().add(i).cast()));
                    vst1q_s32(output.as_mut_ptr().add(i).cast(), integer(op, left, right));
                }
            }
        }
        len
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod sse2 {
    use super::Operation;
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    #[target_feature(enable = "sse2")]
    unsafe fn select(mask: __m128i, a: __m128i, b: __m128i) -> __m128i {
        _mm_or_si128(_mm_and_si128(mask, a), _mm_andnot_si128(mask, b))
    }
    #[target_feature(enable = "sse2")]
    unsafe fn multiply(a: __m128i, b: __m128i) -> __m128i {
        let even = _mm_mul_epu32(a, b);
        let odd = _mm_mul_epu32(_mm_srli_si128::<4>(a), _mm_srli_si128::<4>(b));
        _mm_unpacklo_epi32(_mm_shuffle_epi32::<0x88>(even), _mm_shuffle_epi32::<0x88>(odd))
    }

    #[target_feature(enable = "sse2")]
    unsafe fn integer(op: Operation, a: __m128i, b: __m128i) -> __m128i {
        unsafe {
            match op {
                Operation::Abs => _mm_sub_epi32(_mm_xor_si128(a, _mm_srai_epi32::<31>(a)), _mm_srai_epi32::<31>(a)),
                Operation::Neg => _mm_sub_epi32(_mm_setzero_si128(), a),
                Operation::Invert => _mm_xor_si128(a, _mm_set1_epi32(-1)),
                Operation::Add => _mm_add_epi32(a, b),
                Operation::Sub => _mm_sub_epi32(a, b),
                Operation::Mul => multiply(a, b),
                Operation::And => _mm_and_si128(a, b),
                Operation::Or => _mm_or_si128(a, b),
                Operation::Xor => _mm_xor_si128(a, b),
                Operation::Min => select(_mm_cmpgt_epi32(a, b), b, a),
                Operation::Max => select(_mm_cmpgt_epi32(a, b), a, b),
                Operation::Eq => _mm_and_si128(_mm_cmpeq_epi32(a, b), _mm_set1_epi32(1)),
                Operation::Ne => _mm_and_si128(_mm_xor_si128(_mm_cmpeq_epi32(a, b), _mm_set1_epi32(-1)), _mm_set1_epi32(1)),
                Operation::Lt => _mm_and_si128(_mm_cmpgt_epi32(b, a), _mm_set1_epi32(1)),
                Operation::Le => _mm_and_si128(_mm_xor_si128(_mm_cmpgt_epi32(a, b), _mm_set1_epi32(-1)), _mm_set1_epi32(1)),
                Operation::Gt => _mm_and_si128(_mm_cmpgt_epi32(a, b), _mm_set1_epi32(1)),
                Operation::Ge => _mm_and_si128(_mm_xor_si128(_mm_cmpgt_epi32(b, a), _mm_set1_epi32(-1)), _mm_set1_epi32(1)),
                _ => unreachable!("operation uses scalar fallback"),
            }
        }
    }

    #[target_feature(enable = "sse2")]
    unsafe fn float(op: Operation, a: __m128, b: __m128) -> __m128 {
        match op {
            Operation::Abs => _mm_andnot_ps(_mm_set1_ps(-0.0), a),
            Operation::Neg => _mm_xor_ps(_mm_set1_ps(-0.0), a),
            Operation::Add => _mm_add_ps(a, b),
            Operation::Sub => _mm_sub_ps(a, b),
            Operation::Mul => _mm_mul_ps(a, b),
            Operation::Div => _mm_andnot_ps(_mm_cmpeq_ps(b, _mm_setzero_ps()), _mm_div_ps(a, b)),
            Operation::Min => _mm_min_ps(a, b),
            Operation::Max => _mm_max_ps(a, b),
            Operation::Eq => _mm_and_ps(_mm_cmpeq_ps(a, b), _mm_set1_ps(1.0)),
            Operation::Ne => _mm_and_ps(_mm_cmpneq_ps(a, b), _mm_set1_ps(1.0)),
            Operation::Lt => _mm_and_ps(_mm_cmplt_ps(a, b), _mm_set1_ps(1.0)),
            Operation::Le => _mm_and_ps(_mm_cmple_ps(a, b), _mm_set1_ps(1.0)),
            Operation::Gt => _mm_and_ps(_mm_cmpgt_ps(a, b), _mm_set1_ps(1.0)),
            Operation::Ge => _mm_and_ps(_mm_cmpge_ps(a, b), _mm_set1_ps(1.0)),
            _ => unreachable!("operation uses scalar fallback"),
        }
    }

    #[target_feature(enable = "sse2")]
    pub(super) unsafe fn apply(output: &mut [[u8; 4]], a: &[[u8; 4]], b: Option<&[[u8; 4]]>, op: Operation, float_mode: bool) -> usize {
        let len = output.len() / 4 * 4;
        // SAFETY: Each load/store covers four pixels within the validated slices.
        unsafe {
            for i in (0..len).step_by(4) {
                if float_mode {
                    let left = _mm_loadu_ps(a.as_ptr().add(i).cast());
                    let right = b.map_or_else(|| _mm_setzero_ps(), |b| _mm_loadu_ps(b.as_ptr().add(i).cast()));
                    _mm_storeu_ps(output.as_mut_ptr().add(i).cast(), float(op, left, right));
                } else {
                    let left = _mm_loadu_si128(a.as_ptr().add(i).cast());
                    let right = b.map_or_else(|| _mm_setzero_si128(), |b| _mm_loadu_si128(b.as_ptr().add(i).cast()));
                    _mm_storeu_si128(output.as_mut_ptr().add(i).cast(), integer(op, left, right));
                }
            }
        }
        len
    }
}
