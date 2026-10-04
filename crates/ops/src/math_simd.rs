//! Four-lane ImageMath kernels with scalar fallbacks for unsupported operations.

use std::simd::cmp::{SimdOrd, SimdPartialEq, SimdPartialOrd};
use std::simd::num::{SimdFloat, SimdInt};
use std::simd::{Select, f32x4, i32x4};

use super::Operation;

#[inline]
pub(super) fn apply(output: &mut [[u8; 4]], a: &[[u8; 4]], b: Option<&[[u8; 4]]>, op: Operation, float: bool) -> usize {
    assert!(a.len() >= output.len() && b.is_none_or(|b| b.len() >= output.len()));
    if matches!(op, Operation::Mod | Operation::Pow | Operation::Lshift | Operation::Rshift) || (!float && matches!(op, Operation::Div)) {
        return 0;
    }
    let len = output.len() / 4 * 4;
    for i in (0..len).step_by(4) {
        let left = std::array::from_fn(|lane| a[i + lane]);
        let right = std::array::from_fn(|lane| b.map_or([0; 4], |b| b[i + lane]));
        let result = if float {
            float_operation(
                op,
                f32x4::from_array(left.map(f32::from_le_bytes)),
                f32x4::from_array(right.map(f32::from_le_bytes)),
            )
            .to_array()
            .map(f32::to_le_bytes)
        } else {
            integer(
                op,
                i32x4::from_array(left.map(i32::from_le_bytes)),
                i32x4::from_array(right.map(i32::from_le_bytes)),
            )
            .to_array()
            .map(i32::to_le_bytes)
        };
        output[i..i + 4].copy_from_slice(&result);
    }
    len
}

#[inline]
fn integer(op: Operation, a: i32x4, b: i32x4) -> i32x4 {
    let one = i32x4::splat(1);
    let zero = i32x4::splat(0);
    match op {
        Operation::Abs => a.abs(),
        Operation::Neg => -a,
        Operation::Invert => !a,
        Operation::Add => a + b,
        Operation::Sub => a - b,
        Operation::Mul => a * b,
        Operation::And => a & b,
        Operation::Or => a | b,
        Operation::Xor => a ^ b,
        Operation::Min => a.simd_min(b),
        Operation::Max => a.simd_max(b),
        Operation::Eq => a.simd_eq(b).select(one, zero),
        Operation::Ne => a.simd_ne(b).select(one, zero),
        Operation::Lt => a.simd_lt(b).select(one, zero),
        Operation::Le => a.simd_le(b).select(one, zero),
        Operation::Gt => a.simd_gt(b).select(one, zero),
        Operation::Ge => a.simd_ge(b).select(one, zero),
        _ => unreachable!("operation uses scalar fallback"),
    }
}

#[inline]
fn float_operation(op: Operation, a: f32x4, b: f32x4) -> f32x4 {
    let one = f32x4::splat(1.0);
    let zero = f32x4::splat(0.0);
    match op {
        Operation::Abs => a.abs(),
        Operation::Neg => -a,
        Operation::Add => a + b,
        Operation::Sub => a - b,
        Operation::Mul => a * b,
        Operation::Div => b.simd_eq(zero).select(zero, a / b),
        Operation::Min => a.simd_lt(b).select(a, b),
        Operation::Max => a.simd_gt(b).select(a, b),
        Operation::Eq => a.simd_eq(b).select(one, zero),
        Operation::Ne => a.simd_ne(b).select(one, zero),
        Operation::Lt => a.simd_lt(b).select(one, zero),
        Operation::Le => a.simd_le(b).select(one, zero),
        Operation::Gt => a.simd_gt(b).select(one, zero),
        Operation::Ge => a.simd_ge(b).select(one, zero),
        _ => unreachable!("operation uses scalar fallback"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unaligned_buffers_preserve_tails_and_guards() {
        for float in [false, true] {
            for name in [
                "abs", "neg", "invert", "add", "sub", "mul", "div", "mod", "pow", "and", "or", "xor", "lshift", "rshift", "eq", "ne", "lt", "le",
                "gt", "ge", "min", "max",
            ] {
                let op = Operation::parse(name).unwrap();
                if float && op.integer_only() {
                    continue;
                }
                for len in [0, 1, 3, 4, 5, 7, 8, 9, 15, 16, 17] {
                    for offset in 0..16 {
                        let mut left = vec![0; offset + len * 4];
                        let mut right = left.clone();
                        let a = left[offset..].as_chunks_mut::<4>().0;
                        let b = right[offset..].as_chunks_mut::<4>().0;
                        for i in 0..len {
                            a[i] = if float {
                                [f32::NAN, -0.0, 0.0, f32::INFINITY, f32::NEG_INFINITY, -3.5, f32::from_bits(1)][i % 7].to_le_bytes()
                            } else {
                                [i32::MIN, i32::MAX, -1, 0, 1, 123456789, -33][i % 7].to_le_bytes()
                            };
                            b[i] = if float {
                                [0.0, -0.0, f32::NAN, f32::INFINITY, 2.0][i % 5].to_le_bytes()
                            } else {
                                [-1i32, 0, i32::MAX, 2, i32::MIN][i % 5].to_le_bytes()
                            };
                        }
                        let mut output = vec![0xa5; offset + len * 4 + 16];
                        let done = apply(
                            output[offset..offset + len * 4].as_chunks_mut::<4>().0,
                            a,
                            (!op.unary()).then_some(b),
                            op,
                            float,
                        );
                        let fallback = matches!(op, Operation::Mod | Operation::Pow | Operation::Lshift | Operation::Rshift)
                            || (!float && matches!(op, Operation::Div));
                        assert_eq!(done, if fallback { 0 } else { len / 4 * 4 });
                        assert!(output[..offset].iter().all(|&byte| byte == 0xa5));
                        assert!(output[offset + done * 4..].iter().all(|&byte| byte == 0xa5));
                        for (i, actual) in output[offset..offset + done * 4].as_chunks::<4>().0.iter().enumerate() {
                            if float {
                                let expected = op.float(f32::from_le_bytes(a[i]), f32::from_le_bytes(b[i]));
                                if expected.is_nan() {
                                    assert!(f32::from_le_bytes(*actual).is_nan(), "{name}");
                                } else {
                                    assert_eq!(*actual, expected.to_le_bytes(), "{name}");
                                }
                            } else {
                                assert_eq!(
                                    *actual,
                                    op.integer(i32::from_le_bytes(a[i]), i32::from_le_bytes(b[i])).to_le_bytes(),
                                    "{name}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}
