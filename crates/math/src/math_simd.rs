//! Four-lane ImageMath kernels with scalar fallbacks for unsupported operations.

use std::simd::cmp::{SimdOrd, SimdPartialEq, SimdPartialOrd};
use std::simd::num::{SimdFloat, SimdInt, SimdUint};
use std::simd::{Select, f32x4, i32x4};

use super::Operation;

#[derive(Clone, Copy)]
pub(super) enum IntegerInput<'a> {
    Bytes(&'a [u8]),
    Wide(&'a [[u8; 4]]),
    Scalar(i32),
}

impl<'a> IntegerInput<'a> {
    pub(super) fn new(data: &'a [u8], byte: bool) -> Self {
        if byte { Self::Bytes(data) } else { Self::Wide(data.as_chunks::<4>().0) }
    }

    pub(super) fn slice(self, start: usize, len: usize) -> Self {
        match self {
            Self::Bytes(data) => Self::Bytes(&data[start..start + len]),
            Self::Wide(data) => Self::Wide(&data[start..start + len]),
            Self::Scalar(_) => self,
        }
    }

    pub(super) fn sample(self, index: usize) -> i32 {
        match self {
            Self::Bytes(data) => i32::from(data[index]),
            Self::Wide(data) => i32::from_le_bytes(data[index]),
            Self::Scalar(value) => value,
        }
    }

    fn vector(self, index: usize) -> i32x4 {
        match self {
            Self::Bytes(data) => std::simd::Simd::<u8, 4>::from_slice(&data[index..index + 4]).cast(),
            Self::Wide(data) => i32x4::from_array(std::array::from_fn(|lane| i32::from_le_bytes(data[index + lane]))),
            Self::Scalar(value) => i32x4::splat(value),
        }
    }
}

#[inline(always)]
pub(super) fn apply_integer(output: &mut [[u8; 4]], a: IntegerInput<'_>, b: IntegerInput<'_>, op: Operation, reverse: bool) {
    let end = if matches!(op, Operation::Pow) { 0 } else { output.len() / 4 * 4 };
    for i in (0..end).step_by(4) {
        let (a, b) = if reverse { (b.vector(i), a.vector(i)) } else { (a.vector(i), b.vector(i)) };
        output[i..i + 4].copy_from_slice(&integer(op, a, b).to_array().map(i32::to_le_bytes));
    }
    for (i, out) in output.iter_mut().enumerate().skip(end) {
        let (a, b) = if reverse { (b.sample(i), a.sample(i)) } else { (a.sample(i), b.sample(i)) };
        *out = op.integer(a, b).to_le_bytes();
    }
}

#[inline(always)]
pub(super) fn apply_float(
    output: &mut [std::mem::MaybeUninit<[u8; 4]>], a: &[[u8; 4]], b: Option<&[[u8; 4]]>, scalar: Option<f32>, op: Operation, reverse: bool,
) {
    assert!(a.len() >= output.len() && b.is_none_or(|b| b.len() >= output.len()));
    let end = if matches!(op, Operation::Mod | Operation::Pow) { 0 } else { output.len() / 4 * 4 };
    let write = |dst: &mut [std::mem::MaybeUninit<[u8; 4]>; 4], left: &[[u8; 4]; 4], right| {
        let left = f32x4::from_array(left.map(f32::from_le_bytes));
        let (left, right) = if reverse { (right, left) } else { (left, right) };
        *dst = float_operation(op, left, right).to_array().map(|v| std::mem::MaybeUninit::new(v.to_le_bytes()));
    };
    let dst = output[..end].as_chunks_mut::<4>().0;
    let left = a[..end].as_chunks::<4>().0;
    if let Some(value) = scalar {
        let value = f32x4::splat(value);
        for (dst, left) in dst.iter_mut().zip(left) {
            write(dst, left, value);
        }
    } else if let Some(b) = b {
        for ((dst, left), right) in dst.iter_mut().zip(left).zip(b[..end].as_chunks::<4>().0) {
            write(dst, left, f32x4::from_array(right.map(f32::from_le_bytes)));
        }
    } else {
        for (dst, left) in dst.iter_mut().zip(left) {
            write(dst, left, f32x4::splat(0.0));
        }
    }
    for i in end..output.len() {
        let left = f32::from_le_bytes(a[i]);
        let right = scalar.unwrap_or_else(|| b.map_or(0.0, |b| f32::from_le_bytes(b[i])));
        let (left, right) = if reverse { (right, left) } else { (left, right) };
        output[i].write(op.float(left, right).to_le_bytes());
    }
}

#[inline]
pub(super) fn apply(output: &mut [[u8; 4]], a: &[[u8; 4]], b: Option<&[[u8; 4]]>, op: Operation, float: bool) -> usize {
    assert!(a.len() >= output.len() && b.is_none_or(|b| b.len() >= output.len()));
    if matches!(op, Operation::Pow) || (float && matches!(op, Operation::Mod)) {
        return 0;
    }
    let len = output.len() / 4 * 4;
    for i in (0..len).step_by(4) {
        let left = std::array::from_fn(|lane| a[i + lane]);
        let right = std::array::from_fn(|lane| b.map_or([0; 4], |b| b[i + lane]));
        let result = if float {
            float_operation(op, f32x4::from_array(left.map(f32::from_le_bytes)), f32x4::from_array(right.map(f32::from_le_bytes)))
                .to_array()
                .map(f32::to_le_bytes)
        } else {
            integer(op, i32x4::from_array(left.map(i32::from_le_bytes)), i32x4::from_array(right.map(i32::from_le_bytes)))
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
        Operation::Lshift => a << (b & i32x4::splat(31)),
        Operation::Rshift => a >> (b & i32x4::splat(31)),
        Operation::Div | Operation::Mod => {
            let divisor = b.simd_eq(zero).select(one, b);
            // i32 quotient gaps exceed f64 rounding error, including near integers.
            let quotient = (a.cast::<f64>() / divisor.cast::<f64>()).cast::<i64>().cast::<i32>();
            b.simd_eq(zero).select(zero, if matches!(op, Operation::Div) { quotient } else { a - quotient * b })
        }
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

#[inline(always)]
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
    fn integer_division_matches_wrapping_scalar_across_bit_ranges() {
        let mut state = 0x12345678_u32;
        for shift in 0..32 {
            let mut left = Vec::new();
            let mut right = Vec::new();
            for i in 0..4099 {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                left.push((state as i32).to_le_bytes());
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                let divisor = if i % 17 == 0 {
                    0
                } else if i % 19 == 0 {
                    -1
                } else {
                    (state as i32) >> shift
                };
                right.push(divisor.to_le_bytes());
            }
            for op in [Operation::Div, Operation::Mod] {
                let mut output = vec![[0; 4]; left.len()];
                apply_integer(&mut output, IntegerInput::Wide(&left), IntegerInput::Wide(&right), op, false);
                for ((actual, a), b) in output.iter().zip(&left).zip(&right) {
                    assert_eq!(*actual, op.integer(i32::from_le_bytes(*a), i32::from_le_bytes(*b)).to_le_bytes());
                }
            }
        }
    }

    #[test]
    #[ignore = "native timing; run alone after compilation"]
    fn benchmark_integer_ops() {
        use std::hint::black_box;
        use std::time::Instant;
        let a: Vec<_> = (0..480_000_i32).map(|i| i.wrapping_mul(7919).to_le_bytes()).collect();
        let b: Vec<_> = (0..480_000_i32).map(|i| i.wrapping_mul(37).to_le_bytes()).collect();
        let mut output = vec![[0; 4]; a.len()];
        for op in [Operation::Lshift, Operation::Rshift, Operation::Div, Operation::Mod] {
            for vector in [false, true] {
                let start = Instant::now();
                for _ in 0..1000 {
                    let a = black_box(&a);
                    let b = black_box(&b);
                    let dst = black_box(&mut output);
                    if vector {
                        apply_integer(dst, IntegerInput::Wide(a), IntegerInput::Wide(b), op, false);
                    } else {
                        for ((dst, a), b) in dst.iter_mut().zip(a).zip(b) {
                            *dst = op.integer(i32::from_le_bytes(*a), i32::from_le_bytes(*b)).to_le_bytes();
                        }
                    }
                }
                let name = match op {
                    Operation::Lshift => "lshift",
                    Operation::Rshift => "rshift",
                    Operation::Div => "div",
                    Operation::Mod => "mod",
                    _ => unreachable!(),
                };
                eprintln!("op={name}, vector={vector}, us={:.3}", start.elapsed().as_secs_f64() * 1000.0);
            }
        }
    }

    #[test]
    fn integer_inputs_preserve_unaligned_samples_tails_and_guards() {
        let values = [i32::MIN, i32::MAX, -1, 0, 1, 31, 32, -33, 123456789];
        for len in [0, 1, 3, 4, 5, 15, 16, 17, 67] {
            for offset in 0..16 {
                let bytes: Vec<u8> = (0..offset + len).map(|i| (i * 79) as u8).collect();
                let wide: Vec<u8> = std::iter::repeat_n(0, offset).chain((0..len).flat_map(|i| values[i % values.len()].to_le_bytes())).collect();
                let inputs = [IntegerInput::Bytes(&bytes[offset..]), IntegerInput::new(&wide[offset..], false), IntegerInput::Scalar(-1)];
                for a in inputs {
                    for b in inputs {
                        for reverse in [false, true] {
                            for op in Operation::ALL {
                                let mut output = vec![[0xa5; 4]; len + 2];
                                apply_integer(&mut output[1..len + 1], a, b, op, reverse);
                                assert_eq!(output[0], [0xa5; 4]);
                                assert_eq!(output[len + 1], [0xa5; 4]);
                                for (i, actual) in output[1..len + 1].iter().enumerate() {
                                    let (a, b) = if reverse { (b.sample(i), a.sample(i)) } else { (a.sample(i), b.sample(i)) };
                                    assert_eq!(*actual, op.integer(a, b).to_le_bytes());
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn scalar_broadcast_preserves_operand_order_tails_and_guards() {
        let values = [f32::NAN, f32::NEG_INFINITY, -3.5, -0.0, 0.0, 1.0, f32::INFINITY, f32::from_bits(1)];
        for name in ["add", "sub", "mul", "div", "mod", "pow", "eq", "ne", "lt", "le", "gt", "ge", "min", "max"] {
            let op = Operation::parse(name).unwrap();
            for value in values {
                for reverse in [false, true] {
                    for len in [0, 1, 3, 4, 5, 15, 16, 17, 67] {
                        for offset in 0..16 {
                            let source: Vec<u8> =
                                std::iter::repeat_n(0, offset).chain((0..len).flat_map(|i| values[i % values.len()].to_le_bytes())).collect();
                            let input = source[offset..].as_chunks::<4>().0;
                            let mut uninit = vec![std::mem::MaybeUninit::new([0xa5; 4]); len + 2];
                            apply_float(&mut uninit[1..len + 1], input, None, Some(value), op, reverse);
                            // SAFETY: the guarded elements started initialized and the kernel wrote every output sample.
                            let initialized: Vec<_> = uninit.iter().map(|sample| unsafe { sample.assume_init() }).collect();
                            assert_eq!(initialized[0], [0xa5; 4]);
                            assert_eq!(initialized[len + 1], [0xa5; 4]);
                            for (actual, sample) in initialized[1..len + 1].iter().zip(input) {
                                let sample = f32::from_le_bytes(*sample);
                                let expected = if reverse { op.float(value, sample) } else { op.float(sample, value) };
                                if expected.is_nan() {
                                    assert!(f32::from_le_bytes(*actual).is_nan(), "{name}");
                                } else {
                                    assert_eq!(*actual, expected.to_le_bytes(), "{name}, reverse={reverse}");
                                }
                            }
                        }
                    }
                }
            }
        }
    }

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
                        let done = apply(output[offset..offset + len * 4].as_chunks_mut::<4>().0, a, (!op.unary()).then_some(b), op, float);
                        let fallback = matches!(op, Operation::Pow) || (float && matches!(op, Operation::Mod));
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
                                assert_eq!(*actual, op.integer(i32::from_le_bytes(a[i]), i32::from_le_bytes(b[i])).to_le_bytes(), "{name}");
                            }
                        }
                    }
                }
            }
        }
    }
}
