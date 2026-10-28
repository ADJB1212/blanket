//! Image arithmetic for ImageMath.

use blanket_core::parallel::{CHUNK_PIXELS, chunks_mut, chunks_mut_above};
use blanket_core::{Image, PixelMode};
use pyo3::exceptions::{PyMemoryError, PyOverflowError, PyTypeError, PyValueError};
use pyo3::prelude::*;

#[path = "math_simd.rs"]
mod simd;

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(math_apply, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_math_matches_scalar_with_cropped_rows_and_tails() {
        let integers = [i32::MIN, i32::MAX, -1, 0, 1, 31, 32, -33, 123456789];
        let floats = [f32::NEG_INFINITY, -3.5, -0.0, 0.0, 2.5, f32::INFINITY, f32::NAN, f32::MIN_POSITIVE, f32::from_bits(1)];
        for name in [
            "abs", "neg", "invert", "add", "sub", "mul", "div", "mod", "pow", "and", "or", "xor", "lshift", "rshift", "eq", "ne", "lt", "le", "gt",
            "ge", "min", "max",
        ] {
            let op = Operation::parse(name).unwrap();
            for float in [false, true] {
                if float && op.integer_only() {
                    continue;
                }
                let values: Vec<[u8; 4]> = if float { floats.map(f32::to_le_bytes).to_vec() } else { integers.map(i32::to_le_bytes).to_vec() };
                for width in [1, 3, 4, 5, 17, CHUNK_PIXELS + 3, CHUNK_PIXELS * 4 + 3] {
                    for cropped in [false, true] {
                        let left_width = width + usize::from(cropped) * 3;
                        let right_width = width + usize::from(cropped) * 5;
                        let a: Vec<u8> = (0..left_width * 3).flat_map(|i| values[i % values.len()]).collect();
                        let b: Vec<u8> = (0..right_width * 3).flat_map(|i| values[(i / values.len() + i * 2) % values.len()]).collect();
                        let b = (!op.unary()).then_some(b.as_slice());
                        let scalar = |a, b| {
                            if float {
                                op.float(f32::from_le_bytes(a), f32::from_le_bytes(b)).to_le_bytes()
                            } else {
                                op.integer(i32::from_le_bytes(a), i32::from_le_bytes(b)).to_le_bytes()
                            }
                        };
                        let mut expected = vec![0; width * 3 * 4];
                        let mut actual = expected.clone();
                        apply_pixels(&mut expected, &a, b, [width, left_width, right_width], scalar, |_, _, _| 0);
                        apply_pixels(&mut actual, &a, b, [width, left_width, right_width], scalar, |out, a, b| simd::apply(out, a, b, op, float));
                        if float {
                            let result = float_pixels(expected.len(), &a, b, [width, left_width, right_width], None, op, false).unwrap();
                            for (actual, expected) in result.as_chunks::<4>().0.iter().zip(expected.as_chunks::<4>().0) {
                                if f32::from_le_bytes(*expected).is_nan() {
                                    assert!(f32::from_le_bytes(*actual).is_nan(), "{name}");
                                } else {
                                    assert_eq!(actual, expected, "{name}, width={width}, cropped={cropped}");
                                }
                            }
                        }
                        for (actual, expected) in actual.as_chunks::<4>().0.iter().zip(expected.as_chunks::<4>().0) {
                            if float && f32::from_le_bytes(*expected).is_nan() {
                                assert!(f32::from_le_bytes(*actual).is_nan(), "{name}");
                            } else {
                                assert_eq!(actual, expected, "{name}, float={float}, width={width}, cropped={cropped}");
                            }
                        }
                    }
                }
            }
        }
    }
}

#[derive(Clone, Copy)]
#[repr(usize)]
enum Operation {
    Abs,
    Neg,
    Invert,
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Pow,
    And,
    Or,
    Xor,
    Lshift,
    Rshift,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    Min,
    Max,
}

impl Operation {
    const ALL: [Self; 22] = [
        Self::Abs,
        Self::Neg,
        Self::Invert,
        Self::Add,
        Self::Sub,
        Self::Mul,
        Self::Div,
        Self::Mod,
        Self::Pow,
        Self::And,
        Self::Or,
        Self::Xor,
        Self::Lshift,
        Self::Rshift,
        Self::Eq,
        Self::Ne,
        Self::Lt,
        Self::Le,
        Self::Gt,
        Self::Ge,
        Self::Min,
        Self::Max,
    ];

    fn parse(name: &str) -> PyResult<Self> {
        Ok(match name {
            "abs" => Self::Abs,
            "neg" => Self::Neg,
            "invert" => Self::Invert,
            "add" => Self::Add,
            "sub" => Self::Sub,
            "mul" => Self::Mul,
            "div" => Self::Div,
            "mod" => Self::Mod,
            "pow" => Self::Pow,
            "and" => Self::And,
            "or" => Self::Or,
            "xor" => Self::Xor,
            "lshift" => Self::Lshift,
            "rshift" => Self::Rshift,
            "eq" => Self::Eq,
            "ne" => Self::Ne,
            "lt" => Self::Lt,
            "le" => Self::Le,
            "gt" => Self::Gt,
            "ge" => Self::Ge,
            "min" => Self::Min,
            "max" => Self::Max,
            _ => return Err(PyValueError::new_err("unknown image math operation")),
        })
    }

    fn unary(self) -> bool {
        matches!(self, Self::Abs | Self::Neg | Self::Invert)
    }

    fn integer_only(self) -> bool {
        matches!(self, Self::Invert | Self::And | Self::Or | Self::Xor | Self::Lshift | Self::Rshift)
    }

    #[inline(always)]
    fn integer(self, a: i32, b: i32) -> i32 {
        match self {
            Self::Abs => a.wrapping_abs(),
            Self::Neg => a.wrapping_neg(),
            Self::Invert => !a,
            Self::Add => a.wrapping_add(b),
            Self::Sub => a.wrapping_sub(b),
            Self::Mul => a.wrapping_mul(b),
            Self::Div => {
                if b == 0 {
                    0
                } else {
                    a.wrapping_div(b)
                }
            }
            Self::Mod => {
                if b == 0 {
                    0
                } else {
                    a.wrapping_rem(b)
                }
            }
            Self::Pow => (f64::from(a).powf(f64::from(b)) + 0.5) as i32,
            Self::And => a & b,
            Self::Or => a | b,
            Self::Xor => a ^ b,
            Self::Lshift => a.wrapping_shl(b as u32),
            Self::Rshift => a.wrapping_shr(b as u32),
            Self::Eq => i32::from(a == b),
            Self::Ne => i32::from(a != b),
            Self::Lt => i32::from(a < b),
            Self::Le => i32::from(a <= b),
            Self::Gt => i32::from(a > b),
            Self::Ge => i32::from(a >= b),
            Self::Min => a.min(b),
            Self::Max => a.max(b),
        }
    }

    #[inline(always)]
    fn float(self, a: f32, b: f32) -> f32 {
        match self {
            Self::Abs => a.abs(),
            Self::Neg => -a,
            Self::Add => a + b,
            Self::Sub => a - b,
            Self::Mul => a * b,
            Self::Div => {
                if b == 0.0 {
                    0.0
                } else {
                    a / b
                }
            }
            Self::Mod => {
                if b == 0.0 {
                    0.0
                } else {
                    a % b
                }
            }
            Self::Pow => a.powf(b),
            Self::Eq => u8::from(a == b) as f32,
            Self::Ne => u8::from(a != b) as f32,
            Self::Lt => u8::from(a < b) as f32,
            Self::Le => u8::from(a <= b) as f32,
            Self::Gt => u8::from(a > b) as f32,
            Self::Ge => u8::from(a >= b) as f32,
            Self::Min => {
                if a < b {
                    a
                } else {
                    b
                }
            }
            Self::Max => {
                if a > b {
                    a
                } else {
                    b
                }
            }
            _ => unreachable!("integer operation validated before execution"),
        }
    }
}

fn apply_pixels(
    output: &mut [u8], a: &[u8], b: Option<&[u8]>, [width, left_width, right_width]: [usize; 3],
    operation: impl Fn([u8; 4], [u8; 4]) -> [u8; 4] + Sync + Send,
    vector: impl Fn(&mut [[u8; 4]], &[[u8; 4]], Option<&[[u8; 4]]>) -> usize + Sync + Send,
) {
    let a = a.as_chunks::<4>().0;
    let b = b.map(|data| data.as_chunks::<4>().0);
    chunks_mut(output, CHUNK_PIXELS << 2, |start, chunk| {
        let mut output = chunk.as_chunks_mut::<4>().0;
        let mut index = start * CHUNK_PIXELS;
        while !output.is_empty() {
            let contiguous = left_width == width && (b.is_none() || right_width == width);
            let (ai, bi, len) = if contiguous {
                (index, index, output.len())
            } else {
                let x = index % width;
                let y = index / width;
                (y * left_width + x, y * right_width + x, output.len().min(width - x))
            };
            let (span, rest) = output.split_at_mut(len);
            let left = &a[ai..ai + len];
            let done = vector(span, left, b.map(|b| &b[bi..bi + len]));
            let span = &mut span[done..];
            let left = &left[done..];
            if let Some(b) = b {
                for ((out, &a), &b) in span.iter_mut().zip(left).zip(&b[bi + done..bi + len]) {
                    *out = operation(a, b);
                }
            } else {
                for (out, &a) in span.iter_mut().zip(left) {
                    *out = operation(a, [0; 4]);
                }
            }
            output = rest;
            index += len;
        }
    });
}

fn integer_pixels<const OP: usize>(
    output: &mut [u8], [a, b]: [simd::IntegerInput<'_>; 2], [width, left_width, right_width]: [usize; 3], reverse: bool,
) {
    chunks_mut(output, CHUNK_PIXELS << 2, |start, chunk| {
        let mut output = chunk.as_chunks_mut::<4>().0;
        let mut index = start * CHUNK_PIXELS;
        while !output.is_empty() {
            let contiguous = left_width == width && (matches!(b, simd::IntegerInput::Scalar(_)) || right_width == width);
            let (ai, bi, len) = if contiguous {
                (index, index, output.len())
            } else {
                let x = index % width;
                let y = index / width;
                (y * left_width + x, y * right_width + x, output.len().min(width - x))
            };
            let (span, rest) = output.split_at_mut(len);
            simd::apply_integer(span, a.slice(ai, len), b.slice(bi, len), Operation::ALL[OP], reverse);
            output = rest;
            index += len;
        }
    });
}

fn float_pixels(
    len: usize, a: &[u8], b: Option<&[u8]>, [width, left_width, right_width]: [usize; 3], scalar: Option<f32>, op: Operation, reverse: bool,
) -> PyResult<Vec<u8>> {
    macro_rules! dispatch {
        ($($variant:ident),+ $(,)?) => {
            match op {
                $(Operation::$variant => float_pixels_impl::<{Operation::$variant as usize}>(
                    len, a, b, [width, left_width, right_width], scalar, reverse)),+
            }
        }
    }
    dispatch!(Abs, Neg, Invert, Add, Sub, Mul, Div, Mod, Pow, And, Or, Xor, Lshift, Rshift, Eq, Ne, Lt, Le, Gt, Ge, Min, Max)
}

fn float_pixels_impl<const OP: usize>(
    len: usize, a: &[u8], b: Option<&[u8]>, [width, left_width, right_width]: [usize; 3], scalar: Option<f32>, reverse: bool,
) -> PyResult<Vec<u8>> {
    let mut output = Vec::<[u8; 4]>::new();
    output.try_reserve_exact(len / 4).map_err(|_| PyMemoryError::new_err("cannot allocate image"))?;
    let a = a.as_chunks::<4>().0;
    let b = b.map(|b| b.as_chunks::<4>().0);
    const CHUNK: usize = 64 << 10;
    let threshold = if matches!(Operation::ALL[OP], Operation::Mod | Operation::Pow) { 256 << 10 } else { 4 << 20 };
    chunks_mut_above(&mut output.spare_capacity_mut()[..len / 4], CHUNK, threshold, |chunk, dst| {
        let mut index = chunk * CHUNK;
        let mut dst = dst;
        while !dst.is_empty() {
            let contiguous = left_width == width && (b.is_none() || right_width == width);
            let (ai, bi, count) = if contiguous {
                (index, index, dst.len())
            } else {
                let x = index % width;
                let y = index / width;
                (y * left_width + x, y * right_width + x, dst.len().min(width - x))
            };
            let (span, rest) = dst.split_at_mut(count);
            simd::apply_float(span, &a[ai..ai + count], b.map(|b| &b[bi..bi + count]), scalar, Operation::ALL[OP], reverse);
            index += count;
            dst = rest;
        }
    });
    // SAFETY: each disjoint output span was fully initialized above.
    unsafe { output.set_len(len / 4) };
    Ok(output.into_flattened())
}

#[pyfunction]
#[pyo3(signature = (operation, left, right=None, integer_output=false, *, scalar=None, reverse=false, integer_scalar=None))]
#[allow(clippy::too_many_arguments)]
fn math_apply(
    py: Python<'_>, operation: &str, left: &Image, right: Option<&Image>, integer_output: bool, scalar: Option<f64>, reverse: bool,
    integer_scalar: Option<i32>,
) -> PyResult<Image> {
    let op = Operation::parse(operation)?;
    let has_scalar = scalar.is_some() || integer_scalar.is_some();
    if op.unary() != (right.is_none() && !has_scalar) || (right.is_some() && has_scalar) || (scalar.is_some() && integer_scalar.is_some()) {
        return Err(PyValueError::new_err("wrong number of image math operands"));
    }
    if scalar.is_some() && left.mode != PixelMode::F {
        return Err(PyValueError::new_err("scalar image math requires F mode"));
    }
    let scalar = scalar
        .map(|value| {
            let sample = value as f32;
            if value.is_finite() && !sample.is_finite() { Err(PyOverflowError::new_err("float too large to pack with f format")) } else { Ok(sample) }
        })
        .transpose()?;
    let left_byte = matches!(left.mode, PixelMode::One | PixelMode::L);
    let right_byte = right.is_some_and(|r| matches!(r.mode, PixelMode::One | PixelMode::L));
    let integer_group = |mode: PixelMode| matches!(mode, PixelMode::One | PixelMode::L | PixelMode::I);
    if integer_scalar.is_some() && !integer_group(left.mode) {
        return Err(PyValueError::new_err("integer scalar image math requires 1, L, or I mode"));
    }
    if left_byte || right_byte {
        if !integer_group(left.mode) || right.is_some_and(|r| !integer_group(r.mode)) {
            return Err(PyValueError::new_err("image math requires matching I or F operands"));
        }
    } else if !matches!(left.mode, PixelMode::I | PixelMode::F) || right.is_some_and(|r| r.mode != left.mode) {
        return Err(PyValueError::new_err("image math requires matching I or F operands"));
    }
    let float = left.mode == PixelMode::F;
    if float && op.integer_only() {
        return Err(PyTypeError::new_err(format!("bad operand type for '{operation}'")));
    }
    let a_raw = left.raw_data()?;
    let b_raw = right.map(Image::raw_data).transpose()?;
    let width = right.map_or(left.width, |r| r.width.min(left.width)) as usize;
    let height = right.map_or(left.height, |r| r.height.min(left.height)) as usize;
    let len = width.checked_mul(height).and_then(|n| n.checked_mul(4)).ok_or_else(|| PyMemoryError::new_err("image dimensions are too large"))?;
    if float {
        let output =
            py.detach(|| float_pixels(len, a_raw, b_raw, [width, left.width as usize, right.map_or(0, |r| r.width as usize)], scalar, op, reverse))?;
        return if integer_output {
            Image::from_integer_bytes(width as u32, height as u32, PixelMode::I, output)
        } else {
            Image::from_float_bytes(width as u32, height as u32, output)
        };
    }
    let mut output = Vec::new();
    output.try_reserve_exact(len).map_err(|_| PyMemoryError::new_err("cannot allocate image"))?;
    output.resize(len, 0);
    py.detach(|| {
        let dimensions = [width, left.width as usize, right.map_or(0, |r| r.width as usize)];
        let a = simd::IntegerInput::new(a_raw, left_byte);
        let b = b_raw.map_or_else(|| simd::IntegerInput::Scalar(integer_scalar.unwrap_or(0)), |data| simd::IntegerInput::new(data, right_byte));
        macro_rules! dispatch {
            ($($variant:ident),+ $(,)?) => {
                match op {
                    $(Operation::$variant => {
                        if !left_byte && !right_byte && integer_scalar.is_none() {
                            apply_pixels(&mut output, a_raw, b_raw, dimensions, |a, b| {
                                Operation::$variant.integer(i32::from_le_bytes(a), i32::from_le_bytes(b)).to_le_bytes()
                            }, |out, a, b| simd::apply(out, a, b, Operation::$variant, false));
                        } else {
                            integer_pixels::<{Operation::$variant as usize}>(&mut output, [a, b], dimensions, reverse);
                        }
                    }),+
                }
            }
        }
        dispatch!(Abs, Neg, Invert, Add, Sub, Mul, Div, Mod, Pow, And, Or, Xor, Lshift, Rshift, Eq, Ne, Lt, Le, Gt, Ge, Min, Max);
    });
    Image::from_integer_bytes(width as u32, height as u32, PixelMode::I, output)
}
