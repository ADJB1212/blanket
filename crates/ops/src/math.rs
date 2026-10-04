//! Image arithmetic for ImageMath.

use blanket_core::parallel::{CHUNK_PIXELS, chunks_mut};
use blanket_core::{Image, PixelMode};
use pyo3::exceptions::{PyMemoryError, PyTypeError, PyValueError};
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
        let floats = [
            f32::NEG_INFINITY,
            -3.5,
            -0.0,
            0.0,
            2.5,
            f32::INFINITY,
            f32::NAN,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
        ];
        for name in [
            "abs", "neg", "invert", "add", "sub", "mul", "div", "mod", "pow", "and", "or", "xor", "lshift", "rshift", "eq", "ne", "lt", "le", "gt",
            "ge", "min", "max",
        ] {
            let op = Operation::parse(name).unwrap();
            for float in [false, true] {
                if float && op.integer_only() {
                    continue;
                }
                let values: Vec<[u8; 4]> = if float {
                    floats.map(f32::to_le_bytes).to_vec()
                } else {
                    integers.map(i32::to_le_bytes).to_vec()
                };
                for width in [1, 3, 4, 5, 17, CHUNK_PIXELS + 3] {
                    for cropped in [false, true] {
                        let left_width = width + usize::from(cropped) * 3;
                        let right_width = width + usize::from(cropped) * 5;
                        let a: Vec<u8> = (0..left_width * 3).flat_map(|i| values[i % values.len()]).collect();
                        let b: Vec<u8> = (0..right_width * 3)
                            .flat_map(|i| values[(i / values.len() + i * 2) % values.len()])
                            .collect();
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
                        apply_pixels(&mut actual, &a, b, [width, left_width, right_width], scalar, |out, a, b| {
                            simd::apply(out, a, b, op, float)
                        });
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

fn widen_u8_to_i32(source: &[u8], stride: usize, width: usize, height: usize) -> Vec<u8> {
    let mut out = vec![0u8; width * height * 4];
    for y in 0..height {
        let src_row = &source[y * stride..y * stride + width];
        let dst_row = &mut out[y * width * 4..(y + 1) * width * 4];
        for (dst, &src) in dst_row.as_chunks_mut::<4>().0.iter_mut().zip(src_row) {
            *dst = (src as i32).to_le_bytes();
        }
    }
    out
}

#[pyfunction]
#[pyo3(signature = (operation, left, right=None, integer_output=false))]
fn math_apply(py: Python<'_>, operation: &str, left: &Image, right: Option<&Image>, integer_output: bool) -> PyResult<Image> {
    let op = Operation::parse(operation)?;
    if op.unary() != right.is_none() {
        return Err(PyValueError::new_err("wrong number of image math operands"));
    }
    let left_byte = matches!(left.mode, PixelMode::One | PixelMode::L);
    let right_byte = right.is_some_and(|r| matches!(r.mode, PixelMode::One | PixelMode::L));
    let integer_group = |mode: PixelMode| matches!(mode, PixelMode::One | PixelMode::L | PixelMode::I);
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
    let len = width
        .checked_mul(height)
        .and_then(|n| n.checked_mul(4))
        .ok_or_else(|| PyMemoryError::new_err("image dimensions are too large"))?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(len)
        .map_err(|_| PyMemoryError::new_err("cannot allocate image"))?;
    output.resize(len, 0);
    py.detach(|| {
        let a_widened;
        let b_widened;
        let (a, left_width) = if left_byte {
            a_widened = widen_u8_to_i32(a_raw, left.width as usize, width, height);
            (a_widened.as_slice(), width)
        } else {
            (a_raw, left.width as usize)
        };
        let (b, right_width) = match (b_raw, right) {
            (Some(b_raw), Some(r)) if matches!(r.mode, PixelMode::One | PixelMode::L) => {
                b_widened = widen_u8_to_i32(b_raw, r.width as usize, width, height);
                (Some(b_widened.as_slice()), width)
            }
            (Some(b_raw), Some(r)) => (Some(b_raw), r.width as usize),
            _ => (None, 0),
        };
        macro_rules! dispatch {
            ($($variant:ident),+ $(,)?) => {
                match op {
                    $(Operation::$variant => {
                        if float {
                            apply_pixels(&mut output, a, b, [width, left_width,
                                right_width], |a, b| {
                                    Operation::$variant.float(f32::from_le_bytes(a), f32::from_le_bytes(b)).to_le_bytes()
                                }, |out, a, b| simd::apply(out, a, b, Operation::$variant, true));
                        } else {
                            apply_pixels(&mut output, a, b, [width, left_width,
                                right_width], |a, b| {
                                    Operation::$variant.integer(i32::from_le_bytes(a), i32::from_le_bytes(b)).to_le_bytes()
                                }, |out, a, b| simd::apply(out, a, b, Operation::$variant, false));
                        }
                    }),+
                }
            }
        }
        dispatch!(
            Abs, Neg, Invert, Add, Sub, Mul, Div, Mod, Pow, And, Or, Xor, Lshift, Rshift, Eq, Ne, Lt, Le, Gt, Ge, Min, Max
        );
    });
    if float && !integer_output {
        Image::from_float_bytes(width as u32, height as u32, output)
    } else {
        Image::from_integer_bytes(width as u32, height as u32, PixelMode::I, output)
    }
}
