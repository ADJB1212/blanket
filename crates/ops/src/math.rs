//! Image arithmetic for ImageMath.

use blanket_core::parallel::{CHUNK_PIXELS, chunks_mut};
use blanket_core::{Image, PixelMode};
use pyo3::exceptions::{PyMemoryError, PyTypeError, PyValueError};
use pyo3::prelude::*;

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(math_apply, module)?)?;
    Ok(())
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
    output: &mut [u8], a: &[u8], b: Option<&[u8]>, width: usize, left_width: usize, right_width: usize,
    operation: impl Fn([u8; 4], [u8; 4]) -> [u8; 4] + Sync + Send,
) {
    let a = a.as_chunks::<4>().0;
    let b = b.map(|data| data.as_chunks::<4>().0);
    chunks_mut(output, CHUNK_PIXELS * 4, |start, chunk| {
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
            if let Some(b) = b {
                for ((out, &a), &b) in span.iter_mut().zip(left).zip(&b[bi..bi + len]) {
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

#[pyfunction]
#[pyo3(signature = (operation, left, right=None, integer_output=false))]
fn math_apply(py: Python<'_>, operation: &str, left: &Image, right: Option<&Image>, integer_output: bool) -> PyResult<Image> {
    let op = Operation::parse(operation)?;
    if op.unary() != right.is_none() {
        return Err(PyValueError::new_err("wrong number of image math operands"));
    }
    if !matches!(left.mode, PixelMode::I | PixelMode::F) || right.is_some_and(|r| r.mode != left.mode) {
        return Err(PyValueError::new_err("image math requires matching I or F operands"));
    }
    let float = left.mode == PixelMode::F;
    if float && op.integer_only() {
        return Err(PyTypeError::new_err(format!("bad operand type for '{operation}'")));
    }
    let a = left.raw_data()?;
    let b = right.map(Image::raw_data).transpose()?;
    let width = right.map_or(left.width, |r| r.width.min(left.width));
    let height = right.map_or(left.height, |r| r.height.min(left.height));
    let len = (width as usize)
        .checked_mul(height as usize)
        .and_then(|n| n.checked_mul(4))
        .ok_or_else(|| PyMemoryError::new_err("image dimensions are too large"))?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(len)
        .map_err(|_| PyMemoryError::new_err("cannot allocate image"))?;
    output.resize(len, 0);
    py.detach(|| {
        macro_rules! dispatch {
            ($($variant:ident),+ $(,)?) => {
                match op {
                    $(Operation::$variant => {
                        if float {
                            apply_pixels(&mut output, a, b, width as usize, left.width as usize,
                                right.map_or(0, |r| r.width as usize), |a, b| {
                                    Operation::$variant.float(f32::from_le_bytes(a), f32::from_le_bytes(b)).to_le_bytes()
                                });
                        } else {
                            apply_pixels(&mut output, a, b, width as usize, left.width as usize,
                                right.map_or(0, |r| r.width as usize), |a, b| {
                                    Operation::$variant.integer(i32::from_le_bytes(a), i32::from_le_bytes(b)).to_le_bytes()
                                });
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
        Image::from_float_bytes(width, height, output)
    } else {
        Image::from_integer_bytes(width, height, PixelMode::I, output)
    }
}
