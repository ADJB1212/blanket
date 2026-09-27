//! Scalar image arithmetic for ImageMath.

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
        chunks_mut(&mut output, CHUNK_PIXELS * 4, |start, chunk| {
            for (offset, pixel) in chunk.as_chunks_mut::<4>().0.iter_mut().enumerate() {
                let index = start * CHUNK_PIXELS + offset;
                let x = index % width as usize;
                let y = index / width as usize;
                let ai = (y * left.width as usize + x) * 4;
                let av = a[ai..ai + 4].try_into().unwrap();
                let bv = match (right, b) {
                    (Some(right), Some(b)) => {
                        let bi = (y * right.width as usize + x) * 4;
                        b[bi..bi + 4].try_into().unwrap()
                    }
                    _ => [0; 4],
                };
                let bytes = if float {
                    op.float(f32::from_le_bytes(av), f32::from_le_bytes(bv)).to_le_bytes()
                } else {
                    op.integer(i32::from_le_bytes(av), i32::from_le_bytes(bv)).to_le_bytes()
                };
                pixel.copy_from_slice(&bytes);
            }
        });
    });
    if float && !integer_output {
        Image::from_float_bytes(width, height, output)
    } else {
        Image::from_integer_bytes(width, height, PixelMode::I, output)
    }
}
