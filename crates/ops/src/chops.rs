//! Channel arithmetic and wraparound translation for ImageChops.

use pyo3::exceptions::{PyMemoryError, PyValueError};
use pyo3::prelude::*;

use blanket_core::parallel::{CHUNK_PIXELS, chunks_mut, chunks_mut_above};
use blanket_core::raster::{Image, PixelMode};

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(chops_binary, module)?)?;
    module.add_function(wrap_pyfunction!(chops_invert, module)?)?;
    module.add_function(wrap_pyfunction!(chops_offset, module)?)?;
    Ok(())
}

fn buffer(len: usize) -> PyResult<Vec<u8>> {
    let mut pixels = reserved_buffer(len)?;
    pixels.resize(len, 0);
    Ok(pixels)
}

fn reserved_buffer(len: usize) -> PyResult<Vec<u8>> {
    let mut pixels = Vec::new();
    pixels
        .try_reserve_exact(len)
        .map_err(|_| PyMemoryError::new_err("cannot allocate image"))?;
    Ok(pixels)
}

#[derive(Clone, Copy)]
enum Operation {
    Difference,
    Multiply,
    Screen,
    Lighter,
    Darker,
    Add,
    Subtract,
    AddModulo,
    SubtractModulo,
    SoftLight,
    HardLight,
    Overlay,
    And,
    Or,
    Xor,
}

impl Operation {
    fn clip_scaled(value: f32) -> u8 {
        let temp = if !value.is_finite() || value > i32::MAX as f32 || value < i32::MIN as f32 {
            i32::MIN
        } else {
            value as i32
        };
        if temp <= 0 {
            0
        } else if temp >= 255 {
            255
        } else {
            temp as u8
        }
    }

    fn parse(name: &str) -> PyResult<Self> {
        Ok(match name {
            "difference" => Self::Difference,
            "multiply" => Self::Multiply,
            "screen" => Self::Screen,
            "lighter" => Self::Lighter,
            "darker" => Self::Darker,
            "add" => Self::Add,
            "subtract" => Self::Subtract,
            "add_modulo" => Self::AddModulo,
            "subtract_modulo" => Self::SubtractModulo,
            "soft_light" => Self::SoftLight,
            "hard_light" => Self::HardLight,
            "overlay" => Self::Overlay,
            "and" => Self::And,
            "or" => Self::Or,
            "xor" => Self::Xor,
            _ => return Err(PyValueError::new_err("unknown channel operation")),
        })
    }

    fn apply(self, a: u8, b: u8, scale: f32, offset: i32) -> u8 {
        let x = u32::from(a);
        let y = u32::from(b);
        match self {
            Self::Difference => a.abs_diff(b),
            Self::Multiply => (x * y / 255) as u8,
            Self::Screen => (255 - (255 - x) * (255 - y) / 255) as u8,
            Self::Lighter => a.max(b),
            Self::Darker => a.min(b),
            Self::Add => Self::clip_scaled((x + y) as f32 / scale + offset as f32),
            Self::Subtract => Self::clip_scaled((i32::from(a) - i32::from(b)) as f32 / scale + offset as f32),
            Self::AddModulo => a.wrapping_add(b),
            Self::SubtractModulo => a.wrapping_sub(b),
            Self::SoftLight => (((255 - x) * x * y / 65536) + x * (255 - (255 - x) * (255 - y) / 255) / 255) as u8,
            Self::HardLight | Self::Overlay => {
                let selector = if matches!(self, Self::HardLight) { y } else { x };
                if selector < 128 {
                    (x * y / 127) as u8
                } else {
                    (255 - (255 - x) * (255 - y) / 127) as u8
                }
            }
            Self::And => a & b,
            Self::Or => a | b,
            Self::Xor => a ^ b,
        }
    }
}

fn bitwise_chunk<const OP: u8>(a: &[u8], b: &[u8], output: &mut [u8]) {
    let (words, tail) = output.as_chunks_mut::<8>();
    for ((dst, left), right) in words.iter_mut().zip(a.as_chunks::<8>().0).zip(b.as_chunks::<8>().0) {
        let left = u64::from_ne_bytes(*left);
        let right = u64::from_ne_bytes(*right);
        *dst = match OP {
            0 => left & right,
            1 => left | right,
            _ => left ^ right,
        }
        .to_ne_bytes();
    }
    for ((dst, &left), &right) in tail.iter_mut().zip(&a[words.len() * 8..]).zip(&b[words.len() * 8..]) {
        *dst = match OP {
            0 => left & right,
            1 => left | right,
            _ => left ^ right,
        };
    }
}

#[pyfunction]
#[pyo3(signature = (first, second, operation, scale=1.0, offset=0))]
fn chops_binary(py: Python<'_>, first: &Image, second: &Image, operation: &str, scale: f32, offset: i32) -> PyResult<Image> {
    let operation = Operation::parse(operation)?;
    let a = first.pixel_data()?;
    let b = second.pixel_data()?;
    if first.mode != second.mode {
        return Err(PyValueError::new_err("images do not match"));
    }
    // Like Pillow, differently sized inputs use their top-left intersection.
    let width = first.width.min(second.width);
    let height = first.height.min(second.height);
    let channels = first.mode.channels();
    let row = width as usize * channels;
    let length = row * height as usize;
    if matches!(operation, Operation::And | Operation::Or | Operation::Xor) && first.width == second.width {
        let mut pixels = buffer(length)?;
        py.detach(|| {
            chunks_mut_above(&mut pixels, 256 * 1024, 512 * 1024, |chunk, dst| {
                let start = chunk * 256 * 1024;
                let a = &a[start..start + dst.len()];
                let b = &b[start..start + dst.len()];
                match operation {
                    Operation::And => bitwise_chunk::<0>(a, b, dst),
                    Operation::Or => bitwise_chunk::<1>(a, b, dst),
                    Operation::Xor => bitwise_chunk::<2>(a, b, dst),
                    _ => unreachable!(),
                }
            });
        });
        return Image::from_pixels(width, height, first.mode, pixels, None);
    }
    let mut pixels = reserved_buffer(length)?;
    let rows_per_chunk = (CHUNK_PIXELS / (width as usize).max(1)).max(1);
    let threshold = match operation {
        Operation::AddModulo | Operation::SubtractModulo | Operation::Lighter | Operation::Darker | Operation::Difference => 16 * 1024 * 1024,
        _ => blanket_core::parallel::MIN_PARALLEL_BYTES,
    };
    py.detach(|| {
        let output = &mut pixels.spare_capacity_mut()[..length];
        chunks_mut_above(output, row.max(1) * rows_per_chunk, threshold, |chunk, dst| {
            for (i, output_row) in dst.chunks_exact_mut(row).enumerate() {
                let y = chunk * rows_per_chunk + i;
                let a_start = y * first.width as usize * channels;
                let b_start = y * second.width as usize * channels;
                // Dispatch outside the pixel loop so each simple operation
                // can be vectorized without a per-sample enum branch.
                macro_rules! apply {
                    ($f:expr) => {
                        for ((value, &a), &b) in output_row
                            .iter_mut()
                            .zip(&a[a_start..a_start + row])
                            .zip(&b[b_start..b_start + row])
                        {
                            value.write($f(a, b));
                        }
                    };
                }
                match operation {
                    Operation::AddModulo => apply!(u8::wrapping_add),
                    Operation::SubtractModulo => apply!(u8::wrapping_sub),
                    Operation::Lighter => apply!(u8::max),
                    Operation::Darker => apply!(u8::min),
                    Operation::Difference => apply!(u8::abs_diff),
                    Operation::And => apply!(std::ops::BitAnd::bitand),
                    Operation::Or => apply!(std::ops::BitOr::bitor),
                    Operation::Xor => apply!(std::ops::BitXor::bitxor),
                    Operation::Add => apply!(|a, b| Operation::Add.apply(a, b, scale, offset)),
                    Operation::Subtract => apply!(|a, b| Operation::Subtract.apply(a, b, scale, offset)),
                    Operation::Multiply => apply!(|a, b| Operation::Multiply.apply(a, b, scale, offset)),
                    Operation::Screen => apply!(|a, b| Operation::Screen.apply(a, b, scale, offset)),
                    Operation::SoftLight => apply!(|a, b| Operation::SoftLight.apply(a, b, scale, offset)),
                    Operation::HardLight => apply!(|a, b| Operation::HardLight.apply(a, b, scale, offset)),
                    Operation::Overlay => apply!(|a, b| Operation::Overlay.apply(a, b, scale, offset)),
                }
            }
        });
    });
    // Every output sample is initialized by the disjoint row partitions.
    unsafe { pixels.set_len(length) };
    let mut result = Image::from_pixels(width, height, first.mode, pixels, None)?;
    // Channel arithmetic creates an empty output palette, as Pillow does.
    result.palette = first.palette.as_ref().map(|_| (PixelMode::Rgb, Vec::new()));
    Ok(result)
}

#[pyfunction]
fn chops_invert(py: Python<'_>, image: &Image) -> PyResult<Image> {
    let source = image.pixel_data()?;
    let mut pixels = reserved_buffer(source.len())?;
    py.detach(|| {
        chunks_mut_above(
            &mut pixels.spare_capacity_mut()[..source.len()],
            CHUNK_PIXELS,
            16 * 1024 * 1024,
            |chunk, dst| {
                for (value, &sample) in dst.iter_mut().zip(&source[chunk * CHUNK_PIXELS..]) {
                    value.write(255 - sample);
                }
            },
        );
    });
    // The loop initializes exactly source.len() bytes, including the tail.
    unsafe { pixels.set_len(source.len()) };
    let mut result = Image::from_pixels(image.width, image.height, image.mode, pixels, None)?;
    result.palette = image.palette.as_ref().map(|_| (PixelMode::Rgb, Vec::new()));
    Ok(result)
}

#[pyfunction]
fn chops_offset(py: Python<'_>, image: &Image, xoffset: i64, yoffset: i64) -> PyResult<Image> {
    let source = image.raw_data()?;
    let mut pixels = buffer(source.len())?;
    if image.width != 0 && image.height != 0 {
        let bytes_per_pixel = image.mode.channels() * if image.bit_depth == 8 { 1 } else { 2 };
        let row = image.width as usize * bytes_per_pixel;
        let head = xoffset.rem_euclid(i64::from(image.width)) as usize * bytes_per_pixel;
        let shift_y = yoffset.rem_euclid(i64::from(image.height)) as usize;
        py.detach(|| {
            chunks_mut(&mut pixels, row, |y, dst| {
                let source_y = (y + image.height as usize - shift_y) % image.height as usize;
                let src = &source[source_y * row..(source_y + 1) * row];
                dst[..head].copy_from_slice(&src[row - head..]);
                dst[head..].copy_from_slice(&src[..row - head]);
            });
        });
    }
    Ok(Image {
        width: image.width,
        height: image.height,
        mode: image.mode,
        pixels: Some(pixels),
        bit_depth: image.bit_depth,
        format: None,
        palette: image.palette.clone(),
    })
}
