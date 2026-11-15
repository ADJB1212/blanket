use blanket_core::buffers::{buffer, output};
use blanket_core::{Image, PixelMode};
use blanket_kernels::parallel::chunks_mut;
use blanket_resample_kernels::remap::{Accumulator, lens_map};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[derive(Clone, Copy)]
enum Border {
    Constant,
    Clamp,
    Reflect,
    Wrap,
}

impl Border {
    fn index(self, coordinate: i64, length: usize) -> Option<usize> {
        let length = length as i64;
        match self {
            Self::Constant => (0..length).contains(&coordinate).then_some(coordinate as usize),
            Self::Clamp => Some(coordinate.clamp(0, length - 1) as usize),
            Self::Wrap => Some(coordinate.rem_euclid(length) as usize),
            Self::Reflect => {
                let value = coordinate.rem_euclid(2 * length);
                Some(if value < length { value } else { 2 * length - 1 - value } as usize)
            }
        }
    }

    fn coordinate(self, value: f64, length: usize) -> f64 {
        match self {
            Self::Constant | Self::Clamp => value.clamp(-3.0, length as f64 + 3.0),
            Self::Wrap => value.rem_euclid(length as f64),
            Self::Reflect => value.rem_euclid(2.0 * length as f64),
        }
    }
}

struct Sampler<'a> {
    source: &'a [u8],
    width: usize,
    height: usize,
    channels: usize,
    sample_bytes: usize,
    mode: PixelMode,
    maximum: f64,
    border: Border,
    fill: &'a [f64],
}

impl Sampler<'_> {
    fn read(&self, pixel: &[u8], channel: usize) -> f64 {
        let value = &pixel[channel * self.sample_bytes..(channel + 1) * self.sample_bytes];
        if self.mode.is_float() {
            f64::from(f32::from_le_bytes(value.try_into().unwrap()))
        } else if self.sample_bytes == 1 {
            f64::from(value[0])
        } else if self.mode.is_big_endian() {
            f64::from(u16::from_be_bytes(value.try_into().unwrap()))
        } else {
            f64::from(u16::from_le_bytes(value.try_into().unwrap()))
        }
    }

    fn pixel(&self, x: i64, y: i64) -> Option<&[u8]> {
        let x = self.border.index(x, self.width)?;
        let y = self.border.index(y, self.height)?;
        let offset = (y * self.width + x) * self.channels * self.sample_bytes;
        Some(&self.source[offset..offset + self.channels * self.sample_bytes])
    }

    fn write(&self, values: &[f64], destination: &mut [u8]) {
        for (&value, sample) in values.iter().zip(destination.chunks_exact_mut(self.sample_bytes)) {
            if self.mode.is_float() {
                sample.copy_from_slice(&(value as f32).to_le_bytes());
            } else if self.sample_bytes == 1 {
                sample[0] = value.round().clamp(0.0, self.maximum) as u8;
            } else {
                let value = value.round().clamp(0.0, self.maximum) as u16;
                sample.copy_from_slice(&if self.mode.is_big_endian() { value.to_be_bytes() } else { value.to_le_bytes() });
            }
        }
    }

    fn sample(&self, x: f64, y: f64, method: u8, premultiply: bool, destination: &mut [u8]) {
        if !x.is_finite() || !y.is_finite() {
            self.write(self.fill, destination);
            return;
        }
        let x = self.border.coordinate(x, self.width);
        let y = self.border.coordinate(y, self.height);
        let (ix, iy) = (x.floor() as i64, y.floor() as i64);
        if method == 0 {
            if let Some(pixel) = self.pixel(ix, iy) {
                destination.copy_from_slice(pixel);
            } else {
                self.write(self.fill, destination);
            }
            return;
        }
        let (x, y) = (x - 0.5, y - 0.5);
        let (ix, iy) = (x.floor() as i64, y.floor() as i64);
        let (tx, ty) = (x - x.floor(), y - y.floor());
        // Preserve exact samples, including hidden color and float bit patterns.
        if tx == 0.0 && ty == 0.0 {
            if let Some(pixel) = self.pixel(ix, iy) {
                destination.copy_from_slice(pixel);
            } else {
                self.write(self.fill, destination);
            }
            return;
        }
        let alpha = self.channels - 1;
        let premultiply = premultiply && matches!(self.mode, PixelMode::La | PixelMode::Rgba | PixelMode::RgbaF);
        let mut accumulator = Accumulator::new(premultiply.then_some(alpha), self.maximum);
        let offsets = if method == 2 { 0..2 } else { -1..3 };
        for dy in offsets.clone() {
            for dx in offsets.clone() {
                let weight = if method == 2 {
                    (if dx == 0 { 1.0 - tx } else { tx }) * (if dy == 0 { 1.0 - ty } else { ty })
                } else {
                    super::kernel(tx - f64::from(dx), 3) * super::kernel(ty - f64::from(dy), 3)
                };
                if weight == 0.0 {
                    continue;
                }
                let pixel = self.pixel(ix + i64::from(dx), iy + i64::from(dy));
                let get = |channel| pixel.map_or(self.fill[channel], |pixel| self.read(pixel, channel));
                let samples = std::array::from_fn(|channel| if channel < self.channels { get(channel) } else { 0.0 });
                accumulator.add(samples, weight);
            }
        }
        let values = accumulator.finish();
        self.write(&values[..self.channels], destination);
    }
}

#[pyfunction]
pub(crate) fn ops_remap(
    py: Python<'_>, image: &Image, coordinates: &Image, method: u8, border: &str, fill: Vec<f64>, premultiply_alpha: bool,
) -> PyResult<Image> {
    let mode = image.pixel_mode();
    if image.palette().is_some()
        || !matches!(
            mode,
            PixelMode::L
                | PixelMode::La
                | PixelMode::Rgb
                | PixelMode::Rgba
                | PixelMode::F
                | PixelMode::RgbF
                | PixelMode::RgbaF
                | PixelMode::I16
                | PixelMode::I16L
                | PixelMode::I16B
                | PixelMode::I16N
        )
    {
        return Err(PyValueError::new_err("remap requires L, LA, RGB, RGBA, F, RGB;F, RGBA;F, or I;16 pixels; convert explicitly"));
    }
    if coordinates.pixel_mode() != PixelMode::RgbF {
        return Err(PyValueError::new_err("coordinate maps require RGB;F mode"));
    }
    if image.width() == 0 || image.height() == 0 {
        return Err(PyValueError::new_err("remap requires positive source dimensions"));
    }
    if ![0, 2, 3].contains(&method) {
        return Err(PyValueError::new_err("remap supports NEAREST, BILINEAR, and BICUBIC"));
    }
    let border = match border {
        "constant" => Border::Constant,
        "clamp" => Border::Clamp,
        "reflect" => Border::Reflect,
        "wrap" => Border::Wrap,
        _ => return Err(PyValueError::new_err("border must be constant, clamp, reflect, or wrap")),
    };
    let channels = mode.channels();
    let maximum = if mode.is_float() { 1.0 } else { f64::from((1_u32 << image.bit_depth()) - 1) };
    if fill.len() != channels || fill.iter().any(|v| !v.is_finite() || (!mode.is_float() && !(0.0..=maximum).contains(v))) {
        return Err(PyValueError::new_err("fillcolor must contain finite samples within the image depth range"));
    }
    let stride = image.raw_bytes_per_pixel();
    let sampler = Sampler {
        source: image.raw_data()?,
        width: image.width() as usize,
        height: image.height() as usize,
        channels,
        sample_bytes: stride / channels,
        mode,
        maximum,
        border,
        fill: &fill,
    };
    let map = coordinates.raw_data()?;
    let mut pixels = buffer(coordinates.size(), stride)?;
    py.detach(|| {
        chunks_mut(&mut pixels, stride * 4096, |chunk, destination| {
            for (i, pixel) in destination.chunks_exact_mut(stride).enumerate() {
                let offset = (chunk * 4096 + i) * 12;
                let x = f32::from_le_bytes(map[offset..offset + 4].try_into().unwrap());
                let y = f32::from_le_bytes(map[offset + 4..offset + 8].try_into().unwrap());
                sampler.sample(f64::from(x), f64::from(y), method, premultiply_alpha, pixel);
            }
        });
        output(image, coordinates.size(), pixels)
    })
}

#[pyfunction]
pub(crate) fn ops_lens_map(py: Python<'_>, size: (u32, u32), coefficients: [f64; 3], center: (f64, f64), focal: (f64, f64)) -> PyResult<Image> {
    if size.0 == 0
        || size.1 == 0
        || coefficients.iter().chain([&center.0, &center.1, &focal.0, &focal.1]).any(|v| !v.is_finite())
        || focal.0 <= 0.0
        || focal.1 <= 0.0
    {
        return Err(PyValueError::new_err("lens maps require positive dimensions and focal lengths, and finite parameters"));
    }
    let mut pixels = buffer(size, 12)?;
    py.detach(|| {
        chunks_mut(&mut pixels, 12 * 4096, |chunk, destination| {
            lens_map(destination, size.0 as usize, chunk * 4096, coefficients, center, focal);
        });
        Image::from_color_float_bytes(size.0, size.1, PixelMode::RgbF, pixels)
    })
}
