//! Backend-independent GPU image descriptors and operation preparation.

pub mod plan;
pub mod protocol;

use protocol::{PARAMETER_COUNT, resampling, weights};

// Match Blanket's CPU decoded-image pixel ceiling.
const MAX_PIXELS: u64 = 178_956_970;
// Packed grayscale through RGBA images have at most four channels.
const MAX_CHANNELS: u32 = 4;
// Filter radii are measured in source pixels before downsampling expansion.
const LANCZOS_RADIUS: f64 = 3.0;
const BICUBIC_RADIUS: f64 = 2.0;
const BOX_RADIUS: f64 = 0.5;
const LINEAR_RADIUS: f64 = 1.0;
// Retain the CPU Hamming window's single-precision coefficient values.
const HAMMING_ALPHA: f32 = 0.54;
const HAMMING_BETA: f32 = 0.46;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Descriptor {
    pub width: u32,
    pub height: u32,
    pub channels: u32,
    pub float: bool,
}

impl Descriptor {
    pub fn byte_len(self) -> Result<usize, String> {
        let pixels = u64::from(self.width) * u64::from(self.height);
        if pixels == 0 || pixels > MAX_PIXELS || !(1..=MAX_CHANNELS).contains(&self.channels) {
            return Err("invalid GPU image dimensions or channel count".into());
        }
        let sample_bytes = if self.float { size_of::<f32>() } else { size_of::<u8>() };
        usize::try_from(pixels * u64::from(self.channels) * sample_bytes as u64).map_err(|_| "GPU image allocation overflow".into())
    }
}

/// All arrays are small CPU control data, never image pixels.
pub struct Dispatch {
    pub output: Descriptor,
    pub parameters: [u32; PARAMETER_COUNT],
    pub values: Vec<f32>,
}

pub trait Backend {
    type Image;
    fn upload(&self, descriptor: Descriptor, bytes: &[u8]) -> Result<Self::Image, String>;
    /// # Safety
    /// Validate shader indices, control lengths, sample layouts, and device compatibility.
    /// Sources must not be modified concurrently with GPU execution.
    unsafe fn dispatch(&self, source: &Self::Image, other: &Self::Image, request: &Dispatch) -> Result<Self::Image, String>;
    fn download(&self, image: &Self::Image) -> Result<Vec<u8>, String>;
    fn synchronize(&self, image: &Self::Image) -> Result<(), String>;
}

pub fn blur_parameters(radius: f32, gaussian: bool) -> Result<(u32, [f32; 2]), String> {
    if !radius.is_finite() || radius < 0.0 {
        return Err("radius must be finite and nonnegative".into());
    }
    let radius = if gaussian {
        let variance = radius * radius / protocol::GAUSSIAN_PASSES as f32;
        // Three fractional box passes approximate the requested Gaussian variance.
        let length = (12.0 * f64::from(variance) + 1.0).sqrt() as f32;
        let integer = ((f64::from(length) - 1.0) / 2.0).floor() as f32;
        let mut fraction = (2.0 * integer + 1.0) * (integer * (integer + 1.0) - 3.0 * variance);
        fraction /= 6.0 * (variance - (integer + 1.0) * (integer + 1.0));
        integer + fraction
    } else {
        radius
    };
    if !radius.is_finite() || radius < 0.0 || radius >= i32::MAX as f32 {
        return Err("blur radius is too large".into());
    }
    let integer = radius as u32;
    let weight = (protocol::BOX_WEIGHT_SCALE as f32 / (radius * 2.0 + 1.0)) as u32;
    let far_weight = protocol::BOX_WEIGHT_SCALE.wrapping_sub((2 * integer + 1).wrapping_mul(weight)) / 2;
    Ok((integer, [weight as f32, far_weight as f32]))
}

pub fn lut_controls(size: [u32; 3], channels: u32, table: &[f32], floating: bool) -> Result<Vec<f32>, String> {
    use protocol::lut;
    if !(3..=4).contains(&channels) || size.iter().any(|&v| !(lut::MIN_SIZE..=lut::MAX_SIZE).contains(&(v as usize))) {
        return Err("3D LUT requires 3 or 4 channels and dimensions from 2 to 65".into());
    }
    let expected = size.iter().fold(channels as usize, |count, &v| count * v as usize);
    if table.len() != expected {
        return Err("3D LUT table must have the expected number of finite samples".into());
    }
    let mut values = Vec::with_capacity(lut::HEADER_LEN + table.len());
    values.extend(size.map(|v| v as f32));
    values.extend(size.map(|v| ((f64::from(v - 1) / f64::from(u8::MAX) * f64::from(1 << lut::POSITION_BITS)) as u32) as f32));
    if floating {
        if table.iter().any(|v| !v.is_finite()) {
            return Err("3D LUT table must have the expected number of finite samples".into());
        }
        values.extend_from_slice(table);
    } else {
        for &value in table {
            if !value.is_finite() {
                return Err("3D LUT table must have the expected number of finite samples".into());
            }
            values.push(((value * lut::SAMPLE_SCALE).round() as i32).clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as f32);
        }
    }
    Ok(values)
}

pub fn resize_weights(input: u32, output: u32, method: u8) -> Result<(u32, Vec<f32>), String> {
    if input == 0 || output == 0 || method > resampling::HAMMING {
        return Err("invalid resize arguments".into());
    }
    let scale = f64::from(input) / f64::from(output);
    let support = match method {
        resampling::NEAREST => 0.0,
        resampling::LANCZOS => LANCZOS_RADIUS,
        resampling::BICUBIC => BICUBIC_RADIUS,
        resampling::BOX => BOX_RADIUS,
        _ => LINEAR_RADIUS,
    } * scale.max(1.0);
    // Include the row header and one extra tap for rounded support endpoints.
    let stride = ((2.0 * support).ceil() as usize + weights::HEADER_LEN + 1).min(input as usize + weights::HEADER_LEN);
    let mut result = vec![0.0; stride.checked_mul(output as usize).ok_or("resize weights overflow")?];
    if method == resampling::NEAREST {
        for (i, row) in result.chunks_exact_mut(stride).enumerate() {
            let center = (i as f64 + 0.5) * scale;
            row[weights::START] = (center.floor() as u32).min(input - 1) as f32;
            row[weights::COUNT] = 1.0;
            row[weights::HEADER_LEN] = 1.0;
        }
        return Ok((stride as u32, result));
    }
    let inverse_scale = 1.0 / scale.max(1.0);
    let mut raw = Vec::with_capacity(stride - weights::HEADER_LEN);
    for (i, row) in result.chunks_exact_mut(stride).enumerate() {
        let center = (i as f64 + 0.5) * scale;
        let left = ((center - support + 0.5) as i64).clamp(0, i64::from(input)) as usize;
        let right = ((center + support + 0.5) as i64).clamp(0, i64::from(input)) as usize;
        row[weights::START] = left as f32;
        row[weights::COUNT] = (right - left) as f32;
        raw.clear();
        raw.extend((left..right).map(|x| kernel((x as f64 - center + 0.5) * inverse_scale, method)));
        let sum: f64 = raw.iter().sum();
        for (dst, &value) in row[weights::HEADER_LEN..weights::HEADER_LEN + right - left].iter_mut().zip(&raw) {
            *dst = if sum == 0.0 { 0.0 } else { ((value / sum * weights::SCALE).round() / weights::SCALE) as f32 };
        }
    }
    Ok((stride as u32, result))
}

fn kernel(x: f64, method: u8) -> f64 {
    if method == resampling::BOX {
        return if x > -BOX_RADIUS && x <= BOX_RADIUS { 1.0 } else { 0.0 };
    }
    let x = x.abs();
    match method {
        resampling::BILINEAR => (LINEAR_RADIUS - x).max(0.0),
        // Keys cubic convolution with a = -0.5, in the CPU evaluation order.
        resampling::BICUBIC if x < 1.0 => ((1.5 * x - 2.5) * x) * x + 1.0,
        resampling::BICUBIC if x < BICUBIC_RADIUS => (((x - 5.0) * x + 8.0) * x - 4.0) * -0.5,
        resampling::LANCZOS | resampling::HAMMING if x == 0.0 => 1.0,
        resampling::LANCZOS if x < LANCZOS_RADIUS => {
            let p = x * std::f64::consts::PI;
            p.sin() / p * (p / LANCZOS_RADIUS).sin() / (p / LANCZOS_RADIUS)
        }
        resampling::HAMMING if x < LINEAR_RADIUS => {
            let p = x * std::f64::consts::PI;
            p.sin() / p * (f64::from(HAMMING_ALPHA) + f64::from(HAMMING_BETA) * p.cos())
        }
        _ => 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn validates_allocations() {
        assert!(Descriptor { width: u32::MAX, height: u32::MAX, channels: 4, float: true }.byte_len().is_err());
        assert_eq!(Descriptor { width: 3, height: 2, channels: 4, float: true }.byte_len().unwrap(), 96);
    }
    #[test]
    fn weights_preserve_constants() {
        for method in resampling::NEAREST..=resampling::HAMMING {
            for (input, output) in [(1, 17), (17, 1), (31, 19)] {
                let (stride, weights) = resize_weights(input, output, method).unwrap();
                for row in weights.chunks_exact(stride as usize) {
                    assert!((row[weights::HEADER_LEN..weights::HEADER_LEN + row[weights::COUNT] as usize].iter().sum::<f32>() - 1.0).abs() < 1e-6);
                }
            }
        }
    }
    #[test]
    fn resize_rows_keep_boundaries_and_padding() {
        assert_eq!(resize_weights(2, 3, resampling::BILINEAR).unwrap(), (4, vec![0.0, 1.0, 1.0, 0.0, 0.0, 2.0, 0.5, 0.5, 1.0, 1.0, 1.0, 0.0]));
        assert_eq!(resize_weights(2, 3, resampling::NEAREST).unwrap(), (3, vec![0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]));
        for method in resampling::NEAREST..=resampling::HAMMING {
            let (stride, rows) = resize_weights(1, 31, method).unwrap();
            assert_eq!(stride, 3);
            assert!(rows.as_chunks::<3>().0.iter().all(|row| *row == [0.0, 1.0, 1.0]));
        }
    }
    #[test]
    fn blur_controls_match_fractional_box_weights() {
        assert_eq!(blur_parameters(0.0, false).unwrap(), (0, [protocol::BOX_WEIGHT_SCALE as f32, 0.0]));
        let (radius, weights) = blur_parameters(1.5, false).unwrap();
        assert_eq!(radius, 1);
        assert_eq!(weights, [4_194_304.0, 2_097_152.0]);
        for invalid in [-1.0, f32::INFINITY, f32::NAN, f32::MAX] {
            assert!(blur_parameters(invalid, true).is_err());
        }
    }
    #[test]
    fn lut_controls_preserve_cpu_quantization_and_float_samples() {
        let table = vec![0.5; 3 * 2 * 3 * 4];
        let values = lut_controls([2, 3, 4], 3, &table, false).unwrap();
        assert_eq!(&values[..3], &[2.0, 3.0, 4.0]);
        assert_eq!(&values[protocol::lut::HEADER_LEN..], vec![8160.0; table.len()]);
        assert_eq!(&lut_controls([2, 3, 4], 3, &table, true).unwrap()[protocol::lut::HEADER_LEN..], table);
        assert!(lut_controls([1, 3, 4], 3, &table, true).is_err());
        assert!(lut_controls([2, 3, 4], 4, &table, true).is_err());
        assert!(lut_controls([2, 3, 4], 3, &vec![f32::NAN; table.len()], true).is_err());
    }
    #[test]
    fn lut_controls_handle_extremes_and_late_invalid_samples() {
        use protocol::lut;
        let samples = [-f32::MAX, -0.5, -0.0, 0.5, 1.0, f32::MAX];
        let table: Vec<f32> = samples.into_iter().cycle().take(24).collect();
        let values = lut_controls([2, 2, 2], 3, &table, false).unwrap();
        for chunk in values[lut::HEADER_LEN..].chunks_exact(samples.len()) {
            assert_eq!(chunk, [-32768.0, -8160.0, 0.0, 8160.0, 16320.0, 32767.0]);
        }
        let values = lut_controls([2, 2, 2], 3, &table, true).unwrap();
        assert!(values[lut::HEADER_LEN..].iter().zip(&table).all(|(a, b)| a.to_bits() == b.to_bits()));
        for floating in [false, true] {
            for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                let mut invalid_table = table.clone();
                *invalid_table.last_mut().unwrap() = invalid;
                assert!(lut_controls([2, 2, 2], 3, &invalid_table, floating).is_err());
            }
        }
    }
}
