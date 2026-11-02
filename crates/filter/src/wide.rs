use std::simd::Simd;
use std::simd::cmp::SimdOrd;
use std::simd::num::{SimdFloat, SimdInt};

use blanket_core::{Image, PixelMode};
use pyo3::exceptions::PyMemoryError;
use pyo3::prelude::*;

pub(crate) fn kernel(image: &Image, size: usize, scale: f32, offset: f32, weights: &[f32]) -> PyResult<Image> {
    let source: Vec<i32> = image.raw_data()?.as_chunks::<4>().0.iter().map(|&bytes| i32::from_le_bytes(bytes)).collect();
    let mut values = source.clone();
    let (width, height) = (image.width as usize, image.height as usize);
    convolve(&source, &mut values, size, (width, height), scale, offset, weights);
    Image::from_integer_bytes(image.width, image.height, image.mode, values.into_iter().flat_map(i32::to_le_bytes).collect())
}

fn convolve(source: &[i32], values: &mut [i32], size: usize, dimensions: (usize, usize), scale: f32, offset: f32, weights: &[f32]) {
    let (width, height) = dimensions;
    if width >= size && height >= size {
        let radius = size / 2;
        let weights: Vec<f32> = weights.iter().map(|weight| weight / scale).collect();
        for y in radius..height - radius {
            let first = y * width + radius;
            let last = (y + 1) * width - radius;
            let end = first + (last - first) / 16 * 16;
            for i in (first..end).step_by(16) {
                let mut sum = Simd::splat(offset + 0.5);
                for ky in 0..size {
                    let start = i + radius * width - ky * width - radius;
                    let at = |kx| Simd::<i32, 16>::from_slice(&source[start + kx..]).cast::<f32>();
                    let mut row = super::filter::kernel_add_simd(at(0), Simd::splat(weights[ky * size]), at(1) * Simd::splat(weights[ky * size + 1]));
                    for kx in 2..size {
                        row = super::filter::kernel_add_simd(at(kx), Simd::splat(weights[ky * size + kx]), row);
                    }
                    sum += row;
                }
                sum.cast::<i32>().simd_max(Simd::splat(0)).copy_to_slice(&mut values[i..i + 16]);
            }
            for (relative, value) in values[end..last].iter_mut().enumerate() {
                let i = end + relative;
                let mut sum = offset + 0.5;
                for ky in 0..size {
                    let start = i + radius * width - ky * width - radius;
                    let mut row =
                        super::filter::kernel_add(source[start] as f32, weights[ky * size], source[start + 1] as f32 * weights[ky * size + 1]);
                    for kx in 2..size {
                        row = super::filter::kernel_add(source[start + kx] as f32, weights[ky * size + kx], row);
                    }
                    sum += row;
                }
                *value = (sum as i32).max(0);
            }
        }
    }
}

pub(crate) fn rank(image: &Image, size: usize, rank: usize) -> PyResult<Image> {
    let bytes = image.raw_data()?.as_chunks::<4>().0;
    let pixels = if image.mode == PixelMode::I {
        let source: Vec<i32> = bytes.iter().map(|&value| i32::from_le_bytes(value)).collect();
        rank_values(&source, image.width as usize, image.height as usize, size, rank)?.into_iter().flat_map(i32::to_le_bytes).collect()
    } else {
        let source: Vec<f32> = bytes.iter().map(|&value| f32::from_le_bytes(value)).collect();
        rank_values(&source, image.width as usize, image.height as usize, size, rank)?.into_iter().flat_map(f32::to_le_bytes).collect()
    };
    blanket_core::buffers::output(image, (image.width, image.height), pixels)
}

fn rank_values<T: Copy + PartialOrd>(source: &[T], width: usize, height: usize, size: usize, rank: usize) -> PyResult<Vec<T>> {
    let mut output = source.to_vec();
    if output.is_empty() {
        return Ok(output);
    }
    let mut window = Vec::new();
    window.try_reserve_exact(size * size).map_err(|_| PyMemoryError::new_err("cannot allocate rank window"))?;
    let radius = size / 2;
    for y in 0..height {
        for x in 0..width {
            window.clear();
            for ky in 0..size {
                let yy = (y + ky).saturating_sub(radius).min(height - 1);
                for kx in 0..size {
                    let xx = (x + kx).saturating_sub(radius).min(width - 1);
                    window.push(source[yy * width + xx]);
                }
            }
            output[y * width + x] = select(&mut window, rank);
        }
    }
    Ok(output)
}

fn select<T: Copy + PartialOrd>(values: &mut [T], rank: usize) -> T {
    let (mut left, mut right) = (0_isize, values.len() as isize - 1);
    while left < right {
        let pivot = values[rank];
        let (mut i, mut j) = (left, right);
        loop {
            while values[i as usize] < pivot {
                i += 1;
            }
            while pivot < values[j as usize] {
                j -= 1;
            }
            if i <= j {
                values.swap(i as usize, j as usize);
                i += 1;
                j -= 1;
            }
            if i > j {
                break;
            }
        }
        if j < rank as isize {
            left = i;
        }
        if (rank as isize) < i {
            right = j;
        }
    }
    values[rank]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integer_convolution_covers_unaligned_vectors_tails_borders_and_guards() {
        for size in [3, 5] {
            for width in 0..40 {
                for height in [0, 1, 2, 3, 5, 7] {
                    for offset in 0..4 {
                        let count = width * height;
                        let storage: Vec<i32> = (0..offset + count).map(|i| (i * 37 % 511) as i32 - 255).collect();
                        let source = &storage[offset..];
                        for nonfinite in [false, true] {
                            let weights: Vec<f32> = (0..size * size).map(|i| if nonfinite { f32::NAN } else { (i % 5) as f32 - 2.0 }).collect();
                            let mut expected = source.to_vec();
                            if width >= size && height >= size {
                                let radius = size / 2;
                                for y in radius..height - radius {
                                    for x in radius..width - radius {
                                        let mut sum = 3.0;
                                        for ky in 0..size {
                                            for kx in 0..size {
                                                sum += source[(y + radius - ky) * width + x + kx - radius] as f32 * weights[ky * size + kx];
                                            }
                                        }
                                        expected[y * width + x] = (sum as i32).max(0);
                                    }
                                }
                            }
                            let mut output = vec![0x5a5a; offset + count + 13];
                            output[offset..offset + count].copy_from_slice(source);
                            convolve(source, &mut output[offset..offset + count], size, (width, height), 1.0, 2.5, &weights);
                            assert_eq!(&output[offset..offset + count], expected, "width={width}, height={height}, size={size}");
                            assert!(output[..offset].iter().chain(&output[offset + count..]).all(|&value| value == 0x5a5a));
                        }
                    }
                }
            }
        }
    }
}
