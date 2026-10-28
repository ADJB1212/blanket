//! Perceptual hashes of small, grayscale thumbnails.

use std::simd::Simd;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(image_hash, module)?)?;
    Ok(())
}

fn pack(bits: impl ExactSizeIterator<Item = bool>) -> Vec<u8> {
    let count = bits.len();
    let mut result = vec![0; count.div_ceil(8)];
    for (i, bit) in bits.enumerate() {
        let position = count - 1 - i;
        if bit {
            let index = result.len() - 1 - position / 8;
            result[index] |= 1 << (position % 8);
        }
    }
    result
}

fn phash(pixels: &[u8], size: usize, side: usize) -> Vec<u8> {
    phash_impl::<true>(pixels, size, side)
}

fn phash_impl<const VECTOR: bool>(pixels: &[u8], size: usize, side: usize) -> Vec<u8> {
    let mean = pixels.iter().map(|&v| f64::from(v)).sum::<f64>() / pixels.len() as f64;
    let cosines: Vec<f64> =
        (0..size).flat_map(|k| (0..side).map(move |x| (std::f64::consts::PI * k as f64 * (2 * x + 1) as f64 / (2 * side) as f64).cos())).collect();
    let mut rows = vec![0.0; side * size];
    for (y, row) in pixels.chunks_exact(side).enumerate() {
        let end = if VECTOR { size / 4 * 4 } else { 0 };
        for k in (0..end).step_by(4) {
            let mut sum = Simd::<f64, 4>::splat(0.0);
            for (x, &v) in row.iter().enumerate() {
                let c = Simd::from_array(std::array::from_fn(|lane| cosines[(k + lane) * side + x]));
                sum += Simd::splat(f64::from(v) - mean) * c;
            }
            sum.copy_to_slice(&mut rows[y * size + k..y * size + k + 4]);
        }
        for k in end..size {
            rows[y * size + k] = row.iter().zip(&cosines[k * side..(k + 1) * side]).map(|(&v, &c)| (f64::from(v) - mean) * c).sum();
        }
    }
    let mut coefficients = vec![0.0; size * size];
    for ky in 0..size {
        let end = if VECTOR { size / 4 * 4 } else { 0 };
        for kx in (0..end).step_by(4) {
            let mut sum = Simd::<f64, 4>::splat(0.0);
            for y in 0..side {
                sum += Simd::from_slice(&rows[y * size + kx..]) * Simd::splat(cosines[ky * side + y]);
            }
            sum.copy_to_slice(&mut coefficients[ky * size + kx..ky * size + kx + 4]);
        }
        for kx in end..size {
            coefficients[ky * size + kx] = (0..side).map(|y| rows[y * size + kx] * cosines[ky * side + y]).sum();
        }
    }
    coefficients[0] = mean * pixels.len() as f64;
    // Suppress cancellation noise in mathematically zero AC coefficients.
    let tolerance = 255.0 * pixels.len() as f64 * f64::EPSILON * 32.0;
    for value in &mut coefficients[1..] {
        if value.abs() < tolerance {
            *value = 0.0;
        }
    }
    let scratch = &mut rows[..coefficients.len()];
    scratch.copy_from_slice(&coefficients);
    let median = median(scratch);
    pack(coefficients.iter().map(|&value| value > median))
}

fn median(values: &mut [f64]) -> f64 {
    let middle = values.len() / 2;
    let even = values.len().is_multiple_of(2);
    let (lower, median, _) = values.select_nth_unstable_by(middle, f64::total_cmp);
    if even { (*lower.iter().max_by(|a, b| a.total_cmp(b)).unwrap() + *median) / 2.0 } else { *median }
}

#[pyfunction]
fn image_hash(py: Python<'_>, pixels: &[u8], size: usize, side: usize, algorithm: &str) -> PyResult<Vec<u8>> {
    py.detach(|| hash_pixels(pixels, size, side, algorithm))
}

fn hash_pixels(pixels: &[u8], size: usize, side: usize, algorithm: &str) -> PyResult<Vec<u8>> {
    if !(2..=64).contains(&size) {
        return Err(PyValueError::new_err("hash_size must be between 2 and 64"));
    }
    let (width, height) = match algorithm {
        "average" => (size, size),
        "difference" => (size + 1, size),
        "perceptual" if (size..=256).contains(&side) => (side, side),
        _ => return Err(PyValueError::new_err("invalid hashing algorithm or thumbnail size")),
    };
    if pixels.len() != width * height {
        return Err(PyValueError::new_err("incorrect hash thumbnail buffer length"));
    }
    Ok(match algorithm {
        "average" => {
            let sum: usize = pixels.iter().map(|&v| usize::from(v)).sum();
            pack(pixels.iter().map(|&v| usize::from(v) * pixels.len() > sum))
        }
        "difference" => pack((0..size * size).map(|i| {
            let offset = i / size * width + i % size;
            pixels[offset + 1] > pixels[offset]
        })),
        _ => phash(pixels, size, side),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_phash_matches_scalar() {
        for size in [2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 63, 64] {
            for side in [size, size * 2 + 1] {
                for offset in 0..4 {
                    for pattern in 0..3 {
                        let mut pixels = vec![123; offset + side * side + 8];
                        for (i, v) in pixels[offset..offset + side * side].iter_mut().enumerate() {
                            *v = match pattern {
                                0 => (i * 7919 + i / side * 17) as u8,
                                1 => {
                                    if i % 2 == 0 {
                                        0
                                    } else {
                                        255
                                    }
                                }
                                _ => 127,
                            };
                        }
                        let input = &pixels[offset..offset + side * side];
                        assert_eq!(phash_impl::<true>(input, size, side), phash_impl::<false>(input, size, side));
                        assert!(pixels[..offset].iter().chain(&pixels[offset + side * side..]).all(|&v| v == 123));
                    }
                }
            }
        }
    }

    #[test]
    fn selected_median_matches_sorted_coefficients() {
        for size in 2..=64 {
            for repeated in [false, true] {
                let mut values: Vec<f64> =
                    (0..size * size).map(|i| if repeated { (i % 3) as f64 } else { ((i * 7919 % 4093) as f64 - 2046.0) / 7.0 }).collect();
                let mut sorted = values.clone();
                sorted.sort_unstable_by(f64::total_cmp);
                let middle = sorted.len() / 2;
                let expected = if sorted.len().is_multiple_of(2) { (sorted[middle - 1] + sorted[middle]) / 2.0 } else { sorted[middle] };
                assert_eq!(median(&mut values), expected);
            }
        }
    }

    #[test]
    fn packing_preserves_leading_padding_and_order() {
        assert_eq!(pack([true, false, false, false, false, false, false, false, true].into_iter()), [1, 1]);
        assert_eq!(pack([true, false, false, false, false, false, false, false].into_iter()), [128]);
    }

    #[test]
    fn constant_phash_has_only_dc() {
        for size in [2, 3, 8, 9] {
            for value in [0, 1, 127, 255] {
                let side = size * 4;
                let actual = phash(&vec![value; side * side], size, side);
                let expected = pack((0..size * size).map(|i| i == 0 && value != 0));
                assert_eq!(actual, expected);
            }
        }
    }

    #[test]
    fn rejects_invalid_native_inputs() {
        assert!(hash_pixels(&[], 0, 0, "average").is_err());
        assert!(hash_pixels(&[], 8, 32, "perceptual").is_err());
        assert!(hash_pixels(&[], 8, usize::MAX, "perceptual").is_err());
        assert!(hash_pixels(&[], 8, 8, "unknown").is_err());
    }
}
