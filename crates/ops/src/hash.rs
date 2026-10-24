//! Perceptual hashes of small, grayscale thumbnails.

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
    let mean = pixels.iter().map(|&v| f64::from(v)).sum::<f64>() / pixels.len() as f64;
    let cosines: Vec<f64> = (0..size)
        .flat_map(|k| (0..side).map(move |x| (std::f64::consts::PI * k as f64 * (2 * x + 1) as f64 / (2 * side) as f64).cos()))
        .collect();
    let mut rows = vec![0.0; side * size];
    for (y, row) in pixels.chunks_exact(side).enumerate() {
        for k in 0..size {
            rows[y * size + k] = row
                .iter()
                .zip(&cosines[k * side..(k + 1) * side])
                .map(|(&v, &c)| (f64::from(v) - mean) * c)
                .sum();
        }
    }
    let mut coefficients = vec![0.0; size * size];
    for ky in 0..size {
        for kx in 0..size {
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
    let mut ordered = coefficients.clone();
    ordered.sort_unstable_by(f64::total_cmp);
    let middle = ordered.len() / 2;
    let median = if ordered.len().is_multiple_of(2) {
        (ordered[middle - 1] + ordered[middle]) / 2.0
    } else {
        ordered[middle]
    };
    pack(coefficients.iter().map(|&value| value > median))
}

#[pyfunction]
fn image_hash(pixels: &[u8], size: usize, side: usize, algorithm: &str) -> PyResult<Vec<u8>> {
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
        assert!(image_hash(&[], 0, 0, "average").is_err());
        assert!(image_hash(&[], 8, 32, "perceptual").is_err());
        assert!(image_hash(&[], 8, usize::MAX, "perceptual").is_err());
        assert!(image_hash(&[], 8, 8, "unknown").is_err());
    }
}
