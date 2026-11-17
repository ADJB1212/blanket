use blanket_core::{Image, PixelMode};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

// SplitMix64 constants: https://prng.di.unimi.it/splitmix64.c
const PROJECTION_SEED_STEP: u64 = 0x9e37_79b9_7f4a_7c15;
const MIX_MULTIPLIER_1: u64 = 0xbf58_476d_1ce4_e5b9;
const MIX_MULTIPLIER_2: u64 = 0x94d0_49bb_1331_11eb;

fn mix(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(MIX_MULTIPLIER_1);
    value = (value ^ (value >> 27)).wrapping_mul(MIX_MULTIPLIER_2);
    value ^ (value >> 31)
}

fn project(counts: &[u64], bits: usize) -> Vec<u8> {
    let weights: Vec<(usize, f64)> =
        counts.iter().enumerate().filter(|&(_, &count)| count != 0).map(|(i, &count)| (i, (count as f64).sqrt())).collect();
    super::pack((0..bits).map(|bit| {
        let seed = (bit as u64 + 1).wrapping_mul(PROJECTION_SEED_STEP);
        let sum: f64 = weights.iter().map(|&(bin, weight)| if mix(seed ^ bin as u64) & 1 == 0 { weight } else { -weight }).sum();
        sum > 0.0
    }))
}

fn color_counts(pixels: &[u8]) -> [u64; 64] {
    let mut counts = [0; 64];
    for &[r, g, b] in pixels.as_chunks::<3>().0 {
        let bin = usize::from(r >> 6) * 16 + usize::from(g >> 6) * 4 + usize::from(b >> 6);
        counts[bin] += 1;
    }
    counts
}

fn texture_counts(pixels: &[u8], width: usize, height: usize) -> [u64; 256] {
    let mut counts = [0; 256];
    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let i = y * width + x;
            let neighbors = [i - width - 1, i - width, i - width + 1, i + 1, i + width + 1, i + width, i + width - 1, i - 1];
            let mut bin = 0;
            for (bit, index) in neighbors.into_iter().enumerate() {
                bin |= usize::from(pixels[index] >= pixels[i]) << bit;
            }
            counts[bin] += 1;
        }
    }
    counts
}

#[pyfunction]
pub(super) fn image_histogram_hash_native<'py>(py: Python<'py>, image: &Image, size: usize, algorithm: &str) -> PyResult<Bound<'py, PyBytes>> {
    if !(2..=64).contains(&size) {
        return Err(PyValueError::new_err("hash_size must be between 2 and 64"));
    }
    if image.bit_depth() != 8 || image.width() == 0 || image.height() == 0 {
        return Err(PyValueError::new_err("histogram hashes require nonempty 8-bit pixels"));
    }
    match algorithm {
        "color" if image.pixel_mode() == PixelMode::Rgb => {}
        "crop-resistant" if image.pixel_mode() == PixelMode::L => {
            if image.width() < 3 || image.height() < 3 {
                return Err(PyValueError::new_err("crop-resistant hashes require dimensions of at least 3x3"));
            }
        }
        _ => return Err(PyValueError::new_err("invalid histogram hash algorithm or pixel mode")),
    }
    let pixels = image.pixel_data()?;
    let packed = py.detach(|| match algorithm {
        "color" => project(&color_counts(pixels), size * size),
        _ => project(&texture_counts(pixels, image.width() as usize, image.height() as usize), size * size),
    });
    Ok(PyBytes::new(py, &packed))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn projection_constants_preserve_stored_fingerprints() {
        assert_eq!(mix(PROJECTION_SEED_STEP), 0xe220_a839_7b1d_cdaf);
        let mut counts = [0; 64];
        counts[0] = 1;
        assert_eq!(project(&counts, 64), 0x5550_733c_80e8_5264_u64.to_be_bytes());
    }

    #[test]
    fn color_bins_include_endpoints_and_ignore_order() {
        assert_eq!(color_counts(&[0, 0, 0, 255, 255, 255, 64, 128, 192, 63, 127, 191]), {
            let mut expected = [0; 64];
            for bin in [0, 63, 27, 6] {
                expected[bin] = 1;
            }
            expected
        });
        assert_eq!(color_counts(&[0, 255, 0, 255, 0, 0]), color_counts(&[255, 0, 0, 0, 255, 0]));
    }

    #[test]
    fn local_patterns_use_clockwise_bits_and_inclusive_ties() {
        let mut expected = [0; 256];
        expected[0b1010_0101] = 1;
        assert_eq!(texture_counts(&[5, 4, 6, 8, 5, 0, 1, 9, 2], 3, 3), expected);
        expected = [0; 256];
        expected[255] = 6;
        assert_eq!(texture_counts(&[17; 20], 4, 5), expected);
    }

    #[test]
    fn histogram_projection_preserves_proportions_and_unaligned_inputs() {
        let mut storage = vec![123; 3 + 3 * 65 + 7];
        for (i, value) in storage[3..3 + 3 * 65].iter_mut().enumerate() {
            *value = (i * 37) as u8;
        }
        let counts = color_counts(&storage[3..3 + 3 * 65]);
        let scaled = counts.map(|count| count * 4);
        for size in [2, 3, 8, 9, 64] {
            assert_eq!(project(&counts, size * size), project(&scaled, size * size));
        }
        assert!(storage[..3].iter().chain(&storage[3 + 3 * 65..]).all(|&value| value == 123));
        assert_eq!(project(&[0; 256], 9), [0, 0]);
    }
}
