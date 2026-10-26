//! Native histogram reductions and moments for `ImageStat`.

use std::simd::Simd;
use std::simd::cmp::SimdPartialOrd;
use std::simd::num::SimdUint;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyList;

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(stat_histogram, module)?)?;
    module.add_function(wrap_pyfunction!(stat_count, module)?)?;
    module.add_function(wrap_pyfunction!(stat_extrema, module)?)?;
    module.add_function(wrap_pyfunction!(stat_sum, module)?)?;
    module.add_function(wrap_pyfunction!(stat_variance, module)?)?;
    module.add_function(wrap_pyfunction!(stat_median, module)?)?;
    module.add_function(wrap_pyfunction!(stat_normalize, module)?)?;
    module.add_function(wrap_pyfunction!(stat_sqrt, module)?)?;
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (image, mask=None))]
fn stat_histogram(py: Python<'_>, image: &blanket_core::Image, mask: Option<&blanket_core::Image>) -> PyResult<Vec<u64>> {
    use blanket_core::{PixelMode, samples};
    if !samples::supported(image) {
        return Err(PyValueError::new_err("unsigned high-depth image required"));
    }
    let source = image.raw_data()?;
    let mask = mask
        .map(|mask| {
            if mask.mode != PixelMode::L || mask.bit_depth != 8 || (mask.width, mask.height) != (image.width, image.height) {
                return Err(PyValueError::new_err("bad mask mode or size"));
            }
            mask.pixel_data()
        })
        .transpose()?;
    let channels = image.mode.channels();
    let bins = usize::from(samples::maximum(image)) + 1;
    Ok(py.detach(|| {
        let mut histogram = vec![0; bins * channels];
        for (i, pixel) in source.chunks_exact(channels * 2).enumerate() {
            if mask.is_some_and(|mask| mask[i] == 0) {
                continue;
            }
            for (c, &bytes) in pixel.as_chunks::<2>().0.iter().enumerate() {
                let value = if image.mode == PixelMode::I16B { u16::from_be_bytes(bytes) } else { u16::from_le_bytes(bytes) };
                histogram[c * bins + usize::from(value)] += 1;
            }
        }
        histogram
    }))
}

fn sample_bands(histogram: &[u64], bins: usize) -> PyResult<std::slice::ChunksExact<'_, u64>> {
    if !matches!(bins, 256 | 1024 | 4096 | 65536) || !histogram.len().is_multiple_of(bins) {
        if bins == 256 {
            return Err(PyValueError::new_err("histogram must contain 256 bins per band"));
        }
        return Err(PyValueError::new_err("histogram has wrong number of bins per band"));
    }
    Ok(histogram.chunks_exact(bins))
}

fn bands(histogram: &[u64]) -> PyResult<&[[u64; 256]]> {
    let (bands, remainder) = histogram.as_chunks::<256>();
    if !remainder.is_empty() {
        return Err(PyValueError::new_err("histogram must contain 256 bins per band"));
    }
    Ok(bands)
}

fn list_count(histogram: &Bound<'_, PyList>, index: usize) -> PyResult<u64> {
    histogram.get_item(index)?.extract()
}

#[pyfunction]
#[pyo3(signature = (histogram, bins=256))]
fn stat_count(histogram: &Bound<'_, PyList>, bins: usize) -> PyResult<Vec<u128>> {
    if bins != 256 {
        let values = histogram.extract::<Vec<u64>>()?;
        return Ok(sample_bands(&values, bins)?.map(|band| band.iter().map(|&n| u128::from(n)).sum()).collect());
    }
    if !histogram.is_exact_instance_of::<PyList>() {
        let values = histogram.extract::<Vec<u64>>()?;
        return Ok(bands(&values)?.iter().map(|band| band.iter().map(|&n| u128::from(n)).sum()).collect());
    }
    if !histogram.len().is_multiple_of(256) {
        return Err(PyValueError::new_err("histogram must contain 256 bins per band"));
    }
    let mut counts = vec![0_u128; histogram.len() / 256];
    // Access list entries directly: materializing a Vec through Python's
    // iterator costs more than these small reductions themselves.
    for (i, count) in counts.iter_mut().enumerate() {
        for bin in 0..256 {
            *count += u128::from(list_count(histogram, i * 256 + bin)?);
        }
    }
    Ok(counts)
}

#[pyfunction]
#[pyo3(signature = (histogram, bins=256))]
fn stat_extrema<'py>(histogram: &Bound<'py, PyList>, bins: usize) -> PyResult<Bound<'py, PyList>> {
    if bins != 256 {
        let values = histogram.extract::<Vec<u64>>()?;
        return PyList::new(
            histogram.py(),
            sample_bands(&values, bins)?
                .map(|band| (band.iter().position(|&n| n != 0).unwrap_or(bins - 1), band.iter().rposition(|&n| n != 0).unwrap_or(0))),
        );
    }
    if !histogram.is_exact_instance_of::<PyList>() {
        let values = histogram.extract::<Vec<u64>>()?;
        return PyList::new(
            histogram.py(),
            bands(&values)?
                .iter()
                .map(|band| (band.iter().position(|&n| n != 0).unwrap_or(255) as u8, band.iter().rposition(|&n| n != 0).unwrap_or(0) as u8)),
        );
    }
    if !histogram.len().is_multiple_of(256) {
        return Err(PyValueError::new_err("histogram must contain 256 bins per band"));
    }
    let result = PyList::empty(histogram.py());
    for i in 0..histogram.len() / 256 {
        let mut counts = [0; 256];
        // Validate the full band before finding its occupied endpoints.
        for (bin, count) in counts.iter_mut().enumerate() {
            *count = list_count(histogram, i * 256 + bin)?;
        }
        let minimum = counts.iter().position(|&n| n != 0).unwrap_or(255) as u8;
        let maximum = counts.iter().rposition(|&n| n != 0).unwrap_or(0) as u8;
        result.append((minimum, maximum))?;
    }
    Ok(result)
}

fn weighted_sum(histogram: &[u64], squared: bool) -> f64 {
    let mut result = 0.0;
    let end = if histogram.len() > 256 { histogram.len() / 8 * 8 } else { 0 };
    for i in (0..end).step_by(8) {
        let counts = Simd::<u64, 8>::from_slice(&histogram[i..]);
        let indices = Simd::<u64, 8>::from_array(std::array::from_fn(|lane| (i + lane) as u64));
        if !squared && counts.simd_gt(Simd::splat(u64::MAX / 65535)).any() {
            for (lane, &count) in histogram[i..i + 8].iter().enumerate() {
                result += ((i + lane) as u128 * u128::from(count)) as f64;
            }
        } else {
            let terms = if squared { (indices * indices).cast::<f64>() * counts.cast::<f64>() } else { (indices * counts).cast::<f64>() };
            for term in terms.to_array() {
                result += term;
            }
        }
    }
    for (i, &count) in histogram[end..].iter().enumerate() {
        let value = end + i;
        // Match Pillow's accumulation order and conversion to floating point.
        result += if squared { (value * value) as f64 * count as f64 } else { (value as u128 * u128::from(count)) as f64 };
    }
    result
}

#[pyfunction]
#[pyo3(signature = (histogram, squared, bins=256))]
fn stat_sum(py: Python<'_>, histogram: Vec<u64>, squared: bool, bins: usize) -> PyResult<Vec<f64>> {
    let bands = sample_bands(&histogram, bins)?;
    Ok(py.detach(|| bands.map(|band| weighted_sum(band, squared)).collect()))
}

fn median(histogram: &[u64], count: u128) -> u16 {
    let mut cumulative = 0_u128;
    for (value, &frequency) in histogram.iter().enumerate() {
        cumulative += u128::from(frequency);
        if cumulative > count / 2 {
            return value as u16;
        }
    }
    // Select the upper middle sample, or the sample maximum for empty bands.
    (histogram.len() - 1) as u16
}

#[pyfunction]
#[pyo3(signature = (histogram, counts, bins=256))]
fn stat_median(py: Python<'_>, histogram: Vec<u64>, counts: Vec<u128>, bins: usize) -> PyResult<Vec<u16>> {
    let bands = sample_bands(&histogram, bins)?;
    if bands.len() != counts.len() {
        return Err(PyValueError::new_err("band counts do not match histogram"));
    }
    // PyO3 converts Vec<u8> to bytes, while the public API requires a list.
    Ok(py.detach(|| bands.zip(counts).map(|(band, count)| median(band, count)).collect()))
}

#[pyfunction]
fn stat_variance(py: Python<'_>, histogram: Vec<u64>, counts: Vec<u128>, sums: Vec<f64>, bins: usize) -> PyResult<Vec<f64>> {
    let bands = sample_bands(&histogram, bins)?;
    if bands.len() != counts.len() || bands.len() != sums.len() {
        return Err(PyValueError::new_err("statistics have different band counts"));
    }
    Ok(py.detach(|| {
        bands
            .zip(counts)
            .zip(sums)
            .map(|((band, count), sum)| {
                if count == 0 {
                    return 0.0;
                }
                let mean = sum / count as f64;
                centered_sum(band, mean) / count as f64
            })
            .collect()
    }))
}

fn centered_sum(histogram: &[u64], mean: f64) -> f64 {
    let end = histogram.len() / 8 * 8;
    let mut sum = 0.0;
    for i in (0..end).step_by(8) {
        let counts = Simd::<u64, 8>::from_slice(&histogram[i..]);
        let positions = Simd::<f64, 8>::from_array(std::array::from_fn(|lane| (i + lane) as f64));
        let difference = positions - Simd::splat(mean);
        let terms = (difference * difference * counts.cast::<f64>()).to_array();
        for (lane, term) in terms.into_iter().enumerate() {
            if histogram[i + lane] != 0 {
                sum += term;
            }
        }
    }
    for (i, &frequency) in histogram[end..].iter().enumerate() {
        if frequency != 0 {
            sum += ((end + i) as f64 - mean).powi(2) * frequency as f64;
        }
    }
    sum
}

/// Normalize sums (or squared sums) by count, optionally subtracting the mean
/// contribution first to compute population variance in Pillow's order.
#[pyfunction]
#[pyo3(signature = (values, counts, sums=None))]
fn stat_normalize(py: Python<'_>, values: Vec<f64>, counts: Vec<u128>, sums: Option<Vec<f64>>) -> PyResult<Vec<f64>> {
    if values.len() != counts.len() || sums.as_ref().is_some_and(|sums| sums.len() != values.len()) {
        return Err(PyValueError::new_err("statistics have different band counts"));
    }
    Ok(py.detach(|| {
        values
            .iter()
            .zip(&counts)
            .enumerate()
            .map(|(i, (&value, &count))| {
                if count == 0 {
                    0.0
                } else if let Some(sums) = &sums {
                    (value - sums[i].powi(2) / count as f64) / count as f64
                } else {
                    value / count as f64
                }
            })
            .collect()
    }))
}

#[pyfunction]
fn stat_sqrt(py: Python<'_>, values: Vec<f64>) -> PyResult<Vec<f64>> {
    py.detach(|| {
        values.into_iter().map(|value| if value < 0.0 { Err(PyValueError::new_err("math domain error")) } else { Ok(value.sqrt()) }).collect()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median_uses_upper_middle_and_empty_sentinel() {
        let mut histogram = [0; 256];
        assert_eq!(median(&histogram, 0), 255);
        histogram[10] = 2;
        histogram[200] = 2;
        assert_eq!(median(&histogram, 4), 200);
        histogram[10] += 1;
        assert_eq!(median(&histogram, 5), 10);
    }

    #[test]
    fn weighted_sums_do_not_overflow_integer_intermediates() {
        let mut histogram = [0; 256];
        histogram[255] = u64::MAX;
        assert_eq!(weighted_sum(&histogram, false), (255 * u128::from(u64::MAX)) as f64);
        assert_eq!(weighted_sum(&histogram, true), 65025.0 * u64::MAX as f64);
    }

    #[test]
    fn high_depth_moments_preserve_scalar_accumulation_and_large_counts() {
        for count in [1024, 4096, 65536] {
            let mut backing: Vec<_> = (0..count + 1).map(|i| (i as u64 * 719) % 10001).collect();
            for i in [1, count / 2, count] {
                backing[i] = u64::MAX;
            }
            let histogram = &backing[1..];
            let mut sum = 0.0;
            let mut sum2 = 0.0;
            let mut variance = 0.0;
            let mean = 619.25;
            for (value, &frequency) in histogram.iter().enumerate() {
                sum += (value as u128 * u128::from(frequency)) as f64;
                sum2 += (value * value) as f64 * frequency as f64;
                if frequency != 0 {
                    variance += (value as f64 - mean).powi(2) * frequency as f64;
                }
            }
            assert_eq!(weighted_sum(histogram, false), sum);
            assert_eq!(weighted_sum(histogram, true), sum2);
            assert_eq!(centered_sum(histogram, mean), variance);
        }
    }
}
