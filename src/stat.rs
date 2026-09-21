//! Native histogram reductions and moments for ImageStat.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyList;

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(stat_count, module)?)?;
    module.add_function(wrap_pyfunction!(stat_extrema, module)?)?;
    module.add_function(wrap_pyfunction!(stat_sum, module)?)?;
    module.add_function(wrap_pyfunction!(stat_median, module)?)?;
    module.add_function(wrap_pyfunction!(stat_normalize, module)?)?;
    module.add_function(wrap_pyfunction!(stat_sqrt, module)?)?;
    Ok(())
}

fn bands(histogram: &[u64]) -> PyResult<&[[u64; 256]]> {
    let (bands, remainder) = histogram.as_chunks::<256>();
    if !remainder.is_empty() {
        return Err(PyValueError::new_err("histogram must contain 256 bins per band"));
    }
    Ok(bands)
}

fn list_count(histogram: &Bound<'_, PyList>, index: usize) -> PyResult<u64> {
    let py = histogram.py();
    // The critical section keeps a borrowed list item alive during conversion
    // on free-threaded Python too; it is a no-op when the GIL is enabled.
    // GetItem checks bounds even if an earlier __index__ call mutated the list.
    pyo3::sync::critical_section::with_critical_section(histogram.as_any(), || unsafe {
        let item = pyo3::ffi::PyList_GetItem(histogram.as_ptr(), index as pyo3::ffi::Py_ssize_t);
        if item.is_null() {
            return Err(PyErr::fetch(py));
        }
        let value = pyo3::ffi::PyLong_AsUnsignedLongLong(item);
        if value == u64::MAX && !pyo3::ffi::PyErr_Occurred().is_null() {
            let item = Bound::<PyAny>::from_borrowed_ptr(py, item);
            let error = PyErr::fetch(py);
            if error.is_instance_of::<pyo3::exceptions::PyTypeError>(py) {
                // Preserve integer-like objects with __index__ on the slow path.
                return item.extract();
            }
            return Err(error);
        }
        Ok(value)
    })
}

#[pyfunction]
fn stat_count(histogram: &Bound<'_, PyList>) -> PyResult<Vec<u128>> {
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
fn stat_extrema(histogram: &Bound<'_, PyList>) -> PyResult<Vec<(u8, u8)>> {
    if !histogram.is_exact_instance_of::<PyList>() {
        let values = histogram.extract::<Vec<u64>>()?;
        return Ok(bands(&values)?
            .iter()
            .map(|band| {
                (
                    band.iter().position(|&n| n != 0).unwrap_or(255) as u8,
                    band.iter().rposition(|&n| n != 0).unwrap_or(0) as u8,
                )
            })
            .collect());
    }
    if !histogram.len().is_multiple_of(256) {
        return Err(PyValueError::new_err("histogram must contain 256 bins per band"));
    }
    let mut result = vec![(255, 0); histogram.len() / 256];
    for (i, (minimum, maximum)) in result.iter_mut().enumerate() {
        for bin in 0..256 {
            // Validate every count, including bins between the extrema.
            if list_count(histogram, i * 256 + bin)? != 0 {
                *minimum = (*minimum).min(bin as u8);
                *maximum = bin as u8;
            }
        }
    }
    Ok(result)
}

fn weighted_sum(histogram: &[u64; 256], squared: bool) -> f64 {
    let mut result = 0.0;
    for (value, &count) in histogram.iter().enumerate() {
        // Match Pillow's accumulation order and conversion to floating point.
        result += if squared {
            (value * value) as f64 * count as f64
        } else {
            (value as u128 * u128::from(count)) as f64
        };
    }
    result
}

#[pyfunction]
fn stat_sum(py: Python<'_>, histogram: Vec<u64>, squared: bool) -> PyResult<Vec<f64>> {
    let bands = bands(&histogram)?;
    Ok(py.detach(|| bands.iter().map(|band| weighted_sum(band, squared)).collect()))
}

fn median(histogram: &[u64; 256], count: u128) -> u8 {
    let mut cumulative = 0_u128;
    for (value, &frequency) in histogram.iter().enumerate() {
        cumulative += u128::from(frequency);
        if cumulative > count / 2 {
            return value as u8;
        }
    }
    // Pillow selects the upper middle sample, and returns 255 for empty bands.
    255
}

#[pyfunction]
fn stat_median(py: Python<'_>, histogram: Vec<u64>, counts: Vec<u128>) -> PyResult<Vec<u16>> {
    let bands = bands(&histogram)?;
    if bands.len() != counts.len() {
        return Err(PyValueError::new_err("band counts do not match histogram"));
    }
    // PyO3 converts Vec<u8> to bytes, while the public API requires a list.
    Ok(py.detach(|| bands.iter().zip(counts).map(|(band, count)| u16::from(median(band, count))).collect()))
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
        values
            .into_iter()
            .map(|value| {
                if value < 0.0 {
                    Err(PyValueError::new_err("math domain error"))
                } else {
                    Ok(value.sqrt())
                }
            })
            .collect()
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
}
