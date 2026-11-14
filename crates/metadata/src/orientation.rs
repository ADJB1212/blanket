//! Remove classic TIFF orientation entries without relocating external values.

use pyo3::prelude::*;
use pyo3::types::PyBytes;

#[pyfunction]
pub fn _exif_orientation<'py>(py: Python<'py>, raw: &[u8]) -> (Option<u32>, Bound<'py, PyBytes>) {
    let (orientation, cleaned) = py.detach(|| orientation(raw));
    (orientation, PyBytes::new(py, cleaned.as_deref().unwrap_or(raw)))
}

fn orientation(raw: &[u8]) -> (Option<u32>, Option<Vec<u8>>) {
    let prefix = if raw.starts_with(b"Exif\0\0") { 6 } else { 0 };
    let data = &raw[prefix..];
    let little = match data.get(..2) {
        Some(b"II") => true,
        Some(b"MM") => false,
        _ => return (None, None),
    };
    let short = |offset: usize| -> Option<u16> {
        let bytes = data.get(offset..offset.checked_add(2)?)?.try_into().ok()?;
        Some(if little { u16::from_le_bytes(bytes) } else { u16::from_be_bytes(bytes) })
    };
    let long = |offset: usize| -> Option<u32> {
        let bytes = data.get(offset..offset.checked_add(4)?)?.try_into().ok()?;
        Some(if little { u32::from_le_bytes(bytes) } else { u32::from_be_bytes(bytes) })
    };
    let read = || -> Option<(u32, Option<Vec<u8>>)> {
        if short(2)? != 42 {
            return None;
        }
        let offset = usize::try_from(long(4)?).ok()?;
        let count = short(offset)?;
        let start = offset.checked_add(2)?;
        let end = start.checked_add(12 * usize::from(count))?;
        data.get(..end.checked_add(4)?)?;
        for pos in (start..end).step_by(12) {
            let kind = short(pos + 2)?;
            if short(pos)? != 274 || !matches!(kind, 3 | 4) || long(pos + 4)? != 1 {
                continue;
            }
            let value = if kind == 3 { u32::from(short(pos + 8)?) } else { long(pos + 8)? };
            if !(2..=8).contains(&value) {
                return Some((value, None));
            }
            let mut clean = raw.to_vec();
            let count = if little { (count - 1).to_le_bytes() } else { (count - 1).to_be_bytes() };
            clean[prefix + offset..prefix + offset + 2].copy_from_slice(&count);
            clean.copy_within(prefix + pos + 12..prefix + end + 4, prefix + pos);
            clean[prefix + end - 8..prefix + end + 4].fill(0);
            return Some((value, Some(clean)));
        }
        None
    };
    match read() {
        Some((value, clean)) => (Some(value), clean),
        None => (None, None),
    }
}
