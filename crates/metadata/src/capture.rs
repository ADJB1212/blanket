//! Native EXIF geographic coordinates and capture timestamp fields.

use pyo3::prelude::*;
use pyo3::types::PyBytes;

use super::tiff::{Field, Tiff, invalid};

type Gps = (f64, f64, Option<f64>);
type CaptureTime = (u16, u8, u8, u8, u8, u8, u32, Option<i32>);

fn gps(tiff: &Tiff<'_>) -> PyResult<Option<Gps>> {
    let offset = tiff.nested(34853)?;
    if offset == 0 {
        return Ok(None);
    }
    let mut present = false;
    for tag in 1..=4 {
        present |= tiff.find(offset, tag)?.is_some();
    }
    if !present {
        return Ok(None);
    }
    let coordinate = |tag: u16, positive: &str, negative: &str, limit: f64| -> PyResult<f64> {
        let reference = tiff.text(offset, tag - 1)?.ok_or_else(|| invalid("missing GPS coordinate reference"))?;
        let numbers = tiff.numbers(offset, tag)?.ok_or_else(|| invalid("missing GPS coordinate"))?;
        let [degrees, minutes, seconds] = numbers.as_slice() else { return Err(invalid("GPS coordinates require degrees, minutes, seconds")) };
        if !numbers.iter().all(|v| v.is_finite())
            || !(0.0..=limit).contains(degrees)
            || !(0.0..60.0).contains(minutes)
            || !(0.0..60.0).contains(seconds)
        {
            return Err(invalid("invalid GPS coordinate"));
        }
        let value = degrees + minutes / 60.0 + seconds / 3600.0;
        if value > limit {
            return Err(invalid("GPS coordinate outside geographic range"));
        }
        match reference {
            v if v == positive => Ok(value),
            v if v == negative => Ok(-value),
            _ => Err(invalid("invalid GPS coordinate reference")),
        }
    };
    let latitude = coordinate(2, "N", "S", 90.0)?;
    let longitude = coordinate(4, "E", "W", 180.0)?;
    let altitude = match tiff.numbers(offset, 6)? {
        None => None,
        Some(values) => {
            let [value] = values.as_slice() else { return Err(invalid("invalid GPS altitude")) };
            if !value.is_finite() || *value < 0.0 {
                return Err(invalid("invalid GPS altitude"));
            }
            let reference = tiff.numbers(offset, 5)?.unwrap_or_else(|| vec![0.0]);
            Some(match reference.as_slice() {
                [0.0] => *value,
                [1.0] => -*value,
                _ => return Err(invalid("invalid GPS altitude reference")),
            })
        }
    };
    Ok(Some((latitude, longitude, altitude)))
}

#[pyfunction]
pub fn _get_gps(py: Python<'_>, raw: &[u8]) -> PyResult<Option<Gps>> {
    py.detach(|| gps(&Tiff::new(raw)?))
}

fn field(tag: u16, kind: u16, count: u32, bytes: Vec<u8>) -> (u16, Option<Field>) {
    (tag, Some(Field { tag, kind, count, bytes }))
}

fn ascii(tag: u16, value: String) -> (u16, Option<Field>) {
    let mut bytes = value.into_bytes();
    bytes.push(0);
    field(tag, 2, bytes.len() as u32, bytes)
}

fn update(raw: &[u8], pointer: u16, changes: Option<Vec<(u16, Option<Field>)>>) -> PyResult<Vec<u8>> {
    let tiff = Tiff::new(raw)?;
    let mut output = tiff.data.to_vec();
    let value = if let Some(changes) = changes {
        let offset = tiff.append_directory(&mut output, tiff.nested(pointer)?, &changes)?;
        Some(Field { tag: pointer, kind: 4, count: 1, bytes: tiff.put_long(offset).to_vec() })
    } else {
        None
    };
    let root = tiff.append_directory(&mut output, tiff.root, &[(pointer, value)])?;
    output[4..8].copy_from_slice(&tiff.put_long(root));
    let mut result = b"Exif\0\0".to_vec();
    result.extend_from_slice(&output);
    Ok(result)
}

fn rational(tiff: &Tiff<'_>, value: f64) -> PyResult<Vec<u8>> {
    let mut denominator = 1_000_000_u32;
    while value * f64::from(denominator) > f64::from(u32::MAX) && denominator > 1 {
        denominator /= 10;
    }
    let numerator = (value * f64::from(denominator)).round();
    if !numerator.is_finite() || numerator < 0.0 || numerator > f64::from(u32::MAX) {
        return Err(invalid("GPS rational exceeds EXIF range"));
    }
    let mut bytes = tiff.put_long(numerator as u32).to_vec();
    bytes.extend_from_slice(&tiff.put_long(denominator));
    Ok(bytes)
}

fn set_gps(raw: &[u8], value: Option<Gps>) -> PyResult<Vec<u8>> {
    let Some((latitude, longitude, altitude)) = value else { return update(raw, 34853, None) };
    if !latitude.is_finite() || latitude.abs() > 90.0 || !longitude.is_finite() || longitude.abs() > 180.0 || altitude.is_some_and(|v| !v.is_finite())
    {
        return Err(invalid("GPS coordinates must be finite and within geographic range"));
    }
    let tiff = Tiff::new(raw)?;
    let dms = |value: f64| -> PyResult<Vec<u8>> {
        // Round total seconds first so a carry cannot produce 60 seconds.
        let total = (value.abs() * 3_600_000_000.0).round() as u64;
        let degrees = (total / 3_600_000_000) as f64;
        let minutes = ((total % 3_600_000_000) / 60_000_000) as f64;
        let seconds = (total % 60_000_000) as f64 / 1_000_000.0;
        let mut bytes = rational(&tiff, degrees)?;
        bytes.extend_from_slice(&rational(&tiff, minutes)?);
        bytes.extend_from_slice(&rational(&tiff, seconds)?);
        Ok(bytes)
    };
    let mut changes = vec![
        field(0, 1, 4, vec![2, 3, 0, 0]),
        ascii(1, if latitude.is_sign_negative() { "S" } else { "N" }.into()),
        field(2, 5, 3, dms(latitude)?),
        ascii(3, if longitude.is_sign_negative() { "W" } else { "E" }.into()),
        field(4, 5, 3, dms(longitude)?),
        (5, None),
        (6, None),
    ];
    if let Some(altitude) = altitude {
        changes.push(field(5, 1, 1, vec![u8::from(altitude.is_sign_negative())]));
        changes.push(field(6, 5, 1, rational(&tiff, altitude.abs())?));
    }
    update(raw, 34853, Some(changes))
}

#[pyfunction]
pub fn _set_gps<'py>(py: Python<'py>, raw: &[u8], value: Option<Gps>) -> PyResult<Bound<'py, PyBytes>> {
    let output = py.detach(|| set_gps(raw, value))?;
    Ok(PyBytes::new(py, &output))
}

fn validate_time(value: CaptureTime) -> PyResult<CaptureTime> {
    let (year, month, day, hour, minute, second, microsecond, offset) = value;
    let leap = year.is_multiple_of(4) && (!year.is_multiple_of(100) || year.is_multiple_of(400));
    let days = match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            if leap {
                29
            } else {
                28
            }
        }
        _ => 0,
    };
    if !(1..=9999).contains(&year)
        || day == 0
        || day > days
        || hour > 23
        || minute > 59
        || second > 59
        || microsecond > 999999
        || offset.is_some_and(|v| v.unsigned_abs() >= 1440)
    {
        return Err(invalid("invalid EXIF capture timestamp"));
    }
    Ok(value)
}

fn capture_time(raw: &[u8]) -> PyResult<Option<CaptureTime>> {
    let tiff = Tiff::new(raw)?;
    let directory = tiff.nested(34665)?;
    let Some(text) = tiff.text(directory, 36867)? else { return Ok(None) };
    let bytes = text.as_bytes();
    if bytes.len() != 19 || [4, 7, 10, 13, 16].iter().zip(b":: ::").any(|(i, c)| bytes[*i] != *c) {
        return Err(invalid("invalid EXIF DateTimeOriginal"));
    }
    let number = |start: usize, end: usize| -> PyResult<u16> {
        let part = &bytes[start..end];
        if !part.iter().all(u8::is_ascii_digit) {
            return Err(invalid("invalid EXIF timestamp digits"));
        }
        Ok(part.iter().fold(0_u16, |value, digit| value * 10 + u16::from(digit - b'0')))
    };
    let subsecond = tiff.text(directory, 37521)?.unwrap_or("");
    if !subsecond.bytes().all(|c| c.is_ascii_digit()) {
        return Err(invalid("invalid EXIF SubSecTimeOriginal"));
    }
    let mut digits = subsecond.bytes().chain(std::iter::repeat(b'0'));
    let microsecond = (0..6).fold(0, |v, _| v * 10 + u32::from(digits.next().expect("infinite digits") - b'0'));
    let offset = if let Some(text) = tiff.text(directory, 36881)? {
        let b = text.as_bytes();
        if b.len() != 6 || !matches!(b[0], b'+' | b'-') || b[3] != b':' || ![b[1], b[2], b[4], b[5]].iter().all(u8::is_ascii_digit) {
            return Err(invalid("invalid EXIF OffsetTimeOriginal"));
        }
        let hours = i32::from(b[1] - b'0') * 10 + i32::from(b[2] - b'0');
        let minutes = i32::from(b[4] - b'0') * 10 + i32::from(b[5] - b'0');
        if hours > 23 || minutes > 59 {
            return Err(invalid("invalid EXIF timezone offset"));
        }
        Some((hours * 60 + minutes) * if b[0] == b'-' { -1 } else { 1 })
    } else {
        None
    };
    validate_time((
        number(0, 4)?,
        number(5, 7)? as u8,
        number(8, 10)? as u8,
        number(11, 13)? as u8,
        number(14, 16)? as u8,
        number(17, 19)? as u8,
        microsecond,
        offset,
    ))
    .map(Some)
}

#[pyfunction]
pub fn _get_capture_time(py: Python<'_>, raw: &[u8]) -> PyResult<Option<CaptureTime>> {
    py.detach(|| capture_time(raw))
}

fn set_capture_time(raw: &[u8], value: Option<CaptureTime>) -> PyResult<Vec<u8>> {
    let mut changes = vec![(36867, None), (36881, None), (37521, None)];
    if let Some(value) = value {
        let (year, month, day, hour, minute, second, microsecond, offset) = validate_time(value)?;
        let offset = offset.ok_or_else(|| invalid("capture time requires an explicit timezone offset"))?;
        changes.push(ascii(36867, format!("{year:04}:{month:02}:{day:02} {hour:02}:{minute:02}:{second:02}")));
        changes.push(ascii(36881, format!("{}{:02}:{:02}", if offset < 0 { "-" } else { "+" }, offset.abs() / 60, offset.abs() % 60)));
        changes.push(ascii(37521, format!("{microsecond:06}")));
    }
    update(raw, 34665, Some(changes))
}

#[pyfunction]
pub fn _set_capture_time<'py>(py: Python<'py>, raw: &[u8], value: Option<CaptureTime>) -> PyResult<Bound<'py, PyBytes>> {
    let output = py.detach(|| set_capture_time(raw, value))?;
    Ok(PyBytes::new(py, &output))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty(little: bool) -> Vec<u8> {
        if little { b"II*\0\x08\0\0\0\0\0\0\0\0\0".to_vec() } else { b"MM\0*\0\0\0\x08\0\0\0\0\0\0".to_vec() }
    }

    #[test]
    fn native_capture_roundtrips_both_endians() {
        for little in [false, true] {
            let mut raw = empty(little);
            assert_eq!(capture_time(&raw).unwrap(), None);
            for offset in [-1439, -345, 0, 345, 1439] {
                let value = (2024, 2, 29, 23, 59, 59, 1, Some(offset));
                raw = set_capture_time(&raw, Some(value)).unwrap();
                assert_eq!(capture_time(&raw).unwrap(), Some(value));
            }
            let cleared = set_capture_time(&raw, None).unwrap();
            assert_eq!(capture_time(&cleared).unwrap(), None);
        }
    }

    #[test]
    fn native_gps_roundtrips_boundaries() {
        for little in [false, true] {
            for value in [(90.0, 180.0, None), (-90.0, -180.0, Some(-100.25)), (0.0, 0.0, Some(0.0)), (1.99999999999, -2.99999999999, None)] {
                let raw = set_gps(&empty(little), Some(value)).unwrap();
                let actual = gps(&Tiff::new(&raw).unwrap()).unwrap().unwrap();
                assert!((actual.0 - value.0).abs() < 1e-9);
                assert!((actual.1 - value.1).abs() < 1e-9);
                assert_eq!(actual.2, value.2);
                let cleared = set_gps(&raw, None).unwrap();
                assert_eq!(gps(&Tiff::new(&cleared).unwrap()).unwrap(), None);
            }
        }
    }

    #[test]
    fn native_capture_rejects_invalid_fields_and_offsets() {
        assert!(validate_time((2023, 2, 29, 0, 0, 0, 0, Some(0))).is_err());
        assert!(validate_time((2024, 2, 29, 0, 0, 0, 0, Some(i32::MIN))).is_err());
        assert!(validate_time((2024, 2, 29, 0, 0, 0, 0, Some(1440))).is_err());
        for length in 0..14 {
            assert!(capture_time(&empty(true)[..length]).is_err());
        }
        let raw = b"II*\0\xff\xff\xff\xff";
        assert!(capture_time(raw).is_err());
        assert!(set_gps(&empty(true), Some((f64::NAN, 0.0, None))).is_err());
    }
}
