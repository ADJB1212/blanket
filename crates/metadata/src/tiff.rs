//! Checked classic TIFF fields and append-only directory updates.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub(super) fn invalid(message: &str) -> PyErr {
    PyValueError::new_err(message.to_owned())
}

pub(super) fn slice(data: &[u8], start: usize, length: usize) -> PyResult<&[u8]> {
    data.get(start..start.checked_add(length).ok_or_else(|| invalid("metadata offset overflow"))?).ok_or_else(|| invalid("truncated metadata"))
}

#[derive(Clone, Debug)]
pub(super) struct Field {
    pub tag: u16,
    pub kind: u16,
    pub count: u32,
    pub bytes: Vec<u8>,
}

pub(super) struct Tiff<'a> {
    pub data: &'a [u8],
    pub little: bool,
    pub root: usize,
}

impl<'a> Tiff<'a> {
    pub(super) fn new(raw: &'a [u8]) -> PyResult<Self> {
        let data = raw.strip_prefix(b"Exif\0\0").unwrap_or(raw);
        let little = match slice(data, 0, 2)? {
            b"II" => true,
            b"MM" => false,
            _ => return Err(invalid("invalid EXIF/TIFF byte order")),
        };
        let mut result = Self { data, little, root: 0 };
        if result.short(2)? != 42 {
            return Err(invalid("only classic TIFF metadata is supported"));
        }
        result.root = result.long(4)? as usize;
        Ok(result)
    }

    pub(super) fn short(&self, pos: usize) -> PyResult<u16> {
        let bytes = slice(self.data, pos, 2)?.try_into().expect("two bytes");
        Ok(if self.little { u16::from_le_bytes(bytes) } else { u16::from_be_bytes(bytes) })
    }

    pub(super) fn long(&self, pos: usize) -> PyResult<u32> {
        let bytes = slice(self.data, pos, 4)?.try_into().expect("four bytes");
        Ok(if self.little { u32::from_le_bytes(bytes) } else { u32::from_be_bytes(bytes) })
    }

    pub(super) fn directory(&self, offset: usize) -> PyResult<(Vec<usize>, u32)> {
        if offset == 0 {
            return Ok((Vec::new(), 0));
        }
        let count = usize::from(self.short(offset)?);
        let start = offset.checked_add(2).ok_or_else(|| invalid("TIFF directory overflow"))?;
        slice(self.data, start, count * 12 + 4)?;
        let end = start + count * 12;
        Ok(((start..end).step_by(12).collect(), self.long(end)?))
    }

    pub(super) fn find(&self, offset: usize, tag: u16) -> PyResult<Option<usize>> {
        let mut result = None;
        for pos in self.directory(offset)?.0 {
            if self.short(pos)? == tag {
                if result.is_some() {
                    return Err(invalid("duplicate TIFF metadata tag"));
                }
                result = Some(pos);
            }
        }
        Ok(result)
    }

    pub(super) fn nested(&self, tag: u16) -> PyResult<usize> {
        let Some(pos) = self.find(self.root, tag)? else { return Ok(0) };
        if self.short(pos + 2)? != 4 || self.long(pos + 4)? != 1 {
            return Err(invalid("invalid EXIF directory pointer"));
        }
        Ok(self.long(pos + 8)? as usize)
    }

    pub(super) fn value(&self, pos: usize) -> PyResult<&'a [u8]> {
        let width = match self.short(pos + 2)? {
            1 | 2 | 6 | 7 => 1,
            3 | 8 => 2,
            4 | 9 | 11 | 13 => 4,
            5 | 10 | 12 => 8,
            _ => return Err(invalid("unsupported TIFF metadata field type")),
        };
        let size = (self.long(pos + 4)? as usize).checked_mul(width).ok_or_else(|| invalid("TIFF field size overflow"))?;
        let start = if size <= 4 { pos + 8 } else { self.long(pos + 8)? as usize };
        slice(self.data, start, size)
    }

    pub(super) fn text(&self, offset: usize, tag: u16) -> PyResult<Option<&'a str>> {
        let Some(pos) = self.find(offset, tag)? else { return Ok(None) };
        if self.short(pos + 2)? != 2 {
            return Err(invalid("EXIF timestamp must be ASCII"));
        }
        let raw = self.value(pos)?;
        let raw = raw.strip_suffix(&[0]).unwrap_or(raw);
        let text = std::str::from_utf8(raw).map_err(|_| invalid("invalid EXIF ASCII field"))?;
        Ok(Some(text))
    }

    pub(super) fn numbers(&self, offset: usize, tag: u16) -> PyResult<Option<Vec<f64>>> {
        let Some(pos) = self.find(offset, tag)? else { return Ok(None) };
        let kind = self.short(pos + 2)?;
        let raw = self.value(pos)?;
        let short = |bytes: &[u8]| {
            let bytes = bytes.try_into().expect("two bytes");
            if self.little { u16::from_le_bytes(bytes) } else { u16::from_be_bytes(bytes) }
        };
        let long = |bytes: &[u8]| {
            let bytes = bytes.try_into().expect("four bytes");
            if self.little { u32::from_le_bytes(bytes) } else { u32::from_be_bytes(bytes) }
        };
        let values = match kind {
            1 => raw.iter().map(|v| f64::from(*v)).collect(),
            3 => raw.as_chunks::<2>().0.iter().map(|v| f64::from(short(v))).collect(),
            4 => raw.as_chunks::<4>().0.iter().map(|v| f64::from(long(v))).collect(),
            5 => raw.as_chunks::<8>().0.iter().map(|v| f64::from(long(&v[..4])) / f64::from(long(&v[4..]))).collect(),
            _ => return Err(invalid("unsupported numeric EXIF metadata type")),
        };
        Ok(Some(values))
    }

    pub(super) fn put_short(&self, value: u16) -> [u8; 2] {
        if self.little { value.to_le_bytes() } else { value.to_be_bytes() }
    }

    pub(super) fn put_long(&self, value: u32) -> [u8; 4] {
        if self.little { value.to_le_bytes() } else { value.to_be_bytes() }
    }

    pub(super) fn append_directory(&self, output: &mut Vec<u8>, offset: usize, changes: &[(u16, Option<Field>)]) -> PyResult<u32> {
        let (positions, next) = self.directory(offset)?;
        let mut entries = Vec::new();
        for pos in positions {
            let tag = self.short(pos)?;
            if !changes.iter().any(|(key, _)| *key == tag) {
                entries.push((tag, slice(self.data, pos, 12)?.to_vec()));
            }
        }
        for (_, field) in changes {
            let Some(field) = field else { continue };
            let mut entry = Vec::with_capacity(12);
            entry.extend_from_slice(&self.put_short(field.tag));
            entry.extend_from_slice(&self.put_short(field.kind));
            entry.extend_from_slice(&self.put_long(field.count));
            if field.bytes.len() <= 4 {
                entry.extend_from_slice(&field.bytes);
                entry.resize(12, 0);
            } else {
                if output.len() & 1 != 0 {
                    output.push(0);
                }
                let start = u32::try_from(output.len()).map_err(|_| invalid("TIFF metadata exceeds 4 GiB"))?;
                entry.extend_from_slice(&self.put_long(start));
                output.extend_from_slice(&field.bytes);
            }
            entries.push((field.tag, entry));
        }
        entries.sort_by_key(|(tag, _)| *tag);
        let count = u16::try_from(entries.len()).map_err(|_| invalid("too many TIFF entries"))?;
        if output.len() & 1 != 0 {
            output.push(0);
        }
        let start = u32::try_from(output.len()).map_err(|_| invalid("TIFF metadata exceeds 4 GiB"))?;
        output.extend_from_slice(&self.put_short(count));
        for (_, entry) in entries {
            output.extend_from_slice(&entry);
        }
        output.extend_from_slice(&self.put_long(next));
        Ok(start)
    }
}
