//! XMP field retention using validated XML source ranges.

use pyo3::prelude::*;
use pyo3::types::PyBytes;
use rustc_hash::FxHashSet;

#[pyfunction]
pub(crate) fn _metadata_location(name: &str) -> bool {
    // Casefold expansions used by the ASCII location field names.
    let local = name.rsplit('}').next().unwrap_or(name).to_lowercase().replace('\u{017f}', "s").replace(['\u{fb05}', '\u{fb06}'], "st");
    local.starts_with("gps")
        || matches!(
            local.as_str(),
            "location"
                | "locationcreated"
                | "locationshown"
                | "sublocation"
                | "city"
                | "state"
                | "province"
                | "country"
                | "countrycode"
                | "countryname"
                | "locationdetails"
        )
}

fn keep(namespace: Option<&str>, local: &str, location: bool, fields: Option<&FxHashSet<String>>) -> bool {
    if !location && _metadata_location(local) {
        return false;
    }
    fields.is_none_or(|fields| {
        matches!(namespace, Some("adobe:ns:meta/" | "http://www.w3.org/1999/02/22-rdf-syntax-ns#"))
            || fields.contains(&super::xmp::name(namespace, local, false))
    })
}

fn filter(raw: &[u8], location: bool, fields: Option<&FxHashSet<String>>) -> Vec<u8> {
    if raw.len() > 16 * 1024 * 1024 {
        return Vec::new();
    }
    let Ok(text) = std::str::from_utf8(raw) else { return Vec::new() };
    if raw.windows(9).any(|value| value.eq_ignore_ascii_case(b"<!DOCTYPE")) || raw.windows(8).any(|value| value.eq_ignore_ascii_case(b"<!ENTITY")) {
        return Vec::new();
    }
    let Ok(document) = roxmltree::Document::parse(text) else { return Vec::new() };
    let root = document.root_element();
    let mut removed = Vec::new();
    let start = if raw.starts_with(b"\xef\xbb\xbf") { 3 } else { 0 };
    if text[start..].starts_with("<?xml") {
        if let Some(end) = text[start..].find("?>") {
            removed.push(0..start + end + 2);
        }
    } else if start != 0 {
        removed.push(0..start);
    }
    for node in document.descendants() {
        if node.is_comment() || node.is_pi() {
            removed.push(node.range());
        } else if node.is_element() {
            let tag = node.tag_name();
            if node != root && !keep(tag.namespace(), tag.name(), location, fields) {
                let mut range = node.range();
                let mut following = node.next_sibling();
                while let Some(tail) = following.filter(|tail| !tail.is_element()) {
                    range.end = tail.range().end;
                    following = tail.next_sibling();
                }
                removed.push(range);
            } else {
                for attribute in node.attributes() {
                    if !keep(attribute.namespace(), attribute.name(), location, fields) {
                        removed.push(attribute.range());
                    }
                }
            }
        }
    }
    removed.sort_unstable_by_key(|range| range.start);
    let mut output = Vec::with_capacity(raw.len());
    let mut position = 0;
    for range in removed {
        if range.start > position {
            output.extend_from_slice(&raw[position..range.start]);
        }
        position = position.max(range.end);
    }
    output.extend_from_slice(&raw[position..]);
    output
}

#[pyfunction]
pub(crate) fn _filter_xmp<'py>(py: Python<'py>, raw: &[u8], location: bool, fields: Option<FxHashSet<String>>) -> Bound<'py, PyBytes> {
    let output = py.detach(|| filter(raw, location, fields.as_ref()));
    PyBytes::new(py, &output)
}
