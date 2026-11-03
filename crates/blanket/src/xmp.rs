use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

fn name(namespace: Option<&str>, local: &str, strip: bool) -> String {
    match namespace.filter(|_| !strip) {
        Some(uri) => format!("{{{uri}}}{local}"),
        None => local.to_owned(),
    }
}

fn value<'py>(py: Python<'py>, node: roxmltree::Node<'_, '_>, strip: bool, depth: usize) -> PyResult<Bound<'py, PyAny>> {
    if depth > 256 {
        return Err(PyValueError::new_err("XMP nesting exceeds 256 elements"));
    }
    let result = PyDict::new(py);
    for attr in node.attributes() {
        result.set_item(name(attr.namespace(), attr.name(), strip), attr.value())?;
    }
    let mut has_children = false;
    for child in node.children().filter(roxmltree::Node::is_element) {
        has_children = true;
        let tag = child.tag_name();
        let key = name(tag.namespace(), tag.name(), strip);
        let child_value = value(py, child, strip, depth + 1)?;
        if let Some(existing) = result.get_item(&key)? {
            if let Ok(list) = existing.cast::<PyList>() {
                list.append(child_value)?;
            } else {
                result.set_item(key, PyList::new(py, [existing, child_value])?)?;
            }
        } else {
            result.set_item(key, child_value)?;
        }
    }
    if !has_children {
        let text: String = node.children().filter(|child| child.is_text()).filter_map(|child| child.text()).collect();
        if result.is_empty() {
            return if node.children().any(|child| child.is_text()) { Ok(text.into_pyobject(py)?.into_any()) } else { Ok(py.None().into_bound(py)) };
        }
        if !text.is_empty() {
            result.set_item("text", text)?;
        }
    }
    Ok(result.into_any())
}

#[pyfunction]
pub(crate) fn _getxmp<'py>(py: Python<'py>, raw: &[u8], strip_namespaces: bool) -> PyResult<Bound<'py, PyDict>> {
    let decoded;
    let text = if raw.starts_with(b"\xff\xfe") || raw.starts_with(b"\xfe\xff") || raw.starts_with(b"<\0") || raw.starts_with(b"\0<") {
        let little = raw.starts_with(b"\xff\xfe") || raw.starts_with(b"<\0");
        let bytes = if raw.starts_with(b"\xff\xfe") || raw.starts_with(b"\xfe\xff") { &raw[2..] } else { raw };
        if bytes.len() % 2 != 0 {
            return Err(PyValueError::new_err("Invalid UTF-16 XMP data"));
        }
        let units: Vec<u16> =
            bytes.as_chunks::<2>().0.iter().map(|pair| if little { u16::from_le_bytes(*pair) } else { u16::from_be_bytes(*pair) }).collect();
        decoded = String::from_utf16(&units).map_err(|error| PyValueError::new_err(error.to_string()))?;
        decoded.as_str()
    } else {
        std::str::from_utf8(raw).map_err(|error| PyValueError::new_err(error.to_string()))?
    };
    let text = text.trim_end_matches(['\0', ' ']);
    let text = if let Some(start) = text.find("<x:xmpmeta").filter(|&start| start > 0) {
        text[start..].find("</x:xmpmeta>").map_or(text, |end| &text[start..start + end + 12])
    } else {
        text
    };
    let doc = roxmltree::Document::parse(text).map_err(|error| PyValueError::new_err(error.to_string()))?;
    let root = doc.root_element();
    let tag = root.tag_name();
    let result = PyDict::new(py);
    result.set_item(name(tag.namespace(), tag.name(), strip_namespaces), value(py, root, strip_namespaces, 0)?)?;
    Ok(result)
}
