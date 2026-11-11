use std::collections::BTreeMap;
use std::sync::Arc;

use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use quick_xml::Reader;
use quick_xml::events::Event;
use resvg::{tiny_skia, usvg};
use usvg::roxmltree;

const SVG_NS: &str = "http://www.w3.org/2000/svg";
const XLINK_NS: &str = "http://www.w3.org/1999/xlink";

#[derive(Clone, Copy, Debug)]
pub struct RenderOptions {
    pub size: Option<(u32, u32)>,
    pub dpi: f32,
}

impl Default for RenderOptions {
    fn default() -> Self {
        Self { size: None, dpi: 96.0 }
    }
}

#[must_use]
pub fn detect(data: &[u8]) -> bool {
    let mut reader = quick_xml::NsReader::from_reader(&data[..data.len().min(64 << 10)]);
    for _ in 0..Limits::default().xml_nodes {
        match reader.read_resolved_event() {
            Ok((namespace, Event::Start(element) | Event::Empty(element))) => {
                return element.local_name().as_ref() == "svg"
                    && matches!(namespace, quick_xml::name::ResolveResult::Bound(ns) if ns.as_ref() == SVG_NS);
            }
            Ok((_, Event::Decl(_) | Event::Comment(_) | Event::PI(_) | Event::DocType(_))) => {}
            Ok((_, Event::Text(text))) if text.as_ref().bytes().all(|b| b.is_ascii_whitespace()) => {}
            _ => return false,
        }
    }
    false
}

#[derive(Clone, Copy, Debug)]
pub struct Limits {
    pub input_bytes: usize,
    pub xml_nodes: usize,
    pub xml_depth: usize,
    pub attribute_bytes: usize,
    pub resource_depth: usize,
    pub image_bytes: usize,
    pub image_pixels: u64,
    pub dimension: u32,
    pub output_pixels: u64,
    pub working_bytes: u64,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            input_bytes: 8 << 20,
            xml_nodes: 4096,
            xml_depth: 32,
            attribute_bytes: 64 << 10,
            resource_depth: 4,
            image_bytes: 4 << 20,
            image_pixels: 4 << 20,
            dimension: 4096,
            output_pixels: 4 << 20,
            working_bytes: 128 << 20,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[must_use]
pub enum Error {
    Corrupt(&'static str),
    Unsupported(&'static str),
    Limit(&'static str),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Corrupt(message) => write!(f, "corrupt SVG: {message}"),
            Self::Unsupported(message) => write!(f, "unsupported SVG: {message}"),
            Self::Limit(message) => write!(f, "SVG resource limit exceeded: {message}"),
        }
    }
}

impl std::error::Error for Error {}

pub struct Document {
    tree: usvg::Tree,
    width: u32,
    height: u32,
}

impl std::fmt::Debug for Document {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Document").field("width", &self.width).field("height", &self.height).finish_non_exhaustive()
    }
}

impl Document {
    #[must_use]
    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    pub fn render(&self) -> Result<Vec<u8>, Error> {
        let mut pixmap = tiny_skia::Pixmap::new(self.width, self.height).ok_or(Error::Limit("output allocation"))?;
        let transform =
            tiny_skia::Transform::from_scale(self.width as f32 / self.tree.size().width(), self.height as f32 / self.tree.size().height());
        resvg::render(&self.tree, transform, &mut pixmap.as_mut());
        let mut pixels = pixmap.take();
        for pixel in pixels.as_chunks_mut::<4>().0 {
            let alpha = u32::from(pixel[3]);
            for channel in &mut pixel[..3] {
                *channel = (u32::from(*channel) * 255 + alpha / 2).checked_div(alpha).unwrap_or(0).min(255) as u8;
            }
        }
        Ok(pixels)
    }
}

struct Budget {
    limits: Limits,
    input: usize,
    nodes: usize,
    images: usize,
    pixels: u64,
    working: u64,
    options: RenderOptions,
    validate_samples: bool,
}

fn charge<T: Copy + std::ops::Add<Output = T> + std::ops::Sub<Output = T> + PartialOrd>(
    used: &mut T, amount: T, max: T, name: &'static str,
) -> Result<(), Error> {
    if amount > max || *used > max - amount {
        return Err(Error::Limit(name));
    }
    *used = *used + amount;
    Ok(())
}

pub fn parse(input: &[u8], limits: Limits) -> Result<Document, Error> {
    parse_options(input, limits, RenderOptions::default(), SampleValidation::Decode)
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum SampleValidation {
    Header,
    Decode,
}

pub(super) fn parse_options(input: &[u8], limits: Limits, options: RenderOptions, validation: SampleValidation) -> Result<Document, Error> {
    if !options.dpi.is_finite() || options.dpi <= 0.0 {
        return Err(Error::Unsupported("DPI must be finite and positive"));
    }
    let max = Limits::default();
    if limits.input_bytes > max.input_bytes
        || limits.xml_nodes > max.xml_nodes
        || limits.xml_depth > max.xml_depth
        || limits.attribute_bytes > max.attribute_bytes
        || limits.resource_depth > max.resource_depth
        || limits.image_bytes > max.image_bytes
        || limits.image_pixels > max.image_pixels
        || limits.dimension > max.dimension
        || limits.output_pixels > max.output_pixels
        || limits.working_bytes > max.working_bytes
    {
        return Err(Error::Unsupported("limits may only be lowered"));
    }
    let validate_samples = validation == SampleValidation::Decode;
    parse_inner(input, &mut Budget { limits, input: 0, nodes: 0, images: 0, pixels: 0, working: 0, options, validate_samples }, 0)
}

fn scan(text: &str, budget: &mut Budget) -> Result<(), Error> {
    let mut reader = Reader::from_str(text);
    let mut depth = 0usize;
    loop {
        let event = reader.read_event().map_err(|_| Error::Corrupt("XML token"))?;
        if matches!(event, Event::Eof) {
            break;
        }
        charge(&mut budget.nodes, 1, budget.limits.xml_nodes, "XML nodes")?;
        match event {
            Event::Start(ref element) | Event::Empty(ref element) => {
                if depth >= budget.limits.xml_depth {
                    return Err(Error::Limit("XML depth"));
                }
                for attribute in element.attributes() {
                    let attribute = attribute.map_err(|_| Error::Corrupt("XML attribute"))?;
                    if attribute.value.len() > budget.limits.attribute_bytes {
                        return Err(Error::Limit("attribute bytes"));
                    }
                }
                if matches!(event, Event::Start(_)) {
                    depth += 1;
                }
            }
            Event::End(_) => {
                depth = depth.checked_sub(1).ok_or(Error::Corrupt("XML depth"))?;
            }
            Event::DocType(_) => return Err(Error::Unsupported("DTD/entities")),
            Event::PI(_) => return Err(Error::Unsupported("processing instruction")),
            _ => {}
        }
    }
    if depth != 0 {
        return Err(Error::Corrupt("unclosed XML"));
    }
    Ok(())
}

fn viewport(root: roxmltree::Node<'_, '_>, limits: Limits, options: RenderOptions) -> Result<(u32, u32), Error> {
    let viewbox = root.attribute("viewBox").map(str::parse::<svgtypes::ViewBox>).transpose().map_err(|_| Error::Corrupt("viewBox"))?;
    if let Some(v) = viewbox
        && (![v.x, v.y, v.w, v.h].iter().all(|x| x.is_finite()) || v.w <= 0.0 || v.h <= 0.0)
    {
        return Err(Error::Corrupt("viewBox"));
    }
    let length = |name: &str, axis: Option<f64>| -> Result<Option<f64>, Error> {
        let Some(value) = root.attribute(name) else { return Ok(None) };
        let value: svgtypes::Length = value.parse().map_err(|_| Error::Corrupt("dimension"))?;
        use svgtypes::LengthUnit as U;
        let factor = match value.unit {
            U::None | U::Px => 1.0,
            U::In => f64::from(options.dpi),
            U::Cm => f64::from(options.dpi) / 2.54,
            U::Mm => f64::from(options.dpi) / 25.4,
            U::Pt => f64::from(options.dpi) / 72.0,
            U::Pc => f64::from(options.dpi) / 6.0,
            U::Percent => axis.ok_or(Error::Unsupported("unresolved percentage"))? / 100.0,
            _ => return Err(Error::Unsupported("dimension unit")),
        };
        let n = value.number * factor;
        if !n.is_finite() || n <= 0.0 {
            return Err(Error::Corrupt("dimension"));
        }
        Ok(Some(n))
    };
    let w = length("width", viewbox.map(|v| v.w))?;
    let h = length("height", viewbox.map(|v| v.h))?;
    let (w, h) = match (w, h, viewbox) {
        (Some(w), Some(h), _) => (w, h),
        (Some(w), None, Some(v)) => (w, v.h * w / v.w),
        (None, Some(h), Some(v)) => (v.w * h / v.h, h),
        (None, None, Some(v)) => (v.w, v.h),
        _ => return Err(Error::Unsupported("unresolved size")),
    };
    if !w.is_finite() || !h.is_finite() || w <= 0.0 || h <= 0.0 {
        return Err(Error::Corrupt("dimension"));
    }
    let (w, h) = options.size.map_or_else(|| (w.round().max(1.0), h.round().max(1.0)), |(w, h)| (f64::from(w), f64::from(h)));
    if w == 0.0 || h == 0.0 {
        return Err(Error::Unsupported("output size must be positive"));
    }
    if w > f64::from(limits.dimension) || h > f64::from(limits.dimension) {
        return Err(Error::Limit("output dimensions"));
    }
    let (w, h) = (w as u32, h as u32);
    if u64::from(w) * u64::from(h) > limits.output_pixels {
        return Err(Error::Limit("output pixels"));
    }
    Ok((w, h))
}

fn png_size(data: &[u8]) -> Result<(u32, u32), Error> {
    if !data.starts_with(b"\x89PNG\r\n\x1a\n") {
        return Err(Error::Corrupt("PNG signature"));
    }
    let mut pos = 8usize;
    let mut size = None;
    let mut ended = false;
    while pos < data.len() {
        let header = data.get(pos..pos + 8).ok_or(Error::Corrupt("PNG chunk"))?;
        let len = u32::from_be_bytes(header[..4].try_into().unwrap()) as usize;
        let end = pos.checked_add(12).and_then(|p| p.checked_add(len)).ok_or(Error::Corrupt("PNG chunk length"))?;
        let chunk = data.get(pos + 8..end.checked_sub(4).unwrap()).ok_or(Error::Corrupt("PNG chunk data"))?;
        if end > data.len() || ended {
            return Err(Error::Corrupt("PNG end"));
        }
        match &header[4..8] {
            b"IHDR" if pos == 8 && len == 13 => {
                let w = u32::from_be_bytes(chunk[..4].try_into().unwrap());
                let h = u32::from_be_bytes(chunk[4..8].try_into().unwrap());
                if w == 0 || h == 0 {
                    return Err(Error::Corrupt("PNG size"));
                }
                if chunk[8] != 8 || !matches!(chunk[9], 0 | 2 | 3 | 4 | 6) || chunk[10..13] != [0, 0, 0] {
                    return Err(Error::Unsupported("PNG layout"));
                }
                size = Some((w, h));
            }
            b"PLTE" if len <= 768 => {}
            b"tRNS" if len <= 256 => {}
            b"IDAT" => {}
            b"IEND" if len == 0 => ended = true,
            _ => return Err(Error::Unsupported("PNG ancillary chunk or animation")),
        }
        pos = end;
    }
    if !ended {
        return Err(Error::Corrupt("PNG missing end"));
    }
    size.ok_or(Error::Corrupt("PNG missing header"))
}

fn image(value: &str, budget: &mut Budget, depth: usize) -> Result<(Vec<u8>, usvg::ImageKind), Error> {
    let (prefix, encoded) = value.split_once(',').ok_or(Error::Unsupported("image URL"))?;
    if !matches!(prefix, "data:image/png;base64" | "data:image/svg+xml;base64") {
        return Err(Error::Unsupported("image type or external reference"));
    }
    if depth >= budget.limits.resource_depth {
        return Err(Error::Limit("resource depth"));
    }
    // Check the expansion bound before base64 allocates its destination.
    let padding = encoded.bytes().rev().take_while(|b| *b == b'=').count();
    if !encoded.len().is_multiple_of(4) || padding > 2 {
        return Err(Error::Corrupt("image base64"));
    }
    let bound = (encoded.len() / 4).checked_mul(3).and_then(|n| n.checked_sub(padding)).ok_or(Error::Corrupt("image base64"))?;
    if bound > budget.limits.image_bytes.saturating_sub(budget.images) {
        return Err(Error::Limit("image bytes"));
    }
    let data = STANDARD.decode(encoded).map_err(|_| Error::Corrupt("image base64"))?;
    charge(&mut budget.images, data.len(), budget.limits.image_bytes, "image bytes")?;
    let kind = if prefix == "data:image/svg+xml;base64" {
        usvg::ImageKind::SVG(parse_inner(&data, budget, depth + 1)?.tree)
    } else {
        let (w, h) = png_size(&data)?;
        if w > budget.limits.dimension || h > budget.limits.dimension {
            return Err(Error::Limit("image dimensions"));
        }
        let pixels = u64::from(w) * u64::from(h);
        charge(&mut budget.pixels, pixels, budget.limits.image_pixels, "image pixels")?;
        charge(&mut budget.working, pixels.checked_mul(32).ok_or(Error::Limit("image memory"))?, budget.limits.working_bytes, "working bytes")?;
        // Validate compressed samples now, after header and allocation checks.
        if budget.validate_samples {
            tiny_skia::Pixmap::decode_png(&data).map_err(|_| Error::Corrupt("PNG samples"))?;
        }
        usvg::ImageKind::PNG(Arc::new(data.clone()))
    };
    Ok((data, kind))
}

fn parse_inner(input: &[u8], budget: &mut Budget, depth: usize) -> Result<Document, Error> {
    charge(&mut budget.input, input.len(), budget.limits.input_bytes, "input bytes")?;
    if input.starts_with(&[0x1f, 0x8b]) {
        return Err(Error::Unsupported("SVGZ"));
    }
    let text = std::str::from_utf8(input).map_err(|_| Error::Corrupt("UTF-8"))?;
    scan(text, budget)?;
    let xml = roxmltree::Document::parse_with_options(
        text,
        roxmltree::ParsingOptions { allow_dtd: false, nodes_limit: budget.limits.xml_nodes.min(u32::MAX as usize) as u32, entity_resolver: None },
    )
    .map_err(|_| Error::Corrupt("XML document"))?;
    let root = xml.root_element();
    if root.tag_name().name() != "svg" || root.tag_name().namespace() != Some(SVG_NS) {
        return Err(Error::Unsupported("SVG root namespace"));
    }
    let options = RenderOptions { size: if depth == 0 { budget.options.size } else { None }, ..budget.options };
    let (width, height) = viewport(root, budget.limits, options)?;
    let mut resources = Vec::new();
    let mut ids = BTreeMap::new();
    let mut paints = Vec::new();
    for node in root.descendants().filter(|n| n.is_element()) {
        let name = node.tag_name().name();
        if node.tag_name().namespace() != Some(SVG_NS) {
            return Err(Error::Unsupported("foreign namespace"));
        }
        if node != root && name == "svg" {
            return Err(Error::Unsupported("inline nested viewport"));
        }
        if !matches!(
            name,
            "svg"
                | "g"
                | "defs"
                | "path"
                | "rect"
                | "circle"
                | "ellipse"
                | "line"
                | "polyline"
                | "polygon"
                | "linearGradient"
                | "radialGradient"
                | "stop"
                | "image"
                | "title"
                | "desc"
                | "metadata"
        ) {
            return Err(Error::Unsupported("element requires bounded handling"));
        }
        if let Some(id) = node.attribute("id")
            && ids.insert(id, name).is_some()
        {
            return Err(Error::Corrupt("duplicate id"));
        }
        for attr in node.attributes() {
            let key = attr.name();
            let value = attr.value().trim();
            if attr.namespace().is_some() && !(key == "href" && attr.namespace() == Some(XLINK_NS)) {
                return Err(Error::Unsupported("namespaced attribute"));
            }
            if key.to_ascii_lowercase().starts_with("on")
                || matches!(
                    key,
                    "style"
                        | "class"
                        | "filter"
                        | "mask"
                        | "clip-path"
                        | "marker"
                        | "marker-start"
                        | "marker-mid"
                        | "marker-end"
                        | "stroke-dasharray"
                        | "stroke-dashoffset"
                )
            {
                return Err(Error::Unsupported("active content, CSS or indirect rendering"));
            }
            if key == "href" {
                if name != "image" {
                    return Err(Error::Unsupported("reference expansion"));
                }
                if attr.value() != value {
                    return Err(Error::Unsupported("noncanonical image URI"));
                }
                resources.push(image(value, budget, depth)?);
            } else if value.contains(['\\', '@']) || value.to_ascii_lowercase().contains("url") {
                if matches!(key, "fill" | "stroke") && value.starts_with("url(#") && value.ends_with(')') {
                    paints.push(&value[5..value.len() - 1]);
                } else {
                    return Err(Error::Unsupported("resource expression"));
                }
            }
        }
        if name == "image" && node.attributes().filter(|a| a.name() == "href").count() != 1 {
            return Err(Error::Unsupported("image requires one data URI"));
        }
    }
    for id in paints {
        if !matches!(ids.get(id), Some(&"linearGradient" | &"radialGradient")) {
            return Err(Error::Unsupported("paint reference"));
        }
    }
    let options = usvg::Options {
        dpi: budget.options.dpi,
        resources_dir: None,
        image_href_resolver: usvg::ImageHrefResolver {
            resolve_string: Box::new(|_, _| None),
            resolve_data: Box::new(move |_, data, _| {
                resources.iter().find(|(bytes, _)| bytes.as_slice() == data.as_slice()).map(|(_, kind)| kind.clone())
            }),
        },
        ..usvg::Options::default()
    };
    let tree = usvg::Tree::from_xmltree(&xml, &options).map_err(|_| Error::Corrupt("SVG tree"))?;
    let scale_x = f64::from(width) / f64::from(tree.size().width());
    let scale_y = f64::from(height) / f64::from(tree.size().height());
    // Nested SVG rendering uses the parent's canvas, regardless of intrinsic size.
    let output_bytes = u64::from(width) * u64::from(height) * 4 * (budget.limits.resource_depth as u64 + 2);
    charge(&mut budget.working, output_bytes, budget.limits.working_bytes, "working bytes")?;
    preflight(tree.root(), scale_x, scale_y, budget, depth != 0)?;
    Ok(Document { tree, width, height })
}

fn preflight(group: &usvg::Group, sx: f64, sy: f64, budget: &mut Budget, nested: bool) -> Result<(), Error> {
    if group.clip_path().is_some() || group.mask().is_some() || !group.filters().is_empty() {
        return Err(Error::Unsupported("generated clip, mask or filter"));
    }
    if group.should_isolate() {
        if nested {
            return Err(Error::Unsupported("nested SVG layer"));
        }
        if let Some(rect) = group.abs_layer_bounding_box() {
            let w = (f64::from(rect.width()) * sx).ceil() + 6.0;
            let h = (f64::from(rect.height()) * sy).ceil() + 6.0;
            let bytes = w * h * 4.0;
            if !bytes.is_finite() || bytes > budget.limits.working_bytes as f64 {
                return Err(Error::Limit("layer bytes"));
            }
            charge(&mut budget.working, bytes as u64, budget.limits.working_bytes, "working bytes")?;
        }
    }
    for node in group.children() {
        if let usvg::Node::Group(child) = node {
            preflight(child, sx, sy, budget, nested)?
        }
    }
    Ok(())
}
