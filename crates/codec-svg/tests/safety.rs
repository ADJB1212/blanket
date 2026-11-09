use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use blanket_codec_svg::{Document, Error, Limits, parse};
use resvg::tiny_skia;

fn svg(body: &str) -> String {
    format!("<svg xmlns='http://www.w3.org/2000/svg' width='4' height='3'>{body}</svg>")
}

fn check(xml: &str, limits: Limits) -> Result<Document, Error> {
    parse(xml.as_bytes(), limits)
}

fn error(xml: &str, limits: Limits) -> Error {
    check(xml, limits).expect_err("expected rejection")
}

fn embedded(mime: &str, data: &[u8]) -> String {
    svg(&format!("<image width='4' height='3' href='data:{mime};base64,{}'/>", STANDARD.encode(data)))
}

fn png() -> Vec<u8> {
    let mut pixels = tiny_skia::Pixmap::new(1, 1).unwrap();
    pixels.fill(tiny_skia::Color::from_rgba8(255, 0, 0, 255));
    pixels.encode_png().unwrap()
}

#[test]
fn static_geometry_gradient_and_alpha() {
    let xml = svg(
        "<defs><linearGradient id='g'><stop stop-color='red'/><stop offset='1' stop-color='blue'/></linearGradient></defs><g opacity='0.5'><path fill='url(#g)' d='M0 0H4V3H0Z'/></g>",
    );
    let document = check(&xml, Limits::default()).unwrap();
    assert_eq!(document.size(), (4, 3));
    assert!(document.render().unwrap().as_chunks::<4>().0.iter().all(|p| p[3] == 128));
}

#[test]
fn forbidden_content_and_reference_spellings() {
    for body in [
        "<script/>",
        "<animate/>",
        "<animateTransform/>",
        "<set/>",
        "<foreignObject/>",
        "<style>@import 'file:///secret';</style>",
        "<style>@font-face { src: url(https://example.org/font); }</style>",
        "<rect onclick='anything()'/>",
        "<rect style='fill:red'/>",
        "<path stroke='black' stroke-dasharray='1e-30 1e-30' d='M0 0H1000000'/>",
        "<rect fill='url(https://example.org/a)'/>",
        "<rect fill='u&#114;l(file:///secret)'/>",
        "<rect fill='u\\72l(#a)'/>",
        "<g xml:base='file:///tmp/'/>",
        "<use href='#self' id='self'/>",
        "<filter><feTurbulence/></filter>",
        "<pattern/>",
        "<mask/>",
        "<clipPath/>",
        "<text>Hello</text>",
        "<image href='file:///secret'/>",
        "<image href='../secret'/>",
        "<image href='https://example.org/a'/>",
        "<image href='//example.org/a'/>",
        "<image xmlns:xlink='http://www.w3.org/1999/xlink' xlink:href='secret'/>",
        "<g xmlns:x='urn:foreign'><x:script/></g>",
        "<image href='data:image/jpeg;base64,AA=='/>",
    ] {
        assert!(matches!(error(&svg(body), Limits::default()), Error::Unsupported(_)), "{body}");
    }
    for prefix in [
        "<!DOCTYPE svg>",
        "<!DOCTYPE svg SYSTEM 'file:///secret'>",
        "<!DOCTYPE svg [<!ENTITY e SYSTEM 'https://example.org/'>]>",
        "<?xml-stylesheet href='secret'?>",
    ] {
        assert!(matches!(error(&format!("{prefix}{}", svg("")), Limits::default()), Error::Unsupported(_)));
    }
}

#[test]
fn xml_validation_and_root() {
    for xml in ["<svg>", "<svg><g></svg>", "<svg duplicate='1' duplicate='2'/>", "<svg>&unknown;</svg>"] {
        assert!(matches!(error(xml, Limits::default()), Error::Corrupt(_)));
    }
    for xml in ["<html/>", "<svg/>", "<x:svg xmlns:x='urn:other'/>"] {
        assert!(matches!(error(xml, Limits::default()), Error::Unsupported(_)));
    }
    let xml = format!("\u{feff}<?xml version='1.0'?><!--test-->{}", svg(""));
    check(&xml, Limits::default()).unwrap();
    check("<s:svg xmlns:s='http://www.w3.org/2000/svg' viewBox='0 0 2 3'/>", Limits::default()).unwrap();
    assert_eq!(parse(&[255], Limits::default()).err(), Some(Error::Corrupt("UTF-8")));
    assert_eq!(parse(&[0x1f, 0x8b, 0], Limits::default()).err(), Some(Error::Unsupported("SVGZ")));
}

#[test]
fn input_nodes_depth_and_attributes() {
    let xml = svg("");
    check(&xml, Limits { input_bytes: xml.len(), xml_nodes: 2, xml_depth: 1, ..Limits::default() }).unwrap();
    assert_eq!(error(&xml, Limits { input_bytes: xml.len() - 1, ..Limits::default() }), Error::Limit("input bytes"));
    assert_eq!(error(&xml, Limits { xml_nodes: 1, ..Limits::default() }), Error::Limit("XML nodes"));
    assert_eq!(error(&svg("<g/>"), Limits { xml_depth: 1, ..Limits::default() }), Error::Limit("XML depth"));
    let xml = svg("<path d='M0 0H1V1H0Z'/>");
    // The namespace URI is also an attribute and must fit the same bound.
    check(&xml, Limits { attribute_bytes: 26, ..Limits::default() }).unwrap();
    assert_eq!(error(&xml, Limits { attribute_bytes: 25, ..Limits::default() }), Error::Limit("attribute bytes"));
    assert!(matches!(error(&svg(""), Limits { xml_depth: usize::MAX, ..Limits::default() }), Error::Unsupported(_)));
}

#[test]
fn intrinsic_size_and_output_limits() {
    for (attrs, expected) in [
        ("viewBox='0 0 20 10'", (20, 10)),
        ("width='40' viewBox='0 0 20 10'", (40, 20)),
        ("width='50%' height='25%' viewBox='0 0 20 12'", (10, 3)),
        ("width='1in' height='25.4mm'", (96, 96)),
        ("width='2.5' height='0.2'", (3, 1)),
    ] {
        let xml = format!("<svg xmlns='http://www.w3.org/2000/svg' {attrs}/>");
        assert_eq!(check(&xml, Limits::default()).unwrap().size(), expected);
    }
    for attrs in ["width='0' height='1'", "width='NaN' height='1'", "width='inf' height='1'", "viewBox='0 0 0 1'", "width='1e309' height='1'"] {
        assert!(matches!(error(&format!("<svg xmlns='http://www.w3.org/2000/svg' {attrs}/>"), Limits::default()), Error::Corrupt(_)));
    }
    assert_eq!(error(&svg(""), Limits { dimension: 3, ..Limits::default() }), Error::Limit("output dimensions"));
    check(&svg(""), Limits { dimension: 4, output_pixels: 12, ..Limits::default() }).unwrap();
    assert_eq!(error(&svg(""), Limits { output_pixels: 11, ..Limits::default() }), Error::Limit("output pixels"));
    assert_eq!(error(&svg(""), Limits { working_bytes: 287, ..Limits::default() }), Error::Limit("working bytes"));
    check(&svg(""), Limits { working_bytes: 288, ..Limits::default() }).unwrap();
}

#[test]
fn png_bytes_pixels_and_validation() {
    let data = png();
    let xml = embedded("image/png", &data);
    let doc = check(&xml, Limits { image_bytes: data.len(), image_pixels: 1, ..Limits::default() }).unwrap();
    assert!(doc.render().unwrap().as_chunks::<4>().0.iter().any(|p| p[0] != 0));
    assert_eq!(error(&xml, Limits { image_bytes: data.len() - 1, ..Limits::default() }), Error::Limit("image bytes"));
    assert_eq!(error(&xml, Limits { image_pixels: 0, ..Limits::default() }), Error::Limit("image pixels"));
    assert_eq!(error(&embedded("image/png", b"not a png"), Limits::default()), Error::Corrupt("PNG signature"));
    let mut corrupt = data.clone();
    corrupt[29] ^= 1;
    assert_eq!(error(&embedded("image/png", &corrupt), Limits::default()), Error::Corrupt("PNG samples"));
    let mut bomb = data.clone();
    bomb[16..20].copy_from_slice(&u32::MAX.to_be_bytes());
    assert_eq!(error(&embedded("image/png", &bomb), Limits::default()), Error::Limit("image dimensions"));
    let mut ancillary = data.clone();
    ancillary[37..41].copy_from_slice(b"iCCP");
    assert!(matches!(error(&embedded("image/png", &ancillary), Limits::default()), Error::Unsupported(_)));
    for end in 0..data.len() {
        assert!(check(&embedded("image/png", &data[..end]), Limits::default()).is_err(), "truncation {end}");
    }
    assert_eq!(error(&svg("<image href='data:image/png;base64,A'/>"), Limits::default()), Error::Corrupt("image base64"));
}

#[test]
fn nested_resources_share_budgets_and_policy() {
    let inner = svg("<rect width='4' height='3' fill='red'/>");
    let xml = embedded("image/svg+xml", inner.as_bytes());
    check(&xml, Limits { resource_depth: 1, ..Limits::default() }).unwrap().render().unwrap();
    assert_eq!(error(&xml, Limits { resource_depth: 0, ..Limits::default() }), Error::Limit("resource depth"));
    let nested = embedded("image/svg+xml", xml.as_bytes());
    assert_eq!(error(&nested, Limits { resource_depth: 1, ..Limits::default() }), Error::Limit("resource depth"));
    assert_eq!(error(&xml, Limits { input_bytes: xml.len(), ..Limits::default() }), Error::Limit("input bytes"));
    for inner in [svg("<script/>"), svg("<image href='../file'/>"), svg("<g opacity='0.5'><rect width='4' height='3'/></g>")] {
        assert!(matches!(error(&embedded("image/svg+xml", inner.as_bytes()), Limits::default()), Error::Unsupported(_)));
    }
    let image = format!("<image href='data:image/png;base64,{}'/>", STANDARD.encode(png()));
    assert_eq!(error(&svg(&image.repeat(2)), Limits { image_pixels: 1, ..Limits::default() }), Error::Limit("image pixels"));
}

#[test]
fn isolated_layers_and_reference_expansion() {
    let xml = svg("<g opacity='0.5'><rect width='100000' height='100000'/></g>");
    assert_eq!(error(&xml, Limits::default()), Error::Limit("layer bytes"));
    let xml = svg("<g opacity='0.5'><rect width='4' height='3'/></g>");
    check(&xml, Limits { working_bytes: 648, ..Limits::default() }).unwrap();
    assert_eq!(error(&xml, Limits { working_bytes: 647, ..Limits::default() }), Error::Limit("working bytes"));
    for body in ["<linearGradient id='a' href='#a'/>", "<rect fill='url(#missing)'/>", "<g id='a'/><g id='a'/>"] {
        assert!(check(&svg(body), Limits::default()).is_err());
    }
    let data = STANDARD.encode(png());
    let xml = svg(&format!("<image href='data:image/png;base64,{data}' width='4' height='3' preserveAspectRatio='xMidYMid slice'/>"));
    assert_eq!(error(&xml, Limits::default()), Error::Unsupported("generated clip, mask or filter"));
}

#[test]
fn nested_depth_boundary_and_aggregate_image_bytes() {
    let mut xml = svg("<rect width='4' height='3'/>");
    for _ in 0..4 {
        xml = embedded("image/svg+xml", xml.as_bytes())
    }
    check(&xml, Limits::default()).unwrap().render().unwrap();
    xml = embedded("image/svg+xml", xml.as_bytes());
    assert_eq!(error(&xml, Limits::default()), Error::Limit("resource depth"));
    let data = png();
    let image = format!("<image href='data:image/png;base64,{}'/>", STANDARD.encode(&data));
    check(&svg(&image.repeat(2)), Limits { image_bytes: data.len() * 2, image_pixels: 2, ..Limits::default() }).unwrap();
    assert_eq!(error(&svg(&image.repeat(2)), Limits { image_bytes: data.len() * 2 - 1, ..Limits::default() }), Error::Limit("image bytes"));
}
