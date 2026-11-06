use blanket_codec_common::CodecErrorKind;
use blanket_codec_svg::{Error, RenderOptions, decode, decode_options, detect, inspect};
use blanket_core::PixelMode;

const SVG: &[u8] = b"<svg xmlns='http://www.w3.org/2000/svg' width='2' height='1'><path fill='#804020' fill-opacity='0.5' d='M0 0H1V1H0Z'/></svg>";

#[test]
fn decoded_raster_matches_inspection_and_straight_alpha() {
    let header = inspect(SVG).unwrap();
    assert_eq!((header.width, header.height, header.mode, header.bit_depth), (2, 1, "RGBA", 8));
    let image = decode(SVG).unwrap();
    assert_eq!((image.width, image.height, image.mode, image.bit_depth), (2, 1, PixelMode::Rgba, 8));
    assert_eq!(image.format.as_deref(), Some("SVG"));
    assert_eq!(image.pixel_data().unwrap(), &[128, 64, 32, 128, 0, 0, 0, 0]);
}

#[test]
fn inspect_and_decode_preserve_typed_errors() {
    for (input, kind) in [
        (b"<svg".as_slice(), CodecErrorKind::Corrupt),
        (b"<svg xmlns='http://www.w3.org/2000/svg' width='1' height='1'><script/></svg>".as_slice(), CodecErrorKind::Unsupported),
        (b"<svg xmlns='http://www.w3.org/2000/svg' width='4097' height='1'/>".as_slice(), CodecErrorKind::ResourceLimit),
    ] {
        assert_eq!(decode(input).err().unwrap().kind(), kind);
        assert_eq!(inspect(input).err().unwrap().kind(), kind);
    }
    assert_eq!(Error::Limit("test").to_string(), "SVG resource limit exceeded: test");
}

#[test]
fn detection_is_bounded_and_namespace_aware() {
    assert!(detect(SVG));
    assert!(detect(b"\xef\xbb\xbf<?xml version='1.0'?><!--x--><s:svg xmlns:s='http://www.w3.org/2000/svg'/>"));
    for data in [b"<svg".as_slice(), b"<svg/>", b"<html><svg xmlns='http://www.w3.org/2000/svg'/></html>", b"<svg xmlns='urn:other'/>"] {
        assert!(!detect(data));
    }
    let mut large = b"<!--".to_vec();
    large.extend(std::iter::repeat_n(b'x', 64 << 10));
    large.extend_from_slice(b"-->");
    large.extend_from_slice(SVG);
    assert!(!detect(&large));
}

#[test]
fn render_options_are_checked_before_allocation() {
    assert_eq!(decode_options(SVG, RenderOptions { size: Some((20, 10)), ..Default::default() }).unwrap().width, 20);
    for size in [(0, 1), (4097, 1), (4096, 4096)] {
        assert!(decode_options(SVG, RenderOptions { size: Some(size), ..Default::default() }).is_err());
    }
    for dpi in [f32::NAN, f32::INFINITY, 0.0, -1.0] {
        assert!(decode_options(SVG, RenderOptions { dpi, ..Default::default() }).is_err());
    }
}
