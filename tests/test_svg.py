from __future__ import annotations

import base64
import os
from io import BytesIO, StringIO
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

from blanket import CodecResourceLimitError, CorruptImageError, Image, ImageDraw, UnidentifiedImageError, UnsupportedCodecError, features
from blanket.Compressor import LosslessImageCompressor, LossyImageCompressor

SVG = b"<svg xmlns='http://www.w3.org/2000/svg' width='4' height='2'><rect width='2' height='2' fill='red' fill-opacity='0.5'/></svg>"


@pytest.mark.parametrize("prefix", [b"", b"\xef\xbb\xbf \n", b"<?xml version='1.0'?><!-- test -->\n"])
def test_detection_and_allowlists(prefix: bytes) -> None:
    image = Image.decode(prefix + SVG, formats=["svg"])
    assert (image.format, image.mode, image.size, image.n_frames, image.is_animated) == ("SVG", "RGBA", (4, 2), 1, False)
    assert image.getpixel((0, 0)) == (255, 0, 0, 128)
    assert image.getpixel((3, 0)) == (0, 0, 0, 0)
    with pytest.raises(UnidentifiedImageError):
        Image.decode(prefix + SVG, formats=["PNG"])
    prefixed = SVG.replace(b"<svg ", b"<s:svg ").replace(b"xmlns=", b"xmlns:s=").replace(b"</svg>", b"</s:svg>").replace(b"<rect ", b"<s:rect ")
    assert Image.decode(prefixed).tobytes() == image.tobytes()


@pytest.mark.parametrize("data", [b"<html><svg xmlns='http://www.w3.org/2000/svg'/></html>", b"<svg/>", b"<svg xmlns='urn:wrong'/>", b"<svg", b"<root/>", b"<!--" + b"x" * (64 * 1024) + b"-->" + SVG])
def test_reject_other_xml_and_bounded_detection(data: bytes) -> None:
    with pytest.raises(UnidentifiedImageError):
        Image.decode(data)


def test_input_forms_lifetime_copy_and_edits(tmp_path: Path) -> None:
    path = tmp_path / "sample.svg"
    path.write_bytes(SVG)
    for image in [Image.open(path), Image.open(bytes(path)), Image.open(BytesIO(SVG)), Image.decode(bytearray(SVG)), Image.decode(memoryview(SVG))]:
        assert image.format == "SVG"
        assert image.tell() == 0
        with pytest.raises(EOFError):
            image.seek(1)
        copied = image.copy()
        image.close()
        copied.putpixel((3, 0), (0, 255, 0, 255))
        assert Image.decode(copied.encode("PNG")).getpixel((3, 0)) == (0, 255, 0, 255)
    stream = BytesIO(SVG)
    image = Image.open(stream)
    stream.close()
    assert image.getpixel((0, 0)) == (255, 0, 0, 128)
    with pytest.raises(UnsupportedCodecError) as caught:
        Image.decode(SVG, lazy=True)
    assert caught.value.format == "SVG"

    class Nonseekable(BytesIO):
        def seek(self, *_args: object) -> int:
            raise OSError("not seekable")

    assert Image.open(Nonseekable(SVG)).tobytes() == Image.decode(SVG).tobytes()


def test_render_options_and_physical_units() -> None:
    physical = b"<svg xmlns='http://www.w3.org/2000/svg' width='1in' height='0.5in'><rect width='0.5in' height='0.5in' fill='red'/></svg>"
    assert Image.decode(physical).size == (96, 48)
    image = Image.decode(physical, svg_dpi=144)
    assert image.size == (144, 72)
    assert image.getpixel((71, 20)) == (255, 0, 0, 255)
    assert image.getpixel((73, 20)) == (0, 0, 0, 0)
    resized = Image.open(BytesIO(SVG), svg_size=(40, 20))
    assert resized.size == (40, 20)
    assert resized.getpixel((19, 0)) == (255, 0, 0, 128)
    assert resized.getpixel((20, 0)) == (0, 0, 0, 0)
    # A subpixel feature must be rasterized at the requested resolution.
    detail = b"<svg xmlns='http://www.w3.org/2000/svg' width='1' height='1'><rect width='0.1' height='1' fill='red'/></svg>"
    assert Image.decode(detail, svg_size=(100, 100)).getpixel((5, 5)) == (255, 0, 0, 255)
    assert Image.decode(detail, svg_size=(100, 100)).getpixel((20, 5)) == (0, 0, 0, 0)


def test_open_svg_applies_pixel_threshold_to_requested_raster(monkeypatch):
    source = b"<svg xmlns='http://www.w3.org/2000/svg' width='100' height='100'/>"
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 10)
    with pytest.raises(Image.DecompressionBombError):
        Image.open(BytesIO(source))
    assert Image.open(BytesIO(source), svg_size=(3, 3)).size == (3, 3)
    assert Image.decode(source, svg_size=(3, 3)).size == (3, 3)


def test_svg_trace_preserves_indexed_transparency():
    source = Image.new("P", (16, 16), 0)
    source.putpalette(bytes([255, 0, 0, 0, 0, 255]))
    source.info["transparency"] = 0
    source.paste(1, (8, 0, 16, 16))
    result = Image.decode(source.encode("SVG", svg_mode="pixel", svg_filter_speckle=0))
    assert result.getpixel((4, 4)) == (0, 0, 0, 0)
    assert result.getpixel((12, 4)) == (0, 0, 255, 255)


@pytest.mark.parametrize(
    "options,error",
    [
        ({"svg_size": (1, 1), "svg_dpi": 96}, ValueError),
        ({"svg_size": (0, 1)}, ValueError),
        ({"svg_size": (1,)}, ValueError),
        ({"svg_size": (True, 1)}, TypeError),
        ({"svg_size": (1.5, 1)}, TypeError),
        ({"svg_size": (4097, 1)}, CodecResourceLimitError),
        ({"svg_dpi": float("nan")}, ValueError),
        ({"svg_dpi": float("inf")}, ValueError),
        ({"svg_dpi": 0}, ValueError),
        ({"svg_dpi": True}, TypeError),
        ({"svg_dpi": "96"}, TypeError),
    ],
)
def test_option_validation(options: dict, error: type[Exception]) -> None:
    with pytest.raises(error):
        Image.decode(SVG, **options)


def test_options_reject_other_formats() -> None:
    png = Image.new("RGB", (2, 2)).encode("PNG")
    with pytest.raises(ValueError, match="require SVG"):
        Image.decode(png, svg_size=(4, 4))
    with pytest.raises(UnsupportedCodecError):
        Image.decode(SVG, lazy=True, svg_dpi=96)


@pytest.mark.parametrize("operation", ["decode", "inspect"])
def test_forbidden_dtd_and_resource_limits(operation: str) -> None:
    function = Image.decode if operation == "decode" else lambda data: Image.inspect(BytesIO(data))
    with pytest.raises(UnsupportedCodecError):
        function(b"<!DOCTYPE svg>" + SVG)
    with pytest.raises(CodecResourceLimitError) as caught:
        function(SVG.replace(b"width='4'", b"width='4097'"))
    assert caught.value.operation == operation


def test_inspection_and_diagnostics() -> None:
    stream = BytesIO(SVG)
    stream.seek(7)
    header = Image.inspect(stream, formats=["SVG"])
    assert stream.tell() == 7
    assert (header.format, header.size, header.mode, header.bit_depth) == ("SVG", (4, 2), "RGBA", 8)
    assert header.info == {}
    assert Image.decode(SVG).get_format_mimetype() == "image/svg+xml"
    assert Image.registered_extensions()[".svg"] == "SVG"
    assert features.check("svg") and "svg" in features.get_supported()
    assert features.version("svg") == "resvg 0.50.0"
    capabilities = features.encoding_capabilities("SVG")
    assert capabilities is not None
    assert ("RGBA", (8,)) in capabilities.mode_depths
    output = StringIO()
    features.blanketinfo(output)
    assert "SVG: open, save\n" in output.getvalue()


@pytest.mark.parametrize("body,error", [(b"<script/>", UnsupportedCodecError), (b"<image href='../file'/>", UnsupportedCodecError), (b"<g>", CorruptImageError)])
@pytest.mark.parametrize("operation", ["decode", "inspect"])
def test_structured_errors(body: bytes, error: type[Exception], operation: str) -> None:
    data = SVG.replace(b"<rect width='2' height='2' fill='red' fill-opacity='0.5'/>", body)
    with pytest.raises(error) as caught:
        if operation == "decode":
            Image.decode(data)
        else:
            Image.inspect(BytesIO(data))
    assert caught.value.format == "SVG"
    assert caught.value.operation == operation


def test_inspection_does_not_decode_embedded_png_pixels() -> None:
    png = bytearray(Image.new("RGBA", (1, 1), "red").encode("PNG"))
    # Corrupt the IHDR CRC; dimensions remain readable during inspection.
    png[29] ^= 1
    data = b"<svg xmlns='http://www.w3.org/2000/svg' width='4' height='2'><image href='data:image/png;base64," + base64.b64encode(png) + b"'/></svg>"
    assert Image.inspect(BytesIO(data)).size == (4, 2)
    with pytest.raises(CorruptImageError):
        Image.decode(data)


@pytest.mark.parametrize("mode", ["1", "L", "LA", "RGB", "RGBA", "P", "PA"])
def test_svg_trace_roundtrip_and_outputs(mode: str, tmp_path: Path) -> None:
    image = Image.new("RGB", (12, 8), "red").convert(mode)
    expected = image.convert("RGBA")
    data = image.encode("svg", svg_mode="pixel", svg_filter_speckle=0, svg_color_precision=8)
    assert b"<path " in data and b"<image" not in data
    assert Image.decode(data).tobytes() == expected.tobytes()
    for output in [BytesIO(), tmp_path / "image.svg", bytes(tmp_path / "bytes.svg")]:
        image.save(output, "SVG" if isinstance(output, BytesIO) else None, svg_mode="pixel", svg_filter_speckle=0, svg_color_precision=8)
        saved = output.getvalue() if isinstance(output, BytesIO) else Path(os.fsdecode(output)).read_bytes()
        assert saved == data


@pytest.mark.parametrize("fit", ["pixel", "polygon", "spline"])
def test_svg_trace_transparency_holes_and_paint_order(fit: str) -> None:
    image = Image.new("RGBA", (24, 24), "red")
    for y in range(6, 18):
        for x in range(6, 18):
            image.putpixel((x, y), (0, 0, 255, 255))
    for y in range(10, 14):
        for x in range(10, 14):
            image.putpixel((x, y), (255, 0, 0, 0))
    data = image.encode("SVG", svg_mode=fit, svg_filter_speckle=0, svg_color_precision=8)
    result = Image.decode(data)
    assert result.size == image.size
    assert result.getpixel((2, 2)) == (255, 0, 0, 255)
    assert result.getpixel((8, 8)) == (0, 0, 255, 255)
    assert result.getpixel((12, 12)) == (0, 0, 0, 0)
    assert data == image.encode("SVG", svg_mode=fit, svg_filter_speckle=0, svg_color_precision=8)


def test_svg_trace_alpha_policy_and_empty_document() -> None:
    image = Image.new("RGBA", (8, 8), (40, 80, 120, 127))
    assert Image.decode(image.encode("SVG")).getbbox() is None
    assert Image.decode(image.encode("SVG", svg_alpha_threshold=127)).getpixel((4, 4)) == (40, 80, 120, 255)
    image.putpixel((7, 7), (0, 255, 0, 255))
    result = Image.decode(image.encode("SVG", svg_mode="pixel", svg_filter_speckle=0))
    assert result.getpixel((7, 7)) == (0, 255, 0, 255)
    assert result.getpixel((0, 0)) == (0, 0, 0, 0)


@pytest.mark.parametrize("size", [(1, 1), (1, 17), (17, 1)])
@pytest.mark.parametrize("fit", ["pixel", "polygon", "spline"])
def test_svg_trace_small_dimensions(size: tuple[int, int], fit: str) -> None:
    image = Image.new("RGB", size, "red")
    data = image.encode("SVG", svg_mode=fit, svg_filter_speckle=0)
    result = Image.decode(data)
    assert result.size == size
    assert result.getpixel((0, 0))[0] == 255


@pytest.mark.parametrize(
    "options,error",
    [
        ({"svg_mode": "unknown"}, ValueError),
        ({"svg_mode": 1}, TypeError),
        ({"svg_color_precision": 0}, ValueError),
        ({"svg_color_precision": 9}, ValueError),
        ({"svg_filter_speckle": -1}, ValueError),
        ({"svg_filter_speckle": True}, TypeError),
        ({"svg_layer_difference": 766}, ValueError),
        ({"svg_path_precision": 9}, ValueError),
        ({"svg_alpha_threshold": 0}, ValueError),
        ({"svg_alpha_threshold": 256}, ValueError),
        ({"svg_corner_threshold": float("nan")}, ValueError),
        ({"svg_corner_threshold": 181}, ValueError),
        ({"svg_length_threshold": float("inf")}, ValueError),
        ({"svg_length_threshold": 0}, ValueError),
        ({"svg_splice_threshold": -1}, ValueError),
        ({"svg_splice_threshold": True}, TypeError),
        ({"svg_length_threshold": 10**1000}, ValueError),
        ({"quality": 90}, TypeError),
    ],
)
def test_svg_trace_option_validation(options: dict, error: type[Exception]) -> None:
    with pytest.raises(error):
        Image.new("RGB", (8, 8)).encode("SVG", **options)


def test_svg_trace_modes_limits_and_other_formats() -> None:
    with pytest.raises(TypeError, match="unsupported PNG"):
        Image.new("RGB", (8, 8)).encode("PNG", svg_mode="pixel")
    for image in [Image.new("RGB", (1, 1)).convert("RGB", bit_depth=16), Image.new("F", (1, 1)), Image.new("CMYK", (1, 1))]:
        with pytest.raises(UnsupportedCodecError) as caught:
            image.encode("SVG")
        assert caught.value.format == "SVG"
        assert caught.value.operation == "encode"
    with pytest.raises(ValueError, match="empty"):
        Image.new("RGB", (0, 0)).encode("SVG")
    with pytest.raises(CodecResourceLimitError):
        Image.new("RGB", (4097, 1)).encode("SVG")
    with pytest.raises(ValueError, match="compression optimization"):
        Image.new("RGB", (8, 8)).encode("SVG", compressor=LossyImageCompressor())
    with pytest.raises(ValueError, match="multi-frame"):
        Image.new("RGB", (8, 8)).encode("SVG", save_all=True)


@pytest.mark.parametrize("fit", ["pixel", "polygon", "spline"])
def test_svg_lossless_compressor_minifies_trace(tmp_path: Path, fit: str) -> None:
    image = Image.new("RGBA", (24, 24), "red")
    ImageDraw.Draw(image).ellipse((4, 4, 19, 19), fill="blue")
    baseline = image.encode("SVG", svg_mode=fit)
    expected = baseline.replace(b"\n", b"")
    compressor = LosslessImageCompressor(target_size=len(expected))
    encoded = image.encode("SVG", svg_mode=fit, compressor=compressor)
    assert len(encoded) < len(baseline)
    assert encoded == expected
    assert Image.decode(encoded).tobytes() == Image.decode(baseline).tobytes()

    path = tmp_path / "minified.svg"
    stream = BytesIO()
    image.save(path, svg_mode=fit, compressor=compressor)
    image.save(stream, "SVG", svg_mode=fit, compressor=compressor)
    assert path.read_bytes() == stream.getvalue() == encoded


def test_svg_trace_fitting_color_and_speckle_controls() -> None:
    image = Image.new("RGB", (32, 32), "white")
    ImageDraw.Draw(image).ellipse((4, 4, 27, 27), fill="black")
    paths = {}
    for fit in ("pixel", "polygon", "spline"):
        root = ET.fromstring(image.encode("SVG", svg_mode=fit, svg_filter_speckle=0))
        paths[fit] = "".join(element.attrib["d"] for element in root)
    assert len(paths["polygon"]) < len(paths["pixel"])
    assert any(command in paths["spline"] for command in "CcSs")
    assert not any(command in paths["pixel"] + paths["polygon"] for command in "CcSs")

    image = Image.new("RGB", (32, 16), (100, 100, 100))
    ImageDraw.Draw(image).rectangle((16, 0, 31, 15), fill=(110, 110, 110))
    coarse = image.encode("SVG", svg_color_precision=1, svg_filter_speckle=0, svg_layer_difference=0)
    detailed = image.encode("SVG", svg_color_precision=8, svg_filter_speckle=0, svg_layer_difference=0)
    coarse_image = Image.decode(coarse)
    detailed_image = Image.decode(detailed)
    assert coarse_image.getpixel((8, 8)) == coarse_image.getpixel((24, 8))
    assert detailed_image.getpixel((8, 8)) != detailed_image.getpixel((24, 8))

    image = Image.new("RGB", (32, 32), "white")
    ImageDraw.Draw(image).rectangle((10, 10, 12, 12), fill="red")
    unfiltered = image.encode("SVG", svg_filter_speckle=0, svg_layer_difference=0)
    filtered = image.encode("SVG", svg_filter_speckle=4, svg_layer_difference=0)
    assert filtered.count(b"<path") < unfiltered.count(b"<path")
