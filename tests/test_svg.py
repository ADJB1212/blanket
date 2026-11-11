from __future__ import annotations

import base64
from io import BytesIO, StringIO
from typing import TYPE_CHECKING

import pytest

from blanket import CodecResourceLimitError, CorruptImageError, Image, UnidentifiedImageError, UnsupportedCodecError, features

if TYPE_CHECKING:
    from pathlib import Path

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
    assert features.encoding_capabilities("SVG") is None
    output = StringIO()
    features.blanketinfo(output)
    assert "SVG: open\n" in output.getvalue()


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


def test_svg_save_is_explicitly_unavailable(tmp_path: Path) -> None:
    image = Image.decode(SVG)
    for output in [BytesIO(), tmp_path / "image.svg"]:
        with pytest.raises(UnsupportedCodecError, match="read-only") as caught:
            image.save(output, "SVG" if isinstance(output, BytesIO) else None)
        assert caught.value.format == "SVG"
        assert caught.value.operation == "encode"
