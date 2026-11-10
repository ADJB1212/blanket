from __future__ import annotations

from io import BytesIO
from typing import TYPE_CHECKING

import pytest
from helpers import assert_raises, image_buffer
from PIL import PdfParser

from blanket import Image, UnidentifiedImageError
from blanket._blanket import _encode, _encode_sequence

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(("mode", "channels"), [("L", 1), ("RGB", 3), ("RGBA", 4), ("CMYK", 4)])
@pytest.mark.parametrize("destination", ["stream", "path", "bytes_path"])
@pytest.mark.parametrize("dpi", [None, 300, 96.123456789, 1e10, 1e40, 1e-40])
def test_pdf_save(mode: str, channels: int, destination: str, dpi: float | None, tmp_path: Path) -> None:
    pixels = bytes((i * 37) % 256 for i in range(6 * channels))
    source = Image.frombytes(mode, (2, 3), pixels)
    options = {} if dpi is None else {"dpi": dpi}
    if destination == "stream":
        output = image_buffer(source, format="pdf", **options)
        assert not output.closed
        data = output.getvalue()
    else:
        path = tmp_path / "image.PDF"
        source.save(bytes(path) if destination == "bytes_path" else path, **options)
        data = path.read_bytes()
    # Pillow's independent PDF parser validates the trailer, xref offsets,
    # page tree, indirect references, and binary stream lengths.
    pdf = PdfParser.PdfParser(buf=data)
    assert len(pdf.pages) == 1
    page = pdf.read_indirect(pdf.pages[0])
    dimensions = [2 * 72 / (dpi or 72), 3 * 72 / (dpi or 72)]
    assert page[b"MediaBox"] == pytest.approx([0, 0, *dimensions], abs=0)
    content = pdf.read_indirect(page[b"Contents"]).decode()
    matrix = content.splitlines()[1].split()
    assert list(map(float, matrix[:6])) == pytest.approx([dimensions[0], 0, 0, dimensions[1], 0, 0], abs=0)
    assert matrix[6] == b"cm"
    assert b"/Im0 Do" in content
    raster = pdf.read_indirect(page[b"Resources"][b"XObject"][b"Im0"])
    assert raster.dictionary[b"Width"] == 2
    assert raster.dictionary[b"Height"] == 3
    assert raster.dictionary[b"BitsPerComponent"] == 8
    assert raster.dictionary[b"ColorSpace"] == {"L": b"DeviceGray", "CMYK": b"DeviceCMYK"}.get(mode, b"DeviceRGB")
    if mode == "RGBA":
        assert raster.decode() == bytes(v for i, v in enumerate(pixels) if i % 4 != 3)
        mask = pdf.read_indirect(raster.dictionary[b"SMask"])
        assert mask.dictionary[b"ColorSpace"] == b"DeviceGray"
        assert mask.decode() == pixels[3::4]
    else:
        assert raster.decode() == pixels
        assert b"SMask" not in raster.dictionary
    assert source.tobytes() == pixels
    assert_raises(UnidentifiedImageError, lambda: Image.open(BytesIO(data)))


@pytest.mark.parametrize("save_all", [False, True])
@pytest.mark.parametrize(
    ("dpi", "error"),
    [
        (0, ValueError),
        (-72, ValueError),
        (float("nan"), ValueError),
        (float("inf"), ValueError),
        (-float("inf"), ValueError),
        (10**400, ValueError),
        (5e-324, ValueError),
        (True, TypeError),
        (None, TypeError),
        ("300", TypeError),
        ((300, 300), TypeError),
    ],
)
def test_pdf_rejects_invalid_dpi(dpi: object, error: type[Exception], save_all: bool, tmp_path: Path) -> None:
    source = Image.new("RGB", (2, 3))
    path = tmp_path / "image.pdf"
    path.write_bytes(b"unchanged")
    with pytest.raises(error, match="dpi"):
        source.save(path, dpi=dpi, save_all=save_all)
    assert path.read_bytes() == b"unchanged"
    assert source.tell() == 0


@pytest.mark.parametrize("save_all", [False, True])
def test_pdf_encode_dpi(save_all: bool) -> None:
    source = Image.new("L", (20, 30), 42)
    pdf = PdfParser.PdfParser(buf=source.encode("PDF", dpi=144, save_all=save_all))
    page = pdf.read_indirect(pdf.pages[0])
    assert page[b"MediaBox"] == [0, 0, 10, 15]
    raster = pdf.read_indirect(page[b"Resources"][b"XObject"][b"Im0"])
    assert raster.decode() == source.tobytes()


@pytest.mark.parametrize("dpi", [0, -72, float("nan"), float("inf")])
def test_pdf_native_rejects_invalid_dpi(dpi: float) -> None:
    source = Image.new("RGB", (2, 3))
    with pytest.raises(ValueError, match="dpi must be finite and positive"):
        _encode(source._native, "PDF", 90, 6, False, 7, pdf_dpi=dpi)
    with pytest.raises(ValueError, match="dpi must be finite and positive"):
        _encode_sequence([source._native], "PDF", [0], [0], [0], None, False, 90, 6, False, pdf_dpi=dpi)


def test_pdf_rejects_unsupported_options_and_depth() -> None:
    output = BytesIO()
    assert_raises(TypeError, lambda: Image.frombytes("RGB", (1, 1), b"\0\0\0").save(output, "PDF", quality=90), match="unsupported PDF save option")
    assert_raises(ValueError, lambda: Image.frombytes("L", (1, 1), b"\x00\x01", bit_depth=16).save(output, "PDF"), match="bit_depth=8")
    assert output.getvalue() == b""


@pytest.mark.parametrize("size", [(0, 1), (1, 0)])
def test_pdf_rejects_empty_image(size: tuple[int, int]) -> None:
    assert_raises(ValueError, lambda: Image.frombytes("RGB", size, b"").save(BytesIO(), "PDF"), match="empty PDF")
