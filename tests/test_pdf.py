from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest
from PIL import PdfParser

from blanket import Image, UnidentifiedImageError


@pytest.mark.parametrize("mode,channels", [("L", 1), ("RGB", 3), ("RGBA", 4)])
@pytest.mark.parametrize("destination", ["stream", "path", "bytes_path"])
def test_pdf_save(mode: str, channels: int, destination: str, tmp_path: Path) -> None:
    pixels = bytes((i * 37) % 256 for i in range(6 * channels))
    source = Image.frombytes(mode, (2, 3), pixels)
    if destination == "stream":
        output = BytesIO()
        source.save(output, format="pdf")
        assert not output.closed
        data = output.getvalue()
    else:
        path = tmp_path / "image.PDF"
        source.save(bytes(path) if destination == "bytes_path" else path)
        data = path.read_bytes()
    # Pillow's independent PDF parser validates the trailer, xref offsets,
    # page tree, indirect references, and binary stream lengths.
    pdf = PdfParser.PdfParser(buf=data)
    assert len(pdf.pages) == 1
    page = pdf.read_indirect(pdf.pages[0])
    assert page[b"MediaBox"] == [0, 0, 2, 3]
    content = pdf.read_indirect(page[b"Contents"]).decode()
    assert b"2 0 0 3 0 0 cm" in content
    assert b"/Im0 Do" in content
    raster = pdf.read_indirect(page[b"Resources"][b"XObject"][b"Im0"])
    assert raster.dictionary[b"Width"] == 2
    assert raster.dictionary[b"Height"] == 3
    assert raster.dictionary[b"BitsPerComponent"] == 8
    assert raster.dictionary[b"ColorSpace"] == (b"DeviceGray" if mode == "L" else b"DeviceRGB")
    if mode == "RGBA":
        assert raster.decode() == bytes(v for i, v in enumerate(pixels) if i % 4 != 3)
        mask = pdf.read_indirect(raster.dictionary[b"SMask"])
        assert mask.dictionary[b"ColorSpace"] == b"DeviceGray"
        assert mask.decode() == pixels[3::4]
    else:
        assert raster.decode() == pixels
        assert b"SMask" not in raster.dictionary
    assert source.tobytes() == pixels
    with pytest.raises(UnidentifiedImageError):
        Image.open(BytesIO(data))


def test_pdf_rejects_unsupported_options_and_depth() -> None:
    output = BytesIO()
    with pytest.raises(TypeError, match="unsupported PDF save option"):
        Image.frombytes("RGB", (1, 1), b"\0\0\0").save(output, "PDF", quality=90)
    with pytest.raises(ValueError, match="bit_depth=8"):
        Image.frombytes("L", (1, 1), b"\x00\x01", bit_depth=16).save(output, "PDF")
    assert output.getvalue() == b""


@pytest.mark.parametrize("size", [(0, 1), (1, 0)])
def test_pdf_rejects_empty_image(size: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match="empty PDF"):
        Image.frombytes("RGB", size, b"").save(BytesIO(), "PDF")
