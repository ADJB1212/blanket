from __future__ import annotations

from io import BytesIO
from pathlib import Path
import struct

import pytest
from PIL import Image as PillowImage

from blanket import Image, UnidentifiedImageError


@pytest.mark.parametrize("format", ["TIFF", "WEBP"])
@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
def test_raster_interoperability(format: str, mode: str) -> None:
    channels = {"L": 1, "RGB": 3, "RGBA": 4}[mode]
    raw = bytes((i * 17 + 31) % 256 for i in range(48 * channels))
    source = Image.frombytes(mode, (8, 6), raw)
    options = {"lossless": True} if format == "WEBP" else {}
    output = BytesIO()
    source.save(output, format, **options)
    expected_mode = "RGB" if mode == "L" and format == "WEBP" else mode
    expected = source.convert(expected_mode).tobytes()
    loaded = Image.open(output, formats=[format])
    assert (loaded.format, loaded.mode, loaded.size) == (format, expected_mode, (8, 6))
    assert loaded.tobytes() == expected
    assert PillowImage.open(output).convert(expected_mode).tobytes() == expected

    external = BytesIO()
    PillowImage.frombytes(mode, (8, 6), raw).save(external, format, **options)
    assert Image.open(external).convert(expected_mode).tobytes() == expected
    with pytest.raises(UnidentifiedImageError):
        Image.open(output, formats=["PNG"])


@pytest.mark.parametrize("extension", ["tif", "tiff", "webp"])
def test_extension_inference(tmp_path: Path, extension: str) -> None:
    path = tmp_path / f"image.{extension}"
    Image.frombytes("RGB", (2, 2), bytes([50, 100, 150] * 4)).save(path)
    assert Image.open(path).format == ("WEBP" if extension == "webp" else "TIFF")


def test_lossy_webp() -> None:
    stream = BytesIO()
    Image.frombytes("RGB", (16, 16), bytes([50, 100, 150] * 256)).save(stream, "WEBP", quality=90)
    result = Image.open(stream)
    assert result.size == (16, 16)
    assert max(abs(a - b) for a, b in zip(result.tobytes(), bytes([50, 100, 150] * 256))) < 8


@pytest.mark.parametrize("compression", ["tiff_lzw", "tiff_adobe_deflate", "packbits"])
def test_compressed_tiff(compression: str) -> None:
    source = PillowImage.frombytes("RGB", (8, 6), bytes(range(144)))
    stream = BytesIO()
    source.save(stream, "TIFF", compression=compression)
    assert Image.open(stream).tobytes() == source.tobytes()


@pytest.mark.parametrize("prefix", [b"", b"\xef\xbb\xbf \n", b'<?xml version="1.0"?><!-- test -->\n'])
@pytest.mark.parametrize("formats", [None, ["SVG"]])
def test_svg_is_unsupported(prefix: bytes, formats: list[str] | None) -> None:
    data = prefix + b'''<svg xmlns="http://www.w3.org/2000/svg" width="4" height="2">
      <rect width="2" height="2" fill="red" fill-opacity="0.5"/>
    </svg>'''
    with pytest.raises(UnidentifiedImageError, match="cannot identify image file"):
        Image.open(BytesIO(data), formats=formats)


def make_dng(cfa: bool = False) -> bytes:
    """Generate an uncompressed 16-bit LinearRaw DNG without camera assets."""
    entries: list[tuple[int, int, int, bytes]] = []

    def shorts(tag: int, *values: int) -> None:
        entries.append((tag, 3, len(values), struct.pack("<" + "H" * len(values), *values)))

    def longs(tag: int, *values: int) -> None:
        entries.append((tag, 4, len(values), struct.pack("<" + "I" * len(values), *values)))

    longs(256, 16)
    longs(257, 16)
    shorts(258, *([16] if cfa else [16, 16, 16]))
    shorts(259, 1)
    shorts(262, 32803 if cfa else 34892)
    longs(273, 0)
    shorts(277, 1 if cfa else 3)
    longs(278, 16)
    longs(279, 16 * 16 * (1 if cfa else 3) * 2)
    shorts(284, 1)
    entries.append((50706, 1, 4, bytes([1, 4, 0, 0])))
    entries.append((50707, 1, 4, bytes([1, 1, 0, 0])))
    longs(50717, *([65535] if cfa else [65535, 65535, 65535]))
    if cfa:
        shorts(33421, 2, 2)
        entries.append((33422, 1, 4, bytes([0, 1, 1, 2])))
    matrix = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    entries.append((50721, 10, 9, b"".join(struct.pack("<ii", v, 1) for v in matrix)))
    entries.append((50728, 5, 3, struct.pack("<IIIIII", 1, 1, 1, 1, 1, 1)))
    shorts(50778, 21)
    entries.sort()
    payload_offset = 8 + 2 + len(entries) * 12 + 4
    payload = bytearray()
    directory = bytearray()
    pixel_offset = payload_offset + sum(len(value) for _, _, _, value in entries if len(value) > 4)
    for tag, kind, count, value in entries:
        if tag == 273:
            value = struct.pack("<I", pixel_offset)
        directory.extend(struct.pack("<HHI", tag, kind, count))
        if len(value) > 4:
            directory.extend(struct.pack("<I", payload_offset + len(payload)))
            payload.extend(value)
        else:
            directory.extend(value.ljust(4, b"\0"))
    if cfa:
        samples = [[16000, 24000, 24000, 32000][(y % 2) * 2 + x % 2] for y in range(16) for x in range(16)]
    else:
        samples = [16000, 24000, 32000] * 256
    pixels = struct.pack("<" + "H" * len(samples), *samples)
    return b"II*\0\x08\0\0\0" + struct.pack("<H", len(entries)) + directory + bytes(4) + payload + pixels


@pytest.mark.parametrize("cfa", [False, True])
def test_dng_develops_raw_pixels(cfa: bool) -> None:
    result = Image.open(BytesIO(make_dng(cfa)), formats=["DNG"])
    assert (result.format, result.mode, result.size) == ("DNG", "RGB", (16, 16))
    assert len(result.tobytes()) == 768
    assert any(result.tobytes())
    with pytest.raises(UnidentifiedImageError):
        Image.open(BytesIO(make_dng()), formats=["TIFF"])


@pytest.mark.parametrize("data", [b"<svg", b"<html/>", b"II*\0", b"RIFF\0\0\0\0WEBP", make_dng()[:32]])
def test_invalid_new_formats(data: bytes) -> None:
    with pytest.raises(UnidentifiedImageError):
        Image.open(BytesIO(data))


def test_dng_dimensions_checked_before_decoding() -> None:
    data = bytearray(make_dng())
    struct.pack_into("<I", data, 18, 100_000_000)
    with pytest.raises(UnidentifiedImageError, match="dimensions|pixels"):
        Image.open(BytesIO(data))


@pytest.mark.parametrize("format", ["DNG", "SVG"])
def test_unsupported_save_formats(format: str) -> None:
    with pytest.raises(ValueError, match="unsupported image format"):
        Image.frombytes("L", (1, 1), b"\0").save(BytesIO(), format)
