from __future__ import annotations

import struct
import zlib
from io import BytesIO
from typing import TYPE_CHECKING

import pytest

from blanket import CodecError, CodecResourceLimitError, CorruptImageError, Image, ImageFile, MissingBackendError, UnidentifiedImageError, UnsupportedCodecError

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("operation", ["decode", "inspect"])
@pytest.mark.parametrize("data,format", [(b"\x89PNG\r\n\x1a\n", "PNG"), (b"\xff\xd8\xff", "JPEG"), (b"GIF89a", "GIF"), (b"II*\x00", "TIFF")])
def test_recognized_corrupt_input(data: bytes, format: str, operation: str) -> None:
    function = Image.open if operation == "decode" else Image.inspect
    with pytest.raises(CorruptImageError) as caught:
        function(BytesIO(data))
    assert isinstance(caught.value, CodecError)
    assert isinstance(caught.value, UnidentifiedImageError)
    assert caught.value.format == format
    assert caught.value.operation == operation


@pytest.mark.parametrize("mode,format,depth", [("RGBA", "JPEG", 8), ("LAB", "PNG", 8), ("HSV", "TIFF", 8), ("RGB", "WEBP", 16)])
def test_unsupported_encoding(mode: str, format: str, depth: int) -> None:
    image = Image.new(mode, (8, 8))
    if depth > 8:
        image = image.convert(mode, bit_depth=depth)
    with pytest.raises(UnsupportedCodecError) as caught:
        image.save(BytesIO(), format)
    assert isinstance(caught.value, ValueError)
    assert isinstance(caught.value, CodecError)
    assert caught.value.format == format
    assert caught.value.operation == "encode"


def test_missing_ghostscript(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", "")
    with pytest.raises(MissingBackendError, match="requires Ghostscript") as caught:
        Image.open(BytesIO(b"%!PS-Adobe-3.0 EPSF-3.0\n%%BoundingBox: 0 0 8 8\n"))
    assert caught.value.format == "EPS"
    assert caught.value.operation == "decode"
    assert isinstance(caught.value, UnidentifiedImageError)


@pytest.mark.parametrize("data,error", [(b"%!PS-Adobe\n", CorruptImageError), (b"\xc5\xd0\xd3\xc6", CorruptImageError), (b"%!PS-Adobe\n%%BoundingBox: 0 0 9999999 9999999\n", CodecResourceLimitError)])
def test_eps_validation_precedes_backend_lookup(monkeypatch: pytest.MonkeyPatch, data: bytes, error: type[CodecError]) -> None:
    monkeypatch.setenv("PATH", "")
    with pytest.raises(error) as caught:
        Image.open(BytesIO(data))
    assert caught.value.format == "EPS"


def test_decoded_pixel_limit() -> None:
    stream = BytesIO()
    Image.new("RGB", (1, 1)).save(stream, "PNG")
    data = bytearray(stream.getvalue())
    struct.pack_into(">II", data, 16, 20000, 20000)
    struct.pack_into(">I", data, 29, zlib.crc32(data[12:29]))
    with pytest.raises(CodecResourceLimitError) as caught:
        Image.open(BytesIO(data))
    assert isinstance(caught.value, UnidentifiedImageError)
    assert caught.value.format == "PNG"
    assert caught.value.operation == "decode"


def test_cumulative_frame_limit() -> None:
    stream = BytesIO()
    image = Image.new("RGBA", (2, 2))
    image.save(stream, "PNG", save_all=True, append_images=[Image.new("RGBA", (2, 2), "red")])
    data = bytearray(stream.getvalue())
    start = data.index(b"acTL")
    struct.pack_into(">I", data, start + 4, 0x7FFFFFFF)
    struct.pack_into(">I", data, start + 12, zlib.crc32(data[start : start + 12]))
    with pytest.raises(CodecResourceLimitError):
        Image.open(BytesIO(data))


def test_streaming_corrupt_input() -> None:
    parser = ImageFile.Parser()
    parser.feed(b"\xff\xd8\xff")
    with pytest.raises(CorruptImageError) as caught:
        parser.close()
    assert caught.value.format == "JPEG"


def test_unsupported_tiff_compression_layout() -> None:
    with pytest.raises(UnsupportedCodecError) as caught:
        Image.new("RGBA", (8, 8)).save(BytesIO(), "TIFF", compression="jpeg")
    assert caught.value.format == "TIFF"
    assert caught.value.operation == "encode"


def test_unsupported_animation() -> None:
    with pytest.raises(UnsupportedCodecError) as caught:
        Image.new("RGB", (8, 8)).save(BytesIO(), "JPEG", save_all=True)
    assert caught.value.format == "JPEG"


def test_unsupported_decoding() -> None:
    stream = BytesIO()
    Image.new("RGB", (8, 8)).save(stream, "BMP")
    data = bytearray(stream.getvalue())
    struct.pack_into("<I", data, 30, 4)
    with pytest.raises(UnsupportedCodecError) as caught:
        Image.open(BytesIO(data))
    assert caught.value.format == "BMP"
    assert caught.value.operation == "decode"


def test_invalid_compression_identifier_is_corrupt() -> None:
    stream = BytesIO()
    Image.new("RGB", (8, 8)).save(stream, "BMP")
    data = bytearray(stream.getvalue())
    struct.pack_into("<I", data, 30, 99)
    with pytest.raises(CorruptImageError):
        Image.open(BytesIO(data))


def test_non_codec_failures_keep_existing_types(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        Image.open(tmp_path / "missing.png")
    with pytest.raises(UnidentifiedImageError) as caught:
        Image.open(BytesIO(b"not an image"))
    assert not isinstance(caught.value, CodecError)
    with pytest.raises(ValueError) as caught:
        Image.new("RGB", (1, 1)).save(BytesIO(), "JPEG", quality=101)
    assert not isinstance(caught.value, CodecError)
