"""Exercise OpenEXR through the public imaging API."""

from __future__ import annotations

import struct
from io import BytesIO
from typing import TYPE_CHECKING

import pytest

from blanket import Image, UnidentifiedImageError, features
from blanket.errors import CorruptImageError, UnsupportedCodecError

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("mode", ["RGB;F", "RGBA;F"])
@pytest.mark.parametrize("sample_type", ["half", "float"])
def test_openexr_roundtrip(tmp_path: Path, mode: str, sample_type: str) -> None:
    values = (-1.25, 4.0, 0.123456)
    if mode == "RGBA;F":
        values += (0.5,)
    source = Image.frombytes(mode, (19, 17), struct.pack(f"<{len(values)}f", *values) * (19 * 17))
    path = tmp_path / "plate.EXR"
    if sample_type == "half":
        source.save(path, exr_sample_type="half")
    else:
        source.save(path)
    encoded = path.read_bytes()
    channel_list = encoded.index(b"channels\0chlist\0") + len(b"channels\0chlist\0") + 4
    names = b"BGR" if mode == "RGB;F" else b"ABGR"
    for index, name in enumerate(names):
        offset = channel_list + 18 * index
        assert encoded[offset : offset + 2] == bytes((name, 0))
        assert struct.unpack_from("<i", encoded, offset + 2)[0] == (1 if sample_type == "half" else 2)

    header = Image.inspect(path, formats=["OpenEXR"])
    assert (header.format, header.mode, header.size, header.bit_depth) == ("EXR", mode, source.size, 32)
    with Image.open(path) as actual:
        assert (actual.format, actual.mode, actual.size, actual.bit_depth) == ("EXR", mode, source.size, 32)
        assert actual.get_format_mimetype() == "image/x-exr"
        if sample_type == "float":
            assert actual.tobytes() == source.tobytes()
        else:
            expected = tuple(struct.unpack("<e", struct.pack("<e", value))[0] for value in values)
            assert actual.getpixel((18, 16)) == expected
    with pytest.raises(UnidentifiedImageError):
        Image.open(path, formats=["PNG"])


def test_openexr_mixed_channels_and_offset_window() -> None:
    def attribute(name: bytes, kind: bytes, value: bytes) -> bytes:
        return name + b"\0" + kind + b"\0" + struct.pack("<I", len(value)) + value

    channels = b"".join(bytes((name, 0)) + struct.pack("<iB3xii", sample_type, 0, 1, 1) for name, sample_type in zip(b"ABGRZ", (1, 2, 1, 2, 2), strict=True))
    header = b"".join(
        (
            b"\x76\x2f\x31\x01\x02\0\0\0",
            attribute(b"channels", b"chlist", channels + b"\0"),
            attribute(b"compression", b"compression", b"\0"),
            attribute(b"dataWindow", b"box2i", struct.pack("<4i", -2, 3, -1, 3)),
            attribute(b"displayWindow", b"box2i", struct.pack("<4i", -5, 0, 8, 8)),
            attribute(b"lineOrder", b"lineOrder", b"\0"),
            attribute(b"pixelAspectRatio", b"float", struct.pack("<f", 1.0)),
            attribute(b"screenWindowCenter", b"v2f", struct.pack("<2f", 0.0, 0.0)),
            attribute(b"screenWindowWidth", b"float", struct.pack("<f", 1.0)),
            b"\0",
        )
    )
    pixels = struct.pack("<2e2f2e4f", 0.5, 0.0, 1.5, 2.0, -1.0, 4.0, 2.0, 8.0, 15.0, 16.0)
    encoded = header + struct.pack("<Qii", len(header) + 8, 3, len(pixels)) + pixels
    source = BytesIO(encoded)
    assert Image.inspect(source).size == (2, 1)
    assert source.tell() == 0
    actual = Image.open(source)
    assert actual.getpixel((0, 0)) == (4.0, -2.0, 3.0, 0.5)
    assert actual.getpixel((1, 0)) == (8.0, 4.0, 2.0, 0.0)
    assert Image.decode(actual.encode("OpenEXR")).tobytes() == actual.tobytes()
    with pytest.raises(CorruptImageError) as error:
        Image.decode(encoded[:-1])
    assert error.value.format == "EXR"


def test_openexr_capabilities_and_save_validation() -> None:
    assert Image.registered_extensions()[".exr"] == "EXR"
    assert features.check("openexr")
    assert "exr" in features.get_supported()
    capabilities = features.encoding_capabilities("OpenEXR")
    assert capabilities is not None
    assert dict(capabilities.mode_depths) == {"RGB;F": (32,), "RGBA;F": (32,)}
    source = Image.new("RGB;F", (1, 1))
    with pytest.raises(ValueError, match="exr_sample_type"):
        source.encode("EXR", exr_sample_type="uint")
    with pytest.raises(TypeError, match="exr_sample_type"):
        source.encode("EXR", exr_sample_type=16)
    with pytest.raises(UnsupportedCodecError, match="cannot write mode"):
        Image.new("RGB", (1, 1)).encode("EXR")
