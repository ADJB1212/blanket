from __future__ import annotations

from io import BytesIO
from pathlib import Path
import struct

import numpy as np
import pytest
import pillow_heif

from blanket import Image, UnidentifiedImageError


@pytest.mark.parametrize("mode,channels", [("L", 1), ("RGB", 3), ("RGBA", 4)])
@pytest.mark.parametrize("depth", [10, 12, 16])
def test_wide_pixels_and_png(mode: str, channels: int, depth: int) -> None:
    maximum = (1 << depth) - 1
    samples = np.array([0, 1, 255, 256, maximum // 2, maximum], dtype=np.uint16)
    samples = np.repeat(samples[:, None], channels, axis=1).reshape(2, 3, channels)
    array = samples[:, :, 0] if mode == "L" else samples
    image = Image.fromarray(array, bit_depth=depth)
    assert (image.mode, image.bit_depth, image.size) == (mode, depth, (3, 2))
    expected = maximum if channels == 1 else (maximum,) * channels
    assert image.getpixel((-1, -1)) == expected
    assert image.copy().tobytes() == image.tobytes()
    assert image.copy().bit_depth == depth
    output = BytesIO()
    image.save(output, "PNG")
    result = Image.open(output)
    assert result.bit_depth == depth
    assert result.tobytes() == image.tobytes()
    converted = image.convert("RGBA", bit_depth=8)
    assert converted.bit_depth == 8
    assert converted.getpixel((-1, -1)) == (255, 255, 255, 255)


@pytest.mark.parametrize("depth", [8, 10, 12])
@pytest.mark.parametrize("mode,channels", [("L", 1), ("RGB", 3), ("RGBA", 4)])
def test_heif_roundtrip(depth: int, mode: str, channels: int) -> None:
    maximum = (1 << depth) - 1
    array = np.full((32, 32, channels), maximum // 2, dtype=np.uint8 if depth == 8 else np.uint16)
    if channels == 4:
        array[:, :, 3] = maximum
    source = Image.fromarray(array[:, :, 0] if channels == 1 else array, bit_depth=depth)
    output = BytesIO()
    source.save(output, "HEIC", quality=100)
    loaded = Image.open(output, formats=["HEIC"])
    assert (loaded.format, loaded.bit_depth, loaded.size) == ("HEIF", depth, (32, 32))
    pixel = loaded.getpixel((16, 16))
    assert isinstance(pixel, tuple)
    assert max(abs(value - maximum // 2) for value in pixel[:3]) <= 4
    if depth > 8:
        assert pixel[0] > 255
    if channels == 4:
        assert pixel[3] == maximum
    external = pillow_heif.open_heif(output.getvalue(), convert_hdr_to_8bit=False, hdr_to_16bit=False)
    assert external.info["bit_depth"] == depth
    decoded = np.asarray(external)
    expected = np.frombuffer(loaded.tobytes(), dtype=np.uint8 if depth == 8 else "<u2").reshape(decoded.shape)
    np.testing.assert_array_equal(decoded, expected)
    with pytest.raises(UnidentifiedImageError):
        Image.open(output, formats=["PNG"])


@pytest.mark.parametrize("extension", ["heic", "heif", "HEIC"])
def test_heif_extension(tmp_path: Path, extension: str) -> None:
    path = tmp_path / f"image.{extension}"
    Image.frombytes("RGB", (16, 16), bytes([1, 2, 3] * 256)).save(path)
    assert Image.open(path).format == "HEIF"


def test_high_depth_validation_and_lifecycle() -> None:
    image = Image.frombytes("L", (2, 1), struct.pack("<HH", 256, 1023), bit_depth=10)
    assert image.getpixel((0, 0)) == 256
    assert image.convert("RGB").getpixel((1, 0)) == (1023, 1023, 1023)
    with pytest.raises(ValueError, match="range"):
        Image.frombytes("L", (1, 1), struct.pack("<H", 1024), bit_depth=10)
    with pytest.raises(ValueError):
        Image.frombytes("L", (1, 1), b"\x00", bit_depth=10)
    with pytest.raises(ValueError, match="bit_depth"):
        Image.fromarray(np.zeros((1, 1), dtype=np.uint16), bit_depth=9)
    with pytest.raises(ValueError, match="8-bit"):
        image.to_pillow()
    for format in ("JPEG", "WEBP"):
        with pytest.raises(ValueError, match="high-bit-depth"):
            image.save(BytesIO(), format)
    image.close()
    for operation in (image.load, image.copy, image.tobytes):
        with pytest.raises(ValueError, match="closed"):
            operation()


def test_big_endian_strided_uint16() -> None:
    source = np.arange(24, dtype=">u2").reshape(4, 6)[::-1, ::2]
    image = Image.fromarray(source, bit_depth=10)
    assert image.tobytes() == source.astype("<u2").tobytes()


def test_external_ten_bit_heif() -> None:
    # Independent encoder uses 16-bit input normalized to 10-bit HEVC samples.
    samples = np.linspace(0, 65535, 32 * 32 * 3, dtype=np.uint16).reshape(32, 32, 3)
    external = pillow_heif.from_bytes("RGB;16", (32, 32), samples.tobytes())
    output = BytesIO()
    external.save(output, bit_depth=10, quality=100)
    image = Image.open(output)
    expected = np.asarray(pillow_heif.open_heif(output.getvalue(), convert_hdr_to_8bit=False, hdr_to_16bit=False))
    assert image.bit_depth == 10
    np.testing.assert_array_equal(np.frombuffer(image.tobytes(), dtype="<u2").reshape(32, 32, 3), expected)
    assert np.max(expected) > 255


@pytest.mark.parametrize("format", ["TIFF", "JXL"])
def test_wide_export_uses_full_uint16_range(format: str) -> None:
    source = np.array([[0, 1, 256, 511, 1022, 1023]], dtype=np.uint16)
    image = Image.fromarray(source, bit_depth=10)
    output = BytesIO()
    image.save(output, format, **({"lossless": True} if format == "JXL" else {}))
    loaded = Image.open(output)
    assert loaded.bit_depth == 16
    assert loaded.convert("L", bit_depth=10).tobytes() == image.tobytes()


def test_wide_geometry_and_bands() -> None:
    samples = np.arange(300, 336, dtype=np.uint16).reshape(3, 4, 3)
    image = Image.fromarray(samples, bit_depth=10)
    for operation, expected in (
        (image.crop(), samples),
        (image.crop((1, 1, 3, 3)), samples[1:3, 1:3]),
        (image.transpose(Image.FLIP_LEFT_RIGHT), samples[:, ::-1]),
        (image.transpose(Image.ROTATE_90), np.rot90(samples)),
    ):
        assert operation.bit_depth == 10
        assert operation.tobytes() == expected.tobytes()
    for channel, band in enumerate(image.split()):
        assert band.bit_depth == 10
        assert band.tobytes() == samples[:, :, channel].tobytes()


@pytest.mark.parametrize("method", list(Image.Resampling))
@pytest.mark.parametrize("mode,channels", [("L", 1), ("RGB", 3), ("RGBA", 4)])
def test_wide_resize_preserves_constant_samples(method: int, mode: str, channels: int) -> None:
    samples = np.full((4, 6, channels), 713, dtype=np.uint16)
    if channels == 4:
        samples[:, :, 3] = 511
    image = Image.fromarray(samples[:, :, 0] if channels == 1 else samples, bit_depth=10)
    result = image.resize((9, 7), method)
    assert result.bit_depth == 10
    expected = 713 if channels == 1 else (713, 713, 713, 511) if channels == 4 else (713,) * 3
    assert all(result.getpixel((x, y)) == expected for y in range(7) for x in range(9))


@pytest.mark.parametrize("data", [b"", b"\x00\x00\x00\x18ftypheic", b"\x00\x00\x00\x10ftypheic\0\0\0\0"])
def test_invalid_heif(data: bytes) -> None:
    with pytest.raises(UnidentifiedImageError):
        Image.open(BytesIO(data))
