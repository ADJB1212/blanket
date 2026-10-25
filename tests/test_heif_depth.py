from __future__ import annotations

import struct
from io import BytesIO
from typing import TYPE_CHECKING

import numpy as np
import pillow_heif
import pytest
from PIL import Image as PILImage

from blanket import Image, UnidentifiedImageError

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("mode,channels", [("L", 1), ("RGB", 3), ("RGBA", 4)])
@pytest.mark.parametrize("depth", [10, 12, 16])
@pytest.mark.parametrize("compress_level", [0, 1, 6, 9])
def test_streamed_wide_png_samples(mode: str, channels: int, depth: int, compress_level: int) -> None:
    values = np.random.default_rng(73).integers(0, 1 << depth, (19, 257, channels), dtype=np.uint16)
    image = Image.fromarray(values[:, :, 0] if channels == 1 else values, bit_depth=depth)
    stream = BytesIO()
    image.save(stream, "PNG", compress_level=compress_level)
    reopened = Image.open(BytesIO(stream.getvalue()))
    assert (reopened.mode, reopened.bit_depth) == (mode, depth)
    assert reopened.tobytes() == image.tobytes()
    normalized = (values.astype(np.uint32) * 65535 + ((1 << depth) - 1) // 2) // ((1 << depth) - 1)
    expected = normalized[:, :, 0] if channels == 1 else normalized >> 8
    np.testing.assert_array_equal(np.asarray(PILImage.open(BytesIO(stream.getvalue()))), expected)


@pytest.mark.parametrize("dtype", ["<u2", ">u2"])
@pytest.mark.parametrize("mode,channels", [("L", 1), ("RGB", 3), ("RGBA", 4)])
@pytest.mark.parametrize("strided", [False, True])
def test_wide_array_owns_pixels(dtype: str, mode: str, channels: int, strided: bool) -> None:
    values = np.arange(17 * 19 * channels).reshape(17, 19, channels).astype(dtype)
    values = values[:, :, 0] if channels == 1 else values
    if strided:
        values = values[::-1, ::2]
    expected = values.astype("<u2").tobytes()
    image = Image.fromarray(values, mode, bit_depth=16)
    values[:] = 0
    assert image.tobytes() == expected


@pytest.mark.parametrize(("mode", "channels"), [("L", 1), ("RGB", 3), ("RGBA", 4)])
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
@pytest.mark.parametrize(("mode", "channels"), [("L", 1), ("RGB", 3), ("RGBA", 4)])
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
    if mode == "L":
        assert external.mode.startswith("L") or external.mode.startswith("I;")
        decoded = np.repeat(decoded[:, :, None], 3, axis=2)
    expected = np.frombuffer(loaded.tobytes(), dtype=np.uint8 if depth == 8 else "<u2").reshape(decoded.shape)
    np.testing.assert_array_equal(decoded, expected)
    with pytest.raises(UnidentifiedImageError):
        Image.open(output, formats=["PNG"])


@pytest.mark.parametrize("extension", ["heic", "heif", "HEIC"])
def test_heif_extension(tmp_path: Path, extension: str) -> None:
    path = tmp_path / f"image.{extension}"
    Image.frombytes("RGB", (16, 16), bytes([1, 2, 3] * 256)).save(path)
    assert Image.open(path).format == "HEIF"


@pytest.mark.parametrize("depth", [8, 10, 12])
@pytest.mark.parametrize(("mode", "channels"), [("L", 1), ("RGB", 3), ("RGBA", 4)])
@pytest.mark.parametrize("size", [(64, 32), (67, 35), (513, 515)])
def test_heif_pixel_rows(depth: int, mode: str, channels: int, size: tuple[int, int]) -> None:
    """Exercise packed and padded rows with distinct pixels, including wide alpha."""
    width, height = size
    samples = (np.arange(width * height * channels, dtype=np.uint32) * 37 % (1 << depth)).astype(np.uint8 if depth == 8 else "<u2")
    source = Image.frombytes(mode, size, samples.tobytes(), bit_depth=depth)
    output = BytesIO()
    source.save(output, "HEIF", lossless=True)
    result = Image.open(output)
    expected = np.asarray(pillow_heif.open_heif(output.getvalue(), convert_hdr_to_8bit=False, hdr_to_16bit=False))
    if mode == "L":
        expected = np.repeat(expected[:, :, None], 3, axis=2)
    assert (result.mode, result.size, result.bit_depth) == ("RGB" if mode == "L" else mode, size, depth)
    assert result.tobytes() == expected.astype(np.uint8 if depth == 8 else "<u2").tobytes()
    if mode == "L":
        # Monochrome HEVC must retain each complete sample without color conversion.
        assert result.tobytes() == np.repeat(samples, 3).tobytes()
    elif mode == "RGBA":
        np.testing.assert_array_equal(expected[:, :, 3].ravel(), samples.reshape(-1, 4)[:, 3])


def test_heif_concurrent_codec_lifetimes() -> None:
    from concurrent.futures import ThreadPoolExecutor

    def roundtrip(value: int) -> bytes:
        source = Image.frombytes("RGB", (32, 32), bytes([value] * (32 * 32 * 3)))
        output = BytesIO()
        source.save(output, "HEIF", lossless=True)
        return Image.open(output).tobytes()

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(roundtrip, range(12)))
    assert results == [bytes([value] * (32 * 32 * 3)) for value in range(12)]


@pytest.mark.parametrize("depth", [10, 12, 16])
def test_high_depth_validation_and_lifecycle(depth: int) -> None:
    maximum = (1 << depth) - 1
    image = Image.frombytes("L", (2, 1), struct.pack("<HH", 256, maximum), bit_depth=depth)
    assert image.getpixel((0, 0)) == 256
    assert image.convert("RGB").getpixel((1, 0)) == (maximum, maximum, maximum)
    if depth < 16:
        with pytest.raises(ValueError, match="range"):
            Image.frombytes("L", (1, 1), struct.pack("<H", maximum + 1), bit_depth=depth)
    with pytest.raises(ValueError):
        Image.frombytes("L", (1, 1), b"\x00", bit_depth=depth)
    with pytest.raises(ValueError):
        Image.frombytes("L", (1, 1), bytes(4), bit_depth=depth)
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


@pytest.mark.parametrize(("mode", "channels"), [("L", 1), ("RGB", 3), ("RGBA", 4)])
@pytest.mark.parametrize("depth", [10, 12, 16])
@pytest.mark.parametrize("byteorder", ["<", ">"])
def test_wide_array_storage_and_validation(mode: str, channels: int, depth: int, byteorder: str) -> None:
    samples = (np.arange(19 * 23 * channels, dtype=np.uint32) * 37 % (1 << depth)).astype(f"{byteorder}u2")
    samples = samples.reshape((19, 23) if channels == 1 else (19, 23, channels))
    for array in (samples, samples[::-1, ::2]):
        expected = array.astype("<u2").tobytes()
        image = Image.fromarray(array, mode, bit_depth=depth)
        original = array.copy()
        array[...] = (1 << depth) - 1
        assert image.tobytes() == expected
        array[...] = original
    if depth < 16:
        samples.flat[-1] = 1 << depth
        for construct in (lambda: Image.fromarray(samples, mode, bit_depth=depth), lambda: Image.frombytes(mode, (23, 19), samples.astype("<u2").tobytes(), bit_depth=depth)):
            with pytest.raises(ValueError, match="range"):
                construct()


@pytest.mark.parametrize("source_mode", ["L", "RGB", "RGBA"])
@pytest.mark.parametrize("target_mode", ["L", "RGB", "RGBA"])
@pytest.mark.parametrize(("source_depth", "target_depth"), [(8, 10), (8, 16), (10, 8), (10, 12), (10, 16), (12, 8), (12, 10), (12, 16), (16, 8), (16, 10), (16, 12), (16, 16)])
@pytest.mark.parametrize("count", [17, 262147])
def test_wide_conversion_rounding_and_parallel_tails(source_mode: str, target_mode: str, source_depth: int, target_depth: int, count: int) -> None:
    source_max = (1 << source_depth) - 1
    target_max = (1 << target_depth) - 1
    samples = (np.arange(count * len(source_mode), dtype=np.uint64) * 37 % (source_max + 1)).reshape(count, len(source_mode))
    raw = samples.astype("u1" if source_depth == 8 else "<u2").tobytes()
    image = Image.frombytes(source_mode, (count, 1), raw, bit_depth=source_depth)
    rgb = np.repeat(samples, 3, axis=1) if source_mode == "L" else samples[:, :3]
    if target_mode == "L":
        expected = (rgb @ np.array([19595, 38470, 7471], dtype=np.uint64) + 32768) >> 16
    elif target_mode == "RGB":
        expected = rgb
    else:
        alpha = samples[:, 3:] if source_mode == "RGBA" else np.full((count, 1), source_max, dtype=np.uint64)
        expected = np.concatenate((rgb, alpha), axis=1)
    expected = ((expected * target_max + source_max // 2) // source_max).astype("u1" if target_depth == 8 else "<u2")
    result = image.convert(target_mode, bit_depth=target_depth)
    assert (result.mode, result.size, result.bit_depth) == (target_mode, image.size, target_depth)
    assert result.tobytes() == expected.tobytes()
    assert image.tobytes() == raw


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
@pytest.mark.parametrize(("mode", "channels"), [("L", 1), ("RGB", 3), ("RGBA", 4)])
def test_wide_resize_preserves_constant_samples(method: int, mode: str, channels: int) -> None:
    samples = np.full((4, 6, channels), 713, dtype=np.uint16)
    if channels == 4:
        samples[:, :, 3] = 511
    image = Image.fromarray(samples[:, :, 0] if channels == 1 else samples, bit_depth=10)
    result = image.resize((9, 7), method)
    assert result.bit_depth == 10
    expected = 713 if channels == 1 else (713, 713, 713, 511) if channels == 4 else (713,) * 3
    assert all(result.getpixel((x, y)) == expected for y in range(7) for x in range(9))


@pytest.mark.parametrize("method", list(Image.Resampling))
@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
@pytest.mark.parametrize("depth", [10, 12, 16])
def test_wide_resize_matches_independent_row_regions(method: int, mode: str, depth: int) -> None:
    samples = (np.arange(1026 * 514 * len(mode), dtype=np.uint32) * 37 % (1 << depth)).astype("<u2")
    image = Image.frombytes(mode, (1026, 514), samples.tobytes(), bit_depth=depth)
    result = image.resize((513, 257), method)
    regions = [image.resize((513, bottom - top), method, box=(0, top * 2, 1026, bottom * 2)) for top, bottom in ((0, 64), (64, 128), (128, 257))]
    assert (result.mode, result.bit_depth) == (mode, depth)
    assert result.tobytes() == b"".join(region.tobytes() for region in regions)


@pytest.mark.parametrize("data", [b"", b"\x00\x00\x00\x18ftypheic", b"\x00\x00\x00\x10ftypheic\0\0\0\0"])
def test_invalid_heif(data: bytes) -> None:
    with pytest.raises(UnidentifiedImageError):
        Image.open(BytesIO(data))
