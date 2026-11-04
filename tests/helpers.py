"""Shared image construction, comparisons, and in-memory encoding for tests."""

from __future__ import annotations

import struct
import zlib
from io import BytesIO

import numpy as np
from PIL import Image as PillowImage

from blanket import Image


def image_pair(mode: str, size: tuple[int, int], data: bytes) -> tuple[Image.Image, PillowImage.Image]:
    return Image.frombytes(mode, size, data), PillowImage.frombytes(mode, size, data)


def patterned_pair(mode: str, size: tuple[int, int], seed: int = 0, *, divisor: int = 11) -> tuple[Image.Image, PillowImage.Image]:
    count = size[0] * size[1] * Image.getmodebands(mode)
    data = bytes((i * 37 + i // divisor + seed) % 256 for i in range(count))
    return image_pair(mode, size, data)


def random_pair(mode: str = "RGB", size: tuple[int, int] = (37, 29), *, seed: int = 42) -> tuple[Image.Image, PillowImage.Image]:
    data = np.random.default_rng(seed).integers(0, 256, size[0] * size[1] * Image.getmodebands(mode), dtype=np.uint8).tobytes()
    return image_pair(mode, size, data)


def assert_pixels_equal(actual: Image.Image | PillowImage.Image, expected: Image.Image | PillowImage.Image) -> None:
    assert actual.mode == expected.mode, f"mode: {actual.mode!r} != {expected.mode!r}"
    assert actual.size == expected.size, f"size: {actual.size!r} != {expected.size!r}"
    assert actual.getbands() == expected.getbands()
    assert actual.tobytes() == expected.tobytes(), "pixel buffers differ"


def assert_pillow_equal(actual: Image.Image, expected: PillowImage.Image, *, tolerance: int = 0, info: bool = False) -> None:
    assert isinstance(actual, Image.Image)
    if tolerance:
        assert actual.mode == expected.mode
        assert actual.size == expected.size
        np.testing.assert_allclose(np.frombuffer(actual.tobytes(), dtype=np.uint8), np.frombuffer(expected.tobytes(), dtype=np.uint8), rtol=0, atol=tolerance)
    else:
        assert_pixels_equal(actual, expected)
    if info:
        assert actual.info == expected.info


def encode_image(image: Image.Image | PillowImage.Image, format: str, **options: object) -> bytes:
    stream = BytesIO()
    image.save(stream, format, **options)
    return stream.getvalue()


def png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


def float_samples(image: Image.Image) -> tuple[float, ...]:
    data = image.tobytes()
    return struct.unpack(f"{len(data) // 4}f", data)
