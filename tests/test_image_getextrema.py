"""Pillow 13.0.0 contracts; see PILLOW_TESTS.md."""

from __future__ import annotations

import struct

import pytest
from pillow_helpers import PILLOW_IMAGES, hopper

from blanket import Image


def test_extrema() -> None:
    def extrema(mode: str) -> tuple[float, float] | tuple[tuple[int, int], ...]:
        return hopper(mode).getextrema()

    assert extrema("1") == (0, 255)
    assert extrema("L") == (1, 255)
    assert extrema("I") == (1, 255)
    assert extrema("F") == (1, 255)
    assert extrema("P") == (0, 225)  # fixed palette
    assert extrema("RGB") == ((0, 255), (0, 255), (0, 255))
    assert extrema("RGBA") == ((0, 255), (0, 255), (0, 255), (255, 255))
    assert extrema("CMYK") == ((0, 255), (0, 255), (0, 255), (0, 0))
    assert extrema("I;16") == (1, 255)


def test_true_16() -> None:
    with Image.open(PILLOW_IMAGES / "16_bit_noise.tif") as im:
        assert im.mode == "I;16"
        extrema = im.getextrema()
    assert extrema == (106, 285)


@pytest.mark.parametrize("mode,format,minimum,maximum", [("I", "<i", -(2**31), 2**31 - 1), ("I;16L", "<H", 0, 65535), ("I;16B", ">H", 0, 65535), ("I;16N", "=H", 0, 65535)])
@pytest.mark.parametrize("width", [1, 15, 16, 17, 35])
def test_integer_extrema_include_vector_blocks_and_tails(mode, format, minimum, maximum, width) -> None:
    values = [573] * width
    values[0] = minimum
    values[-1] = maximum
    pixels = b"".join(struct.pack(format, value) for value in values)
    image = Image.frombytes(mode, (width, 1), pixels)
    assert image.getextrema() == (min(values), max(values))
