"""Polar geometry and integration with native coordinate sampling."""

from __future__ import annotations

import math
import struct
from io import BytesIO

import pytest
from helpers import image_buffer
from PIL import Image as PillowImage

from blanket import Image


@pytest.mark.parametrize("scaling", ["linear", "log"])
@pytest.mark.parametrize("size", [(1, 1), (7, 9), (129, 67)])
def test_polar_map_geometry(scaling: str, size: tuple[int, int]) -> None:
    source = Image.new("L", (13, 11))
    center, radius = (2.25, -0.75), 17.5
    mapping = source.polar_map(size, center=center, radius=radius, scaling=scaling)
    assert mapping.mode == "RGB;F"
    assert mapping.size == size
    for index in range(size[0] * size[1]):
        x, y = index % size[0], index // size[0]
        t = (x + 0.5) / size[0]
        r = radius * t if scaling == "linear" else math.expm1(t * math.log1p(radius))
        theta = math.tau * (y + 0.5) / size[1]
        expected = (center[0] + r * math.cos(theta), center[1] + r * math.sin(theta), 0)
        assert mapping.getpixel((x, y)) == pytest.approx(expected, abs=2e-6)


def test_defaults_and_clockwise_orientation() -> None:
    source = Image.new("L", (9, 7))
    assert source.polar_map().tobytes() == source.polar_map(source.size, center=(4.5, 3.5), radius=3.5, scaling="linear").tobytes()
    mapping = source.polar_map((1, 4), center=(4.5, 3.5), radius=2 * math.sqrt(2))
    for y, expected in enumerate([(5.5, 4.5, 0), (3.5, 4.5, 0), (3.5, 2.5, 0), (5.5, 2.5, 0)]):
        assert mapping.getpixel((0, y)) == pytest.approx(expected)


@pytest.mark.parametrize("mode,depth", [("L", 8), ("LA", 8), ("RGB", 8), ("RGBA", 8), ("L", 10), ("RGB", 12), ("RGBA", 16), ("I;16", 16), ("I;16L", 16), ("I;16B", 16), ("I;16N", 16), ("F", 32), ("RGB;F", 32), ("RGBA;F", 32)])
@pytest.mark.parametrize("scaling", ["linear", "log"])
@pytest.mark.parametrize("resample", [0, 2, 3])
@pytest.mark.parametrize("border", ["constant", "clamp", "reflect", "wrap"])
def test_polar_sampling(mode: str, depth: int, scaling: str, resample: int, border: str) -> None:
    channels = Image.getmodebands(mode)
    if depth == 32:
        raw = b"".join(struct.pack("<f", i / 17 - 2) for i in range(9 * 7 * channels))
        source = Image.frombytes(mode, (9, 7), raw)
    else:
        endian = "big" if mode == "I;16B" else "little"
        raw = b"".join(((i * 7919 + 129) % (1 << depth)).to_bytes(1 if depth == 8 else 2, endian) for i in range(9 * 7 * channels))
        source = Image.frombytes(mode, (9, 7), raw, bit_depth=depth)
    source.info["custom"] = "retained"
    size, center, radius = (11, 13), (2.25, 1.75), 15
    points = []
    for y in range(size[1]):
        theta = math.tau * (y + 0.5) / size[1]
        for x in range(size[0]):
            t = (x + 0.5) / size[0]
            r = radius * t if scaling == "linear" else math.expm1(t * math.log1p(radius))
            points.append(struct.pack("<3f", center[0] + r * math.cos(theta), center[1] + r * math.sin(theta), 0))
    mapping = Image.frombytes("RGB;F", size, b"".join(points))
    expected = source.remap(mapping, resample, border=border, fillcolor=1)
    actual = source.polar(size, resample, center=center, radius=radius, scaling=scaling, border=border, fillcolor=1)
    assert actual.size == size
    assert actual.mode == mode
    assert actual.bit_depth == source.bit_depth
    assert actual.tobytes() == expected.tobytes()
    assert actual.info == source.info
    assert actual.info is not source.info
    assert source.tobytes() == raw
    source.close()
    assert actual.tobytes() == expected.tobytes()


def test_polar_alpha_option() -> None:
    source = Image.frombytes("RGBA", (2, 1), bytes((255, 0, 0, 0, 0, 0, 255, 255)))
    options = {"center": (1.5, 0.5), "radius": 1, "border": "clamp"}
    assert source.polar((1, 1), **options).getpixel((0, 0)) == (0, 0, 255, 128)
    assert source.polar((1, 1), premultiply_alpha=False, **options).getpixel((0, 0)) == (128, 0, 128, 128)


def test_current_frame_and_map_reuse() -> None:
    first = PillowImage.new("L", (3, 3), 10)
    stream = image_buffer(first, "TIFF", save_all=True, append_images=[PillowImage.new("L", (3, 3), 20)])
    with Image.open(stream) as source:
        mapping = source.polar_map((5, 7))
        source.seek(1)
        source.info["custom"] = "second"
        result = source.polar((5, 7), border="clamp")
        assert result.tobytes() == source.remap(mapping, 2, border="clamp").tobytes() == bytes([20] * 35)
        assert result.info["custom"] == "second"
        assert source.tell() == 1


@pytest.mark.parametrize("options", [{"size": (0, 1)}, {"size": (1, 0)}, {"radius": 0}, {"radius": -1}, {"radius": math.nan}, {"radius": math.inf}, {"center": (math.nan, 0)}, {"center": (0, math.inf)}, {"scaling": "invalid"}])
def test_invalid_geometry(options: dict) -> None:
    source = Image.new("L", (3, 3))
    with pytest.raises(ValueError):
        source.polar_map(**options)
    with pytest.raises(ValueError):
        source.polar(**options)


def test_empty_closed_and_unsupported_source() -> None:
    source = Image.new("L", (0, 1))
    with pytest.raises(ValueError, match="positive"):
        source.polar_map((1, 1), radius=1)
    source.close()
    with pytest.raises(ValueError, match="closed"):
        source.polar()
    with pytest.raises(ValueError, match="convert explicitly"):
        Image.new("P", (3, 3)).polar()


@pytest.mark.parametrize("scaling", ["linear", "log"])
def test_extreme_coordinates_use_remap_fill(scaling: str) -> None:
    source = Image.new("L", (3, 3), 42)
    result = source.polar((3, 5), center=(1e300, -1e300), radius=1e300, scaling=scaling, border="wrap", fillcolor=77)
    assert result.tobytes() == bytes([77] * 15)
