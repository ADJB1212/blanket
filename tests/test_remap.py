"""Reusable coordinate sampling and calibrated radial correction."""

from __future__ import annotations

import math
import struct
from io import BytesIO

import pytest
from helpers import image_buffer
from PIL import Image as PillowImage

from blanket import Image


def coordinate_map(points: list[tuple[float, float]], size: tuple[int, int] | None = None) -> Image.Image:
    return Image.frombytes("RGB;F", size or (len(points), 1), b"".join(struct.pack("<3f", x, y, math.nan) for x, y in points))


@pytest.mark.parametrize("mode,depth", [("L", 8), ("LA", 8), ("RGB", 8), ("RGBA", 8), ("L", 10), ("RGB", 12), ("RGBA", 16), ("I;16", 16), ("I;16L", 16), ("I;16B", 16), ("I;16N", 16), ("F", 32), ("RGB;F", 32), ("RGBA;F", 32)])
@pytest.mark.parametrize("resample", [0, 2, 3])
def test_identity_preserves_samples_and_depth(mode: str, depth: int, resample: int) -> None:
    channels = Image.getmodebands(mode)
    if depth == 32:
        raw = b"".join(struct.pack("<f", i * 0.125 - 12) for i in range(17 * 9 * channels))
    else:
        endian = "big" if mode == "I;16B" else "little"
        raw = b"".join(((i * 7919 + 129) % (1 << depth)).to_bytes(1 if depth == 8 else 2, endian) for i in range(17 * 9 * channels))
    source = Image.frombytes(mode, (17, 9), raw) if depth == 32 else Image.frombytes(mode, (17, 9), raw, bit_depth=depth)
    source.info["custom"] = "retained"
    coordinates = source.lens_map()
    map_raw = coordinates.tobytes()
    result = source.remap(coordinates, resample)
    corrected = source.lens_correct(resample=resample)
    assert result.tobytes() == corrected.tobytes() == source.tobytes() == raw
    assert result.mode == mode
    assert result.bit_depth == source.bit_depth
    assert result.info == source.info
    assert result.info is not source.info
    assert coordinates.tobytes() == map_raw
    source.close()
    coordinates.close()
    assert result.tobytes() == raw


@pytest.mark.parametrize("border,expected", [("constant", [77, 77, 10, 20, 30, 77, 77, 77]), ("clamp", [10, 10, 10, 20, 30, 30, 30, 30]), ("reflect", [20, 10, 10, 20, 30, 30, 20, 10]), ("wrap", [20, 30, 10, 20, 30, 10, 20, 30])])
def test_nearest_borders(border: str, expected: list[int]) -> None:
    source = Image.frombytes("L", (3, 1), bytes((10, 20, 30)))
    mapping = coordinate_map([(x + 0.5, 0.5) for x in range(-2, 6)])
    assert list(source.remap(mapping, border=border, fillcolor=77).tobytes()) == expected


@pytest.mark.parametrize("border,expected", [("constant", [55, 15, 65]), ("clamp", [10, 15, 30]), ("reflect", [10, 15, 30]), ("wrap", [20, 15, 20])])
def test_bilinear_border_taps(border: str, expected: list[int]) -> None:
    source = Image.frombytes("L", (3, 1), bytes((10, 20, 30)))
    mapping = coordinate_map([(0, 0.5), (1, 0.5), (3, 0.5)])
    assert list(source.remap(mapping, 2, border=border, fillcolor=100).tobytes()) == expected


@pytest.mark.parametrize("resample", [0, 2, 3])
@pytest.mark.parametrize("border", ["constant", "clamp", "reflect", "wrap"])
def test_nonfinite_and_extreme_coordinates(resample: int, border: str) -> None:
    source = Image.new("L", (1, 1), 42)
    mapping = coordinate_map([(math.nan, 0.5), (0.5, math.inf), (-math.inf, 0.5), (3e38, -3e38)])
    actual = source.remap(mapping, resample, border=border, fillcolor=77).tobytes()
    assert actual[:3] == bytes([77] * 3)
    assert actual[3] == (77 if border == "constant" else 42)


@pytest.mark.parametrize("resample", [2, 3])
@pytest.mark.parametrize("mode", ["F", "RGB;F"])
def test_interpolation_on_a_plane_retains_float_range(mode: str, resample: int) -> None:
    channels = 1 if mode == "F" else 3
    raw = b"".join(struct.pack("<f", (x + 2 * y - 15) * 0.25 + c) for y in range(7) for x in range(9) for c in range(channels))
    source = Image.frombytes(mode, (9, 7), raw)
    mapping = coordinate_map([(2.75, 3.25), (5.125, 4.75)])
    result = struct.unpack(f"<{2 * channels}f", source.remap(mapping, resample).tobytes())
    expected = [((x - 0.5) + 2 * (y - 0.5) - 15) * 0.25 + c for x, y in [(2.75, 3.25), (5.125, 4.75)] for c in range(channels)]
    assert result == pytest.approx(expected)


def test_bicubic_impulse_and_integer_saturation() -> None:
    source = Image.frombytes("F", (5, 1), struct.pack("<5f", 0, 0, 1, 0, 0))
    mapping = coordinate_map([(1, 0.5), (2, 0.5)])
    assert struct.unpack("<2f", source.remap(mapping, 3, border="clamp").tobytes()) == pytest.approx((-1 / 16, 9 / 16))
    assert struct.unpack("<2f", source.remap(mapping, 2, border="clamp").tobytes()) == pytest.approx((0, 0.5))
    integer = Image.frombytes("L", (5, 1), bytes((0, 0, 255, 0, 0)))
    assert integer.remap(mapping, 3, border="clamp").tobytes() == bytes((0, 143))


@pytest.mark.parametrize("mode,depth", [("L", 10), ("RGB", 12), ("RGBA", 16), ("I;16B", 16)])
def test_high_depth_interpolation_and_fill(mode: str, depth: int) -> None:
    channels = 1 if mode.startswith("I;") else len(mode)
    maximum = (1 << depth) - 1
    endian = "big" if mode == "I;16B" else "little"
    raw = b"".join(value.to_bytes(2, endian) for pixel in (0, maximum) for value in [pixel] * channels)
    source = Image.frombytes(mode, (2, 1), raw, bit_depth=depth)
    result = source.remap(coordinate_map([(1, 0.5), (-5, 0.5)]), 2, fillcolor=maximum, premultiply_alpha=False)
    samples = [int.from_bytes(result.tobytes()[i : i + 2], endian) for i in range(0, 4 * channels, 2)]
    assert samples == [maximum // 2 + 1] * channels + [maximum] * channels
    assert result.bit_depth == depth


@pytest.mark.parametrize("resample", [2, 3])
def test_alpha_interpolation_avoids_hidden_color_fringe(resample: int) -> None:
    source = Image.frombytes("RGBA", (2, 1), bytes((255, 0, 0, 0, 0, 0, 255, 255)))
    mapping = coordinate_map([(1, 0.5)])
    assert source.remap(mapping, resample, border="clamp").getpixel((0, 0)) == (0, 0, 255, 128)
    assert source.remap(mapping, resample, border="clamp", premultiply_alpha=False).getpixel((0, 0)) == (128, 0, 128, 128)


def test_float_alpha_interpolation_ignores_nonfinite_hidden_colors() -> None:
    source = Image.frombytes("RGBA;F", (2, 1), struct.pack("<8f", math.nan, math.inf, -math.inf, 0, 0.25, 2, -1, 1))
    result = source.remap(coordinate_map([(1, 0.5)]), 2).getpixel((0, 0))
    assert result == pytest.approx((0.25, 2, -1, 0.5))


@pytest.mark.parametrize("resample", [0, 2, 3])
def test_parallel_map_and_remap_preserve_chunk_boundaries(resample: int) -> None:
    size = (513, 257)
    raw = bytes((i * 37 + i // 19) % 256 for i in range(size[0] * size[1] * 3))
    source = Image.frombytes("RGB", size, raw)
    mapping = source.lens_map()
    for index in [4095, 4096, 4097, size[0] * size[1] - 1]:
        x, y = index % size[0], index // size[0]
        assert mapping.getpixel((x, y)) == (x + 0.5, y + 0.5, 0)
    assert source.remap(mapping, resample).tobytes() == raw


def test_map_reuse_output_dimensions_and_current_frame() -> None:
    source = Image.frombytes("L", (3, 2), bytes(range(6)))
    mapping = coordinate_map([(2.5, 0.5), (0.5, 1.5), (1.5, 1.5), (0.5, 0.5)], (2, 2))
    assert source.remap(mapping).tobytes() == bytes((2, 3, 4, 0))
    assert source.point(lambda v: v + 10).remap(mapping).tobytes() == bytes((12, 13, 14, 10))
    stream = BytesIO()
    first = PillowImage.frombytes("L", source.size, source.tobytes())
    second = PillowImage.frombytes("L", source.size, source.point(lambda v: v + 10).tobytes())
    first.save(stream, "TIFF", save_all=True, append_images=[second])
    with Image.open(BytesIO(stream.getvalue())) as sequence:
        sequence.seek(1)
        sequence.info["custom"] = "second"
        result = sequence.remap(mapping)
        assert sequence.tell() == 1
        assert result.tobytes() == bytes((12, 13, 14, 10))
        assert result.info["custom"] == "second"


def test_radial_map_matches_calibrated_equation() -> None:
    source = Image.new("L", (9, 7))
    mapping = source.lens_map(k1=-0.2, k2=0.03, k3=0.005, center=(3.25, 2.75), focal=(4, 3))
    assert mapping.mode == "RGB;F"
    for x, y in [(0, 0), (4, 3), (8, 6)]:
        u, v = (x + 0.5 - 3.25) / 4, (y + 0.5 - 2.75) / 3
        r2 = u * u + v * v
        scale = 1 - 0.2 * r2 + 0.03 * r2**2 + 0.005 * r2**3
        assert mapping.getpixel((x, y)) == pytest.approx((3.25 + 4 * u * scale, 2.75 + 3 * v * scale, 0))


def test_zero_coefficients_remain_identity_with_extreme_calibration() -> None:
    source = Image.frombytes("L", (3, 2), bytes(range(6)))
    mapping = source.lens_map(center=(1e300, -1e300), focal=(1e-300, 1e-300))
    assert source.remap(mapping).tobytes() == source.tobytes()


def test_empty_geometry() -> None:
    empty = Image.new("L", (0, 1))
    with pytest.raises(ValueError, match="positive"):
        empty.lens_map()
    source = Image.new("L", (2, 2))
    mapping = Image.new("RGB;F", (0, 1))
    assert source.remap(mapping).size == (0, 1)
    with pytest.raises(ValueError, match="positive"):
        empty.remap(source.lens_map())


@pytest.mark.parametrize("resample", [0, 2, 3])
def test_lens_correction_matches_explicit_map(resample: int) -> None:
    source = Image.frombytes("RGB", (33, 19), bytes((i * 37) % 256 for i in range(33 * 19 * 3)))
    mapping = source.lens_map(k1=0.3, k2=-0.01)
    actual = source.lens_correct(k1=0.3, k2=-0.01, resample=resample, border="reflect")
    assert actual.tobytes() == source.remap(mapping, resample, border="reflect").tobytes()
    assert actual.tobytes() != source.tobytes()


@pytest.mark.parametrize("options", [{"k1": math.nan}, {"k2": math.inf}, {"center": (0, math.nan)}, {"focal": (0, 1)}, {"focal": (1, -1)}])
def test_invalid_lens_parameters(options: dict) -> None:
    with pytest.raises(ValueError):
        Image.new("L", (3, 2)).lens_map(**options)


@pytest.mark.parametrize(
    "options,error",
    [
        ({"resample": 1}, ValueError),
        ({"border": "invalid"}, ValueError),
        ({"border": []}, ValueError),
        ({"fillcolor": [0, 1]}, ValueError),
        ({"fillcolor": math.nan}, ValueError),
        ({"fillcolor": 256}, ValueError),
        ({"fillcolor": "red"}, TypeError),
        ({"premultiply_alpha": 1}, TypeError),
    ],
)
def test_invalid_remap_parameters(options: dict, error: type[Exception]) -> None:
    source = Image.new("L", (3, 2))
    with pytest.raises(error):
        source.remap(source.lens_map(), **options)


@pytest.mark.parametrize("mode", ["1", "P", "PA", "I", "CMYK", "HSV"])
def test_unsupported_modes_require_explicit_conversion(mode: str) -> None:
    source = Image.new(mode, (3, 2))
    with pytest.raises(ValueError, match="convert explicitly"):
        source.remap(source.lens_map())


def test_closed_images_and_invalid_map_mode() -> None:
    source = Image.new("L", (3, 2))
    with pytest.raises(TypeError):
        source.remap(None)
    with pytest.raises(ValueError, match="RGB;F"):
        source.remap(source)
    mapping = source.lens_map()
    mapping.close()
    with pytest.raises(ValueError, match="closed"):
        source.remap(mapping)
    source.close()
    with pytest.raises(ValueError, match="closed"):
        source.lens_correct()
