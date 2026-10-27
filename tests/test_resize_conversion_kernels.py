from __future__ import annotations

import random
import struct

import pytest
from PIL import Image as PillowImage

from blanket import Image


@pytest.mark.parametrize("source", ["L", "RGB", "RGBA"])
@pytest.mark.parametrize("target", ["L", "RGB", "RGBA"])
@pytest.mark.parametrize("source_depth", [8, 10, 12, 16])
@pytest.mark.parametrize("target_depth", [8, 10, 12, 16])
def test_conversion_depth_and_mode_rounding(source: str, target: str, source_depth: int, target_depth: int) -> None:
    maximum = (1 << source_depth) - 1
    target_max = (1 << target_depth) - 1
    rng = random.Random(73)
    samples = [rng.randrange(maximum + 1) for _ in range(33 * len(source))]
    samples[: len(source)] = [maximum] * len(source)
    samples[len(source) : 2 * len(source)] = [0] * len(source)
    raw = bytes(samples) if source_depth == 8 else struct.pack(f"<{len(samples)}H", *samples)
    image = Image.frombytes(source, (33, 1), raw, bit_depth=source_depth)
    expected = []
    for i in range(0, len(samples), len(source)):
        pixel = samples[i : i + len(source)]
        if target == source:
            values = pixel
        elif target == "L":
            values = [(pixel[0] * 19595 + pixel[1] * 38470 + pixel[2] * 7471 + 32768) >> 16]
        else:
            values = pixel[:3] if source != "L" else pixel * 3
            if target == "RGBA":
                values.append(maximum)
        expected.extend((value * target_max + maximum // 2) // maximum for value in values)
    encoded = bytes(expected) if target_depth == 8 else struct.pack(f"<{len(expected)}H", *expected)
    actual = image.convert(target, bit_depth=target_depth)
    assert (actual.mode, actual.bit_depth, actual.size) == (target, target_depth, (33, 1))
    assert actual.tobytes() == encoded
    assert image.tobytes() == raw


@pytest.mark.parametrize("mode", ["L", "LA", "RGB", "RGBA", "CMYK"])
@pytest.mark.parametrize("height", [1, 3, 4, 5, 15, 16, 17])
@pytest.mark.parametrize("method", [1, 2, 3, 4, 5])
def test_filtered_resize_row_batches_and_fractional_boxes(mode: str, height: int, method: int) -> None:
    size = (39, height)
    raw = random.Random(83).randbytes(size[0] * size[1] * len(mode))
    actual = Image.frombytes(mode, size, raw)
    expected = PillowImage.frombytes(mode, size, raw)
    for target in [(17, height), (67, height), (13, 7)]:
        for box in [None, (0.25, 0.0, 38.5, float(height))]:
            assert actual.resize(target, method, box=box).tobytes() == expected.resize(target, method, box=box).tobytes()
    assert actual.tobytes() == raw


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
@pytest.mark.parametrize("size", [(37, 29), (38, 26), (803, 607)])
@pytest.mark.parametrize("target", [(19, 13), (91, 79), (7, 5)])
def test_nearest_gathers_and_repeated_rows(mode: str, size: tuple[int, int], target: tuple[int, int]) -> None:
    raw = random.Random(42).randbytes(size[0] * size[1] * len(mode))
    image = Image.frombytes(mode, size, raw)
    reference = PillowImage.frombytes(mode, size, raw)
    for box in (None, (1.25, 2.5, size[0] - 0.75, size[1] - 1.5)):
        actual = image.resize(target, Image.Resampling.NEAREST, box=box)
        expected = reference.resize(target, PillowImage.Resampling.NEAREST, box=box)
        assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("mode", ["RGB", "HSV", "YCbCr", "LAB"])
@pytest.mark.parametrize("size", [(1, 1), (15, 17), (16, 16), (17, 19), (803, 607)])
def test_nearest_double_rgb_layout(mode: str, size: tuple[int, int]) -> None:
    raw = random.Random(42).randbytes(size[0] * size[1] * 3)
    actual = Image.frombytes(mode, size, raw)
    expected = PillowImage.frombytes(mode, size, raw)
    target = (size[0] * 2, size[1] * 2)
    assert actual.resize(target, Image.Resampling.NEAREST).tobytes() == expected.resize(target, PillowImage.Resampling.NEAREST).tobytes()


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
@pytest.mark.parametrize("target", [(101, 77), (37, 29), (13, 11)])
@pytest.mark.parametrize("size", [(803, 607), (800, 600)])
def test_parallel_thumbnail_reduction(mode: str, target: tuple[int, int], size: tuple[int, int]) -> None:
    raw = random.Random(81).randbytes(size[0] * size[1] * len(mode))
    image = Image.frombytes(mode, size, raw)
    reference = PillowImage.frombytes(mode, size, raw)
    options = {"resample": Image.Resampling.LANCZOS, "reducing_gap": 3.0}
    assert image.resize(target, **options).tobytes() == reference.resize(target, **options).tobytes()


@pytest.mark.parametrize("source", ["L", "RGB", "RGBA"])
@pytest.mark.parametrize("target", ["L", "RGB", "RGBA"])
@pytest.mark.parametrize("count", [15, 16, 17, 262147, 2800001])
def test_conversion_vector_tails_and_parallel_chunks(source: str, target: str, count: int) -> None:
    raw = random.Random(17).randbytes(count * len(source))
    image = Image.frombytes(source, (count, 1), raw)
    reference = PillowImage.frombytes(source, (count, 1), raw)
    actual = image.convert(target)
    expected = reference.convert(target)
    assert (actual.mode, actual.size) == (expected.mode, expected.size)
    assert actual.tobytes() == expected.tobytes()
