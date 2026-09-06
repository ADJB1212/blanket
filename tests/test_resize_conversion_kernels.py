from __future__ import annotations

import random

import pytest
from PIL import Image as PillowImage

from blanket import Image


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


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
@pytest.mark.parametrize("target", [(101, 77), (37, 29), (13, 11)])
@pytest.mark.parametrize("size", [(803, 607), (800, 600)])
def test_parallel_thumbnail_reduction(mode: str, target: tuple[int, int], size: tuple[int, int]) -> None:
    raw = random.Random(81).randbytes(size[0] * size[1] * len(mode))
    image = Image.frombytes(mode, size, raw)
    reference = PillowImage.frombytes(mode, size, raw)
    options = dict(resample=Image.Resampling.LANCZOS, reducing_gap=3.0)
    assert image.resize(target, **options).tobytes() == reference.resize(target, **options).tobytes()


@pytest.mark.parametrize("source", ["L", "RGB", "RGBA"])
@pytest.mark.parametrize("target", ["L", "RGB", "RGBA"])
@pytest.mark.parametrize("count", [15, 16, 17, 262147, 2800001])
def test_conversion_vector_tails_and_parallel_chunks(source: str, target: str, count: int) -> None:
    raw = random.Random(17).randbytes(count * len(source))
    image = Image.frombytes(source, (count, 1), raw)
    reference = PillowImage.frombytes(source, (count, 1), raw)
    assert image.convert(target).tobytes() == reference.convert(target).tobytes()
