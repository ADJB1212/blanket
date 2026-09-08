from __future__ import annotations

import random

import pytest
from blanket import Image, ImageOps
from PIL import Image as PillowImage
from PIL import ImageOps as PillowOps


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
@pytest.mark.parametrize("size", [(1, 33), (33, 1), (31, 65), (32, 64), (33, 63), (803, 607)])
def test_transpose_tile_edges(mode: str, size: tuple[int, int]) -> None:
    raw = random.Random(19).randbytes(size[0] * size[1] * len(mode))
    image = Image.frombytes(mode, size, raw)
    reference = PillowImage.frombytes(mode, size, raw)
    for method in Image.Transpose:
        actual = image.transpose(method)
        expected = reference.transpose(int(method))
        assert actual.size == expected.size
        assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
@pytest.mark.parametrize("size", [(30, 7), (32, 17), (34, 33), (800, 600), (1034, 1033)])
def test_memory_kernels(mode: str, size: tuple[int, int]) -> None:
    raw = random.Random(42).randbytes(size[0] * size[1] * len(mode))
    image = Image.frombytes(mode, size, raw)
    reference = PillowImage.frombytes(mode, size, raw)
    assert [band.tobytes() for band in image.split()] == [band.tobytes() for band in reference.split()]
    target = (size[0] // 2, max(1, size[1] // 2))
    assert image.resize(target, Image.Resampling.NEAREST).tobytes() == reference.resize(target, PillowImage.Resampling.NEAREST).tobytes()
    assert ImageOps.flip(image).tobytes() == PillowOps.flip(reference).tobytes()
    box = (1, 2, size[0] - 1, size[1] - 1)
    assert image.crop(box).tobytes() == reference.crop(box).tobytes()
    for border in ((1, 3, 5, 7), (0, 0, 0, 0)):
        fill = 37 if mode == "L" else (37, 81, 123, 149)[: len(mode)]
        assert ImageOps.expand(image, border, fill).tobytes() == PillowOps.expand(reference, border, fill).tobytes()
    for extent in ((0.5, 0.25, size[0] - 0.5, size[1] - 0.25), (size[0] - 0.5, size[1] - 0.25, 0.5, 0.25), (-2, -1, size[0] + 3, size[1] + 2)):
        assert image.transform(target, Image.EXTENT, extent).tobytes() == reference.transform(target, PillowImage.Transform.EXTENT, extent).tobytes()
