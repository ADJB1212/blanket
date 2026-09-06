from __future__ import annotations

import inspect
import math

import numpy as np
import pytest
from blanket import Image, ImageEnhance
from PIL import Image as PILImage
from PIL import ImageEnhance as PILEnhance


def pair(mode: str = "RGB", size: tuple[int, int] = (37, 29)) -> tuple[Image.Image, PILImage.Image]:
    raw = np.random.default_rng(84).integers(0, 256, size[0] * size[1] * len(mode), dtype=np.uint8).tobytes()
    return Image.frombytes(mode, size, raw), PILImage.frombytes(mode, size, raw)


def same(actual: Image.Image, expected: PILImage.Image) -> None:
    assert isinstance(actual, Image.Image)
    assert (actual.mode, actual.size) == (expected.mode, expected.size)
    assert actual.tobytes() == expected.tobytes()


def test_public_classes_match_pillow() -> None:
    expected = {
        name
        for name, value in vars(PILEnhance).items()
        if not name.startswith("_") and inspect.isclass(value) and value.__module__ == PILEnhance.__name__
    }
    assert set(ImageEnhance.__all__) == expected


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
@pytest.mark.parametrize("name", ImageEnhance.__all__)
@pytest.mark.parametrize("factor", [-2, -0.5, 0, 0.1, 0.5, 1, 1.5, 2, 10, math.nan, math.inf, -math.inf])
def test_enhancers_match_pillow(mode: str, name: str, factor: float) -> None:
    blanket, pillow = pair(mode)
    actual = getattr(ImageEnhance, name)(blanket).enhance(factor)
    expected = getattr(PILEnhance, name)(pillow).enhance(factor)
    same(actual, expected)


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
@pytest.mark.parametrize("name", ImageEnhance.__all__)
def test_degenerate_images_and_attributes_match_pillow(mode: str, name: str) -> None:
    blanket, pillow = pair(mode)
    actual = getattr(ImageEnhance, name)(blanket)
    expected = getattr(PILEnhance, name)(pillow)
    assert actual.image is blanket
    same(actual.degenerate, expected.degenerate)
    assert getattr(actual, "intermediate_mode", None) == getattr(expected, "intermediate_mode", None)
    if name == "Color" and mode == "L":
        assert actual.degenerate is blanket


@pytest.mark.parametrize("size", [(0, 0), (1, 1), (1, 4), (2, 2), (3, 2), (3, 3)])
@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
@pytest.mark.parametrize("name", ImageEnhance.__all__)
def test_small_images_match_pillow(size: tuple[int, int], mode: str, name: str) -> None:
    channels = len(mode)
    raw = bytes((i * 37 + 11) % 256 for i in range(size[0] * size[1] * channels))
    blanket = Image.frombytes(mode, size, raw)
    pillow = PILImage.frombytes(mode, size, raw)
    actual = getattr(ImageEnhance, name)(blanket)
    expected = getattr(PILEnhance, name)(pillow)
    same(actual.degenerate, expected.degenerate)
    same(actual.enhance(1.75), expected.enhance(1.75))


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
@pytest.mark.parametrize("name", ImageEnhance.__all__)
def test_parallel_enhancers_match_pillow(mode: str, name: str) -> None:
    blanket, pillow = pair(mode, (521, 509))
    actual = getattr(ImageEnhance, name)(blanket)
    expected = getattr(PILEnhance, name)(pillow)
    same(actual.degenerate, expected.degenerate)
    same(actual.enhance(1.75), expected.enhance(1.75))


@pytest.mark.parametrize("name", ImageEnhance.__all__)
def test_enhance_returns_fresh_image_with_pillow_metadata(name: str) -> None:
    blanket, pillow = pair("RGBA")
    blanket.info["custom"] = pillow.info["custom"] = "not copied"
    actual = getattr(ImageEnhance, name)(blanket).enhance(1)
    expected = getattr(PILEnhance, name)(pillow).enhance(1)
    same(actual, expected)
    assert actual is not blanket
    assert actual.info == expected.info
