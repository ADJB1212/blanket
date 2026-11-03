"""Pillow 13.0.0 contracts and Blanket regressions; see PILLOW_TESTS.md."""

from __future__ import annotations

import inspect

import numpy as np
import pytest
from PIL import Image as PILImage, ImageEnhance as PILEnhance
from pillow_helpers import assert_image_equal, hopper

from blanket import Image, ImageEnhance


def test_sanity() -> None:
    # FIXME: assert_image
    # Implicit asserts no exception:
    ImageEnhance.Color(hopper()).enhance(0.5)
    ImageEnhance.Contrast(hopper()).enhance(0.5)
    ImageEnhance.Brightness(hopper()).enhance(0.5)
    ImageEnhance.Sharpness(hopper()).enhance(0.5)


def test_crash() -> None:
    # crashes on small images
    im = Image.new("RGB", (1, 1))
    ImageEnhance.Sharpness(im).enhance(0.5)


def _half_transparent_image() -> Image.Image:
    # returns an image, half transparent, half solid
    im = hopper("RGB")

    transparent = Image.new("L", im.size, 0)
    solid = Image.new("L", (im.size[0] // 2, im.size[1]), 255)
    transparent.paste(solid, (0, 0))
    im.putalpha(transparent)

    return im


def _check_alpha(im: Image.Image, original: Image.Image, op: str, amount: float) -> None:
    assert im.getbands() == original.getbands()
    assert_image_equal(im.getchannel("A"), original.getchannel("A"), f"Diff on {op}: {amount}")


@pytest.mark.parametrize("op", ("Color", "Brightness", "Contrast", "Sharpness"))
def test_alpha(op: str) -> None:
    # Issue https://github.com/python-pillow/Pillow/issues/899
    # Is alpha preserved through image enhancement?

    original = _half_transparent_image()

    for amount in [0, 0.5, 1.0]:
        _check_alpha(getattr(ImageEnhance, op)(original).enhance(amount), original, op, amount)


def pair(mode: str = "RGB", size: tuple[int, int] = (37, 29)) -> tuple[Image.Image, PILImage.Image]:
    raw = np.random.default_rng(84).integers(0, 256, size[0] * size[1] * len(mode), dtype=np.uint8).tobytes()
    return Image.frombytes(mode, size, raw), PILImage.frombytes(mode, size, raw)


def same(actual: Image.Image, expected: PILImage.Image) -> None:
    assert isinstance(actual, Image.Image)
    assert (actual.mode, actual.size) == (expected.mode, expected.size)
    np.testing.assert_allclose(np.frombuffer(actual.tobytes(), dtype=np.uint8), np.frombuffer(expected.tobytes(), dtype=np.uint8), rtol=0, atol=3)


def test_public_classes_match_pillow() -> None:
    expected = {name for name, value in vars(PILEnhance).items() if not name.startswith("_") and inspect.isclass(value) and value.__module__ == PILEnhance.__name__}
    assert set(ImageEnhance.__all__) == expected


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
@pytest.mark.parametrize("name", ImageEnhance.__all__)
def test_small_images_match_pillow(mode: str, name: str, subtests: pytest.Subtests) -> None:
    channels = len(mode)
    for size in ((0, 0), (1, 1), (1, 4), (2, 2), (3, 2), (3, 3)):
        with subtests.test(size=size):
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
def test_enhance_returns_fresh_image_with_source_metadata(name: str) -> None:
    blanket, pillow = pair("RGBA")
    blanket.info["custom"] = pillow.info["custom"] = "retained"
    actual = getattr(ImageEnhance, name)(blanket).enhance(1)
    expected = getattr(PILEnhance, name)(pillow).enhance(1)
    same(actual, expected)
    assert actual is not blanket
    assert actual.info == blanket.info
    assert actual.info is not blanket.info


@pytest.mark.parametrize("name", ImageEnhance.__all__)
def test_enhance_preserves_metadata_edits_after_construction(name: str) -> None:
    image = Image.new("RGB", (3, 3), (40, 80, 120))
    enhancer = getattr(ImageEnhance, name)(image)
    image.getexif()[305] = "edited"
    image.info["custom"] = "edited"
    result = enhancer.enhance(1.5)
    assert result.getexif()[305] == "edited"
    assert result.info["custom"] == "edited"
    result.getexif()[305] = "result"
    assert image.getexif()[305] == "edited"


@pytest.mark.parametrize("mode", ["L", "LA", "RGB", "RGBA", "CMYK"])
@pytest.mark.parametrize("size", [(0, 0), (17, 1), (16385, 1), (521, 509)])
def test_blend_initializes_output_across_vector_and_parallel_boundaries(mode: str, size: tuple[int, int]) -> None:
    first, reference = pair(mode, size)
    raw = first.tobytes()
    reversed_raw = raw[::-1]
    second = Image.frombytes(mode, size, reversed_raw)
    second_reference = PILImage.frombytes(mode, size, reversed_raw)
    actual = Image.blend(first, second, 0.5)
    assert actual.tobytes() == PILImage.blend(reference, second_reference, 0.5).tobytes()
    assert (actual.mode, actual.size, actual.bit_depth) == (mode, size, 8)
    assert first.tobytes() == raw
    assert second.tobytes() == reversed_raw
