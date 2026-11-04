"""Pillow 13.0.0 contracts and Blanket regressions; see PILLOW_TESTS.md."""

from __future__ import annotations

import inspect

import pytest
from helpers import assert_pillow_equal, image_pair, random_pair
from PIL import Image as PILImage, ImageEnhance as PILEnhance
from pillow_helpers import assert_image_equal, hopper

from blanket import Image, ImageEnhance


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
            blanket, pillow = image_pair(mode, size, raw)
            actual = getattr(ImageEnhance, name)(blanket)
            expected = getattr(PILEnhance, name)(pillow)
            assert_pillow_equal(actual.degenerate, expected.degenerate, tolerance=3)
            assert_pillow_equal(actual.enhance(1.75), expected.enhance(1.75), tolerance=3)


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
@pytest.mark.parametrize("name", ImageEnhance.__all__)
def test_parallel_enhancers_match_pillow(mode: str, name: str) -> None:
    blanket, pillow = random_pair(mode, (521, 509), seed=84)
    actual = getattr(ImageEnhance, name)(blanket)
    expected = getattr(PILEnhance, name)(pillow)
    assert_pillow_equal(actual.degenerate, expected.degenerate, tolerance=3)
    assert_pillow_equal(actual.enhance(1.75), expected.enhance(1.75), tolerance=3)


@pytest.mark.parametrize("name", ImageEnhance.__all__)
def test_enhance_returns_fresh_image_with_source_metadata(name: str) -> None:
    blanket, pillow = random_pair("RGBA", seed=84)
    blanket.info["custom"] = pillow.info["custom"] = "retained"
    actual = getattr(ImageEnhance, name)(blanket).enhance(1)
    expected = getattr(PILEnhance, name)(pillow).enhance(1)
    assert_pillow_equal(actual, expected, tolerance=3)
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
    first, reference = random_pair(mode, size, seed=84)
    raw = first.tobytes()
    reversed_raw = raw[::-1]
    second, second_reference = image_pair(mode, size, reversed_raw)
    actual = Image.blend(first, second, 0.5)
    assert actual.tobytes() == PILImage.blend(reference, second_reference, 0.5).tobytes()
    assert (actual.mode, actual.size, actual.bit_depth) == (mode, size, 8)
    assert first.tobytes() == raw
    assert second.tobytes() == reversed_raw
