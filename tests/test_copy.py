from __future__ import annotations

from copy import copy, deepcopy
from typing import TYPE_CHECKING

import pytest

from blanket import Image

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.mark.parametrize("copier", [Image.Image.copy, copy, deepcopy])
def test_copy_pixels_and_metadata_lists_are_independent(copier: Callable[[Image.Image], Image.Image]) -> None:
    image = Image.new("RGBA", (3, 2), (12, 34, 56, 78))
    image.info["keywords"] = ["original"]
    image.getexif()[305] = "original"
    result = copier(image)
    assert result.size == image.size
    assert result.tobytes() == image.tobytes()
    result.info["keywords"].append("changed")
    result.getexif()[305] = "changed"
    result.putpixel((0, 0), (1, 2, 3, 4))
    assert image.info["keywords"] == ["original"]
    assert image.getexif()[305] == "original"
    assert image.getpixel((0, 0)) == (12, 34, 56, 78)


@pytest.mark.parametrize("copier", [Image.Image.copy, copy, deepcopy])
def test_copy_preserves_metadata_references_to_image(copier: Callable[[Image.Image], Image.Image]) -> None:
    image = Image.new("L", (2, 1), 30)
    image.info["source"] = image
    result = copier(image)
    assert result.info["source"] is (result if copier is deepcopy else image)


def test_deepcopy_nested_metadata_is_independent() -> None:
    image = Image.new("L", (2, 1), 30)
    image.info["metadata"] = {"nested": [1, 2]}
    result = deepcopy(image)
    result.info["metadata"]["nested"].append(3)
    assert image.info["metadata"] == {"nested": [1, 2]}


def test_copy_retains_dictionary_metadata_references_like_pillow() -> None:
    image = Image.new("L", (2, 1), 30)
    image.info["metadata"] = {"nested": [1, 2]}
    result = image.copy()
    assert result.info["metadata"] is image.info["metadata"]


@pytest.mark.parametrize("copier", [Image.Image.copy, copy, deepcopy])
def test_copy_indexed_palette_is_independent(copier: Callable[[Image.Image], Image.Image]) -> None:
    image = Image.new("P", (3, 2), 0)
    image.putpalette([20, 40, 60, 100, 120, 140])
    result = copier(image)
    result.putpalette([10, 30, 50, 100, 120, 140])
    assert image.convert("RGB").getpixel((0, 0)) == (20, 40, 60)
    assert result.convert("RGB").getpixel((0, 0)) == (10, 30, 50)


def test_transform_copies_nested_metadata() -> None:
    image = Image.new("RGB", (3, 2))
    image.info["keywords"] = ["original"]
    transformed = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    transformed.info["keywords"].append("changed")
    assert image.info["keywords"] == ["original"]


@pytest.mark.parametrize("bit_depth", [10, 12, 16])
@pytest.mark.parametrize("copier", [Image.Image.copy, copy, deepcopy])
def test_copy_preserves_high_depth_samples(bit_depth: int, copier: Callable[[Image.Image], Image.Image]) -> None:
    image = Image.frombytes("RGBA", (3, 2), b"\x00\x02\x2c\x01\xbc\x02\x20\x03" * 6, bit_depth=bit_depth)
    result = copier(image)
    assert result.bit_depth == bit_depth
    assert result.mode == image.mode
    assert result.tobytes() == image.tobytes()


@pytest.mark.parametrize("copier", [Image.Image.copy, copy, deepcopy])
def test_copy_closed_image_raises(copier: Callable[[Image.Image], Image.Image]) -> None:
    image = Image.new("RGB", (1, 1))
    image.close()
    with pytest.raises(ValueError, match="closed"):
        copier(image)
