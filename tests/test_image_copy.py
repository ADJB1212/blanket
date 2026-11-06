"""Passing Pillow 13.0.0 contracts; see PILLOW_TESTS.md."""

from __future__ import annotations

import copy

import pytest
from helpers import assert_image_properties
from pillow_helpers import PILLOW_IMAGES, hopper, skip_unless_feature

from blanket import Image


@pytest.mark.parametrize("mode", ("1", "P", "L", "RGB", "I", "F"))
def test_copy(mode: str) -> None:
    cropped_coordinates = (10, 10, 20, 20)
    cropped_size = (10, 10)

    # Internal copy method
    im = hopper(mode)
    out = im.copy()
    assert_image_properties(out, mode=im.mode, size=im.size)

    # Python's copy method
    im = hopper(mode)
    out = copy.copy(im)
    assert_image_properties(out, mode=im.mode, size=im.size)

    # Internal copy method on a cropped image
    im = hopper(mode)
    out = im.crop(cropped_coordinates).copy()
    assert_image_properties(out, mode=im.mode, size=cropped_size)

    # Python's copy method on a cropped image
    im = hopper(mode)
    out = copy.copy(im.crop(cropped_coordinates))
    assert_image_properties(out, mode=im.mode, size=cropped_size)


def test_copy_zero() -> None:
    im = Image.new("RGB", (0, 0))
    out = im.copy()
    assert_image_properties(out, mode=im.mode, size=im.size)


@skip_unless_feature("libtiff")
def test_deepcopy() -> None:
    with Image.open(PILLOW_IMAGES / "g4_orientation_5.tif") as im:
        assert im.size == (590, 88)

        out = copy.deepcopy(im)
    assert out.size == (590, 88)
