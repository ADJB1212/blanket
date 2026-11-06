"""Pillow 13.0.0 contracts; see PILLOW_TESTS.md."""

from __future__ import annotations

import pillow_helpers as helper
import pytest
from helpers import assert_image_properties
from pillow_helpers import assert_image_equal

from blanket.Image import Transpose

TYPE_CHECKING = False
if TYPE_CHECKING:
    from blanket import Image

HOPPER = {mode: helper.hopper(mode).crop((0, 0, 121, 127)).copy() for mode in ["L", "RGB", "I;16", "I;16L", "I;16B"]}


@pytest.mark.parametrize("mode", HOPPER)
@pytest.mark.parametrize(
    "method,swap,point",
    [
        (Transpose.FLIP_LEFT_RIGHT, False, lambda x, y, w, h: (w - 1 - x, y)),
        (Transpose.FLIP_TOP_BOTTOM, False, lambda x, y, w, h: (x, h - 1 - y)),
        (Transpose.ROTATE_90, True, lambda x, y, w, h: (y, w - 1 - x)),
        (Transpose.ROTATE_180, False, lambda x, y, w, h: (w - 1 - x, h - 1 - y)),
        (Transpose.ROTATE_270, True, lambda x, y, w, h: (h - 1 - y, x)),
        (Transpose.TRANSPOSE, True, lambda x, y, w, h: (y, x)),
        (Transpose.TRANSVERSE, True, lambda x, y, w, h: (h - 1 - y, w - 1 - x)),
    ],
)
def test_corner_mapping(mode, method, swap, point) -> None:
    im = HOPPER[mode]
    out = im.transpose(method)
    assert_image_properties(out, mode=mode, size=im.size[::-1] if swap else im.size)
    w, h = im.size
    for x, y in ((1, 1), (w - 2, 1), (1, h - 2), (w - 2, h - 2)):
        assert im.getpixel((x, y)) == out.getpixel(point(x, y, w, h))


@pytest.mark.parametrize("mode", HOPPER)
def test_roundtrip(mode: str) -> None:
    im = HOPPER[mode]

    def transpose(first: Transpose, second: Transpose) -> Image.Image:
        return im.transpose(first).transpose(second)

    assert_image_equal(im, transpose(Transpose.FLIP_LEFT_RIGHT, Transpose.FLIP_LEFT_RIGHT))
    assert_image_equal(im, transpose(Transpose.FLIP_TOP_BOTTOM, Transpose.FLIP_TOP_BOTTOM))
    assert_image_equal(im, transpose(Transpose.ROTATE_90, Transpose.ROTATE_270))
    assert_image_equal(im, transpose(Transpose.ROTATE_180, Transpose.ROTATE_180))
    assert_image_equal(im.transpose(Transpose.TRANSPOSE), transpose(Transpose.ROTATE_90, Transpose.FLIP_TOP_BOTTOM))
    assert_image_equal(im.transpose(Transpose.TRANSPOSE), transpose(Transpose.ROTATE_270, Transpose.FLIP_LEFT_RIGHT))
    assert_image_equal(im.transpose(Transpose.TRANSVERSE), transpose(Transpose.ROTATE_90, Transpose.FLIP_LEFT_RIGHT))
    assert_image_equal(im.transpose(Transpose.TRANSVERSE), transpose(Transpose.ROTATE_270, Transpose.FLIP_TOP_BOTTOM))
    assert_image_equal(im.transpose(Transpose.TRANSVERSE), transpose(Transpose.ROTATE_180, Transpose.TRANSPOSE))
