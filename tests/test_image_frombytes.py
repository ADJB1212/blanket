"""Pillow 13.0.0 contracts; see PILLOW_TESTS.md."""

from __future__ import annotations

import pytest
from pillow_helpers import assert_image_equal, hopper

from blanket import Image


@pytest.mark.parametrize("data_type", ("bytes", "memoryview"))
def test_sanity(data_type: str) -> None:
    im1 = hopper()

    data: bytes | memoryview = im1.tobytes()
    if data_type == "memoryview":
        data = memoryview(data)
    im2 = Image.frombytes(im1.mode, im1.size, data)

    assert_image_equal(im1, im2)
