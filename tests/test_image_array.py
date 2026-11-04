"""Passing Pillow 13.0.0 contracts; see PILLOW_TESTS.md."""

from __future__ import annotations

from typing import Any

import pytest
from helpers import assert_raises
from pillow_helpers import hopper

from blanket import Image

numpy = pytest.importorskip("numpy", reason="NumPy not installed")

im = hopper().resize((128, 100))

TYPE_CHECKING = False


def test_fromarray() -> None:
    class Wrapper:
        """Class with API matching Image.fromarray"""

        def __init__(self, img: Image.Image, arr_params: dict[str, Any]) -> None:
            self.img = img
            self.__array_interface__ = arr_params

        def tobytes(self) -> bytes:
            return self.img.tobytes()

    def test(mode: str) -> tuple[str, tuple[int, int], bool]:
        i = im.convert(mode)
        a = numpy.array(i)
        # Make wrapper instance for image, new array interface
        wrapped = Wrapper(
            i,
            {
                "shape": a.shape,
                "typestr": a.dtype.str,
                "version": 3,
                "data": a.data,
                "strides": 1,  # pretend it's non-contiguous
            },
        )
        out = Image.fromarray(wrapped)
        return out.mode, out.size, i.get_flattened_data() == out.get_flattened_data()

    # assert test("1") == ("1", (128, 100), True)
    assert test("L") == ("L", (128, 100), True)
    assert test("I") == ("I", (128, 100), True)
    assert test("F") == ("F", (128, 100), True)
    assert test("LA") == ("LA", (128, 100), True)
    assert test("RGB") == ("RGB", (128, 100), True)
    assert test("RGBA") == ("RGBA", (128, 100), True)
    assert test("RGBX") == ("RGBA", (128, 100), True)

    # Test mode is None with no "typestr" in the array interface
    wrapped = Wrapper(hopper("L"), {"shape": (100, 128)})
    assert_raises(TypeError, lambda: Image.fromarray(wrapped))


def test_fromarray_strides_without_tobytes() -> None:
    class Wrapper:
        def __init__(self, arr_params: dict[str, Any]) -> None:
            self.__array_interface__ = arr_params

    with pytest.raises(ValueError):
        wrapped = Wrapper({"shape": (1, 1), "strides": (1, 1), "typestr": "|u1"})
        Image.fromarray(wrapped, "L")
