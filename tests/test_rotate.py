"""Rotation parity with Pillow."""

from __future__ import annotations

import pytest
from blanket import Image
from PIL import Image as PillowImage


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
@pytest.mark.parametrize("size", [(13, 8), (9, 9), (1, 7)])
@pytest.mark.parametrize("method", [0, 2, 3])
@pytest.mark.parametrize("angle", [0, 90, 180, 270, 45, -32.5, 721])
@pytest.mark.parametrize("options", [{}, {"expand": True}, {"center": (2.25, 4.5)}, {"translate": (3, -2)}, {"expand": True, "center": (1, 3), "translate": (-2.5, 1)}, {"expand": True, "fillcolor": "#12345680"}])
def test_rotate_parity(mode, size, method, angle, options):
    data = bytes((i * 37 + i // 7) % 256 for i in range(size[0] * size[1] * len(mode)))
    image = Image.frombytes(mode, size, data)
    reference = PillowImage.frombytes(mode, size, data)
    actual = image.rotate(angle, method, **options)
    expected = reference.rotate(angle, method, **options)
    assert actual.size == expected.size
    assert actual.mode == expected.mode
    assert actual.tobytes() == expected.tobytes()
    assert image.tobytes() == data


def test_rotate_metadata_and_independence():
    image = Image.frombytes("L", (2, 2), b"abcd")
    image.info["test"] = 42
    result = image.rotate(0)
    assert result is not image
    assert result.info == image.info
    assert result.info is not image.info
    image.close()
    assert result.tobytes() == b"abcd"
    with pytest.raises(ValueError):
        image.rotate(12)


@pytest.mark.parametrize("method", [1, 4, 5, -1, 100])
def test_rotate_invalid_filter(method):
    image = Image.frombytes("L", (2, 2), b"abcd")
    with pytest.raises(ValueError):
        image.rotate(12, method)


@pytest.mark.parametrize("angle", [float("nan"), float("inf")])
def test_rotate_nonfinite_angle(angle):
    image = Image.frombytes("L", (2, 2), b"abcd")
    with pytest.raises(ValueError):
        image.rotate(angle)


@pytest.mark.parametrize("fill", [None, 0, (18, 52, 86, 128), (100, 20, 30, 0)])
@pytest.mark.parametrize("method", [0, 2, 3])
def test_rotate_rgba_fill(fill, method):
    data = bytes(range(64))
    image = Image.frombytes("RGBA", (4, 4), data)
    reference = PillowImage.frombytes("RGBA", (4, 4), data)
    assert image.rotate(33, method, expand=True, fillcolor=fill).tobytes() == reference.rotate(33, method, expand=True, fillcolor=fill).tobytes()


@pytest.mark.parametrize("option", ["center", "translate"])
def test_rotate_nonfinite_coordinates(option):
    image = Image.frombytes("L", (2, 2), b"abcd")
    with pytest.raises(ValueError):
        image.rotate(12, **{option: (float("inf"), 0)})
