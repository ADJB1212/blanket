"""Passing Pillow 13.0.0 contracts; see PILLOW_TESTS.md."""

from __future__ import annotations

from blanket import Image


def test_interface() -> None:
    im = Image.new("RGBA", (1, 1), (1, 2, 3, 0))
    assert im.getpixel((0, 0)) == (1, 2, 3, 0)

    im = Image.new("RGBA", (1, 1), (1, 2, 3))
    assert im.getpixel((0, 0)) == (1, 2, 3, 255)

    im.putalpha(Image.new("L", im.size, 4))
    assert im.getpixel((0, 0)) == (1, 2, 3, 4)

    im.putalpha(5)
    assert im.getpixel((0, 0)) == (1, 2, 3, 5)


def test_readonly() -> None:
    im = Image.new("RGB", (1, 1), (1, 2, 3))
    im.readonly = 1

    im.putalpha(4)
    assert not im.readonly
    assert im.mode == "RGBA"
    assert im.getpixel((0, 0)) == (1, 2, 3, 4)
