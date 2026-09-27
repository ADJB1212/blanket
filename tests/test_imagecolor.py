from __future__ import annotations

import pytest

from blanket import ImageColor


def test_getrgb_named_hex_and_functional_colors() -> None:
    assert ImageColor.getrgb("ReBeccapurple") == (102, 51, 153)
    assert ImageColor.getrgb("#1234") == (17, 34, 51, 68)
    assert ImageColor.getrgb("transparent") == (0, 0, 0, 0)
    assert ImageColor.getrgb("rgb(100%, 0%, 50%)") == (255, 0, 128)


def test_getcolor_converts_to_requested_mode() -> None:
    assert ImageColor.getcolor("red", "RGB") == (255, 0, 0)
    assert ImageColor.getcolor("#10203080", "RGBA") == (16, 32, 48, 128)
    assert ImageColor.getcolor("white", "L") == 255
    assert ImageColor.getcolor("black", "LA") == (0, 255)
    assert ImageColor.getcolor("white", "1") == 255
    assert ImageColor.getcolor("red", "CMYK") == (0, 255, 255, 0)


@pytest.mark.parametrize(
    ("color", "mode", "expected"),
    [
        ("#7f7f7f", "1", 0),
        ("#808080", "1", 255),
        ("red", "I", 76),
        ("red", "F", 76),
        ("#ff000080", "LA", (76, 128)),
        ("red", "RGBA", (255, 0, 0, 255)),
        ("#10203080", "RGB", (16, 32, 48)),
        ("black", "CMYK", (0, 0, 0, 255)),
        ("#4080ff", "CMYK", (191, 127, 0, 0)),
    ],
)
def test_getcolor_native_conversions(color: str, mode: str, expected: int | tuple[int, ...]) -> None:
    assert ImageColor.getcolor(color, mode) == expected


def test_invalid_colors_and_modes_raise() -> None:
    with pytest.raises(ValueError):
        ImageColor.getrgb("not a color")
    with pytest.raises(TypeError):
        ImageColor.getrgb(123)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        ImageColor.getcolor("red", "P")
