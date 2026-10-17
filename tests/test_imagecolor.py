from __future__ import annotations

import pytest
from PIL import ImageColor as PillowColor

from blanket import Image, ImageColor, ImageDraw, ImageOps


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


@pytest.mark.parametrize(
    ("color", "expected"),
    [
        (" RGB(100% 0 50% / 50%) ", (255, 0, 128, 128)),
        ("rgb(-1 3e2 +1.25e2)", (0, 255, 125)),
        ("rgba(255 0 0 / .5)", (255, 0, 0, 128)),
        ("rgba(255, 0, 0, 0.5)", (255, 0, 0, 128)),
        ("rgba(1, 2, 3, 4)", (1, 2, 3, 4)),
        ("rgb(100%, 0%, 0%, 50%)", (255, 0, 0, 128)),
        ("hsl(-240deg 100% 50%)", (0, 255, 0)),
        ("hsla(.5turn 100% 50% / 25%)", (0, 255, 255, 64)),
        ("hsl(200grad 100 50)", (0, 255, 255)),
        ("hsl(3.141592653589793rad 100% 50%)", (0, 255, 255)),
        ("hsla(120, 100%, 50%, .5)", (0, 255, 0, 128)),
        ("hwb(0 20% 30%)", (179, 51, 51)),
        ("hwb(120 80% 80% / 2)", (128, 128, 128, 255)),
        ("hwb(0 200% 100%)", (170, 170, 170)),
        ("rgb(none 50% none / none)", (0, 128, 0, 0)),
        ("hsl(none none 50% / -1)", (128, 128, 128, 0)),
        ("lab(50% none none)", (119, 119, 119)),
        ("oklab(50% 0 0)", (99, 99, 99)),
        ("lab(54.2917% 80.8125 69.8851)", (255, 0, 0)),
        ("lch(54.2917% 106.839 40.8526)", (255, 0, 0)),
        ("oklab(62.7955% 0.224863 0.125846)", (255, 0, 0)),
        ("oklch(62.7955% 0.257683 29.2339)", (255, 0, 0)),
        ("color(srgb 1 0 0 / 50%)", (255, 0, 0, 128)),
        ("color(srgb-linear .5 .5 .5)", (188, 188, 188)),
        ("color(display-p3 .4 .5 .6)", (95, 128, 156)),
        ("color(a98-rgb .4 .5 .6)", (89, 129, 155)),
        ("color(prophoto-rgb .4 .5 .6)", (69, 151, 173)),
        ("color(rec2020 .4 .5 .6)", (64, 123, 151)),
        ("color(xyz-d50 .2005 .14089 .4472)", (118, 84, 205)),
        ("color(xyz-d65 .21661 .14602 .59452)", (118, 84, 205)),
        ("color(xyz .21661 .14602 .59452)", (118, 84, 205)),
        ("color(display-p3 1 0 0)", (255, 11, 12)),
        ("oklch(.7 .4 30)", (255, 88, 67)),
    ],
)
def test_css4_colors(color: str, expected: tuple[int, ...]) -> None:
    assert ImageColor.getrgb(color) == expected


@pytest.mark.parametrize("space", ["srgb", "srgb-linear", "display-p3", "display-p3-linear", "a98-rgb", "prophoto-rgb", "rec2020"])
@pytest.mark.parametrize("value", ["0", "100%"])
def test_css4_rgb_space_endpoints(space: str, value: str) -> None:
    expected = (0, 0, 0) if value == "0" else (255, 255, 255)
    assert ImageColor.getrgb(f"color({space} {value} {value} {value})") == expected


@pytest.mark.parametrize("color", ["red", "#abcd", "rgb(12,34,56)", "rgba(1,2,3,4)", "hsl(120,30%,50%)", "hsv(250,50%,80%)", "hsb(250,50%,80%)"])
def test_legacy_color_parity(color: str) -> None:
    assert ImageColor.getrgb(color) == PillowColor.getrgb(color)


@pytest.mark.parametrize("color", PillowColor.colormap)
def test_css_named_colors_match_pillow(color: str) -> None:
    assert ImageColor.getrgb(color) == PillowColor.getrgb(color)


@pytest.mark.parametrize(
    ("color", "equivalent"),
    [
        ("lab(50% 20% -20%)", "lab(50 25 -25)"),
        ("lch(50% 20% .5turn)", "lch(50 30 180)"),
        ("oklab(50% 20% -20%)", "oklab(.5 .08 -.08)"),
        ("oklch(50% 20% .5turn)", "oklch(.5 .08 180)"),
        ("color(display-p3-linear .25 .25 .25)", "color(srgb-linear .25 .25 .25)"),
        ("lch(50 -10 90)", "lab(50 0 0)"),
        ("oklch(50% -1 90)", "oklab(.5 0 0)"),
    ],
)
def test_css4_component_scales(color: str, equivalent: str) -> None:
    assert ImageColor.getrgb(color) == ImageColor.getrgb(equivalent)


@pytest.mark.parametrize(
    "color",
    [
        "rgb(1, 2 3)",
        "rgb(1, 2, 3 / .5)",
        "rgb(1 2 3 /)",
        "rgb(1 2 3 / .5 / .5)",
        "rgb(1%, 2, 3)",
        "rgb(none, 0, 0)",
        "hsl(30% 50% 50%)",
        "hsl(30, 50, 50)",
        "hwb(0, 0%, 0%)",
        "color(unknown 1 0 0)",
        "lab(50 0)",
        "rgb(nan 0 0)",
        "rgb(1e999 0 0)",
        "currentcolor",
        "CanvasText",
        "red trailing",
        "#12345",
        "rgb(1deg 0 0)",
        "red" * 400,
    ],
)
def test_invalid_css4_colors(color: str) -> None:
    with pytest.raises(ValueError):
        ImageColor.getrgb(color)


def test_css4_colors_in_image_operations() -> None:
    color = "hwb(0 20% 30% / 50%)"
    image = Image.new("RGBA", (2, 2), color)
    assert image.getpixel((0, 0)) == (179, 51, 51, 128)
    ImageDraw.Draw(image).point((1, 1), fill="oklab(50% 0 0 / 25%)")
    assert image.getpixel((1, 1)) == (99, 99, 99, 64)
    expanded = ImageOps.expand(image, 1, fill=color)
    assert expanded.getpixel((0, 0)) == (179, 51, 51, 128)
