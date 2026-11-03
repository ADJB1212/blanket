"""Pillow 13.0.0 contracts and Blanket regressions; see PILLOW_TESTS.md."""

from __future__ import annotations

import pytest
from PIL import ImageColor as PillowColor

from blanket import Image, ImageColor, ImageDraw, ImageOps


def test_hash() -> None:
    # short 3 components
    assert ImageColor.getrgb("#f00") == (255, 0, 0)
    assert ImageColor.getrgb("#0f0") == (0, 255, 0)
    assert ImageColor.getrgb("#00f") == (0, 0, 255)

    # short 4 components
    assert ImageColor.getrgb("#f000") == (255, 0, 0, 0)
    assert ImageColor.getrgb("#0f00") == (0, 255, 0, 0)
    assert ImageColor.getrgb("#00f0") == (0, 0, 255, 0)
    assert ImageColor.getrgb("#000f") == (0, 0, 0, 255)

    # long 3 components
    assert ImageColor.getrgb("#de0000") == (222, 0, 0)
    assert ImageColor.getrgb("#00de00") == (0, 222, 0)
    assert ImageColor.getrgb("#0000de") == (0, 0, 222)

    # long 4 components
    assert ImageColor.getrgb("#de000000") == (222, 0, 0, 0)
    assert ImageColor.getrgb("#00de0000") == (0, 222, 0, 0)
    assert ImageColor.getrgb("#0000de00") == (0, 0, 222, 0)
    assert ImageColor.getrgb("#000000de") == (0, 0, 0, 222)

    # case insensitivity
    assert ImageColor.getrgb("#DEF") == ImageColor.getrgb("#def")
    assert ImageColor.getrgb("#CDEF") == ImageColor.getrgb("#cdef")
    assert ImageColor.getrgb("#DEFDEF") == ImageColor.getrgb("#defdef")
    assert ImageColor.getrgb("#CDEFCDEF") == ImageColor.getrgb("#cdefcdef")

    # not a number
    with pytest.raises(ValueError):
        ImageColor.getrgb("#fo0")
    with pytest.raises(ValueError):
        ImageColor.getrgb("#fo00")
    with pytest.raises(ValueError):
        ImageColor.getrgb("#fo0000")
    with pytest.raises(ValueError):
        ImageColor.getrgb("#fo000000")

    # wrong number of components
    with pytest.raises(ValueError):
        ImageColor.getrgb("#f0000")
    with pytest.raises(ValueError):
        ImageColor.getrgb("#f000000")
    with pytest.raises(ValueError):
        ImageColor.getrgb("#f00000000")
    with pytest.raises(ValueError):
        ImageColor.getrgb("#f000000000")
    with pytest.raises(ValueError):
        ImageColor.getrgb("#f00000 ")


def test_transparent_keyword_extension() -> None:
    assert ImageColor.getrgb("transparent") == (0, 0, 0, 0)


def test_colormap() -> None:
    assert ImageColor.getrgb("black") == (0, 0, 0)
    assert ImageColor.getrgb("white") == (255, 255, 255)
    assert ImageColor.getrgb("WHITE") == (255, 255, 255)

    with pytest.raises(ValueError):
        ImageColor.getrgb("black ")


def test_functions() -> None:
    # rgb numbers
    assert ImageColor.getrgb("rgb(255,0,0)") == (255, 0, 0)
    assert ImageColor.getrgb("rgb(0,255,0)") == (0, 255, 0)
    assert ImageColor.getrgb("rgb(0,0,255)") == (0, 0, 255)

    # percents
    assert ImageColor.getrgb("rgb(100%,0%,0%)") == (255, 0, 0)
    assert ImageColor.getrgb("rgb(0%,100%,0%)") == (0, 255, 0)
    assert ImageColor.getrgb("rgb(0%,0%,100%)") == (0, 0, 255)

    # rgba numbers
    assert ImageColor.getrgb("rgba(255,0,0,0)") == (255, 0, 0, 0)
    assert ImageColor.getrgb("rgba(0,255,0,0)") == (0, 255, 0, 0)
    assert ImageColor.getrgb("rgba(0,0,255,0)") == (0, 0, 255, 0)
    assert ImageColor.getrgb("rgba(0,0,0,255)") == (0, 0, 0, 255)

    assert ImageColor.getrgb("hsl(0,100%,50%)") == (255, 0, 0)
    assert ImageColor.getrgb("hsl(360,100%,50%)") == (255, 0, 0)
    assert ImageColor.getrgb("hsl(180,100%,50%)") == (0, 255, 255)

    assert ImageColor.getrgb("hsv(0,100%,100%)") == (255, 0, 0)
    assert ImageColor.getrgb("hsv(360,100%,100%)") == (255, 0, 0)
    assert ImageColor.getrgb("hsv(180,100%,100%)") == (0, 255, 255)

    # alternate format
    assert ImageColor.getrgb("hsb(0,100%,50%)") == ImageColor.getrgb("hsv(0,100%,50%)")

    # floats
    assert ImageColor.getrgb("hsl(0.1,99.2%,50.3%)") == (254, 3, 3)
    assert ImageColor.getrgb("hsl(360.,100.0%,50%)") == (255, 0, 0)

    assert ImageColor.getrgb("hsv(0.1,99.2%,99.3%)") == (253, 2, 2)
    assert ImageColor.getrgb("hsv(360.,100.0%,100%)") == (255, 0, 0)

    # case insensitivity
    assert ImageColor.getrgb("RGB(255,0,0)") == ImageColor.getrgb("rgb(255,0,0)")
    assert ImageColor.getrgb("RGB(100%,0%,0%)") == ImageColor.getrgb("rgb(100%,0%,0%)")
    assert ImageColor.getrgb("RGBA(255,0,0,0)") == ImageColor.getrgb("rgba(255,0,0,0)")
    assert ImageColor.getrgb("HSL(0,100%,50%)") == ImageColor.getrgb("hsl(0,100%,50%)")
    assert ImageColor.getrgb("HSV(0,100%,50%)") == ImageColor.getrgb("hsv(0,100%,50%)")
    assert ImageColor.getrgb("HSB(0,100%,50%)") == ImageColor.getrgb("hsb(0,100%,50%)")

    # space agnosticism
    assert ImageColor.getrgb("rgb(  255  ,  0  ,  0  )") == (255, 0, 0)
    assert ImageColor.getrgb("rgb(  100%  ,  0%  ,  0%  )") == (255, 0, 0)
    assert ImageColor.getrgb("rgba(  255  ,  0  ,  0  ,  0  )") == (255, 0, 0, 0)
    assert ImageColor.getrgb("hsl(  0  ,  100%  ,  50%  )") == (255, 0, 0)
    assert ImageColor.getrgb("hsv(  0  ,  100%  ,  100%  )") == (255, 0, 0)

    # wrong number of components
    with pytest.raises(ValueError):
        ImageColor.getrgb("rgb(255,0)")
    with pytest.raises(ValueError):
        ImageColor.getrgb("rgb(255,0,0,0)")

    with pytest.raises(ValueError):
        ImageColor.getrgb("rgb(100%,0%)")
    with pytest.raises(ValueError):
        ImageColor.getrgb("rgb(100%,0%,0)")
    with pytest.raises(ValueError):
        ImageColor.getrgb("rgb(100%,0%,0 %)")
    with pytest.raises(ValueError):
        ImageColor.getrgb("rgb(100%,0%,0%,0%)")

    with pytest.raises(ValueError):
        ImageColor.getrgb("rgba(255,0,0)")
    with pytest.raises(ValueError):
        ImageColor.getrgb("rgba(255,0,0,0,0)")

    with pytest.raises(ValueError):
        ImageColor.getrgb("hsl(0,100%)")
    with pytest.raises(ValueError):
        ImageColor.getrgb("hsl(0,100%,0%,0%)")
    with pytest.raises(ValueError):
        ImageColor.getrgb("hsl(0%,100%,50%)")
    with pytest.raises(ValueError):
        ImageColor.getrgb("hsl(0,100,50%)")
    with pytest.raises(ValueError):
        ImageColor.getrgb("hsl(0,100%,50)")

    with pytest.raises(ValueError):
        ImageColor.getrgb("hsv(0,100%)")
    with pytest.raises(ValueError):
        ImageColor.getrgb("hsv(0,100%,0%,0%)")
    with pytest.raises(ValueError):
        ImageColor.getrgb("hsv(0%,100%,50%)")
    with pytest.raises(ValueError):
        ImageColor.getrgb("hsv(0,100,50%)")
    with pytest.raises(ValueError):
        ImageColor.getrgb("hsv(0,100%,50)")


# look for rounding errors (based on code by Tim Hatch)
def test_rounding_errors() -> None:
    for color in ImageColor.colormap:
        expected = Image.new("RGB", (1, 1), color).convert("L").getpixel((0, 0))
        actual = ImageColor.getcolor(color, "L")
        assert expected == actual

    assert ImageColor.getcolor("rgba(0, 255, 115, 33)", "RGB") == (0, 255, 115)
    Image.new("RGB", (1, 1), "white")

    assert ImageColor.getcolor("black", "RGBA") == (0, 0, 0, 255)
    assert ImageColor.getcolor("white", "RGBA") == (255, 255, 255, 255)
    assert ImageColor.getcolor("rgba(0, 255, 115, 33)", "RGBA") == (0, 255, 115, 33)
    Image.new("RGBA", (1, 1), "white")

    assert ImageColor.getcolor("black", "L") == 0
    assert ImageColor.getcolor("white", "L") == 255
    assert ImageColor.getcolor("rgba(0, 255, 115, 33)", "L") == 163
    Image.new("L", (1, 1), "white")

    assert ImageColor.getcolor("black", "1") == 0
    assert ImageColor.getcolor("white", "1") == 255
    # The following test is wrong, but is current behavior
    # The correct result should be 255 due to the mode 1
    assert ImageColor.getcolor("rgba(0, 255, 115, 33)", "1") == 163
    # Correct behavior
    # assert
    #     255, ImageColor.getcolor("rgba(0, 255, 115, 33)", "1"))
    Image.new("1", (1, 1), "white")

    assert ImageColor.getcolor("black", "LA") == (0, 255)
    assert ImageColor.getcolor("white", "LA") == (255, 255)
    assert ImageColor.getcolor("rgba(0, 255, 115, 33)", "LA") == (163, 33)
    Image.new("LA", (1, 1), "white")


def test_color_hsv() -> None:
    assert ImageColor.getcolor("hsv(240, 100%, 100%)", "HSV") == (170, 255, 255)


@pytest.mark.parametrize("mode, premultiplied_mode", (("LA", "La"), ("RGBA", "RGBa")))
@pytest.mark.parametrize("color", ("#f00", "#ff000080", "#ff000050"))
def test_color_premultiplied_alpha(mode: str, premultiplied_mode: str, color: str) -> None:
    im = Image.new(mode, (1, 1), color)
    im_premultiplied = im.convert(premultiplied_mode)
    expected = im_premultiplied.getpixel((0, 0))

    assert ImageColor.getcolor(color, premultiplied_mode) == expected


def test_color_too_long() -> None:
    # Arrange
    color_too_long = "hsl(" + "1" * 40 + "," + "1" * 40 + "%," + "1" * 40 + "%)"

    # Act / Assert
    with pytest.raises(ValueError):
        ImageColor.getrgb(color_too_long)


def test_public_colormap_accepts_custom_hex_and_tuple_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(ImageColor.colormap, "blanket-hex-color", "#123456")
    monkeypatch.setitem(ImageColor.colormap, "blanket-tuple-color", (23, 45, 67))
    assert ImageColor.getrgb("blanket-hex-color") == (18, 52, 86)
    assert ImageColor.getrgb("BLANKET-TUPLE-COLOR") == (23, 45, 67)
    assert set(PillowColor.colormap) <= set(ImageColor.colormap)


@pytest.mark.parametrize(("rgb", "expected"), [((2, 2, 2), (1, 1, 1)), ((-1, -1, -1), (0, 0, 0)), ((1.001, 0, 0), (1, 0, 0))])
def test_gamut_mapping_extreme_lightness_and_small_excursions(rgb, expected) -> None:
    from blanket._css_color import _gamut_map

    assert _gamut_map(rgb) == pytest.approx(expected)


@pytest.mark.parametrize("rgb", [(float("inf"), 0, 0), (float("nan"), 0, 0)])
def test_gamut_mapping_rejects_nonfinite_conversion(rgb) -> None:
    from blanket._css_color import _gamut_map

    with pytest.raises(ValueError, match="overflow"):
        _gamut_map(rgb)


def test_gamut_mapping_converges_at_chroma_tolerance() -> None:
    assert ImageColor.getrgb("color(srgb 2.194630862304452 -0.9097089088587733 -1.816498245207691)") == (251, 180, 0)


@pytest.mark.parametrize("color", ["unknown(1 2 3)", "color(srgb 1.1 0.1 0.1)", "color(srgb 0.1 1.1 0.1)", "color(srgb 0.1 0.1 1.1)"])
def test_color_function_dispatch_and_out_of_gamut_primaries(color: str) -> None:
    if color.startswith("unknown"):
        with pytest.raises(ValueError):
            ImageColor.getrgb(color)
    else:
        assert all(0 <= value <= 255 for value in ImageColor.getrgb(color))


def test_invalid_colors_and_modes_raise() -> None:
    with pytest.raises(ValueError):
        ImageColor.getrgb("not a color")
    with pytest.raises(TypeError):
        ImageColor.getrgb(123)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        ImageColor.getcolor("red", "invalid")


@pytest.mark.parametrize(
    ("color", "expected"),
    [
        (" RGB(100% 0 50% / 50%) ", (255, 0, 128, 128)),
        ("rgb(-1 3e2 +1.25e2)", (0, 255, 125)),
        ("rgba(255 0 0 / .5)", (255, 0, 0, 128)),
        ("rgba(255, 0, 0, 0.5)", (255, 0, 0, 128)),
        ("rgba(1, 2, 3, 4)", (1, 2, 3, 4)),
        ("rgba(100%, 0%, 0%, 50%)", (255, 0, 0, 128)),
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
