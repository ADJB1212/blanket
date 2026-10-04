from __future__ import annotations

import pytest
from PIL import Image as PILImage, ImageDraw as PILDraw, ImageFont as PILFont

from blanket import Image, ImageDraw


@pytest.mark.parametrize("points", [[(1, 2), (12, 9)], [(12, 9), (1, 2)], [(-4, -3), (20, 10)], [(0, 0), (7, 14)], [(7, 14), (0, 0)], [(2, 2), (2, 2)], [(2, 2), (2, 9), (8, 9)]])
@pytest.mark.parametrize("mode,ink", [("L", 117), ("RGB", (23, 81, 157)), ("RGBA", (23, 81, 157, 91)), ("I", -100), ("F", 1.5), ("I;16", 500)])
def test_thin_lines_match_pillow(points: list[tuple[int, int]], mode: str, ink: object) -> None:
    actual = Image.new(mode, (15, 15))
    expected = PILImage.new(mode, actual.size)
    ImageDraw.Draw(actual).line(points, fill=ink)
    PILDraw.Draw(expected).line(points, fill=ink)
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("operation", ["line", "polygon", "ellipse", "arc", "chord", "pieslice", "rounded_rectangle", "regular_polygon"])
@pytest.mark.parametrize(
    "mode,ink",
    [
        ("1", 1),
        ("L", 117),
        ("LA", (117, 81)),
        ("RGB", (23, 81, 157)),
        ("RGBA", (23, 81, 157, 91)),
        ("HSV", (23, 81, 157)),
        ("YCbCr", (23, 81, 157)),
        ("LAB", (23, 81, 157)),
        ("CMYK", (23, 81, 157, 91)),
        ("P", 7),
        ("PA", (7, 81)),
        ("I", -123456),
        ("F", 1.5),
        ("I;16", 12000),
        ("I;16L", 12000),
        ("I;16B", 12000),
    ],
)
def test_geometry_supports_modes(operation: str, mode: str, ink: object) -> None:
    image = Image.new(mode, (19, 17))
    draw = ImageDraw.Draw(image)
    args = [(2, 2, 15, 13)]
    if operation in ("arc", "chord", "pieslice"):
        args += [20, 280]
    elif operation == "regular_polygon":
        args = [((9, 8), 6), 5]
    elif operation == "polygon":
        args = [[(2, 2), (15, 2), (9, 13)]]
    before = image.tobytes()
    getattr(draw, operation)(*args, fill=ink, width=2)
    assert image.tobytes() != before
    assert image.mode == mode
    assert image.size == (19, 17)


@pytest.mark.parametrize("operation", ["ellipse", "chord", "pieslice", "polygon", "rounded_rectangle", "regular_polygon"])
def test_fill_and_outline(operation: str) -> None:
    image = Image.new("L", (31, 31))
    args = [(3, 3, 27, 27)]
    if operation in ("chord", "pieslice"):
        args += [0, 270]
    elif operation == "regular_polygon":
        args = [((15, 15), 12), 6]
    elif operation == "polygon":
        args = [[(3, 3), (27, 3), (15, 27)]]
    ImageDraw.Draw(image).__getattribute__(operation)(*args, fill=71, outline=201, width=3)
    assert 71 in image.tobytes()
    assert 201 in image.tobytes()
    assert image.getpixel((0, 0)) == 0


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
@pytest.mark.parametrize("operation", ["line", "ellipse", "polygon", "rounded_rectangle"])
def test_antialias_produces_coverage(mode: str, operation: str) -> None:
    image = Image.new(mode, (24, 24))
    draw = ImageDraw.Draw(image, antialias=True)
    coords = [(3, 3), (20, 8), (7, 20)] if operation == "polygon" else (3, 3, 20, 20)
    kwargs = {"radius": 5} if operation == "rounded_rectangle" else {}
    getattr(draw, operation)(coords, fill=255 if mode == "L" else "white", **kwargs)
    assert any(0 < sample < 255 for sample in image.tobytes())


@pytest.mark.parametrize("mode", ["1", "P", "PA", "I", "F", "I;16", "I;16L", "I;16B"])
def test_antialias_rejects_noncoverage_modes(mode: str) -> None:
    image = Image.new(mode, (10, 10))
    before = image.tobytes()
    with pytest.raises(ValueError, match="anti-aliasing"):
        ImageDraw.Draw(image, antialias=True).ellipse((1, 1, 8, 8), fill=1)
    assert image.tobytes() == before


def test_clipping_and_single_alpha_blend() -> None:
    image = Image.new("RGB", (10, 10), (0, 0, 100))
    draw = ImageDraw.Draw(image, "RGBA")
    draw.polygon([(-10, -10), (20, -10), (20, 20), (-10, 20)], fill=(200, 0, 0, 128))
    assert image.getpixel((5, 5)) == (100, 0, 50)
    assert len(set(image.tobytes()[::3])) == 1


@pytest.mark.parametrize("mode,ink", [("L", 117), ("RGB", (23, 81, 157)), ("RGBA", (23, 81, 157, 91)), ("I", -100), ("F", 1.5), ("I;16B", 500)])
def test_filled_circle_matches_pillow(mode: str, ink: object) -> None:
    actual = Image.new(mode, (15, 15))
    expected = PILImage.new(mode, actual.size)
    ImageDraw.Draw(actual).ellipse((2, 2, 12, 12), fill=ink)
    PILDraw.Draw(expected).ellipse((2, 2, 12, 12), fill=ink)
    assert actual.tobytes() == expected.tobytes()


def test_curved_shape_interiors_and_selected_corners() -> None:
    chord = Image.new("L", (25, 25))
    pie = Image.new("L", chord.size)
    ImageDraw.Draw(chord).chord((2, 2, 22, 22), 0, 90, fill=255)
    ImageDraw.Draw(pie).pieslice((2, 2, 22, 22), 0, 90, fill=255)
    assert chord.getpixel((12, 12)) == 0
    assert pie.getpixel((12, 12)) == 255
    assert chord.getpixel((18, 18)) == pie.getpixel((18, 18)) == 255
    assert pie.getpixel((6, 6)) == 0
    rounded = Image.new("L", chord.size)
    ImageDraw.Draw(rounded).rounded_rectangle((2, 2, 22, 22), radius=5, corners=(False, True, True, True), fill=255)
    assert rounded.getpixel((2, 2)) == 255
    assert rounded.getpixel((22, 2)) == 0
    assert rounded.getpixel((12, 12)) == 255


def test_polygon_fill_includes_edges() -> None:
    actual = Image.new("L", (15, 15))
    expected = PILImage.new("L", actual.size)
    coords = [(2, 2), (12, 2), (12, 12), (2, 12)]
    ImageDraw.Draw(actual).polygon(coords, fill=255, outline=100, width=1)
    PILDraw.Draw(expected).polygon(coords, fill=255, outline=100, width=1)
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("operation", ["arc", "chord", "pieslice"])
def test_zero_sweep_draws_nothing(operation: str) -> None:
    image = Image.new("L", (10, 10))
    getattr(ImageDraw.Draw(image), operation)((1, 1, 8, 8), 30, 30, fill=255)
    assert image.tobytes() == bytes(100)


@pytest.mark.parametrize("border", [None, 200])
@pytest.mark.parametrize("threshold", [0, 3, 10])
def test_floodfill_matches_pillow(border: int | None, threshold: int) -> None:
    raw = bytes([200] * 7 + [200, 10, 11, 14, 19, 30, 200] * 5 + [200] * 7)
    actual = Image.frombytes("L", (7, 7), raw)
    expected = PILImage.frombytes("L", actual.size, raw)
    ImageDraw.floodfill(actual, (2, 2), 100, border=border, thresh=threshold)
    PILDraw.floodfill(expected, (2, 2), 100, border=border, thresh=threshold)
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("mode,value,background", [("RGB", (1, 2, 3), (1, 2, 4)), ("RGBA", (1, 2, 3, 4), (1, 2, 3, 5)), ("I", -7, -8), ("F", 1.25, 1.5), ("I;16B", 1000, 1001), ("P", 7, 8)])
def test_floodfill_modes_and_diagonal_separation(mode: str, value: object, background: object) -> None:
    image = Image.new(mode, (3, 3), background)
    image.putpixel((0, 0), value)
    image.putpixel((1, 1), value)
    ImageDraw.floodfill(image, (0, 0), background)
    assert image.getpixel((0, 0)) == background
    assert image.getpixel((1, 1)) == value
    ImageDraw.floodfill(image, (-1, 0), value)
    ImageDraw.floodfill(image, (3, 3), value)
    assert image.getpixel((2, 2)) == background


@pytest.mark.parametrize("mode", ["1", "L", "RGB", "RGBA", "I", "F", "P"])
def test_default_bitmap_text(mode: str) -> None:
    image = Image.new(mode, (100, 40))
    draw = ImageDraw.Draw(image)
    draw.text((2, 2), "Hello 123!", fill=1 if mode in ("1", "I", "F", "P") else "white")
    assert any(image.tobytes())
    assert draw.textlength("Hello") == 30
    assert draw.textbbox((2, 2), "Hello") == (2, 2, 32, 9)
    assert draw.textbbox((20, 20), "A", anchor="mm") == (17, 16.5, 23, 23.5)
    draw.multiline_text((2, 15), "A\nBB", fill=1, align="right")
    assert draw.multiline_textbbox((2, 15), "A\nBB", align="right") == (2, 15, 14, 33)


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
@pytest.mark.parametrize("anchor", [None, "la", "mm", "rs"])
def test_supplied_font_matches_pillow(mode: str, anchor: str | None) -> None:
    font = PILFont.load_default(size=13)
    actual = Image.new(mode, (100, 40))
    expected = PILImage.new(mode, actual.size)
    for module, image in ((ImageDraw, actual), (PILDraw, expected)):
        draw = module.Draw(image)
        draw.text((40, 20), "Hello!", font=font, fill="white", anchor=anchor)
    assert actual.tobytes() == expected.tobytes()
    draw = ImageDraw.Draw(actual)
    reference = PILDraw.Draw(expected)
    assert draw.textlength("Hello!", font=font) == reference.textlength("Hello!", font=font)
    assert draw.textbbox((40, 20), "Hello!", font=font, anchor=anchor) == reference.textbbox((40, 20), "Hello!", font=font, anchor=anchor)


@pytest.mark.parametrize("anchor", [None, "la", "mm", "rd", "ms"])
@pytest.mark.parametrize("align", ["left", "center", "right"])
def test_multiline_text_layout_and_stroke_bounds(anchor: str | None, align: str) -> None:
    font = PILFont.load_default(size=13)
    actual = Image.new("L", (100, 60))
    expected = PILImage.new("L", actual.size)
    draw = ImageDraw.Draw(actual)
    reference = PILDraw.Draw(expected)
    draw.multiline_text((40, 25), "Hello\nWorld!", fill=255, font=font, align=align, anchor=anchor, spacing=6)
    reference.multiline_text((40, 25), "Hello\nWorld!", fill=255, font=font, align=align, anchor=anchor, spacing=6)
    assert actual.tobytes() == expected.tobytes()
    assert draw.multiline_textbbox((40, 25), "Hello\nWorld!", font=font, align=align, anchor=anchor, spacing=6) == reference.multiline_textbbox(
        (40, 25), "Hello\nWorld!", font=font, align=align, anchor=anchor, spacing=6
    )
    default = Image.new("L", (30, 20))
    ImageDraw.Draw(default).text((3, 3), "A", fill=100, stroke_width=2, stroke_fill=255)
    assert {0, 100, 255} <= set(default.tobytes())
    assert ImageDraw.Draw(default).textbbox((3, 3), "A", stroke_width=2) == (1, 1, 11, 12)


@pytest.mark.parametrize("mode,draw_mode", [("L", None), ("RGB", None), ("RGBA", None), ("RGB", "RGBA")])
@pytest.mark.parametrize("stroke_fill", [None, "red"])
def test_supplied_font_stroke_and_fractional_position(mode: str, draw_mode: str | None, stroke_fill: str | None) -> None:
    font = PILFont.load_default(size=15)
    actual = Image.new(mode, (100, 40), 37)
    expected = PILImage.new(mode, actual.size, 37)
    for module, image in ((ImageDraw, actual), (PILDraw, expected)):
        module.Draw(image, draw_mode).text((20.5, 10.75), "Ab!", font=font, fill="white", stroke_fill=stroke_fill, stroke_width=2)
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("background", [(30, 80, 110, 91), (30, 80, 110, 0), (30, 80, 110, 255)])
def test_translucent_rgba_font_masks_match_pillow(background: tuple[int, ...]) -> None:
    font = PILFont.load_default(size=15)
    actual = Image.new("RGBA", (50, 30), background)
    expected = PILImage.new("RGBA", actual.size, background)
    for module, image in ((ImageDraw, actual), (PILDraw, expected)):
        module.Draw(image).text((3, 3), "Ab", fill=(211, 20, 141, 128), font=font)
    assert actual.tobytes() == expected.tobytes()


def test_invalid_geometry_and_text() -> None:
    draw = ImageDraw.Draw(Image.new("L", (10, 10)))
    with pytest.raises(ValueError):
        draw.regular_polygon((5, 5, -1), 4)
    with pytest.raises(ValueError):
        draw.polygon([(1, 1)])
    with pytest.raises(ValueError):
        draw.ellipse((5, 5, 2, 2), fill=255)
    with pytest.raises(ValueError):
        draw.arc((0, 0, 5, 5), float("nan"), 90)
    with pytest.raises(ValueError):
        draw.textlength("A\nB")
    with pytest.raises(ValueError):
        draw.multiline_text((0, 0), "A\nB", align="invalid")
    with pytest.raises(ValueError):
        draw.text((0, 0), "A", stroke_width=-1)
    with pytest.raises(ValueError):
        ImageDraw.floodfill(draw.im, (0, 0), 1, thresh=-1)
