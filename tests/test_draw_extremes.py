import pytest

from blanket import Image, ImageDraw


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
@pytest.mark.parametrize("shape", ["polygon", "line", "ellipse"])
def test_drawing_clips_coordinates_across_the_i64_range(mode: str, shape: str) -> None:
    image = Image.new(mode, (3, 3))
    draw = ImageDraw.Draw(image)
    low, high = -(1 << 63), (1 << 63) - 1
    if shape == "polygon":
        draw.polygon([(low, low), (high, low), (high, high), (low, high)], fill="white")
    elif shape == "line":
        draw.line([(low, 1), (high, 1)], fill="white", width=3)
    else:
        draw.ellipse((low, low, high, high), fill="white")
    assert image.tobytes() == Image.new(mode, image.size, "white").tobytes()


@pytest.mark.parametrize("origin", [-(1 << 63), (1 << 63) - 11])
def test_ellipse_outside_canvas_with_extreme_origin_and_width(origin: int) -> None:
    image = Image.new("L", (3, 3))
    ImageDraw.Draw(image).ellipse((origin, origin, origin + 10, origin + 10), outline=255, width=(1 << 63) - 1)
    assert image.tobytes() == bytes(9)


def test_curve_rejects_unrepresentable_extents_without_panicking() -> None:
    image = Image.new("L", (3, 3))
    low, high = -(1 << 63), (1 << 63) - 1
    with pytest.raises(ValueError, match="ellipse bounds are too large"):
        ImageDraw.Draw(image).arc((low, low, high, high), 0, 90, fill=255)


def test_ellipse_extreme_stroke_width_matches_filled_ellipse() -> None:
    actual = Image.new("L", (7, 7))
    expected = actual.copy()
    ImageDraw.Draw(actual).ellipse((1, 1, 5, 5), outline=255, width=(1 << 63) - 1)
    ImageDraw.Draw(expected).ellipse((1, 1, 5, 5), fill=255)
    assert actual.tobytes() == expected.tobytes()
