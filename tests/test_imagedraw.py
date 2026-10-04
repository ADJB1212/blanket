from __future__ import annotations

import pytest
from PIL import Image as PILImage, ImageDraw as PILDraw

from blanket import Image, ImageDraw


@pytest.mark.parametrize(
    "mode,ink",
    [
        ("1", 1),
        ("L", 117),
        ("LA", (117, 81)),
        ("RGB", (27, 139, 241)),
        ("RGBA", (27, 139, 241, 81)),
        ("HSV", (27, 139, 241)),
        ("CMYK", (27, 139, 241, 81)),
        ("YCbCr", (27, 139, 241)),
        ("LAB", (27, 139, 241)),
        ("P", 7),
        ("PA", (7, 81)),
        ("I", -123456),
        ("I;16", 12000),
        ("I;16L", 12000),
        ("I;16B", 12000),
        ("F", 1.25),
    ],
)
@pytest.mark.parametrize("operation", ["point", "rectangle"])
def test_primitives_match_pillow(mode: str, ink: object, operation: str) -> None:
    actual = Image.new(mode, (13, 11))
    expected = PILImage.new(mode, actual.size)
    points = [(-2.5, 1), (3.9, 4.9), (12, 10), (13, 11)] if operation == "point" else [(-2.5, 1), (8.9, 7.9)]
    getattr(ImageDraw.Draw(actual), operation)(points, fill=ink)
    getattr(PILDraw.Draw(expected), operation)(points, fill=ink)
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("width", [-1, 0, 1, 2, 6, 30])
@pytest.mark.parametrize("bounds", [(2, 2, 8, 7), (-3, -2, 4, 5), (4, 4, 4, 4), (20, 20, 30, 30)])
@pytest.mark.parametrize("blend", [False, True])
def test_rectangle_outline_matches_pillow(width: int, bounds: tuple[int, ...], blend: bool) -> None:
    actual = Image.new("RGB", (13, 11), (81, 93, 117))
    expected = PILImage.new("RGB", actual.size, (81, 93, 117))
    mode = "RGBA" if blend else "RGB"
    fill = (31, 173, 59, 83) if blend else "green"
    outline = (237, 41, 121, 129) if blend else "red"
    ImageDraw.Draw(actual, mode).rectangle(bounds, fill=fill, outline=outline, width=width)
    PILDraw.Draw(expected, mode).rectangle(bounds, fill=fill, outline=outline, width=width)
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("mode", ["1", "L", "LA", "RGB", "RGBA", "CMYK", "HSV", "YCbCr", "LAB", "I", "I;16", "I;16L", "I;16B", "F", "P", "PA"])
def test_default_point_ink(mode: str) -> None:
    actual = Image.new(mode, (2, 2))
    expected = PILImage.new(mode, actual.size)
    ImageDraw.Draw(actual).point((0, 0))
    PILDraw.Draw(expected).point((0, 0))
    assert actual.tobytes() == expected.tobytes()


def test_point_blending_and_flat_coordinates() -> None:
    actual = Image.new("RGB", (3, 3), "blue")
    expected = PILImage.new("RGB", actual.size, "blue")
    points = [-1, 0, 1, 1, 1, 1, 2, 2]
    ImageDraw.Draw(actual, "RGBA").point(points, fill=(231, 37, 93, 128))
    PILDraw.Draw(expected, "RGBA").point(points, fill=(231, 37, 93, 128))
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("image_mode,draw_mode", [("RGBA", "RGB"), ("L", "RGBA"), ("RGB", "L")])
def test_context_rejects_mode_mismatch(image_mode: str, draw_mode: str) -> None:
    with pytest.raises(ValueError, match="mode mismatch"):
        ImageDraw.Draw(Image.new(image_mode, (2, 2)), draw_mode)


@pytest.mark.parametrize("bounds", [(4, 0, 2, 3), (0, 4, 2, 3), (0, 1), (0, 1, 2)])
def test_invalid_rectangle_does_not_modify_image(bounds: tuple[int, ...]) -> None:
    image = Image.new("RGB", (5, 5), "blue")
    before = image.tobytes()
    with pytest.raises(ValueError):
        ImageDraw.Draw(image).rectangle(bounds, fill="red")
    assert image.tobytes() == before


def test_empty_image_drawing() -> None:
    image = Image.new("RGB", (0, 0))
    draw = ImageDraw.Draw(image)
    draw.point([])
    draw.rectangle((-10, -10, 10, 10), fill="red")
    assert image.tobytes() == b""
    draw.ellipse((0, 0, 2, 2), fill="red")
    draw.line((0, 0, 2, 2), fill="red")
    draw.text((0, 0), "A", fill="red")
    ImageDraw.floodfill(image, (0, 0), "red")
    assert image.tobytes() == b""


@pytest.mark.parametrize("operation", ["point", "rectangle"])
def test_multichannel_high_depth_requires_conversion(operation: str) -> None:
    image = Image.frombytes("RGB", (2, 2), bytes(24), bit_depth=16)
    before = image.tobytes()
    points = (0, 0) if operation == "point" else (0, 0, 1, 1)
    with pytest.raises(ValueError, match="8-bit"):
        getattr(ImageDraw.Draw(image), operation)(points, fill="red")
    assert image.tobytes() == before


def test_drawing_on_closed_image_fails() -> None:
    image = Image.new("RGB", (2, 2))
    draw = ImageDraw.Draw(image)
    image.close()
    with pytest.raises(ValueError):
        draw.point((0, 0), fill="red")
