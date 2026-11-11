"""Pillow 13.0.0 drawing contracts and Blanket regressions; see PILLOW_TESTS.md."""

from __future__ import annotations

from pillow_helpers import PILLOW_IMAGES

PILLOW_IMAGE_PREFIX = str(PILLOW_IMAGES) + "/"

import os.path

import pytest
from helpers import assert_drawing_equal, assert_raises, image_pair, new_pair
from pillow_helpers import assert_image_equal, assert_image_equal_tofile, assert_image_similar_tofile, hopper

from blanket import Image, ImageColor, ImageDraw

TYPE_CHECKING = False
if TYPE_CHECKING:
    from collections.abc import Sequence

    from blanket._typing import Coords

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (190, 190, 190)
DEFAULT_MODE = "RGB"
IMAGES_PATH = os.path.join(PILLOW_IMAGES, "imagedraw")

# Image size
W, H = 100, 100

# Bounding box points
X0 = int(W / 4)
X1 = int(X0 * 3)
Y0 = int(H / 4)
Y1 = int(X0 * 3)

# Bounding boxes
BBOX = (((X0, Y0), (X1, Y1)), [(X0, Y0), (X1, Y1)], (X0, Y0, X1, Y1), [X0, Y0, X1, Y1])

# Coordinate sequences
POINTS = (((10, 10), (20, 40), (30, 30)), [(10, 10), (20, 40), (30, 30)], ([10, 10], [20, 40], [30, 30]), [[10, 10], [20, 40], [30, 30]], (10, 10, 20, 40, 30, 30), [10, 10, 20, 40, 30, 30])

KITE_POINTS = (((10, 50), (70, 10), (90, 50), (70, 90), (10, 50)), [(10, 50), (70, 10), (90, 50), (70, 90), (10, 50)], ([10, 50], [70, 10], [90, 50], [70, 90], [10, 50]), [[10, 50], [70, 10], [90, 50], [70, 90], [10, 50]])


def assert_draw_reference(filename, method, *args, mode="RGB", size=(W, H), background=0, draw_mode=None, epsilon=0, **options):
    image = Image.new(mode, size, background)
    getattr(ImageDraw.Draw(image, draw_mode), method)(*args, **options)
    if epsilon:
        assert_image_similar_tofile(image, filename, epsilon)
    else:
        assert_image_equal_tofile(image, filename)


@pytest.mark.parametrize("bbox", (BBOX[0], BBOX[3]))
@pytest.mark.parametrize(
    "method,args,options,draw_mode,filename,epsilon",
    [
        pytest.param("arc", (), {"start": 270.5, "end": 0}, None, "imagedraw_arc_end_le_start.png", 0, id="arc_end_le_start"),
        pytest.param("arc", (), {"start": 5, "end": 370}, None, "imagedraw_arc_no_loops.png", 1, id="arc_no_loops"),
        pytest.param("arc", (10, 260), {"width": 5}, None, "imagedraw_arc_width.png", 1, id="arc_width"),
        pytest.param("arc", (10, 260), {"fill": "yellow", "width": 100}, None, "imagedraw_arc_width_pieslice.png", 1, id="arc_width_pieslice_large"),
        pytest.param("arc", (10, 260), {"fill": "yellow", "width": 5}, None, "imagedraw_arc_width_fill.png", 1, id="arc_width_fill"),
        pytest.param("arc", (10, 259.5), {"width": 5}, None, "imagedraw_arc_width_non_whole_angle.png", 1, id="arc_width_non_whole_angle"),
        pytest.param("chord", (10, 260), {"outline": "yellow", "width": 5}, None, "imagedraw_chord_width.png", 1, id="chord_width"),
        pytest.param("chord", (10, 260), {"fill": "red", "outline": "yellow", "width": 5}, None, "imagedraw_chord_width_fill.png", 1, id="chord_width_fill"),
        pytest.param("chord", (10, 260), {"fill": "red", "outline": "yellow", "width": 0}, None, "imagedraw_chord_zero_width.png", 0, id="chord_zero_width"),
        pytest.param("ellipse", (), {"fill": (0, 255, 0, 127)}, "RGBA", "imagedraw_ellipse_translucent.png", 1, id="ellipse_translucent"),
        pytest.param("ellipse", (), {"outline": "blue", "width": 5}, None, "imagedraw_ellipse_width.png", 1, id="ellipse_width"),
        pytest.param("ellipse", (), {"fill": "green", "outline": "blue", "width": 5}, None, "imagedraw_ellipse_width_fill.png", 1, id="ellipse_width_fill"),
        pytest.param("ellipse", (), {"fill": "green", "outline": "blue", "width": 0}, None, "imagedraw_ellipse_zero_width.png", 0, id="ellipse_zero_width"),
        pytest.param("pieslice", (10, 260), {"outline": "blue", "width": 5}, None, "imagedraw_pieslice_width.png", 1, id="pieslice_width"),
        pytest.param("pieslice", (10, 260), {"fill": "white", "outline": "blue", "width": 5}, None, "imagedraw_pieslice_width_fill.png", 1, id="pieslice_width_fill"),
        pytest.param("pieslice", (10, 260), {"fill": "white", "outline": "blue", "width": 0}, None, "imagedraw_pieslice_zero_width.png", 0, id="pieslice_zero_width"),
        pytest.param("rectangle", (), {"fill": "black", "outline": "green"}, None, "imagedraw_rectangle.png", 0, id="rectangle"),
        pytest.param("rectangle", (), {"outline": "green", "width": 5}, None, "imagedraw_rectangle_width.png", 0, id="rectangle_width"),
        pytest.param("rectangle", (), {"fill": "blue", "outline": "green", "width": 5}, None, "imagedraw_rectangle_width_fill.png", 0, id="rectangle_width_fill"),
        pytest.param("rectangle", (), {"fill": "blue", "outline": "green", "width": 0}, None, "imagedraw_rectangle_zero_width.png", 0, id="rectangle_zero_width"),
        pytest.param("rectangle", (), {"fill": "black", "outline": (0, 255, 0, 127), "width": 5}, "RGBA", "imagedraw_rectangle_translucent_outline.png", 0, id="rectangle_translucent_outline"),
        pytest.param("rounded_rectangle", (0,), {"fill": "blue", "outline": "green", "width": 5}, None, "imagedraw_rectangle_width_fill.png", 0, id="rounded_rectangle_zero_radius"),
    ],
)
def test_shapes_match_reference(bbox: Coords, method: str, args: tuple[float, ...], options: dict[str, object], draw_mode: str | None, filename: str, epsilon: float) -> None:
    image = Image.new("RGB", (W, H))
    draw = ImageDraw.Draw(image, draw_mode)
    getattr(draw, method)(bbox, *args, **options)
    reference = PILLOW_IMAGES / filename
    if epsilon:
        assert_image_similar_tofile(image, reference, epsilon)
    else:
        assert_image_equal_tofile(image, reference)


@pytest.mark.parametrize(
    "size,draw_mode,operation,filename,epsilon",
    [
        pytest.param((100, 100), None, lambda draw: draw.chord([-150, -150, 99, 99], 15, 60, width=10, fill="white", outline="red"), "imagedraw_chord_too_fat.png", 0, id="chord_too_fat"),
        pytest.param((W, H), None, lambda draw: draw.ellipse(((0, 0), (W - 1, H - 1)), fill="white"), "imagedraw_ellipse_edge.png", 1, id="ellipse_edge"),
        pytest.param((500, 500), None, lambda draw: draw.ellipse((25, 25, 475, 475), outline="blue", width=75), "imagedraw_ellipse_width_large.png", 1, id="ellipse_width_large"),
        pytest.param((200, 100), None, lambda draw: draw.pieslice([0, 0, 199, 99], 190, 170, width=10, fill="white", outline="red"), "imagedraw_pieslice_wide.png", 0, id="pieslice_wide"),
        pytest.param((3, 3), None, lambda draw: draw.polygon([(0, 1), (0, 1), (2, 1), (2, 1)], "#f00"), "imagedraw_polygon_1px_high.png", 0, id="polygon_1px_high"),
        pytest.param((4, 3), "RGBA", lambda draw: draw.polygon([(1, 1), (1, 1), (3, 1), (3, 1)], (255, 0, 0, 127)), "imagedraw_polygon_1px_high_translucent.png", 0, id="polygon_1px_high_translucent"),
        pytest.param((W, H), "RGBA", lambda draw: draw.polygon([(20, 80), (80, 80), (80, 20)], fill=(0, 255, 0, 127)), "imagedraw_polygon_translucent.png", 0, id="polygon_translucent"),
        pytest.param(
            (W, H),
            "RGBA",
            lambda draw: draw.rounded_rectangle((20, 10, 80, 90), 30, fill="red", outline="green", width=5, corners=(True, False, False, False)),
            "imagedraw_rounded_rectangle_joined_x_different_corners.png",
            0,
            id="rounded_rectangle_joined_x_different_corners",
        ),
        pytest.param((W, H), "RGB", lambda draw: draw.rounded_rectangle((25, 25, 75, 75), 24, fill="red", outline="green", width=5), "imagedraw_rounded_rectangle_radius.png", 0, id="rounded_rectangle_radius"),
        pytest.param((W, H), None, lambda draw: draw.line([(50, 50), (50, 50)], width=3), "imagedraw_wide_line_dot.png", 1, id="wide_line_dot"),
        pytest.param((W, H), None, lambda draw: draw.line([(0, 0), (32768, 32768)], width=3), "imagedraw_wide_line_larger_than_int.png", 1, id="wide_line_larger_than_int"),
        pytest.param((W, H), None, lambda draw: draw.polygon([(18, 30), (19, 31), (18, 30), (85, 30), (60, 72)], "red"), "imagedraw_outline_polygon_RGB.png", 1, id="polygon2"),
    ],
)
def test_drawing_reference_cases(size, draw_mode, operation, filename, epsilon) -> None:
    image = Image.new("RGB", size)
    operation(ImageDraw.Draw(image, draw_mode))
    if epsilon:
        assert_image_similar_tofile(image, PILLOW_IMAGES / filename, epsilon)
    else:
        assert_image_equal_tofile(image, PILLOW_IMAGES / filename)


def test_sanity() -> None:
    im = hopper("RGB").copy()

    draw = ImageDraw.ImageDraw(im)
    draw = ImageDraw.Draw(im)

    draw.ellipse(list(range(4)))
    draw.line(list(range(10)))
    draw.polygon(list(range(100)))
    draw.rectangle(list(range(4)))


def test_new_color() -> None:
    with Image.open(PILLOW_IMAGE_PREFIX + "chi.gif") as im:
        draw = ImageDraw.Draw(im)
        assert im.palette is not None
        assert len(im.palette.colors) == 249

        # Test drawing a new color onto the palette
        draw.line((0, 0), fill=(0, 0, 0))
        assert im.palette is not None
        assert len(im.palette.colors) == 250
        assert im.palette.dirty

        # Test drawing another new color, now that the palette is dirty
        draw.point((0, 0), fill=(1, 0, 0))
        assert im.convert("RGB").getpixel((0, 0)) == (1, 0, 0)
        # Since line() did not draw any pixels, the same palette index was re-used
        assert len(im.palette.colors) == 250
        assert (0, 0, 0) not in im.palette.colors


def test_mode_mismatch() -> None:
    im = hopper("RGB").copy()

    assert_raises(ValueError, lambda: ImageDraw.ImageDraw(im, mode="L"))


@pytest.mark.parametrize("bbox", BBOX)
@pytest.mark.parametrize("start, end", ((0, 180), (0.5, 180.4)))
def test_arc(bbox: Coords, start: float, end: float) -> None:
    assert_draw_reference(PILLOW_IMAGE_PREFIX + "imagedraw_arc.png", "arc", bbox, start, end, epsilon=1)


def test_arc_high() -> None:
    im = Image.new("RGB", (200, 200))
    draw = ImageDraw.Draw(im)
    draw.arc([10, 10, 89, 189], 20, 330, width=20, fill="white")
    draw.arc([110, 10, 189, 189], 20, 150, width=20, fill="white")
    assert_image_equal_tofile(im, (PILLOW_IMAGE_PREFIX + "imagedraw_arc_high.png"))


def test_bitmap() -> None:
    im = Image.new("RGB", (W, H))
    draw = ImageDraw.Draw(im)
    with Image.open(PILLOW_IMAGE_PREFIX + "pil123rgba.png") as small:
        small_resized = small.resize((50, 50), Image.Resampling.NEAREST)
        draw.bitmap((10, 10), small_resized)
    assert_image_equal_tofile(im, (PILLOW_IMAGE_PREFIX + "imagedraw_bitmap.png"))


@pytest.mark.parametrize("mode", ("RGB", "L"))
@pytest.mark.parametrize("bbox", BBOX)
def test_chord(mode: str, bbox: Coords) -> None:
    assert_draw_reference(PILLOW_IMAGE_PREFIX + f"imagedraw_chord_{mode}.png", "chord", bbox, 0, 180, fill="red", outline="yellow", mode=mode, epsilon=1)


@pytest.mark.parametrize("mode", ("RGB", "L"))
@pytest.mark.parametrize("xy", ((W / 2, H / 2), [W / 2, H / 2]))
def test_circle(mode: str, xy: Sequence[float]) -> None:
    assert_draw_reference(PILLOW_IMAGE_PREFIX + f"imagedraw_ellipse_{mode}.png", "circle", xy, 25, fill="green", outline="blue", mode=mode, epsilon=1)


@pytest.mark.parametrize("mode", ("RGB", "L"))
@pytest.mark.parametrize("bbox", BBOX)
def test_ellipse(mode: str, bbox: Coords) -> None:
    assert_draw_reference(PILLOW_IMAGE_PREFIX + f"imagedraw_ellipse_{mode}.png", "ellipse", bbox, fill="green", outline="blue", mode=mode, epsilon=1)


def test_ellipse_symmetric() -> None:
    for width, bbox in ((100, (24, 24, 75, 75)), (101, (25, 25, 75, 75))):
        im = Image.new("RGB", (width, 100))
        draw = ImageDraw.Draw(im)
        draw.ellipse(bbox, fill="green", outline="blue")
        assert_image_equal(im, im.transpose(Image.Transpose.FLIP_LEFT_RIGHT))


def ellipse_various_sizes_helper(filled: bool) -> Image.Image:
    ellipse_sizes = range(32)
    image_size = sum(ellipse_sizes) + len(ellipse_sizes) + 1
    im = Image.new("RGB", (image_size, image_size))
    draw = ImageDraw.Draw(im)

    x = 1
    for w in ellipse_sizes:
        y = 1
        for h in ellipse_sizes:
            x1 = x + w
            if w:
                x1 -= 1
            y1 = y + h
            if h:
                y1 -= 1
            border = [x, y, x1, y1]
            if filled:
                draw.ellipse(border, fill="white")
            else:
                draw.ellipse(border, outline="white")
            y += h + 1
        x += w + 1

    return im


@pytest.mark.parametrize("filled,suffix", [(False, ""), (True, "_filled")])
def test_ellipse_various_sizes(filled, suffix) -> None:
    assert_image_equal_tofile(ellipse_various_sizes_helper(filled), PILLOW_IMAGES / f"imagedraw_ellipse_various_sizes{suffix}.png")


@pytest.mark.parametrize("points", POINTS)
def test_line(points: Coords) -> None:
    assert_draw_reference(PILLOW_IMAGE_PREFIX + "imagedraw_line.png", "line", points, fill="yellow", width=2)


def curve_outline(points=((5, 5), (5, 50), (95, 50), (95, 5))):
    outline = ImageDraw.Outline()
    outline.move(*points[0])
    outline.curve(*(value for point in points[1:] for value in point))
    outline.line(*points[0])
    return outline


@pytest.mark.parametrize("points,options,filename", [(((5, 5), (5, 50), (95, 50), (95, 5)), {"fill": 1}, "imagedraw_shape1.png"), (((95, 95), (95, 50), (5, 50), (5, 95)), {"outline": "blue"}, "imagedraw_shape2.png")])
def test_shape(points, options, filename) -> None:
    assert_draw_reference(PILLOW_IMAGES / filename, "shape", curve_outline(points), background="white", **options)


@pytest.mark.parametrize("bbox", BBOX)
@pytest.mark.parametrize("start, end", ((-92, 46), (-92.2, 46.2)))
def test_pieslice(bbox: Coords, start: float, end: float) -> None:
    assert_draw_reference(PILLOW_IMAGE_PREFIX + "imagedraw_pieslice.png", "pieslice", bbox, start, end, fill="white", outline="blue", epsilon=1)


def test_pieslice_no_spikes() -> None:
    im = Image.new("RGB", (161, 161), "white")
    draw = ImageDraw.Draw(im)
    cxs = [140] * 3 + list(range(140, 19, -20)) + [20] * 5 + list(range(20, 141, 20)) + [140] * 2
    cys = list(range(80, 141, 20)) + [140] * 5 + list(range(140, 19, -20)) + [20] * 5 + list(range(20, 80, 20))

    for cx, cy, angle in zip(cxs, cys, range(0, 360, 15), strict=False):
        draw.pieslice([cx - 100, cy - 100, cx + 100, cy + 100], angle, angle + 1, fill="black")
        draw.point([cx, cy], fill="red")

    im_pre_erase = im.copy()
    draw.rectangle([21, 21, 139, 139], fill="white")

    assert_image_equal(im, im_pre_erase)


@pytest.mark.parametrize("points", POINTS)
def test_point(points: Coords) -> None:
    assert_draw_reference(PILLOW_IMAGE_PREFIX + "imagedraw_point.png", "point", points, fill="yellow")


@pytest.mark.parametrize("points", POINTS)
def test_polygon(points: Coords) -> None:
    assert_draw_reference(PILLOW_IMAGE_PREFIX + "imagedraw_polygon.png", "polygon", points, fill="red", outline="blue")


@pytest.mark.parametrize("mode", ("RGB", "L"))
@pytest.mark.parametrize("kite_points", KITE_POINTS)
def test_polygon_kite(mode: str, kite_points: tuple[tuple[int, int], ...] | list[tuple[int, int]]) -> None:
    # Test drawing lines of different gradients (dx>dy, dy>dx) and
    # vertical (dx==0) and horizontal (dy==0) lines
    assert_draw_reference(PILLOW_IMAGE_PREFIX + f"imagedraw_polygon_kite_{mode}.png", "polygon", kite_points, fill="blue", outline="yellow", mode=mode)


def test_big_rectangle() -> None:
    # Test drawing a rectangle bigger than the image
    im = Image.new("RGB", (W, H))
    bbox = [(-1, -1), (W + 1, H + 1)]
    draw = ImageDraw.Draw(im)
    draw.rectangle(bbox, fill="orange")
    assert_image_similar_tofile(im, (PILLOW_IMAGE_PREFIX + "imagedraw_big_rectangle.png"), 1)


@pytest.mark.parametrize("xy", [(10, 20, 190, 180), ([10, 20], [190, 180]), ((10, 20), (190, 180))])
def test_rounded_rectangle(xy: (tuple[int, int, int, int] | tuple[list[int]] | tuple[tuple[int, int], tuple[int, int]])) -> None:
    assert_draw_reference(PILLOW_IMAGE_PREFIX + "imagedraw_rounded_rectangle.png", "rounded_rectangle", xy, 30, fill="red", outline="green", width=5, size=(200, 200))


@pytest.mark.parametrize("top_left", (True, False))
@pytest.mark.parametrize("top_right", (True, False))
@pytest.mark.parametrize("bottom_right", (True, False))
@pytest.mark.parametrize("bottom_left", (True, False))
def test_rounded_rectangle_corners(top_left: bool, top_right: bool, bottom_right: bool, bottom_left: bool) -> None:
    corners = (top_left, top_right, bottom_right, bottom_left)
    im = Image.new("RGB", (200, 200))
    draw = ImageDraw.Draw(im)
    draw.rounded_rectangle((10, 20, 190, 180), 30, fill="red", outline="green", width=5, corners=corners)
    suffix = "".join((("y" if top_left else "n"), ("y" if top_right else "n"), ("y" if bottom_right else "n"), ("y" if bottom_left else "n")))
    assert_image_equal_tofile(im, (PILLOW_IMAGE_PREFIX + "imagedraw_rounded_rectangle_corners_") + suffix + ".png")


@pytest.mark.parametrize("xy, radius, type", [((10, 20, 190, 180), 30.5, "given"), ((10, 10, 181, 190), 90, "width"), ((10, 20, 190, 181), 85, "height")])
def test_rounded_rectangle_non_integer_radius(xy: tuple[int, int, int, int], radius: float, type: str) -> None:
    assert_draw_reference(PILLOW_IMAGE_PREFIX + "imagedraw_rounded_rectangle_non_integer_radius_" + type + ".png", "rounded_rectangle", xy, radius, fill="red", outline="green", width=5, size=(200, 200))


@pytest.mark.parametrize("w, h", ((200, 100), (100, 200)))
def test_rounded_rectangle_large_radius(w: int, h: int) -> None:
    im = Image.new("RGB", (w, h))
    draw = ImageDraw.Draw(im)
    draw.rounded_rectangle((0, 0, w, h), 100, "red", corners=(True, False, False, False))

    expected = Image.new("RGB", (w, h))
    draw = ImageDraw.Draw(expected)
    draw.rounded_rectangle((0, 0, w, h), 50, "red", corners=(True, False, False, False))

    assert_image_equal(im, expected)


@pytest.mark.parametrize("xy, suffix", [((20, 10, 80, 90), "x"), ((20, 10, 81, 90), "x_odd"), ((20, 10, 81.1, 90), "x_odd"), ((10, 20, 90, 80), "y"), ((10, 20, 90, 81), "y_odd"), ((10, 20, 90, 81.1), "y_odd"), ((20, 20, 80, 80), "both")])
def test_rounded_rectangle_translucent(xy: tuple[int, int, int, int], suffix: str) -> None:
    assert_draw_reference(PILLOW_IMAGE_PREFIX + "imagedraw_rounded_rectangle_" + suffix + ".png", "rounded_rectangle", xy, 30, fill=(255, 0, 0, 127), outline=(0, 255, 0, 127), width=5, draw_mode="RGBA")


@pytest.mark.parametrize("bbox", BBOX)
def test_floodfill(bbox: Coords) -> None:
    red = ImageColor.getrgb("red")

    mode_values: list[tuple[str, int | tuple[int, ...]]] = [("L", 1), ("RGBA", (255, 0, 0, 0)), ("RGB", red)]
    for mode, value in mode_values:
        im = Image.new(mode, (W, H))
        draw = ImageDraw.Draw(im)
        draw.rectangle(bbox, outline="yellow", fill="green")
        centre_point = (int(W / 2), int(H / 2))
        ImageDraw.floodfill(im, centre_point, value)
        expected = (PILLOW_IMAGE_PREFIX + "imagedraw_floodfill_") + mode + ".png"
        with Image.open(expected) as im_floodfill:
            assert_image_equal(im, im_floodfill)

    # Test that using the same colour does not change the image
    ImageDraw.floodfill(im, centre_point, red)
    assert_image_equal(im, im_floodfill)

    # Test that filling outside the image does not change the image
    ImageDraw.floodfill(im, (W, H), red)
    assert_image_equal(im, im_floodfill)

    # Test filling at the edge of an image
    im = Image.new("RGB", (1, 1))
    ImageDraw.floodfill(im, (0, 0), red)
    assert_image_equal(im, Image.new("RGB", (1, 1), red))


@pytest.mark.parametrize("bbox", BBOX)
def test_floodfill_border(bbox: Coords) -> None:
    # floodfill() is experimental
    im = Image.new("RGB", (W, H))
    draw = ImageDraw.Draw(im)
    draw.rectangle(bbox, outline="yellow", fill="green")
    centre_point = (int(W / 2), int(H / 2))
    ImageDraw.floodfill(im, centre_point, ImageColor.getrgb("red"), border=ImageColor.getrgb("black"))
    assert_image_equal_tofile(im, (PILLOW_IMAGE_PREFIX + "imagedraw_floodfill2.png"))


@pytest.mark.parametrize("bbox", BBOX)
def test_floodfill_thresh(bbox: Coords) -> None:
    # floodfill() is experimental
    im = Image.new("RGB", (W, H))
    draw = ImageDraw.Draw(im)
    draw.rectangle(bbox, outline="darkgreen", fill="green")
    centre_point = (int(W / 2), int(H / 2))
    ImageDraw.floodfill(im, centre_point, ImageColor.getrgb("red"), thresh=30)
    assert_image_equal_tofile(im, (PILLOW_IMAGE_PREFIX + "imagedraw_floodfill2.png"))


def test_floodfill_not_negative() -> None:
    # floodfill() is experimental
    # Test that floodfill does not extend into negative coordinates
    im = Image.new("RGB", (W, H))
    draw = ImageDraw.Draw(im)
    draw.line((W / 2, 0, W / 2, H / 2), fill="green")
    draw.line((0, H / 2, W / 2, H / 2), fill="green")
    ImageDraw.floodfill(im, (int(W / 4), int(H / 4)), ImageColor.getrgb("red"))
    assert_image_equal_tofile(im, (PILLOW_IMAGE_PREFIX + "imagedraw_floodfill_not_negative.png"))


def create_base_image_draw(size: tuple[int, int], mode: str = DEFAULT_MODE, background1: tuple[int, int, int] = WHITE, background2: tuple[int, int, int] = GRAY) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new(mode, size, background1)
    for x in range(size[0]):
        for y in range(size[1]):
            if (x + y) % 2 == 0:
                img.putpixel((x, y), background2)
    return img, ImageDraw.Draw(img)


@pytest.mark.parametrize(
    "size,operation,filename",
    [
        pytest.param((10, 10), lambda draw: draw.polygon([(2, 2), (2, 7), (7, 7), (7, 2)], BLACK), "square.png", id="square_1"),
        pytest.param((10, 10), lambda draw: draw.polygon([(7, 7), (7, 2), (2, 2), (2, 7)], BLACK), "square.png", id="square_2"),
        pytest.param((10, 10), lambda draw: draw.rectangle((2, 2, 7, 7), BLACK), "square.png", id="square_3"),
        pytest.param((20, 20), lambda draw: draw.polygon([(3, 5), (17, 5), (10, 12)], BLACK), "triangle_right.png", id="triangle_right_1"),
        pytest.param((20, 20), lambda draw: draw.line((5, 5, 14, 5), BLACK, 2), "line_horizontal_w2px_normal.png", id="line_horizontal_1"),
        pytest.param((20, 20), lambda draw: draw.line((14, 5, 5, 5), BLACK, 2), "line_horizontal_w2px_inverted.png", id="line_horizontal_2"),
        pytest.param((20, 20), lambda draw: draw.line((5, 5, 14, 5), BLACK, 3), "line_horizontal_w3px.png", id="line_horizontal_3"),
        pytest.param((20, 20), lambda draw: draw.line((14, 5, 5, 5), BLACK, 3), "line_horizontal_w3px.png", id="line_horizontal_4"),
        pytest.param((200, 110), lambda draw: draw.line((5, 55, 195, 55), BLACK, 101), "line_horizontal_w101px.png", id="line_horizontal_5"),
        pytest.param((20, 20), lambda draw: draw.line((5, 5, 5, 14), BLACK, 2), "line_vertical_w2px_normal.png", id="line_vertical_1"),
        pytest.param((20, 20), lambda draw: draw.line((5, 14, 5, 5), BLACK, 2), "line_vertical_w2px_inverted.png", id="line_vertical_2"),
        pytest.param((20, 20), lambda draw: draw.line((5, 5, 5, 14), BLACK, 3), "line_vertical_w3px.png", id="line_vertical_3"),
        pytest.param((20, 20), lambda draw: draw.line((5, 14, 5, 5), BLACK, 3), "line_vertical_w3px.png", id="line_vertical_4"),
        pytest.param((110, 200), lambda draw: draw.line((55, 5, 55, 195), BLACK, 101), "line_vertical_w101px.png", id="line_vertical_5"),
        pytest.param((20, 20), lambda draw: draw.line((5, 5, 6, 14), BLACK, 2), "line_vertical_slope1px_w2px.png", id="line_vertical_6"),
        pytest.param((20, 20), lambda draw: draw.line((5, 5, 14, 14), BLACK, 3), "line_oblique_45_w3px_a.png", id="line_oblique_45_1"),
        pytest.param((20, 20), lambda draw: draw.line((14, 14, 5, 5), BLACK, 3), "line_oblique_45_w3px_a.png", id="line_oblique_45_2"),
        pytest.param((20, 20), lambda draw: draw.line((14, 5, 5, 14), BLACK, 3), "line_oblique_45_w3px_b.png", id="line_oblique_45_3"),
        pytest.param((20, 20), lambda draw: draw.line((5, 14, 14, 5), BLACK, 3), "line_oblique_45_w3px_b.png", id="line_oblique_45_4"),
    ],
)
def test_line_and_polygon_references(size, operation, filename) -> None:
    image, draw = create_base_image_draw(size)
    operation(draw)
    assert_image_equal_tofile(image, os.path.join(IMAGES_PATH, filename))


@pytest.mark.parametrize("fill, suffix", ((BLACK, "width"), (None, "width_no_fill")))
def test_triangle_right_width(fill: tuple[int, int, int] | None, suffix: str) -> None:
    img, draw = create_base_image_draw((100, 100))
    draw.polygon([(15, 25), (85, 25), (50, 60)], fill, WHITE, width=5)
    assert_image_equal_tofile(img, os.path.join(IMAGES_PATH, "triangle_right_" + suffix + ".png"))


@pytest.mark.parametrize(
    "xy",
    [
        [(400, 280), (380, 280), (450, 280), (440, 120), (350, 200), (310, 280), (300, 280), (250, 280), (250, 200), (150, 200), (150, 260), (50, 200), (150, 50), (250, 100)],
        (400, 280, 380, 280, 450, 280, 440, 120, 350, 200, 310, 280, 300, 280, 250, 280, 250, 200, 150, 200, 150, 260, 50, 200, 150, 50, 250, 100),
        [400, 280, 380, 280, 450, 280, 440, 120, 350, 200, 310, 280, 300, 280, 250, 280, 250, 200, 150, 200, 150, 260, 50, 200, 150, 50, 250, 100],
    ],
)
def test_line_joint(xy: list[tuple[int, int]] | tuple[int, ...] | list[int]) -> None:
    assert_draw_reference(PILLOW_IMAGE_PREFIX + "imagedraw_line_joint_curve.png", "line", xy, GRAY, 50, "curve", size=(500, 325), epsilon=3)


def test_textsize_empty_string() -> None:
    # https://github.com/python-pillow/Pillow/issues/2783
    im = Image.new("RGB", (W, H))
    draw = ImageDraw.Draw(im)
    # Should not cause 'SystemError: <built-in method getsize of
    # ImagingFont object at 0x...> returned NULL without setting an error'
    draw.textbbox((0, 0), "")
    draw.textbbox((0, 0), "\n")
    draw.textbbox((0, 0), "test\n")
    draw.textlength("")


@pytest.mark.parametrize("bbox", BBOX)
def test_same_color_outline(bbox: Coords) -> None:
    s = curve_outline()

    # Begin
    for mode in ["RGB", "L"]:
        fill = "red"
        for outline in [None, "red", "#f00"]:
            for operation, args in {"chord": [bbox, 0, 180], "ellipse": [bbox], "shape": [s], "pieslice": [bbox, -90, 45], "polygon": [[(18, 30), (85, 30), (60, 72)]], "rectangle": [bbox]}.items():
                im = Image.new(mode, (W, H))
                draw = ImageDraw.Draw(im)
                draw_method = getattr(draw, operation)
                assert isinstance(args, list)
                args += [fill, outline]
                draw_method(*args)
                expected = PILLOW_IMAGE_PREFIX + f"imagedraw_outline_{operation}_{mode}.png"
                assert_image_similar_tofile(im, expected, 1)


@pytest.mark.parametrize("n_sides, polygon_name, args", [(4, "square", {}), (8, "regular_octagon", {}), (4, "square_rotate_45", {"rotation": 45}), (3, "triangle_width", {"outline": "yellow", "width": 5})])
def test_draw_regular_polygon(n_sides: int, polygon_name: str, args: dict[str, int | str]) -> None:
    im = Image.new("RGBA", size=(W, H), color=(255, 0, 0, 0))
    filename = PILLOW_IMAGE_PREFIX + f"imagedraw_{polygon_name}.png"
    draw = ImageDraw.Draw(im)
    bounding_circle = ((W // 2, H // 2), 25)
    rotation = int(args.get("rotation", 0))
    outline = args.get("outline")
    width = int(args.get("width", 1))
    draw.regular_polygon(bounding_circle, n_sides, rotation, "red", outline, width)
    assert_image_equal_tofile(im, filename)


@pytest.mark.parametrize(
    "n_sides, expected_vertices",
    [
        (3, [(28.35, 62.5), (71.65, 62.5), (50.0, 25.0)]),
        (4, [(32.32, 67.68), (67.68, 67.68), (67.68, 32.32), (32.32, 32.32)]),
        (5, [(35.31, 70.23), (64.69, 70.23), (73.78, 42.27), (50.0, 25.0), (26.22, 42.27)]),
        (6, [(37.5, 71.65), (62.5, 71.65), (75.0, 50.0), (62.5, 28.35), (37.5, 28.35), (25.0, 50.0)]),
    ],
)
def test_compute_regular_polygon_vertices(n_sides: int, expected_vertices: list[tuple[float, float]]) -> None:
    bounding_circle = (W // 2, H // 2, 25)
    vertices = ImageDraw._compute_regular_polygon_vertices(bounding_circle, n_sides, 0)
    assert vertices == expected_vertices


@pytest.mark.parametrize(
    "n_sides, bounding_circle, rotation, expected_error, error_message",
    [
        (None, (50, 50, 25), 0, TypeError, "n_sides should be an int"),
        (1, (50, 50, 25), 0, ValueError, "n_sides should be an int > 2"),
        (3, 50, 0, TypeError, "bounding_circle should be a sequence"),
        (
            3,
            (50, 50, 100, 100),
            0,
            ValueError,
            (
                "bounding_circle should contain 2D coordinates "
                r"and a radius \(e.g. \(x, y, r\) or \(\(x, y\), r\) \)"
            ),
        ),
        (3, (50, 50, "25"), 0, ValueError, "bounding_circle should only contain numeric data"),
        (3, ((50, 50, 50), 25), 0, ValueError, r"bounding_circle centre should contain 2D coordinates \(e.g. \(x, y\)\)"),
        (3, (50, 50, 0), 0, ValueError, "bounding_circle radius should be > 0"),
        (3, (50, 50, 25), "0", ValueError, "rotation should be an int or float"),
    ],
)
def test_compute_regular_polygon_vertices_input_error_handling(n_sides: int, bounding_circle: int | tuple[int | tuple[int] | str, ...], rotation: int | str, expected_error: type[Exception], error_message: str) -> None:
    assert_raises(expected_error, lambda: ImageDraw._compute_regular_polygon_vertices(bounding_circle, n_sides, rotation), match=error_message)  # type: ignore[arg-type]


def test_continuous_horizontal_edges_polygon() -> None:
    xy = [(2, 6), (6, 6), (12, 6), (12, 12), (8, 12), (8, 8), (4, 8), (2, 8)]
    img, draw = create_base_image_draw((16, 16))
    draw.polygon(xy, BLACK)
    expected = os.path.join(IMAGES_PATH, "continuous_horizontal_edges_polygon.png")
    assert_image_equal_tofile(img, expected, "continuous horizontal edges polygon failed")


def test_discontiguous_corners_polygon() -> None:
    img, draw = create_base_image_draw((84, 68))
    draw.polygon(((1, 21), (34, 4), (71, 1), (38, 18)), BLACK)
    draw.polygon(((82, 29), (82, 26), (82, 24), (67, 22), (52, 29), (52, 15), (67, 22)), BLACK)
    draw.polygon(((71, 44), (38, 27), (1, 24)), BLACK)
    draw.polygon(((38, 66), (5, 49), (77, 49), (47, 66), (82, 63), (82, 47), (1, 47), (1, 63)), BLACK)
    expected = os.path.join(IMAGES_PATH, "discontiguous_corners_polygon.png")
    assert_image_equal_tofile(img, expected)


@pytest.mark.parametrize("xy", ((1, 1, 0, 1), (1, 1, 1, 0)))
def test_incorrectly_ordered_coordinates(xy: tuple[int, int, int, int]) -> None:
    im = Image.new("RGB", (W, H))
    draw = ImageDraw.Draw(im)
    assert_raises(ValueError, lambda: draw.arc(xy, 10, 260))
    assert_raises(ValueError, lambda: draw.chord(xy, 10, 260))
    assert_raises(ValueError, lambda: draw.ellipse(xy))
    assert_raises(ValueError, lambda: draw.pieslice(xy, 10, 260))
    assert_raises(ValueError, lambda: draw.rectangle(xy))
    assert_raises(ValueError, lambda: draw.rounded_rectangle(xy))


import pytest
from PIL import Image as PILImage, ImageDraw as PILDraw


def test_palette_drawing_reuses_unused_ink_after_single_point_line() -> None:
    palette = bytes(value for i in range(249) for value in (i, i, i)) + bytes(21)
    actual, expected = image_pair("P", (249, 1), bytes(range(249)))
    for module, image in ((ImageDraw, actual), (PILDraw, expected)):
        image.putpalette(palette)
        draw = module.Draw(image)
        draw.line((0, 0), fill=(255, 0, 0))
        assert len(image.palette.colors) == 250
        draw.point((0, 0), fill=(0, 0, 255))
        assert len(image.palette.colors) == 250
        assert (255, 0, 0) not in image.palette.colors
    assert actual.getpalette() == expected.getpalette()
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize(
    "operation",
    [
        lambda draw: draw.line((0, 0, 1, 1), joint="miter"),
        lambda draw: draw.ellipse((0, 0, 1, 1, 2, 2)),
        lambda draw: draw.regular_polygon((1, 1, 1), 2),
        lambda draw: draw.rounded_rectangle((0, 0, 1, 1), radius=-1),
        lambda draw: draw.rounded_rectangle((0, 0, 1, 1), corners=(True,)),
        lambda draw: draw.text((0, 0), "a", stroke_width=-1),
        lambda draw: draw.text((0, 0), "a", embedded_color=True),
        lambda draw: draw.multiline_text((0, 0), "a", direction="ttb"),
        lambda draw: draw.multiline_text((0, 0), "a", anchor="lt"),
        lambda draw: draw.multiline_text((0, 0), "a", spacing=float("inf")),
        lambda draw: draw.textbbox((0, 0), "a", stroke_width=-1),
    ],
)
def test_drawing_rejects_invalid_options(operation) -> None:
    assert_raises((ValueError, NotImplementedError), lambda: operation(ImageDraw.Draw(Image.new("L", (4, 4)))))


def test_drawing_rejects_foreign_image_and_string_integer_ink() -> None:
    assert_raises(TypeError, lambda: ImageDraw.Draw(object()), match="Blanket Image")
    assert_raises(ValueError, lambda: ImageDraw.Draw(Image.new("I", (1, 1))).point((0, 0), "red"), match="numeric ink")


def test_drawing_degenerate_curves_and_antialiased_rectangles() -> None:
    image = Image.new("L", (8, 8))
    draw = ImageDraw.Draw(image, antialias=True)
    draw.pieslice((0, 0, 0, 0), 0, 0, fill=255)
    assert not any(image.tobytes())
    draw.rectangle((1, 1, 6, 6), fill=255)
    assert image.getpixel((3, 3)) == 255
    assert draw.textbbox((0, 0), "a\nb") == draw.multiline_textbbox((0, 0), "a\nb")
    draw.text((0, 0), "a\nb", fill=255)


def test_font_without_getmask2() -> None:
    class Font:
        def getmask(self, text, mode, **kwargs):
            return PILImage.new("L", (1, 1), 255).im

        def getbbox(self, text, **kwargs):
            return (0, 0, 1, 1)

    image = Image.new("L", (2, 2))
    ImageDraw.Draw(image).text((0, 0), "a", font=Font(), fill=255)
    assert image.getpixel((0, 0)) == 255


@pytest.mark.parametrize("count", [1, 15, 16, 17, 33, 65])
@pytest.mark.parametrize("alpha", [None, 0, 1, 127, 254, 255])
def test_rectangle_row_tiles_and_blending_preserve_guards(count: int, alpha: int | None) -> None:
    actual, expected = new_pair("RGB", (count + 2, 3), (81, 93, 117))
    mode = "RGB" if alpha is None else "RGBA"
    ink = (31, 173, 59) if alpha is None else (31, 173, 59, alpha)
    assert_drawing_equal(actual, expected, "rectangle", (1, 1, count, 1), fill=ink, draw_mode=mode)


@pytest.mark.parametrize("blend", [False, True])
def test_rectangle_enormous_outline_preserves_overlapping_edges(blend: bool) -> None:
    actual, expected = new_pair("RGB", (13, 11), (81, 93, 117))
    mode = "RGBA" if blend else "RGB"
    ink = (237, 41, 121, 129) if blend else "red"
    ImageDraw.Draw(actual, mode).rectangle((2, 2, 8, 7), outline=ink, width=10**12)
    PILDraw.Draw(expected, mode).rectangle((2, 2, 8, 7), outline=ink, width=100)
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("bounds", [(4, 0, 2, 3), (0, 4, 2, 3), (0, 1), (0, 1, 2)])
def test_invalid_rectangle_does_not_modify_image(bounds: tuple[int, ...]) -> None:
    image = Image.new("RGB", (5, 5), "blue")
    before = image.tobytes()
    assert_raises(ValueError, lambda: ImageDraw.Draw(image).rectangle(bounds, fill="red"))
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
    assert_raises(ValueError, lambda: getattr(ImageDraw.Draw(image), operation)(points, fill="red"), match="8-bit")
    assert image.tobytes() == before


def test_drawing_on_closed_image_fails() -> None:
    image = Image.new("RGB", (2, 2))
    draw = ImageDraw.Draw(image)
    image.close()
    assert_raises(ValueError, lambda: draw.point((0, 0), fill="red"))
