"""Pillow 13.0.0 contracts and Blanket regressions; see PILLOW_TESTS.md."""

from __future__ import annotations

import math

import pytest
from helpers import assert_image_properties, assert_pillow_equal, assert_pixels_equal, assert_raises, image_pair, patterned_pair
from PIL import Image as PILImage, ImageChops as PILChops
from pillow_helpers import PILLOW_IMAGES, assert_image_equal, hopper

from blanket import Image, ImageChops

TYPE_CHECKING = False
if TYPE_CHECKING:
    from collections.abc import Callable

BLACK = (0, 0, 0)
BROWN = (127, 64, 0)
CYAN = (0, 255, 255)
DARK_GREEN = (0, 128, 0)
GREEN = (0, 255, 0)
ORANGE = (255, 128, 0)
WHITE = (255, 255, 255)

GRAY = 128


@pytest.mark.parametrize(
    "operation,first_file,second_file,options,bbox,pixels",
    [
        pytest.param("add", "imagedraw_ellipse_RGB.png", "imagedraw_floodfill_RGB.png", {}, (25, 25, 76, 76), {(50, 50): ORANGE}, id="add"),
        pytest.param("add", "imagedraw_ellipse_RGB.png", "imagedraw_floodfill_RGB.png", {"scale": 2.5, "offset": 100}, (0, 0, 100, 100), {(50, 50): (202, 151, 100)}, id="add_scale_offset"),
        pytest.param("add_modulo", "imagedraw_ellipse_RGB.png", "imagedraw_floodfill_RGB.png", {}, (25, 25, 76, 76), {(50, 50): ORANGE}, id="add_modulo"),
        pytest.param("blend", "imagedraw_ellipse_RGB.png", "imagedraw_floodfill_RGB.png", {"alpha": 0.5}, (25, 25, 76, 76), {(50, 50): BROWN}, id="blend"),
        pytest.param("difference", "imagedraw_arc_end_le_start.png", "imagedraw_arc_no_loops.png", {}, (25, 25, 76, 76), {}, id="difference"),
        pytest.param("screen", "imagedraw_ellipse_RGB.png", "imagedraw_floodfill_RGB.png", {}, (25, 25, 76, 76), {(50, 50): ORANGE}, id="screen"),
        pytest.param("subtract", "imagedraw_chord_RGB.png", "imagedraw_outline_chord_RGB.png", {}, (25, 50, 76, 76), {(50, 51): GREEN, (50, 52): BLACK}, id="subtract"),
        pytest.param("subtract", "imagedraw_chord_RGB.png", "imagedraw_outline_chord_RGB.png", {"scale": 2.5, "offset": 100}, (0, 0, 100, 100), {(50, 50): (100, 202, 100)}, id="subtract_scale_offset"),
        pytest.param("subtract", None, "imagedraw_chord_RGB.png", {}, None, {(50, 50): (0, 0, 127)}, id="subtract_clip"),
        pytest.param("subtract_modulo", "imagedraw_chord_RGB.png", "imagedraw_outline_chord_RGB.png", {}, (25, 50, 76, 76), {(50, 51): GREEN, (50, 52): BLACK}, id="subtract_modulo"),
        pytest.param("subtract_modulo", None, "imagedraw_chord_RGB.png", {}, None, {(50, 50): (241, 167, 127)}, id="subtract_modulo_no_clip"),
        pytest.param("soft_light", "hopper.png", "hopper-XYZ.png", {}, None, {(64, 64): (163, 54, 32), (15, 100): (1, 1, 3)}, id="soft_light"),
        pytest.param("hard_light", "hopper.png", "hopper-XYZ.png", {}, None, {(64, 64): (144, 50, 27), (15, 100): (1, 1, 2)}, id="hard_light"),
        pytest.param("overlay", "hopper.png", "hopper-XYZ.png", {}, None, {(64, 64): (159, 50, 27), (15, 100): (1, 1, 2)}, id="overlay"),
        pytest.param("add", None, None, {}, None, {(50, 50): (255, 255, 254)}, id="add_clip"),
        pytest.param("add_modulo", None, None, {}, None, {(50, 50): (224, 76, 254)}, id="add_modulo_no_clip"),
    ],
)
def test_binary_reference_samples(operation, first_file, second_file, options, bbox, pixels) -> None:
    first = Image.open(PILLOW_IMAGES / first_file) if first_file else hopper()
    second = Image.open(PILLOW_IMAGES / second_file) if second_file else first
    try:
        result = getattr(ImageChops, operation)(first, second, **options)
    finally:
        first.close()
        second.close()
    if bbox is not None:
        assert result.getbbox() == bbox
    for point, color in pixels.items():
        assert result.getpixel(point) == color


def test_constant() -> None:
    im = Image.new("RGB", (20, 10))
    new = ImageChops.constant(im, GRAY)
    assert new.size == im.size
    assert new.getpixel((0, 0)) == GRAY
    assert new.getpixel((19, 9)) == GRAY


def test_darker_image() -> None:
    with Image.open(PILLOW_IMAGES / "imagedraw_chord_RGB.png") as im1, Image.open(PILLOW_IMAGES / "imagedraw_outline_chord_RGB.png") as im2:
        new = ImageChops.darker(im1, im2)
        assert_image_equal(new, im2)


def test_duplicate() -> None:
    im = hopper()
    new = ImageChops.duplicate(im)
    assert_image_equal(new, im)


def test_invert() -> None:
    with Image.open(PILLOW_IMAGES / "imagedraw_floodfill_RGB.png") as im:
        new = ImageChops.invert(im)
    assert new.getbbox() == (0, 0, 100, 100)
    assert new.getpixel((0, 0)) == WHITE
    assert new.getpixel((50, 50)) == CYAN


def test_lighter_image() -> None:
    with Image.open(PILLOW_IMAGES / "imagedraw_chord_RGB.png") as im1:
        with Image.open(PILLOW_IMAGES / "imagedraw_outline_chord_RGB.png") as im2:
            new = ImageChops.lighter(im1, im2)
        assert_image_equal(new, im1)


def test_multiply_black() -> None:
    """If you multiply an image with a solid black image,
    the result is black."""
    im1 = hopper()
    black = Image.new("RGB", im1.size, "black")
    new = ImageChops.multiply(im1, black)
    assert_image_equal(new, black)


def test_multiply_green() -> None:
    with Image.open(PILLOW_IMAGES / "imagedraw_floodfill_RGB.png") as im:
        green = Image.new("RGB", im.size, "green")
        new = ImageChops.multiply(im, green)
    assert new.getbbox() == (25, 25, 76, 76)
    assert new.getpixel((25, 25)) == DARK_GREEN
    assert new.getpixel((50, 50)) == BLACK


def test_multiply_white() -> None:
    """If you multiply with a solid white image, the image is unaffected."""
    im1 = hopper()
    white = Image.new("RGB", im1.size, "white")
    new = ImageChops.multiply(im1, white)
    assert_image_equal(new, im1)


def test_offset() -> None:
    xoffset = 45
    yoffset = 20
    with Image.open(PILLOW_IMAGES / "imagedraw_ellipse_RGB.png") as im:
        new = ImageChops.offset(im, xoffset, yoffset)
        assert new.getbbox() == (0, 45, 100, 96)
        assert new.getpixel((50, 50)) == BLACK
        assert new.getpixel((50 + xoffset, 50 + yoffset)) == DARK_GREEN

        # Test no yoffset
        assert ImageChops.offset(im, xoffset) == ImageChops.offset(im, xoffset, xoffset)


@pytest.mark.parametrize("mode", Image.MODES)
def test_offset_modes(mode: str) -> None:
    im = hopper(mode)
    assert_image_equal(ImageChops.offset(im, 0, 0), im)  # check no-op

    xoffset = 45
    yoffset = 20
    new = ImageChops.offset(im, xoffset, yoffset)
    w, h = im.size
    for x in range(w):
        for y in range(h):
            wrapped_coord = ((x + xoffset) % w, (y + yoffset) % h)
            assert new.getpixel(wrapped_coord) == im.getpixel((x, y))


@pytest.mark.parametrize("size", ((1, 0), (0, 1), (0, 0)))
def test_offset_zero_size(size: tuple[int, int]) -> None:
    im = Image.new("RGB", size)

    assert_image_equal(ImageChops.offset(im, 1, 1), im)


def test_logical() -> None:
    def table(op: Callable[[Image.Image, Image.Image], Image.Image], a: int, b: int) -> list[float]:
        out = []
        for x in (a, b):
            imx = Image.new("1", (1, 1), x)
            for y in (a, b):
                imy = Image.new("1", (1, 1), y)
                value = op(imx, imy).getpixel((0, 0))
                assert not isinstance(value, tuple)
                assert value is not None
                out.append(value)
        return out

    assert table(ImageChops.logical_and, 0, 1) == [0, 0, 0, 255]
    assert table(ImageChops.logical_or, 0, 1) == [0, 255, 255, 255]
    assert table(ImageChops.logical_xor, 0, 1) == [0, 255, 255, 0]

    assert table(ImageChops.logical_and, 0, 128) == [0, 0, 0, 255]
    assert table(ImageChops.logical_or, 0, 128) == [0, 255, 255, 255]
    assert table(ImageChops.logical_xor, 0, 128) == [0, 255, 255, 0]

    assert table(ImageChops.logical_and, 0, 255) == [0, 0, 0, 255]
    assert table(ImageChops.logical_or, 0, 255) == [0, 255, 255, 255]
    assert table(ImageChops.logical_xor, 0, 255) == [0, 255, 255, 0]


BINARY = ["difference", "multiply", "screen", "lighter", "darker", "add", "subtract", "add_modulo", "subtract_modulo", "soft_light", "hard_light", "overlay"]


@pytest.mark.parametrize("name", ["logical_and", "logical_or", "logical_xor"])
@pytest.mark.parametrize("size", [(0, 3), (1, 1), (17, 9), (1025, 1027)])
def test_logical_operations_match_pillow_across_chunks(name: str, size: tuple[int, int]) -> None:
    raw = bytes((i * 37 + i // 11) % 256 for i in range(((size[0] + 7) // 8) * size[1]))
    other = bytes(value ^ 0x71 for value in raw)
    first, pil_first = image_pair("1", size, raw)
    second, pil_second = image_pair("1", size, other)
    assert_pillow_equal(getattr(ImageChops, name)(first, second), getattr(PILChops, name)(pil_first, pil_second), info=True)


@pytest.mark.parametrize("name", ["logical_and", "logical_or", "logical_xor"])
def test_logical_operations_preserve_first_image_metadata(name: str) -> None:
    first, second = Image.new("1", (3, 2), 1), Image.new("1", (3, 2))
    first.getexif()[305] = "edited"
    first.info["custom"] = "first"
    second.info["custom"] = "second"
    result = getattr(ImageChops, name)(first, second)
    assert result.getexif()[305] == "edited"
    assert result.info["custom"] == "first"
    result.getexif()[305] = "result"
    result.info["custom"] = "result"
    assert first.getexif()[305] == "edited"
    assert first.info["custom"] == "first"


@pytest.mark.parametrize("name", ["difference", "lighter", "darker", "add_modulo", "subtract_modulo", "invert"])
@pytest.mark.parametrize("size", [(1025, 1027), (2049, 2051)])
def test_memory_operations_large_parallel_tail(name: str, size: tuple[int, int]) -> None:
    raw = (bytes(range(256)) * ((size[0] * size[1] * 4 + 255) // 256))[: size[0] * size[1] * 4]
    actual, expected = image_pair("RGBA", size, raw)
    args = [] if name == "invert" else [Image.frombytes("RGBA", size, raw[::-1])]
    reference = [] if name == "invert" else [PILImage.frombytes("RGBA", size, raw[::-1])]
    assert getattr(ImageChops, name)(actual, *args).tobytes() == getattr(PILChops, name)(expected, *reference).tobytes()


@pytest.mark.parametrize("name", BINARY)
def test_every_channel_pair_matches_pillow(name: str) -> None:
    # Exhaustive coverage of integer rounding and the 127/128 blend thresholds.
    a = bytes(range(256)) * 256
    b = bytes(i // 256 for i in range(65536))
    first = Image.frombytes("L", (256, 256), a)
    second = Image.frombytes("L", (256, 256), b)
    expected = getattr(PILChops, name)(PILImage.frombytes("L", first.size, a), PILImage.frombytes("L", second.size, b))
    assert_pillow_equal(getattr(ImageChops, name)(first, second), expected, info=True)


@pytest.mark.parametrize("name", BINARY)
@pytest.mark.parametrize("mode,sizes", [(mode, ((17, 9), (11, 13))) for mode in ("L", "RGB", "RGBA")] + [("L", ((0, 3), (4, 0))), ("RGBA", ((513, 193), (513, 193)))])
def test_binary_modes_sizes_and_parallel_chunks(mode: str, name: str, sizes: tuple[tuple[int, int], tuple[int, int]]) -> None:
    first, pil_first = patterned_pair(mode, sizes[0])
    second, pil_second = patterned_pair(mode, sizes[1], 71)
    first.info["test"] = pil_first.info["test"] = "retained"
    before = first.tobytes(), second.tobytes()
    result = getattr(ImageChops, name)(first, second)
    assert_pillow_equal(result, getattr(PILChops, name)(pil_first, pil_second), info=True)
    assert result.info is not first.info
    assert (first.tobytes(), second.tobytes()) == before
    if result.width and result.height:
        result.putpixel((0, 0), 0)
        assert (first.tobytes(), second.tobytes()) == before


@pytest.mark.parametrize("mode", ["LA", "PA", "HSV", "CMYK", "YCbCr", "LAB"])
@pytest.mark.parametrize("name", ["hard_light", "overlay"])
@pytest.mark.parametrize("width", [15, 16, 17, 33])
def test_blend_channel_modes_at_vector_boundaries(mode: str, name: str, width: int) -> None:
    first, pil_first = patterned_pair(mode, (width + 2, 3))
    second, pil_second = patterned_pair(mode, (width, 4), 71)
    assert_pillow_equal(getattr(ImageChops, name)(first, second), getattr(PILChops, name)(pil_first, pil_second), info=True)


@pytest.mark.parametrize("name", ["add", "subtract"])
@pytest.mark.parametrize(("scale", "offset"), [(2, 0), (0.7, -13), (-1.3, 123), (1, 300), (1, -300), (math.inf, 17), (math.nan, 0), (1e-5, 0)])
def test_scaled_arithmetic(name: str, scale: float, offset: int) -> None:
    first, pil_first = patterned_pair("RGBA", (31, 19))
    second, pil_second = patterned_pair("RGBA", first.size, 99)
    assert_pillow_equal(getattr(ImageChops, name)(first, second, scale, offset), getattr(PILChops, name)(pil_first, pil_second, scale, offset), info=True)


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA", "P"])
@pytest.mark.parametrize("size,shift", [((0, 3), (39, -27)), ((1, 1), (0, 0)), ((17, 9), (3, None)), ((513, 193), (-4, None)), ((1025, 1027), (-100000, 100001))])
def test_offset_regression(mode: str, size: tuple[int, int], shift: tuple[int, int | None]) -> None:
    image, pillow = patterned_pair(mode, size)
    image.info["test"] = pillow.info["test"] = 1
    before = image.tobytes()
    assert_pillow_equal(ImageChops.offset(image, *shift), PILChops.offset(pillow, *shift), info=True)
    assert image.tobytes() == before


@pytest.mark.parametrize("name", BINARY)
def test_invalid_modes_and_closed_inputs(name: str) -> None:
    first = Image.new("RGB", (3, 2))
    second = Image.new("RGBA", first.size)
    assert_raises(ValueError, lambda: getattr(ImageChops, name)(first, second), match="images do not match")
    second = first.copy()
    second.close()
    assert_raises(ValueError, lambda: getattr(ImageChops, name)(first, second), match="closed image")


@pytest.mark.parametrize("name", ["invert", "offset", "duplicate"])
def test_closed_unary_inputs(name: str) -> None:
    image = Image.new("L", (2, 2))
    image.close()
    assert_raises(ValueError, lambda: getattr(ImageChops, name)(image, 1) if name == "offset" else getattr(ImageChops, name)(image), match="closed image")


@pytest.mark.parametrize("name", [*BINARY, "invert", "offset"])
def test_palette_indices_and_palette_copy(name: str) -> None:
    first, pil_first = patterned_pair("L", (19, 11))
    second, pil_second = patterned_pair("L", first.size, 29)
    palette = bytes(range(256)) * 3
    for image in (first, second, pil_first, pil_second):
        image.putpalette(palette)
    if name in BINARY:
        result = getattr(ImageChops, name)(first, second)
        expected = getattr(PILChops, name)(pil_first, pil_second)
    elif name == "offset":
        result, expected = ImageChops.offset(first, -3), PILChops.offset(pil_first, -3)
    else:
        result, expected = ImageChops.invert(first), PILChops.invert(pil_first)
    assert_pillow_equal(result, expected, info=True)
    assert result.getpalette() == expected.getpalette()
    result.putpalette(bytes(768))
    assert first.getpalette() == list(palette)


@pytest.mark.parametrize("bit_depth", [10, 12, 16])
def test_high_depth_offset_and_arithmetic_preserve_depth(bit_depth: int) -> None:
    samples = [0, 256, (1 << bit_depth) - 1, 17, 1000, 88]
    image = Image.frombytes("L", (3, 2), b"".join(v.to_bytes(2, "little") for v in samples), bit_depth=bit_depth)
    result = ImageChops.offset(image, 1, -1)
    assert_image_properties(result, bit_depth=bit_depth, pixels=b"".join(samples[i].to_bytes(2, "little") for i in [5, 3, 4, 2, 0, 1]))
    for name in BINARY:
        assert getattr(ImageChops, name)(image, image).bit_depth == bit_depth
    assert ImageChops.invert(image).bit_depth == bit_depth
