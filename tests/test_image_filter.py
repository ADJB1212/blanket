"""Copied Pillow 13.0.0 contracts; see PILLOW_TESTS.md."""

from __future__ import annotations

import math

import pytest
from helpers import assert_image_properties, assert_raises
from pillow_helpers import PILLOW_IMAGES, assert_image_equal, hopper

from blanket import Image, ImageFilter

MODES = ("L", "LA", "La", "I", "I;16", "I;16B", "I;16L", "I;16N", "RGB", "RGBA", "RGBa", "RGBX", "CMYK")


@pytest.mark.parametrize(
    ("filter_to_apply", "mode"),
    [
        (filter_to_apply, mode)
        for filter_to_apply in (
            ImageFilter.BLUR,
            ImageFilter.CONTOUR,
            ImageFilter.DETAIL,
            ImageFilter.EDGE_ENHANCE,
            ImageFilter.EDGE_ENHANCE_MORE,
            ImageFilter.EMBOSS,
            ImageFilter.FIND_EDGES,
            ImageFilter.SMOOTH,
            ImageFilter.SMOOTH_MORE,
            ImageFilter.SHARPEN,
            ImageFilter.MaxFilter,
            ImageFilter.MedianFilter,
            ImageFilter.MinFilter,
            ImageFilter.ModeFilter,
            ImageFilter.GaussianBlur,
            ImageFilter.GaussianBlur(0),
            ImageFilter.GaussianBlur(5),
            ImageFilter.GaussianBlur((2, 5)),
            ImageFilter.BoxBlur(0),
            ImageFilter.BoxBlur(5),
            ImageFilter.BoxBlur((2, 5)),
            ImageFilter.UnsharpMask,
            ImageFilter.UnsharpMask(10),
        )
        for mode in MODES
        if not mode.startswith("I") or (isinstance(filter_to_apply, type) and issubclass(filter_to_apply, ImageFilter.BuiltinFilter))
        if mode == "RGB" or filter_to_apply in (ImageFilter.BLUR, ImageFilter.GaussianBlur)
    ],
)
def test_sanity(filter_to_apply: ImageFilter.Filter | type[ImageFilter.Filter], mode: str) -> None:
    im = hopper(mode)
    out = im.filter(filter_to_apply)
    assert_image_properties(out, mode=im.mode, size=im.size)


@pytest.mark.parametrize("mode", MODES)
def test_sanity_error(mode: str) -> None:
    im = hopper(mode)
    assert_raises(TypeError, lambda: im.filter("hello"))  # type: ignore[arg-type]


def test_noop_on_small_images() -> None:
    # If image is smaller than the kernel size, return it as-is.
    kernel_size: tuple[int, int] = ImageFilter.SMOOTH_MORE.filterargs[0]
    kernel_w, kernel_h = kernel_size
    for w in range(1, kernel_w):
        for h in range(1, kernel_h):
            im = hopper("RGB").resize((w, h))
            # Precondition for the below equality test:
            # filter is larger or equal to image.
            assert im.size < kernel_size
            assert_image_equal(im.filter(ImageFilter.SMOOTH_MORE), im)


@pytest.mark.parametrize("mode, expected", (("1", (4, 0)), ("L", (4, 0)), ("P", (4, 0)), ("RGB", ((4, 0, 0), (0, 0, 0)))))
def test_modefilter(mode: str, expected: tuple[int, int] | tuple[tuple[int, int, int], tuple[int, int, int]]) -> None:
    im = Image.new(mode, (3, 3), None)
    im.putdata(list(range(9)))
    # image is:
    #   0 1 2
    #   3 4 5
    #   6 7 8
    mod = im.filter(ImageFilter.ModeFilter).getpixel((1, 1))
    im.putdata([0, 0, 1, 2, 5, 1, 5, 2, 0])  # mode=0
    mod2 = im.filter(ImageFilter.ModeFilter).getpixel((1, 1))
    assert (mod, mod2) == expected


@pytest.mark.parametrize("mode, expected", (("1", (0, 4, 8)), ("L", (0, 4, 8)), ("RGB", ((0, 0, 0), (4, 0, 0), (8, 0, 0))), ("I", (0, 4, 8)), ("F", (0.0, 4.0, 8.0))))
def test_rankfilter(mode: str, expected: (tuple[float, float, float] | tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]])) -> None:
    im = Image.new(mode, (3, 3), None)
    im.putdata(list(range(9)))
    # image is:
    #   0 1 2
    #   3 4 5
    #   6 7 8
    minimum = im.filter(ImageFilter.MinFilter).getpixel((1, 1))
    med = im.filter(ImageFilter.MedianFilter).getpixel((1, 1))
    maximum = im.filter(ImageFilter.MaxFilter).getpixel((1, 1))
    assert (minimum, med, maximum) == expected


@pytest.mark.parametrize("filter", (ImageFilter.MinFilter, ImageFilter.MedianFilter, ImageFilter.MaxFilter))
def test_rankfilter_error(filter: ImageFilter.RankFilter) -> None:
    with pytest.raises(ValueError):
        im = Image.new("P", (3, 3), None)
        im.putdata(list(range(9)))
        # image is:
        #   0 1 2
        #   3 4 5
        #   6 7 8
        im.filter(filter).getpixel((1, 1))


@pytest.mark.parametrize("filter", (ImageFilter.MinFilter(1), ImageFilter.MedianFilter(1), ImageFilter.MaxFilter(1), ImageFilter.RankFilter(3, 1)))
@pytest.mark.parametrize("size", ((0, 1), (1, 0), (0, 0)))
def test_rankfilter_zero_dimension(filter: ImageFilter.RankFilter, size: tuple[int, int]) -> None:
    im = Image.new("L", size)
    assert im.filter(filter).size == size


def test_rankfilter_overflow() -> None:
    # Large margins used to overflow the ImagingExpand overflow guard itself (SIGFPE),
    # by mutating RankFilter.size after construction, bypassing __init__'s validation.
    im = Image.new("L", (16, 16))
    rankfilter = ImageFilter.RankFilter(3, 0)

    for size in (2**31, 2**32 - 1):  # margins of 2**30 and INT_MAX
        rankfilter.size = size
        assert_raises(ValueError, lambda: im.filter(rankfilter), match="filter size too large")


def test_kernel_not_enough_coefficients() -> None:
    assert_raises(ValueError, lambda: ImageFilter.Kernel((3, 3), (0, 0)))


EMBOSS_MATRIX = {
    3: (
        -1, -1, 0,
        -1, 0, 1,
         0, 1, 1,
    ),
    5: (
        -1, -1, -1, -1, 0,
        -1, -1, -1, 0, 1,
        -1, -1, 0, 1, 1,
        -1, 0, 1, 1, 1,
         0, 1, 1, 1, 1,
    ),
}  # fmt: skip


@pytest.mark.parametrize("size", (3, 5))
def test_consistency(size: int) -> None:
    kernel = ImageFilter.Kernel((size, size), EMBOSS_MATRIX[size], 0.3)
    with Image.open(PILLOW_IMAGES / "hopper.bmp") as source, Image.open(PILLOW_IMAGES / f"hopper_emboss_{size}x{size}.bmp") as reference:
        assert_image_equal(source.filter(kernel), reference)


@pytest.mark.parametrize("size", (3, 5))
@pytest.mark.parametrize("mode", ("I;16", "I;16L", "I;16B", "I;16N"))
def test_consistency_i16(size: int, mode: str) -> None:
    kernel = ImageFilter.Kernel((size, size), EMBOSS_MATRIX[size], 0.3)
    reference = hopper("I").filter(kernel)
    result = hopper(mode).filter(kernel)
    assert_image_equal(result.convert("I"), reference)


@pytest.mark.parametrize("mode", ("I;16", "I;16L", "I;16B", "I;16N"))
def test_consistency_i16_high_byte(mode: str) -> None:
    # Exercise filters with a 16bpc image that has content in the high byte, too.
    im = Image.new(mode, (8, 8), 1000)
    # Ensure repeated smoothing retains the exact same color value,
    # rather than drifting due to rounding errors.
    for _ in range(5):
        im = im.filter(ImageFilter.SMOOTH)
        assert im.getpixel((4, 4)) == 1000


@pytest.mark.parametrize("size", (-1, math.nan, math.inf, 2**31))
def test_invalid_box_blur_filter(size: int) -> None:
    for radius in (size, (size, size), (size, 1), (1, size)):
        assert_raises(ValueError, lambda: ImageFilter.BoxBlur(radius), match="radius")

        im = hopper()
        box_blur_filter = ImageFilter.BoxBlur(2)
        box_blur_filter.radius = radius
        assert_raises(ValueError, lambda: im.filter(box_blur_filter), match="radius")


@pytest.mark.parametrize("radius", (math.nan, math.inf, 2**31))
def test_invalid_gaussian_blur_filter(radius: int) -> None:
    im = hopper()
    assert_raises(ValueError, lambda: im.filter(ImageFilter.GaussianBlur(radius)), match="radius")


def test_rankfilter_size_1() -> None:
    im = Image.new("L", (3, 3), 128)

    # Size 1 should not crash (margin is 0)
    assert im.filter(ImageFilter.MinFilter(1)).getpixel((1, 1)) == 128
    assert im.filter(ImageFilter.MaxFilter(1)).getpixel((1, 1)) == 128
    assert im.filter(ImageFilter.MedianFilter(1)).getpixel((1, 1)) == 128
    assert im.filter(ImageFilter.RankFilter(1, 0)).getpixel((1, 1)) == 128
