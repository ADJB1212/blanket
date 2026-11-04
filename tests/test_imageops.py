"""Pillow 13.0.0 contracts and Blanket regressions; see PILLOW_TESTS.md."""

from __future__ import annotations

from pillow_helpers import PILLOW_IMAGES

PILLOW_IMAGE_PREFIX = str(PILLOW_IMAGES) + "/"

import pytest
from helpers import assert_pillow_equal, assert_pixels_equal, assert_raises, encode_image, image_buffer, image_pair, new_pair, random_pair
from pillow_helpers import assert_image_equal, assert_image_similar, assert_image_similar_tofile, hopper

from blanket import ExifTags, Image, ImageDraw, ImageOps, ImageStat, features

TYPE_CHECKING = False
if TYPE_CHECKING:
    from collections.abc import Callable


class Deformer(ImageOps.SupportsGetMesh):
    def getmesh(self, im: Image.Image) -> list[tuple[tuple[int, int, int, int], tuple[int, int, int, int, int, int, int, int]]]:
        x, y = im.size
        return [((0, 0, x, y), (0, 0, x, 0, x, y, y, 0))]


deformer = Deformer()


@pytest.mark.parametrize("mode", ["L", "RGB"])
@pytest.mark.parametrize(
    "operation,args",
    [
        ("autocontrast", ()),
        ("pad", ((128, 128),)),
        ("contain", ((128, 128),)),
        ("cover", ((128, 128),)),
        ("crop", (1,)),
        ("deform", (deformer,)),
        ("equalize", ()),
        ("expand", (1,)),
        ("expand", (2, "blue")),
        ("fit", ((128, 128),)),
        ("flip", ()),
        ("grayscale", ()),
        ("invert", ()),
        ("mirror", ()),
        ("posterize", (4,)),
        ("solarize", ()),
        ("exif_transpose", ()),
    ],
)
def test_sanity(mode, operation, args) -> None:
    getattr(ImageOps, operation)(hopper(mode), *args)


def test_sanity_special_options() -> None:
    for options in ({"cutoff": 10}, {"cutoff": (2, 10)}, {"ignore": [0, 255]}, {"mask": hopper("L")}, {"preserve_tone": True}):
        ImageOps.autocontrast(hopper("L"), **options)
    for black, white in (((0, 0, 0), (255, 255, 255)), ("black", "white")):
        ImageOps.colorize(hopper("L"), black, white)
    ImageOps.invert(hopper("1"))


@pytest.mark.parametrize("border, expected_box", ((1, (1, 1, 127, 127)), ((1, 2), (1, 2, 127, 126)), ((1, 2, 3, 4), (1, 2, 125, 124))))
def test_crop(border: int | tuple[int, ...], expected_box: tuple[int, int, int, int]) -> None:
    im = hopper()
    assert_image_equal(ImageOps.crop(im, border), im.crop(expected_box))


@pytest.mark.parametrize("size", [(1, 1), (1, 100), (100, 1)])
def test_1pxfit(size) -> None:
    assert ImageOps.fit(hopper("RGB").resize(size), (35, 35)).size == (35, 35)


def test_fit_same_ratio() -> None:
    # The ratio for this image is 1000.0 / 755 = 1.3245033112582782
    # If the ratios are not acknowledged to be the same,
    # and Pillow attempts to adjust the width to
    # 1.3245033112582782 * 755 = 1000.0000000000001
    # then centering this greater width causes a negative x offset when cropping
    with Image.new("RGB", (1000, 755)) as im:
        new_im = ImageOps.fit(im, (1000, 755))
        assert new_im.size == (1000, 755)


def test_fit_zero() -> None:
    im = Image.new("1", (5, 5))
    new_im = ImageOps.fit(im, (0, 5))
    assert new_im.width == 0

    im = Image.new("1", (5, 0))
    new_im = ImageOps.fit(im, (5, 5))
    assert new_im.size == (5, 5)

    im = Image.new("1", (5, 5))
    new_im = ImageOps.fit(im, (10, 0))
    assert new_im.size == (10, 0)


@pytest.mark.parametrize("new_size", ((256, 256), (512, 256), (256, 512)))
def test_contain(new_size: tuple[int, int]) -> None:
    im = hopper()
    new_im = ImageOps.contain(im, new_size)
    assert new_im.size == (256, 256)


def test_contain_round() -> None:
    im = Image.new("1", (43, 63), 1)
    new_im = ImageOps.contain(im, (5, 7))
    assert new_im.width == 5

    im = Image.new("1", (63, 43), 1)
    new_im = ImageOps.contain(im, (7, 5))
    assert new_im.height == 5


def test_contain_zero() -> None:
    im = Image.new("1", (1, 10))
    new_im = ImageOps.contain(im, (5, 5))
    assert new_im.width == 0

    im = Image.new("1", (10, 1))
    new_im = ImageOps.contain(im, (5, 5))
    assert new_im.height == 0

    im = Image.new("1", (10, 0))
    new_im = ImageOps.contain(im, (5, 5))
    assert new_im.size == (5, 0)

    im = Image.new("1", (5, 5))
    new_im = ImageOps.contain(im, (2, 0))
    assert new_im.size == (0, 0)


@pytest.mark.parametrize(
    "image_name, expected_size",
    (
        ("colr_bungee.png", (1024, 256)),  # landscape
        ("imagedraw_stroke_multiline.png", (256, 640)),  # portrait
        ("hopper.png", (256, 256)),  # square
    ),
)
def test_cover(image_name: str, expected_size: tuple[int, int]) -> None:
    with Image.open((PILLOW_IMAGE_PREFIX + "") + image_name) as im:
        new_im = ImageOps.cover(im, (256, 256))
        assert new_im.size == expected_size


def test_cover_zero() -> None:
    im = Image.new("1", (10, 0))
    new_im = ImageOps.cover(im, (5, 5))
    assert new_im.size == (5, 0)

    im = Image.new("1", (5, 5))
    new_im = ImageOps.cover(im, (2, 0))
    assert new_im.size == (0, 0)


def test_pad() -> None:
    # Same ratio
    im = hopper()
    new_size = (im.width * 2, im.height * 2)
    new_im = ImageOps.pad(im, new_size)
    assert new_im.size == new_size

    for label, color, new_size in [("h", None, (im.width * 4, im.height * 2)), ("v", "#f00", (im.width * 2, im.height * 4))]:
        for i, centering in enumerate([(0, 0), (0.5, 0.5), (1, 1)]):
            new_im = ImageOps.pad(im, new_size, color=color, centering=centering)
            assert new_im.size == new_size

            assert_image_similar_tofile(new_im, (PILLOW_IMAGE_PREFIX + "imageops_pad_") + label + "_" + str(i) + ".jpg", 6)


def test_pad_round() -> None:
    im = Image.new("1", (1, 1), 1)
    new_im = ImageOps.pad(im, (4, 1))
    assert new_im.getpixel((2, 0)) == 1

    new_im = ImageOps.pad(im, (1, 4))
    assert new_im.getpixel((0, 2)) == 1


def test_pad_palette() -> None:
    with Image.open(PILLOW_IMAGE_PREFIX + "p_16.tga") as im:
        im_padded = ImageOps.pad(im, (im.width * 2, im.height), color=(255, 0, 0))
        im_rgb = im_padded.convert("RGB")

        # Verify that the left and right sides are now red
        px = im_rgb.load()
        assert px is not None
        for y in range(im_rgb.height):
            for x in range(im.width // 2):
                assert px[x, y] == (255, 0, 0)
                assert px[im_rgb.width - 1 - x, y] == (255, 0, 0)

        im_cropped = im_rgb.crop((im.width * 0.5, 0, im.width * 1.5, im.height))
        assert_image_equal(im_cropped, im.convert("RGB"))


@pytest.mark.parametrize("mode", ("P", "PA"))
def test_palette(mode: str) -> None:
    im = hopper(mode)

    # Expand
    expanded_im = ImageOps.expand(im)
    assert_image_equal(im.convert("RGB"), expanded_im.convert("RGB"))

    # Pad
    padded_im = ImageOps.pad(im, (256, 128), centering=(0, 0))
    assert_image_equal(im.convert("RGB"), padded_im.convert("RGB").crop((0, 0, 128, 128)))


def test_rgba_palette() -> None:
    im = Image.new("P", (1, 1))

    red = (255, 0, 0, 255)
    translucent_black = (0, 0, 0, 127)
    im.putpalette(red + translucent_black, "RGBA")

    expanded_im = ImageOps.expand(im, 1, 1)

    palette = expanded_im.palette
    assert palette is not None
    assert palette.mode == "RGBA"
    assert expanded_im.convert("RGBA").getpixel((0, 0)) == translucent_black


def test_transparency() -> None:
    im = Image.new("P", (1, 1))
    im.info["transparency"] = 0

    expanded_im = ImageOps.expand(im, 1, "blue")
    assert expanded_im.info["transparency"] == 0
    expanded_im_rgba = expanded_im.convert("RGBA")
    assert expanded_im_rgba.getpixel((0, 0)) == (0, 0, 255, 255)
    assert expanded_im_rgba.getpixel((1, 1)) == (0, 0, 0, 0)

    padded_im = ImageOps.pad(im, (3, 1), color="blue")
    assert padded_im.info["transparency"] == 0
    padded_im_rgba = padded_im.convert("RGBA")
    assert padded_im_rgba.getpixel((0, 0)) == (0, 0, 255, 255)
    assert padded_im_rgba.getpixel((1, 0)) == (0, 0, 0, 0)


def test_pil163() -> None:
    # Division by zero in equalize if < 255 pixels in image (@PIL163)

    i = hopper("RGB").resize((15, 16))

    ImageOps.equalize(i.convert("L"))
    ImageOps.equalize(i.convert("P"))
    ImageOps.equalize(i.convert("RGB"))


def test_scale() -> None:
    # Test the scaling function
    i = hopper("L").resize((50, 50))

    assert_raises(ValueError, lambda: ImageOps.scale(i, -1))

    newimg = ImageOps.scale(i, 1)
    assert newimg.size == (50, 50)

    newimg = ImageOps.scale(i, 2)
    assert newimg.size == (100, 100)

    newimg = ImageOps.scale(i, 0.5)
    assert newimg.size == (25, 25)


@pytest.mark.parametrize("border", (10, (1, 2, 3, 4)))
def test_expand_palette(border: int | tuple[int, int, int, int]) -> None:
    with Image.open(PILLOW_IMAGE_PREFIX + "p_16.tga") as im:
        im_expanded = ImageOps.expand(im, border, (255, 0, 0))

        if isinstance(border, int):
            left = top = right = bottom = border
        else:
            left, top, right, bottom = border
        px = im_expanded.convert("RGB").load()
        assert px is not None
        for x in range(im_expanded.width):
            for b in range(top):
                assert px[x, b] == (255, 0, 0)
            for b in range(bottom):
                assert px[x, im_expanded.height - 1 - b] == (255, 0, 0)
        for y in range(im_expanded.height):
            for b in range(left):
                assert px[b, y] == (255, 0, 0)
            for b in range(right):
                assert px[im_expanded.width - 1 - b, y] == (255, 0, 0)

        im_cropped = im_expanded.crop((left, top, im_expanded.width - right, im_expanded.height - bottom))
        assert_image_equal(im_cropped, im)


@pytest.mark.parametrize("border", ((1,), (1, 2, 3), (1, 2, 3, 4, 5)))
def test_expand_invalid_border(border: tuple[int, ...]) -> None:
    im = Image.new("1", (1, 1))
    assert_raises(ValueError, lambda: ImageOps.expand(im, border))


@pytest.mark.parametrize(
    "options,samples",
    [
        ({}, [(0, (255, 0, 0)), (127, (127, 63, 0)), (255, (0, 127, 0))]),
        ({"blackpoint": 50, "whitepoint": 100}, [(25, (255, 0, 0)), (75, (127, 63, 0)), (125, (0, 127, 0))]),
        ({"mid": "blue", "blackpoint": 50, "whitepoint": 200, "midpoint": 100}, [(25, (255, 0, 0)), (75, (127, 0, 127)), (100, (0, 0, 255)), (150, (0, 63, 127)), (225, (0, 127, 0))]),
    ],
)
def test_colorize_reference_samples(options, samples) -> None:
    with Image.open(PILLOW_IMAGES / "bw_gradient.png") as image:
        result = ImageOps.colorize(image.convert("L"), black="red", white="green", **options)
    for x, color in samples:
        assert result.getpixel((x, 1)) == pytest.approx(color, abs=1)


def test_colorize_invalid_mode() -> None:
    assert_raises(ValueError, lambda: ImageOps.colorize(hopper("RGB"), "black", "white"), match="mode must be L, not RGB")


@pytest.mark.parametrize("bits", (0, 9))
def test_posterize_invalid_bits(bits: int) -> None:
    assert_raises(ValueError, lambda: ImageOps.posterize(hopper("L"), bits), match="bits must be between or equal to 1 and 8")


@pytest.mark.parametrize("blackpoint, whitepoint", ((-1, 255), (0, 256), (200, 50)))
def test_colorize_invalid_points(blackpoint: int, whitepoint: int) -> None:
    assert_raises(ValueError, lambda: ImageOps.colorize(hopper("L"), "black", "white", blackpoint=blackpoint, whitepoint=whitepoint), match="blackpoint and whitepoint must each be")


@pytest.mark.parametrize("midpoint", (100, 200))
def test_colorize_invalid_midpoint(midpoint: int) -> None:
    assert_raises(ValueError, lambda: ImageOps.colorize(hopper("L"), "black", "white", mid="blue", blackpoint=125, midpoint=midpoint, whitepoint=175), match="midpoint must be between or equal to")


def test_exif_transpose() -> None:
    exts = [".jpg"]
    if features.check("webp"):
        exts.append(".webp")
    for ext in exts:
        with Image.open((PILLOW_IMAGE_PREFIX + "hopper") + ext) as base_im:

            def check(orientation_im: Image.Image) -> None:
                for im in [orientation_im, orientation_im.copy()]:  # ImageFile  # Image
                    if orientation_im is base_im:
                        assert "exif" not in im.info
                    else:
                        original_exif = im.info["exif"]
                    transposed_im = ImageOps.exif_transpose(im)
                    assert_image_similar(base_im, transposed_im, 17)
                    if orientation_im is base_im:
                        assert "exif" not in im.info
                    else:
                        assert transposed_im.info["exif"] != original_exif

                        assert 0x0112 in im.getexif()
                        assert 0x0112 not in transposed_im.getexif()

                    # Repeat the operation to test that it does not keep transposing
                    transposed_im2 = ImageOps.exif_transpose(transposed_im)
                    assert_image_equal(transposed_im2, transposed_im)

            check(base_im)
            for i in range(2, 9):
                with Image.open((PILLOW_IMAGE_PREFIX + "hopper_orientation_") + str(i) + ext) as orientation_im:
                    check(orientation_im)

    # Orientation from "XML:com.adobe.xmp" info key
    for suffix in ("", "_exiftool"):
        with Image.open((PILLOW_IMAGE_PREFIX + "xmp_tags_orientation") + suffix + ".png") as im:
            assert im.getexif()[0x0112] == 3

            transposed_im = ImageOps.exif_transpose(im)
            assert 0x0112 not in transposed_im.getexif()

            transposed_im._reload_exif()
            assert 0x0112 not in transposed_im.getexif()

    # Orientation from "Raw profile type exif" info key
    # This test image has been manually hexedited from exif_imagemagick.png
    # to have a different orientation
    with Image.open(PILLOW_IMAGE_PREFIX + "exif_imagemagick_orientation.png") as im:
        assert im.getexif()[0x0112] == 3

        transposed_im = ImageOps.exif_transpose(im)
        assert 0x0112 not in transposed_im.getexif()

    # Orientation set directly on Image.Exif
    im1 = hopper()
    im1.getexif()[0x0112] = 3
    transposed_im = ImageOps.exif_transpose(im1)
    assert 0x0112 not in transposed_im.getexif()


def test_exif_transpose_xml_without_xmp() -> None:
    with Image.open(PILLOW_IMAGE_PREFIX + "xmp_tags_orientation.png") as im:
        assert im.getexif()[0x0112] == 3
        assert "XML:com.adobe.xmp" in im.info

        del im.info["xmp"]
        transposed_im = ImageOps.exif_transpose(im)
        assert 0x0112 not in transposed_im.getexif()


def test_exif_transpose_raw_profile() -> None:
    exif = Image.Exif()
    exif[0x0112] = 3
    exif[ExifTags.Base.UserComment] = "Pillow"
    data = exif.tobytes()

    im = Image.new("1", (1, 1))
    im.info["Raw profile type exif"] = f"\nexif\n{len(data):8}\n{data.hex()}\n"
    assert im.getexif()[0x0112] == 3

    # Assert that the updated info is still present after transposing and copying
    transposed_im = ImageOps.exif_transpose(im)
    transposed_exif = transposed_im.copy().getexif()
    assert 0x0112 not in transposed_exif
    assert transposed_exif[ExifTags.Base.UserComment] == "Pillow"


def test_exif_transpose_in_place() -> None:
    with Image.open(PILLOW_IMAGE_PREFIX + "orientation_rectangle.jpg") as im:
        assert im.size == (2, 1)
        assert im.getexif()[0x0112] == 8
        expected = im.rotate(90, expand=True)

        ImageOps.exif_transpose(im, in_place=True)
        assert im.size == (1, 2)
        assert 0x0112 not in im.getexif()
        assert_image_equal(im, expected)


def test_autocontrast_unsupported_mode() -> None:
    im = Image.new("RGBA", (1, 1))
    assert_raises(OSError, lambda: ImageOps.autocontrast(im))


def test_autocontrast_cutoff() -> None:
    # Test the cutoff argument of autocontrast
    with Image.open(PILLOW_IMAGE_PREFIX + "bw_gradient.png") as img:

        def autocontrast(cutoff: int | tuple[int, int]) -> list[int]:
            return ImageOps.autocontrast(img, cutoff).histogram()

        assert autocontrast(10) == autocontrast((10, 10))
        assert autocontrast(10) != autocontrast((1, 10))


def test_autocontrast_mask_toy_input() -> None:
    # Test the mask argument of autocontrast
    with Image.open(PILLOW_IMAGE_PREFIX + "bw_gradient.png") as img:
        rect_mask = Image.new("L", img.size, 0)
        draw = ImageDraw.Draw(rect_mask)
        x0 = img.size[0] // 4
        y0 = img.size[1] // 4
        x1 = 3 * img.size[0] // 4
        y1 = 3 * img.size[1] // 4
        draw.rectangle((x0, y0, x1, y1), fill=255)

        result = ImageOps.autocontrast(img, mask=rect_mask)
        result_nomask = ImageOps.autocontrast(img)

        assert result != result_nomask
        assert ImageStat.Stat(result, mask=rect_mask).median == [127]
        assert ImageStat.Stat(result_nomask).median == [128]


def test_autocontrast_mask_real_input() -> None:
    # Test the autocontrast with a rectangular mask
    with Image.open(PILLOW_IMAGE_PREFIX + "iptc.jpg") as img:
        rect_mask = Image.new("L", img.size, 0)
        draw = ImageDraw.Draw(rect_mask)
        x0, y0 = img.size[0] // 2, img.size[1] // 2
        x1, y1 = img.size[0] - 40, img.size[1]
        draw.rectangle((x0, y0, x1, y1), fill=255)

        result = ImageOps.autocontrast(img, mask=rect_mask)
        result_nomask = ImageOps.autocontrast(img)

        assert result_nomask != result
        assert ImageStat.Stat(result, mask=rect_mask).median == pytest.approx((195, 202, 184), abs=2), "autocontrast with mask pixel incorrect"
        assert ImageStat.Stat(result_nomask).median == pytest.approx((119, 106, 79), abs=2), "autocontrast without mask pixel incorrect"


def test_autocontrast_preserve_tone() -> None:
    def autocontrast(mode: str, preserve_tone: bool) -> list[int]:
        im = hopper(mode)
        return ImageOps.autocontrast(im, preserve_tone=preserve_tone).histogram()

    assert autocontrast("RGB", True) != autocontrast("RGB", False)
    assert autocontrast("L", True) == autocontrast("L", False)


def test_autocontrast_preserve_gradient() -> None:
    gradient = Image.linear_gradient("L")

    # test with a grayscale gradient that extends to 0,255.
    # Should be a noop.
    out = ImageOps.autocontrast(gradient, cutoff=0, preserve_tone=True)

    assert_image_equal(gradient, out)

    # cutoff the top and bottom
    # autocontrast should make the first and last histogram entries equal
    # and, with rounding, should be 10% of the image pixels
    out = ImageOps.autocontrast(gradient, cutoff=10, preserve_tone=True)
    hist = out.histogram()
    assert hist[0] == hist[-1]
    assert hist[-1] == 256 * round(256 * 0.10)

    # in rgb
    img = gradient.convert("RGB")
    out = ImageOps.autocontrast(img, cutoff=0, preserve_tone=True)
    assert_image_equal(img, out)


@pytest.mark.parametrize("color", ((255, 255, 255), (127, 255, 0), (127, 127, 127), (0, 0, 0)))
def test_autocontrast_preserve_one_color(color: tuple[int, int, int]) -> None:
    img = Image.new("RGB", (10, 10), color)

    # single color images shouldn't change
    out = ImageOps.autocontrast(img, cutoff=0, preserve_tone=True)
    assert_image_equal(img, out)  # single color, no cutoff

    # even if there is a cutoff
    out = ImageOps.autocontrast(img, cutoff=10, preserve_tone=True)  # single color 10 cutoff
    assert_image_equal(img, out)


@pytest.mark.parametrize("operation", (ImageOps.invert, lambda im: ImageOps.posterize(im, 3), ImageOps.solarize))
def test_palette_operation(operation: Callable[[Image.Image], Image.Image]) -> None:
    im = hopper("P")
    result = operation(im)

    assert result.mode == "P"
    assert_image_equal(result.convert("RGB"), operation(im.convert("RGB")))


def test_palette_operation_keeps_alpha() -> None:
    im = Image.new("P", (1, 2))
    im.putpalette((0, 0, 0, 0, 255, 0, 0, 128), "RGBA")
    im.putpixel((0, 1), 1)

    result = ImageOps.invert(im)

    palette = result.getpalette("RGBA")
    assert palette is not None
    assert palette == [255, 255, 255, 0, 0, 255, 255, 128]


def test_palette_histogram_operation_not_supported() -> None:
    im = hopper("P")
    assert_raises(NotImplementedError, lambda: ImageOps.autocontrast(im), match="mode P support coming soon")


import subprocess
import sys
from io import BytesIO
from typing import TYPE_CHECKING, cast

import pytest
from PIL import Image as PILImage, ImageOps as PILOps
from PIL.PngImagePlugin import PngInfo

from blanket import ImageOps

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.mark.parametrize("mode", ["L", "RGB"])
def test_deformer_can_subclass_public_protocol(mode: str) -> None:
    class IdentityDeformer(ImageOps.SupportsGetMesh):
        def getmesh(self, image: Image.Image) -> list[tuple[tuple[int, int, int, int], tuple[float, float, float, float, float, float, float, float]]]:
            width, height = image.size
            return [((0, 0, width, height), (0, 0, 0, height, width, height, width, 0))]

    image = Image.frombytes(mode, (7, 5), bytes((i * 17) % 256 for i in range(35 * Image.getmodebands(mode))))
    result = ImageOps.deform(image, IdentityDeformer(), Image.Resampling.NEAREST)
    assert result.size == image.size
    assert result.mode == image.mode
    assert result.tobytes() == image.tobytes()


@pytest.mark.parametrize("mode", ["L", "RGB"])
@pytest.mark.parametrize("name", ["autocontrast", "equalize"])
@pytest.mark.parametrize("size", [(0, 0), (1, 1), (33, 17)])
def test_native_histogram_operations_avoid_python_tables(mode: str, name: str, size: tuple[int, int], monkeypatch: pytest.MonkeyPatch) -> None:
    image, reference = random_pair(mode, size)
    image.info["description"] = "native tone mapping"
    mask, reference_mask = new_pair("L", size, 1)
    options = {"cutoff": (2.5, 9), "ignore": [-1, 0], "preserve_tone": True} if name == "autocontrast" else {}
    expected = getattr(PILOps, name)(reference, mask=reference_mask, **options)

    def unexpected(*args: object, **kwargs: object) -> None:
        raise AssertionError("histograms and tone tables must stay in Rust")

    with monkeypatch.context() as patch:
        patch.setattr(ImageOps, "ops_histogram", unexpected)
        patch.setattr(ImageOps, "_lut", unexpected)
        result = getattr(ImageOps, name)(image, mask=mask, **options)
    assert_pillow_equal(result, expected)
    assert result.info == image.info and result.info is not image.info


@pytest.mark.parametrize("cutoff", [0.1, (0.3, 0.7), (33.33333333333333, 66.66666666666666), (100, 100), (-3, 110), (2, 4, 6)])
def test_autocontrast_fractional_and_unusual_cutoffs(cutoff: object) -> None:
    image, reference = random_pair("RGB", (53, 41))
    assert_pillow_equal(ImageOps.autocontrast(image, cutoff=cutoff), PILOps.autocontrast(reference, cutoff=cutoff))


@pytest.mark.parametrize("points", [(0, 0, 255), (0, 255, 255), (70, 70, 200), (70, 200, 200), (255, 255, 255), (0, 0, 0)])
def test_colorize_coincident_stops_preserve_floor_rounding(points: tuple[int, int, int]) -> None:
    image, reference = image_pair("L", (256, 1), bytes(range(256)))
    options = {"black": (250, 30, 17), "mid": (10, 230, 90), "white": (20, 40, 55), "blackpoint": points[0], "midpoint": points[1], "whitepoint": points[2]}
    assert_pillow_equal(ImageOps.colorize(image, **options), PILOps.colorize(reference, **options))


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
@pytest.mark.parametrize("name", ["flip", "mirror", "grayscale"])
def test_flips_and_grayscale(mode: str, name: str) -> None:
    b, p = random_pair(mode)
    b.info["custom"] = p.info["custom"] = "kept"
    result = getattr(ImageOps, name)(b)
    assert_pillow_equal(result, getattr(PILOps, name)(p))
    assert result.info == p.info


@pytest.mark.parametrize("mode", ["P", "PA"])
@pytest.mark.parametrize("operation", [*range(7), "mirror", "flip", "scale", "expand", "deform"])
def test_geometry_preserves_pending_palette_edits(mode, operation):
    indices = bytes([0, 1, 2, 2, 1, 0])
    raw = indices if mode == "P" else bytes(value for index in indices for value in (index, 127))
    image = Image.frombytes(mode, (3, 2), raw)
    reference = PILImage.frombytes(mode, image.size, raw)
    palette = bytes([255, 0, 0, 0, 255, 0, 0, 0, 255]) + bytes(759)
    image.palette.palette = palette
    reference.putpalette(palette)
    if isinstance(operation, int):
        result = image.transpose(operation)
        expected = reference.transpose(operation)
    elif operation == "deform":
        deformer = Deformer_regression([((0, 0, 3, 2), (0, 0, 0, 2, 3, 2, 3, 0))])
        result = ImageOps.deform(image, deformer)
        expected = PILOps.deform(reference, deformer)
    else:
        args = (1,) if operation in ("scale", "expand") else ()
        result = getattr(ImageOps, operation)(image, *args)
        expected = getattr(PILOps, operation)(reference, *args)
    assert result.palette.tobytes() == palette
    assert result.tobytes() == expected.tobytes()
    assert result.convert("RGBA").tobytes() == expected.convert("RGBA").tobytes()
    assert image.tobytes() == raw


@pytest.mark.parametrize("mode", ["I", "I;16", "I;16L", "I;16B"])
@pytest.mark.parametrize("fill", [None, 0, 255, 1000, 65535, -10, -(1 << 31), (1 << 31) - 1])
@pytest.mark.parametrize("operation", ["expand", "pad"])
def test_integer_canvas_fill_preserves_samples_and_byte_order(mode, fill, operation):
    image = Image.new(mode, (3, 2), 512)
    reference = PILImage.frombytes(mode, image.size, image.tobytes())
    if operation == "expand":
        result = ImageOps.expand(image, (1, 2), fill=fill)
        expected = PILOps.expand(reference, (1, 2), fill=fill)
    else:
        result = ImageOps.pad(image, (5, 2), color=fill)
        expected = PILOps.pad(reference, (5, 2), color=fill)
    assert_pillow_equal(result, expected)
    assert result.bit_depth == image.bit_depth
    assert image.getpixel((0, 0)) == 512


@pytest.mark.parametrize("mode,depth", [("F", 32), *[(mode, depth) for mode in ("L", "RGB", "RGBA") for depth in (10, 12, 16)]])
@pytest.mark.parametrize("operation", ["expand", "pad"])
def test_canvas_preserves_float_and_high_depth_samples(mode, depth, operation):
    image = Image.new(mode, (3, 2), 73)
    if mode != "F":
        image = image.convert(mode, bit_depth=depth)
    fill = -1.25 if mode == "F" else "white"
    if operation == "expand":
        result = ImageOps.expand(image, 1, fill=fill)
        source_corner = (1, 1)
    else:
        result = ImageOps.pad(image, (5, 2), color=fill)
        source_corner = (1, 0)
    expected = -1.25 if mode == "F" else (1 << depth) - 1
    if mode in ("RGB", "RGBA"):
        expected = (expected,) * len(mode)
    assert result.getpixel((0, 0)) == expected
    assert result.getpixel(source_corner) == image.getpixel((0, 0))
    assert result.mode == mode
    assert result.bit_depth == depth


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
def test_crop_border_and_box_are_distinct(mode: str) -> None:
    b, p = random_pair(mode)
    b.info["custom"] = "preserved"
    border = ImageOps.crop(b, border=2)
    assert_pillow_equal(border, PILOps.crop(p, border=2))
    assert border.size == (b.width - 4, b.height - 4)
    box = (1, 2, 3, 4)
    region = b.crop(box=box)
    assert_pillow_equal(region, p.crop(box=box))
    assert region.size == (2, 2)
    # Pillow retains tuple border support despite its integer annotation.
    sides = ImageOps.crop(b, border=box)
    assert_pillow_equal(sides, PILOps.crop(p, border=box))
    assert sides.size == (b.width - 4, b.height - 6)
    for result in (border, region, sides, ImageOps.crop(b), b.crop()):
        assert result is not b
        assert result.info == b.info
        assert result.info is not b.info
    assert_raises(TypeError, lambda: cast("Callable[..., object]", b.crop)(border=2))
    assert_raises(TypeError, lambda: cast("Callable[..., object]", ImageOps.crop)(b, box=box))


class Deformer_regression:
    def __init__(self, mesh: list) -> None:
        self.mesh = mesh
        self.image = None

    def getmesh(self, image: object) -> list:
        self.image = image
        return self.mesh


@pytest.mark.parametrize("key", ["XML:com.adobe.xmp", "xmp"])
@pytest.mark.parametrize("kind", [str, bytes, tuple])
def test_exif_cleans_xmp(key: str, kind: type) -> None:
    b, p = random_pair()
    exif = PILImage.Exif()
    exif[274] = 6
    b.info["exif"] = p.info["exif"] = exif.tobytes()
    value = '<rdf tiff:Orientation="6"><tiff:Orientation>6</tiff:Orientation><keep/></rdf>'
    if kind is bytes:
        value = value.encode()
    elif kind is tuple:
        value = (value.encode(), b"unrelated")
    b.info[key] = p.info[key] = value
    if kind is tuple:
        result = ImageOps.exif_transpose(b)
        assert result.info[key] == (b"<rdf ><keep/></rdf>", b"unrelated")
        assert b.info[key] == value
        return
    actual, expected = ImageOps.exif_transpose(b), PILOps.exif_transpose(p)
    assert_pillow_equal(actual, expected)
    assert actual.info[key] == expected.info[key]
    assert b.info[key] == value


def test_xmp_only_png() -> None:
    _, p = random_pair()
    metadata = PngInfo()
    metadata.add_itxt("XML:com.adobe.xmp", '<rdf tiff:Orientation="8"/>', zip=True)
    stream = image_buffer(p, "PNG", pnginfo=metadata)
    b, p = Image.open(stream), PILImage.open(stream)
    assert_pillow_equal(ImageOps.exif_transpose(b), PILOps.exif_transpose(p))


@pytest.mark.parametrize(("mode", "size"), [("RGB", (2, 2)), ("L", (1, 1))])
def test_invalid_masks(mode: str, size: tuple[int, int]) -> None:
    b, _ = random_pair()
    mask, _ = random_pair(mode, size)
    for op in (ImageOps.autocontrast, ImageOps.equalize):
        assert_raises(ValueError, lambda: op(b, mask=mask))


def test_closed_image() -> None:
    b, _ = random_pair()
    b.close()
    for op in (ImageOps.invert, ImageOps.flip, ImageOps.equalize, ImageOps.exif_transpose):
        assert_raises(ValueError, lambda: op(b), match="closed")


def test_no_pillow_runtime_dependency() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
sys.modules['PIL'] = None
from blanket import Image, ImageOps
im = Image.frombytes('RGB', (2, 2), bytes(range(12)))
assert ImageOps.fit(ImageOps.invert(im), (4, 3)).size == (4, 3)
assert ImageOps.expand(im, 1, 'rebeccapurple').size == (4, 4)
""",
        ],
        check=True,
    )


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
def test_processed_png_interop(mode: str) -> None:
    b, p = random_pair(mode)
    b = ImageOps.fit(ImageOps.mirror(b), (21, 17))
    p = PILOps.fit(PILOps.mirror(p), (21, 17))
    stream = image_buffer(b, "PNG")
    loaded = PILImage.open(stream)
    assert loaded.tobytes() == p.tobytes()


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
@pytest.mark.parametrize(("source", "target"), [((1, 1), (7, 9)), ((1, 7), (3, 11)), ((7, 1), (11, 3)), ((4, 4), (2, 4)), ((4, 4), (4, 2)), ((4, 4), (3, 3)), ((10, 10), (15, 15)), ((2, 205), (5, 61))])
def test_resize_thin_and_small_images(mode: str, source: tuple, target: tuple, subtests: pytest.Subtests) -> None:
    b, p = random_pair(mode, source)
    for method in Image.Resampling:
        with subtests.test(method=method):
            assert_pillow_equal(ImageOps.fit(b, target, method), PILOps.fit(p, target, method))


@pytest.mark.parametrize("method", [0, 2, 3])
def test_deform_clipped_and_overlapping_boxes(method: int) -> None:
    b, p = random_pair()
    mesh = [((-5, -3, 20, 24), (0, 0, 0, 29, 37, 29, 37, 0)), ((10, 5, 40, 30), (3, 4, -2, 25, 28, 35, 37, 0))]
    assert_pillow_equal(ImageOps.deform(b, Deformer_regression(mesh), method), PILOps.deform(p, Deformer_regression(mesh), method))


@pytest.mark.parametrize("endian", ["<", ">"])
def test_exif_byte_order_and_profile(endian: str) -> None:
    import struct

    b, p = random_pair()
    raw = (b"II" if endian == "<" else b"MM") + struct.pack(endian + "HIHHHIH", 42, 8, 1, 274, 3, 1, 8) + bytes(6)
    b.info["Raw profile type exif"] = p.info["Raw profile type exif"] = f"\nexif\n{len(raw)}\n{raw.hex()}"
    actual, expected = ImageOps.exif_transpose(b), PILOps.exif_transpose(p)
    assert_pillow_equal(actual, expected)
    exif = PILImage.Exif()
    exif.load(bytes.fromhex(actual.info["Raw profile type exif"]))
    assert 274 not in exif


@pytest.mark.parametrize("compressed", [False, True])
def test_png_exif_profile(compressed: bool) -> None:
    _, p = random_pair()
    exif = PILImage.Exif()
    exif[274] = 3
    raw = exif.tobytes()
    metadata = PngInfo()
    metadata.add_text("Raw profile type exif", f"\nexif\n{len(raw)}\n{raw.hex()}", zip=compressed)
    stream = image_buffer(p, "PNG", pnginfo=metadata)
    assert_pillow_equal(ImageOps.exif_transpose(Image.open(stream)), PILOps.exif_transpose(PILImage.open(stream)))


def test_jxl_exif_container() -> None:
    b, p = random_pair()
    encoded = encode_image(b, "JXL", lossless=True)
    exif = PILImage.Exif()
    exif[274] = 6
    p.info["exif"] = exif.tobytes()
    payload = bytes(4) + exif.tobytes()[6:]
    box = (len(payload) + 8).to_bytes(4, "big") + b"Exif" + payload
    if encoded.startswith(b"\xff\x0a"):
        signature = b"\0\0\0\x0cJXL \r\n\x87\n"
        filetype = b"\0\0\0\x14ftypjxl \0\0\0\0jxl "
        encoded = signature + filetype + box + (len(encoded) + 8).to_bytes(4, "big") + b"jxlc" + encoded
    else:
        # Insert after the signature so a final zero-length codestream box
        # cannot swallow the metadata box.
        encoded = encoded[:12] + box + encoded[12:]
    assert_pillow_equal(ImageOps.exif_transpose(Image.open(BytesIO(encoded))), PILOps.exif_transpose(p))


@pytest.mark.parametrize("raw", [b"", b"garbage", b"Exif\0\0II", b"II*\0\xff\xff\xff\xff"])
def test_malformed_exif_does_not_break_pixels(raw: bytes) -> None:
    b, p = random_pair()
    b.info["exif"] = raw
    assert_pillow_equal(ImageOps.exif_transpose(b), p)


def test_deform_zero_area_box() -> None:
    b, p = random_pair()
    deformer = Deformer_regression([((1, 0, 1, 10), (0, 0, 0, 10, 10, 10, 10, 0))])
    for ops, image in ((ImageOps, b), (PILOps, p)):
        assert_raises(ZeroDivisionError, lambda: ops.deform(image, deformer))


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
@pytest.mark.parametrize(
    ("name", "kwargs"),
    [
        ("crop", {"border": (7, 9, 11, 13)}),
        ("expand", {"border": (3, 5), "fill": "#1234"}),
        ("flip", {}),
        ("mirror", {}),
        ("grayscale", {}),
        ("fit", {"size": (377, 311), "bleed": 0.07}),
        ("scale", {"factor": 1.5, "resample": Image.Resampling.NEAREST}),
    ],
)
def test_parallel_imageops_match_pillow(mode: str, name: str, kwargs: dict) -> None:
    # Large enough for worker partitions, with unaligned row and SIMD tails.
    b, p = random_pair(mode, (521, 509))
    assert_pillow_equal(getattr(ImageOps, name)(b, **kwargs), getattr(PILOps, name)(p, **kwargs))


@pytest.mark.parametrize("mode", ["L", "RGB"])
@pytest.mark.parametrize(("name", "kwargs"), [("autocontrast", {"cutoff": (2, 7), "preserve_tone": True}), ("equalize", {}), ("invert", {}), ("posterize", {"bits": 3}), ("solarize", {"threshold": 173})])
def test_parallel_luts_and_histograms(mode: str, name: str, kwargs: dict) -> None:
    b, p = random_pair(mode, (521, 509))
    b_options, p_options = kwargs.copy(), kwargs.copy()
    if name in ("autocontrast", "equalize"):
        data = bytes(0 if i % 3 == 0 else 255 for i in range(521 * 509))
        b_options["mask"] = Image.frombytes("L", b.size, data)
        p_options["mask"] = PILImage.frombytes("L", p.size, data)
    assert_pillow_equal(getattr(ImageOps, name)(b, **b_options), getattr(PILOps, name)(p, **p_options))


def test_parallel_colorize() -> None:
    b, p = random_pair("L", (521, 509))
    assert_pillow_equal(ImageOps.colorize(b, "navy", "gold", "red"), PILOps.colorize(p, "navy", "gold", "red"))


@pytest.mark.parametrize("orientation", range(2, 9))
@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
def test_parallel_exif_tiles(mode: str, orientation: int) -> None:
    b, p = random_pair(mode, (521, 509))
    exif = PILImage.Exif()
    exif[274] = orientation
    b.info["exif"] = p.info["exif"] = exif.tobytes()
    assert_pillow_equal(ImageOps.exif_transpose(b), PILOps.exif_transpose(p))


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
def test_parallel_mesh_overwrite_order(mode: str) -> None:
    b, p = random_pair(mode, (521, 509))
    mesh = [((0, 0, 521, 509), (521, 0, 521, 509, 0, 509, 0, 0)), ((1, 1, 520, 508), (0, 0, 0, 507, 519, 507, 519, 0))]
    assert_pillow_equal(ImageOps.deform(b, Deformer_regression(mesh)), PILOps.deform(p, Deformer_regression(mesh)))
