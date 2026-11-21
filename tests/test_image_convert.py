"""Copied passing Pillow 13.0.0 contracts; see PILLOW_TESTS.md."""

from __future__ import annotations

import pytest
from helpers import assert_image_properties, assert_pixels_equal, assert_raises, image_pair
from PIL import Image as PillowImage
from pillow_helpers import PILLOW_IMAGES, assert_image, assert_image_equal, assert_image_similar, hopper

from blanket import Image

TYPE_CHECKING = False
if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("width", [1, 16, 257])
@pytest.mark.parametrize(
    "source,target",
    [("RGB", "RGBA"), ("RGB", "RGBX"), ("RGBX", "RGB"), ("RGBX", "L"), ("RGBX", "RGBA"), ("RGBA", "RGBX"), ("RGBA", "RGBa"), ("RGBa", "RGBA"), ("LA", "La"), ("La", "LA"), ("RGB", "F"), ("RGBA", "F"), ("CMYK", "RGB"), ("YCbCr", "RGB")],
)
def test_conversion_buffers_preserve_pixels_and_source(width, source, target):
    size = (width, 3)
    channels = PillowImage.getmodebands(source)
    raw = bytes((i * 137 + i // 7) % 256 for i in range(width * 3 * channels))
    image, pillow = image_pair(source, size, raw)
    expected = pillow.convert(target)
    actual = image.convert(target)
    assert_pixels_equal(actual, expected)
    actual.putpixel((0, 0), 0 if target in ("L", "F") else (0,) * len(actual.getbands()))
    assert image.tobytes() == raw


@pytest.mark.parametrize("target", ["La", "RGBa"])
@pytest.mark.parametrize("source_mode", ["RGBA;F", "P:palette", "P:key", "P:table", "PA"])
def test_premultiplied_conversion_retains_alpha(source_mode, target):
    if source_mode.startswith("P:"):
        image = Image.frombytes("P", (2, 1), b"\x00\x01")
        if source_mode == "P:palette":
            image.putpalette(bytes((130, 80, 200, 0, 50, 110, 70, 127)), "RGBA")
        else:
            image.putpalette(bytes((130, 80, 200, 50, 110, 70)))
            image.info["transparency"] = 0 if source_mode == "P:key" else b"\x00\x7f"
    elif source_mode == "RGBA;F":
        image = Image.new(source_mode, (2, 1), (0.25, 0.4, 0.5, 0.125))
    else:
        image = Image.new("PA", (2, 1), (0, 127))
        image.putpalette(bytes((130, 80, 200)))
    straight_mode = "LA" if target == "La" else "RGBA"
    straight = image.convert("RGBA", bit_depth=8) if source_mode == "RGBA;F" else image
    expected = straight.convert(straight_mode).convert(target)

    result = image.convert(target)

    assert result.tobytes() == expected.tobytes()
    assert "transparency" not in result.info


def test_premultiplied_to_float_keeps_alpha():
    source = Image.new("RGBa", (1, 1), (11, 22, 33, 64))

    actual = source.convert("RGBA;F")
    expected = source.convert("RGBA").convert("RGBA;F")

    assert actual.tobytes() == expected.tobytes()
    assert actual.getpixel((0, 0))[-1] == pytest.approx(64 / 255)


def test_sanity() -> None:
    def convert(im: Image.Image, mode: str) -> None:
        out = im.convert(mode)
        assert_image_properties(out, mode=mode, size=im.size)

    modes = ("1", "L", "LA", "P", "PA", "I", "F", "RGB", "RGBA", "RGBX", "CMYK", "YCbCr", "HSV")

    for input_mode in modes:
        im = hopper(input_mode)
        for output_mode in modes:
            convert(im, output_mode)

        # Check 0
        im = Image.new(input_mode, (0, 0))
        for output_mode in modes:
            convert(im, output_mode)


@pytest.mark.parametrize("depth", [8, 12])
def test_extracting_rgb_channels_removes_the_unrepresentable_color_key(depth):
    image = Image.new("RGB", (2, 1), (40, 50, 60)).convert("RGB", bit_depth=depth)
    image.info["transparency"] = image.getpixel((0, 0))
    image.info["label"] = "kept"

    for band in (*image.split(), image.getchannel("G")):
        assert band.info == {"label": "kept"}
        assert band.convert("RGBA").getpixel((0, 0))[-1] == (1 << depth) - 1
    assert image.info["transparency"] == image.getpixel((0, 0))


def test_extracting_the_grayscale_channel_keeps_its_transparency():
    image = Image.new("L", (2, 1), 40)
    image.info["transparency"] = 40

    for band in (*image.split(), image.getchannel("L")):
        assert band.convert("RGBA").getpixel((0, 0))[-1] == 0


def test_unsupported_conversion() -> None:
    im = hopper()
    assert_raises(ValueError, lambda: im.convert("INVALID"), match="image has wrong mode")


# ref https://github.com/python-pillow/Pillow/issues/274


def _test_float_conversion(im: Image.Image) -> None:
    orig = im.getpixel((5, 5))
    converted = im.convert("F").getpixel((5, 5))
    assert orig == converted


def test_16bit() -> None:
    with Image.open(PILLOW_IMAGES / "16bit.cropped.tif") as im:
        _test_float_conversion(im)

    for color in (65535, 65536):
        im_i = Image.new("I", (1, 1), color)
        im_i16 = im_i.convert("I;16")
        assert im_i16.getpixel((0, 0)) == 65535


def test_16bit_workaround() -> None:
    with Image.open(PILLOW_IMAGES / "16bit.cropped.tif") as im:
        _test_float_conversion(im.convert("I"))


def test_opaque() -> None:
    alpha = hopper("P").convert("PA").getchannel("A")

    solid = Image.new("L", (128, 128), 255)
    assert_image_equal(alpha, solid)


def test_rgba() -> None:
    with Image.open(PILLOW_IMAGES / "transparent.png") as im:
        assert im.mode == "RGBA"

        assert_image_similar(im.convert("RGBa").convert("RGB"), im.convert("RGB"), 1.5)


def test_rgba_p() -> None:
    im = hopper("RGBA")
    im.putalpha(hopper("L"))

    converted = im.convert("P")
    comparable = converted.convert("RGBA")

    assert_image_similar(im, comparable, 20)


def test_rgba_pa() -> None:
    im = hopper("RGBA").convert("PA").convert("RGB")
    expected = hopper("RGB")

    assert_image_similar(im, expected, 9.3)


def test_pa() -> None:
    im = hopper().convert("PA")

    palette = im.palette
    assert palette is not None
    assert palette.colors != {}


def test_trns_p(tmp_path: Path) -> None:
    im = hopper("P")
    im.info["transparency"] = 0

    f = tmp_path / "temp.png"

    im_l = im.convert("L")
    assert im_l.info["transparency"] == 0
    im_l.save(f)

    im_rgb = im.convert("RGB")
    assert im_rgb.info["transparency"] == (0, 0, 0)
    im_rgb.save(f)


# ref https://github.com/python-pillow/Pillow/issues/664


@pytest.mark.parametrize("mode", ("LA", "PA", "RGBA"))
def test_trns_p_transparency(mode: str) -> None:
    im = hopper("P")
    im.info["transparency"] = 128
    converted_im = im.convert(mode)
    assert "transparency" not in converted_im.info
    if mode == "PA":
        assert converted_im.palette is not None
    else:
        # https://github.com/python-pillow/Pillow/issues/2702
        assert converted_im.palette is None


def test_trns_l(tmp_path: Path) -> None:
    im = hopper("L")
    im.info["transparency"] = 128

    f = tmp_path / "temp.png"

    im_la = im.convert("LA")
    assert "transparency" not in im_la.info
    im_la.save(f)

    im_rgb = im.convert("RGB")
    assert im_rgb.info["transparency"] == (128, 128, 128)  # undone
    im_rgb.save(f)

    im_p = im.convert("P")
    assert "transparency" in im_p.info
    im_p.save(f)

    im_p = im.convert("P", palette=Image.Palette.ADAPTIVE)
    assert "transparency" in im_p.info
    im_p.save(f)


def test_trns_RGB(tmp_path: Path) -> None:
    im = hopper("RGB")
    im.info["transparency"] = im.getpixel((0, 0))

    f = tmp_path / "temp.png"

    im_l = im.convert("L")
    assert im_l.info["transparency"] == im_l.getpixel((0, 0))  # undone
    im_l.save(f)

    im_la = im.convert("LA")
    assert "transparency" not in im_la.info
    im_la.save(f)

    im_la = im.convert("La")
    assert "transparency" not in im_la.info
    assert im_la.getpixel((0, 0)) == (0, 0)

    im_p = im.convert("P")
    assert "transparency" in im_p.info
    im_p.save(f)

    im_rgba = im.convert("RGBA")
    assert "transparency" not in im_rgba.info
    im_rgba.save(f)

    im_rgba = im.convert("RGBa")
    assert "transparency" not in im_rgba.info
    assert im_rgba.getpixel((0, 0)) == (0, 0, 0, 0)

    with pytest.warns(UserWarning, match="Couldn't allocate palette entry for transparency"):
        im_p = im.convert("P", palette=Image.Palette.ADAPTIVE)
    assert "transparency" not in im_p.info
    im_p.save(f)

    im.info["transparency"] = list(im.info["transparency"])
    im_rgba = im.convert("RGBA")
    assert "transparency" not in im_rgba.info

    im = Image.new("RGB", (1, 1))
    im.info["transparency"] = im.getpixel((0, 0))
    im_p = im.convert("P", palette=Image.Palette.ADAPTIVE)
    assert im_p.info["transparency"] == im_p.getpixel((0, 0))
    im_p.save(f)


def test_adaptive_grayscale_conversion_remaps_transparent_index() -> None:
    image = Image.frombytes("L", (3, 1), bytes([40, 90, 40]))
    image.info["transparency"] = 40

    indexed = image.convert("P", palette=Image.Palette.ADAPTIVE)

    assert indexed.info["transparency"] == indexed.getpixel((0, 0))
    assert indexed.convert("RGBA").tobytes() == image.convert("RGBA").tobytes()


@pytest.mark.parametrize("convert_mode", ("L", "LA", "I"))
def test_l_macro_rounding(convert_mode: str) -> None:
    for mode in ("P", "PA"):
        im = Image.new(mode, (1, 1))
        assert im.palette is not None
        im.palette.getcolor((0, 1, 2))

        converted_im = im.convert(convert_mode)
        converted_color = converted_im.getpixel((0, 0))
        if convert_mode == "LA":
            assert isinstance(converted_color, tuple)
            converted_color = converted_color[0]
        assert converted_color == 1


@pytest.mark.parametrize("mode", ["L", "RGB", "P"])
@pytest.mark.parametrize("key", [0, 1])
def test_bilevel_transparency_survives_conversion(mode: str, key: int) -> None:
    image = Image.frombytes("1", (2, 1), b"\x40")
    image.info["transparency"] = key

    result = image.convert(mode)
    assert result.convert("RGBA").tobytes() == image.convert("RGBA").tobytes()


def test_gif_with_rgba_palette_to_p() -> None:
    # See https://github.com/python-pillow/Pillow/issues/2433
    with Image.open(PILLOW_IMAGES / "hopper.gif") as im:
        im.info["transparency"] = 255
        im.load()
        assert im.palette is not None
        assert im.palette.mode == "RGB"
        im_p = im.convert("P")

    # Should not raise ValueError: unrecognized raw mode
    im_p.load()


def test_p_la() -> None:
    im = hopper("RGBA")
    alpha = hopper("L")
    im.putalpha(alpha)

    comparable = im.convert("P").convert("LA").getchannel("A")

    assert_image_similar(alpha, comparable, 5)


def test_p2pa_alpha() -> None:
    with Image.open(PILLOW_IMAGES / "tiny.png") as im:
        assert im.mode == "P"

        im_pa = im.convert("PA")
    assert im_pa.mode == "PA"

    im_a = im_pa.getchannel("A")
    for x in range(4):
        alpha = 255 if x > 1 else 0
        for y in range(4):
            assert im_a.getpixel((x, y)) == alpha


def test_p2pa_palette() -> None:
    with Image.open(PILLOW_IMAGES / "tiny.png") as im:
        im_pa = im.convert("PA")
    assert im_pa.getpalette(None) == im.getpalette()


rgb2xyz_matrix = (
    0.412453, 0.357580, 0.180423, 0,
    0.212671, 0.715160, 0.072169, 0,
    0.019334, 0.119193, 0.950227, 0,
)  # fmt: skip


def test_matrix_illegal_conversion() -> None:
    im = hopper("CMYK")
    assert im.mode != "RGB"
    assert_raises(ValueError, lambda: im.convert(mode="CMYK", matrix=rgb2xyz_matrix), match="illegal conversion")


def test_matrix_wrong_mode() -> None:
    im = hopper("L")
    assert_raises(ValueError, lambda: im.convert(mode="L", matrix=rgb2xyz_matrix), match="image has wrong mode")


def test_matrix_truncated() -> None:
    im = hopper()
    assert_raises(TypeError, lambda: im.convert(mode="L", matrix=(0,)), match="matrix must be tuple of length 4 or 12")


@pytest.mark.parametrize("mode,matrix", [("L", (0.5, 0, 0, 10)), ("RGB", (0.5, 0, 0, 10, 0, 1, 0, 0, 0, 0, 1, 0))])
@pytest.mark.parametrize("key", [(41, 67, 89), [41, 67, 89]])
def test_matrix_conversion_uses_same_rounding_for_pixels_and_transparency(mode, matrix, key) -> None:
    image = Image.new("RGB", (17, 1), (41, 67, 89))
    image.putpixel((1, 0), (101, 67, 89))
    image.info["transparency"] = key

    result = image.convert(mode, matrix)

    assert result.info["transparency"] == result.getpixel((0, 0)) == result.getpixel((16, 0))
    assert result.convert("RGBA").getchannel("A").tobytes() == b"\x00\xff" + bytes(15)


@pytest.mark.parametrize("mode", ("RGB", "L"))
def test_matrix_xyz(mode: str) -> None:
    im = hopper("RGB")
    im.info["transparency"] = (255, 0, 0)
    # Convert an RGB image to the CIE XYZ colour space
    converted_im = im.convert(mode=mode, matrix=rgb2xyz_matrix)
    assert_image_properties(converted_im, mode=mode, size=im.size)
    with Image.open(PILLOW_IMAGES / "hopper-XYZ.png") as target:
        if converted_im.mode == "RGB":
            assert_image_similar(converted_im, target, 3)
            assert converted_im.info["transparency"] == (105, 54, 5)
        else:
            assert_image_similar(converted_im, target.getchannel(0), 1)
            assert converted_im.info["transparency"] == 105


def test_matrix_identity() -> None:
    im = hopper("RGB")
    identity_matrix = (
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
    )  # fmt: skip
    # Convert with an identity matrix
    converted_im = im.convert("RGB", identity_matrix)
    # No change
    assert_image_equal(converted_im, im)

    # Test list
    converted_im = im.convert("RGB", list(identity_matrix))
    assert_image_equal(converted_im, im)
