from __future__ import annotations

import copy
import struct
import sys
from io import BytesIO

import numpy as np
import pytest
from helpers import assert_pixels_equal, assert_raises, image_pair
from PIL import Image as PillowImage, ImageChops as PillowChops, ImageColor as PillowColor, ImageEnhance as PillowEnhance, ImageFilter as PillowFilter

from blanket import Image, ImageChops, ImageColor, ImageDraw, ImageEnhance, ImageFilter, ImageHdr, ImageOps, ImageStat

MODES = ("La", "RGBa", "RGBX", "I;16N")


def images(mode: str, width: int = 33, height: int = 5) -> tuple[Image.Image, PillowImage.Image]:
    channels = Image.getmodebands(mode)
    stride = 2 if mode == "I;16N" else channels
    raw = bytes((i * 53 + i // 7) % 256 for i in range(width * height * stride))
    return Image.frombytes(mode, (width, height), raw), PillowImage.frombytes(mode, (width, height), raw)


@pytest.mark.parametrize("mode", MODES)
def test_storage_and_geometry(mode: str, subtests: pytest.Subtests) -> None:
    for width in [0, 1, 15, 16, 17, 33]:
        with subtests.test(mode=mode, width=width):
            actual, expected = images(mode, width)
            assert_pixels_equal(actual, expected)
            assert actual.getdata() == list(expected.get_flattened_data())
            assert_pixels_equal(copy.copy(actual), expected.copy())
            assert_pixels_equal(copy.deepcopy(actual), expected.copy())
            assert_pixels_equal(actual.crop((-1, 1, width + 1, 7)), expected.crop((-1, 1, width + 1, 7)))
            for method in range(7):
                assert_pixels_equal(actual.transpose(method), expected.transpose(method))
            for alpha_only in (False, True):
                assert actual.getbbox(alpha_only=alpha_only) == expected.getbbox(alpha_only=alpha_only)
            if width:
                reference = expected.convert("I") if mode == "I;16N" else expected
                assert actual.getextrema() == reference.getextrema()
                assert_pixels_equal(ImageChops.offset(actual, -3, 2), PillowChops.offset(expected, -3, 2))
            for band, reference in zip(actual.split(), expected.split(), strict=True):
                assert_pixels_equal(band, reference)
            assert_pixels_equal(Image.merge(mode, actual.split()), expected)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("orientation", [-1, 1])
def test_raw_layout_and_array_interoperability(mode: str, orientation: int) -> None:
    actual, expected = images(mode)
    row = len(actual.tobytes()) // actual.height
    encoded = actual.tobytes("raw", mode, row + 5, orientation)
    assert encoded == expected.tobytes("raw", mode, row + 5, orientation)
    assert_pixels_equal(Image.frombytes(mode, actual.size, encoded, "raw", mode, row + 5, orientation), expected)
    assert_pixels_equal(actual.to_pillow(), expected)
    array = np.asarray(actual)
    assert array.dtype == np.asarray(expected).dtype
    np.testing.assert_array_equal(array, np.asarray(expected))
    assert_pixels_equal(Image.fromarray(array, mode=mode), expected)


@pytest.mark.parametrize("mode", ["LA", "RGBA"])
@pytest.mark.parametrize("width", [0, 1, 15, 16, 17, 33, 65536])
def test_premultiplied_conversion_arithmetic(mode: str, width: int) -> None:
    channels = Image.getmodebands(mode)
    raw = bytes(value for i in range(width) for value in (*([i % 256] * (channels - 1)), (i // 256) % 256))
    actual, expected = image_pair(mode, (width, 1), raw)
    destination = "La" if mode == "LA" else "RGBa"
    assert_pixels_equal(actual.convert(destination), expected.convert(destination))
    assert_pixels_equal(actual.convert(destination).convert(mode), expected.convert(destination).convert(mode))
    actual, expected = image_pair(destination, (width, 1), raw)
    assert_pixels_equal(actual.convert(mode), expected.convert(mode))


@pytest.mark.parametrize("mode", ["La", "RGBa", "RGBX"])
@pytest.mark.parametrize("color", ["red", "#ff000080", "#12345600", "#123456ff"])
def test_color_strings(mode: str, color: str) -> None:
    assert ImageColor.getcolor(color, mode) == PillowColor.getcolor(color, mode)
    assert_pixels_equal(Image.new(mode, (3, 2), color), PillowImage.new(mode, (3, 2), color))


@pytest.mark.parametrize(
    "source,destination",
    [
        (source, destination)
        for source in ("1", "L", "LA", "La", "RGB", "RGBA", "RGBa", "RGBX", "CMYK", "YCbCr", "I", "I;16N", "F")
        for destination in ("L", "LA", "La", "RGB", "RGBA", "RGBa", "RGBX", "I", "I;16N", "F")
        if source in MODES or destination in MODES
    ],
)
def test_conversions(source: str, destination: str) -> None:
    expected = PillowImage.new(source, (17, 3), 71)
    actual = Image.frombytes(source, expected.size, expected.tobytes())
    try:
        reference = expected.convert(destination)
    except ValueError:
        assert_raises(ValueError, lambda: actual.convert(destination))
    else:
        assert_pixels_equal(actual.convert(destination), reference)


@pytest.mark.parametrize("mode", ["La", "RGBa", "RGBX"])
def test_filters(mode: str, subtests: pytest.Subtests) -> None:
    for name in ["BLUR", "SMOOTH", "FIND_EDGES", "MinFilter", "MaxFilter", "MedianFilter", "ModeFilter", "BoxBlur", "GaussianBlur", "UnsharpMask"]:
        with subtests.test(mode=mode, name=name):
            actual, expected = images(mode)
            if name == "BoxBlur":
                assert_pixels_equal(actual.filter(ImageFilter.BoxBlur(2)), expected.filter(PillowFilter.BoxBlur(2)))
                return
            assert_pixels_equal(actual.filter(getattr(ImageFilter, name)), expected.filter(getattr(PillowFilter, name)))


@pytest.mark.parametrize("mode", ["La", "RGBa", "RGBX"])
@pytest.mark.parametrize("method", range(6))
def test_resize_and_reduce(mode: str, method: int) -> None:
    actual, expected = images(mode)
    assert_pixels_equal(actual.resize((17, 9), method), expected.resize((17, 9), method))
    assert_pixels_equal(actual.reduce((3, 2)), expected.reduce((3, 2)))


@pytest.mark.parametrize("mode", ["La", "RGBa", "RGBX"])
@pytest.mark.parametrize("method", [0, 2, 3])
def test_transforms(mode: str, method: int) -> None:
    actual, expected = images(mode)
    assert_pixels_equal(actual.rotate(17, method, expand=True), expected.rotate(17, method, expand=True))
    matrix = (0.8, 0.1, -1.0, 0.2, 1.2, 0.5)
    assert_pixels_equal(actual.transform((19, 7), Image.AFFINE, matrix, method), expected.transform((19, 7), PillowImage.Transform.AFFINE, matrix, method))


@pytest.mark.parametrize("mode", ["La", "RGBa", "RGBX"])
def test_channel_operations(mode: str) -> None:
    actual, expected = images(mode)
    assert actual.has_transparency_data == expected.has_transparency_data
    assert actual.histogram() == expected.histogram()
    for name in ("invert", "multiply", "difference", "add", "screen"):
        operation = getattr(ImageChops, name)
        reference = getattr(PillowChops, name)
        assert_pixels_equal(operation(actual) if name == "invert" else operation(actual, actual), reference(expected) if name == "invert" else reference(expected, expected))
    for name in ("Brightness", "Sharpness", "Color", "Contrast"):
        try:
            reference = getattr(PillowEnhance, name)(expected).enhance(0.7)
        except ValueError:
            assert_raises(ValueError, lambda: getattr(ImageEnhance, name)(actual))
        else:
            assert_pixels_equal(getattr(ImageEnhance, name)(actual).enhance(0.7), reference)
    mask = Image.new("L", actual.size, 83)
    actual.paste(Image.new(mode, actual.size, 71), mask=mask)
    expected.paste(PillowImage.new(mode, expected.size, 71), mask=PillowImage.new("L", expected.size, 83))
    assert_pixels_equal(actual, expected)


@pytest.mark.parametrize("mode", ["La", "RGBa", "RGBX"])
@pytest.mark.parametrize("alpha", [0, 71, 255, "mask"])
def test_alpha_promotion(mode: str, alpha: int | str) -> None:
    actual, expected = images(mode)
    actual.putalpha(Image.new("L", actual.size, 83) if alpha == "mask" else alpha)
    expected.putalpha(PillowImage.new("L", expected.size, 83) if alpha == "mask" else alpha)
    assert_pixels_equal(actual, expected)
    actual, expected = images(mode)
    before = actual.tobytes()
    assert_raises(ValueError, lambda: actual.putalpha(Image.new("L", (1, 1))))
    assert_pixels_equal(actual, expected)
    assert actual.tobytes() == before


def test_native_u16_byte_order_and_mutation() -> None:
    values = [0, 1, 255, 256, 1000, 32768, 65535] * 5
    raw = struct.pack(f"={len(values)}H", *values)
    actual = Image.frombytes("I;16N", (len(values), 1), memoryview(b"x" + raw + b"y")[1:-1])
    expected = PillowImage.frombytes("I;16N", actual.size, raw)
    assert_pixels_equal(actual, expected)
    assert_pixels_equal(actual.resize((17, 3), Image.NEAREST), expected.resize((17, 3), PillowImage.Resampling.NEAREST))
    for destination, byteorder in (("I;16L", "little"), ("I;16B", "big"), ("I;16N", sys.byteorder)):
        converted = actual.convert(destination)
        assert converted.tobytes() == b"".join(v.to_bytes(2, byteorder) for v in values)
        assert converted.convert("I;16N").tobytes() == raw
    for dtype in ("<u2", ">u2", "=u2"):
        assert Image.fromarray(np.array([values], dtype=dtype), mode="I;16N").tobytes() == raw
    assert Image.new("I;16N", (2, 1), 1000).tobytes() == struct.pack("=HH", 1000, 1000)
    actual.putdata(values)
    assert actual.getdata() == values
    actual.putpixel((0, 0), 50000)
    actual.load()[1, 0] = (1000,)
    assert actual.getpixel((0, 0)) == 50000
    assert actual.getpixel((1, 0)) == 1000
    actual.paste(71, (2, 0, 3, 1))
    assert actual.getpixel((2, 0)) == 71


@pytest.mark.parametrize("name", ["SMOOTH", "FIND_EDGES"])
def test_native_u16_convolution(name: str) -> None:
    actual, expected = images("I;16N", 17, 7)
    assert_pixels_equal(actual.filter(getattr(ImageFilter, name)), expected.filter(getattr(PillowFilter, name)))


@pytest.mark.parametrize("format", ["PNG", "TIFF"])
def test_native_u16_codec_interoperability(format: str) -> None:
    actual, expected = images("I;16N")
    data = actual.encode(format)
    reopened = Image.decode(data)
    reference = PillowImage.open(BytesIO(data))
    assert np.asarray(reopened).tolist() == np.asarray(expected).tolist()
    np.testing.assert_array_equal(np.asarray(reference), np.asarray(expected))


def test_native_u16_shared_operations() -> None:
    image = Image.new("I;16N", (17, 5), 1000)
    expanded = ImageOps.expand(image, 1, 50000)
    assert expanded.mode == "I;16N"
    assert expanded.getpixel((0, 0)) == 50000
    assert expanded.getpixel((1, 1)) == 1000
    ImageDraw.Draw(image).rectangle((0, 0, 2, 2), fill=50000)
    assert image.getpixel((1, 1)) == 50000
    assert image.getpixel((3, 3)) == 1000
    reference = image.convert("I;16L")
    assert ImageStat.Stat(image).mean == ImageStat.Stat(reference).mean
    assert ImageChops.invert(image).convert("I;16L").tobytes() == ImageChops.invert(reference).tobytes()
    assert ImageEnhance.Brightness(image).enhance(0.7).convert("I;16L").tobytes() == ImageEnhance.Brightness(reference).enhance(0.7).tobytes()
    assert ImageHdr.tone_map(image).tobytes() == ImageHdr.tone_map(reference).tobytes()
    data = image.encode("PNG", save_all=True, append_images=[image])
    decoded = Image.decode(data)
    assert decoded.n_frames == 2
    assert decoded.getchannel(0).getdata() == image.getdata()
    decoded.seek(1)
    assert decoded.getchannel(0).getdata() == image.getdata()


@pytest.mark.parametrize("mode", ["RGB", "RGBA"])
def test_float_luminance_conversion_preserves_fractional_values(mode: str) -> None:
    actual, expected = images(mode)
    assert_pixels_equal(actual.convert("F"), expected.convert("F"))


@pytest.mark.parametrize("mode", MODES)
def test_closed_mode_conversion(mode: str) -> None:
    actual, _ = images(mode)
    actual.close()
    assert_raises(ValueError, lambda: actual.convert("RGBA" if mode == "RGBa" else "LA" if mode == "La" else "RGB"), match="closed")


@pytest.mark.parametrize("mode", ["La", "RGBa", "RGBX"])
def test_codec_restrictions_and_explicit_conversion(mode: str) -> None:
    actual, _ = images(mode)
    assert_raises(Image.UnsupportedCodecError, lambda: actual.encode("PNG"))
    converted = actual.convert({"La": "LA", "RGBa": "RGBA", "RGBX": "RGB"}[mode])
    assert_pixels_equal(Image.open(BytesIO(converted.encode("PNG"))).to_pillow(), converted.to_pillow())
