from __future__ import annotations

import random
from io import BytesIO

import pytest
from PIL import Image as PIL, ImageChops as PILChops, ImageEnhance as PILEnhance, ImageFilter as PILFilter

from blanket import Image, ImageChops, ImageEnhance, ImageFilter


@pytest.mark.parametrize("mode", ["CMYK", "YCbCr"])
@pytest.mark.parametrize("count", [17, 4097, 300007])
def test_raw_conversion_chunks(mode: str, count: int) -> None:
    raw = random.Random(73).randbytes(count * PIL.getmodebands(mode))
    actual, expected = Image.frombytes(mode, (count, 1), raw), PIL.frombytes(mode, (count, 1), raw)
    for target in ("RGB", "RGBA", "L", "LA", "HSV", "CMYK", "YCbCr", "F", "I", "1"):
        assert actual.convert(target).tobytes() == expected.convert(target, dither=PIL.Dither.NONE).tobytes()


@pytest.mark.parametrize("mode", ["CMYK", "YCbCr", "LAB"])
def test_arrays_palettes_and_depth(mode: str) -> None:
    np = pytest.importorskip("numpy")
    channels = PIL.getmodebands(mode)
    values = np.arange(35 * channels, dtype=np.uint8).reshape(5, 7, channels)
    image = Image.fromarray(values, mode)
    assert image.tobytes() == values.tobytes()
    assert np.array_equal(np.asarray(image.to_pillow()), values)
    for source in ("P", "PA"):
        reference = PIL.new(source, (7, 5))
        reference.putpalette(random.Random(7).randbytes(768))
        actual = Image.frombytes(source, reference.size, reference.tobytes())
        actual.putpalette(reference.getpalette())
        assert actual.convert(mode).tobytes() == reference.convert(mode).tobytes()
    for source in ("L", "RGB", "RGBA"):
        wide = Image.frombytes(source, (1, 1), bytes([0, 128] * len(source)), bit_depth=16)
        assert wide.convert(mode).tobytes() == wide.convert(source, bit_depth=8).convert(mode).tobytes()


@pytest.mark.parametrize("mode", ["CMYK", "YCbCr"])
@pytest.mark.parametrize("other", ["1", "L", "LA", "RGB", "RGBA", "HSV", "CMYK", "YCbCr", "I", "F"])
@pytest.mark.parametrize("reverse", [False, True])
def test_conversions(mode: str, other: str, reverse: bool) -> None:
    source, target = (other, mode) if reverse else (mode, other)
    rgb = PIL.frombytes("RGB", (257, 1), random.Random(7).randbytes(257 * 3))
    expected = rgb.convert(source)
    actual = Image.frombytes(source, expected.size, expected.tobytes())
    assert actual.convert(target).tobytes() == expected.convert(target, dither=PIL.Dither.NONE).tobytes()


@pytest.mark.parametrize("mode", ["CMYK", "YCbCr", "LAB"])
def test_storage_and_bands(mode: str) -> None:
    channels = PIL.getmodebands(mode)
    raw = random.Random(42).randbytes(35 * channels)
    actual, expected = Image.frombytes(mode, (7, 5), raw), PIL.frombytes(mode, (7, 5), raw)
    assert actual.getbands() == expected.getbands()
    assert actual.to_pillow().tobytes() == raw
    assert actual.getdata() == list(expected.get_flattened_data())
    assert actual.getextrema() == expected.getextrema()
    assert actual.histogram() == expected.histogram()
    assert actual.getbbox() == expected.getbbox()
    assert Image.merge(mode, actual.split()).tobytes() == raw
    for band in actual.getbands():
        assert actual.getchannel(band).tobytes() == expected.getchannel(band).tobytes()
    for color in [None, 0, 123456, "red", tuple(range(channels)), (10, 20, 30)]:
        assert Image.new(mode, (7, 5), color).tobytes() == PIL.new(mode, (7, 5), color).tobytes()
    actual.putpixel((1, 1), tuple(range(channels)))
    expected.putpixel((1, 1), tuple(range(channels)))
    assert actual.tobytes() == expected.tobytes()
    actual.putdata(list(expected.get_flattened_data()))
    assert actual.tobytes() == expected.tobytes()
    assert actual.copy().tobytes() == actual.tobytes()
    actual.close()
    with pytest.raises(ValueError):
        actual.convert("RGB")


@pytest.mark.parametrize("mode", ["CMYK", "YCbCr", "LAB"])
@pytest.mark.parametrize("method", range(6))
def test_geometry(mode: str, method: int) -> None:
    raw = random.Random(19).randbytes(17 * 13 * PIL.getmodebands(mode))
    actual, expected = Image.frombytes(mode, (17, 13), raw), PIL.frombytes(mode, (17, 13), raw)
    assert actual.resize((9, 7), method).tobytes() == expected.resize((9, 7), method).tobytes()
    assert actual.transpose(method).tobytes() == expected.transpose(method).tobytes()
    assert actual.crop((-1, 2, 12, 17)).tobytes() == expected.crop((-1, 2, 12, 17)).tobytes()
    assert actual.reduce(3).tobytes() == expected.reduce(3).tobytes()
    if method in (0, 2, 3):
        assert actual.rotate(23, method).tobytes() == expected.rotate(23, method).tobytes()
        assert actual.transform((11, 9), Image.Transform.EXTENT, (1, 2, 15, 11), method).tobytes() == expected.transform((11, 9), PIL.Transform.EXTENT, (1, 2, 15, 11), method).tobytes()


@pytest.mark.parametrize("mode", ["CMYK", "YCbCr"])
def test_operations(mode: str) -> None:
    raw = random.Random(31).randbytes(17 * 13 * PIL.getmodebands(mode))
    actual, expected = Image.frombytes(mode, (17, 13), raw), PIL.frombytes(mode, (17, 13), raw)
    for name in ("Color", "Contrast", "Brightness", "Sharpness"):
        assert getattr(ImageEnhance, name)(actual).enhance(0.6).tobytes() == getattr(PILEnhance, name)(expected).enhance(0.6).tobytes()
    for name in ("BLUR", "SMOOTH", "GaussianBlur", "MedianFilter"):
        if mode == "YCbCr" and name == "GaussianBlur":
            continue
        assert actual.filter(getattr(ImageFilter, name)).tobytes() == expected.filter(getattr(PILFilter, name)).tobytes()
    assert actual.point(list(reversed(range(256))) * PIL.getmodebands(mode)).tobytes() == expected.point(list(reversed(range(256))) * PIL.getmodebands(mode)).tobytes()
    assert ImageChops.invert(actual).tobytes() == PILChops.invert(expected).tobytes()
    actual.paste(0, (0, 0, 17, 13))
    expected.paste(0, (0, 0, 17, 13))
    color = tuple([200] * PIL.getmodebands(mode))
    actual.paste(color, (0, 0, 17, 13), Image.new("L", actual.size, 128))
    expected.paste(color, (0, 0, 17, 13), PIL.new("L", expected.size, 128))
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("format", ["JPEG", "TIFF"])
def test_cmyk_codecs(format: str) -> None:
    raw = random.Random(13).randbytes(17 * 13 * 4)
    actual, expected = Image.frombytes("CMYK", (17, 13), raw), PIL.frombytes("CMYK", (17, 13), raw)
    for image in (actual, expected):
        stream = BytesIO()
        image.save(stream, format)
        decoded = Image.open(BytesIO(stream.getvalue()))
        reference = PIL.open(BytesIO(stream.getvalue()))
        assert decoded.mode == reference.mode == "CMYK"
        assert decoded.tobytes() == reference.tobytes()
        if format == "TIFF":
            assert decoded.tobytes() == raw


@pytest.mark.parametrize("mode", ["CMYK", "YCbCr"])
def test_save_and_validation(mode: str) -> None:
    image = Image.new(mode, (3, 2))
    with pytest.raises(OSError, match=f"cannot write mode {mode}"):
        image.save(BytesIO(), "PNG")
    with pytest.raises(ValueError):
        image.convert(mode, bit_depth=16)
    with pytest.raises(ValueError):
        Image.frombytes(mode, (3, 2), bytes(6 * PIL.getmodebands(mode) * 2), bit_depth=16)
    with pytest.raises(ValueError):
        image.quantize()
    assert image.convert("P").mode == "P"
    stream = BytesIO()
    image.save(stream, "JPEG")
    assert PIL.open(BytesIO(stream.getvalue())).size == image.size
    if mode == "CMYK":
        stream = BytesIO()
        image.save(stream, "PDF")
        assert b"/DeviceCMYK" in stream.getvalue()
