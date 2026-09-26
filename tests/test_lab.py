from __future__ import annotations

import random
from io import BytesIO

import pytest
from PIL import Image as PIL, ImageChops as PILChops, ImageEnhance as PILEnhance, ImageFilter as PILFilter

from blanket import Image, ImageChops, ImageEnhance, ImageFilter


@pytest.mark.parametrize("count", [17, 4097, 300007])
def test_srgb_conversion(count: int) -> None:
    raw = random.Random(73).randbytes(count * 3)
    actual = Image.frombytes("RGB", (count, 1), raw).convert("LAB")
    expected = PIL.frombytes("RGB", (count, 1), raw).convert("LAB")
    assert actual.tobytes() == expected.tobytes()
    reference = PIL.frombytes("LAB", actual.size, actual.tobytes()).convert("RGB")
    assert actual.convert("RGB").tobytes() == reference.tobytes()
    actual = Image.frombytes("LAB", (count, 1), raw)
    expected = PIL.frombytes("LAB", actual.size, raw)
    assert actual.convert("RGB").tobytes() == expected.convert("RGB").tobytes()


def test_reference_colors_and_raw_bytes() -> None:
    for rgb, lab in [((0, 0, 0), (0, 128, 128)), ((255, 255, 255), (255, 128, 128)), ((255, 0, 0), (138, 209, 198))]:
        image = Image.new("RGB", (1, 1), rgb).convert("LAB")
        assert image.getpixel((0, 0)) == lab
    image = Image.frombytes("LAB", (1, 1), bytes([50, 0, 255]))
    assert image.getpixel((0, 0)) == (50, 128, 127)
    assert image.tobytes() == bytes([50, 0, 255])
    assert Image.new("LAB", (1, 1), (50, 0, 255)).tobytes() == bytes([50, 128, 127])


@pytest.mark.parametrize("mode", ["1", "L", "LA", "RGB", "RGBA", "HSV", "CMYK", "YCbCr", "I", "I;16", "F", "P", "PA"])
def test_conversion_routes(mode: str) -> None:
    source = Image.new(mode, (2, 3))
    assert source.convert("LAB").tobytes() == source.convert("RGB").convert("LAB").tobytes()
    lab = Image.new("LAB", (2, 3), (128, 128, 128))
    assert lab.convert(mode).tobytes() == lab.convert("RGB").convert(mode).tobytes()


def test_operations() -> None:
    raw = random.Random(31).randbytes(17 * 13 * 3)
    actual, expected = Image.frombytes("LAB", (17, 13), raw), PIL.frombytes("LAB", (17, 13), raw)
    for name in ("Brightness", "Sharpness"):
        reference = PIL.frombytes("RGB", expected.size, bytes(v for pixel in actual.getdata() for v in pixel))
        enhanced = getattr(PILEnhance, name)(reference).enhance(0.6)
        assert getattr(ImageEnhance, name)(actual).enhance(0.6).getdata() == list(enhanced.get_flattened_data())
    for name in ("BLUR", "SMOOTH", "MedianFilter"):
        assert actual.filter(getattr(ImageFilter, name)()).tobytes() == expected.filter(getattr(PILFilter, name)()).tobytes()
    for name in ("invert", "add", "multiply", "difference"):
        args, ref_args = ([actual], [expected]) if name == "invert" else ([actual, actual], [expected, expected])
        assert getattr(ImageChops, name)(*args).tobytes() == getattr(PILChops, name)(*ref_args).tobytes()
    assert actual.point(list(range(256)) * 3).tobytes() == raw
    actual.paste((12, 34, 56), (1, 2, 5, 6))
    expected.paste((12, 34, 56), (1, 2, 5, 6))
    assert actual.tobytes() == expected.tobytes()
    for name in ("Color", "Contrast"):
        assert getattr(ImageEnhance, name)(actual).enhance(0.6).mode == "LAB"


@pytest.mark.parametrize("compression,byteorder,big_tiff", [("raw", "<", False), ("raw", ">", False), ("raw", "<", True), ("tiff_lzw", "<", False), ("tiff_adobe_deflate", "<", False)])
def test_tiff_interoperability(compression: str, byteorder: str, big_tiff: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    from PIL import TiffImagePlugin

    raw = random.Random(13).randbytes(17 * 13 * 3)
    expected = PIL.frombytes("LAB", (17, 13), raw)
    stream = BytesIO()
    rawmode, _, *options = TiffImagePlugin.SAVE_INFO["LAB"]
    prefix = b"II" if byteorder == "<" else b"MM"
    monkeypatch.setitem(TiffImagePlugin.SAVE_INFO, "LAB", (rawmode, prefix, *options))
    expected.save(stream, "TIFF", compression=compression, big_tiff=big_tiff)
    assert stream.getvalue().startswith(prefix)
    decoded = Image.open(BytesIO(stream.getvalue()))
    assert decoded.mode == "LAB"
    assert decoded.tobytes() == raw
    output = BytesIO()
    decoded.save(output, "TIFF")
    for module in (Image, PIL):
        reopened = module.open(BytesIO(output.getvalue()))
        assert reopened.mode == "LAB"
        assert reopened.tobytes() == raw


def test_icc_lab_tiff() -> None:
    pixels = random.Random(13).randbytes(17 * 13 * 3)
    stream = BytesIO()
    PIL.frombytes("RGB", (17, 13), pixels).save(stream, "TIFF", tiffinfo={262: 9})
    image = Image.open(BytesIO(stream.getvalue()))
    assert image.mode == "LAB"
    assert image.getdata() == list(zip(pixels[::3], pixels[1::3], pixels[2::3], strict=True))


def test_empty_and_high_depth() -> None:
    np = pytest.importorskip("numpy")
    for mode in ("RGB", "LAB"):
        image = Image.new(mode, (0, 2))
        for target in ("RGB", "LAB"):
            assert image.convert(target).tobytes() == b""
    with pytest.raises(ValueError):
        Image.fromarray(np.zeros((2, 3, 3), dtype=np.uint16), "LAB")
    image = Image.new("LAB", (2, 3), (128, 128, 128))
    assert image.convert("RGB", bit_depth=16).tobytes() == image.convert("RGB").convert("RGB", bit_depth=16).tobytes()
    assert image.convert("LAB").tobytes() == image.tobytes()
    assert image.convert("RGBA").getchannel("A").getextrema() == (255, 255)


def test_validation() -> None:
    from blanket.Compressor import LosslessImageCompressor

    image = Image.new("LAB", (3, 2))
    for mode in ("RGB", "LAB"):
        with pytest.raises(ValueError):
            Image.new(mode, (1, 1)).convert("LAB", bit_depth=16)
    with pytest.raises(ValueError):
        Image.frombytes("LAB", (1, 1), bytes(6), bit_depth=16)
    with pytest.raises(ValueError):
        Image.frombytes("LAB", (1, 1), bytes(2))
    with pytest.raises(ValueError):
        image.quantize()
    with pytest.raises(OSError):
        image.save(BytesIO(), "PNG")
    with pytest.raises(ValueError):
        image.save(BytesIO(), "TIFF", compressor=LosslessImageCompressor())
    image.close()
    with pytest.raises(ValueError):
        image.convert("RGB")
