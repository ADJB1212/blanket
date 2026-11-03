from __future__ import annotations

import struct
from io import BytesIO
from typing import ClassVar

import numpy as np
import pytest
from PIL import Image as PILImage, ImageDraw as PILDraw

from blanket import Image, ImageDraw


@pytest.mark.parametrize("mode", ["L", "RGB"])
@pytest.mark.parametrize(("target", "dither"), [(target, dither) for target in ("1", "P", "L", "RGB", "LA", "La", "RGBA", "RGBa") for dither in ((Image.Dither.NONE, Image.Dither.FLOYDSTEINBERG) if target in ("1", "P") else (Image.Dither.NONE,))])
def test_conversion_palette_dither_and_transparency(mode: str, target: str, dither: Image.Dither) -> None:
    channels = len(mode)
    raw = bytes((i * 37 + i // 11) % 256 for i in range(19 * 7 * channels))
    actual = Image.frombytes(mode, (19, 7), raw)
    expected = PILImage.frombytes(mode, actual.size, raw)
    key = actual.getpixel((0, 0))
    actual.info["transparency"] = expected.info["transparency"] = key
    if mode == "L" and target in ("La", "RGBa"):
        for image in (actual, expected):
            with pytest.raises(ValueError):
                image.convert(target, dither=dither)
        return
    result, reference = actual.convert(target, dither=dither), expected.convert(target, dither=dither)
    assert result.mode == reference.mode
    assert result.tobytes() == reference.tobytes()
    assert result.getpalette() == reference.getpalette()
    assert result.info == reference.info
    assert actual.tobytes() == raw


@pytest.mark.parametrize("target", ["L", "RGB", "LA", "RGBA", "PA"])
def test_indexed_transparency_conversion(target: str) -> None:
    actual, expected = Image.new("P", (5, 3)), PILImage.new("P", (5, 3))
    palette = bytes([20, 40, 70, 90, 140, 210, 39, 211, 103])
    for image in (actual, expected):
        image.putpalette(palette)
        image.frombytes(bytes(i % 3 for i in range(15)))
        image.info["transparency"] = 1
    result, reference = actual.convert(target), expected.convert(target)
    assert result.tobytes() == reference.tobytes()
    assert result.getpalette() == reference.getpalette()
    assert result.info == reference.info


@pytest.mark.parametrize("target", ["L", "RGB"])
def test_matrix_conversion_matches_pillow(target: str) -> None:
    raw = bytes((i * 37 + i // 11) % 256 for i in range(19 * 7 * 3))
    actual = Image.frombytes("RGB", (19, 7), raw)
    expected = PILImage.frombytes("RGB", actual.size, raw)
    matrix = (0.412453, 0.357580, 0.180423, 0, 0.212671, 0.715160, 0.072169, 0, 0.019334, 0.119193, 0.950227, 0)
    for image in (actual, expected):
        image.info["transparency"] = (255, 0, 0)
    result, reference = actual.convert(target, matrix), expected.convert(target, matrix)
    assert result.tobytes() == reference.tobytes()
    assert result.info == reference.info


def test_uint16_tiff_decodes_as_integer_mode() -> None:
    values = [0, 1, 255, 256, 447, 32768, 65535] * 5
    raw = struct.pack(f"<{len(values)}H", *values)
    source = PILImage.frombytes("I;16", (7, 5), raw)
    encoded = BytesIO()
    source.save(encoded, "TIFF")
    image = Image.open(BytesIO(encoded.getvalue()))
    assert image.mode == "I;16"
    assert image.tobytes() == raw
    assert image.getextrema() == (0, 65535)
    assert image.convert("F").tobytes() == source.convert("F").tobytes()
    assert Image.inspect(BytesIO(encoded.getvalue())).mode == "I;16"


def test_high_depth_grayscale_to_float_preserves_sample_values() -> None:
    samples = [0, 1, 255, 256, 447, 32768, 65535] * 5
    raw = struct.pack(f"<{len(samples)}H", *samples)
    image = Image.frombytes("L", (len(samples), 1), raw, bit_depth=16)
    expected = PILImage.frombytes("I;16", image.size, raw).convert("F")
    actual = image.convert("F")
    assert actual.tobytes() == expected.tobytes()
    assert image.tobytes() == raw


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA", "P", "PA"])
@pytest.mark.parametrize("transparent", [False, True])
def test_default_conversion(mode: str, transparent: bool) -> None:
    actual, expected = Image.new(mode, (3, 2)), PILImage.new(mode, (3, 2))
    if transparent:
        actual.info["transparency"] = expected.info["transparency"] = 0
    result, reference = actual.convert(), expected.convert()
    assert actual.convert("").tobytes() == reference.tobytes()
    assert result.mode == reference.mode
    assert result.tobytes() == reference.tobytes()
    result.putpixel((0, 0), 71)
    assert actual.getpixel((0, 0)) != result.getpixel((0, 0))


def test_array_validation_and_indexed_import() -> None:
    class MissingType:
        __array_interface__: ClassVar[dict[str, object]] = {"shape": (2, 3)}

    class MissingBytes:
        __array_interface__: ClassVar[dict[str, object]] = {"shape": (2, 3), "typestr": "|u1", "strides": (6, 2)}

    with pytest.raises(TypeError):
        Image.fromarray(MissingType())
    with pytest.raises(ValueError):
        Image.fromarray(MissingBytes(), "L")
    array = np.arange(24, dtype=np.uint8).reshape(4, 6)[:, ::2]
    actual, expected = Image.fromarray(array, "P"), PILImage.fromarray(array, "P")
    assert actual.mode == "P"
    assert actual.tobytes() == expected.tobytes()
    assert actual.getpalette() == expected.getpalette()


@pytest.mark.parametrize("mode", ["1", "L", "RGB", "RGBA", "I;16B", "F"])
def test_replace_raw_pixels_is_atomic_and_preserves_metadata(mode: str) -> None:
    source = PILImage.new(mode, (3, 2), 71)
    image = Image.new(mode, source.size)
    image.info["note"] = "retained"
    image.frombytes(source.tobytes())
    assert image.tobytes() == source.tobytes()
    assert image.info["note"] == "retained"
    with pytest.raises(ValueError):
        image.frombytes(b"")
    assert image.tobytes() == source.tobytes()


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
@pytest.mark.parametrize("mask_mode", ["1", "L", "LA", "RGBA"])
def test_bitmap_and_circle_match_pillow(mode: str, mask_mode: str) -> None:
    actual, expected = Image.new(mode, (19, 17)), PILImage.new(mode, (19, 17))
    mask = PILImage.new(mask_mode, (5, 4), 71 if mask_mode in ("1", "L") else (71, 83) if mask_mode == "LA" else (10, 20, 30, 83))
    if mask_mode == "LA":
        with pytest.raises(ValueError):
            ImageDraw.Draw(actual).bitmap((0, 0), Image.frombytes(mask.mode, mask.size, mask.tobytes()))
        return
    ImageDraw.Draw(actual).bitmap((-1, 2), Image.frombytes(mask.mode, mask.size, mask.tobytes()), fill="white")
    PILDraw.Draw(expected).bitmap((-1, 2), mask, fill="white")
    assert actual.tobytes() == expected.tobytes()
    ImageDraw.Draw(actual).circle((9, 8), 5, fill="red", outline="white", width=2)
    PILDraw.Draw(expected).circle((9, 8), 5, fill="red", outline="white", width=2)
    assert actual.tobytes() == expected.tobytes()
