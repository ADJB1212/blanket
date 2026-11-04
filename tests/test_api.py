from __future__ import annotations

from io import BytesIO

import numpy as np
import pytest
from helpers import assert_raises, image_buffer

from blanket import Image, UnidentifiedImageError


def pixels(mode: str, size: tuple[int, int] = (17, 13)) -> bytes:
    channels = {"L": 1, "LA": 2, "RGB": 3, "RGBA": 4}[mode]
    width, height = size
    return bytes((x * 17 + y * 29 + channel * 53) % 256 for y in range(height) for x in range(width) for channel in range(channels))


@pytest.mark.parametrize(("mode", "channels"), [("L", ()), ("LA", (2,)), ("RGB", (3,)), ("RGBA", (4,))])
def test_fromarray(mode: str, channels: tuple[int, ...]) -> None:
    raw = pixels(mode)
    array = np.frombuffer(raw, dtype=np.uint8).reshape((13, 17, *channels)).copy()
    image = Image.fromarray(array)

    array[...] = 0
    assert (image.mode, image.size) == (mode, (17, 13))
    assert image.tobytes() == raw


def test_fromarray_supports_strided_arrays() -> None:
    source = np.arange(12 * 16 * 3, dtype=np.uint8).reshape(12, 16, 3)
    array = source[::2, ::2]
    image = Image.fromarray(array)
    assert (image.mode, image.size) == ("RGB", (8, 6))
    assert image.tobytes() == array.tobytes()


@pytest.mark.parametrize(("shape", "dtype"), [((3, 5, 3), np.float16), ((3, 5, 5), np.uint8), ((3, 5, 2), np.uint16), ((3, 5, 2), np.float32)])
def test_fromarray_rejects_unsupported_arrays(shape: tuple[int, ...], dtype: object) -> None:
    assert_raises(TypeError, lambda: Image.fromarray(np.zeros(shape, dtype=dtype)), match="cannot handle this data type")


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("channels,mode", [(3, "RGB;F"), (4, "RGBA;F")])
def test_fromarray_supports_float_color(dtype: object, channels: int, mode: str) -> None:
    values = np.arange(3 * 5 * channels, dtype=dtype).reshape(3, 5, channels) / 4
    image = Image.fromarray(values)
    assert (image.mode, image.size, image.bit_depth) == (mode, (5, 3), 32)
    assert image.tobytes() == values.astype("<f4").tobytes()


def test_path_format_inference(tmp_path: object) -> None:
    path = tmp_path / "image.png"
    source = Image.frombytes("L", (17, 13), pixels("L"))
    source.save(path)
    assert Image.open(path).tobytes() == source.tobytes()


def test_context_manager() -> None:
    source = Image.frombytes("RGBA", (17, 13), pixels("RGBA"))
    with source as entered:
        assert entered is source
    assert source.tobytes() == pixels("RGBA")
    source.close()
    assert_raises(ValueError, lambda: source.tobytes(), match="closed image")


def test_crop_metadata_is_independent() -> None:
    image = Image.frombytes("RGB", (17, 13), pixels("RGB"))
    image.info["test"] = 42
    result = image.crop((2, 3, 11, 10))

    assert result is not image
    assert result.info == image.info
    assert result.info is not image.info


def test_open_filters_and_errors() -> None:
    output = image_buffer(Image.frombytes("L", (2, 2), b"\0\x7f\x80\xff"), "PNG")
    assert Image.open(output, formats=["PNG"]).format == "PNG"
    assert_raises(UnidentifiedImageError, lambda: Image.open(output, formats=["JPEG"]))
    assert_raises(UnidentifiedImageError, lambda: Image.open(BytesIO(b"not an image")))
    assert_raises(ValueError, lambda: Image.open(output, mode="w"), match="bad mode")


def test_save_option_validation() -> None:
    source = Image.frombytes("RGBA", (1, 1), b"\0\0\0\xff")
    assert_raises(OSError, lambda: source.save(BytesIO(), "JPEG"), match="cannot write mode RGBA as JPEG")
    assert_raises(TypeError, lambda: source.save(BytesIO(), "PNG", quality=80), match="unsupported PNG save option")
    assert_raises(ValueError, lambda: source.save(BytesIO(), "PNG", compress_level=10), match="compress_level")
    assert_raises(ValueError, lambda: source.save(BytesIO()), match="unknown file extension")
