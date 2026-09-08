from __future__ import annotations

from io import BytesIO

import numpy as np
import pytest
from blanket import Image, UnidentifiedImageError


def pixels(mode: str, size: tuple[int, int] = (17, 13)) -> bytes:
    channels = {"L": 1, "RGB": 3, "RGBA": 4}[mode]
    width, height = size
    return bytes((x * 17 + y * 29 + channel * 53) % 256 for y in range(height) for x in range(width) for channel in range(channels))


@pytest.mark.parametrize("mode, channels", [("L", ()), ("RGB", (3,)), ("RGBA", (4,))])
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


def test_fromarray_rejects_unsupported_arrays() -> None:
    with pytest.raises(TypeError, match="cannot handle this data type"):
        Image.fromarray(np.zeros((3, 5), dtype=np.float32))
    with pytest.raises(TypeError, match="cannot handle this data type"):
        Image.fromarray(np.zeros((3, 5, 2), dtype=np.uint8))


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
def test_png_roundtrip(mode: str) -> None:
    raw = pixels(mode)
    source = Image.frombytes(mode, (17, 13), raw)
    output = BytesIO()
    source.save(output, "PNG", compress_level=6)

    loaded = Image.open(output)
    assert loaded.format == "PNG"
    assert loaded.mode == mode
    assert loaded.size == (17, 13)
    assert loaded.tobytes() == raw


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
def test_jxl_lossless_roundtrip(mode: str) -> None:
    raw = pixels(mode, (8, 6))
    output = BytesIO()
    Image.frombytes(mode, (8, 6), raw).save(output, "JXL", lossless=True, effort=1)
    loaded = Image.open(output)
    assert (loaded.format, loaded.mode, loaded.size) == ("JXL", mode, (8, 6))
    assert loaded.tobytes() == raw


@pytest.mark.parametrize("mode", ["L", "RGB"])
def test_jpeg_roundtrip(mode: str) -> None:
    output = BytesIO()
    Image.frombytes(mode, (17, 13), pixels(mode)).save(output, "JPEG", quality=85)
    loaded = Image.open(output)
    assert (loaded.format, loaded.mode, loaded.size) == ("JPEG", mode, (17, 13))


def test_path_format_inference(tmp_path: object) -> None:
    path = tmp_path / "image.png"
    source = Image.frombytes("L", (17, 13), pixels("L"))
    source.save(path)
    assert Image.open(path).tobytes() == source.tobytes()


def test_conversions_and_context_manager() -> None:
    source = Image.frombytes("RGBA", (17, 13), pixels("RGBA"))
    assert source.convert("RGB").mode == "RGB"
    assert source.convert("L").mode == "L"

    with source as entered:
        assert entered is source
    with pytest.raises(ValueError, match="closed image"):
        source.tobytes()


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
@pytest.mark.parametrize("box", [None, (2, 3, 11, 10), (-3, -2, 8, 7), (12, 9, 21, 17), (0.5, 1.5, 9.5, 8.5), (2, 2, 2, 2)])
def test_crop_matches_pillow(mode: str, box: tuple[float, float, float, float] | None) -> None:
    from PIL import Image as PillowImage

    raw = pixels(mode)
    image = Image.frombytes(mode, (17, 13), raw)
    image.info["test"] = 42
    result = image.crop(box)
    expected = PillowImage.frombytes(mode, image.size, raw).crop(box)

    assert result is not image
    assert result.size == expected.size
    assert result.tobytes() == expected.tobytes()
    assert result.info == image.info
    assert result.info is not image.info


@pytest.mark.parametrize("box, message", [((2, 0, 1, 1), "right"), ((0, 2, 1, 1), "lower")])
def test_crop_rejects_reversed_box(box: tuple[int, int, int, int], message: str) -> None:
    image = Image.frombytes("L", (2, 2), b"abcd")
    with pytest.raises(ValueError, match=message):
        image.crop(box)


def test_open_filters_and_errors() -> None:
    output = BytesIO()
    Image.frombytes("L", (2, 2), b"\0\x7f\x80\xff").save(output, "PNG")
    assert Image.open(output, formats=["PNG"]).format == "PNG"
    with pytest.raises(UnidentifiedImageError):
        Image.open(output, formats=["JPEG"])
    with pytest.raises(UnidentifiedImageError):
        Image.open(BytesIO(b"not an image"))
    with pytest.raises(ValueError, match="bad mode"):
        Image.open(output, mode="w")


def test_save_option_validation() -> None:
    source = Image.frombytes("RGBA", (1, 1), b"\0\0\0\xff")
    with pytest.raises(OSError, match="cannot write mode RGBA as JPEG"):
        source.save(BytesIO(), "JPEG")
    with pytest.raises(TypeError, match="unsupported PNG save option"):
        source.save(BytesIO(), "PNG", quality=80)
    with pytest.raises(ValueError, match="compress_level"):
        source.save(BytesIO(), "PNG", compress_level=10)
    with pytest.raises(ValueError, match="unknown file extension"):
        source.save(BytesIO())
