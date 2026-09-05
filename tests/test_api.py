from __future__ import annotations

from io import BytesIO

import pytest

from blanket import Image, UnidentifiedImageError


def pixels(mode: str, size: tuple[int, int] = (17, 13)) -> bytes:
    channels = {"L": 1, "RGB": 3, "RGBA": 4}[mode]
    width, height = size
    return bytes(
        (x * 17 + y * 29 + channel * 53) % 256
        for y in range(height)
        for x in range(width)
        for channel in range(channels)
    )


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
    Image.frombytes(mode, (8, 6), raw).save(
        output, "JXL", lossless=True, effort=1
    )
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
    path = tmp_path / "image.png"  # type: ignore[operator]
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
