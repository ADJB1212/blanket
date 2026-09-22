from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest
from blanket import Image, UnidentifiedImageError
from PIL import Image as PillowImage


@pytest.mark.parametrize("format", ["BMP", "GIF", "ICO"])
@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA", "P"])
def test_interoperability(format: str, mode: str) -> None:
    source = PillowImage.new(mode, (32, 32))
    if mode == "P":
        source.putpalette([255, 0, 0, 0, 255, 0] + [0] * 762)
    for y in range(32):
        for x in range(32):
            value = (x + y) % 2
            color = {"L": value * 255, "RGB": (value * 255, 50, 100), "RGBA": (value * 255, 50, 100, value * 255), "P": value}[mode]
            source.putpixel((x, y), color)
    blanket = Image.frombytes(mode, source.size, source.tobytes())
    if mode == "P":
        blanket.putpalette(source.getpalette())
    for writer in (source, blanket):
        output = BytesIO()
        writer.save(output, format)
        loaded = Image.open(output, formats=[format.lower()])
        expected = PillowImage.open(output).convert("RGBA")
        assert (loaded.format, loaded.size) == (format, source.size)
        assert loaded.convert("RGBA").tobytes() == expected.tobytes()
        # All fixture colors fit GIF's palette, so encoding should retain RGB.
        assert expected.convert("RGB").tobytes() == source.convert("RGB").tobytes()
        with pytest.raises(UnidentifiedImageError):
            Image.open(output, formats=["PNG"])


@pytest.mark.parametrize("format", ["BMP", "GIF", "ICO"])
def test_extension_and_validation(tmp_path: Path, format: str) -> None:
    source = Image.new("RGB", (16, 16), "red")
    path = tmp_path / f"image.{format}"
    source.save(path)
    assert Image.open(path).format == format
    with pytest.raises(TypeError, match="unsupported"):
        source.save(BytesIO(), format, quality=90)
    with pytest.raises(ValueError, match="high-bit-depth"):
        Image.frombytes("L", (1, 1), b"\0\0", bit_depth=16).save(BytesIO(), format)


@pytest.mark.parametrize("data", [b"BM", b"GIF87a", b"GIF89a", b"\0\0\x01\0"])
def test_truncated_input(data: bytes) -> None:
    with pytest.raises(UnidentifiedImageError):
        Image.open(BytesIO(data))


def test_animated_gif_first_frame() -> None:
    output = BytesIO()
    first = PillowImage.new("RGB", (32, 32), "red")
    first.save(output, "GIF", save_all=True, append_images=[PillowImage.new("RGB", first.size, "blue")])
    assert Image.open(output).convert("RGB").tobytes() == first.tobytes()


@pytest.mark.parametrize("bitmap_format", ["png", "bmp"])
def test_ico_largest_entry(bitmap_format: str) -> None:
    source = PillowImage.new("RGBA", (64, 64), (20, 40, 60, 255))
    output = BytesIO()
    source.save(output, "ICO", sizes=[(16, 16), (32, 32), (64, 64)], bitmap_format=bitmap_format)
    loaded = Image.open(output)
    assert loaded.size == source.size
    assert loaded.convert("RGBA").tobytes() == source.tobytes()


@pytest.mark.parametrize("size", [(257, 1), (1, 257), (0, 0)])
def test_ico_invalid_dimensions(size: tuple[int, int]) -> None:
    with pytest.raises(ValueError):
        Image.new("RGB", size).save(BytesIO(), "ICO")
