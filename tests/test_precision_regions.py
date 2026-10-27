from __future__ import annotations

import math
import struct
from io import BytesIO

import pytest
from PIL import Image as Pillow

from blanket import Image, UnidentifiedImageError


@pytest.mark.parametrize("mode,channels", [("RGB;F", 3), ("RGBA;F", 4)])
def test_float_color_storage(mode: str, channels: int) -> None:
    values = (-2.0, float("nan"), float("inf"), 0.5)[:channels]
    raw = struct.pack("<" + "f" * channels, *values)
    image = Image.frombytes(mode, (1, 1), raw)
    assert image.bit_depth == 32
    assert image.tobytes() == raw
    assert image.copy().tobytes() == raw
    assert image.crop((-1, 0, 1, 1)).tobytes() == bytes(channels * 4) + raw
    assert image.getpixel((0, 0))[0] == -2.0
    assert math.isnan(image.getpixel((0, 0))[1])
    assert image.getchannel("R").getpixel((0, 0)) == -2.0
    assert len(image.split()) == channels
    image.putpixel((0, 0), (0.25,) * channels)
    assert image.getpixel((0, 0)) == (0.25,) * channels
    image.putdata([(0.5,) * channels])
    assert image.getdata()[0] == (0.5,) * channels
    image.close()
    with pytest.raises(ValueError, match="closed"):
        image.resize((2, 2))


def test_float_conversion_and_nonfinite_policy() -> None:
    image = Image.new("RGB;F", (1, 1), (-1.0, float("nan"), float("inf")))
    assert image.convert("RGB").getpixel((0, 0)) == (0, 0, 255)
    assert Image.new("RGB", (1, 1), (0, 128, 255)).convert("RGB;F").getpixel((0, 0)) == pytest.approx((0, 128 / 255, 1))
    array = pytest.importorskip("numpy")
    for dtype in ("<f4", ">f4", "<f8", ">f8"):
        pixels = array.array([[[0.25, -1.0, 2.0]]], dtype=dtype)
        result = Image.fromarray(pixels)
        assert result.mode == "RGB;F"
        assert result.getpixel((0, 0)) == (0.25, -1.0, 2.0)


def test_linear_resize_and_alpha() -> None:
    image = Image.frombytes("RGB", (2, 1), bytes([0, 0, 0, 255, 255, 255]))
    assert image.resize((1, 1), Image.Resampling.BOX).getpixel((0, 0)) == (128,) * 3
    assert image.resize((1, 1), Image.Resampling.BOX, linear_light=True).getpixel((0, 0)) == (188,) * 3
    assert image.resize((1, 1), Image.Resampling.BOX, linear_light=True, transfer="linear").getpixel((0, 0)) == (128,) * 3
    rgba = Image.frombytes("RGBA;F", (2, 1), struct.pack("<8f", 1, 0, 0, 0, 0, 0, 1, 1))
    assert rgba.resize((1, 1), Image.Resampling.BOX).getpixel((0, 0)) == (0, 0, 1, 0.5)
    assert rgba.resize((1, 1), Image.Resampling.BOX, premultiply_alpha=False).getpixel((0, 0)) == (0.5, 0, 0.5, 0.5)
    with pytest.raises(ValueError, match="transfer"):
        image.resize((1, 1), linear_light=True, transfer="unknown")


@pytest.mark.parametrize("depth", [8, 10, 12, 16])
def test_linear_composite_depth(depth: int) -> None:
    maximum = (1 << depth) - 1

    def make(pixel: tuple[int, ...]) -> Image.Image:
        raw = bytes(pixel) if depth == 8 else struct.pack("<4H", *pixel)
        return Image.frombytes("RGBA", (1, 1), raw, bit_depth=depth)

    background = make((0, 0, 0, maximum))
    foreground = make((maximum, maximum, maximum, maximum // 2))
    result = Image.alpha_composite(background, foreground, linear_light=True)
    assert result.bit_depth == depth
    assert result.getpixel((0, 0))[0] / maximum == pytest.approx(0.735, abs=0.005)
    assert result.getpixel((0, 0))[3] == maximum


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA", "I;16"])
@pytest.mark.parametrize("compression", ["raw", "tiff_lzw", "tiff_adobe_deflate"])
def test_tiff_region_interoperability(mode: str, compression: str) -> None:
    source = Pillow.new(mode, (19, 13))
    for y in range(13):
        for x in range(19):
            value = x + y * 19
            pixel = value if mode in ("L", "I;16") else (value % 256, x, y) + ((128,) if mode == "RGBA" else ())
            source.putpixel((x, y), pixel)
    stream = BytesIO()
    source.save(stream, format="TIFF", compression=compression)
    bounds = (3, 2, 18, 12)
    result = Image.open_region(BytesIO(stream.getvalue()), bounds)
    assert result.info["decode_fallback"] is False
    expected = source.crop(bounds)
    assert result.tobytes() == expected.tobytes()


def test_region_fallback_and_validation() -> None:
    stream = BytesIO()
    Image.new("RGB", (5, 4), "red").save(stream, format="PNG")
    data = stream.getvalue()
    with pytest.raises(NotImplementedError, match="fallback"):
        Image.open_region(BytesIO(data), (1, 1, 4, 3))
    result = Image.open_region(BytesIO(data), (1, 1, 4, 3), fallback="full")
    assert result.size == (3, 2)
    assert result.getpixel((0, 0)) == (255, 0, 0)
    assert result.info["decode_fallback"] is True
    for bounds in ((-1, 0, 1, 1), (0, 0, 6, 1), (1, 1, 1, 2)):
        with pytest.raises(ValueError, match="region"):
            Image.open_region(BytesIO(data), bounds)
    with pytest.raises(UnidentifiedImageError):
        Image.open_region(BytesIO(data), (0, 0, 1, 1), formats=["TIFF"])


@pytest.mark.parametrize("channels,depth", [(1, 8), (3, 8), (4, 16)])
def test_tiled_region_edges_and_unrequested_corrupt_tile(channels: int, depth: int) -> None:
    width, height, tile = 5, 3, 2
    chunks = []
    for cy in range(2):
        for cx in range(3):
            values = [((cy * tile + y) * width + cx * tile + x) * channels + c for y in range(tile) for x in range(tile) for c in range(channels)]
            chunks.append(bytes(values) if depth == 8 else struct.pack("<" + "H" * len(values), *values))
    tags = [
        (256, 4, 1, width),
        (257, 4, 1, height),
        (258, 3, channels, depth),
        (259, 3, 1, 1),
        (262, 3, 1, 1 if channels == 1 else 2),
        (277, 3, 1, channels),
        (284, 3, 1, 1),
        (322, 4, 1, tile),
        (323, 4, 1, tile),
        (324, 4, 6, 0),
        (325, 4, 6, 0),
    ]
    if channels == 4:
        tags.append((338, 3, 1, 2))
    extra = 8 + 2 + len(tags) * 12 + 4
    bits_offset = extra
    offsets_offset = extra + (channels * 2 if channels > 1 else 0)
    counts_offset = offsets_offset + 24
    data_offset = counts_offset + 24
    tags = [(tag, kind, count, bits_offset if tag == 258 and channels > 1 else offsets_offset if tag == 324 else counts_offset if tag == 325 else value) for tag, kind, count, value in tags]
    header = b"II*\0" + struct.pack("<I", 8) + struct.pack("<H", len(tags))
    header += b"".join(struct.pack("<HHII", *tag) for tag in sorted(tags)) + bytes(4)
    if channels > 1:
        header += struct.pack("<" + "H" * channels, *([depth] * channels))
    offsets = [data_offset + i * len(chunks[0]) for i in range(6)]
    data = header + struct.pack("<6I", *offsets) + struct.pack("<6I", *([len(chunks[0])] * 6)) + b"".join(chunks)
    bounds = (1, 1, 5, 3)
    expected = Pillow.open(BytesIO(data)).crop(bounds)
    result = Image.open_region(BytesIO(data), bounds)
    if depth == 8:
        assert result.tobytes() == expected.tobytes()
    else:
        assert result.getpixel((3, 1)) == (56, 57, 58, 59)
    damaged = bytearray(data)
    struct.pack_into("<I", damaged, offsets_offset + 5 * 4, len(data) + 100)
    assert Image.open_region(BytesIO(damaged), (0, 0, 1, 1)).getpixel((0, 0)) == (0 if channels == 1 else tuple(range(channels)))
    with pytest.raises(OSError):
        Image.open_region(BytesIO(damaged), (4, 2, 5, 3))


@pytest.mark.parametrize("mode", ["RGB;F", "RGBA;F"])
@pytest.mark.parametrize("compression", ["raw", "tiff_lzw", "tiff_adobe_deflate"])
def test_float_tiff_roundtrip(mode: str, compression: str) -> None:
    image = Image.new(mode, (3, 2), (-0.5, 2.0, 0.25, 0.75)[: 3 if mode == "RGB;F" else 4])
    stream = BytesIO()
    image.save(stream, format="TIFF", compression=compression)
    assert Image.inspect(BytesIO(stream.getvalue())).mode == mode
    result = Image.open(BytesIO(stream.getvalue()))
    assert result.mode == mode
    assert result.tobytes() == image.tobytes()


def test_float_inplace_composite_and_transpose() -> None:
    image = Image.new("RGBA;F", (3, 2), (0.0, 0.0, 0.0, 1.0))
    image.alpha_composite(Image.new("RGBA;F", (1, 1), (2.0, -1.0, 0.0, 0.5)), dest=(1, 0))
    assert image.getpixel((1, 0)) == (1.0, -0.5, 0.0, 1.0)
    transposed = image.transpose(Image.Transpose.ROTATE_90)
    assert transposed.size == (2, 3)
    assert transposed.getpixel((0, 1)) == (1.0, -0.5, 0.0, 1.0)


def test_float_zero_and_nonfinite_alpha() -> None:
    zero = Image.new("RGBA;F", (2, 2), (-0.0, -0.0, -0.0, -0.0))
    assert zero.getbbox() is None
    assert zero.getcolors() == [(4, (-0.0, -0.0, -0.0, -0.0))]
    foreground = Image.new("RGBA;F", (2, 2), (float("inf"), float("nan"), -2.0, float("nan")))
    result = Image.alpha_composite(zero, foreground)
    assert result.getpixel((0, 0)) == (0.0, 0.0, 0.0, 0.0)
    assert foreground.resize((1, 1)).getpixel((0, 0)) == (0.0, 0.0, 0.0, 0.0)
    for mode in ("L", "F", "CMYK"):
        with pytest.raises(ValueError, match="RGB"):
            Image.new(mode, (2, 2)).resize((1, 1), linear_light=True)
