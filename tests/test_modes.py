from __future__ import annotations

from io import BytesIO

import pytest
from PIL import Image as PillowImage

from blanket import Image, ImageChops


@pytest.mark.parametrize(("mode", "fill", "raw", "pixel"), [("1", 1, b"\xff\x80", 255), ("LA", (47, 128), bytes([47, 128] * 9), (47, 128)), ("PA", (2, 128), bytes([2, 128] * 9), (2, 128))])
def test_new_modes_pixel_access_and_bytes(mode: str, fill: int | tuple[int, int], raw: bytes, pixel: int | tuple[int, int]) -> None:
    image = Image.new(mode, (9, 1), fill)
    assert image.mode == mode
    assert image.getpixel((0, 0)) == fill
    assert image.getbands() == PillowImage.new(mode, (9, 1)).getbands()
    assert image.tobytes() == raw
    image.putpixel((0, 0), pixel)
    assert image.getpixel((0, 0)) == pixel
    assert Image.frombytes(mode, image.size, raw).tobytes() == raw


@pytest.mark.parametrize("mode", ["1", "LA", "PA"])
def test_new_modes_crop_and_transpose(mode: str) -> None:
    raw = {"1": b"\xa0\x40", "LA": bytes(range(20)), "PA": bytes(range(20))}[mode]
    image = Image.frombytes(mode, (5, 2), raw)
    pillow = PillowImage.frombytes(mode, (5, 2), raw)
    for operation in (lambda im: im.crop((1, 0, 4, 2)), lambda im: im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)):
        assert operation(image).getdata() == list(operation(pillow).get_flattened_data())


def test_la_conversion_and_png_roundtrip() -> None:
    image = Image.frombytes("LA", (2, 1), bytes([40, 70, 210, 255]))
    assert image.convert("RGBA").getdata() == [(40, 40, 40, 70), (210, 210, 210, 255)]
    assert image.convert("L").getdata() == [40, 210]
    output = BytesIO()
    image.save(output, "PNG")
    assert Image.open(BytesIO(output.getvalue())).mode == "LA"
    assert Image.open(BytesIO(output.getvalue())).tobytes() == image.tobytes()


def test_pa_palette_alpha_conversion() -> None:
    image = Image.frombytes("PA", (2, 1), bytes([0, 128, 1, 255]))
    image.putpalette([255, 0, 0, 0, 0, 255])
    assert image.convert("RGBA").getdata() == [(255, 0, 0, 128), (0, 0, 255, 255)]
    assert image.convert("PA").getdata() == image.getdata()
    assert image.convert("P").getdata() == [0, 1]
    output = BytesIO()
    image.save(output, "PNG")
    assert Image.open(BytesIO(output.getvalue())).convert("RGBA").getdata() == image.convert("RGBA").getdata()


def test_rgba_to_pa_retains_pixel_alpha() -> None:
    source = Image.frombytes("RGBA", (2, 1), bytes([255, 0, 0, 80, 0, 0, 255, 200]))
    result = source.convert("PA")
    assert result.mode == "PA"
    assert [pixel[1] for pixel in result.getdata()] == [80, 200]
    assert result.convert("RGBA").getdata() == source.getdata()


def test_pa_alpha_overrides_palette_alpha() -> None:
    image = Image.frombytes("PA", (1, 1), bytes([0, 128]))
    image.putpalette(bytes([255, 0, 0, 40]), "RGBA")
    assert image.convert("RGBA").getpixel((0, 0)) == (255, 0, 0, 128)
    palette_image = Image.frombytes("P", (1, 1), bytes([0]))
    palette_image.putpalette(bytes([255, 0, 0, 40]), "RGBA")
    assert palette_image.convert("PA").getpixel((0, 0)) == (0, 40)


@pytest.mark.parametrize("mode", ["LA", "PA"])
def test_two_channel_masked_paste(mode: str) -> None:
    target = Image.new(mode, (2, 1), (10, 20))
    source = Image.new(mode, (2, 1), (200, 240))
    mask = Image.frombytes("L", (2, 1), bytes([0, 255]))
    target.paste(source, (0, 0), mask)
    assert target.getdata() == [(10, 20), (200, 240)]


def test_la_resize_premultiplies_alpha() -> None:
    image = Image.frombytes("LA", (2, 1), bytes([255, 0, 0, 255]))
    pillow = PillowImage.frombytes("LA", image.size, image.tobytes())
    actual = image.resize((5, 1), Image.Resampling.BILINEAR)
    expected = pillow.resize((5, 1), PillowImage.Resampling.BILINEAR)
    assert actual.getdata() == list(expected.get_flattened_data())


@pytest.mark.parametrize("mode", ["1", "LA", "PA"])
def test_new_modes_reject_high_bit_depth(mode: str) -> None:
    with pytest.raises(ValueError, match="requires 8-bit"):
        Image.frombytes(mode, (1, 1), bytes([0, 0]) * (2 if mode != "1" else 1), bit_depth=16)


def test_bilevel_png_and_logical_operations() -> None:
    first = Image.frombytes("1", (9, 1), b"\xa5\x80")
    second = Image.frombytes("1", (9, 1), b"\x3c\x00")
    pillow_first = PillowImage.frombytes("1", first.size, first.tobytes())
    pillow_second = PillowImage.frombytes("1", second.size, second.tobytes())
    from PIL import ImageChops as PillowChops

    for operation in ("logical_and", "logical_or", "logical_xor"):
        assert getattr(ImageChops, operation)(first, second).tobytes() == getattr(PillowChops, operation)(pillow_first, pillow_second).tobytes()
    output = BytesIO()
    first.save(output, "PNG")
    reopened = Image.open(BytesIO(output.getvalue()))
    assert reopened.mode == "1"
    assert reopened.tobytes() == first.tobytes()
