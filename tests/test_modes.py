from __future__ import annotations

import struct
from io import BytesIO

import pytest
from PIL import Image as PillowImage, ImageOps as PillowOps

from blanket import Image, ImageChops, ImageOps


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


@pytest.mark.parametrize("mode", ["1", "LA"])
@pytest.mark.parametrize("destination", ["L", "RGB", "RGBA"])
def test_mode_conversion_matches_pillow_at_simd_boundaries(mode: str, destination: str) -> None:
    size = (37, 3)
    raw = bytes((i * 53 + 17) % 256 for i in range(size[1] * ((size[0] + 7) // 8 if mode == "1" else size[0] * 2)))
    blanket = Image.frombytes(mode, size, raw)
    pillow = PillowImage.frombytes(mode, size, raw)
    assert blanket.convert(destination).tobytes() == pillow.convert(destination).tobytes()


def test_pa_palette_alpha_conversion() -> None:
    image = Image.frombytes("PA", (2, 1), bytes([0, 128, 1, 255]))
    image.putpalette([255, 0, 0, 0, 0, 255])
    assert image.convert("RGBA").getdata() == [(255, 0, 0, 128), (0, 0, 255, 255)]
    assert image.convert("PA").getdata() == image.getdata()
    assert image.convert("P").getdata() == [0, 1]
    output = BytesIO()
    image.save(output, "PNG")
    assert Image.open(BytesIO(output.getvalue())).convert("RGBA").getdata() == image.convert("RGBA").getdata()


def test_pa_conversion_with_short_palette() -> None:
    image = Image.frombytes("PA", (3, 1), bytes([0, 128, 1, 200, 2, 255]))
    image.putpalette([255, 0, 0, 0, 0, 255])
    assert image.convert("L").getdata() == [76, 29, 0]
    assert image.convert("RGB").getdata() == [(255, 0, 0), (0, 0, 255), (0, 0, 0)]
    assert image.convert("RGBA").getdata() == [(255, 0, 0, 128), (0, 0, 255, 200), (0, 0, 0, 255)]
    assert image.convert("P").getdata() == [0, 1, 2]


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


@pytest.mark.parametrize("widths", [(9, 9), (17, 11), (7, 19)])
def test_bilevel_logical_operations_with_different_widths(widths: tuple[int, int]) -> None:
    from PIL import ImageChops as PillowChops

    first = Image.frombytes("1", (widths[0], 3), bytes((i * 53 + 17) % 256 for i in range((widths[0] + 7) // 8 * 3)))
    second = Image.frombytes("1", (widths[1], 2), bytes((i * 41 + 29) % 256 for i in range((widths[1] + 7) // 8 * 2)))
    pillow_first = PillowImage.frombytes("1", first.size, first.tobytes())
    pillow_second = PillowImage.frombytes("1", second.size, second.tobytes())
    for operation in ("logical_and", "logical_or", "logical_xor"):
        expected = getattr(PillowChops, operation)(pillow_first, pillow_second)
        actual = getattr(ImageChops, operation)(first, second)
        assert actual.size == expected.size
        assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("mode", ["I", "I;16", "I;16L", "I;16B"])
def test_integer_modes_pixel_storage_and_geometry(mode: str) -> None:
    values = [-42, 258, 65535] if mode == "I" else [0, 258, 65535]
    pillow = PillowImage.new(mode, (3, 1))
    blanket = Image.new(mode, (3, 1))
    pillow.putdata(values)
    blanket.putdata(values)
    assert blanket.mode == mode
    assert blanket.getbands() == pillow.getbands()
    assert blanket.getpixel((-1, 0)) == values[-1]
    assert blanket.getdata() == list(pillow.get_flattened_data())
    assert blanket.tobytes() == pillow.tobytes()
    assert Image.frombytes(mode, blanket.size, blanket.tobytes()).getdata() == values
    assert blanket.getextrema() == (min(values), max(values))
    assert list(blanket.to_pillow().get_flattened_data()) == values
    assert blanket.transpose(Image.Transpose.FLIP_LEFT_RIGHT).getdata() == list(pillow.transpose(PillowImage.Transpose.FLIP_LEFT_RIGHT).get_flattened_data())
    assert blanket.crop((1, 0, 3, 1)).getdata() == values[1:]
    blanket.putpixel((0, 0), values[-1])
    assert blanket.getpixel((0, 0)) == values[-1]


@pytest.mark.parametrize("mode", ["I", "I;16", "I;16L", "I;16B"])
def test_integer_mode_conversion_matches_pillow(mode: str) -> None:
    values = [-10, 10, 258, 65535] if mode == "I" else [0, 10, 258, 65535]
    pillow = PillowImage.new(mode, (4, 1))
    blanket = Image.new(mode, (4, 1))
    pillow.putdata(values)
    blanket.putdata(values)
    for destination in ("L", "RGB", "I"):
        assert blanket.convert(destination).getdata() == list(pillow.convert(destination).get_flattened_data())


@pytest.mark.parametrize("mode", ["I", "I;16", "I;16L", "I;16B"])
def test_integer_mode_conversion_simd_boundaries(mode: str) -> None:
    values = [-65536, -1, 0, 1, 254, 255, 256, 65535, 65536] if mode == "I" else [0, 1, 254, 255, 256, 32767, 32768, 65534, 65535]
    values = (values * 5)[:37]
    blanket = Image.new(mode, (len(values), 1))
    pillow = PillowImage.new(mode, (len(values), 1))
    blanket.putdata(values)
    pillow.putdata(values)
    for destination in ("L", "RGB", "RGBA", "I"):
        assert blanket.convert(destination).tobytes() == pillow.convert(destination).tobytes()


@pytest.mark.parametrize("mode", ["I", "I;16", "I;16L", "I;16B"])
@pytest.mark.parametrize("filter", [Image.Resampling.NEAREST, Image.Resampling.BILINEAR])
def test_integer_mode_resize_matches_pillow(mode: str, filter: Image.Resampling) -> None:
    values = [-10, 1000] if mode == "I" else [0, 65535]
    blanket = Image.new(mode, (2, 1))
    pillow = PillowImage.new(mode, (2, 1))
    blanket.putdata(values)
    pillow.putdata(values)
    assert blanket.resize((3, 1), filter).getdata() == list(pillow.resize((3, 1), filter).get_flattened_data())


def test_big_endian_integer_bilinear_resize_matches_pillow_byte_order() -> None:
    values = [1, 258, 1025, 32767]
    blanket = Image.new("I;16B", (4, 1))
    pillow = PillowImage.new("I;16B", (4, 1))
    blanket.putdata(values)
    pillow.putdata(values)
    assert blanket.resize((7, 1), Image.Resampling.BILINEAR).tobytes() == pillow.resize((7, 1), PillowImage.Resampling.BILINEAR).tobytes()


@pytest.mark.parametrize("mode,value", [("I", -1024), ("I;16", 1024), ("I;16L", 1024), ("I;16B", 1024)])
def test_integer_mode_new_fills_full_sample(mode: str, value: int) -> None:
    blanket = Image.new(mode, (37, 3), value)
    pillow = PillowImage.new(mode, (37, 3), value)
    assert blanket.tobytes() == pillow.tobytes()


@pytest.mark.parametrize("dtype", ["<i4", "<u2", ">u2"])
def test_integer_mode_fromarray(dtype: str) -> None:
    import numpy as np

    values = np.array([[0, 258, 65535]], dtype=dtype)
    blanket = Image.fromarray(values)
    pillow = PillowImage.fromarray(values)
    assert blanket.mode == pillow.mode
    assert blanket.getdata() == list(pillow.get_flattened_data())


@pytest.mark.parametrize("mode", ["I", "I;16", "I;16L", "I;16B"])
def test_integer_mode_png_save_preserves_sample_values(mode: str) -> None:
    values = [-1, 258, 65536] if mode == "I" else [0, 258, 65535]
    image = Image.new(mode, (3, 1))
    image.putdata(values)
    output = BytesIO()
    image.save(output, "PNG")
    reopened = PillowImage.open(BytesIO(output.getvalue()))
    expected = [max(0, min(65535, value)) for value in values]
    assert list(reopened.get_flattened_data()) == expected


def test_signed_integer_to_uint16_clips_out_of_range_values() -> None:
    image = Image.new("I", (3, 1))
    image.putdata([-1, 258, 70000])
    assert image.convert("I;16").getdata() == [0, 258, 65535]


@pytest.mark.parametrize("mode", ["I", "I;16", "I;16B"])
def test_integer_mode_paste(mode: str) -> None:
    target = Image.new(mode, (3, 1), 10)
    source = Image.new(mode, (2, 1), 1000)
    target.paste(source, (1, 0))
    assert target.getdata() == [10, 1000, 1000]
    target.paste(42, (0, 0, 1, 1))
    assert target.getdata() == [42, 1000, 1000]


def test_signed_integer_masked_paste_matches_pillow() -> None:
    mode = "I"
    target = Image.new(mode, (3, 1), 10)
    source = Image.new(mode, (3, 1), 1000)
    pillow_target = PillowImage.new(mode, (3, 1), 10)
    pillow_source = PillowImage.new(mode, (3, 1), 1000)
    mask = Image.frombytes("L", (3, 1), bytes([0, 128, 255]))
    pillow_mask = PillowImage.frombytes("L", (3, 1), mask.tobytes())
    target.paste(source, (0, 0), mask)
    pillow_target.paste(pillow_source, (0, 0), pillow_mask)
    assert target.getdata() == list(pillow_target.get_flattened_data())


def test_float_mode_storage_conversion_and_geometry() -> None:
    values = [-10.75, -1.5, -0.0, 0.25, 1.5, 254.9, 255.1, 300.5]
    blanket = Image.new("F", (len(values), 1))
    pillow = PillowImage.new("F", (len(values), 1))
    blanket.putdata(values)
    pillow.putdata(values)
    assert blanket.mode == "F"
    assert blanket.bit_depth == 32
    assert blanket.getbands() == ("F",)
    assert blanket.tobytes() == pillow.tobytes()
    assert blanket.getdata() == list(pillow.get_flattened_data())
    assert blanket.getextrema() == pillow.getextrema()
    assert Image.frombytes("F", blanket.size, blanket.tobytes()).getdata() == blanket.getdata()
    for mode in ("L", "I", "RGB", "RGBA", "LA", "HSV"):
        assert blanket.convert(mode).tobytes() == pillow.convert(mode).tobytes()
    for operation in (lambda im: im.crop((1, 0, 7, 1)), lambda im: im.transpose(PillowImage.Transpose.FLIP_LEFT_RIGHT)):
        assert operation(blanket).getdata() == list(operation(pillow).get_flattened_data())
    assert blanket.resize((13, 1), Image.Resampling.BILINEAR).getdata() == pytest.approx(list(pillow.resize((13, 1), PillowImage.Resampling.BILINEAR).get_flattened_data()), abs=0.0001)
    blanket.putpixel((0, 0), 1.25)
    pillow.putpixel((0, 0), 1.25)
    assert blanket.getpixel((0, 0)) == pillow.getpixel((0, 0))
    assert blanket.to_pillow().tobytes() == blanket.tobytes()


def test_float_mode_fromarray_and_paste() -> None:
    import numpy as np

    values = np.array([[1.25, -2.5, 1000.5]], dtype=np.float32)
    blanket = Image.fromarray(values)
    pillow = PillowImage.fromarray(values)
    assert blanket.mode == pillow.mode == "F"
    assert blanket.tobytes() == pillow.tobytes()
    blanket.paste(Image.new("F", (1, 1), 3.25), (1, 0))
    pillow.paste(PillowImage.new("F", (1, 1), 3.25), (1, 0))
    assert blanket.tobytes() == pillow.tobytes()
    assert Image.frombytes("F", (2, 1), struct.pack("<ff", -0.0, 0.0)).getbbox() == (0, 0, 1, 1)


@pytest.mark.parametrize("dtype", ["<f4", ">f4", "<f8", ">f8"])
def test_float_mode_fromarray_byte_order_and_width(dtype: str) -> None:
    import numpy as np

    values = np.array([[1.25, -2.5]], dtype=dtype)
    assert Image.fromarray(values).tobytes() == PillowImage.fromarray(values).tobytes()


def test_float_mode_reduce_transform_and_tiff_roundtrip() -> None:
    values = [1.25, 2.5, -3.75, 100.0, 0.0, 9.5]
    blanket = Image.new("F", (3, 2))
    pillow = PillowImage.new("F", (3, 2))
    blanket.putdata(values)
    pillow.putdata(values)
    assert blanket.reduce(2).tobytes() == pillow.reduce(2).tobytes()
    assert blanket.point(lambda value: value * 2).tobytes() == pillow.point(lambda value: value * 2).tobytes()
    assert ImageOps.expand(blanket, 1, 2.5).tobytes() == PillowOps.expand(pillow, 1, 2.5).tobytes()
    for operation in (
        lambda im: im.transform((3, 2), PillowImage.Transform.AFFINE, (1, 0, 0.25, 0, 1, 0), resample=PillowImage.Resampling.BILINEAR, fillcolor=-2.5),
        lambda im: im.rotate(30, resample=PillowImage.Resampling.BILINEAR, fillcolor=-2.5),
        lambda im: im.transform((3, 2), PillowImage.Transform.QUAD, (0, 0, 0, 2, 3, 2, 3, 0)),
    ):
        assert operation(blanket).getdata() == pytest.approx(list(operation(pillow).get_flattened_data()), abs=0.0001)
    output = BytesIO()
    blanket.save(output, "TIFF")
    assert list(PillowImage.open(BytesIO(output.getvalue())).get_flattened_data()) == blanket.getdata()
    pillow_output = BytesIO()
    pillow.save(pillow_output, "TIFF")
    assert Image.open(BytesIO(pillow_output.getvalue())).getdata() == blanket.getdata()
    with pytest.raises(OSError, match="cannot write mode F as PNG"):
        blanket.save(BytesIO(), "PNG")


@pytest.mark.parametrize(("source", "destination"), [("RGB", "HSV"), ("RGBA", "HSV"), ("L", "HSV"), ("LA", "HSV"), ("HSV", "RGB"), ("HSV", "RGBA"), ("HSV", "L")])
@pytest.mark.parametrize("count", [15, 16, 17, 257, 1800007])
def test_hsv_conversion_chunks_match_pillow(source: str, destination: str, count: int) -> None:
    import random

    channels = len(source)
    raw = random.Random(42).randbytes(count * channels)
    blanket = Image.frombytes(source, (count, 1), raw)
    pillow = PillowImage.frombytes(source, (count, 1), raw)
    assert blanket.convert(destination).tobytes() == pillow.convert(destination).tobytes()


def test_hsv_storage_conversion_and_geometry_match_pillow() -> None:
    raw = bytes(i * 17 % 256 for i in range(37 * 3))
    blanket = Image.frombytes("HSV", (37, 1), raw)
    pillow = PillowImage.frombytes("HSV", (37, 1), raw)
    assert blanket.mode == "HSV"
    assert blanket.getbands() == ("H", "S", "V")
    assert blanket.getpixel((0, 0)) == pillow.getpixel((0, 0))
    assert blanket.tobytes() == pillow.tobytes()
    blanket.putpixel((1, 0), (10, 20, 30))
    pillow.putpixel((1, 0), (10, 20, 30))
    assert blanket.getpixel((1, 0)) == (10, 20, 30)
    for destination in ("L", "RGB", "RGBA", "I", "LA"):
        assert blanket.convert(destination).tobytes() == pillow.convert(destination).tobytes()
    rgb = Image.frombytes("RGB", blanket.size, bytes((i * 29 + 3) % 256 for i in range(37 * 3)))
    assert rgb.convert("HSV").tobytes() == PillowImage.frombytes("RGB", rgb.size, rgb.tobytes()).convert("HSV").tobytes()
    gray = Image.frombytes("L", (8, 1), bytes(range(0, 256, 32)))
    assert gray.convert("HSV").tobytes() == PillowImage.frombytes("L", gray.size, gray.tobytes()).convert("HSV").tobytes()
    la = Image.frombytes("LA", (2, 1), bytes([40, 10, 200, 255]))
    assert la.convert("HSV").tobytes() == PillowImage.frombytes("LA", la.size, la.tobytes()).convert("HSV").tobytes()
    rgba = Image.frombytes("RGBA", (2, 1), bytes([255, 0, 0, 80, 0, 255, 0, 10]))
    assert rgba.convert("HSV").tobytes() == PillowImage.frombytes("RGBA", rgba.size, rgba.tobytes()).convert("HSV").tobytes()
    assert Image.new("HSV", (1, 1), "red").getpixel((0, 0)) == PillowImage.new("HSV", (1, 1), "red").getpixel((0, 0))
    wide = Image.frombytes("RGB", (2, 1), b"".join(v.to_bytes(2, "little") for v in (0, 0, 65535, 65535, 0, 0)), bit_depth=16)
    assert wide.convert("HSV").tobytes() == wide.convert("RGB", bit_depth=8).convert("HSV").tobytes()
    assert blanket.resize((11, 2), Image.Resampling.BILINEAR).tobytes() == pillow.resize((11, 2), PillowImage.Resampling.BILINEAR).tobytes()
    assert blanket.crop((2, 0, 9, 1)).getdata() == list(pillow.crop((2, 0, 9, 1)).get_flattened_data())
    assert blanket.split()[2].getdata() == list(pillow.split()[2].get_flattened_data())
    assert Image.merge("HSV", blanket.split()).tobytes() == blanket.tobytes()
    with pytest.raises(OSError, match="cannot write mode HSV"):
        blanket.save(BytesIO(), "PNG")
