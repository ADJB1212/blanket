from __future__ import annotations

import math

import pytest
from blanket import Image, ImageChops
from PIL import Image as PILImage
from PIL import ImageChops as PILChops

BINARY = ["difference", "multiply", "screen", "lighter", "darker", "add", "subtract", "add_modulo", "subtract_modulo", "soft_light", "hard_light", "overlay"]


@pytest.mark.parametrize("name", ["difference", "lighter", "darker", "add_modulo", "subtract_modulo", "invert"])
def test_memory_operations_large_parallel_tail(name: str) -> None:
    size = (1025, 1027)
    raw = (bytes(range(256)) * ((size[0] * size[1] * 4 + 255) // 256))[: size[0] * size[1] * 4]
    actual = Image.frombytes("RGBA", size, raw)
    expected = PILImage.frombytes("RGBA", size, raw)
    args = [] if name == "invert" else [Image.frombytes("RGBA", size, raw[::-1])]
    reference = [] if name == "invert" else [PILImage.frombytes("RGBA", size, raw[::-1])]
    assert getattr(ImageChops, name)(actual, *args).tobytes() == getattr(PILChops, name)(expected, *reference).tobytes()


def pair(mode: str, size: tuple[int, int], seed: int = 0) -> tuple[Image.Image, PILImage.Image]:
    raw = bytes((i * 37 + i // 11 + seed) % 256 for i in range(size[0] * size[1] * len(mode)))
    return Image.frombytes(mode, size, raw), PILImage.frombytes(mode, size, raw)


def same(actual: Image.Image, expected: PILImage.Image) -> None:
    assert (actual.mode, actual.size) == (expected.mode, expected.size)
    assert actual.tobytes() == expected.tobytes()
    assert actual.info == expected.info


@pytest.mark.parametrize("name", BINARY)
def test_every_channel_pair_matches_pillow(name: str) -> None:
    # Exhaustive coverage of integer rounding and the 127/128 blend thresholds.
    a = bytes(range(256)) * 256
    b = bytes(i // 256 for i in range(65536))
    first = Image.frombytes("L", (256, 256), a)
    second = Image.frombytes("L", (256, 256), b)
    expected = getattr(PILChops, name)(PILImage.frombytes("L", first.size, a), PILImage.frombytes("L", second.size, b))
    same(getattr(ImageChops, name)(first, second), expected)


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
@pytest.mark.parametrize("name", BINARY)
@pytest.mark.parametrize("sizes", [((17, 9), (11, 13)), ((0, 3), (4, 0)), ((513, 193), (513, 193))])
def test_binary_modes_sizes_and_parallel_chunks(mode: str, name: str, sizes: tuple[tuple[int, int], tuple[int, int]]) -> None:
    first, pil_first = pair(mode, sizes[0])
    second, pil_second = pair(mode, sizes[1], 71)
    first.info["test"] = pil_first.info["test"] = "retained"
    before = first.tobytes(), second.tobytes()
    result = getattr(ImageChops, name)(first, second)
    same(result, getattr(PILChops, name)(pil_first, pil_second))
    assert result.info is not first.info
    assert (first.tobytes(), second.tobytes()) == before
    if result.width and result.height:
        result.putpixel((0, 0), 0)
        assert (first.tobytes(), second.tobytes()) == before


@pytest.mark.parametrize("name", ["add", "subtract"])
@pytest.mark.parametrize("scale,offset", [(2, 0), (0.7, -13), (-1.3, 123), (1, 300), (1, -300), (math.inf, 17), (math.nan, 0), (1e-5, 0)])
def test_scaled_arithmetic(name: str, scale: float, offset: int) -> None:
    first, pil_first = pair("RGBA", (31, 19))
    second, pil_second = pair("RGBA", first.size, 99)
    same(getattr(ImageChops, name)(first, second, scale, offset), getattr(PILChops, name)(pil_first, pil_second, scale, offset))


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
@pytest.mark.parametrize("size", [(1, 1), (17, 9), (513, 193)])
@pytest.mark.parametrize("shift", [(0, 0), (3, None), (-4, None), (39, -27), (-100000, 100001)])
def test_offset(mode: str, size: tuple[int, int], shift: tuple[int, int | None]) -> None:
    image, pillow = pair(mode, size)
    image.info["test"] = pillow.info["test"] = 1
    before = image.tobytes()
    same(ImageChops.offset(image, *shift), PILChops.offset(pillow, *shift))
    assert image.tobytes() == before


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
def test_unary_and_compositing_helpers(mode: str) -> None:
    first, pil_first = pair(mode, (19, 11))
    second, pil_second = pair(mode, first.size, 99)
    mask, pil_mask = pair("L", first.size, 17)
    first.info["test"] = pil_first.info["test"] = 1
    same(ImageChops.invert(first), PILChops.invert(pil_first))
    same(ImageChops.duplicate(first), PILChops.duplicate(pil_first))
    same(ImageChops.constant(first, 73), PILChops.constant(pil_first, 73))
    same(ImageChops.blend(first, second, 0.3), PILChops.blend(pil_first, pil_second, 0.3))
    same(ImageChops.composite(first, second, mask), PILChops.composite(pil_first, pil_second, pil_mask))


@pytest.mark.parametrize("name", BINARY)
def test_invalid_modes_and_closed_inputs(name: str) -> None:
    first = Image.new("RGB", (3, 2))
    second = Image.new("RGBA", first.size)
    with pytest.raises(ValueError, match="images do not match"):
        getattr(ImageChops, name)(first, second)
    second = first.copy()
    second.close()
    with pytest.raises(ValueError, match="closed image"):
        getattr(ImageChops, name)(first, second)


@pytest.mark.parametrize("name", ["invert", "offset", "duplicate"])
def test_closed_unary_inputs(name: str) -> None:
    image = Image.new("L", (2, 2))
    image.close()
    with pytest.raises(ValueError, match="closed image"):
        getattr(ImageChops, name)(image, 1) if name == "offset" else getattr(ImageChops, name)(image)


@pytest.mark.parametrize("name", BINARY + ["invert", "offset"])
def test_palette_indices_and_palette_copy(name: str) -> None:
    first, pil_first = pair("L", (19, 11))
    second, pil_second = pair("L", first.size, 29)
    palette = bytes(range(256)) * 3
    for image in (first, second, pil_first, pil_second):
        image.putpalette(palette)
    if name in BINARY:
        result = getattr(ImageChops, name)(first, second)
        expected = getattr(PILChops, name)(pil_first, pil_second)
    elif name == "offset":
        result, expected = ImageChops.offset(first, -3), PILChops.offset(pil_first, -3)
    else:
        result, expected = ImageChops.invert(first), PILChops.invert(pil_first)
    same(result, expected)
    assert result.getpalette() == expected.getpalette()
    result.putpalette(bytes(768))
    assert first.getpalette() == list(palette)


@pytest.mark.parametrize("bit_depth", [10, 12, 16])
def test_high_depth_offset_preserves_samples_and_arithmetic_rejects(bit_depth: int) -> None:
    samples = [0, 256, (1 << bit_depth) - 1, 17, 1000, 88]
    image = Image.frombytes("L", (3, 2), b"".join(v.to_bytes(2, "little") for v in samples), bit_depth=bit_depth)
    result = ImageChops.offset(image, 1, -1)
    assert result.bit_depth == bit_depth
    assert result.tobytes() == b"".join(samples[i].to_bytes(2, "little") for i in [5, 3, 4, 2, 0, 1])
    for name in BINARY:
        with pytest.raises(ValueError, match="8-bit"):
            getattr(ImageChops, name)(image, image)
    with pytest.raises(ValueError, match="8-bit"):
        ImageChops.invert(image)
