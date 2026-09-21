from __future__ import annotations

from collections import Counter

import pytest
from PIL import Image as PIL

from blanket import Image


@pytest.mark.parametrize("mode", ["RGB", "RGBA"])
@pytest.mark.parametrize("limit", [16, 127, 200, 256])
def test_color_counts_merge_partitions_and_partial_tail(mode: str, limit: int) -> None:
    chunk = 128 * 1024
    tiles = [b"".join(bytes((v, v ^ 31, v ^ 127, v ^ 255)[: len(mode)]) for v in range(start, start + 128)) for start in (0, 128)]
    raw = tiles[0] * (chunk // 128) + tiles[1] * (chunk // 128) + tiles[0][: 17 * len(mode)]
    size = (chunk * 2 + 17, 1)
    actual = Image.frombytes(mode, size, raw).getcolors(limit)
    expected = PIL.frombytes(mode, size, raw).getcolors(limit)
    assert (sorted(actual) if actual is not None else None) == (sorted(expected) if expected is not None else None)


@pytest.mark.parametrize("container", [list, tuple])
def test_putdata_preserves_sequence_overrides(container: type) -> None:
    class Values(container):
        def __getitem__(self, index: int) -> int:
            return super().__getitem__(index) + 10

    image = Image.new("L", (3, 1))
    image.putdata(Values([1, 2, 3]))
    assert image.getdata() == [11, 12, 13]


def test_tuple_pixel_subclass_keeps_index_conversion() -> None:
    class Pixel(tuple):
        def __index__(self) -> int:
            return 0x030201

    image = Image.new("RGB", (1, 1))
    image.putdata([Pixel((9, 8, 7))])
    assert image.getpixel((0, 0)) == (1, 2, 3)


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
@pytest.mark.parametrize("alpha_only", [False, True])
def test_bbox_sparse_edges(mode: str, alpha_only: bool) -> None:
    actual, expected = Image.new(mode, (37, 19)), PIL.new(mode, (37, 19))
    for xy in [(18, 9), (5, 12), (31, 3), (0, 18), (36, 0)]:
        value = 7 if mode == "L" else (7, 0, 0) if mode == "RGB" else (7, 0, 0, 0)
        actual.putpixel(xy, value)
        expected.putpixel(xy, value)
        assert actual.getbbox(alpha_only=alpha_only) == expected.getbbox(alpha_only=alpha_only)
    if mode == "RGBA":
        actual.putpixel((20, 10), (0, 0, 0, 1))
        expected.putpixel((20, 10), (0, 0, 0, 1))
        assert actual.getbbox(alpha_only=alpha_only) == expected.getbbox(alpha_only=alpha_only)


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
def test_extrema_late_band_extremes(mode: str) -> None:
    values = [127] * (1025 * len(mode))
    for band in range(len(mode)):
        values[band * 257 * len(mode) + band] = 0
        values[(1024 - band * 257) * len(mode) + band] = 255
    raw = bytes(values)
    actual = Image.frombytes(mode, (1025, 1), raw)
    expected = PIL.frombytes(mode, (1025, 1), raw)
    assert actual.getextrema() == expected.getextrema()


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
@pytest.mark.parametrize("limit", [0, 1, 255, 256, 257, 1000])
def test_color_count_table_limits(mode: str, limit: int) -> None:
    raw = bytes(component for i in range(256) for component in (i, (i * 73) % 256, 0, 255)[:len(mode)]) * 3
    actual = Image.frombytes(mode, (256, 3), raw).getcolors(limit)
    expected = PIL.frombytes(mode, (256, 3), raw).getcolors(limit)
    assert (None if actual is None else sorted(actual)) == (None if expected is None else sorted(expected))


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA", "P"])
def test_data_and_colors(mode: str) -> None:
    raw = bytes((i * 31) % 256 for i in range(18 * len(mode)))
    actual = Image.frombytes(mode, (6, 3), raw)
    expected = PIL.frombytes(mode, (6, 3), raw)
    assert actual.getdata() == list(expected.get_flattened_data())
    assert actual.get_flattened_data() == expected.get_flattened_data()
    for band in range(len(mode)):
        assert actual.getdata(band) == list(expected.get_flattened_data(band))
    assert Counter(dict((v, n) for n, v in actual.getcolors())) == Counter(actual.getdata())
    assert sorted(actual.getcolors()) == sorted(expected.getcolors())
    assert actual.getcolors(1) is None
    values = actual.getdata()[::-1][:5]
    actual.putdata(values)
    expected.putdata(values)
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA", "P"])
@pytest.mark.parametrize("value", [-5, 270, 0x12345678])
def test_putpixel(mode: str, value: int) -> None:
    actual, expected = Image.new(mode, (3, 2)), PIL.new(mode, (3, 2))
    actual.putpixel((-1, -2), value)
    expected.putpixel((-1, -2), value)
    assert actual.tobytes() == expected.tobytes()
    with pytest.raises(IndexError):
        actual.putpixel((3, 0), value)


@pytest.mark.parametrize("mode,value", [("L", (400,)), ("RGB", (-5, 120, 400)), ("RGBA", (1, 2, 3)), ("RGBA", (1, 2, 3, 4))])
def test_tuple_pixels(mode: str, value: tuple[int, ...]) -> None:
    actual, expected = Image.new(mode, (2, 1)), PIL.new(mode, (2, 1))
    actual.putpixel((1, 0), value)
    expected.putpixel((1, 0), value)
    assert actual.tobytes() == expected.tobytes()
    if mode == "L":
        for image in (actual, expected):
            with pytest.raises(TypeError):
                image.putdata([value, value])
        return
    actual.putdata([value, value])
    expected.putdata([value, value])
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("scale,offset", [(1, 0), (1.7, -20), (-2, 255)])
def test_scaled_putdata(scale: float, offset: float) -> None:
    actual, expected = Image.new("L", (6, 1)), PIL.new("L", (6, 1))
    values = [-10, 1.9, 10, 120, 255, 300]
    actual.putdata(values, scale, offset)
    expected.putdata(values, scale, offset)
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
@pytest.mark.parametrize("alpha", [-1.5, 0, 0.1, 0.5, 1, 2.3])
def test_blend(mode: str, alpha: float) -> None:
    first = bytes((i * 31) % 256 for i in range(18 * len(mode)))
    second = first[::-1]
    a, b = (Image.frombytes(mode, (6, 3), v) for v in (first, second))
    pa, pb = (PIL.frombytes(mode, (6, 3), v) for v in (first, second))
    assert Image.blend(a, b, alpha).tobytes() == PIL.blend(pa, pb, alpha).tobytes()


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA", "P"])
@pytest.mark.parametrize("mask_mode", ["L", "RGBA"])
def test_composite(mode: str, mask_mode: str) -> None:
    first = bytes((i * 31) % 256 for i in range(18 * len(mode)))
    mask = bytes((i * 43) % 256 for i in range(18 * len(mask_mode)))
    a, b = (Image.frombytes(mode, (6, 3), v) for v in (first, first[::-1]))
    pa, pb = (PIL.frombytes(mode, (6, 3), v) for v in (first, first[::-1]))
    result = Image.composite(a, b, Image.frombytes(mask_mode, (6, 3), mask))
    assert result.tobytes() == PIL.composite(pa, pb, PIL.frombytes(mask_mode, (6, 3), mask)).tobytes()
    assert a.tobytes() == first and b.tobytes() == first[::-1]


def test_high_depth_reads() -> None:
    image = Image.frombytes("RGB", (2, 1), b"\x00\x01\x00\x02\x00\x03\x00\x04\x00\x05\x00\x06", bit_depth=12)
    assert image.getextrema() == ((256, 1024), (512, 1280), (768, 1536))
    assert image.getdata() == [(256, 512, 768), (1024, 1280, 1536)]
    assert image.getdata(1) == [512, 1280]
    assert sorted(image.getcolors()) == [(1, (256, 512, 768)), (1, (1024, 1280, 1536))]


def test_errors_and_empty() -> None:
    image = Image.new("RGB", (1, 1))
    with pytest.raises(TypeError):
        image.putdata([(1, 2, 3)] * 2)
    with pytest.raises(ValueError):
        Image.blend(image, Image.new("L", (1, 1)), 0.5)
    with pytest.raises(ValueError):
        Image.composite(image, image, Image.new("L", (2, 1)))
    empty = Image.new("L", (0, 0))
    assert empty.getdata() == [] and empty.getcolors() == []
    image.close()
    for method in (image.getdata, image.getcolors, image.getextrema):
        with pytest.raises(ValueError):
            method()
