from __future__ import annotations

from collections import Counter
from io import BytesIO
from typing import SupportsIndex, overload

import pytest
from PIL import Image as PIL

from blanket import Image


@pytest.mark.parametrize("mode", ["1", "L", "LA", "P", "PA", "RGB", "RGBA", "HSV", "CMYK", "YCbCr", "LAB", "I", "I;16", "I;16L", "I;16B", "F"])
def test_load_returns_mutable_pixel_access(mode):
    expected = PIL.new(mode, (3, 2))
    actual = Image.frombytes(mode, expected.size, expected.tobytes())
    pixels, reference = actual.load(), expected.load()
    channels = len(expected.getbands())
    value = 1.25 if mode == "F" else -123456 if mode == "I" else 60000 if mode.startswith("I;16") else 93 if channels == 1 else tuple(range(31, 31 + channels))
    for xy in [(0, 0), (-1, -1), (-3, -2), (1.9, 0.2)]:
        assert pixels[xy] == reference[xy]
        pixels[xy] = value
        reference[xy] = value
        assert pixels[xy] == reference[xy]
        assert actual.tobytes() == expected.tobytes()
    actual.putpixel((0, 0), value)
    assert pixels[0, 0] == actual.getpixel((0, 0))
    assert actual.load()[0, 0] == pixels[0, 0]


@pytest.mark.parametrize("mode", ["1", "L", "LA", "P", "PA", "I", "I;16B"])
def test_pixel_access_single_component_tuple(mode):
    actual, expected = Image.new(mode, (1, 1)), PIL.new(mode, (1, 1))
    pixels, reference = actual.load(), expected.load()
    pixels[0, 0] = (7,)
    reference[0, 0] = (7,)
    assert pixels[0, 0] == reference[0, 0]


@pytest.mark.parametrize(
    "xy,error",
    [
        ((3, 0), IndexError),
        ((0, 2), IndexError),
        ((-4, 0), IndexError),
        ((0, -3), IndexError),
        ((0,), TypeError),
        ((0, 0, 0), TypeError),
        ([0, 0], TypeError),
        ("ab", TypeError),
        (("0", 0), TypeError),
    ],
)
def test_pixel_access_invalid_coordinates(xy, error):
    actual, expected = Image.new("L", (3, 2)), PIL.new("L", (3, 2))
    for pixels in (actual.load(), expected.load()):
        with pytest.raises(error):
            pixels[xy]
        with pytest.raises(error):
            pixels[xy] = 7
    assert actual.tobytes() == expected.tobytes()


def test_pixel_access_index_protocol():
    class Coordinate:
        def __index__(self):
            return -1

    image = Image.new("L", (2, 2))
    pixels = image.load()
    pixels[Coordinate(), Coordinate()] = 42
    assert pixels[Coordinate(), Coordinate()] == image.getpixel((1, 1)) == 42


@pytest.mark.parametrize(
    "mode,depth,value",
    [("L", 10, 1023), ("RGB", 12, (0, 2048, 4095)), ("RGB", 16, (0, 32768, 65535)), ("RGBA", 16, (1, 2, 3, 65535)), ("RGB;F", 32, (-1.5, 2.0, 3.0)), ("RGBA;F", 32, (-1.5, 2.0, 3.0, 0.5))],
)
def test_pixel_access_retains_sample_precision(mode, depth, value):
    channels = len(value) if isinstance(value, tuple) else 1
    image = Image.frombytes(mode, (2, 2), bytes(4 * channels * (4 if depth == 32 else 2)), bit_depth=depth)
    pixels = image.load()
    pixels[-1, -1] = value
    assert pixels[1, 1] == image.getpixel((1, 1)) == value


def test_pixel_access_lifetime_and_closed_image():
    pixels = Image.new("L", (1, 1), 7).load()
    assert pixels[0, 0] == 7
    image = Image.new("L", (1, 1))
    pixels = image.load()
    image.close()
    for operation in (image.load, lambda: pixels[0, 0], lambda: pixels.__setitem__((0, 0), 7)):
        with pytest.raises(ValueError, match="closed"):
            operation()


@pytest.mark.parametrize("lazy", [False, True])
def test_pixel_access_stays_bound_to_selected_frame(lazy):
    stream = BytesIO()
    PIL.new("L", (2, 2), 10).save(stream, "TIFF", save_all=True, append_images=[PIL.new("L", (3, 1), 20)])
    image = Image.open(stream, lazy=lazy, frame_cache_size=2)
    first = image.load()
    image.seek(1)
    second = image.load()
    first[0, 0] = 30
    second[0, 0] = 40
    assert first[0, 0] == 30
    assert image.getpixel((0, 0)) == 40
    image.seek(0)
    assert image.load()[0, 0] == 30
    image.close()
    for pixels in (first, second):
        with pytest.raises(ValueError, match="closed"):
            pixels[0, 0]


@pytest.mark.parametrize("mode", ["1", "L", "LA", "P", "PA", "RGB", "RGBA", "HSV", "CMYK", "YCbCr", "LAB", "I", "I;16", "I;16L", "I;16B", "F"])
@pytest.mark.parametrize("size", [(0, 3), (257, 1)])
def test_pixel_snapshots_are_independent_and_immutable(mode: str, size: tuple[int, int]) -> None:
    expected = PIL.new(mode, size)
    if size[0]:
        channels = len(expected.getbands())
        expected.putdata([i % 256 if channels == 1 else tuple((i + c * 71) % 256 for c in range(channels)) for i in range(size[0])])
    expected = PIL.frombytes(mode, size, expected.tobytes())
    actual = Image.frombytes(mode, size, expected.tobytes())
    snapshot = actual.get_flattened_data()
    values = actual.getdata()
    assert type(snapshot) is tuple
    assert type(values) is list
    assert snapshot == expected.get_flattened_data()
    assert values == list(snapshot)
    for band in range(len(expected.getbands())):
        assert actual.get_flattened_data(band) == (snapshot if len(expected.getbands()) == 1 else expected.get_flattened_data(band))
    if size[0]:
        actual.putpixel((0, 0), 0)
    actual.close()
    assert snapshot == expected.get_flattened_data()
    assert values == list(snapshot)
    with pytest.raises(ValueError, match="closed"):
        actual.get_flattened_data()


class ListValues(list[int]):
    @overload
    def __getitem__(self, item: SupportsIndex) -> int: ...

    @overload
    def __getitem__(self, item: slice) -> list[int]: ...

    def __getitem__(self, item: SupportsIndex | slice) -> int | list[int]:
        value = super().__getitem__(item)
        return value if isinstance(item, slice) else value + 10


class TupleValues(tuple[int, ...]):
    @overload
    def __getitem__(self, item: SupportsIndex) -> int: ...

    @overload
    def __getitem__(self, item: slice) -> tuple[int, ...]: ...

    def __getitem__(self, item: SupportsIndex | slice) -> int | tuple[int, ...]:
        value = super().__getitem__(item)
        return value if isinstance(item, slice) else value + 10


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


@pytest.mark.parametrize("container", [ListValues, TupleValues])
def test_putdata_preserves_sequence_overrides(container: type[list[int] | tuple[int, ...]]) -> None:
    image = Image.new("L", (3, 1))
    image.putdata(container([1, 2, 3]))
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
    raw = bytes(component for i in range(256) for component in (i, (i * 73) % 256, 0, 255)[: len(mode)]) * 3
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
    assert Counter({v: n for n, v in actual.getcolors()}) == Counter(actual.getdata())
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


@pytest.mark.parametrize(("mode", "value"), [("L", (400,)), ("RGB", (-5, 120, 400)), ("RGBA", (1, 2, 3)), ("RGBA", (1, 2, 3, 4))])
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


@pytest.mark.parametrize(("scale", "offset"), [(1, 0), (1.7, -20), (-2, 255)])
def test_scaled_putdata(scale: float, offset: float) -> None:
    actual, expected = Image.new("L", (6, 1)), PIL.new("L", (6, 1))
    values = [-10, 1.9, 10, 120, 255, 300]
    actual.putdata(values, scale, offset)
    expected.putdata(values, scale, offset)
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("mode", ["I", "I;16", "I;16L", "I;16B", "F"])
@pytest.mark.parametrize(("size", "length"), [((0, 1), 0), ((1, 1), 0), ((1, 1), 1), ((3, 1), 1)])
def test_putdata_rejects_iterators_exceeding_reported_length(mode: str, size: tuple[int, int], length: int) -> None:
    class Values:
        def __len__(self):
            return length

        def __iter__(self):
            return iter([7] * (length + 1))

    image = Image.new(mode, size, 99)
    with pytest.raises(TypeError, match="too many data entries"):
        image.putdata(Values())
    assert image.getdata() == [7] * length + [99] * (size[0] - length)


@pytest.mark.parametrize("mode", ["I", "I;16", "I;16L", "I;16B", "F"])
def test_putdata_short_iterator_preserves_remaining_pixels(mode: str) -> None:
    class Values:
        def __len__(self):
            return 2

        def __iter__(self):
            return iter([7])

    image = Image.new(mode, (3, 1), 99)
    image.putdata(Values(), scale=2, offset=1)
    assert image.getdata() == [15, 99, 99]


@pytest.mark.parametrize("container", [list, tuple])
@pytest.mark.parametrize(("scale", "offset"), [(1, 0), (0.75, 20)])
@pytest.mark.parametrize("mode", ["1", "L", "P"])
def test_putdata_integer_conversion(container: type, scale: float, offset: float, mode: str) -> None:
    class Sample(int):
        def __float__(self) -> float:
            return 42.0

    values = container([-1, 0, 255, 256, 2**100, -(2**100), Sample(7), 1.5])
    actual, expected = Image.new(mode, (10, 1), 99), PIL.new(mode, (10, 1), 99)
    actual.putdata(values, scale, offset)
    expected.putdata(values, scale, offset)
    assert actual.tobytes() == expected.tobytes()
    with pytest.raises(OverflowError):
        actual.putdata([2**10000])
    actual.putdata([-1, True, False], scale, offset)
    expected.putdata([-1, True, False], scale, offset)
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
    assert a.tobytes() == first
    assert b.tobytes() == first[::-1]


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
    assert empty.getdata() == []
    assert empty.getcolors() == []
    image.close()
    for method in (image.getdata, image.getcolors, image.getextrema):
        with pytest.raises(ValueError):
            method()
