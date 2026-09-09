from __future__ import annotations

import random
from array import array
from io import BytesIO

import numpy as np
import pytest
from blanket import Image, ImageFilter
from PIL import Image as PillowImage
from PIL import ImageFilter as PillowFilter

BUILTINS = ("BLUR", "CONTOUR", "DETAIL", "EDGE_ENHANCE", "EDGE_ENHANCE_MORE", "EMBOSS", "FIND_EDGES", "SHARPEN", "SMOOTH", "SMOOTH_MORE")
NEIGHBORHOOD_SIZES = [
    pytest.param(1, marks=pytest.mark.skip(reason="Pillow crashes with filter size 1")),
    3,
    5,
    9,
]


def images(mode: str, size: tuple[int, int]) -> tuple[Image.Image, PillowImage.Image]:
    data = random.Random(71).randbytes(size[0] * size[1] * len(mode))
    return Image.frombytes(mode, size, data), PillowImage.frombytes(mode, size, data)


def compare(mode: str, size: tuple[int, int], actual_filter: ImageFilter.Filter, expected_filter: PillowFilter.Filter) -> None:
    image, pillow = images(mode, size)
    original = image.tobytes()
    image.info["test"] = pillow.info["test"] = "metadata"
    actual = image.filter(actual_filter)
    expected = pillow.filter(expected_filter)
    assert (actual.mode, actual.size, actual.format, actual.info) == (expected.mode, expected.size, expected.format, expected.info)
    assert actual.tobytes() == expected.tobytes()
    assert image.tobytes() == original
    actual.close()
    assert image.tobytes() == original


@pytest.mark.parametrize("name", BUILTINS)
@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
@pytest.mark.parametrize("size", [(1, 1), (2, 7), (4, 4), (17, 13)])
def test_builtin_filters(name: str, mode: str, size: tuple[int, int]) -> None:
    cls, reference = getattr(ImageFilter, name), getattr(PillowFilter, name)
    assert (cls.name, cls.filterargs) == (reference.name, reference.filterargs)
    compare(mode, size, cls, reference)
    compare(mode, size, cls(), reference())


@pytest.mark.parametrize("size", [3, 5])
@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
@pytest.mark.parametrize("scale,offset", [(None, 0), (1, -1.5), (2.75, 20.1), (-3, 128), (0, 0)])
def test_custom_kernel(size: int, mode: str, scale: float | None, offset: float) -> None:
    kernel = [random.Random(43 + i).uniform(-2, 3) for i in range(size * size)]
    actual = ImageFilter.Kernel((size, size), kernel, scale, offset)
    expected = PillowFilter.Kernel((size, size), kernel, scale, offset)
    assert actual.filterargs == expected.filterargs
    compare(mode, (19, 13), actual, expected)


@pytest.mark.parametrize("name", ["RankFilter", "MedianFilter", "MinFilter", "MaxFilter", "ModeFilter"])
@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
@pytest.mark.parametrize("size", NEIGHBORHOOD_SIZES)
def test_neighborhood_filters(name: str, mode: str, size: int) -> None:
    args = (size, size * size // 3) if name == "RankFilter" else (size,)
    compare(mode, (7, 5), getattr(ImageFilter, name)(*args), getattr(PillowFilter, name)(*args))


@pytest.mark.parametrize("size", [-3, -1, 0, 2, 4, 12])
def test_mode_filter_special_sizes(size: int) -> None:
    compare("RGB", (7, 5), ImageFilter.ModeFilter(size), PillowFilter.ModeFilter(size))


def test_mode_filter_ties_and_rare_values() -> None:
    for values in (bytes([1, 1, 1, 2, 2, 2, 9, 8, 7]), bytes(range(9))):
        image = Image.frombytes("L", (3, 3), values)
        pillow = PillowImage.frombytes("L", (3, 3), values)
        assert image.filter(ImageFilter.ModeFilter).tobytes() == pillow.filter(PillowFilter.ModeFilter).tobytes()


@pytest.mark.parametrize("name", ["BoxBlur", "GaussianBlur"])
@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
@pytest.mark.parametrize("radius", [0, 0.1, 0.5, 1, 1.3, 2, 10, 100.5, (0, 2), (2.5, 0), (0.3, 1.7), [1, 2]])
@pytest.mark.parametrize("size", [(1, 1), (1, 7), (17, 13)])
def test_blurs(name: str, mode: str, radius: float | tuple[float, float], size: tuple[int, int]) -> None:
    compare(mode, size, getattr(ImageFilter, name)(radius), getattr(PillowFilter, name)(radius))


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
@pytest.mark.parametrize("args", [(), (0,), (0.5, 75, 0), (2, 150, 3), (3, -150, -1), (-2, 120, 5), (1, 0, 255)])
def test_unsharp(mode: str, args: tuple[float, ...]) -> None:
    compare(mode, (19, 13), ImageFilter.UnsharpMask(*args), PillowFilter.UnsharpMask(*args))


@pytest.mark.parametrize("name,args", [("Kernel", ((2, 2), [1] * 4)), ("RankFilter", (3, 9)), ("RankFilter", (2, 1)), ("MedianFilter", (0,)), ("MedianFilter", (-1,)), ("MinFilter", (23171,)), ("ModeFilter", (1.5,)), ("BoxBlur", (-1,)), ("BoxBlur", ((1, -1),)), ("GaussianBlur", ((1, 2, 3),)), ("UnsharpMask", (1, 1.5))])
def test_invalid_parameters(name: str, args: tuple[object, ...]) -> None:
    image, pillow = images("RGB", (7, 7))
    with pytest.raises(Exception) as expected:
        pillow.filter(getattr(PillowFilter, name)(*args))
    with pytest.raises(type(expected.value)):
        image.filter(getattr(ImageFilter, name)(*args))


@pytest.mark.parametrize("size", [(0, 0), (0, 5), (5, 0)])
@pytest.mark.parametrize("name,args", [("BLUR", ()), ("GaussianBlur", (2,)), ("BoxBlur", (3,)), ("UnsharpMask", ())])
def test_empty_images(size: tuple[int, int], name: str, args: tuple[object, ...]) -> None:
    compare("RGBA", size, getattr(ImageFilter, name)(*args), getattr(PillowFilter, name)(*args))


@pytest.mark.parametrize("name,args", [("SMOOTH", ()), ("BLUR", ()), ("MedianFilter", (3,)), ("ModeFilter", (5,)), ("GaussianBlur", (2.1,)), ("BoxBlur", (1.75,)), ("UnsharpMask", ())])
def test_parallel_filters(name: str, args: tuple[object, ...]) -> None:
    compare("RGBA", (321, 289), getattr(ImageFilter, name)(*args), getattr(PillowFilter, name)(*args))


@pytest.mark.parametrize("mode", ["RGB", "RGBA"])
@pytest.mark.parametrize("size", [2, 3, (3, 4, 5), 17, 65])
@pytest.mark.parametrize("channels", [3, 4])
def test_color_lut(mode: str, size: int | tuple[int, int, int], channels: int) -> None:
    callback = lambda r, g, b: (r * r + b * 0.2, 1.2 - g, b - r * 0.3) + ((r * g,) if channels == 4 else ())
    target = "RGBA" if channels == 4 else None
    actual = ImageFilter.Color3DLUT.generate(size, callback, channels, target)
    expected = PillowFilter.Color3DLUT.generate(size, callback, channels, target)
    assert (actual.size, actual.channels, actual.mode, actual.table) == (expected.size, expected.channels, expected.mode, expected.table)
    assert repr(actual) == repr(expected)
    compare(mode, (37, 29), actual, expected)


@pytest.mark.parametrize("storage", [list, tuple, lambda values: array("f", values), lambda values: np.array(values, dtype=np.float32), lambda values: np.array(values, dtype=np.float64).reshape(2, 2, 2, 3)])
def test_lut_storage_and_clipping(storage: type) -> None:
    values = [random.Random(29 + i).uniform(-4, 4) for i in range(24)]
    actual = ImageFilter.Color3DLUT(2, storage(values))
    expected = PillowFilter.Color3DLUT(2, storage(values))
    compare("RGB", (31, 27), actual, expected)


@pytest.mark.parametrize("channels", [3, 4])
def test_parallel_lut(channels: int) -> None:
    def callback(r: float, g: float, b: float) -> tuple[float, ...]:
        return (b, r * r, 1 - g) + ((r * b,) if channels == 4 else ())

    actual = ImageFilter.Color3DLUT.generate((3, 5, 7), callback, channels)
    expected = PillowFilter.Color3DLUT.generate((3, 5, 7), callback, channels)
    compare("RGBA", (321, 289), actual, expected)


def test_kernel_coefficients_remain_mutable() -> None:
    coefficients = [0.0] * 9
    actual = ImageFilter.Kernel((3, 3), coefficients, 1)
    expected = PillowFilter.Kernel((3, 3), coefficients, 1)
    for position in (0, 4, 8):
        coefficients[:] = [0.0] * 9
        coefficients[position] = 1.0
        compare("RGB", (7, 9), actual, expected)


def test_native_filter_size_and_nonfinite_radii_are_safe() -> None:
    from blanket._blanket import filter_rank

    image, _ = images("RGB", (2, 2))
    with pytest.raises(ValueError, match="filter size too large"):
        filter_rank(image._native, 2**31 - 1, 0)
    for radius in (float("inf"), float("nan"), 1e30):
        for cls in (ImageFilter.GaussianBlur, ImageFilter.BoxBlur):
            with pytest.raises(ValueError, match="radius"):
                image.filter(cls(radius))


def test_lut_transform_copy_and_mutation() -> None:
    actual = ImageFilter.Color3DLUT.generate((2, 3, 4), lambda r, g, b: (r, g, b))
    expected = PillowFilter.Color3DLUT.generate((2, 3, 4), lambda r, g, b: (r, g, b))
    for normals in (False, True):
        callback = lambda *values: tuple(1 - value for value in values[-3:]) + (0.5,)
        transformed = actual.transform(callback, normals, 4, "RGBA")
        reference = expected.transform(callback, normals, 4, "RGBA")
        assert transformed.table == reference.table
        assert repr(transformed) == repr(reference)
        compare("RGB", (13, 9), transformed, reference)
    values = [0.0] * 24
    copied = ImageFilter.Color3DLUT(2, values)
    shared = ImageFilter.Color3DLUT(2, values, _copy_table=False)
    assert copied.table is not values
    assert shared.table is values
    values[:] = [1.0] * 24
    image, _ = images("RGB", (2, 2))
    assert image.filter(shared).tobytes() == b"\xff" * 12
    shared.table[:] = [0.0] * 24
    assert image.filter(shared).tobytes() == b"\0" * 12


@pytest.mark.parametrize("size,table,channels", [(1, [0] * 3, 3), (66, [], 3), ((2, 3), [], 3), (2, [], 2), (2, [0] * 23, 3), (2, [(0, 0)] * 8, 3), (2, np.zeros((3, 8)), 3)])
def test_invalid_lut(size: object, table: object, channels: int) -> None:
    with pytest.raises(Exception) as expected:
        PillowFilter.Color3DLUT(size, table, channels)
    with pytest.raises(type(expected.value)) as actual:
        ImageFilter.Color3DLUT(size, table, channels)
    assert str(actual.value) == str(expected.value)


@pytest.mark.parametrize("mode,channels,target", [("L", 3, None), ("RGB", 4, None), ("RGB", 3, "RGBA"), ("RGBA", 4, "RGB")])
def test_lut_incompatible_modes(mode: str, channels: int, target: str | None) -> None:
    image, pillow = images(mode, (5, 5))
    for source, module in ((image, ImageFilter), (pillow, PillowFilter)):
        with pytest.raises(ValueError):
            source.filter(module.Color3DLUT(2, [0] * (8 * channels), channels, target))


def test_custom_filters_and_errors() -> None:
    class CopyBand(ImageFilter.Filter):
        def filter(self, image: object) -> object:
            assert image.mode == "L"
            return image.copy()

    class CopyImage(ImageFilter.MultibandFilter):
        def filter(self, image: object) -> object:
            assert image.mode == "RGBA"
            return image.copy()

    image, _ = images("RGBA", (3, 4))
    for custom in (CopyBand, CopyBand(), CopyImage, CopyImage(), lambda: CopyBand()):
        assert image.filter(custom).tobytes() == image.tobytes()
    for invalid in (None, 1, "blur", object()):
        with pytest.raises(TypeError, match="filter argument should be ImageFilter.Filter instance or class"):
            image.filter(invalid)
    with pytest.raises(TypeError):
        ImageFilter.Filter()
    image.close()
    with pytest.raises(ValueError, match="closed"):
        image.filter(ImageFilter.BLUR)


def test_opened_image_metadata_and_filename() -> None:
    payload = BytesIO()
    PillowImage.new("RGB", (5, 5)).save(payload, "PNG")
    image = Image.open(payload)
    image.info["test"] = "value"
    result = image.filter(ImageFilter.BLUR)
    assert result.format is None
    assert result.filename == ""
    assert result.info == image.info and result.info is not image.info
