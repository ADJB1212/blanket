from __future__ import annotations

import struct

import pytest
from PIL import Image as PILImage, ImageChops as PILChops, ImageFilter as PILFilter, ImageOps as PILOps

from blanket import Image, ImageChops, ImageFilter, ImageOps


@pytest.mark.parametrize("width", [1, 2, 4, 15, 16, 17, 18, 19, 20, 21, 31, 32, 33, 34])
def test_integer_convolution_matches_pillow_with_unaligned_source(width: int, subtests: pytest.Subtests) -> None:
    for kernel_size in [3, 5]:
        for scale, offset in [(1.0, 0.0), (2.75, -1.5), (-3.0, 128.0)]:
            with subtests.test(width=width, kernel_size=kernel_size, scale=scale, offset=offset):
                size = width, 7
                values = [-2147483648, 2147483647, -1000, 16777217, 0, 127, 255]
                raw = struct.pack(f"<{width * 7}i", *(values[i % len(values)] for i in range(width * 7)))
                storage = bytearray(b"x" + raw + b"guard")
                actual = Image.frombytes("I", size, memoryview(storage)[1 : 1 + len(raw)])
                expected = PILImage.frombytes("I", size, raw)
                coefficients = [float(i % 5 - 2) for i in range(kernel_size * kernel_size)]
                result = actual.filter(ImageFilter.Kernel((kernel_size, kernel_size), coefficients, scale, offset))
                reference = expected.filter(PILFilter.Kernel((kernel_size, kernel_size), coefficients, scale, offset))
                assert result.mode == "I"
                assert result.tobytes() == reference.tobytes()
                assert storage == b"x" + raw + b"guard"
                assert actual.tobytes() == raw


@pytest.mark.parametrize("mode", ["I", "F"])
def test_wide_rank_filters_match_pillow(mode: str, subtests: pytest.Subtests) -> None:
    for size in [(0, 1), (1, 0), (1, 1), (2, 7), (19, 5)]:
        for window, rank in [(3, 0), (3, 4), (3, 8), (5, 1), (5, 12), (5, 24)]:
            with subtests.test(mode=mode, size=size, window=window, rank=rank):
                values = [-2147483648, 2147483647, -1, 0, 255, 256, 1000] if mode == "I" else [float("nan"), float("inf"), -float("inf"), -0.0, 0.0, 1.25, -3.5]
                count = size[0] * size[1]
                raw = struct.pack(f"<{count}{'i' if mode == 'I' else 'f'}", *(values[i % len(values)] for i in range(count)))
                actual = Image.frombytes(mode, size, raw)
                expected = PILImage.frombytes(mode, size, raw)
                result = actual.filter(ImageFilter.RankFilter(window, rank))
                reference = expected.filter(PILFilter.RankFilter(window, rank))
                assert result.mode == mode
                assert result.size == size
                assert result.tobytes() == reference.tobytes()
                assert actual.tobytes() == raw


@pytest.mark.parametrize("mode", ["I", "F"])
def test_wide_rank_one_is_an_independent_copy(mode: str) -> None:
    raw = struct.pack("<3i", -3, 0, 2147483647) if mode == "I" else struct.pack("<3f", float("nan"), -0.0, float("inf"))
    image = Image.frombytes(mode, (3, 1), raw)
    result = image.filter(ImageFilter.RankFilter(1, 0))
    assert result.tobytes() == raw
    result.putpixel((0, 0), 3)
    assert image.tobytes() == raw


@pytest.mark.parametrize("radius", [float("nan"), float("inf"), -float("inf"), 2**31, pytest.param(2**10000, id="huge_integer"), -1])
@pytest.mark.parametrize("shape", ["scalar", "horizontal", "vertical"])
def test_box_blur_checks_constructor_and_mutated_radii(radius: float, shape: str) -> None:
    value = radius if shape == "scalar" else (radius, 1) if shape == "horizontal" else (1, radius)
    with pytest.raises(ValueError, match="radius"):
        ImageFilter.BoxBlur(value)
    operation = ImageFilter.BoxBlur(1)
    operation.radius = value
    with pytest.raises(ValueError, match="radius"):
        Image.new("RGB", (0, 1)).filter(operation)


@pytest.mark.parametrize("radius", [float("nan"), float("inf"), -float("inf"), 2**31])
@pytest.mark.parametrize("size", [(0, 1), (3, 2)])
def test_gaussian_and_unsharp_reject_invalid_radii_on_empty_images(radius: float, size: tuple[int, int]) -> None:
    image = Image.new("RGB", size)
    for operation in (ImageFilter.GaussianBlur(radius), ImageFilter.UnsharpMask(radius)):
        with pytest.raises(ValueError, match="radius"):
            image.filter(operation)


def test_negative_gaussian_radius_matches_pillow() -> None:
    raw = bytes(range(35))
    image = Image.frombytes("L", (7, 5), raw)
    reference = PILImage.frombytes("L", image.size, raw)
    assert image.filter(ImageFilter.GaussianBlur(-1)).tobytes() == reference.filter(PILFilter.GaussianBlur(-1)).tobytes()


@pytest.mark.parametrize("size", [23171, 2**31, 2**32 - 1, 2**100])
def test_rank_filter_revalidates_mutated_size(size: int) -> None:
    operation = ImageFilter.RankFilter(3, 0)
    operation.size = size
    with pytest.raises(ValueError, match="filter size too large"):
        Image.new("L", (2, 2)).filter(operation)


@pytest.mark.parametrize("operation", ["invert", "posterize", "solarize"])
def test_palette_lookup_tables_keep_indices_alpha_and_metadata(operation: str) -> None:
    image = Image.frombytes("P", (3, 1), b"\x00\x01\x00")
    expected = PILImage.frombytes("P", image.size, image.tobytes())
    palette = bytes([10, 20, 30, 0, 40, 50, 60, 128])
    image.putpalette(palette, "RGBA")
    expected.putpalette(palette, "RGBA")
    image.info["metadata"] = ["original"]
    kwargs = {"bits": 4} if operation == "posterize" else {}
    result = getattr(ImageOps, operation)(image, **kwargs)
    reference = getattr(PILOps, operation)(expected, **kwargs)
    assert result.tobytes() == image.tobytes()
    assert result.getpalette("RGBA") == reference.getpalette("RGBA")
    result.info["metadata"].append("changed")
    assert image.info["metadata"] == ["original"]
    assert image.getpalette("RGBA") == list(palette)


@pytest.mark.parametrize("transparency", [0, bytes([0, 128])])
@pytest.mark.parametrize("palette_mode", ["RGB", "RGBA"])
def test_palette_transparency_survives_expand_and_pad(transparency: int | bytes, palette_mode: str) -> None:
    actual = Image.frombytes("P", (2, 1), b"\x00\x01")
    expected = PILImage.frombytes("P", actual.size, actual.tobytes())
    palette = bytes([10, 20, 30, 40, 50, 60]) if palette_mode == "RGB" else bytes([10, 20, 30, 64, 40, 50, 60, 192])
    for image in (actual, expected):
        image.putpalette(palette, palette_mode)
        image.info["transparency"] = transparency
    for target in ("LA", "RGBA"):
        assert actual.convert(target).tobytes() == expected.convert(target).tobytes()
    for name, args, options in (("expand", (1,), {"fill": "blue"}), ("pad", ((4, 3),), {"color": "blue"})):
        result = getattr(ImageOps, name)(actual, *args, **options)
        reference = getattr(PILOps, name)(expected, *args, **options)
        assert result.convert("RGBA").tobytes() == reference.convert("RGBA").tobytes()
    assert actual.getpalette(palette_mode) == list(palette)


def test_kernel_default_scale_and_empty_coefficients_match_pillow() -> None:
    coefficients = [1e16, 1.0, -1e16, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    actual = ImageFilter.Kernel((3, 3), coefficients)
    expected = PILFilter.Kernel((3, 3), coefficients)
    assert actual.filterargs == expected.filterargs
    image = Image.frombytes("L", (5, 5), bytes(range(25)))
    reference = PILImage.frombytes("L", image.size, image.tobytes())
    assert image.filter(actual).tobytes() == reference.filter(expected).tobytes()
    for module in (ImageFilter, PILFilter):
        with pytest.raises(ValueError, match="not enough coefficients"):
            module.Kernel((3, 3), [])


@pytest.mark.parametrize("mode", ["1", "L", "P", "I", "F", "I;16", "I;16L", "I;16B"])
def test_single_band_fills_split_and_merge_match_pillow(mode: str) -> None:
    actual = Image.new(mode, (7, 5), (37,))
    expected = PILImage.new(mode, actual.size, (37,))
    actual.paste((71,), (1, 1, 5, 4))
    expected.paste((71,), (1, 1, 5, 4))
    assert actual.tobytes() == expected.tobytes()
    (band,) = actual.split()
    assert band.mode == mode
    merged = Image.merge(mode, (band,))
    assert merged.tobytes() == actual.tobytes()
    merged.putpixel((0, 0), 0)
    assert actual.getpixel((0, 0)) != 0


@pytest.mark.parametrize("name", ["logical_and", "logical_or", "logical_xor"])
def test_logical_operations_normalize_noncanonical_true_values(name: str) -> None:
    actual = Image.new("1", (33, 1))
    other = Image.new("1", actual.size)
    expected = PILImage.new("1", actual.size)
    reference = PILImage.new("1", actual.size)
    for i in range(33):
        for image, value in ((actual, (0, 1, 2, 128, 255)[i % 5]), (expected, (0, 1, 2, 128, 255)[i % 5]), (other, (0, 2, 1)[i % 3]), (reference, (0, 2, 1)[i % 3])):
            image.putpixel((i, 0), value)
    assert getattr(ImageChops, name)(actual, other).getdata() == list(getattr(PILChops, name)(expected, reference).get_flattened_data())


@pytest.mark.parametrize("mode", ["L", "F"])
def test_linear_gradient_matches_pillow(mode: str) -> None:
    actual = Image.linear_gradient(mode)
    expected = PILImage.linear_gradient(mode)
    assert actual.size == expected.size
    assert actual.mode == mode
    assert actual.tobytes() == expected.tobytes()


def test_image_equality_compares_pixels_metadata_and_palette() -> None:
    image = Image.new("P", (3, 2), 1)
    image.putpalette(bytes(range(30)))
    other = image.copy()
    assert image == other
    assert image != object()
    other.info["key"] = "value"
    assert image != other
    other.info.clear()
    other.putpalette(bytes(reversed(range(30))))
    assert image != other
    other = image.copy()
    other.putpixel((0, 0), 0)
    assert image != other


def test_zero_geometry_and_matching_ratio_fit_match_pillow() -> None:
    for source_size, target in (((10, 0), (5, 5)), ((5, 5), (2, 0))):
        actual = ImageOps.cover(Image.new("1", source_size), target)
        expected = PILOps.cover(PILImage.new("1", source_size), target)
        assert actual.size == expected.size
    image = Image.new("RGB", (1000, 755))
    assert ImageOps.fit(image, image.size).tobytes() == image.tobytes()


@pytest.mark.parametrize("mode,values", [("I", [-3, -2, 0, 4, 100]), ("F", [-3.5, -2.25, 0.0, 4.5, 100.25])])
def test_wide_histograms_match_pillow(mode: str, values: list[int | float]) -> None:
    raw = struct.pack(f"<5{'i' if mode == 'I' else 'f'}", *values)
    actual = Image.frombytes(mode, (5, 1), raw)
    expected = PILImage.frombytes(mode, actual.size, raw)
    assert actual.histogram() == expected.histogram()
    assert actual.histogram(extrema=(-5, 105)) == expected.histogram(extrema=(-5, 105))
    with pytest.raises(ValueError, match="image has wrong mode"):
        actual.histogram(Image.new("1", actual.size))


@pytest.mark.parametrize("maximum", [291, 301, 347, 1023, 65535])
@pytest.mark.parametrize("mode", ["I", "F"])
def test_wide_histogram_float32_bin_boundaries(mode: str, maximum: int) -> None:
    values = [i * maximum / 4096 for i in range(4097)]
    if mode == "I":
        values = [int(value) for value in values]
    raw = struct.pack(f"<4097{'i' if mode == 'I' else 'f'}", *values)
    image = Image.frombytes(mode, (4097, 1), raw)
    reference = PILImage.frombytes(mode, image.size, raw)
    assert image.histogram() == reference.histogram()


@pytest.mark.parametrize("mode", ["L", "LA", "RGB", "RGBA", "I", "F", "CMYK", "HSV", "YCbCr"])
def test_bilevel_conversion_normalizes_noncanonical_true_values(mode: str) -> None:
    actual = Image.new("1", (33, 1))
    expected = PILImage.new("1", actual.size)
    for i in range(33):
        value = (0, 1, 2, 128, 255)[i % 5]
        actual.putpixel((i, 0), value)
        expected.putpixel((i, 0), value)
    assert actual.convert(mode).tobytes() == expected.convert(mode).tobytes()
