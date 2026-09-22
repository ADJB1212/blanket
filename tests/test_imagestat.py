from __future__ import annotations

import pytest
from blanket import Image, ImageStat
from PIL import Image as PILImage, ImageStat as PILStat

STATISTICS = ("extrema", "count", "sum", "sum2", "mean", "median", "rms", "var", "stddev")


def pair(mode: str, size: tuple[int, int]) -> tuple[Image.Image, PILImage.Image]:
    raw = bytes((i * 37 + i // 11) % 256 for i in range(size[0] * size[1] * len(mode)))
    return Image.frombytes(mode, size, raw), PILImage.frombytes(mode, size, raw)


def same(actual: ImageStat.Stat, expected: PILStat.Stat) -> None:
    assert actual.h == expected.h
    assert actual.bands == expected.bands
    for name in STATISTICS:
        result = getattr(actual, name)
        reference = getattr(expected, name)
        assert isinstance(result, list)
        if name in ("extrema", "count", "median"):
            assert result == reference, name
        else:
            assert result == pytest.approx(reference, rel=1e-14, abs=1e-12), name
        assert getattr(actual, name) is result


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
@pytest.mark.parametrize("size", [(0, 0), (0, 3), (3, 0), (1, 1), (19, 11), (513, 529)])
@pytest.mark.parametrize("mask_kind", ["none", "empty", "full", "partial"])
def test_image_statistics_match_pillow(mode: str, size: tuple[int, int], mask_kind: str) -> None:
    image, pillow = pair(mode, size)
    mask = pil_mask = None
    if mask_kind != "none":
        data = bytes(0 if mask_kind == "empty" else 255 if mask_kind == "full" else (0, 1, 128, 255)[i % 4] for i in range(size[0] * size[1]))
        mask = Image.frombytes("L", size, data)
        pil_mask = PILImage.frombytes("L", size, data)
    same(ImageStat.Stat(image, mask), PILStat.Stat(pillow, pil_mask))


@pytest.mark.parametrize("channels", [0, 1, 2, 3, 4, 7])
def test_precomputed_histograms(channels: int) -> None:
    histogram = [(i * 37 + i // 13) % 67 for i in range(channels * 256)]
    actual = ImageStat.Stat(histogram)
    assert actual.h is histogram
    same(actual, PILStat.Stat(histogram))


def test_global_alias_and_nonzero_mask_is_not_a_weight() -> None:
    assert ImageStat.Global is ImageStat.Stat
    image = Image.frombytes("L", (4, 1), bytes([10, 20, 40, 250]))
    mask = Image.frombytes("L", image.size, bytes([1, 128, 255, 0]))
    stat = ImageStat.Global(image, mask)
    assert stat.count == [3]
    assert stat.sum == [70.0]
    assert stat.mean == [70 / 3]
    assert stat.extrema == [(10, 40)]


@pytest.mark.parametrize("samples,median", [([10, 200], 200), ([10, 10, 200], 10), ([0, 0, 255, 255], 255), ([30], 30)])
def test_median_selects_upper_middle(samples: list[int], median: int) -> None:
    image = Image.frombytes("L", (len(samples), 1), bytes(samples))
    assert ImageStat.Stat(image).median == [median]


def test_empty_histogram_and_mask_results() -> None:
    stat = ImageStat.Stat([0] * 256)
    assert stat.extrema == [(255, 0)]
    assert stat.count == [0]
    assert stat.median == [255]
    for name in ("sum", "sum2", "mean", "rms", "var", "stddev"):
        assert getattr(stat, name) == [0]


def test_histogram_snapshot_survives_source_changes_and_close() -> None:
    image, pillow = pair("RGBA", (19, 11))
    stat, reference = ImageStat.Stat(image), PILStat.Stat(pillow)
    image.paste((10, 20, 30, 40), (0, 0, *image.size))
    image.close()
    pillow.close()
    same(stat, reference)


def test_lazy_properties_follow_histogram_mutation_and_cache_independently() -> None:
    histogram = [0] * 256
    histogram[10] = 2
    actual, expected = ImageStat.Stat(histogram), PILStat.Stat(histogram)
    assert actual.sum == expected.sum == [20]
    assert "count" not in actual.__dict__
    histogram[20] = 2
    assert actual.mean == expected.mean == [5]
    same(actual, expected)
    del actual.sum
    del expected.sum
    assert actual.sum == expected.sum == [60]


def test_list_input_ignores_mask() -> None:
    histogram = [1] * 256
    mask = Image.new("RGB", (1, 1))
    mask.close()
    same(ImageStat.Stat(histogram, mask), PILStat.Stat(histogram))


def test_palette_statistics_use_indices() -> None:
    image, pillow = pair("L", (23, 19))
    palette = bytes(range(256)) * 3
    image.putpalette(palette)
    pillow.putpalette(palette)
    same(ImageStat.Stat(image), PILStat.Stat(pillow))
    assert len(ImageStat.Stat(image).mean) == 1


@pytest.mark.parametrize("frequency", [2**32, 2**53 + 1, 2**64 - 1])
def test_large_histogram_counts_do_not_overflow(frequency: int) -> None:
    histogram = [0] * 256
    histogram[0] = frequency
    histogram[127] = frequency
    histogram[255] = frequency
    stat = ImageStat.Stat(histogram)
    assert stat.count == [frequency * 3]
    same(stat, PILStat.Stat(histogram))


@pytest.mark.parametrize("value", [0, 17, 127, 255])
def test_constant_images_have_zero_variance(value: int) -> None:
    stat = ImageStat.Stat(Image.new("RGB", (29, 31), (value,) * 3))
    assert stat.mean == [value] * 3
    assert stat.rms == [value] * 3
    assert stat.var == [0] * 3
    assert stat.stddev == [0] * 3


@pytest.mark.parametrize("value", [None, 1, "histogram", tuple([0] * 256)])
def test_invalid_input_type(value: object) -> None:
    with pytest.raises(TypeError, match="first argument must be image or list"):
        ImageStat.Stat(value)


@pytest.mark.parametrize("length", [1, 255, 257])
def test_incomplete_histogram_is_rejected(length: int) -> None:
    with pytest.raises(ValueError, match="256 bins per band"):
        ImageStat.Stat([0] * length)


@pytest.mark.parametrize("value,error", [(-1, OverflowError), (2**64, OverflowError), (1.5, TypeError)])
def test_invalid_histogram_count(value: object, error: type[Exception]) -> None:
    stat = ImageStat.Stat([value] + [0] * 255)
    with pytest.raises(error):
        _ = stat.count


@pytest.mark.parametrize("property_name", ["count", "extrema"])
@pytest.mark.parametrize("position", [0, 127, 255, 256, 767])
@pytest.mark.parametrize("value,error", [(-1, OverflowError), (2**64, OverflowError), (1.5, TypeError)])
def test_histogram_reductions_validate_every_bin(property_name: str, position: int, value: object, error: type[Exception]) -> None:
    histogram = [1] * 768
    histogram[position] = value
    with pytest.raises(error):
        getattr(ImageStat.Stat(histogram), property_name)


def test_histogram_reductions_accept_integer_protocol_and_list_subclasses() -> None:
    class Count:
        def __index__(self) -> int:
            return 7

    histogram = [0] * 256
    histogram[37] = Count()
    stat = ImageStat.Stat(histogram)
    assert stat.count == [7]
    assert stat.extrema == [(37, 37)]

    class Histogram(list):
        def __iter__(self):
            return iter(histogram)

    stat = ImageStat.Stat(Histogram([0] * 256))
    assert stat.count == [7]
    assert stat.extrema == [(37, 37)]


def test_lazy_band_indices_preserve_construction_count() -> None:
    histogram = [0] * 768
    stat = ImageStat.Stat(histogram)
    histogram.extend([0] * 256)
    assert stat.bands == [0, 1, 2]
    stat.bands.append(7)
    assert stat.bands == [0, 1, 2, 7]


def test_invalid_masks_and_closed_images() -> None:
    image = Image.new("RGBA", (3, 2))
    with pytest.raises(ValueError, match="mask"):
        ImageStat.Stat(image, Image.new("RGB", image.size))
    with pytest.raises(ValueError, match="images do not match"):
        ImageStat.Stat(image, Image.new("L", (1, 1)))
    mask = Image.new("L", image.size)
    mask.close()
    with pytest.raises(ValueError, match="closed image"):
        ImageStat.Stat(image, mask)
    image.close()
    with pytest.raises(ValueError, match="closed image"):
        ImageStat.Stat(image)


@pytest.mark.parametrize("bit_depth", [10, 12, 16])
def test_high_depth_requires_explicit_conversion(bit_depth: int) -> None:
    image = Image.frombytes("L", (1, 1), b"\x00\x01", bit_depth=bit_depth)
    with pytest.raises(ValueError, match="8-bit"):
        ImageStat.Stat(image)


def test_native_reductions_validate_band_lengths() -> None:
    from blanket._blanket import stat_count, stat_median, stat_normalize, stat_sqrt

    with pytest.raises(ValueError, match="256 bins per band"):
        stat_count([1])
    with pytest.raises(ValueError, match="band counts"):
        stat_median([0] * 256, [])
    with pytest.raises(ValueError, match="band counts"):
        stat_normalize([1.0], [])
    with pytest.raises(ValueError, match="band counts"):
        stat_normalize([1.0], [1], [])
    with pytest.raises(ValueError, match="math domain error"):
        stat_sqrt([-1.0])
