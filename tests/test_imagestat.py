"""Pillow 13.0.0 contracts and Blanket regressions; see PILLOW_TESTS.md."""

from __future__ import annotations

import pytest
from helpers import assert_raises, new_pair, patterned_pair
from PIL import Image as PILImage, ImageStat as PILStat
from pillow_helpers import hopper

from blanket import Image, ImageStat

# ruff: noqa: B018


def test_sanity() -> None:
    im = hopper()

    st = ImageStat.Stat(im)
    st = ImageStat.Stat(im.histogram())
    st = ImageStat.Stat(im, Image.new("1", im.size, 1))

    # Check these run. Exceptions will cause failures.
    st.extrema
    st.sum
    st.mean
    st.median
    st.rms
    st.sum2
    st.var
    st.stddev

    assert_raises(AttributeError, lambda: st.spam())  # type: ignore[attr-defined]

    assert_raises(TypeError, lambda: ImageStat.Stat(1))  # type: ignore[arg-type]


def test_hopper() -> None:
    im = hopper()

    st = ImageStat.Stat(im)

    # verify a few values
    assert st.extrema[0] == (0, 255)
    assert st.median[0] == 72
    assert st.sum[0] == 1470218
    assert st.sum[1] == 1311896
    assert st.sum[2] == 1563008


def test_constant() -> None:
    im = Image.new("L", (128, 128), 128)

    st = ImageStat.Stat(im)

    assert st.extrema[0] == (128, 128)
    assert st.sum[0] == 128**3
    assert st.sum2[0] == 128**4
    assert st.mean[0] == 128
    assert st.median[0] == 128
    assert st.rms[0] == 128
    assert st.var[0] == 0
    assert st.stddev[0] == 0


def test_zero_count() -> None:
    im = Image.new("L", (0, 0))

    st = ImageStat.Stat(im)

    assert st.mean == [0]
    assert st.rms == [0]
    assert st.var == [0]


def test_variance_rounding() -> None:
    im = Image.new("L", (919, 405), 255)
    st = ImageStat.Stat(im)
    assert st.var == [0.0]


STATISTICS = ("extrema", "count", "sum", "sum2", "mean", "median", "rms", "var", "stddev")


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
@pytest.mark.parametrize("fill", [0, 1])
def test_bilevel_masks_match_pillow(mode: str, fill: int) -> None:
    image, pillow = patterned_pair(mode, (19, 11))
    mask, reference = new_pair("1", image.size, fill)
    mask.putpixel((0, 0), 1 - fill)
    reference.putpixel((0, 0), 1 - fill)
    assert image.histogram(mask) == pillow.histogram(reference)
    same(ImageStat.Stat(image, mask), PILStat.Stat(pillow, reference))


def test_large_histogram_moments_remain_exact_integers() -> None:
    histogram = [0] * 256
    histogram[253] = (1 << 64) - 1
    stats = ImageStat.Stat(histogram)
    reference = PILStat.Stat(histogram)
    assert stats.sum == reference.sum
    assert stats.sum2 == reference.sum2
    assert stats.var == reference.var


@pytest.mark.parametrize("bit_depth", [10, 12, 16])
def test_high_depth_statistics_accept_bilevel_masks(bit_depth: int) -> None:
    image = Image.frombytes("L", (3, 1), b"\x00\x01\x00\x02\x00\x03", bit_depth=bit_depth)
    mask = Image.new("1", image.size, 0)
    mask.putpixel((1, 0), 1)
    stats = ImageStat.Stat(image, mask)
    assert stats.count == [1]
    assert stats.mean == [512]
    assert stats.median == [512]
    assert stats.var == [0.0]


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


def test_global_alias_and_nonzero_mask_is_not_a_weight() -> None:
    assert ImageStat.Global is ImageStat.Stat
    image = Image.frombytes("L", (4, 1), bytes([10, 20, 40, 250]))
    mask = Image.frombytes("L", image.size, bytes([1, 128, 255, 0]))
    stat = ImageStat.Global(image, mask)
    assert stat.count == [3]
    assert stat.sum == [70.0]
    assert stat.mean == [70 / 3]
    assert stat.extrema == [(10, 40)]


@pytest.mark.parametrize(("samples", "median"), [([10, 200], 200), ([10, 10, 200], 10), ([0, 0, 255, 255], 255), ([30], 30)])
def test_median_selects_upper_middle(samples: list[int], median: int) -> None:
    image = Image.frombytes("L", (len(samples), 1), bytes(samples))
    assert ImageStat.Stat(image).median == [median]


def test_histogram_snapshot_survives_source_changes_and_close() -> None:
    image, pillow = patterned_pair("RGBA", (19, 11))
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
    image, pillow = patterned_pair("L", (23, 19))
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


@pytest.mark.parametrize("value", [None, 1, "histogram", tuple([0] * 256)])
def test_invalid_input_type(value: object) -> None:
    assert_raises(TypeError, lambda: ImageStat.Stat(value), match="first argument must be image or list")


@pytest.mark.parametrize("length", [1, 255, 257])
def test_incomplete_histogram_is_rejected(length: int) -> None:
    assert_raises(ValueError, lambda: ImageStat.Stat([0] * length), match="256 bins per band")


@pytest.mark.parametrize(("value", "error"), [(-1, OverflowError), (2**64, OverflowError), (1.5, TypeError)])
def test_invalid_histogram_count(value: object, error: type[Exception]) -> None:
    stat = ImageStat.Stat([value] + [0] * 255)
    with pytest.raises(error):
        _ = stat.count


@pytest.mark.parametrize("property_name", ["count", "extrema"])
@pytest.mark.parametrize("position", [0, 127, 255, 256, 767])
@pytest.mark.parametrize(("value", "error"), [(-1, OverflowError), (2**64, OverflowError), (1.5, TypeError)])
def test_histogram_reductions_validate_every_bin(property_name: str, position: int, value: object, error: type[Exception]) -> None:
    histogram = [1] * 768
    histogram[position] = value
    assert_raises(error, lambda: getattr(ImageStat.Stat(histogram), property_name))


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


@pytest.mark.parametrize("property_name", ["count", "extrema"])
def test_histogram_conversion_handles_reentrant_list_mutation(property_name: str) -> None:
    class Count:
        def __index__(self) -> int:
            histogram.clear()
            return 7

    histogram = [Count()] + [0] * 255
    stat = ImageStat.Stat(histogram)
    assert_raises(IndexError, lambda: getattr(stat, property_name))
    assert histogram == []


def test_invalid_masks_and_closed_images() -> None:
    image = Image.new("RGBA", (3, 2))
    assert_raises(ValueError, lambda: ImageStat.Stat(image, Image.new("RGB", image.size)), match="mask")
    assert_raises(ValueError, lambda: ImageStat.Stat(image, Image.new("L", (1, 1))), match="images do not match")
    mask = Image.new("L", image.size)
    mask.close()
    assert_raises(ValueError, lambda: ImageStat.Stat(image, mask), match="closed image")
    image.close()
    assert_raises(ValueError, lambda: ImageStat.Stat(image), match="closed image")


@pytest.mark.parametrize("bit_depth", [10, 12, 16])
def test_high_depth_retains_sample_values(bit_depth: int) -> None:
    image = Image.frombytes("L", (1, 1), b"\x00\x01", bit_depth=bit_depth)
    assert ImageStat.Stat(image).mean == [256.0]


def test_native_reductions_validate_band_lengths() -> None:
    from blanket._blanket import stat_count, stat_median, stat_normalize, stat_sqrt

    assert_raises(ValueError, lambda: stat_count([1]), match="256 bins per band")
    assert_raises(ValueError, lambda: stat_median([0] * 256, []), match="band counts")
    assert_raises(ValueError, lambda: stat_normalize([1.0], []), match="band counts")
    assert_raises(ValueError, lambda: stat_normalize([1.0], [1], []), match="band counts")
    assert_raises(ValueError, lambda: stat_sqrt([-1.0]), match="math domain error")
