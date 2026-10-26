"""Per-band image statistics backed by native histogram and reduction kernels."""

from __future__ import annotations

from functools import cached_property

from . import Image
from ._blanket import stat_count, stat_extrema, stat_histogram, stat_median, stat_normalize, stat_sqrt, stat_sum, stat_variance

__all__ = ["Global", "Stat"]


class Stat:
    """Calculate statistics from an image or a list of histogram counts.

    Supported image modes include `1`, `L`, `LA`, `P`, `PA`, `RGB`, `RGBA`,
    `HSV`, `CMYK`, `YCbCr`, and `LAB`. A mask selects pixels with any nonzero
    L value. Histograms contain 256 nonnegative integer counts per band, each
    fitting in an unsigned 64-bit integer. Results are computed lazily and
    cached, as in Pillow.

    High-depth `L`/`RGB`/`RGBA` and `I;16` variants use full sample values,
    including alpha. Their `h` histogram has `2 ** bit_depth` bins per band.
    Empty extrema are `(maximum, 0)` and the empty median is `maximum`.
    Histogram-list input continues to use 256 bins per band.
    """

    def __init__(self, image_or_list: Image.Image | list[int], mask: Image.Image | None = None) -> None:
        """Configure Stat.

        Args:
            image_or_list: An image or a histogram with 256 nonnegative integer counts per band.
            mask: Optional 8-bit L mask; pixels with a zero mask value are excluded.

        Examples:
            ```python
            from blanket import Image, ImageStat

            image = Image.new("RGB", (8, 8), (40, 100, 180))
            statistics = ImageStat.Stat(image)
            print(statistics.mean)
            ```
        """
        self._bins = 256
        if isinstance(image_or_list, list):
            self.h = image_or_list
        elif isinstance(image_or_list, Image.Image):
            if image_or_list.bit_depth in (10, 12, 16):
                self._bins = 1 << image_or_list.bit_depth
                self.h = stat_histogram(image_or_list._native, mask._native if mask is not None else None)
            else:
                self.h = image_or_list.histogram(mask)
        else:
            raise TypeError("first argument must be image or list")
        length = len(self.h)
        if length % self._bins:
            raise ValueError("histogram must contain 256 bins per band")
        self._band_count = length // self._bins

    @cached_property
    def bands(self) -> list[int]:
        """Band indices, materialized only when requested.

        Examples:
            ```python
            from blanket import Image, ImageStat

            image = Image.new("RGB", (8, 8), (40, 100, 180))
            statistics = ImageStat.Stat(image)
            print(statistics.bands)
            ```
        """
        return list(range(self._band_count))

    @cached_property
    def extrema(self) -> list[tuple[int, int]]:
        """Lowest and highest occupied bin per band; (maximum, 0) if empty.

        Examples:
            ```python
            from blanket import Image, ImageStat

            image = Image.new("RGB", (8, 8), (40, 100, 180))
            statistics = ImageStat.Stat(image)
            print(statistics.extrema)
            ```
        """
        return stat_extrema(self.h, self._bins)

    @cached_property
    def count(self) -> list[int]:
        """Number of selected pixels in each band.

        Examples:
            ```python
            from blanket import Image, ImageStat

            image = Image.new("RGB", (8, 8), (40, 100, 180))
            statistics = ImageStat.Stat(image)
            print(statistics.count)
            ```
        """
        return stat_count(self.h, self._bins)

    @cached_property
    def sum(self) -> list[float]:
        """Sum of pixel values in each band.

        Examples:
            ```python
            from blanket import Image, ImageStat

            image = Image.new("RGB", (8, 8), (40, 100, 180))
            statistics = ImageStat.Stat(image)
            print(statistics.sum)
            ```
        """
        return stat_sum(self.h, False, self._bins)

    @cached_property
    def sum2(self) -> list[float]:
        """Sum of squared pixel values in each band.

        Examples:
            ```python
            from blanket import Image, ImageStat

            image = Image.new("RGB", (8, 8), (40, 100, 180))
            statistics = ImageStat.Stat(image)
            print(statistics.sum2)
            ```
        """
        return stat_sum(self.h, True, self._bins)

    @cached_property
    def mean(self) -> list[float]:
        """Arithmetic mean in each band, or zero for empty bands.

        Examples:
            ```python
            from blanket import Image, ImageStat

            image = Image.new("RGB", (8, 8), (40, 100, 180))
            statistics = ImageStat.Stat(image)
            print(statistics.mean)
            ```
        """
        return stat_normalize(self.sum, self.count)

    @cached_property
    def median(self) -> list[int]:
        """Upper median in each band, or the maximum sample value if empty.

        Examples:
            ```python
            from blanket import Image, ImageStat

            image = Image.new("RGB", (8, 8), (40, 100, 180))
            statistics = ImageStat.Stat(image)
            print(statistics.median)
            ```
        """
        return stat_median(self.h, self.count, self._bins)

    @cached_property
    def rms(self) -> list[float]:
        """Root mean square in each band, or zero for empty bands.

        Examples:
            ```python
            from blanket import Image, ImageStat

            image = Image.new("RGB", (8, 8), (40, 100, 180))
            statistics = ImageStat.Stat(image)
            print(statistics.rms)
            ```
        """
        return stat_sqrt(stat_normalize(self.sum2, self.count))

    @cached_property
    def var(self) -> list[float]:
        """Population variance in each band, or zero for empty bands.

        Examples:
            ```python
            from blanket import Image, ImageStat

            image = Image.new("RGB", (8, 8), (40, 100, 180))
            statistics = ImageStat.Stat(image)
            print(statistics.var)
            ```
        """
        if self._bins > 256:
            return stat_variance(self.h, self.count, self.sum, self._bins)
        return stat_normalize(self.sum2, self.count, self.sum)

    @cached_property
    def stddev(self) -> list[float]:
        """Population standard deviation in each band.

        Examples:
            ```python
            from blanket import Image, ImageStat

            image = Image.new("RGB", (8, 8), (40, 100, 180))
            statistics = ImageStat.Stat(image)
            print(statistics.stddev)
            ```
        """
        return stat_sqrt(self.var)


Global = Stat
