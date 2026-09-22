"""Per-band image statistics backed by native histogram and reduction kernels."""

from __future__ import annotations

from functools import cached_property

from . import Image
from ._blanket import stat_count, stat_extrema, stat_median, stat_normalize, stat_sqrt, stat_sum

__all__ = ["Global", "Stat"]


class Stat:
    """Calculate statistics from an 8-bit image or a list of histogram counts.

    A mask selects pixels with any nonzero L value. Histograms contain 256
    nonnegative integer counts per band, each fitting in an unsigned 64-bit
    integer. Results are computed lazily and cached, as in Pillow.
    """

    def __init__(self, image_or_list: Image.Image | list[int], mask: Image.Image | None = None) -> None:
        if isinstance(image_or_list, list):
            self.h = image_or_list
        elif isinstance(image_or_list, Image.Image):
            self.h = image_or_list.histogram(mask)
        else:
            raise TypeError("first argument must be image or list")
        length = len(self.h)
        if length % 256:
            raise ValueError("histogram must contain 256 bins per band")
        self._band_count = length // 256

    @cached_property
    def bands(self) -> list[int]:
        """Band indices, materialized only when requested."""
        return list(range(self._band_count))

    @cached_property
    def extrema(self) -> list[tuple[int, int]]:
        """Lowest and highest occupied bin per band; (255, 0) if empty."""
        return stat_extrema(self.h)

    @cached_property
    def count(self) -> list[int]:
        """Number of selected pixels in each band."""
        return stat_count(self.h)

    @cached_property
    def sum(self) -> list[float]:
        """Sum of pixel values in each band."""
        return stat_sum(self.h, False)

    @cached_property
    def sum2(self) -> list[float]:
        """Sum of squared pixel values in each band."""
        return stat_sum(self.h, True)

    @cached_property
    def mean(self) -> list[float]:
        """Arithmetic mean in each band, or zero for empty bands."""
        return stat_normalize(self.sum, self.count)

    @cached_property
    def median(self) -> list[int]:
        """Upper median in each band, or 255 for empty bands."""
        return stat_median(self.h, self.count)

    @cached_property
    def rms(self) -> list[float]:
        """Root mean square in each band, or zero for empty bands."""
        return stat_sqrt(stat_normalize(self.sum2, self.count))

    @cached_property
    def var(self) -> list[float]:
        """Population variance in each band, or zero for empty bands."""
        return stat_normalize(self.sum2, self.count, self.sum)

    @cached_property
    def stddev(self) -> list[float]:
        """Population standard deviation in each band."""
        return stat_sqrt(self.var)


Global = Stat
