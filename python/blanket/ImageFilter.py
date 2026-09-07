"""Pillow-compatible filters backed by native Rust pixel kernels."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from functools import reduce
from operator import add
from typing import Any

from ._blanket import _Image, filter_blur, filter_kernel, filter_lut, filter_mode, filter_rank, filter_unsharp


class Filter(ABC):
    @abstractmethod
    def filter(self, image: _Image) -> _Image:
        """Filter a native image band and return a native image."""
        raise NotImplementedError


class MultibandFilter(Filter):
    """A filter receiving all image bands together."""


class BuiltinFilter(MultibandFilter):
    name: str
    filterargs: tuple[tuple[int, int], float, float, Sequence[float]]

    def filter(self, image: _Image) -> _Image:
        return filter_kernel(image, *self.filterargs)


class Kernel(BuiltinFilter):
    """A 3x3 or 5x5 convolution, with vertically flipped kernel rows."""

    name = "Kernel"

    def __init__(self, size: tuple[int, int], kernel: Sequence[float], scale: float | None = None, offset: float = 0) -> None:
        if scale is None:
            scale = reduce(add, kernel)
        if size[0] * size[1] != len(kernel):
            raise ValueError("not enough coefficients in kernel")
        self.filterargs = size, scale, offset, kernel


class RankFilter(Filter):
    """Select the zero-based rank in an odd-sized, edge-extended window."""

    name = "Rank"

    def __init__(self, size: int, rank: int) -> None:
        if size % 2 == 0:
            raise ValueError("bad filter size")
        if size * size * 4 > 2**31 - 1:
            raise ValueError("filter size too large")
        if rank < 0 or rank >= size * size:
            raise ValueError("bad rank value")
        self.size = size
        self.rank = rank

    def filter(self, image: _Image) -> _Image:
        if self.size < 0:
            raise ValueError("bad kernel size")
        return filter_rank(image, self.size, self.rank)


class MedianFilter(RankFilter):
    name = "Median"

    def __init__(self, size: int = 3) -> None:
        super().__init__(size, size * size // 2)


class MinFilter(RankFilter):
    name = "Min"

    def __init__(self, size: int = 3) -> None:
        super().__init__(size, 0)


class MaxFilter(RankFilter):
    name = "Max"

    def __init__(self, size: int = 3) -> None:
        super().__init__(size, size * size - 1)


class ModeFilter(Filter):
    """Use the most frequent nearby value when it appears at least three times."""

    name = "Mode"

    def __init__(self, size: int = 3) -> None:
        self.size = size

    def filter(self, image: _Image) -> _Image:
        return filter_mode(image, self.size)


def _radii(radius: float | Sequence[float]) -> tuple[float, float]:
    if isinstance(radius, (int, float)):
        return radius, radius
    if len(radius) != 2:
        raise TypeError("argument 1 must be sequence of length 2, not " + str(len(radius)))
    return radius[0], radius[1]


class GaussianBlur(MultibandFilter):
    """Approximate Gaussian blur with three extended box passes per axis."""

    name = "GaussianBlur"

    def __init__(self, radius: float | Sequence[float] = 2) -> None:
        self.radius = radius

    def filter(self, image: _Image) -> _Image:
        return filter_blur(image, _radii(self.radius), True)


class BoxBlur(MultibandFilter):
    """Blur with a fractional box radius, in linear time per pass."""

    name = "BoxBlur"

    def __init__(self, radius: float | Sequence[float]) -> None:
        xy: Any = radius if isinstance(radius, (tuple, list)) else (radius, radius)
        if xy[0] < 0 or xy[1] < 0:
            raise ValueError("radius must be >= 0")
        self.radius = radius

    def filter(self, image: _Image) -> _Image:
        return filter_blur(image, _radii(self.radius), False)


class UnsharpMask(MultibandFilter):
    name = "UnsharpMask"

    def __init__(self, radius: float = 2, percent: int = 150, threshold: int = 3) -> None:
        self.radius = radius
        self.percent = percent
        self.threshold = threshold

    def filter(self, image: _Image) -> _Image:
        return filter_unsharp(image, self.radius, self.percent, self.threshold)


class BLUR(BuiltinFilter):
    name = "Blur"
    filterargs = (5, 5), 16, 0, (1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1)


class CONTOUR(BuiltinFilter):
    name = "Contour"
    filterargs = (3, 3), 1, 255, (-1, -1, -1, -1, 8, -1, -1, -1, -1)


class DETAIL(BuiltinFilter):
    name = "Detail"
    filterargs = (3, 3), 6, 0, (0, -1, 0, -1, 10, -1, 0, -1, 0)


class EDGE_ENHANCE(BuiltinFilter):
    name = "Edge-enhance"
    filterargs = (3, 3), 2, 0, (-1, -1, -1, -1, 10, -1, -1, -1, -1)


class EDGE_ENHANCE_MORE(BuiltinFilter):
    name = "Edge-enhance More"
    filterargs = (3, 3), 1, 0, (-1, -1, -1, -1, 9, -1, -1, -1, -1)


class EMBOSS(BuiltinFilter):
    name = "Emboss"
    filterargs = (3, 3), 1, 128, (-1, 0, 0, 0, 1, 0, 0, 0, 0)


class FIND_EDGES(BuiltinFilter):
    name = "Find Edges"
    filterargs = (3, 3), 1, 0, (-1, -1, -1, -1, 8, -1, -1, -1, -1)


class SHARPEN(BuiltinFilter):
    name = "Sharpen"
    filterargs = (3, 3), 16, 0, (-2, -2, -2, -2, 32, -2, -2, -2, -2)


class SMOOTH(BuiltinFilter):
    name = "Smooth"
    filterargs = (3, 3), 13, 0, (1, 1, 1, 1, 5, 1, 1, 1, 1)


class SMOOTH_MORE(BuiltinFilter):
    name = "Smooth More"
    filterargs = (5, 5), 100, 0, (1, 1, 1, 1, 1, 1, 5, 5, 5, 1, 1, 5, 44, 5, 1, 1, 5, 5, 5, 1, 1, 1, 1, 1, 1)


class Color3DLUT(MultibandFilter):
    """A mutable three-dimensional color table with native trilinear interpolation."""

    name = "Color 3D LUT"

    def __init__(self, size: int | tuple[int, int, int], table: Sequence[float] | Sequence[Sequence[float]] | Any, channels: int = 3, target_mode: str | None = None, **kwargs: bool) -> None:
        if channels not in (3, 4):
            raise ValueError("Only 3 or 4 output channels are supported")
        self.size = self._check_size(size)
        self.channels = channels
        self.mode = target_mode
        items = self.size[0] * self.size[1] * self.size[2]
        copy_table = kwargs.get("_copy_table", True)
        wrong_size = False
        # Array adaptation does not import or require NumPy at runtime.
        if hasattr(table, "shape") and hasattr(table, "reshape"):
            if copy_table:
                table = table.copy()
            if table.shape in ((items * channels,), (items, channels), (*reversed(self.size), channels)):
                table = table.reshape(items * channels)
            else:
                wrong_size = True
        else:
            if copy_table:
                table = list(table)
            if table and isinstance(table[0], (list, tuple)):
                flattened = []
                for pixel in table:
                    if len(pixel) != channels:
                        raise ValueError(f"The elements of the table should have a length of {channels}.")
                    flattened.extend(pixel)
                table = flattened
        if wrong_size or len(table) != items * channels:
            raise ValueError(
                "The table should have either channels * size**3 float items or size**3 items of channels-sized tuples with floats. "
                f"Table should be: {channels}x{self.size[0]}x{self.size[1]}x{self.size[2]}. Actual length: {len(table)}"
            )
        self.table = table

    @staticmethod
    def _check_size(size: Any) -> tuple[int, int, int]:
        try:
            _, _, _ = size
        except ValueError as error:
            raise ValueError("Size should be either an integer or a tuple of three integers.") from error
        except TypeError:
            size = (size, size, size)
        dimensions = tuple(int(value) for value in size)
        if any(not 2 <= value <= 65 for value in dimensions):
            raise ValueError("Size should be in [2, 65] range.")
        return dimensions

    @classmethod
    def generate(cls, size: int | tuple[int, int, int], callback: Callable[[float, float, float], tuple[float, ...]], channels: int = 3, target_mode: str | None = None) -> Color3DLUT:
        dimensions = cls._check_size(size)
        if channels not in (3, 4):
            raise ValueError("Only 3 or 4 output channels are supported")
        table = [0.0] * (dimensions[0] * dimensions[1] * dimensions[2] * channels)
        offset = 0
        for b in range(dimensions[2]):
            for g in range(dimensions[1]):
                for r in range(dimensions[0]):
                    table[offset : offset + channels] = callback(r / (dimensions[0] - 1), g / (dimensions[1] - 1), b / (dimensions[2] - 1))
                    offset += channels
        return cls(dimensions, table, channels, target_mode, _copy_table=False)

    def transform(self, callback: Callable[..., tuple[float, ...]], with_normals: bool = False, channels: int | None = None, target_mode: str | None = None) -> Color3DLUT:
        if channels not in (None, 3, 4):
            raise ValueError("Only 3 or 4 output channels are supported")
        channels = channels or self.channels
        table = [0.0] * (self.size[0] * self.size[1] * self.size[2] * channels)
        source = destination = 0
        for b in range(self.size[2]):
            for g in range(self.size[1]):
                for r in range(self.size[0]):
                    values = self.table[source : source + self.channels]
                    normals = (r / (self.size[0] - 1), g / (self.size[1] - 1), b / (self.size[2] - 1)) if with_normals else ()
                    table[destination : destination + channels] = callback(*normals, *values)
                    source += self.channels
                    destination += channels
        return type(self)(self.size, table, channels, target_mode or self.mode, _copy_table=False)

    def __repr__(self) -> str:
        text = f"{type(self).__name__} from {type(self.table).__name__} size={self.size[0]}x{self.size[1]}x{self.size[2]} channels={self.channels}"
        if self.mode:
            text += f" target_mode={self.mode}"
        return f"<{text}>"

    def filter(self, image: _Image) -> _Image:
        return filter_lut(image, self.mode or image.mode, self.channels, self.size, self.table)
