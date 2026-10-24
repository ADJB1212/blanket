"""Mutable two-dimensional paths for coordinate manipulation and drawing."""

from __future__ import annotations

__lazy_modules__ = {"struct", "math", "collections.abc"}

import struct
from math import fmod
from operator import index
from typing import TYPE_CHECKING, overload

if TYPE_CHECKING:
    from collections.abc import Buffer, Callable, Iterator, Sequence

__all__ = ["Path"]


class Path:
    """Store coordinate pairs, flat coordinates, native float buffers, or a copy of a path.

    An integer creates that many zero-valued vertices. Item assignment is
    supported; slices produce independent paths. Slice assignment and deletion
    are unsupported. Paths can be passed directly to ImageDraw methods.
    """

    def __init__(self, xy: int | Sequence[float] | Sequence[tuple[float, float]] | Buffer | Path) -> None:
        """Copy coordinates or native float pairs, or allocate an integer count of zero vertices."""
        self._mapping = False
        if isinstance(xy, int):
            if xy < 0:
                raise MemoryError("negative path size")
            self._points = [(0.0, 0.0)] * xy
            return
        if isinstance(xy, Path):
            self._points = xy._points.copy()
            return
        try:
            buffer = memoryview(xy)
        except TypeError:
            buffer = None
        if buffer is not None and buffer.c_contiguous:
            data = buffer.cast("B")
            self._points = list(struct.iter_unpack("=ff", data[: len(data) // 8 * 8]))
            return
        if not hasattr(xy, "__getitem__") or isinstance(xy, str):
            raise TypeError("argument must be a coordinate sequence")
        values: list[float] = []
        for value in xy:
            if isinstance(value, (tuple, list)):
                if len(value) != 2:
                    raise ValueError("coordinate list must contain exactly 2 coordinates")
                values.extend(self._pair(value))
            else:
                values.append(self._number(value))
        if len(values) % 2:
            raise ValueError("wrong number of coordinates")
        self._points = list(zip(values[::2], values[1::2], strict=True))

    @staticmethod
    def _number(value: float) -> float:
        if isinstance(value, (str, bytes, bytearray)):
            raise ValueError("coordinate list must contain numbers")
        return float(value)

    @classmethod
    def _pair(cls, value: Sequence[float]) -> tuple[float, float]:
        x, y = value
        return cls._number(x), cls._number(y)

    def __len__(self) -> int:
        """Return the number of vertices."""
        return len(self._points)

    def __iter__(self) -> Iterator[tuple[float, float]]:
        """Iterate over the coordinate pairs in path order."""
        return iter(self._points)

    @overload
    def __getitem__(self, key: int) -> tuple[float, float]: ...

    @overload
    def __getitem__(self, key: slice) -> Path: ...

    def __getitem__(self, key: int | slice) -> tuple[float, float] | Path:
        """Return a coordinate pair or an independent path for a contiguous slice."""
        if isinstance(key, slice):
            points = self._points[key]
            if points and key.step not in (None, 1):
                raise TypeError("slice steps not supported")
            return Path(points)
        return self._points[index(key)]

    def __setitem__(self, key: int, value: tuple[float, float]) -> None:
        """Replace the vertex at an index with a numeric coordinate tuple."""
        key = index(key)
        if not -len(self) <= key < len(self):
            raise IndexError("path assignment index out of range")
        if not isinstance(value, tuple):
            raise TypeError("coordinate must be a tuple")
        self._points[key] = self._pair(value)

    def __delitem__(self, key: int | slice) -> None:
        """Raise TypeError because deleting vertices by index or slice is unsupported."""
        raise TypeError("cannot delete from path")

    def compact(self, distance: float = 2) -> int:
        """Remove vertices closer than the Manhattan distance; return the number removed."""
        if self._mapping:
            raise ValueError("Path compacted during mapping")
        distance = float(distance)
        if not self._points:
            return 0
        points = [self._points[0]]
        for x, y in self._points[1:]:
            if abs(x - points[-1][0]) + abs(y - points[-1][1]) >= distance:
                points.append((x, y))
        removed = len(self) - len(points)
        self._points[:] = points
        return removed

    def getbbox(self) -> tuple[float, float, float, float]:
        """Return (left, top, right, bottom), or four zeros for an empty path."""
        if not self._points:
            return 0.0, 0.0, 0.0, 0.0
        xs, ys = zip(*self._points, strict=True)
        return min(xs), min(ys), max(xs), max(ys)

    def map(self, function: Callable[[float, float], tuple[float, float]]) -> None:
        """Replace each vertex with function(x, y), updating the path in place."""
        previous = self._mapping
        self._mapping = True
        try:
            for i, (x, y) in enumerate(self._points):
                self[i] = function(x, y)
        finally:
            self._mapping = previous

    def tolist(self, flat: bool = False) -> list[tuple[float, float]] | list[float]:
        """Return copied coordinate pairs, or a flat list when flat is true."""
        if flat:
            return [value for point in self._points for value in point]
        return self._points.copy()

    def transform(self, matrix: Sequence[float], wrap: float = 0) -> None:
        """Apply (a*x + b*y + c, d*x + e*y + f), optionally wrapping x with fmod."""
        a, b, c, d, e, f = (self._number(value) for value in matrix)
        wrap = float(wrap)
        for i, (x, y) in enumerate(self._points):
            tx, ty = a * x + b * y + c, d * x + e * y + f
            self._points[i] = (fmod(tx, wrap) if wrap else tx, ty)
