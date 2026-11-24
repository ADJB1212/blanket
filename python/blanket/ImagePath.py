"""Mutable two-dimensional paths for coordinate manipulation and drawing."""

from __future__ import annotations

__lazy_modules__ = {"collections.abc"}

from operator import index, truth
from typing import TYPE_CHECKING, cast, overload

from ._blanket import _Path

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
        """Copy coordinates or native float pairs, or allocate an integer count of zero vertices.

        Args:
            xy: Vertex count, coordinate sequence, flat buffer, or another path.

        Raises:
            MemoryError: ``xy`` is a negative vertex count.
            TypeError: ``xy`` is not a coordinate sequence or buffer.
        """
        self._mapping = False
        if isinstance(xy, int):
            if xy < 0:
                raise MemoryError("negative path size")
            self._points = _Path.zeros(xy)
            return
        if isinstance(xy, Path):
            self._points = xy._points.copy()
            return
        if isinstance(xy, bytes):
            self._points = _Path.from_buffer(xy)
            return
        try:
            buffer = memoryview(cast("Buffer", xy))
        except TypeError:
            buffer = None
        if buffer is not None and buffer.c_contiguous:
            data = buffer.cast("B")
            self._points = _Path.from_buffer(data.tobytes())
            return
        if not hasattr(xy, "__getitem__") or isinstance(xy, str):
            raise TypeError("argument must be a coordinate sequence")
        self._points = _Path(xy)

    @staticmethod
    def _number(value: float) -> float:
        """Convert a coordinate value to a float, rejecting string and byte inputs."""
        if isinstance(value, (str, bytes, bytearray)):
            raise ValueError("coordinate list must contain numbers")
        return float(value)

    @classmethod
    def _pair(cls, value: Sequence[float]) -> tuple[float, float]:
        """Convert a two-element sequence to a pair of float coordinates."""
        x, y = value
        return cls._number(x), cls._number(y)

    def __len__(self) -> int:
        """Return the number of vertices."""
        return len(self._points)

    def __iter__(self) -> Iterator[tuple[float, float]]:
        """Iterate over the coordinate pairs in path order."""
        return iter(self._points)

    @overload
    def __getitem__(self, key: int) -> tuple[float, float]:
        """Return the coordinate pair at the specified index.

        Examples:
            ```python
            from blanket.ImagePath import Path

            path = Path([(0, 0), (10, 20)])
            point = path[1]
            ```
        """

    @overload
    def __getitem__(self, key: slice) -> Path:
        """Return a new Path containing a slice of vertices.

        Examples:
            ```python
            from blanket.ImagePath import Path

            path = Path([(0, 0), (10, 20), (30, 40)])
            subpath = path[1:]
            ```
        """

    def __getitem__(self, key: int | slice) -> tuple[float, float] | Path:
        """Return a coordinate pair or an independent path for a contiguous slice."""
        if isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            if step != 1:
                if range(start, stop, step):
                    raise TypeError("slice steps not supported")
                return Path(0)
            result = Path.__new__(Path)
            result._mapping = False
            result._points = self._points.slice(start, max(start, stop))
            return result
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
        """Remove vertices closer than the Manhattan distance.

        Args:
            distance: Minimum Manhattan separation to retain. Closer vertices are removed.

        Returns:
            The number of vertices removed.

        Raises:
            ValueError: The path is being mapped.
        """
        if self._mapping:
            raise ValueError("Path compacted during mapping")
        return self._points.compact(float(distance))

    def getbbox(self) -> tuple[float, float, float, float]:
        """Return (left, top, right, bottom), or four zeros for an empty path.

        Returns:
            The axis-aligned bounds of the vertices.
        """
        return self._points.getbbox()

    def map(self, function: Callable[[float, float], tuple[float, float]]) -> None:
        """Replace each vertex with function(x, y), updating the path in place.

        Args:
            function: Callable receiving ``x`` and ``y`` and returning a new pair.
        """
        previous = self._mapping
        self._mapping = True
        try:
            self._points.map(function)
        finally:
            self._mapping = previous

    def tolist(self, flat: bool = False) -> list[tuple[float, float]] | list[float]:
        """Return copied coordinate pairs, or a flat list when flat is true.

        Args:
            flat: Return ``[x, y, ...]`` when true, otherwise ``[(x, y), ...]``.

        Returns:
            A copy of the coordinates.
        """
        return self._points.tolist(truth(flat))

    def transform(self, matrix: Sequence[float], wrap: float = 0) -> None:
        """Apply (a*x + b*y + c, d*x + e*y + f), optionally wrapping x with fmod.

        Args:
            matrix: Six coefficients ``(a, b, c, d, e, f)``.
            wrap: Period used to wrap x with ``fmod``. Zero leaves x unwrapped.
        """
        a, b, c, d, e, f = (self._number(value) for value in matrix)
        self._points.transform((a, b, c, d, e, f), float(wrap))
