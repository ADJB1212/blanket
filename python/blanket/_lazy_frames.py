"""Bounded frame storage for opt-in sequence decoding."""

from __future__ import annotations

__lazy_modules__ = {"copy", "itertools", "blanket.Image"}

from collections import OrderedDict
from copy import deepcopy
from itertools import islice
from typing import TYPE_CHECKING

from ._blanket import _open_sequence
from ._exif import read_metadata, read_tiff_metadata

if TYPE_CHECKING:
    from collections.abc import Iterator

    from .Image import Image


class LazyFrames:
    """Bounded cache and on-demand decoder for sequence frames."""

    def __init__(self, data: bytes, count: int, capacity: int, loop: int | None, default_image: bool) -> None:
        """Store encoded sequence bytes, frame count, cache capacity, and animation controls."""
        self.data = data
        self.count = count
        self.capacity = capacity
        self.loop = loop
        self.default_image = default_image
        self.cache: OrderedDict[int, Image] = OrderedDict()
        self._tiff_metadata: Iterator[dict[object, object]] | None = None
        self._tiff_frame = 0
        self._metadata: dict[object, object] | None = None

    def get(self, frame: int) -> Image:
        """Decode or retrieve a cached frame by its zero-based index."""
        from .Image import Image

        if frame in self.cache:
            self.cache.move_to_end(frame)
            return self.cache[frame]
        frames, controls, _, _ = _open_sequence(self.data, None, frame if self.count > 1 else None)
        selected = Image(frames[0])
        if selected.format == "TIFF":
            if self._tiff_metadata is None or frame < self._tiff_frame:
                self._tiff_metadata = read_tiff_metadata(self.data, self.count, decoded=True)
                self._tiff_frame = 0
            selected.info.update(next(islice(self._tiff_metadata, frame - self._tiff_frame, None)))
            self._tiff_frame = frame + 1
        else:
            if self._metadata is None:
                self._metadata = read_metadata(self.data, selected.format)
            selected.info.update(deepcopy(self._metadata))
        duration, disposal, blend = controls[0]
        if duration is not None:
            selected.info["duration"] = duration
        if self.loop is not None:
            selected.info["loop"] = self.loop
        if self.default_image:
            selected.info["default_image"] = True
        if selected.format == "GIF":
            selected.info["disposal"] = disposal
        elif selected.format == "PNG" and (self.count > 1 or self.loop is not None):
            selected.info.update(disposal=disposal, blend=blend)
        self.cache[frame] = selected
        while len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
        return selected

    def close(self) -> None:
        """Close all cached frame images and release the encoded sequence data."""
        for frame in self.cache.values():
            frame.close()
        self.cache.clear()
        self._tiff_metadata = None
        self._metadata = None
        self.data = b""
