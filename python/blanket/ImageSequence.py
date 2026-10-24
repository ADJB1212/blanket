"""Iteration and independent copies of image frames and pages."""

from __future__ import annotations

__lazy_modules__ = {"collections.abc"}

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

from .Image import Image


class Iterator:
    """Iterate from frame zero, yielding the source image at each position."""

    def __init__(self, image: Image) -> None:
        """Retain the source image and start iteration at frame zero."""
        self.image = image
        self.position = 0

    def __iter__(self) -> Iterator:
        """Return this iterator without resetting its position."""
        return self

    def __next__(self) -> Image:
        """Select and return the next frame on the shared source image."""
        try:
            frame = self[self.position]
        except EOFError:
            raise StopIteration from None
        self.position += 1
        return frame

    def __getitem__(self, index: int) -> Image:
        """Seek the source image to a frame and return it, raising EOFError if absent."""
        self.image.seek(index)
        return self.image


def all_frames(image: Image | Sequence[Image], func: Callable[[Image], Image] | None = None) -> list[Image]:
    """Copy all frames, optionally transform them, and restore source positions."""
    images = [image] if isinstance(image, Image) else image
    frames: list[Image] = []
    for source in images:
        position = source.tell()
        try:
            for frame in Iterator(source):
                copied = frame.copy()
                frames.append(copied if func is None else func(copied))
        finally:
            source.seek(position)
    return frames


__all__ = ["Iterator", "all_frames"]
