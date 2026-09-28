"""Color name and color string helpers."""

from __future__ import annotations

from ._blanket import color_getcolor
from ._color import _rgb


def getrgb(color: str) -> tuple[int, ...]:
    """Convert a CSS color string to an RGB or RGBA tuple."""
    return _rgb(color)


def getcolor(color: str, mode: str) -> int | tuple[int, ...]:
    """Convert a CSS color string to a pixel value for *mode*."""
    return color_getcolor(_rgb(color), mode)


__all__ = ["getcolor", "getrgb"]
