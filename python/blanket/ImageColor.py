"""Color name and color string helpers."""

from __future__ import annotations

from functools import lru_cache

from ._blanket import color_getcolor
from ._color import _rgb
from ._color_names import CSS_COLORS

colormap = CSS_COLORS


@lru_cache(maxsize=512)
def getrgb(color: str) -> tuple[int, ...]:
    """Convert an absolute CSS color literal to an 8-bit sRGB or RGBA tuple.

    Supports CSS4 functions and predefined spaces, plus Pillow's HSV and
    integer-alpha RGBA extensions. Contextual CSS values require resolution
    by the caller. Out-of-gamut colors use CSS local MINDE gamut mapping.

    Args:
        color: CSS color name or absolute color function.

    Returns:
        ``(r, g, b)`` or ``(r, g, b, a)`` samples in the range 0 through 255.

    Raises:
        TypeError: ``color`` is not a string.
        ValueError: ``color`` is not a supported absolute color.
    """
    return _rgb(color)


@lru_cache(maxsize=512)
def getcolor(color: str, mode: str) -> int | tuple[int, ...]:
    """Return Pillow-compatible color samples for *mode*.

    Mode 1 returns luminance rather than thresholding it. Color modes other
    than HSV return RGB samples, with alpha for PA and RGBA; use image
    conversion when actual CMYK, YCbCr, or LAB pixel values are needed.

    Args:
        color: CSS color name or absolute color function.
        mode: Pillow-style destination mode, such as ``"RGB"`` or ``"L"``.

    Returns:
        A single sample for single-band modes, or a channel tuple otherwise.

    Raises:
        TypeError: ``color`` is not a string.
        ValueError: ``color`` or ``mode`` is not supported.
    """
    return color_getcolor(_rgb(color), mode)


__all__ = ["colormap", "getcolor", "getrgb"]
