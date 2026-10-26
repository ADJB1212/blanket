"""Color argument parsing for fills, transforms, and ImageOps; no optional dependencies required."""

from __future__ import annotations

import re
from functools import lru_cache

from ._color_names import CSS_COLORS
from ._css_color import functional_rgb

# Bound parser input and cached results independently.
_MAX_COLOR_LENGTH = 1024
_COLOR_CACHE_SIZE = 512

_HEX = re.compile(r"#[0-9a-f]{3,8}")


def _rgb(color: str) -> tuple[int, ...]:
    """Validate a color string and return its RGB or RGBA samples."""
    if not isinstance(color, str):
        raise TypeError("color must be a string")
    return _parse_rgb(color)


@lru_cache(maxsize=_COLOR_CACHE_SIZE)
def _parse_rgb(color: str) -> tuple[int, ...]:
    """Parse a named, hexadecimal, or functional color into 8-bit samples."""
    if len(color) > _MAX_COLOR_LENGTH:
        raise ValueError("color specifier is too long")
    value = color.strip().lower()
    value = CSS_COLORS.get(value, value)
    if _HEX.fullmatch(value):
        digits = value[1:]
        if len(digits) in (3, 4):
            return tuple(int(c * 2, 16) for c in digits)
        if len(digits) in (6, 8):
            return tuple(int(digits[i : i + 2], 16) for i in range(0, len(digits), 2))
    try:
        return functional_rgb(value)
    except (ValueError, OverflowError):
        pass
    raise ValueError(f"unknown color specifier: {color!r}")


def color_pixel(color: str | int | tuple[int, ...] | None, mode: str, bit_depth: int = 8) -> list[int]:
    """Normalize fill values to the native interleaved channel representation."""
    channels = 3 if mode == "YCbCr" else len(mode)
    maximum = (1 << bit_depth) - 1
    if color is None:
        return [0] * channels
    if isinstance(color, str):
        rgb = _rgb(color)
        if mode in ("1", "L", "LA"):
            color = (rgb[0] * 19595 + rgb[1] * 38470 + rgb[2] * 7471 + 32768) >> 16
            if mode == "LA":
                color = (color, rgb[3] if len(rgb) == 4 else 255)
        else:
            color = rgb[:3] + ((rgb[3] if len(rgb) == 4 else 255,) if mode == "RGBA" else ())
        if bit_depth != 8:
            if isinstance(color, int):
                color = (color * maximum + 127) // 255
            else:
                color = tuple((v * maximum + 127) // 255 for v in color)
    if isinstance(color, int):
        if mode in ("1", "L"):
            return [max(0, min(maximum, color))]
        return [(color >> (bit_depth * i)) & maximum for i in range(channels)]
    if not isinstance(color, tuple):
        raise TypeError("color must be int or tuple")
    if mode in ("1", "L"):
        if len(color) != 1:
            raise TypeError("color must be int or single-element tuple")
    elif (len(color) == 1 and mode == "LA") or (len(color) == 3 and mode in ("RGBA", "CMYK")):
        color = (*color, maximum)
    elif mode in ("RGB", "YCbCr") and len(color) == 4:
        color = color[:3]
    elif len(color) != channels:
        raise TypeError("color must be int, or tuple of the appropriate channel count")
    if not all(isinstance(v, int) for v in color):
        raise TypeError("color components must be integers")
    if mode == "1":
        return [max(0, min(255, color[0]))]
    return [max(0, min(maximum, v)) for v in color]
