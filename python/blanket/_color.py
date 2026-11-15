"""Color argument parsing for fills, transforms, and ImageOps; no optional dependencies required."""

from __future__ import annotations

import re
from functools import lru_cache

from ._blanket import color_pixel as _native_color_pixel
from ._color_names import CSS_COLORS
from ._css_color import functional_rgb

# Bound parser input and cached results independently.
_MAX_COLOR_LENGTH = 100
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
    value = color.lower()
    stripped = value.strip()
    if value != stripped:
        if "(" not in stripped or "," in stripped:
            raise ValueError(f"unknown color specifier: {color!r}")
        value = stripped
    value = CSS_COLORS.get(value, value)
    if isinstance(value, tuple):
        return value
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
    if isinstance(color, str):
        return _native_color_pixel(None, mode, bit_depth, _rgb(color))
    return _native_color_pixel(color, mode, bit_depth)
