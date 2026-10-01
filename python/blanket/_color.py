"""Color argument parsing for fills, transforms, and ImageOps; no optional dependencies required."""

from __future__ import annotations

import colorsys
import re
from functools import lru_cache

from ._color_names import CSS_COLORS

_HEX = re.compile(r"#[0-9a-f]{3,8}")
_FUNCTION = re.compile(r"(rgb|rgba|hsl|hsv|hsb)\((.*)\)")
_INTEGER = re.compile(r"\d+")
_PERCENT = re.compile(r"\d+%")
_DECIMAL = re.compile(r"\d+\.?\d*")
_DECIMAL_PERCENT = re.compile(r"\d+\.?\d*%")


def _rgb(color: str) -> tuple[int, ...]:
    if not isinstance(color, str):
        raise TypeError("color must be a string")
    return _parse_rgb(color)


@lru_cache(maxsize=512)
def _parse_rgb(color: str) -> tuple[int, ...]:
    if len(color) > 100:
        raise ValueError("color specifier is too long")
    value = color.lower()
    value = CSS_COLORS.get(value, value)
    if _HEX.fullmatch(value):
        digits = value[1:]
        if len(digits) in (3, 4):
            return tuple(int(c * 2, 16) for c in digits)
        if len(digits) in (6, 8):
            return tuple(int(digits[i : i + 2], 16) for i in range(0, len(digits), 2))
    match = _FUNCTION.fullmatch(value)
    if match:
        model, body = match.groups()
        parts = [p.strip() for p in body.split(",")]
        if model in ("rgb", "rgba"):
            if len(parts) == (4 if model == "rgba" else 3) and all(_INTEGER.fullmatch(p) for p in parts):
                return tuple(int(p) for p in parts)
            if model == "rgb" and len(parts) == 3 and all(_PERCENT.fullmatch(p) for p in parts):
                return tuple(int(int(p[:-1]) * 255 / 100 + 0.5) for p in parts)
        elif len(parts) == 3 and _DECIMAL.fullmatch(parts[0]) and all(_DECIMAL_PERCENT.fullmatch(p) for p in parts[1:]):
            h, s, v = float(parts[0]) / 360, float(parts[1][:-1]) / 100, float(parts[2][:-1]) / 100
            rgb = colorsys.hls_to_rgb(h, v, s) if model == "hsl" else colorsys.hsv_to_rgb(h, s, v)
            return tuple(int(c * 255 + 0.5) for c in rgb)
    raise ValueError(f"unknown color specifier: {color!r}")


def color_pixel(color: str | int | tuple[int, ...] | None, mode: str) -> list[int]:
    """Normalize fill values to the native interleaved channel representation."""
    channels = 3 if mode == "YCbCr" else len(mode)
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
    if isinstance(color, int):
        if mode in ("1", "L"):
            return [max(0, min(255, color))]
        return [(color >> (8 * i)) & 255 for i in range(channels)]
    if not isinstance(color, tuple):
        raise TypeError("color must be int or tuple")
    if mode in ("1", "L"):
        if len(color) != 1:
            raise TypeError("color must be int or single-element tuple")
    elif (len(color) == 1 and mode == "LA") or (len(color) == 3 and mode in ("RGBA", "CMYK")):
        color = (*color, 255)
    elif mode in ("RGB", "YCbCr") and len(color) == 4:
        color = color[:3]
    elif len(color) != channels:
        raise TypeError("color must be int, or tuple of the appropriate channel count")
    if not all(isinstance(v, int) for v in color):
        raise TypeError("color components must be integers")
    if mode == "1":
        return [max(0, min(255, color[0]))]
    return [max(0, min(255, v)) for v in color]
