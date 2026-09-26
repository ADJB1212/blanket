"""Color argument parsing for ImageOps; no optional dependencies required."""

from __future__ import annotations

import colorsys
import re

from ._color_names import CSS_COLORS


def _rgb(color: str) -> tuple[int, ...]:
    if len(color) > 100:
        raise ValueError("color specifier is too long")
    value = color.lower()
    value = CSS_COLORS.get(value, value)
    if re.fullmatch(r"#[0-9a-f]{3,8}", value):
        digits = value[1:]
        if len(digits) in (3, 4):
            return tuple(int(c * 2, 16) for c in digits)
        if len(digits) in (6, 8):
            return tuple(int(digits[i : i + 2], 16) for i in range(0, len(digits), 2))
    match = re.fullmatch(r"(rgb|rgba|hsl|hsv|hsb)\((.*)\)", value)
    if match:
        model, body = match.groups()
        parts = [p.strip() for p in body.split(",")]
        if model in ("rgb", "rgba"):
            if len(parts) == (4 if model == "rgba" else 3) and all(re.fullmatch(r"\d+", p) for p in parts):
                return tuple(int(p) for p in parts)
            if model == "rgb" and len(parts) == 3 and all(re.fullmatch(r"\d+%", p) for p in parts):
                return tuple(int(int(p[:-1]) * 255 / 100 + 0.5) for p in parts)
        elif len(parts) == 3 and re.fullmatch(r"\d+\.?\d*", parts[0]) and all(re.fullmatch(r"\d+\.?\d*%", p) for p in parts[1:]):
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
        if mode == "1":
            return [max(0, min(255, color))]
        if mode == "L":
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
