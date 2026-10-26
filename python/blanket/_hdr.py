"""PNG HDR metadata serialization, in physical chromaticity and nit units."""

from __future__ import annotations

__lazy_modules__ = {"math"}

import math
import struct

HDR_KEYS = {b"cICP": "cicp", b"mDCV": "mastering_display", b"cLLI": "content_light_level"}


def encode_chunk(key: str, value: object) -> bytes:
    """Validate and encode one public HDR metadata tuple."""
    if not isinstance(value, (tuple, list)):
        raise TypeError(f"{key} must be a tuple or list")
    if key == "cicp":
        if len(value) != 4 or any(isinstance(v, bool) or not isinstance(v, int) or not 0 <= v <= 255 for v in value):
            raise ValueError("cicp must contain four byte-sized integers")
        if value[2] != 0 or value[3] != 1:
            raise ValueError("PNG cICP requires identity matrix coefficients and full range")
        return bytes(value)
    count = 10 if key == "mastering_display" else 2
    if len(value) != count or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in value):
        raise ValueError(f"{key} requires {count} finite nonnegative numbers")
    if key == "mastering_display":
        if any(v > 1 for v in value[:8]) or value[9] > value[8]:
            raise ValueError("invalid mastering display chromaticities or luminance range")
        scales = [50000] * 8 + [10000] * 2
        limits = [65535] * 8 + [0xFFFFFFFF] * 2
        layout = ">8H2I"
    else:
        scales, limits, layout = [10000] * 2, [0xFFFFFFFF] * 2, ">2I"
    packed = [round(v * scale) for v, scale in zip(value, scales, strict=True)]
    if any(v > limit for v, limit in zip(packed, limits, strict=True)):
        raise ValueError(f"{key} exceeds the PNG metadata range")
    return struct.pack(layout, *packed)


def decode_chunk(kind: bytes, payload: bytes) -> tuple[int, ...] | tuple[float, ...] | None:
    """Ignore malformed HDR chunks consistently with other optional metadata."""
    try:
        if kind == b"cICP":
            value = struct.unpack(">4B", payload)
        elif kind == b"mDCV":
            raw = struct.unpack(">8H2I", payload)
            value = tuple(v / (50000 if i < 8 else 10000) for i, v in enumerate(raw))
        else:
            value = tuple(v / 10000 for v in struct.unpack(">2I", payload))
        encode_chunk(HDR_KEYS[kind], value)
    except (ValueError, TypeError, struct.error):
        return None
    return value
