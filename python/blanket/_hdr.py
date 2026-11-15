"""PNG HDR metadata serialization, in physical chromaticity and nit units."""

from __future__ import annotations

__lazy_modules__ = {"math"}

import math

from ._blanket import _decode_hdr, _encode_hdr

HDR_KEYS = {b"cICP": "cicp", b"mDCV": "mastering_display", b"cLLI": "content_light_level"}


def encode_chunk(key: str, value: object) -> bytes:
    """Validate and encode one public HDR metadata tuple."""
    if not isinstance(value, (tuple, list)):
        raise TypeError(f"{key} must be a tuple or list")
    if key == "cicp":
        if len(value) != 4 or any(isinstance(v, bool) or not isinstance(v, int) or not 0 <= v <= 255 for v in value):
            raise ValueError("cicp must contain four byte-sized integers")
    else:
        count = 10 if key == "mastering_display" else 2
        if len(value) != count or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in value):
            raise ValueError(f"{key} requires {count} finite nonnegative numbers")
    return _encode_hdr(key, value)


def decode_chunk(kind: bytes, payload: bytes) -> tuple[int, ...] | tuple[float, ...] | None:
    """Ignore malformed HDR chunks consistently with other optional metadata."""
    return _decode_hdr(kind, payload)
