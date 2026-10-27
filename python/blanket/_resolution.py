"""Physical resolution parsing and normalization."""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import Literal

from ._ifd import Exif


@dataclass(frozen=True)
class Resolution:
    """Normalized DPI and physical pixel width/height ratio.

    dpi is None for unitless densities. unit names the source unit before
    conversion; conflicts names other sources that disagree beyond rounding.
    Missing metadata has source None, unit None, and square pixels.
    """

    dpi: tuple[float, float] | None
    pixel_aspect: float
    source: str | None
    unit: str | None
    conflicts: tuple[str, ...] = ()


def _value(x: float, y: float, unit: int, source: str, units: dict[int, tuple[str, float | None]]) -> Resolution | None:
    if not math.isfinite(x) or not math.isfinite(y) or x <= 0 or y <= 0 or unit not in units:
        return None
    name, factor = units[unit]
    dpi = None if factor is None else (x * factor, y * factor)
    aspect = y / x
    if not math.isfinite(aspect) or aspect <= 0 or (dpi is not None and any(not math.isfinite(v) or v <= 0 for v in dpi)):
        return None
    return Resolution(dpi, aspect, source, name)


def parse_resolution(data: bytes, format: str, info: dict[object, object], conflict: Literal["container", "exif", "error"]) -> Resolution:
    if conflict not in ("container", "exif", "error"):
        raise ValueError("conflict must be 'container', 'exif', or 'error'")
    container = None
    if format == "PNG":
        pos = 8
        while pos + 12 <= len(data):
            length = int.from_bytes(data[pos : pos + 4], "big")
            end = pos + length + 12
            if end > len(data) or data[pos + 4 : pos + 8] == b"IEND":
                break
            if data[pos + 4 : pos + 8] == b"pHYs" and length == 9:
                x, y, unit = struct.unpack_from(">IIB", data, pos + 8)
                container = _value(x, y, unit, "png", {0: ("relative", None), 1: ("meter", 0.0254)})
                break
            pos = end
    elif format == "JPEG":
        pos = 2
        while pos + 4 <= len(data) and data[pos] == 255:
            while pos < len(data) and data[pos] == 255:
                pos += 1
            if pos >= len(data) or data[pos] in (0xDA, 0xD9):
                break
            marker = data[pos]
            pos += 1
            if marker == 1 or 0xD0 <= marker <= 0xD8:
                continue
            length = int.from_bytes(data[pos : pos + 2], "big")
            end = pos + length
            if length < 2 or end > len(data):
                break
            if marker == 0xE0 and data[pos + 2 : pos + 7] == b"JFIF\0" and length >= 16:
                unit, x, y = struct.unpack_from(">BHH", data, pos + 9)
                container = _value(x, y, unit, "jfif", {0: ("relative", None), 1: ("inch", 1.0), 2: ("cm", 2.54)})
                break
            pos = end
    elif format == "BMP" and len(data) >= 46 and int.from_bytes(data[14:18], "little") >= 40:
        x, y = struct.unpack_from("<ii", data, 38)
        container = _value(x, y, 1, "bmp", {1: ("meter", 0.0254)})
    exif_resolution = None
    raw = info.get("exif")
    if isinstance(raw, bytes):
        try:
            exif = Exif()
            exif.load(raw)
            x, y = float(exif[282]), float(exif[283])
            unit = exif.get(296, 2)
            if isinstance(unit, int):
                exif_resolution = _value(x, y, unit, "tiff" if format == "TIFF" else "exif", {1: ("relative", None), 2: ("inch", 1.0), 3: ("cm", 2.54)})
        except (KeyError, TypeError, ValueError, OverflowError, struct.error):
            pass
    candidates = [v for v in (container, exif_resolution) if v is not None]
    if not candidates:
        return Resolution(None, 1.0, None, None)
    preferred = list(reversed(candidates)) if conflict == "exif" else candidates
    selected = next((v for v in preferred if v.dpi is not None), preferred[0])
    conflicts = tuple(
        v.source
        for v in candidates
        if v is not selected
        and v.source is not None
        and (
            not math.isclose(v.pixel_aspect, selected.pixel_aspect, rel_tol=1e-3)
            or (v.dpi is not None and selected.dpi is not None and any(not math.isclose(a, b, rel_tol=1e-3) for a, b in zip(v.dpi, selected.dpi, strict=True)))
        )
    )
    if conflicts and conflict == "error":
        raise ValueError("conflicting resolution metadata")
    return Resolution(selected.dpi, selected.pixel_aspect, selected.source, selected.unit, conflicts)
