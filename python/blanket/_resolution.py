"""Physical resolution parsing and normalization."""

from __future__ import annotations

__lazy_modules__ = {"struct"}

import struct
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from ._blanket import _resolve_resolution
from ._ifd import Exif

if TYPE_CHECKING:
    from typing import Literal

    from ._typing import FloatInput


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


def parse_resolution(data: bytes, format: str, info: dict[object, object], conflict: Literal["container", "exif", "error"]) -> Resolution:
    """Parse and reconcile container chunks and EXIF tags to determine resolution."""
    if conflict not in ("container", "exif", "error"):
        raise ValueError("conflict must be 'container', 'exif', or 'error'")
    candidates: list[tuple[float, float, int, str]] = []
    if format == "PNG":
        pos = 8
        while pos + 12 <= len(data):
            length = int.from_bytes(data[pos : pos + 4], "big")
            end = pos + length + 12
            if end > len(data) or data[pos + 4 : pos + 8] == b"IEND":
                break
            if data[pos + 4 : pos + 8] == b"pHYs" and length == 9:
                x, y, unit = struct.unpack_from(">IIB", data, pos + 8)
                candidates.append((x, y, unit, "png"))
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
                candidates.append((x, y, unit, "jfif"))
                break
            pos = end
    elif format == "BMP" and len(data) >= 46 and int.from_bytes(data[14:18], "little") >= 40:
        x, y = struct.unpack_from("<ii", data, 38)
        candidates.append((x, y, 1, "bmp"))
    raw = info.get("exif")
    if isinstance(raw, bytes):
        try:
            exif = Exif()
            exif.load(raw)
            x, y = float(cast("FloatInput", exif[282])), float(cast("FloatInput", exif[283]))
            unit = exif.get(296, 2)
            if isinstance(unit, int) and unit in (1, 2, 3):
                candidates.append((x, y, unit, "tiff" if format == "TIFF" else "exif"))
        except (KeyError, TypeError, ValueError, OverflowError, struct.error):
            pass
    dpi, aspect, source, unit_name, conflicts = _resolve_resolution(candidates, conflict)
    return Resolution(dpi, aspect, source, unit_name, tuple(conflicts))
