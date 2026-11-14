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
    dpi, aspect, source, unit_name, conflicts = _resolve_resolution(data, format, candidates, conflict)
    return Resolution(dpi, aspect, source, unit_name, tuple(conflicts))
