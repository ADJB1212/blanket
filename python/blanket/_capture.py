"""Python values for native EXIF GPS and capture-time helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone, tzinfo
from typing import TYPE_CHECKING

from ._blanket import _get_capture_time, _get_gps, _set_capture_time, _set_gps

if TYPE_CHECKING:
    from ._ifd import Exif


@dataclass(frozen=True)
class GPS:
    """Decimal latitude/longitude in degrees and optional altitude in meters."""

    latitude: float
    longitude: float
    altitude: float | None = None


def get_gps(exif: Exif) -> GPS | None:
    """Read native EXIF GPS fields; return None when coordinates are absent.

    Incomplete or invalid coordinates raise ValueError. Altitude is optional
    and positive above sea level. No image pixels are decoded.
    """
    value = _get_gps(exif.tobytes())
    return GPS(*value) if value is not None else None


def set_gps(exif: Exif, value: GPS | None) -> None:
    """Set validated GPS coordinates natively; None removes the GPS IFD.

    Unrelated GPS fields are retained. Coordinates must be finite and in range.
    EXIF rationals round coordinates to microseconds of arc.
    """
    if value is not None and not isinstance(value, GPS):
        raise TypeError("value must be GPS or None")
    fields = (value.latitude, value.longitude, value.altitude) if value is not None else None
    exif.load(_set_gps(exif.tobytes(), fields))


def get_capture_time(exif: Exif, *, default_timezone: tzinfo | None = None) -> datetime | None:
    """Read native DateTimeOriginal, subseconds, and OffsetTimeOriginal.

    Missing DateTimeOriginal returns None. A missing offset requires an explicit
    default_timezone. Subseconds truncate to microseconds. Invalid fields raise
    ValueError; the host's timezone is never assumed.
    """
    value = _get_capture_time(exif.tobytes())
    if value is None:
        return None
    year, month, day, hour, minute, second, microsecond, offset = value
    zone = timezone(timedelta(minutes=offset)) if offset is not None else default_timezone
    if not isinstance(zone, tzinfo):
        raise ValueError("capture time has no timezone; supply default_timezone")
    result = datetime(year, month, day, hour, minute, second, microsecond, tzinfo=zone)
    if result.utcoffset() is None:
        raise ValueError("capture time requires an explicit timezone offset")
    return result


def set_capture_time(exif: Exif, value: datetime | None) -> None:
    """Write an aware capture timestamp natively, retaining unrelated EXIF tags.

    None removes DateTimeOriginal, SubSecTimeOriginal, and OffsetTimeOriginal.
    Naive datetimes and offsets containing seconds are rejected before mutation.
    """
    fields = None
    if value is not None:
        if not isinstance(value, datetime):
            raise TypeError("value must be a datetime or None")
        offset = value.utcoffset()
        if offset is None or offset.total_seconds() % 60:
            raise ValueError("capture time requires a timezone offset in whole minutes")
        fields = (value.year, value.month, value.day, value.hour, value.minute, value.second, value.microsecond, int(offset.total_seconds() / 60))
    exif.load(_set_capture_time(exif.tobytes(), fields))
