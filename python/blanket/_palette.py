"""Readers for the legacy palette formats accepted by ImagePalette.load."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from ._blanket import palette_gradient

if TYPE_CHECKING:
    from typing import BinaryIO

_GIMP_COMMENT = re.compile(rb"\w+:|#")


def _text_palette(stream: BinaryIO) -> tuple[bytes, str]:
    """Read indexed grayscale or RGB text entries into a 256-color palette."""
    entries = [bytes([value]) * 3 for value in range(256)]
    for line in stream:
        if line.startswith(b"#"):
            continue
        if len(line) > 100:
            raise SyntaxError("bad palette file")
        values = [int(value) for value in line.split()]
        if len(values) == 4:
            slot, red, green, blue = values
        else:
            slot, red = values
            green = blue = red
        if 0 <= slot <= 255:
            entries[slot] = bytes((red & 255, green & 255, blue & 255))
    return b"".join(entries), "RGB"


def _gimp_palette(stream: BinaryIO) -> tuple[bytes, str]:
    """Read up to 256 RGB entries from a GIMP palette."""
    if not stream.readline().startswith(b"GIMP Palette"):
        raise SyntaxError("not a GIMP palette file")
    entries = bytearray()
    for _ in range(259):
        line = stream.readline()
        if not line:
            break
        if _GIMP_COMMENT.match(line):
            continue
        if len(line) > 100:
            raise SyntaxError("bad palette file")
        values = line.split(maxsplit=3)
        if len(values) < 3:
            raise ValueError("bad palette entry")
        entries.extend(int(value) for value in values[:3])
        if len(entries) == 768:
            break
    return bytes(entries), "RGB"


def _gimp_gradient(stream: BinaryIO) -> tuple[bytes, str]:
    """Sample a GIMP gradient with RGB segments into an RGBA palette."""
    if not stream.readline().startswith(b"GIMP Gradient"):
        raise SyntaxError("not a GIMP gradient file")
    count = stream.readline()
    if count.startswith(b"Name: "):
        count = stream.readline()
    segments: list[tuple[list[float], int]] = []
    for _ in range(int(count)):
        fields = stream.readline().split()
        values = [float(value) for value in fields[:11]]
        kind = (0, 1, 2, 3, 4)[int(fields[11])]
        if int(fields[12]) != 0:
            raise OSError("cannot handle HSV colour space")
        segments.append((values, kind))
    return palette_gradient(segments), "RGBA"


def load_palette(stream: BinaryIO) -> tuple[bytes, str]:
    """Try supported palette readers and return palette bytes and mode."""
    for reader in (_gimp_palette, _gimp_gradient, _text_palette):
        stream.seek(0)
        try:
            return reader(stream)
        except (SyntaxError, ValueError):
            continue
    raise OSError("cannot load palette")
