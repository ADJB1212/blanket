"""JPEG marker metadata and public header attributes."""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

_ZIGZAG = (
    0,
    1,
    5,
    6,
    14,
    15,
    27,
    28,
    2,
    4,
    7,
    13,
    16,
    26,
    29,
    42,
    3,
    8,
    12,
    17,
    25,
    30,
    41,
    43,
    9,
    11,
    18,
    24,
    31,
    40,
    44,
    53,
    10,
    19,
    23,
    32,
    39,
    45,
    52,
    54,
    20,
    22,
    33,
    38,
    46,
    51,
    55,
    60,
    21,
    34,
    37,
    47,
    50,
    56,
    59,
    61,
    35,
    36,
    48,
    49,
    57,
    58,
    62,
    63,
)


def segments(data: bytes) -> Iterator[tuple[int, memoryview]]:
    """Yield bounded header segments without copying unknown payloads."""
    view = memoryview(data)
    position = 2
    while position < len(data):
        if data[position] != 255:
            position += 1
            continue
        while position < len(data) and data[position] == 255:
            position += 1
        if position == len(data):
            return
        marker = data[position]
        position += 1
        if marker in (0xDA, 0xD9):
            return
        if marker in (0, 1) or 0xD0 <= marker <= 0xD8:
            continue
        if position + 2 > len(data):
            return
        length = int.from_bytes(view[position : position + 2], "big")
        if length < 2 or position + length > len(data):
            return
        yield marker, view[position + 2 : position + length]
        position += length


def read_metadata(data: bytes) -> dict[object, object]:
    """Read JFIF density, Adobe interpretation, comments, and progression."""
    info: dict[object, object] = {}
    for marker, payload in segments(data):
        if marker == 0xE0 and payload[:5] == b"JFIF\0" and len(payload) >= 12:
            version = int.from_bytes(payload[5:7], "big")
            density = struct.unpack_from(">HH", payload, 8)
            info.update(jfif=version, jfif_version=divmod(version, 256), jfif_unit=payload[7], jfif_density=density)
            if payload[7] == 1:
                info["dpi"] = density
            elif payload[7] == 2:
                info["dpi"] = tuple(value * 2.54 for value in density)
        elif marker == 0xEE and payload[:5] == b"Adobe" and len(payload) >= 7:
            info["adobe"] = int.from_bytes(payload[5:7], "big")
            if len(payload) >= 12:
                info["adobe_transform"] = payload[11]
        elif marker == 0xFE:
            info["comment"] = bytes(payload)
        elif marker in (0xC2, 0xC6, 0xCA, 0xCE):
            info["progressive"] = info["progression"] = 1
    return info


def read_attributes(data: bytes) -> dict[str, object]:
    """Read application markers, component sampling, and quantization tables."""
    app: dict[str, bytes] = {}
    applist: list[tuple[str, bytes]] = []
    quantization: dict[int, list[int]] = {}
    attributes: dict[str, object] = {"app": app, "applist": applist, "quantization": quantization}
    for marker, payload in segments(data):
        if 0xE0 <= marker <= 0xEF or marker == 0xFE:
            name = "COM" if marker == 0xFE else f"APP{marker - 0xE0}"
            value = bytes(payload)
            app[name] = value
            applist.append((name, value))
        elif marker == 0xDB:
            position = 0
            while position < len(payload):
                header = payload[position]
                precision = 2 if header >> 4 else 1
                if position + 1 + precision * 64 > len(payload):
                    break
                values = struct.unpack_from(">64H" if precision == 2 else "64B", payload, position + 1)
                quantization[header & 15] = [values[index] for index in _ZIGZAG]
                position += 1 + precision * 64
        elif 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC) and len(payload) >= 6:
            layers = payload[5]
            if len(payload) >= 6 + 3 * layers:
                attributes.update(bits=payload[0], layers=layers, layer=[(payload[i], payload[i + 1] >> 4, payload[i + 1] & 15, payload[i + 2]) for i in range(6, 6 + 3 * layers, 3)])
    return attributes
