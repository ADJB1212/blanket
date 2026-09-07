"""Read orientation metadata without decoding pixels or depending on Pillow."""

from __future__ import annotations

import re
import struct
import zlib


def _orientation(raw: bytes) -> tuple[int | None, bytes]:
    """Remove an IFD0 orientation entry without relocating referenced data."""
    prefix = b"Exif\0\0" if raw.startswith(b"Exif\0\0") else b""
    data = raw[len(prefix) :]
    if len(data) < 8 or data[:2] not in (b"II", b"MM"):
        return None, raw
    endian = "<" if data[:2] == b"II" else ">"
    try:
        magic, offset = struct.unpack_from(endian + "HI", data, 2)
        if magic != 42:
            return None, raw
        count = struct.unpack_from(endian + "H", data, offset)[0]
        end = offset + 2 + 12 * count
        if end + 4 > len(data):
            return None, raw
        for i in range(count):
            pos = offset + 2 + i * 12
            tag, kind, length = struct.unpack_from(endian + "HHI", data, pos)
            if tag != 0x112 or kind not in (3, 4) or length != 1:
                continue
            orientation = struct.unpack_from(endian + ("H" if kind == 3 else "I"), data, pos + 8)[0]
            if orientation not in range(2, 9):
                return orientation, raw
            clean = bytearray(data)
            struct.pack_into(endian + "H", clean, offset, count - 1)
            # Shift the remaining entries and next-IFD pointer within IFD0.
            # The trailing gap keeps every external value/thumbnail offset valid.
            clean[pos : end - 8] = data[pos + 12 : end + 4]
            clean[end - 8 : end + 4] = bytes(12)
            return orientation, prefix + bytes(clean)
    except (struct.error, OverflowError):
        pass
    return None, raw


ORIENTATION_RE = re.compile(r'tiff:Orientation="([0-9])"|<tiff:Orientation>([0-9])</tiff:Orientation>')

ORIENTATION_REMOVE_RE = re.compile(r'tiff:Orientation="[0-9]"|<tiff:Orientation>[0-9]</tiff:Orientation>')

ORIENTATION_REMOVE_BYTES_RE = re.compile(ORIENTATION_REMOVE_RE.pattern.encode())


def transpose_metadata(info: dict[object, object]) -> tuple[int, dict[object, object]]:
    result = info.copy()
    raw = info.get("exif")
    profile = False
    if raw is None and isinstance(info.get("Raw profile type exif"), str):
        try:
            value = info["Raw profile type exif"]
            raw = bytes.fromhex("".join(value.split("\n")[3:]) if "\n" in value else value)
            profile = True
        except ValueError:
            pass
    orientation, cleaned = _orientation(raw) if isinstance(raw, bytes) else (None, b"")
    if orientation is None:
        for key in ("XML:com.adobe.xmp", "xmp"):
            xmp = info.get(key, b"")
            if isinstance(xmp, tuple):
                xmp = b"".join(xmp)
            if isinstance(xmp, bytes):
                xmp = xmp.decode("utf-8", "replace")
            if isinstance(xmp, str):
                match = ORIENTATION_RE.search(xmp)
                if match:
                    orientation = int(match[1] or match[2])
                    break
    if orientation is None:
        orientation = 1
    if orientation in range(2, 9):
        if raw is not None:
            result["Raw profile type exif" if profile else "exif"] = cleaned.hex() if profile else cleaned
        for key in ("XML:com.adobe.xmp", "xmp"):
            value = result.get(key)
            if isinstance(value, str):
                result[key] = ORIENTATION_REMOVE_RE.sub("", value)
            elif isinstance(value, bytes):
                result[key] = ORIENTATION_REMOVE_BYTES_RE.sub(b"", value)
            elif isinstance(value, tuple):
                result[key] = tuple(ORIENTATION_REMOVE_BYTES_RE.sub(b"", v) for v in value)
    return orientation, result


def read_metadata(data: bytes, format: str | None) -> dict[object, object]:
    """Retain EXIF and XMP carried by PNG/JPEG/JXL containers."""
    info: dict[object, object] = {}
    if format == "JPEG":
        pos = 2
        while pos + 4 <= len(data) and data[pos] == 255:
            pos += 1
            while pos < len(data) and data[pos] == 255:
                pos += 1
            if pos >= len(data):
                break
            marker = data[pos]
            pos += 1
            if marker in (0xDA, 0xD9):
                break
            if marker == 1 or 0xD0 <= marker <= 0xD8:
                continue
            if pos + 2 > len(data):
                break
            length = int.from_bytes(data[pos : pos + 2], "big")
            if length < 2 or pos + length > len(data):
                break
            payload = data[pos + 2 : pos + length]
            if marker == 0xE1:
                if payload.startswith(b"Exif\0\0"):
                    info.setdefault("exif", payload)
                elif payload.startswith(b"http://ns.adobe.com/xap/1.0/\0"):
                    info["xmp"] = payload.split(b"\0", 1)[1]
            pos += length
    elif format == "PNG":
        pos = 8
        while pos + 12 <= len(data):
            length = int.from_bytes(data[pos : pos + 4], "big")
            kind = data[pos + 4 : pos + 8]
            if pos + 12 + length > len(data):
                break
            payload = data[pos + 8 : pos + 8 + length]
            if kind == b"eXIf":
                info["exif"] = b"Exif\0\0" + payload
            elif kind in (b"tEXt", b"zTXt") and payload.startswith(b"Raw profile type exif\0"):
                text = payload.split(b"\0", 1)[1]
                if kind == b"zTXt":
                    try:
                        text = zlib.decompressobj().decompress(text[1:], 1024 * 1024) if text[:1] == b"\0" else b""
                    except zlib.error:
                        text = b""
                info["Raw profile type exif"] = text.decode("latin-1")
            elif kind == b"iTXt" and payload.startswith(b"XML:com.adobe.xmp\0"):
                parts = payload.split(b"\0", 1)[1]
                if len(parts) >= 2:
                    compressed, method = parts[:2]
                    fields = parts[2:].split(b"\0", 2)
                    if len(fields) == 3:
                        text = fields[2]
                        if compressed == 1 and method == 0:
                            try:
                                # Match Pillow's bounded text decompression.
                                text = zlib.decompressobj().decompress(text, 1024 * 1024)
                            except zlib.error:
                                text = b""
                        info["XML:com.adobe.xmp"] = text.decode("utf-8", "replace")
            pos += length + 12
    elif format == "JXL" and data.startswith(b"\0\0\0\x0cJXL "):
        pos = 0
        while pos + 8 <= len(data):
            length = int.from_bytes(data[pos : pos + 4], "big")
            kind = data[pos + 4 : pos + 8]
            header = 8
            if length == 1:
                if pos + 16 > len(data):
                    break
                length = int.from_bytes(data[pos + 8 : pos + 16], "big")
                header = 16
            elif length == 0:
                length = len(data) - pos
            if length < header or pos + length > len(data):
                break
            payload = data[pos + header : pos + length]
            if kind == b"Exif" and len(payload) >= 4:
                offset = int.from_bytes(payload[:4], "big")
                if offset + 4 < len(payload):
                    info["exif"] = b"Exif\0\0" + payload[4 + offset :]
            elif kind == b"xml ":
                info["xmp"] = payload
            pos += length
    return info
