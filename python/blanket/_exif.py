"""Metadata parsing and save option adaptation without Pillow."""

from __future__ import annotations

__lazy_modules__ = {"collections.abc", "zlib"}

import contextlib
import re
import struct
import zlib
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Iterator

from ._hdr import HDR_KEYS, decode_chunk, encode_chunk
from ._ifd import Exif
from .PngImagePlugin import PngInfo, iTXt

# Limit expansion of compressed PNG text while reading metadata.
_MAX_TEXT_BYTES = 1024 << 10
_MAX_PNG_METADATA_BYTES = _MAX_TEXT_BYTES << 6
# Classic TIFF identifies its header and IFD0 orientation with these values.
_TIFF_MAGIC = 42
_ORIENTATION_TAG = 0x112
_IFD_ENTRY_BYTES = 12


def _orientation(raw: bytes) -> tuple[int | None, bytes]:
    """Remove an IFD0 orientation entry without relocating referenced data."""
    prefix = b"Exif\0\0" if raw.startswith(b"Exif\0\0") else b""
    data = raw[len(prefix) :]
    if len(data) < 8 or data[:2] not in (b"II", b"MM"):
        return None, raw
    endian = "<" if data[:2] == b"II" else ">"
    try:
        magic, offset = struct.unpack_from(endian + "HI", data, 2)
        if magic != _TIFF_MAGIC:
            return None, raw
        count = struct.unpack_from(endian + "H", data, offset)[0]
        end = offset + 2 + _IFD_ENTRY_BYTES * count
        if end + 4 > len(data):
            return None, raw
        for i in range(count):
            pos = offset + 2 + i * _IFD_ENTRY_BYTES
            tag, kind, length = struct.unpack_from(endian + "HHI", data, pos)
            if tag != _ORIENTATION_TAG or kind not in (3, 4) or length != 1:
                continue
            orientation = struct.unpack_from(endian + ("H" if kind == 3 else "I"), data, pos + 8)[0]
            if orientation not in range(2, 9):
                return orientation, raw
            clean = bytearray(data)
            struct.pack_into(endian + "H", clean, offset, count - 1)
            # Shift the remaining entries and next-IFD pointer within IFD0.
            # The trailing gap keeps every external value/thumbnail offset valid.
            clean[pos : end - 8] = data[pos + _IFD_ENTRY_BYTES : end + 4]
            clean[end - 8 : end + 4] = bytes(_IFD_ENTRY_BYTES)
            return orientation, prefix + bytes(clean)
    except (struct.error, OverflowError):
        pass
    return None, raw


ORIENTATION_RE = re.compile(r'tiff:Orientation="([0-9])"|<tiff:Orientation>([0-9])</tiff:Orientation>')

ORIENTATION_REMOVE_RE = re.compile(r'tiff:Orientation="[0-9]"|<tiff:Orientation>[0-9]</tiff:Orientation>')

ORIENTATION_REMOVE_BYTES_RE = re.compile(ORIENTATION_REMOVE_RE.pattern.encode())


def transpose_metadata(info: dict[object, object]) -> tuple[int, dict[object, object]]:
    """Return the orientation and a metadata copy with applied orientation removed."""
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


def read_metadata(data: bytes, format: str | None, frame: int = 0) -> dict[object, object]:
    """Read metadata independently of pixel decoding."""
    info: dict[object, object] = {}
    if format == "JPEG":
        profiles: dict[int, bytes] = {}
        profile_count = 0
        valid_profile = True
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
            if marker == 0xE1:
                payload = data[pos + 2 : pos + length]
                if payload.startswith(b"Exif\0\0"):
                    info.setdefault("exif", payload)
                elif payload.startswith(b"http://ns.adobe.com/xap/1.0/\0"):
                    info["xmp"] = payload.split(b"\0", 1)[1]
            elif marker == 0xE2 and data[pos + 2 : pos + 14] == b"ICC_PROFILE\0" and length >= 16:
                part, count = data[pos + 14 : pos + 16]
                if part in profiles or not 1 <= part <= count or profile_count not in (0, count):
                    valid_profile = False
                profile_count = count
                profiles[part] = data[pos + 16 : pos + length]
            pos += length
        if valid_profile and profile_count and sorted(profiles) == list(range(1, profile_count + 1)):
            info["icc_profile"] = b"".join(profiles[i] for i in range(1, profile_count + 1))
    elif format == "PNG":
        pos = 8
        remaining = _MAX_PNG_METADATA_BYTES
        while pos + 12 <= len(data):
            length = int.from_bytes(data[pos : pos + 4], "big")
            kind = data[pos + 4 : pos + 8]
            if pos + 12 + length > len(data):
                break
            if kind == b"IEND":
                break
            if kind not in (b"eXIf", b"tEXt", b"zTXt", b"iTXt", b"iCCP", *HDR_KEYS):
                pos += length + 12
                continue
            if kind in HDR_KEYS and length != {b"cICP": 4, b"mDCV": 24, b"cLLI": 8}[kind]:
                pos += length + 12
                continue
            payload = data[pos + 8 : pos + 8 + length]
            if kind in HDR_KEYS:
                value = decode_chunk(kind, payload)
                if value is not None:
                    info[HDR_KEYS[kind]] = value
            elif kind == b"eXIf":
                if length + 6 <= remaining:
                    info["exif"] = b"Exif\0\0" + payload
                    remaining -= length + 6
            else:
                remaining -= _png_metadata(info, kind, payload, remaining)
            if remaining == 0:
                break
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
            if kind not in (b"Exif", b"xml "):
                pos += length
                continue
            payload = data[pos + header : pos + length]
            if kind == b"Exif" and len(payload) >= 4:
                offset = int.from_bytes(payload[:4], "big")
                if offset + 4 < len(payload):
                    info["exif"] = b"Exif\0\0" + payload[4 + offset :]
            elif kind == b"xml ":
                info["xmp"] = payload
            pos += length
    elif format == "WEBP":
        pos = 12
        while pos + 8 <= len(data):
            kind, length = data[pos : pos + 4], int.from_bytes(data[pos + 4 : pos + 8], "little")
            if pos + 8 + length > len(data):
                break
            key = {b"EXIF": "exif", b"XMP ": "xmp", b"ICCP": "icc_profile"}.get(kind)
            if key:
                raw = data[pos + 8 : pos + 8 + length]
                info[key] = b"Exif\0\0" + raw.removeprefix(b"Exif\0\0") if key == "exif" else raw
            pos += 8 + length + (length & 1)
    elif format == "TIFF":
        for page in read_tiff_metadata(data, frame + 1):
            info = page
        return info
    if format in ("HEIF", "JXL"):
        from ._blanket import _read_native_metadata

        with contextlib.suppress(OSError):
            info.update(_read_native_metadata(data, format))
    return info


def read_tiff_metadata(data: bytes, count: int) -> Iterator[dict[object, object]]:
    """Read page metadata in one traversal of the TIFF directory chain."""
    offset = 0
    endian = "<" if data[:2] == b"II" else ">"
    if len(data) >= 8 and data[:2] in (b"II", b"MM"):
        magic, first = struct.unpack_from(endian + "HI", data, 2)
        if magic == _TIFF_MAGIC:
            offset = first
    for _ in range(count):
        info: dict[object, object] = {}
        try:
            exif = Exif()
            exif.endian, exif._raw = endian, data
            exif._data, offset = exif._read_ifd(offset, 0)
            thumbnail_offset = exif.pop(330, None)
            if isinstance(thumbnail_offset, int):
                exif._load_thumbnail(thumbnail_offset)
            for tag in (34665, 34853):
                if tag in exif:
                    exif.get_ifd(tag)
            if 40965 in exif._ifds.get(34665, {}):
                exif.get_ifd(40965)
            for tag, key in ((700, "xmp"), (34675, "icc_profile")):
                value = exif.get(tag)
                if value is not None:
                    info[key] = value if isinstance(value, bytes) else bytes(value if isinstance(value, tuple) else (value,))
            exif._raw, exif._entries, exif._next = b"", {}, 0
            if exif or exif.thumbnail is not None:
                info["exif"] = exif.tobytes()
        except (ValueError, struct.error):
            pass
        yield info


def _decompress(raw: bytes, maximum: int = _MAX_TEXT_BYTES) -> bytes:
    """Decompress a complete PNG metadata stream within the text size limit."""
    decoder = zlib.decompressobj()
    maximum = min(maximum, _MAX_TEXT_BYTES)
    value = decoder.decompress(raw, maximum + 1)
    if len(value) > maximum or not decoder.eof:
        raise ValueError("compressed PNG metadata exceeds limit or is truncated")
    return value


def _png_metadata(info: dict[object, object], kind: bytes, payload: bytes, maximum: int = _MAX_PNG_METADATA_BYTES) -> int:
    """Extract a PNG text or ICC chunk into info, ignoring malformed metadata."""
    try:
        key, raw = payload.split(b"\0", 1)
        if not 1 <= len(key) <= 79 or len(key) + 1 > maximum:
            return 0
        overhead = len(key) + 1
        maximum -= overhead
        keyword = key.decode("latin-1")
        if kind == b"iCCP":
            if raw[:1] == b"\0":
                value = _decompress(raw[1:], maximum)
                info["icc_profile"] = value
                return overhead + len(value)
            return 0
        if kind == b"zTXt":
            if raw[:1] != b"\0":
                return 0
            raw = _decompress(raw[1:], maximum)
        if kind == b"iTXt":
            compressed, method = raw[:2]
            lang, tkey, raw = raw[2:].split(b"\0", 2)
            if compressed not in (0, 1) or method != 0:
                return 0
            fields = len(lang) + len(tkey) + 4
            if fields > maximum:
                return 0
            overhead += fields
            maximum -= fields
            if compressed:
                raw = _decompress(raw, maximum)
            if len(raw) > maximum:
                return 0
            info[keyword] = iTXt(raw.decode("utf-8"), lang.decode("ascii"), tkey.decode("utf-8"))
            if keyword == "XML:com.adobe.xmp":
                info["xmp"] = raw
        else:
            if len(raw) > maximum:
                return 0
            info[keyword] = raw.decode("latin-1")
        return overhead + len(raw)
    except (ValueError, zlib.error):
        return 0


def save_metadata(image: object, format: str, options: dict[str, object]) -> tuple[bytes, bytes, bytes, list[tuple[bytes, bytes, bool]]]:
    """Remove metadata options and resolve source defaults before encoding."""
    policy = options.pop("metadata_policy", None)
    if policy is not None:
        from ._metadata_policy import MetadataPolicy

        if not isinstance(policy, MetadataPolicy):
            raise TypeError("metadata_policy must be a MetadataPolicy or None")
    supported = format in ("JPEG", "PNG", "WEBP", "JXL", "HEIF", "TIFF")
    keys = {"exif", "xmp", "icc_profile", "pnginfo", *HDR_KEYS.values()} & options.keys()
    if format != "PNG" and not keys.isdisjoint(HDR_KEYS.values()):
        raise TypeError("HDR metadata saving is supported only for PNG")
    if keys and not supported:
        raise TypeError(f"metadata saving is not supported for {format}")
    if "pnginfo" in keys and format != "PNG":
        raise TypeError("pnginfo is supported only for PNG")
    if not supported:
        return b"", b"", b"", []
    values = []
    for key in ("exif", "xmp", "icc_profile"):
        value = options.pop(key) if key in options else image._metadata_value(key)
        if key == "exif" and hasattr(value, "tobytes"):
            value = value.tobytes()
        if key == "xmp" and isinstance(value, str):
            value = value.encode("utf-8")
        if value is None:
            value = b""
        if not isinstance(value, (bytes, bytearray, memoryview)):
            raise TypeError(f"{key} must be bytes" + (" or an Exif object" if key == "exif" else ""))
        values.append(bytes(value))
    pnginfo = options.pop("pnginfo", None)
    if pnginfo is not None and not isinstance(pnginfo, PngInfo):
        raise TypeError("pnginfo must be a PngInfo object or None")
    chunks = [] if pnginfo is None else list(pnginfo.chunks)
    if format == "PNG" and pnginfo is None:
        text = PngInfo()
        for key, value in image.info.items():
            if isinstance(key, str) and isinstance(value, str) and key not in ("XML:com.adobe.xmp", "Raw profile type exif"):
                text.add_text(key, value)
        chunks = text.chunks
    if format == "PNG":
        seen = set()
        for kind, payload, after_idat in chunks:
            if kind in HDR_KEYS:
                if kind in seen or after_idat or decode_chunk(kind, payload) is None:
                    raise ValueError("invalid, duplicate, or misplaced HDR PNG chunk")
                seen.add(kind)
        for kind, key in HDR_KEYS.items():
            value = options.pop(key) if key in options else image.info.get(key)
            if value is not None:
                if any(chunk[0] == kind for chunk in chunks):
                    raise ValueError(f"duplicate {key} metadata in pnginfo")
                chunks.append((kind, encode_chunk(key, value), False))
        if any(kind == b"mDCV" for kind, _, _ in chunks) and not any(kind == b"cICP" for kind, _, _ in chunks):
            raise ValueError("mastering_display requires cicp metadata")
    if policy is not None:
        info = policy.apply(dict(zip(("exif", "xmp", "icc_profile"), values, strict=True)))
        values = [cast("bytes", info.get(key, b"")) for key in ("exif", "xmp", "icc_profile")]
        filtered = []
        for kind, payload, after_idat in chunks:
            if kind not in (b"tEXt", b"zTXt", b"iTXt"):
                filtered.append((kind, payload, after_idat))
                continue
            text_info: dict[object, object] = {}
            _png_metadata(text_info, kind, payload)
            retained = policy.apply(text_info)
            if "Raw profile type exif" in retained:
                retained.pop("Raw profile type exif")
            text = PngInfo()
            raw_xmp = retained.get("xmp")
            if isinstance(raw_xmp, bytes) and raw_xmp:
                text.add_itxt("XML:com.adobe.xmp", raw_xmp)
            for key, value in retained.items():
                if isinstance(key, str) and isinstance(value, str) and key != "XML:com.adobe.xmp":
                    text.add_text(key, value)
            filtered.extend((kind, raw, after_idat) for kind, raw, _ in text.chunks)
        chunks = filtered
    return values[0], values[1], values[2], chunks
