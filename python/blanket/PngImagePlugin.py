"""PNG text metadata compatible with Pillow's PngInfo API."""

from __future__ import annotations

__lazy_modules__ = {"zlib", "struct"}

import struct
import zlib
from enum import IntEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Self

    from ._typing import Writable

from ._io import write_all
from ._plugin import PluginImage

_MAGIC = b"\x89PNG\r\n\x1a\n"
_SYNTAX_ERROR = SyntaxError


def sanitize_apng(data: bytes) -> tuple[bytes, BaseException | None, int | None]:
    """Strip invalid APNG control chunks and classify deferred frame errors."""
    if not data.startswith(_MAGIC):
        return data, None, None
    for pos in _chunk_offsets(data):
        if data.startswith(b"acTL", pos + 4):
            break
        if data.startswith((b"IDAT", b"IEND"), pos + 4):
            return data, None, None
    else:
        return data, None, None
    pos = 8
    view = memoryview(data)
    chunks: list[tuple[int, int, bytes, memoryview]] = []
    actl_indices: list[int] = []
    idat_seen = False
    while pos + 12 <= len(data):
        length = int.from_bytes(view[pos : pos + 4], "big")
        if pos + 12 + length > len(data):
            return data, None, None
        cid = bytes(view[pos + 4 : pos + 8])
        payload = view[pos + 8 : pos + 8 + length]
        if cid == b"acTL" and not idat_seen:
            actl_indices.append(len(chunks))
        elif cid == b"IDAT":
            idat_seen = True
        chunks.append((pos, pos + 12 + length, cid, payload))
        pos += 12 + length
        if cid == b"IEND":
            break
    if not actl_indices:
        return data, None, None
    actl_payload = chunks[actl_indices[0]][3]
    if len(actl_payload) < 8:
        return data, None, None
    num_frames = int.from_bytes(actl_payload[:4], "big")
    if len(actl_indices) > 1 or num_frames == 0 or num_frames > 0x80000000:
        import warnings

        warnings.warn("Invalid APNG, will use default PNG image if possible", UserWarning, stacklevel=3)
        output = bytearray(_MAGIC)
        for start, end, cid, _ in chunks:
            if cid == b"IEND" and not idat_seen:
                ihdr = chunks[0][3]
                width, height = int.from_bytes(ihdr[:4], "big"), int.from_bytes(ihdr[4:8], "big")
                putchunk(_Buffer(output), b"IDAT", zlib.compress(bytes((width * 8 + 1) * height)))
            if cid not in (b"acTL", b"fcTL", b"fdAT"):
                output.extend(view[start:end])
        return bytes(output), None if idat_seen else OSError("IDAT or fdAT chunk is missing"), None
    expected_seq = 0
    fctl_count = 0
    need_data = False
    has_idat = False
    has_fctl_before_idat = False
    for index, (_, _, cid, payload) in enumerate(chunks):
        if cid == b"fcTL":
            if index < actl_indices[0] and not has_idat:
                return data, None, None
            if not has_idat:
                has_fctl_before_idat = True
            if need_data:
                return bytes(_strip_apng(view, chunks)), _SYNTAX_ERROR("APNG missing frame data"), num_frames + int(has_idat and not has_fctl_before_idat)
            if len(payload) < 26 or int.from_bytes(payload[:4], "big") != expected_seq:
                return bytes(_strip_apng(view, chunks)), _SYNTAX_ERROR("APNG contains frame sequence errors"), num_frames + int(has_idat and not has_fctl_before_idat)
            expected_seq += 1
            fctl_count += 1
            need_data = True
        elif cid == b"IDAT":
            has_idat = True
            need_data = False
        elif cid == b"fdAT":
            if not fctl_count or len(payload) < 4 or int.from_bytes(payload[:4], "big") != expected_seq:
                return bytes(_strip_apng(view, chunks)), _SYNTAX_ERROR("APNG contains frame sequence errors"), num_frames + int(has_idat and not has_fctl_before_idat)
            expected_seq += 1
            need_data = False
    total_frames = num_frames + int(has_idat and not has_fctl_before_idat)
    ihdr = chunks[0][3]
    if len(ihdr) >= 8 and int.from_bytes(ihdr[:4], "big") * int.from_bytes(ihdr[4:8], "big") * total_frames > 178_956_970:
        return data, None, None
    if need_data or fctl_count == 0:
        return bytes(_strip_apng(view, chunks)), _SYNTAX_ERROR("APNG missing frame data"), total_frames
    if fctl_count < num_frames:
        return bytes(_strip_apng(view, chunks)), OSError("no more images in APNG file"), total_frames
    if fctl_count > num_frames and has_idat:
        output = bytearray(_MAGIC)
        for start, end, cid, payload in chunks:
            if cid == b"acTL":
                putchunk(output_stream := _Buffer(output), b"acTL", o32(fctl_count), bytes(payload[4:8]))
                del output_stream
            else:
                output.extend(view[start:end])
        return bytes(output), None, None
    return data, None, None


def _chunk_offsets(data: bytes) -> Iterator[int]:
    """Yield complete PNG chunks by offset."""
    pos = 8
    while pos + 12 <= len(data):
        end = pos + 12 + struct.unpack_from(">I", data, pos)[0]
        if end > len(data):
            return
        yield pos
        pos = end


class _Buffer:
    def __init__(self, target: bytearray) -> None:
        """Initialize the buffer wrapper around a bytearray."""
        self.target = target

    def write(self, data: bytes) -> int:
        """Append data to the bytearray target and return bytes written."""
        self.target.extend(data)
        return len(data)


def _strip_apng(view: memoryview, chunks: list[tuple[int, int, bytes, memoryview]]) -> bytearray:
    """Strip APNG animation chunks to produce a static PNG."""
    output = bytearray(_MAGIC)
    seen_idat = False
    for start, end, cid, _ in chunks:
        if cid == b"IDAT":
            seen_idat = True
            output.extend(view[start:end])
        elif (cid not in (b"acTL", b"fcTL", b"fdAT") or (not seen_idat and cid == b"fcTL")) and cid != b"fcTL":
            output.extend(view[start:end])
    return output


class PngImageFile(PluginImage):
    """Open a PNG or APNG using the native decoder."""

    _format = "PNG"


class Disposal(IntEnum):
    """Specify how to clear an APNG frame before rendering the next frame."""

    OP_NONE = 0
    OP_BACKGROUND = 1
    OP_PREVIOUS = 2


class Blend(IntEnum):
    """Specify how to combine an APNG frame with the existing canvas."""

    OP_SOURCE = 0
    OP_OVER = 1


def o16(value: int) -> bytes:
    """Pack an integer as a 16-bit unsigned big-endian bytes object.

    Args:
        value: Integer value in range 0 to 65535.

    Examples:
        ```python
        from blanket.PngImagePlugin import o16

        packed = o16(42)
        ```
    """
    return struct.pack(">H", value)


def o32(value: int) -> bytes:
    """Pack an integer as a 32-bit unsigned big-endian bytes object.

    Args:
        value: Integer value in range 0 to 4294967295.

    Examples:
        ```python
        from blanket.PngImagePlugin import o32

        packed = o32(1000)
        ```
    """
    return struct.pack(">I", value)


def putchunk(fp: Writable, cid: bytes, *data: bytes) -> None:
    """Write a PNG chunk with its length and checksum."""
    payload = b"".join(data)
    checksum = zlib.crc32(payload, zlib.crc32(cid))
    write_all(fp, o32(len(payload)) + cid + payload + o32(checksum))


class iTXt(str):
    """International text with its language and translated keyword."""

    lang: str
    tkey: str

    def __new__(cls, text: str, lang: str = "", tkey: str = "") -> Self:
        """Create a text value carrying a language tag and translated keyword."""
        result = super().__new__(cls, text)
        result.lang, result.tkey = lang, tkey
        return result


class PngInfo:
    """Container for ancillary PNG chunks passed to ``save(pnginfo=...)``."""

    def __init__(self) -> None:
        """Create an empty collection of ancillary PNG chunks."""
        self.chunks: list[tuple[bytes, bytes, bool]] = []

    def add(self, cid: bytes, data: bytes, after_idat: bool = False) -> None:
        """Append an ancillary chunk, optionally after the pixel data."""
        if cid not in (b"tEXt", b"zTXt", b"iTXt") and (len(cid) != 4 or not cid.isalpha() or not cid[:1].islower() or not cid[2:3].isupper()):
            raise ValueError("PNG chunk must have a valid ancillary identifier")
        self.chunks.append((cid, bytes(data), bool(after_idat)))

    def add_text(self, key: str | bytes, value: str | bytes, zip: bool = False) -> None:
        """Add Latin-1 text, using international text for Unicode values."""
        if isinstance(value, iTXt):
            self.add_itxt(key, value, value.lang, value.tkey, zip)
            return
        try:
            raw = value.encode("latin-1") if isinstance(value, str) else bytes(value)
        except UnicodeEncodeError:
            self.add_itxt(key, value, zip=zip)
            return
        keyword = self._keyword(key)
        self.add(b"zTXt" if zip else b"tEXt", keyword + b"\0" + (b"\0" + zlib.compress(raw) if zip else raw))

    def add_itxt(self, key: str | bytes, value: str | bytes, lang: str = "", tkey: str = "", zip: bool = False) -> None:
        """Add UTF-8 text with optional language and translated keyword."""
        raw = value.encode("utf-8") if isinstance(value, str) else bytes(value)
        fields = self._keyword(key) + b"\0" + bytes((int(zip), 0)) + lang.encode("ascii") + b"\0" + tkey.encode("utf-8") + b"\0"
        self.add(b"iTXt", fields + (zlib.compress(raw) if zip else raw))

    @staticmethod
    def _keyword(key: str | bytes) -> bytes:
        """Validate and encode a PNG text chunk keyword into Latin-1 bytes."""
        raw = key.encode("latin-1") if isinstance(key, str) else bytes(key)
        if not 1 <= len(raw) <= 79 or b"\0" in raw:
            raise ValueError("PNG text keyword must contain 1 to 79 non-null Latin-1 bytes")
        return raw
