"""Classic TIFF IFD parsing and editable EXIF metadata."""

from __future__ import annotations

import struct
from collections.abc import Iterator, Mapping, MutableMapping
from typing import BinaryIO, Self

_FORMATS = {1: "B", 3: "H", 4: "I", 6: "b", 8: "h", 9: "i", 11: "f", 12: "d", 13: "I"}
_SIZES = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8, 13: 4}
_POINTERS = {34665, 34853, 40965}
_TYPES = {
    274: 3,
    282: 5,
    283: 5,
    296: 3,
    513: 4,
    514: 4,
    531: 3,
    33434: 5,
    33437: 5,
    34850: 3,
    34855: 3,
    36864: 7,
    37121: 7,
    37377: 10,
    37378: 5,
    37380: 10,
    37386: 5,
    37500: 7,
    37510: 7,
    40960: 7,
    40961: 3,
}


class IFDRational(float):
    """A TIFF rational, retaining its numerator and denominator."""

    def __new__(cls, numerator: float, denominator: int = 1) -> Self:
        value = super().__new__(cls, numerator / denominator if denominator else float("nan"))
        value.numerator = numerator
        value.denominator = denominator
        return value


class Exif(MutableMapping[int, object]):
    """Editable EXIF tags with lazy access to EXIF, GPS, and interoperability IFDs.

    JPEG thumbnails are available as ``thumbnail`` bytes and through
    ``get_ifd(1)``. Assign JPEG bytes to embed a thumbnail on serialization.
    """

    def __init__(self) -> None:
        self.endian: str | None = None
        self.bigtiff = False
        self._data: dict[int, object] = {}
        self._raw = b""
        self._types: dict[int, dict[int, int]] = {}
        self._entries: dict[int, dict[int, tuple[bytes, object]]] = {}
        self._ifds: dict[int, dict[int, object]] = {}
        self._hidden: dict[int, object] = {}
        self._next = 0
        self.thumbnail: bytes | None = None

    def __getitem__(self, tag: int) -> object:
        return self._data[tag]

    def __setitem__(self, tag: int, value: object) -> None:
        self._data[tag] = value
        self._hidden.pop(tag, None)
        if tag in _POINTERS:
            self._ifds.pop(tag, None)

    def __delitem__(self, tag: int) -> None:
        del self._data[tag]
        self._ifds.pop(tag, None)

    def __iter__(self) -> Iterator[int]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def load(self, data: bytes) -> None:
        """Load an EXIF APP1 payload or a classic TIFF byte stream."""
        self.__init__()
        data = bytes(data).removeprefix(b"Exif\0\0")
        if not data:
            return
        if len(data) < 8 or data[:2] not in (b"II", b"MM"):
            raise ValueError("invalid EXIF header")
        self.endian = "<" if data[:2] == b"II" else ">"
        magic, offset = struct.unpack_from(self.endian + "HI", data, 2)
        if magic != 42:
            raise ValueError("only classic TIFF EXIF is supported")
        self._raw = data
        self._data, self._next = self._read_ifd(offset, 0)
        if self._next:
            self._load_thumbnail(self._next)

    def _load_thumbnail(self, offset: int) -> None:
        ifd, _ = self._read_ifd(offset, 1)
        self._ifds[1] = ifd
        start, length = ifd.get(513), ifd.get(514)
        if isinstance(start, int) and isinstance(length, int) and 0 <= start <= len(self._raw) and 0 <= length <= len(self._raw) - start:
            self.thumbnail = self._raw[start : start + length]

    def load_from_fp(self, fp: BinaryIO, offset: int | None = None) -> None:
        """Read EXIF from a seekable TIFF stream."""
        fp.seek(0)
        self.load(fp.read())
        if offset is not None:
            self._data, self._next = self._read_ifd(offset, 0)

    def _read_ifd(self, offset: int, key: int) -> tuple[dict[int, object], int]:
        data, endian = self._raw, self.endian or ">"
        if offset < 8 or offset + 2 > len(data):
            raise ValueError("invalid EXIF IFD offset")
        count = struct.unpack_from(endian + "H", data, offset)[0]
        end = offset + 2 + count * 12
        if end + 4 > len(data):
            raise ValueError("truncated EXIF IFD")
        result: dict[int, object] = {}
        types: dict[int, int] = {}
        entries: dict[int, tuple[bytes, object]] = {}
        for pos in range(offset + 2, end, 12):
            tag, kind, length, field = struct.unpack_from(endian + "HHII", data, pos)
            size = _SIZES.get(kind, 0) * length
            if not size:
                continue
            start = pos + 8 if size <= 4 else field
            if start > len(data) or size > len(data) - start:
                raise ValueError("truncated EXIF value")
            if kind == 2:
                value = data[start : start + size].rstrip(b"\0").decode("ascii", "replace")
            elif kind == 7 or (kind == 1 and tag in (700, 34675)):
                value = data[start : start + size]
            elif kind in (5, 10):
                fmt = endian + ("II" if kind == 5 else "ii")
                if length == 1:
                    value = IFDRational(*struct.unpack_from(fmt, data, start))
                else:
                    value = tuple(IFDRational(n, d) for n, d in struct.iter_unpack(fmt, memoryview(data)[start : start + size]))
            else:
                values = struct.unpack_from(f"{endian}{length}{_FORMATS[kind]}", data, start)
                value = values[0] if length == 1 else values
            result[tag], types[tag], entries[tag] = value, kind, (data[pos : pos + 12], value)
        self._types[key], self._entries[key] = types, entries
        return result, struct.unpack_from(endian + "I", data, end)[0]

    def get_ifd(self, tag: int) -> dict[int, object]:
        """Return the mutable IFD selected by its pointer tag, or IFD1 (1)."""
        if tag in self._ifds:
            return self._ifds[tag]
        parent = self.get_ifd(34665) if tag == 40965 else self._data
        value = parent.get(tag)
        if isinstance(value, Mapping):
            result = dict(value)
        elif isinstance(value, int) and value:
            result, _ = self._read_ifd(value, tag)
        else:
            result = {}
        self._ifds[tag] = result
        return result

    def hide_offsets(self) -> None:
        """Hide EXIF and GPS pointers while retaining their editable IFDs."""
        for tag in (34665, 34853):
            if tag in self:
                self.get_ifd(tag)
                self._hidden[tag] = self._data.pop(tag)

    def tobytes(self, offset: int = 8) -> bytes:
        """Serialize tags, nested IFDs, and an optional JPEG thumbnail."""
        endian = self.endian or ">"
        output = bytearray(self._raw or ((b"II" if endian == "<" else b"MM") + struct.pack(endian + "HI", 42, offset)))
        if not self._raw:
            output.extend(bytes(max(0, offset - len(output))))

        def append(raw: bytes) -> int:
            if len(output) & 1:
                output.append(0)
            start = len(output)
            output.extend(raw)
            return start

        active: set[int] = set()

        def write(values: Mapping[int, object], key: int, following: int = 0) -> int:
            if id(values) in active:
                raise ValueError("cyclic EXIF IFD")
            active.add(id(values))
            records = []
            previous_entries = self._entries.get(key, {})
            types = self._types.get(key, {})
            for tag, value in sorted(values.items()):
                if tag in _POINTERS:
                    nested = self._ifds.get(tag)
                    if nested is None:
                        nested = value if isinstance(value, Mapping) else self.get_ifd(tag)
                    value = write(nested, tag)
                    kind = 4
                else:
                    previous = previous_entries.get(tag)
                    if previous is not None and value == previous[1]:
                        records.append(previous[0])
                        continue
                    kind = types.get(tag) or _TYPES.get(tag)
                if kind is None:
                    items = value if isinstance(value, (tuple, list)) else (value,)
                    if isinstance(value, str):
                        kind = 2
                    elif isinstance(value, bytes):
                        kind = 7
                    elif any(isinstance(v, float) or (hasattr(v, "denominator") and not isinstance(v, int)) for v in items):
                        kind = 5 if all(float(v) >= 0 for v in items) else 10
                    else:
                        kind = 3 if all(isinstance(v, int) and 0 <= v <= 65535 for v in items) else 9 if any(v < 0 for v in items) else 4
                if kind == 2:
                    raw = str(value).encode("ascii", "replace") + b"\0"
                    count = len(raw)
                elif kind == 7 or (kind == 1 and isinstance(value, bytes)):
                    raw = bytes(value)
                    count = len(raw)
                else:
                    items = value if isinstance(value, (tuple, list)) else (value,)
                    count = len(items)
                    if kind in (5, 10):
                        from fractions import Fraction

                        numbers = []
                        for item in items:
                            rational = item if hasattr(item, "denominator") and not isinstance(item, int) else Fraction(item).limit_denominator(1000000)
                            numbers.extend((int(rational.numerator), int(rational.denominator)))
                        raw = struct.pack(f"{endian}{len(numbers)}{'I' if kind == 5 else 'i'}", *numbers)
                    else:
                        raw = struct.pack(f"{endian}{count}{_FORMATS[kind]}", *items)
                field = raw.ljust(4, b"\0") if len(raw) <= 4 else struct.pack(endian + "I", append(raw))
                records.append(struct.pack(endian + "HHI", tag, kind, count) + field)
            active.remove(id(values))
            if len(records) > 65535:
                raise ValueError("too many EXIF tags")
            return append(struct.pack(endian + "H", len(records)) + b"".join(records) + struct.pack(endian + "I", following))

        following = 0
        if self.thumbnail is not None or self._ifds.get(1):
            thumbnail_ifd = self.get_ifd(1).copy()
            if self.thumbnail is not None:
                thumbnail_ifd.update({259: 6, 513: append(self.thumbnail), 514: len(self.thumbnail)})
            following = write(thumbnail_ifd, 1)
        root = write({**self._hidden, **self._data} if self._hidden else self._data, 0, following)
        struct.pack_into(endian + "I", output, 4, root)
        return b"Exif\0\0" + bytes(output)
