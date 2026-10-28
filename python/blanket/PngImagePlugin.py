"""PNG text metadata compatible with Pillow's PngInfo API."""

from __future__ import annotations

__lazy_modules__ = {"zlib"}

import zlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Self


class iTXt(str):
    """International text with its language and translated keyword."""

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
        if len(cid) != 4 or not cid.isalpha() or not cid[:1].islower() or not cid[2:3].isupper():
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
