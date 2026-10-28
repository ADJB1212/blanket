"""Load bitmap and TrueType/OpenType fonts without FreeType or Pillow.

Outline fonts use ab_glyph coverage masks and rustybuzz OpenType shaping.
Glyph availability depends on the selected font; system fallback is not used.
"""

from __future__ import annotations

__lazy_modules__ = {"collections.abc"}

import os
import struct
from enum import IntEnum
from math import ceil, floor, isfinite
from operator import index as integer_index
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO

from . import Image
from ._blanket import NativeFont, _font_stroke
from ._draw_font import BitmapFont, BitmapMask

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["MAX_STRING_LENGTH", "ImageFont", "Layout", "TrueTypeFont", "load", "load_default", "truetype"]

# Bound shaping work for a single text request.
MAX_STRING_LENGTH: int | None = 1_000_000
# Bound mask allocation and stroke expansion in both font implementations.
_MAX_MASK_PIXELS = 100_000_000
_MAX_GLYPH_CACHE_PIXELS = 1024 << 10
_MAX_STROKE_WIDTH = 1024
# Match the native font scale limit before passing values to Rust.
_MAX_FONT_SIZE = 1_000_000
_BINARY_TABLE = bytes(255 if value >= 128 else 0 for value in range(256))


class Layout(IntEnum):
    """Layout engines. Both values use Blanket's native OpenType shaper."""

    BASIC = 0
    RAQM = 1


def _text(text: str | bytes) -> str:
    """Decode UTF-8 text and enforce single-line and length limits."""
    if isinstance(text, bytes):
        text = text.decode("utf-8")
    if not isinstance(text, str):
        raise TypeError("text must be a string or UTF-8 bytes")
    if MAX_STRING_LENGTH is not None and len(text) > MAX_STRING_LENGTH:
        raise ValueError("text exceeds MAX_STRING_LENGTH")
    if "\n" in text or "\r" in text:
        raise ValueError("font methods require a single line")
    return text


def _stroke_width(value: float) -> int:
    """Validate a stroke width and round it up to whole pixels."""
    if not isfinite(value) or value < 0 or value > _MAX_STROKE_WIDTH:
        raise ValueError("stroke_width must be finite and between 0 and 1024")
    return ceil(value)


class ImageFont:
    """Bitmap font returned by :func:`load` and :func:`load_default`.

    The built-in fallback retains the fixed-width 5 by 7 glyphs used by
    ImageDraw. Loaded PIL fonts contain up to 256 Latin-1 glyphs.
    """

    def __init__(self, size: float | None = None) -> None:
        """Create the built-in bitmap fallback, optionally scaled to a pixel size."""
        self._fallback = BitmapFont(size)
        self._atlas: Image.Image | None = None
        self._glyphs: list[tuple[int, ...]] = []
        self._ascent = 7 * self._fallback.scale
        self._descent = 0
        self.size = self._ascent

    def getmetrics(self) -> tuple[int, int]:
        """Return the ascent and positive descent in pixels."""
        return self._ascent, self._descent

    def _layout(
        self, text: str, direction: str | None, features: Sequence[str] | None, language: str | None, render: bool, start: tuple[float, float] = (0, 0)
    ) -> tuple[float, tuple[int, int, int, int], bytes]:
        """Measure bitmap text and optionally render its coverage mask."""
        del start
        if direction not in (None, "ltr") or features or language:
            raise ValueError("bitmap fonts support only unshaped left-to-right text")
        if self._atlas is None:
            width = int(self._fallback.getlength(text))
            height = self._ascent if text else 0
            data = bytes(self._fallback.getmask2(text)[0]) if render else b""
            return float(width), (0, -height, width, 0), data
        encoded = text.encode("latin-1")
        width = max(0, sum(self._glyphs[char][0] for char in encoded))
        height = self._ascent + self._descent
        if width * height > _MAX_MASK_PIXELS:
            raise ValueError("font mask is too large")
        data = b""
        if render:
            mask = Image.new("L", (width, height))
            glyphs: dict[int, Image.Image] = {}
            cache_remaining = _MAX_GLYPH_CACHE_PIXELS
            x, y = 0, self._ascent
            for char in encoded:
                dx, dy, x0, y0, _, _, sx0, sy0, sx1, sy1 = self._glyphs[char]
                glyph = glyphs.get(char)
                if glyph is None:
                    glyph = self._atlas.crop((sx0, sy0, sx1, sy1))
                    pixels = (sx1 - sx0) * (sy1 - sy0)
                    if pixels <= cache_remaining:
                        glyphs[char] = glyph
                        cache_remaining -= pixels
                mask.paste(glyph, (x + x0, y + y0))
                x += dx
                y += dy
            data = mask.tobytes()
        return float(width), (0, -self._ascent, width, self._descent), data

    def _anchor(self, bounds: tuple[int, int, int, int], advance: float, anchor: str | None, direction: str | None) -> tuple[int, int]:
        """Resolve a text anchor to integer offsets from the layout origin."""
        left, top, right, bottom = bounds
        if direction == "ttb":
            anchor = "lt" if anchor is None else anchor
            if len(anchor) != 2 or anchor[0] not in "lmrs" or anchor[1] not in "tmb":
                raise ValueError("invalid vertical text anchor")
            x = {"l": -left, "m": -(left + right) / 2, "r": -right, "s": 0}[anchor[0]]
            y = {"t": -top, "m": -advance / 2, "b": -advance}[anchor[1]]
        else:
            anchor = "la" if anchor is None else anchor
            if len(anchor) != 2 or anchor[0] not in "lmr" or anchor[1] not in "atmsbd":
                raise ValueError("invalid text anchor")
            x = {"l": 0, "m": -advance / 2, "r": -advance}[anchor[0]]
            y = {"a": self._ascent, "t": -top, "m": (self._ascent - self._descent) / 2, "s": 0, "b": -bottom, "d": -self._descent}[anchor[1]]
        return floor(x + 0.5), floor(y + 0.5)

    def getlength(self, text: str | bytes, mode: str = "", direction: str | None = None, features: Sequence[str] | None = None, language: str | None = None) -> float:
        """Return the shaped advance in pixels, including kerning.

        `direction` accepts `ltr`, `rtl`, or `ttb` for outline fonts. OpenType
        feature strings include `kern`, `-kern`, and `liga=0`. `language` is a
        BCP 47 language tag. Bitmap fonts accept only left-to-right layout.
        The optional `mode` is accepted for Pillow compatibility.
        """
        del mode
        return self._layout(_text(text), direction, features, language, False)[0]

    def getbbox(
        self, text: str | bytes, mode: str = "", direction: str | None = None, features: Sequence[str] | None = None, language: str | None = None, stroke_width: float = 0, anchor: str | None = None
    ) -> tuple[int, int, int, int]:
        """Return anchored bounds, expanded by the rounded-up stroke width.

        Horizontal anchors combine `l`, `m`, or `r` with `a` (ascender), `t`
        (ink top), `m` (metrics middle), `s` (baseline), `b` (ink bottom), or
        `d` (descender). Vertical anchors use `l/m/r/s` and `t/m/b`.
        """
        del mode
        stroke = _stroke_width(stroke_width)
        advance, bounds, _ = self._layout(_text(text), direction, features, language, False)
        x, y = self._anchor(bounds, advance, anchor, direction)
        return bounds[0] + x - stroke, bounds[1] + y - stroke, bounds[2] + x + stroke, bounds[3] + y + stroke

    def getmask2(
        self,
        text: str | bytes,
        mode: str = "L",
        direction: str | None = None,
        features: Sequence[str] | None = None,
        language: str | None = None,
        stroke_width: float = 0,
        anchor: str | None = None,
        ink: int = 0,
        start: tuple[float, float] | None = None,
        *,
        stroke_filled: bool = True,
    ) -> tuple[BitmapMask, tuple[int, int]]:
        """Return a coverage mask and its offset relative to the text anchor.

        `mode="1"` thresholds coverage; `L` preserves antialiasing. Strokes
        expand the filled mask. `start` supplies fractional pixel offsets for
        outline fonts. `ink` and `stroke_filled` are compatibility arguments.
        """
        del ink, stroke_filled
        if mode not in ("", "1", "L"):
            raise ValueError("font mask mode must be 1 or L")
        stroke = _stroke_width(stroke_width)
        advance, bounds, data = self._layout(_text(text), direction, features, language, True, (0, 0) if start is None else start)
        x, y = self._anchor(bounds, advance, anchor, direction)
        width, height = bounds[2] - bounds[0], bounds[3] - bounds[1]
        if stroke:
            data = _font_stroke(data, (width, height), stroke)
            width += 2 * stroke
            height += 2 * stroke
        if mode == "1":
            data = data.translate(_BINARY_TABLE)
        return BitmapMask((width, height), data), (bounds[0] + x - stroke, bounds[1] + y - stroke)

    def getmask(
        self, text: str | bytes, mode: str = "L", direction: str | None = None, features: Sequence[str] | None = None, language: str | None = None, stroke_width: float = 0, anchor: str | None = None
    ) -> BitmapMask:
        """Return the font coverage mask; see :meth:`getmask2` for options."""
        return self.getmask2(text, mode, direction, features, language, stroke_width, anchor)[0]


class TrueTypeFont(ImageFont):
    """A TrueType/OpenType face rasterized and shaped in Rust.

    Fonts can be filesystem paths or binary file objects. Font collections
    select a face with `index`. Sizes use pixels per em; rasterization and
    hinting differ from Pillow/FreeType. Missing glyphs use the font's .notdef.
    """

    def __init__(self, font: str | bytes | os.PathLike[str] | BinaryIO, size: float = 10, index: int = 0, encoding: str = "", layout_engine: Layout | int | None = None) -> None:
        """Load a font face from a path or binary stream at the given pixels per em.

        ``index`` selects a collection face. Only Unicode encoding is supported;
        both layout engines use the native OpenType shaper.
        """
        if not isfinite(size) or size <= 0 or size > _MAX_FONT_SIZE:
            raise ValueError("font size must be finite, positive, and at most 1000000")
        index = integer_index(index)
        if index < 0 or index > 0xFFFFFFFF:
            raise ValueError("font index must be a nonnegative 32-bit integer")
        if encoding not in ("", "unic"):
            raise ValueError("only Unicode font encoding is supported")
        self.layout_engine = Layout.RAQM if layout_engine is None else Layout(layout_engine)
        self.path = font
        self.size = size
        self.index = index
        self.encoding = encoding
        if isinstance(font, (str, bytes, os.PathLike)):
            self.font_bytes = Path(os.fsdecode(font)).read_bytes()
        else:
            self.font_bytes = font.read()
        self._native = NativeFont(self.font_bytes, index)
        ascent, descent = self._native.metrics(size)
        self._ascent, self._descent = ceil(ascent), ceil(descent)

    def _layout(
        self, text: str, direction: str | None, features: Sequence[str] | None, language: str | None, render: bool, start: tuple[float, float] = (0, 0)
    ) -> tuple[float, tuple[int, int, int, int], bytes]:
        """Shape outline text and optionally rasterize it through the native font."""
        return self._native.layout(text, self.size, direction, [] if features is None else list(features), language, render, start)

    def getname(self) -> tuple[str, str]:
        """Return the font family and style names when present."""
        return self._native.name()

    def getlength(self, text: str | bytes, mode: str = "", direction: str | None = None, features: Sequence[str] | None = None, language: str | None = None) -> float:
        """Return the shaped advance without constructing glyph outlines."""
        del mode
        return self._native.layout(_text(text), self.size, direction, [] if features is None else list(features), language, False, (0, 0), True)[0]

    def font_variant(
        self, font: str | bytes | os.PathLike[str] | BinaryIO | None = None, size: float | None = None, index: int | None = None, encoding: str | None = None, layout_engine: Layout | int | None = None
    ) -> TrueTypeFont:
        """Create another face, reusing the loaded bytes unless `font` is set."""
        from io import BytesIO

        return TrueTypeFont(
            BytesIO(self.font_bytes) if font is None else font,
            self.size if size is None else size,
            self.index if index is None else index,
            self.encoding if encoding is None else encoding,
            self.layout_engine if layout_engine is None else layout_engine,
        )


def truetype(font: str | bytes | os.PathLike[str] | BinaryIO, size: float = 10, index: int = 0, encoding: str = "", layout_engine: Layout | int | None = None) -> TrueTypeFont:
    """Load a .ttf/.otf file or a collection face from a path or binary stream.

    The path is used directly; system font discovery is not performed. Both
    `Layout` values select the native shaper and require no system RAQM library.
    """
    return TrueTypeFont(font, size, index, encoding, layout_engine)


def load_default(size: float | None = None) -> ImageFont:
    """Return the built-in bitmap fallback, optionally scaled in multiples of seven."""
    return ImageFont(size)


def load(filename: str | bytes | os.PathLike[str]) -> ImageFont:
    """Load a legacy PIL bitmap font and its PNG, GIF, or binary PBM atlas.

    The atlas shares the font's basename. Glyphs use Latin-1 encoding. Invalid
    or truncated files raise `OSError`; the font requires no Pillow installation.
    """
    path = Path(os.fsdecode(filename))
    with path.open("rb") as stream:
        if stream.readline() != b"PILfont\n":
            raise OSError("not a PIL bitmap font")
        while True:
            line = stream.readline(65536)
            if not line:
                raise OSError("missing PIL font data")
            if line == b"DATA\n":
                break
        data = stream.read(256 * 20)
    if len(data) != 256 * 20:
        raise OSError("truncated PIL font metrics")
    atlas = None
    for suffix in (".png", ".gif", ".pbm"):
        candidate = path.with_suffix(suffix)
        if candidate.is_file():
            with candidate.open("rb") as stream:
                magic = stream.read(2)
            if magic == b"P4":
                atlas = _load_pbm(candidate)
            else:
                with Image.open(candidate) as image:
                    atlas = image.convert("L")
            break
    if atlas is None:
        raise OSError("PIL font atlas not found")
    font = ImageFont()
    font._atlas = atlas
    font._glyphs = list(struct.iter_unpack(">10h", data))
    font._ascent = 0
    font._descent = 0
    for glyph in font._glyphs:
        _, _, x0, y0, x1, y1, sx0, sy0, sx1, sy1 = glyph
        if not (0 <= sx0 <= sx1 <= atlas.width and 0 <= sy0 <= sy1 <= atlas.height) or (x1 - x0, y1 - y0) != (sx1 - sx0, sy1 - sy0):
            raise OSError("invalid PIL font glyph bounds")
        font._ascent = max(font._ascent, -y0)
        font._descent = max(font._descent, y1)
    font.size = font._ascent + font._descent
    return font


def _load_pbm(path: Path) -> Image.Image:
    """Load a binary PBM font atlas as an L coverage image."""
    with path.open("rb") as stream:
        if stream.readline().strip() != b"P4":
            raise OSError("font atlas must be a binary PBM")
        tokens: list[bytes] = []
        while len(tokens) < 2:
            line = stream.readline(65536)
            if not line:
                raise OSError("truncated PBM font atlas")
            tokens.extend(line.split(b"#", 1)[0].split())
        if len(tokens) != 2:
            raise OSError("invalid PBM font atlas dimensions")
        try:
            width, height = map(int, tokens)
        except ValueError as exc:
            raise OSError("invalid PBM font atlas dimensions") from exc
        if width <= 0 or height <= 0 or width * height > _MAX_MASK_PIXELS:
            raise OSError("invalid PBM font atlas size")
        length = (width + 7) // 8 * height
        data = stream.read(length)
    if len(data) != length:
        raise OSError("truncated PBM font atlas")
    return Image.frombytes("1", (width, height), data).convert("L").point([255 - value for value in range(256)])
