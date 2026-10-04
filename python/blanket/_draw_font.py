"""Provide the bitmap fallback and mask adapter used by ImageDraw.

Glyphs use a 5 by 7 grid with one column of advance spacing. Rasterization
runs in the native drawing crate. This fallback performs no font shaping;
lowercase letters use uppercase glyphs and unsupported characters use `?`.
"""

from __future__ import annotations

from math import ceil, isfinite

from ._blanket import raster_font_mask

_GLYPHS = {
    "A": "0e11111f111111",
    "B": "1e11111e11111e",
    "C": "0e11101010110e",
    "D": "1e11111111111e",
    "E": "1f10101e10101f",
    "F": "1f10101e101010",
    "G": "0e11101711110f",
    "H": "1111111f111111",
    "I": "0e04040404040e",
    "J": "0702020212120c",
    "K": "11121418141211",
    "L": "1010101010101f",
    "M": "111b1515111111",
    "N": "11191513111111",
    "O": "0e11111111110e",
    "P": "1e11111e101010",
    "Q": "0e11111115120d",
    "R": "1e11111e141211",
    "S": "0f10100e01011e",
    "T": "1f040404040404",
    "U": "1111111111110e",
    "V": "11111111110a04",
    "W": "11111115151b11",
    "X": "11110a040a1111",
    "Y": "11110a04040404",
    "Z": "1f01020408101f",
    "0": "0e11131519110e",
    "1": "040c040404040e",
    "2": "0e11010204081f",
    "3": "1e01010e01011e",
    "4": "02060a121f0202",
    "5": "1f10101e01011e",
    "6": "0e10101e11110e",
    "7": "1f010204080808",
    "8": "0e11110e11110e",
    "9": "0e11110f01010e",
    " ": "00000000000000",
    "!": "04040404040004",
    '"': "0a0a0a00000000",
    "#": "0a0a1f0a1f0a0a",
    "$": "040f140e051e04",
    "%": "19190204081313",
    "&": "0c12140c15120d",
    "'": "04040800000000",
    "(": "02040808080402",
    ")": "08040202020408",
    "*": "000a041f040a00",
    "+": "0004041f040400",
    ",": "00000000000408",
    "-": "0000001f000000",
    ".": "00000000000004",
    "/": "01010204081010",
    ":": "00000400000400",
    ";": "00000400000408",
    "<": "01020408040201",
    "=": "00001f001f0000",
    ">": "10080402040810",
    "?": "0e110102040004",
    "@": "0e11171517100e",
    "[": "0e08080808080e",
    "\\": "10100804020101",
    "]": "0e02020202020e",
    "^": "040a1100000000",
    "_": "0000000000001f",
    "`": "08040200000000",
    "{": "02040408040402",
    "|": "04040404040404",
    "}": "08040402040408",
    "~": "00000916000000",
}


class BitmapMask:
    """Byte-convertible coverage mask consumed by native text drawing.

    Attributes:
        size: Mask dimensions as `(width, height)` in pixels.
        data: Row-major, one-byte-per-pixel coverage values.
    """

    def __init__(self, size: tuple[int, int], data: bytes) -> None:
        """Store mask dimensions and coverage bytes without copying or validation.

        Args:
            size: Mask width and height in pixels.
            data: Coverage bytes, with zero for background and 255 for ink.
        """
        self.size = size
        self.data = data

    def __bytes__(self) -> bytes:
        """Return the row-major coverage bytes."""
        return self.data


class BitmapFont:
    """Fixed-width bitmap font with integer nearest-neighbor scaling.

    Each character advances six scaled pixels, including a spacing column.
    Glyphs occupy five columns and seven rows. Lowercase letters use uppercase
    glyphs; unsupported characters use `?`.

    Attributes:
        scale: Positive integer enlargement factor for the glyph grid.
    """

    def __init__(self, size: float | None = None) -> None:
        """Choose an integer scale for the requested glyph height.

        Args:
            size: Finite, positive glyph height in pixels, rounded up to
                a multiple of seven. `None` selects the unscaled seven-pixel font.

        Raises:
            ValueError: If size is nonfinite or nonpositive.
        """
        size = 7 if size is None else size
        if not isfinite(size) or size <= 0:
            raise ValueError("font_size must be finite and positive")
        self.scale = max(1, ceil(size / 7))

    def getlength(self, text: str, **_kwargs: object) -> float:
        """Return the fixed advance length of a string.

        Args:
            text: String to measure. Every character occupies one advance cell.
            **_kwargs: Font shaping options accepted for compatibility and ignored.

        Returns:
            Character count multiplied by six scaled pixels, as a float.
        """
        return float(len(text) * 6 * self.scale)

    def getbbox(self, text: str, *, anchor: str | None = None, stroke_width: float = 0, **_kwargs: object) -> tuple[float, float, float, float]:
        """Return anchored bounds for a single line, including advance spacing.

        Args:
            text: String to measure. An empty string has zero width and height
                before stroke expansion.
            anchor: Horizontal `l`, `m`, or `r` followed by vertical `a`, `t`,
                `m`, `b`, `s`, or `d`. Defaults to `"la"`. The fallback treats
                `a` and `t` as the top and `b`, `s`, and `d` as the bottom.
            stroke_width: Amount added to each side of the bounding box.
                The drawing context validates this value before use.
            **_kwargs: Font shaping options accepted for compatibility and ignored.

        Returns:
            `(left, top, right, bottom)` relative to the anchor origin.

        Raises:
            ValueError: If the anchor is invalid.
        """
        width = self.getlength(text)
        height = 7 * self.scale if text else 0
        anchor = "la" if anchor is None else anchor
        if len(anchor) != 2 or anchor[0] not in "lmr" or anchor[1] not in "atmbsd":
            raise ValueError("invalid text anchor")
        x = {"l": 0, "m": -width / 2, "r": -width}[anchor[0]]
        y = {"a": 0, "t": 0, "m": -height / 2, "b": -height, "s": -height, "d": -height}[anchor[1]]
        return x - stroke_width, y - stroke_width, x + width + stroke_width, y + height + stroke_width

    def getmask2(self, text: str, _mode: str = "L", *, anchor: str | None = None, **_kwargs: object) -> tuple[BitmapMask, tuple[float, float]]:
        """Rasterize a single line and return its mask and anchored offset.

        Args:
            text: String to render. Unsupported characters use the `?` glyph.
            _mode: Font mask mode accepted for compatibility. Coverage remains
                binary, with one byte per pixel, in every requested mode.
            anchor: Anchor used to position the mask; see :meth:`getbbox`.
            **_kwargs: Additional font options accepted and ignored. Strokes
                for this font are applied by the native drawing kernel.

        Returns:
            A mask and its `(left, top)` offset relative to the anchor origin.

        Raises:
            ValueError: If the anchor is invalid or the native mask exceeds
                its allocation limit.
        """
        left, top = self.getbbox(text, anchor=anchor)[:2]
        glyphs = [bytes.fromhex(_GLYPHS.get(char.upper(), _GLYPHS["?"])) for char in text]
        width, height, data = raster_font_mask(glyphs, self.scale)
        return BitmapMask((width, height), bytes(data)), (left, top)
