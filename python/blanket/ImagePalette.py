"""Pillow-compatible palette objects, factories, and palette file loading."""

from __future__ import annotations

import math
from array import array
from collections.abc import Sequence
from contextlib import nullcontext
from typing import IO, Protocol

from ._blanket import ops_histogram, palette_colors, palette_gamma, palette_linear, palette_ramp, palette_sepia
from ._color import _rgb
from ._palette import load_palette
from .Image import Image


class _PaletteImage(Protocol):
    @property
    def info(self) -> dict[object, object]: ...

    def histogram(self) -> list[int]: ...


class ImagePalette:
    """An interleaved color palette, with RGB entries by default."""

    def __init__(self, mode: str = "RGB", palette: Sequence[int] | bytes | bytearray | None = None) -> None:
        self.mode = mode
        self.rawmode: str | None = None
        self.palette = palette or bytearray()
        self.dirty: int | None = None

    @property
    def palette(self) -> Sequence[int] | bytes | bytearray:
        return self._palette

    @palette.setter
    def palette(self, palette: Sequence[int] | bytes | bytearray) -> None:
        self._palette = palette
        self._colors: dict[tuple[int, ...], int] | None = None

    @property
    def colors(self) -> dict[tuple[int, ...], int]:
        if self._colors is None:
            self._colors = palette_colors(self.palette, len(self.mode))
        return self._colors

    @colors.setter
    def colors(self, colors: dict[tuple[int, ...], int]) -> None:
        self._colors = colors

    def copy(self) -> ImagePalette:
        """Copy the palette data and flags, rebuilding color lookup on demand."""
        result = ImagePalette(self.mode)
        result.rawmode = self.rawmode
        result.palette = self.palette[:]
        result.dirty = self.dirty
        return result

    def getdata(self) -> tuple[str, Sequence[int] | bytes | bytearray]:
        """Return the raw mode and data, or the mode and serialized entries."""
        return (self.rawmode, self.palette) if self.rawmode else (self.mode, self.tobytes())

    def tobytes(self) -> bytes:
        """Serialize a non-raw palette to bytes."""
        self._check_raw()
        return self.palette if isinstance(self.palette, bytes) else array("B", self.palette).tobytes()

    tostring = tobytes

    def _check_raw(self) -> None:
        if self.rawmode:
            raise ValueError("palette contains raw palette data")

    def _new_color_index(self, image: Image | _PaletteImage | None = None, e: Exception | None = None) -> int:
        if not isinstance(self.palette, bytearray):
            self._palette = bytearray(self.palette)
        slot = len(self.palette) // len(self.mode)
        reserved = (image.info.get("background"), image.info.get("transparency")) if image else ()
        while slot in reserved:
            slot += 1
        if slot >= 256 and image:
            histogram = ops_histogram(image._native, None) if isinstance(image, Image) else image.histogram()
            for candidate, count in reversed(list(enumerate(histogram))):
                if count == 0 and candidate not in reserved:
                    slot = candidate
                    break
        if slot >= 256:
            raise ValueError("cannot allocate more than 256 colors") from e
        return slot

    def getcolor(self, color: tuple[int, ...], image: Image | _PaletteImage | None = None) -> int:
        """Find or allocate a color, optionally reusing an unused image index."""
        self._check_raw()
        if not isinstance(color, tuple):
            raise ValueError(f"unknown color specifier: {color!r}")
        if self.mode == "RGB" and len(color) == 4:
            if color[3] != 255:
                raise ValueError("cannot add non-opaque RGBA color to RGB palette")
            color = color[:3]
        elif self.mode == "RGBA" and len(color) == 3:
            color += (255,)
        try:
            return self.colors[color]
        except KeyError as error:
            slot = self._new_color_index(image, error)
        self.colors[color] = slot
        assert isinstance(self._palette, bytearray)
        offset = slot * len(self.mode)
        if offset < len(self.palette):
            self._palette = self._palette[:offset] + bytes(color) + self._palette[offset + len(self.mode) :]
        else:
            self._palette += bytes(color)
        self.dirty = 1
        return slot

    def save(self, fp: str | IO[str]) -> None:
        """Write a 256-entry text palette to a filename or text stream."""
        self._check_raw()
        with open(fp, "w") if isinstance(fp, str) else nullcontext(fp) as stream:
            stream.write(f"# Palette\n# Mode: {self.mode}\n")
            for slot in range(256):
                offset = slot * len(self.mode)
                values = [self.palette[i] if i < len(self.palette) else 0 for i in range(offset, offset + len(self.mode))]
                stream.write(str(slot) + "".join(f" {value}" for value in values) + "\n")


def raw(rawmode: str, data: Sequence[int] | bytes | bytearray) -> ImagePalette:
    """Wrap encoded palette data without interpreting its layout."""
    result = ImagePalette()
    result.rawmode = rawmode
    result.palette = data
    result.dirty = 1
    return result


def make_linear_lut(black: int, white: float) -> list[int]:
    if black != 0:
        raise NotImplementedError("unavailable when black is non-zero")
    if isinstance(white, int) and -(1 << 45) <= white <= 1 << 45:
        return palette_linear(white)
    return [int(white * value // 255) for value in range(256)]


def make_gamma_lut(exp: float) -> list[int]:
    if isinstance(exp, (int, float)) and 0 <= exp <= 1e300 and math.isfinite(exp):
        return palette_gamma(exp)
    return [int((value / 255.0) ** exp * 255.0 + 0.5) for value in range(256)]


def negative(mode: str = "RGB") -> ImagePalette:
    return ImagePalette(mode, palette_ramp(len(mode), True))


def random(mode: str = "RGB") -> ImagePalette:
    from random import randint

    return ImagePalette(mode, [randint(0, 255) for _ in range(256 * len(mode))])


def sepia(white: str = "#fff0c0") -> ImagePalette:
    channels = _rgb(white)
    if all(-(1 << 45) <= channel <= 1 << 45 for channel in channels):
        return ImagePalette("RGB", palette_sepia(channels[:3]))
    bands = [make_linear_lut(0, channel) for channel in channels]
    return ImagePalette("RGB", [bands[channel][value] for value in range(256) for channel in range(3)])


def wedge(mode: str = "RGB") -> ImagePalette:
    return ImagePalette(mode, palette_ramp(len(mode), False))


def load(filename: str) -> tuple[bytes, str]:
    """Load a text palette, GIMP palette, or GIMP RGB gradient."""
    with open(filename, "rb") as stream:
        return load_palette(stream)
