"""Pillow-compatible palette objects, factories, and palette file loading."""

from __future__ import annotations

import math
from array import array
from contextlib import nullcontext
from typing import IO, TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Sequence

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
        """Configure ImagePalette.

        Args:
            mode: Palette channel layout, normally `RGB` or `RGBA`.
            palette: Interleaved channel values, or None for an empty palette.

        Examples:
            ```python
            from blanket import ImagePalette

            palette = ImagePalette.ImagePalette("RGB", [0, 0, 0, 255, 0, 0])
            ```
        """
        self.mode = mode
        self.rawmode: str | None = None
        self.palette = palette or bytearray()
        self.dirty: int | None = None

    @property
    def palette(self) -> Sequence[int] | bytes | bytearray:
        """Mutable interleaved palette entries.

        Examples:
            ```python
            from blanket import ImagePalette

            palette = ImagePalette.wedge()
            print(palette.palette)
            ```
        """
        return self._palette

    @palette.setter
    def palette(self, palette: Sequence[int] | bytes | bytearray) -> None:
        self._palette = palette
        self._colors: dict[tuple[int, ...], int] | None = None

    @property
    def colors(self) -> dict[tuple[int, ...], int]:
        """Map channel tuples to their first palette index.

        Examples:
            ```python
            from blanket import ImagePalette

            palette = ImagePalette.wedge()
            print(palette.colors)
            ```
        """
        if self._colors is None:
            self._colors = palette_colors(self.palette, len(self.mode))
        return self._colors

    @colors.setter
    def colors(self, colors: dict[tuple[int, ...], int]) -> None:
        self._colors = colors

    def copy(self) -> ImagePalette:
        """Copy the palette data and flags, rebuilding color lookup on demand.

        Examples:
            ```python
            from blanket import ImagePalette

            palette = ImagePalette.wedge()
            result = palette.copy()
            ```
        """
        result = ImagePalette(self.mode)
        result.rawmode = self.rawmode
        result.palette = self.palette[:]
        result.dirty = self.dirty
        return result

    def getdata(self) -> tuple[str, Sequence[int] | bytes | bytearray]:
        """Return the raw mode and data, or the mode and serialized entries.

        Examples:
            ```python
            from blanket import ImagePalette

            palette = ImagePalette.wedge()
            mode, entries = palette.getdata()
            ```
        """
        return (self.rawmode, self.palette) if self.rawmode else (self.mode, self.tobytes())

    def tobytes(self) -> bytes:
        """Serialize a non-raw palette to bytes.

        Examples:
            ```python
            from blanket import ImagePalette

            palette = ImagePalette.wedge()
            entries = palette.tobytes()
            ```
        """
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
        """Find or allocate a color, optionally reusing an unused image index.

        Args:
            color: RGB or RGBA tuple matching the palette mode.
            image: Optional indexed image whose unused entries may be reused.

        Examples:
            ```python
            from blanket import ImagePalette

            palette = ImagePalette.wedge()
            palette = ImagePalette.ImagePalette("RGB")
            index = palette.getcolor((255, 0, 0))
            ```
        """
        if self.rawmode:
            raise ValueError("palette contains raw palette data")
        if not isinstance(color, tuple):
            raise ValueError(f"unknown color specifier: {color!r}")
        if self.mode == "RGB" and len(color) == 4:
            if color[3] != 255:
                raise ValueError("cannot add non-opaque RGBA color to RGB palette")
            color = color[:3]
        elif self.mode == "RGBA" and len(color) == 3:
            color += (255,)
        colors = self._colors
        if colors is None:
            colors = self.colors
        try:
            return colors[color]
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
        """Write a 256-entry text palette to a filename or text stream.

        Args:
            fp: Filename or writable text stream.

        Examples:
            ```python
            from blanket import ImagePalette

            palette = ImagePalette.wedge()
            from io import StringIO

            output = StringIO()
            palette.save(output)
            ```
        """
        self._check_raw()
        channels = len(self.mode)
        palette = self.palette
        # Pad once, rather than repeatedly checking bounds and properties for
        # each component of every output entry.
        values = list(palette[: 256 * channels])
        values.extend([0] * (256 * channels - len(values)))
        with open(fp, "w") if isinstance(fp, str) else nullcontext(fp) as stream:
            stream.write(f"# Palette\n# Mode: {self.mode}\n")
            for slot in range(256):
                offset = slot * channels
                stream.write(" ".join(map(str, (slot, *values[offset : offset + channels]))) + "\n")


def raw(rawmode: str, data: Sequence[int] | bytes | bytearray) -> ImagePalette:
    """Wrap encoded palette data without interpreting its layout.

    Args:
        rawmode: Channel layout of the palette data, normally `RGB` or `RGBA`.
        data: Raw encoded palette entries.

    Examples:
        ```python
        from blanket import ImagePalette

        result = ImagePalette.raw("RGB", bytes([0, 0, 0, 255, 0, 0]))
        ```
    """
    result = ImagePalette()
    result.rawmode = rawmode
    result.palette = data
    result.dirty = 1
    return result


def make_linear_lut(black: int, white: float) -> list[int]:
    """Return a 256-entry linear intensity lookup table.

    Args:
        black: Black endpoint; only 0 is supported.
        white: White endpoint of the linear ramp.

    Examples:
        ```python
        from blanket import ImagePalette

        result = ImagePalette.make_linear_lut(0, 255)
        ```
    """
    if black != 0:
        raise NotImplementedError("unavailable when black is non-zero")
    if isinstance(white, int) and -(1 << 45) <= white <= 1 << 45:
        return palette_linear(white)
    return [int(white * value // 255) for value in range(256)]


def make_gamma_lut(exp: float) -> list[int]:
    """Return a 256-entry gamma correction lookup table.

    Args:
        exp: Gamma exponent applied to normalized intensity values.

    Examples:
        ```python
        from blanket import ImagePalette

        result = ImagePalette.make_gamma_lut(2.2)
        ```
    """
    if isinstance(exp, (int, float)) and 0 <= exp <= 1e300 and math.isfinite(exp):
        return palette_gamma(exp)
    return [int((value / 255.0) ** exp * 255.0 + 0.5) for value in range(256)]


def negative(mode: str = "RGB") -> ImagePalette:
    """Create a descending grayscale palette.

    Args:
        mode: Palette channel layout, normally `RGB` or `RGBA`.

    Examples:
        ```python
        from blanket import ImagePalette

        result = ImagePalette.negative()
        ```
    """
    return ImagePalette(mode, palette_ramp(len(mode), True))


def random(mode: str = "RGB") -> ImagePalette:
    """Create a palette with 256 random colors.

    Args:
        mode: Palette channel layout, normally `RGB` or `RGBA`.

    Examples:
        ```python
        from blanket import ImagePalette

        result = ImagePalette.random()
        ```
    """
    from random import randint

    return ImagePalette(mode, [randint(0, 255) for _ in range(256 * len(mode))])


def sepia(white: str = "#fff0c0") -> ImagePalette:
    """Create a black-to-sepia RGB palette.

    Args:
        white: CSS color for the lightest palette entry.

    Examples:
        ```python
        from blanket import ImagePalette

        result = ImagePalette.sepia("#fff0c0")
        ```
    """
    channels = _rgb(white)
    if all(-(1 << 45) <= channel <= 1 << 45 for channel in channels):
        return ImagePalette("RGB", palette_sepia(channels[:3]))
    bands = [make_linear_lut(0, channel) for channel in channels]
    return ImagePalette("RGB", [bands[channel][value] for value in range(256) for channel in range(3)])


def wedge(mode: str = "RGB") -> ImagePalette:
    """Create an ascending grayscale palette.

    Args:
        mode: Palette channel layout, normally `RGB` or `RGBA`.

    Examples:
        ```python
        from blanket import ImagePalette

        result = ImagePalette.wedge()
        ```
    """
    return ImagePalette(mode, palette_ramp(len(mode), False))


def load(filename: str) -> tuple[bytes, str]:
    """Load a text palette, GIMP palette, or GIMP RGB gradient.

    Args:
        filename: Path to a text palette, GIMP palette, or GIMP gradient.

    Examples:
        ```python
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from blanket import ImagePalette

        with TemporaryDirectory() as directory:
            path = str(Path(directory) / "palette.txt")
            ImagePalette.wedge().save(path)
            entries, mode = ImagePalette.load(path)
        ```
    """
    with open(filename, "rb") as stream:
        return load_palette(stream)
