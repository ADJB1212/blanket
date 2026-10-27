"""Pillow-shaped image constructors and the Image type, backed by Blanket's Rust extension.

Supported modes are `1`, `L`, `LA`, `P`, `PA`, `RGB`, `RGBA`, `HSV`, `CMYK`,
`YCbCr`, `LAB`, integer `I`/`I;16` variants, and float `F`/`RGB;F`/`RGBA;F`.
"""

from __future__ import annotations

__lazy_modules__ = {"subprocess", "tempfile", "shlex", "sys", "io", "collections.abc"}

import contextlib
import math
import os
import shlex
import struct
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from enum import IntEnum
from io import BytesIO
from operator import index
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO, Protocol, Self, cast

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from .ImageFilter import Filter
    from .ImagePalette import ImagePalette

from ._blanket import (
    _decode_region,
    _encode,
    _Image,
    _inspect,
    _open_sequence,
    enhance_blend,
    filter_merge,
    fromarray as _native_fromarray,
    frombytes as _native_frombytes,
    image_alpha_composite,
    image_alpha_composite_inplace,
    image_new,
    image_paste,
    image_putalpha,
    ops_affine,
    ops_canvas,
    ops_color_composite,
    ops_color_resize,
    ops_entropy,
    ops_histogram,
    ops_lut,
    ops_reduce,
    ops_remap_indices,
    ops_resize,
    ops_split,
    ops_spread,
    ops_transpose,
    ops_warp,
    ops_xbm_bits,
    quantize as _native_quantize,
)
from ._color import color_pixel
from ._exif import ORIENTATION_RE, read_metadata, read_tiff_metadata, save_metadata
from ._ifd import Exif
from ._io import write_all
from .Compressor import LosslessImageCompressor, LossyImageCompressor
from .ImageFilter import ModeFilter, MultibandFilter, RankFilter

_EXTENSIONS = {
    ".pcx": "PCX",
    ".sgi": "SGI",
    ".rgb": "SGI",
    ".rgba": "SGI",
    ".bw": "SGI",
    ".xbm": "XBM",
    ".xpm": "XPM",
    ".msp": "MSP",
    ".palm": "PALM",
    ".bmp": "BMP",
    ".gif": "GIF",
    ".ico": "ICO",
    ".png": "PNG",
    ".apng": "PNG",
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".jxl": "JXL",
    ".tif": "TIFF",
    ".tiff": "TIFF",
    ".webp": "WEBP",
    ".heic": "HEIF",
    ".heif": "HEIF",
    ".avif": "AVIF",
    ".pdf": "PDF",
    ".ppm": "PPM",
    ".pgm": "PPM",
    ".pbm": "PPM",
    ".pnm": "PPM",
    ".pam": "PAM",
    ".tga": "TGA",
    ".qoi": "QOI",
    ".jp2": "JPEG2000",
    ".j2k": "JPEG2000",
    ".j2c": "JPEG2000",
    ".jpc": "JPEG2000",
    ".jpf": "JPEG2000",
    ".jpx": "JPEG2000",
}
_BANDS = {
    "RGB;F": ("R", "G", "B"),
    "RGBA;F": ("R", "G", "B", "A"),
    "1": ("1",),
    "L": ("L",),
    "I": ("I",),
    "F": ("F",),
    "I;16": ("I",),
    "I;16L": ("I",),
    "I;16B": ("I",),
    "LA": ("L", "A"),
    "P": ("P",),
    "PA": ("P", "A"),
    "RGB": ("R", "G", "B"),
    "RGBA": ("R", "G", "B", "A"),
    "HSV": ("H", "S", "V"),
    "CMYK": ("C", "M", "Y", "K"),
    "YCbCr": ("Y", "Cb", "Cr"),
    "LAB": ("L", "A", "B"),
}
_GRAY_PALETTE = bytes(value for value in range(256) for _ in range(3))
_RESIZE_SUPPORT = {1: 3, 2: 1, 3: 2, 4: 0.5, 5: 1}
_FORMAT_ALIASES = {
    "JPG": "JPEG",
    "JPEGXL": "JXL",
    "TIF": "TIFF",
    "HEIC": "HEIF",
    "PNM": "PPM",
    "PBM": "PPM",
    "PGM": "PPM",
    "TARGA": "TGA",
    "JP2": "JPEG2000",
    "J2K": "JPEG2000",
    "J2C": "JPEG2000",
    "JPC": "JPEG2000",
}
_SAVE_OPTIONS = {
    "PCX": frozenset(),
    "SGI": frozenset(),
    "XBM": frozenset(),
    "XPM": frozenset(),
    "MSP": frozenset(),
    "PALM": frozenset(),
    "PPM": frozenset(),
    "PAM": frozenset(),
    "TGA": frozenset(),
    "QOI": frozenset(),
    "JPEG2000": frozenset(),
    "BMP": frozenset(),
    "GIF": frozenset(),
    "ICO": frozenset(),
    "PNG": frozenset({"compress_level"}),
    "JPEG": frozenset({"quality"}),
    "JXL": frozenset({"quality", "lossless", "effort"}),
    "TIFF": frozenset({"compression", "quality"}),
    "PDF": frozenset(),
    "WEBP": frozenset({"quality", "lossless"}),
    "HEIF": frozenset({"quality", "lossless"}),
    "AVIF": frozenset({"quality", "effort"}),
}


class Resampling(IntEnum):
    """Pillow-compatible resampling filter identifiers."""

    NEAREST = 0
    LANCZOS = 1
    BILINEAR = 2
    BICUBIC = 3
    BOX = 4
    HAMMING = 5


NEAREST, LANCZOS, BILINEAR, BICUBIC, BOX, HAMMING = Resampling


class Quantize(IntEnum):
    """Pillow-compatible palette quantization methods."""

    MEDIANCUT = 0
    MAXCOVERAGE = 1
    FASTOCTREE = 2
    LIBIMAGEQUANT = 3


class Dither(IntEnum):
    """Pillow-compatible dithering identifiers."""

    NONE = 0
    ORDERED = 1
    RASTERIZE = 2
    FLOYDSTEINBERG = 3


MEDIANCUT, MAXCOVERAGE, FASTOCTREE, LIBIMAGEQUANT = Quantize
NONE, ORDERED, RASTERIZE, FLOYDSTEINBERG = Dither


class Transpose(IntEnum):
    """Pillow-compatible flip and right-angle rotation identifiers."""

    FLIP_LEFT_RIGHT = 0
    FLIP_TOP_BOTTOM = 1
    ROTATE_90 = 2
    ROTATE_180 = 3
    ROTATE_270 = 4
    TRANSPOSE = 5
    TRANSVERSE = 6


class Transform(IntEnum):
    """Pillow-compatible geometric transformation identifiers."""

    AFFINE = 0
    EXTENT = 1
    PERSPECTIVE = 2
    QUAD = 3
    MESH = 4


FLIP_LEFT_RIGHT, FLIP_TOP_BOTTOM, ROTATE_90, ROTATE_180, ROTATE_270, TRANSPOSE, TRANSVERSE = Transpose
AFFINE, EXTENT, PERSPECTIVE, QUAD, MESH = Transform


class SupportsGetData(Protocol):
    """Protocol for objects supplying a transformation method and coefficients."""

    def getdata(self) -> tuple[int, Sequence[object]]:
        """Return the transformation identifier and its associated data.

        Examples:
            ```python
            from blanket import Image


            class IdentityTransform:
                def getdata(self):
                    return Image.Transform.AFFINE, (1, 0, 0, 0, 1, 0)


            image = Image.new("RGB", (8, 8), "navy")
            result = image.transform(image.size, IdentityTransform())
            ```
        """
        ...


class ImageTransformHandler:
    """Base class for custom transformation handlers."""

    def transform(self, size: tuple[int, int], image: Image, resample: int = Resampling.NEAREST, fill: int = 1) -> Image:
        """Implement this method in a custom handler to return a transformed image.

        Args:
            size: Output `(width, height)` in pixels.
            image: Input image.
            resample: Resampling filter from `Image.Resampling`; supported filters depend on the operation.
            fill: Pillow-compatible fill flag.

        Examples:
            ```python
            from blanket import Image


            class CopyHandler(Image.ImageTransformHandler):
                def transform(self, size, image, resample=Image.Resampling.NEAREST, fill=1):
                    return image.resize(size, resample)


            image = Image.new("RGB", (8, 8), "navy")
            result = image.transform((16, 16), CopyHandler())
            ```
        """
        raise NotImplementedError


class Image:
    """An image whose pixels and supported operations live in Rust.

    Modes include `1`, `L`, `LA`, `P`, `PA`, `RGB`, `RGBA`, `HSV`, `CMYK`,
    `YCbCr`, `LAB`, `I`, `I;16` variants, and `F`. Indexed, integer, and float
    modes support a smaller operation set than 8-bit color modes.
    """

    def __init__(self, native: _Image) -> None:
        """Wrap a native image and initialize its Python metadata.

        Args:
            native: Image storage supplied by the Rust extension.

        Examples:
            ```python
            from blanket import Image

            source = Image.new("RGB", (8, 8), (40, 100, 180))
            image = Image.Image(source._native)
            ```
        """
        self._native = native
        self._bands = _BANDS[native.mode]
        self.filename: str | bytes = ""
        self.palette: ImagePalette | None = None
        self._info: dict[object, object] = {}
        self._frames: list[Image] | None = None
        self._frame = 0
        self._exif: Exif | None = None
        self._exif_source: object = None
        palette_data = native.palette_data()
        if palette_data is not None:
            from .ImagePalette import ImagePalette

            self.palette = ImagePalette(palette_data[0], bytes(palette_data[1]))

    @property
    def mode(self) -> str:
        """Pixel mode of this image.

        Examples:
            ```python
            from blanket import Image

            image = Image.new("RGB", (8, 8), (40, 100, 180))
            print(image.mode)
            ```
        """
        return self._native.mode

    @property
    def bit_depth(self) -> int:
        """Significant bits per channel (8, 10, 12, 16, or 32 for `I` and `F`).

        Examples:
            ```python
            from blanket import Image

            image = Image.new("RGB", (8, 8), (40, 100, 180))
            print(image.bit_depth)
            ```
        """
        return self._native.bit_depth

    @property
    def size(self) -> tuple[int, int]:
        """Image dimensions as (width, height).

        Examples:
            ```python
            from blanket import Image

            image = Image.new("RGB", (8, 8), (40, 100, 180))
            print(image.size)
            ```
        """
        return self._native.size

    @property
    def width(self) -> int:
        """Image width in pixels.

        Examples:
            ```python
            from blanket import Image

            image = Image.new("RGB", (8, 8), (40, 100, 180))
            print(image.width)
            ```
        """
        return self._native.width

    @property
    def height(self) -> int:
        """Image height in pixels.

        Examples:
            ```python
            from blanket import Image

            image = Image.new("RGB", (8, 8), (40, 100, 180))
            print(image.height)
            ```
        """
        return self._native.height

    @property
    def format(self) -> str | None:
        """Detected input format, or None for a newly created image.

        Examples:
            ```python
            from blanket import Image

            image = Image.new("RGB", (8, 8), (40, 100, 180))
            print(image.format)
            ```
        """
        return self._native.format

    @property
    def info(self) -> dict[object, object]:
        """Mutable metadata dictionary.

        Examples:
            ```python
            from blanket import Image

            image = Image.new("RGB", (8, 8), (40, 100, 180))
            print(image.info)
            ```
        """
        return self._info

    @property
    def is_animated(self) -> bool:
        """Whether the image contains more than one frame or page.

        Examples:
            ```python
            from blanket import Image

            image = Image.new("RGB", (8, 8), (40, 100, 180))
            print(image.is_animated)
            ```
        """
        return self.n_frames > 1

    @property
    def n_frames(self) -> int:
        """Number of eagerly decoded frames or pages.

        Examples:
            ```python
            from blanket import Image

            assert Image.new("RGB", (2, 2)).n_frames == 1
            ```
        """
        return len(self._frames) if self._frames is not None else 1

    def tell(self) -> int:
        """Return the current zero-based frame index.

        Examples:
            ```python
            from blanket import Image

            assert Image.new("RGB", (2, 2)).tell() == 0
            ```
        """
        self._native.load()
        return self._frame

    def seek(self, frame: int) -> None:
        """Select a frame; raise EOFError when the index is outside the image.

        Args:
            frame: Zero-based frame or page index.

        Examples:
            ```python
            from blanket import Image

            image = Image.new("RGB", (2, 2))
            image.seek(0)
            ```
        """
        self._native.load()
        frame = index(frame)
        if not 0 <= frame < self.n_frames:
            raise EOFError("attempt to seek outside sequence")
        if self._frames is None or frame == self._frame:
            return
        self._sync_palette()
        current = self._frames[self._frame]
        current._native, current.palette, current._info = self._native, self.palette, self._info
        current._exif, current._exif_source = self._exif, self._exif_source
        selected = self._frames[frame]
        self._native, self.palette, self._info = selected._native, selected.palette, selected._info
        self._exif, self._exif_source = selected._exif, selected._exif_source
        self._bands = _BANDS[self._native.mode]
        self._frame = frame

    @property
    def has_transparency_data(self) -> bool:
        """Whether alpha or transparency metadata exists, even if opaque.

        Examples:
            ```python
            from blanket import Image

            image = Image.new("RGB", (8, 8), (40, 100, 180))
            print(image.has_transparency_data)
            ```
        """
        return self.mode in ("LA", "PA", "RGBA") or "transparency" in self.info or (self.palette is not None and self.palette.mode == "RGBA")

    def load(self) -> None:
        """Validate that this eagerly loaded image remains open.

        Examples:
            ```python
            from blanket import Image

            image = Image.new("RGB", (8, 8), (40, 100, 180))
            image.load()
            ```
        """
        self._native.load()
        self._sync_palette()

    @property
    def im(self) -> _Image:
        """Blanket's native image core, available after loading the image.

        This is a Blanket ``_Image`` object, not a Pillow ``ImagingCore``.

        Examples:
            ```python
            from blanket import Image

            image = Image.new("RGB", (2, 2))
            core = image.im
            ```
        """
        self.load()
        return self._native

    def getim(self) -> _Image:
        """Return Blanket's native image core.

        The result is the same object as ``im``; it is not a Pillow capsule.

        Examples:
            ```python
            from blanket import Image

            image = Image.new("RGB", (2, 2))
            assert image.getim() is image.im
            ```
        """
        return self.im

    def verify(self) -> None:
        """Confirm that the eagerly decoded pixel buffer is available.

        File decoding and integrity checks take place in ``Image.open``.
        This method raises ``ValueError`` if the image has been closed.

        Examples:
            ```python
            from blanket import Image

            image = Image.new("RGB", (2, 2))
            image.verify()
            ```
        """
        self.load()

    def draft(self, mode: str | None, size: tuple[int, int] | None) -> None:
        """Return ``None`` because Blanket decodes images when opening them.

        Args:
            mode: Requested decoder mode; accepted for Pillow compatibility.
            size: Requested decoder size; accepted for Pillow compatibility.

        Examples:
            ```python
            from blanket import Image

            image = Image.new("RGB", (8, 8))
            image.draft("L", (4, 4))
            ```
        """
        self.load()

    def show(self, title: str | None = None) -> None:
        """Save a temporary PNG and launch the system image viewer.

        Set ``BLANKET_IMAGE_VIEWER`` to a command and optional arguments to
        select another viewer. The temporary file remains available while the
        viewer opens it.

        Args:
            title: Optional window title; accepted for Pillow compatibility.

        Examples:
            ```python
            from blanket import Image

            image = Image.new("RGB", (8, 8), "navy")
            image.show()
            ```
        """
        self.load()
        display = self if self.mode in ("1", "L", "LA", "P", "PA", "RGB", "RGBA") else self.convert("RGB")
        with tempfile.NamedTemporaryFile(prefix="blanket-", suffix=".png", delete=False) as stream:
            path = stream.name
            try:
                display.save(stream, format="PNG")
            except Exception:
                Path(path).unlink(missing_ok=True)
                raise
        viewer = os.environ.get("BLANKET_IMAGE_VIEWER")
        if viewer:
            command = [*shlex.split(viewer), path]
        elif sys.platform == "darwin":
            command = ["open", path]
        elif sys.platform == "win32":
            os.startfile(path)  # type: ignore[attr-defined]
            return
        else:
            command = ["xdg-open", path]
        try:
            subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            Path(path).unlink(missing_ok=True)
            raise

    def close(self) -> None:
        """Release the image's pixel buffer.

        Examples:
            ```python
            from blanket import Image

            image = Image.new("RGB", (8, 8), (40, 100, 180))
            image.close()
            ```
        """
        self._native.close()
        if self._frames is not None:
            for frame in self._frames:
                frame._native.close()

    def convert(self, mode: str, *, bit_depth: int | None = None) -> Image:
        """Return a new image converted to a supported mode.

        Args:
            mode: Destination mode: `1`, `L`, `LA`, `P`, `PA`, `RGB`, `RGBA`, `HSV`, `CMYK`, `YCbCr`, `LAB`, `I`, `F`, or an `I;16` variant.
            bit_depth: Output sample depth for high-bit-depth storage, or None to retain the input depth. `I` and `F` use 32-bit samples.

        Examples:
            ```python
            from blanket import Image

            image = Image.new("RGB", (8, 8), (40, 100, 180))
            result = image.convert("L")
            ```
        """
        if mode == "PA" and self.mode != "PA":
            source = self if self.mode == "P" else self.convert("RGB").quantize()
            if self.mode in ("LA", "RGBA"):
                alpha = self.getchannel("A").tobytes()
            elif self.mode == "P" and self.palette is not None and self.palette.mode == "RGBA":
                entries = self.palette.tobytes()
                table = bytes(entries[slot * 4 + 3] if slot * 4 + 3 < len(entries) else 255 for slot in range(256))
                alpha = source.tobytes().translate(table)
            else:
                alpha = b"\xff" * (self.width * self.height)
            indices = source.tobytes()
            pixels = bytearray(2 * len(indices))
            pixels[0::2] = indices
            pixels[1::2] = alpha
            result = frombytes("PA", source.size, pixels)
            if source.palette is not None:
                result.putpalette(source.palette)
            return result
        if mode == "PA" and self.mode == "PA":
            return self.copy()
        if mode == "P" and self.mode == "PA":
            result = self.getchannel("P")
            if self.palette is not None:
                result.putpalette(self.palette)
            return result
        if mode == "P" and self.mode != "P":
            return (self.convert("RGB") if self.mode in ("1", "LA", "HSV", "CMYK", "YCbCr", "LAB", "F", "I", "I;16", "I;16L", "I;16B") else self).quantize()
        self._sync_palette()
        result = Image(self._native.convert(mode, bit_depth))
        result.info.update(self._metadata_info())
        return result

    def _sync_palette(self) -> None:
        """Copy the Python palette to the native indexed raster when present."""
        if self.mode in ("P", "PA") and self.palette is not None:
            self._native.set_palette(self.palette.mode, self.palette.tobytes())

    def copy(self) -> Image:
        """Return an independent copy of the pixels, palette, and metadata.

        Examples:
            ```python
            from blanket import Image

            image = Image.new("RGB", (8, 8), (40, 100, 180))
            result = image.copy()
            ```
        """
        self._sync_palette()
        result = Image(self._native.copy())
        result.info.update(self._metadata_info())
        return result

    def getexif(self) -> Exif:
        """Return the image's editable EXIF tags, including nested IFDs.

        Examples:
            ```python
            from blanket import Image

            image = Image.new("RGB", (8, 8))
            image.getexif()[305] = "Blanket"
            ```
        """
        self._native.load()
        info = self._info
        raw = info.get("exif")
        if raw is None and isinstance(info.get("Raw profile type exif"), str):
            value = info["Raw profile type exif"]
            raw = bytes.fromhex("".join(value.split("\n")[3:]) if "\n" in value else value)
        if self._exif is None or raw != self._exif_source:
            self._exif = Exif()
            if isinstance(raw, bytes):
                self._exif.load(raw)
            if 274 not in self._exif:
                xmp = info.get("xmp", info.get("XML:com.adobe.xmp", ""))
                if isinstance(xmp, bytes):
                    xmp = xmp.decode("utf-8", "replace")
                if isinstance(xmp, str) and (match := ORIENTATION_RE.search(xmp)):
                    self._exif[274] = int(match[1] or match[2])
            self._exif_source = raw
        return self._exif

    def _metadata_value(self, key: str) -> object:
        """Resolve a metadata value, including cached EXIF edits and the XMP alias."""
        if key == "exif" and self._exif is not None and self.info.get("exif") == self._exif_source:
            return self._exif.tobytes() if self._exif or self._exif._hidden or self._exif.thumbnail is not None else b""
        return self.info.get(key, self.info.get("XML:com.adobe.xmp") if key == "xmp" else None)

    def _metadata_info(self) -> dict[object, object]:
        """Copy image metadata with any cached EXIF edits serialized."""
        info = self.info.copy()
        if self._exif is not None:
            info["exif"] = self._metadata_value("exif")
        return info

    def getpixel(self, xy: tuple[int, int] | list[int]) -> int | tuple[int, ...]:
        """Return a pixel value, accepting negative coordinates as Pillow does.

        Args:
            xy: Pixel `(x, y)` coordinates. Negative coordinates count from the right or bottom.

        Examples:
            ```python
            from blanket import Image

            image = Image.new("RGB", (8, 8), (40, 100, 180))
            pixel = image.getpixel((0, 0))
            ```
        """
        if isinstance(xy, (tuple, list)) and len(xy) == 2 and isinstance(xy[0], int) and isinstance(xy[1], int):
            # Native access already validates the image and coordinates.
            return self._native.getpixel((xy[0], xy[1]))
        self.load()
        if not isinstance(xy, (tuple, list)):
            raise TypeError("argument must be a sequence")
        if len(xy) != 2:
            raise TypeError("argument must be sequence of length 2")
        coordinates = tuple(int(v) if isinstance(v, float) else index(v) for v in xy)
        return self._native.getpixel(coordinates)

    def getpalette(self, rawmode: str | None = "RGB") -> list[int] | None:
        """Return interleaved palette entries, or None for non-palette images.

        Args:
            rawmode: Channel layout of the palette data, normally `RGB` or `RGBA`.

        Examples:
            ```python
            from blanket import Image

            image = Image.new("RGB", (8, 8), (40, 100, 180))
            palette = image.quantize(colors=8).getpalette()
            ```
        """
        self.load()
        if self.palette is None:
            return None
        mode = self.palette.mode if rawmode is None else rawmode
        if mode not in ("RGB", "RGBA"):
            raise ValueError("unrecognized raw mode")
        data = self.palette.tobytes()
        if mode == self.palette.mode:
            return list(data)
        if mode == "RGB":
            rgb = bytearray(data)
            del rgb[3::4]
            return list(rgb)
        rgba = bytearray(b"\xff" * (len(data) // 3 * 4))
        for channel in range(3):
            rgba[channel::4] = data[channel::3]
        return list(rgba)

    def remap_palette(self, dest_map: Sequence[int], source_palette: bytes | bytearray | None = None) -> Image:
        """Return a P image whose palette follows ``dest_map``.

        The map lists old palette indices in their new order. Unlisted source
        indices become index zero, matching Pillow's remapping behavior.

        Args:
            dest_map: Old indices in the desired palette order.
            source_palette: Optional interleaved RGB or RGBA palette bytes.

        Examples:
            ```python
            from blanket import Image

            image = Image.new("P", (2, 1))
            image.putpalette([0, 0, 0, 255, 0, 0])
            swapped = image.remap_palette([1, 0])
            ```
        """

        self.load()
        if self.mode not in ("L", "P"):
            raise ValueError("illegal image mode")
        positions = [index(value) for value in dest_map]
        if len(positions) > 256 or len(set(positions)) != len(positions) or any(not 0 <= value < 256 for value in positions):
            raise ValueError("dest_map must contain unique palette indices from 0 through 255")
        palette_mode = self.palette.mode if self.mode == "P" and self.palette is not None else "RGB"
        if source_palette is None:
            entries = self.palette.tobytes() if self.mode == "P" and self.palette is not None else _GRAY_PALETTE
        else:
            entries = bytes(source_palette)
            palette_mode = "RGBA" if len(entries) > 768 else "RGB"
        channels = len(palette_mode)
        if len(entries) % channels or any(value * channels + channels > len(entries) for value in positions):
            raise ValueError("source palette has too few entries")
        mapping = bytearray(256)
        for new_position, old_position in enumerate(positions):
            mapping[old_position] = new_position
        result = Image(ops_remap_indices(self._native, bytes(mapping)))
        result.putpalette(b"".join(entries[value * channels : (value + 1) * channels] for value in positions), palette_mode)
        result.info.update(self._metadata_info())
        if isinstance(self.info.get("transparency"), int):
            old_transparency = self.info["transparency"]
            if old_transparency in positions:
                result.info["transparency"] = mapping[old_transparency]
            else:
                result.info.pop("transparency", None)
        return result

    def putpixel(self, xy: tuple[int, int], value: float | tuple[int, ...]) -> None:
        """Write a pixel, accepting negative coordinates.

        Palette images accept numeric indices, not RGB color allocation.

        Args:
            xy: Pixel `(x, y)` coordinates. Negative coordinates count from the right or bottom.
            value: Pixel value or channel tuple.

        Examples:
            ```python
            from blanket import Image

            image = Image.new("RGB", (8, 8), (40, 100, 180))
            image.putpixel((0, 0), (255, 0, 0))
            ```
        """
        if self.mode == "P" and isinstance(value, tuple) and len(value) != 1:
            raise ValueError("palette putpixel requires an index")
        if type(xy) is not tuple or len(xy) != 2 or type(xy[0]) is not int or type(xy[1]) is not int:
            xy = tuple(index(v) for v in xy)
        self._native.putpixel(xy, value)

    def getdata(self, band: int | None = None) -> list[int | float | tuple[int, ...]]:
        """Return a flat pixel snapshot, optionally selecting one band.

        Args:
            band: Zero-based channel index, or None for all channels.

        Examples:
            ```python
            from blanket import Image

            image = Image.new("RGB", (8, 8), (40, 100, 180))
            pixels = image.getdata()
            ```
        """
        source = self if band is None else self.getchannel(index(band))
        return source._native.getdata()

    def get_flattened_data(self, band: int | None = None) -> tuple[int | float | tuple[int, ...], ...]:
        """Return an immutable flat pixel snapshot, as in recent Pillow versions.

        Args:
            band: Zero-based channel index, or None for all channels.

        Examples:
            ```python
            from blanket import Image

            image = Image.new("RGB", (8, 8), (40, 100, 180))
            pixels = image.get_flattened_data()
            ```
        """
        source = self if band is None else self.getchannel(index(band))
        return source._native.get_flattened_data()

    def putdata(self, data: Sequence[int | float | tuple[int, ...]], scale: float = 1.0, offset: float = 0.0) -> None:
        """Write pixels in row order; scale/offset apply to single-band data.

        Args:
            data: Pixel values in row order. Short input leaves remaining pixels unchanged.
            scale: Multiplier for single-band input values.
            offset: Offset added after multiplying single-band values.

        Examples:
            ```python
            from blanket import Image

            image = Image.new("RGB", (8, 8), (40, 100, 180))
            image.putdata([(255, 0, 0)] * 64)
            ```
        """
        self._native.putdata(data, scale, offset)

    def getcolors(self, maxcolors: int = 256) -> list[tuple[int, int | float | tuple[int, ...]]] | None:
        """Return unordered (count, pixel) pairs, or None above maxcolors.

        Args:
            maxcolors: Maximum number of distinct colors to return; exceeding it returns None.

        Examples:
            ```python
            from blanket import Image

            image = Image.new("RGB", (8, 8), (40, 100, 180))
            counts = image.getcolors(maxcolors=256)
            ```
        """
        return self._native.getcolors(max(0, index(maxcolors)))

    def putpalette(self, data: Sequence[int] | bytes | ImagePalette, rawmode: str = "RGB") -> None:
        """Attach an RGB or RGBA palette to an L, P, or PA image.

        Args:
            data: Interleaved palette entries or an ImagePalette object.
            rawmode: Channel layout of the palette data, normally `RGB` or `RGBA`.

        Examples:
            ```python
            from blanket import Image

            image = Image.new("RGB", (8, 8), (40, 100, 180))
            image = Image.new("P", (8, 8))
            image.putpalette([0, 0, 0, 255, 0, 0])
            ```
        """
        from .ImagePalette import ImagePalette

        self.load()
        if self.mode not in ("L", "P", "PA"):
            raise ValueError("illegal image mode")
        palette = data.copy() if isinstance(data, ImagePalette) else ImagePalette(rawmode, bytes(data))
        self._native.set_palette(palette.mode, palette.tobytes())
        self.palette = palette
        self._bands = _BANDS["PA" if self.mode == "PA" else "P"]

    def quantize(self, colors: int = 256, method: int | None = None, kmeans: int = 0, palette: Image | None = None, dither: Dither = Dither.FLOYDSTEINBERG) -> Image:
        """Return an indexed P image using a generated or supplied palette.

        MEDIANCUT, MAXCOVERAGE, and FASTOCTREE run in the native backend.
        LIBIMAGEQUANT is unavailable in this build. Generated palette ordering
        and color choices may differ from Pillow's implementations.
        Non-RGB sources convert through RGB except RGBA, which quantizes directly.

        Args:
            colors: Maximum palette size, from 1 through 256.
            method: Quantizer from `Image.Quantize`; defaults to FASTOCTREE for RGBA and MEDIANCUT otherwise.
            kmeans: Number of k-means refinement iterations for supported quantizers.
            palette: Optional indexed image supplying the destination palette.
            dither: Dithering mode from `Image.Dither`.

        Examples:
            ```python
            from blanket import Image

            image = Image.new("RGB", (8, 8), (40, 100, 180))
            result = image.quantize(colors=8)
            ```
        """

        self.load()
        method = Quantize.FASTOCTREE if method is None and self.mode == "RGBA" else Quantize.MEDIANCUT if method is None else index(method)
        if self.mode == "RGBA" and method not in (2, 3):
            raise ValueError("Fast Octree (method == 2) and libimagequant (method == 3) are the only valid methods for quantizing RGBA images")
        if palette is not None:
            palette.load()
            if palette.mode != "P":
                raise ValueError("bad mode for palette image")
            if self.mode not in ("RGB", "L"):
                raise ValueError("only RGB or L mode images can be quantized to a palette")
            palette._sync_palette()
            # Pillow ignores the generated-palette options on this path.
            colors, method, kmeans = 256, 0, 0
        else:
            colors, kmeans = index(colors), index(kmeans)
            if not 1 <= colors <= 256:
                raise ValueError("bad number of colors")
            if kmeans < 0:
                raise ValueError("kmeans must not be negative")
            if method not in range(4):
                raise ValueError("quantization error")
        source = self.convert("RGB") if self.mode == "P" else self
        result = Image(_native_quantize(source._native, colors, method, kmeans, None if palette is None else palette._native, index(dither)))
        result.info.update(self._metadata_info())
        return result

    def tobytes(self) -> bytes:
        """Return packed pixels.

        Mode `1` uses row-packed bits. `I` is little-endian int32, `F` is
        little-endian float32, and `I;16B` is big-endian uint16. Other
        high-depth samples use little-endian uint16.

        Examples:
            ```python
            from blanket import Image

            image = Image.new("RGB", (8, 8), (40, 100, 180))
            pixels = image.tobytes()
            ```
        """
        return self._native.tobytes()

    def tobitmap(self, name: str = "image") -> bytes:
        """Return a mode ``1`` image as X11 bitmap source bytes.

        Args:
            name: ASCII prefix for the bitmap width, height, and data names.

        Examples:
            ```python
            from blanket import Image

            image = Image.new("1", (8, 8))
            bitmap = image.tobitmap("icon")
            ```
        """

        self.load()
        if self.mode != "1":
            raise ValueError("not a bitmap")
        name.encode("ascii")
        bits = [f"0x{value:02x}" for value in ops_xbm_bits(self._native)]
        body = ",".join(bits)
        return (f"#define {name}_width {self.width}\n#define {name}_height {self.height}\nstatic char {name}_bits[] = {{\n{body}\n}};").encode("ascii")

    def effect_spread(self, distance: int) -> Image:
        """Return an image with pixels randomly sampled from nearby positions.

        Args:
            distance: Maximum horizontal and vertical sample displacement.

        Examples:
            ```python
            from blanket import Image

            image = Image.new("RGB", (8, 8), "navy")
            spread = image.effect_spread(2)
            ```
        """

        self.load()
        result = Image(ops_spread(self._native, index(distance)))
        result.info.update(self._metadata_info())
        return result

    def filter(self, filter: Filter | type[Filter]) -> Image:
        """Return a filtered image, accepting a filter instance or class.

        Args:
            filter: Filter instance or filter class from `ImageFilter`.

        Examples:
            ```python
            from blanket import Image

            image = Image.new("RGB", (8, 8), (40, 100, 180))
            from blanket import ImageFilter

            result = image.filter(ImageFilter.GaussianBlur(radius=2))
            ```
        """

        self.load()
        if callable(filter):
            filter = filter()
        if not hasattr(filter, "filter"):
            raise TypeError("filter argument should be ImageFilter.Filter instance or class")
        neighborhood_filter = isinstance(filter, (RankFilter, ModeFilter)) and getattr(filter.filter, "__func__", None) in (RankFilter.filter, ModeFilter.filter)
        if self.mode in ("L", "I;16", "I;16L", "I;16B") or isinstance(filter, MultibandFilter) or neighborhood_filter:
            result = Image(filter.filter(self._native))
            result.info.update(self._metadata_info())
            return result
        bands = [filter.filter(band._native) for band in self.split()]
        result = Image(filter_merge(self.mode, bands))
        result.info.update(self._metadata_info())
        return result

    def paste(self, im: Image | str | int | tuple[int, ...], box: Image | tuple[int, ...] | None = None, mask: Image | None = None) -> None:
        """Paste pixels or a color, optionally interpolating through an L/RGBA mask.

        High-depth images retain sample precision. Masks can be 8-bit or match
        the image depth. Numeric fills use sample units; CSS colors scale to
        the sample range. Source images must have the same sample depth.

        Args:
            im: Source image or fill color.
            box: Destination origin or rectangle, or a mask image as the second positional argument.
            mask: Optional L or RGBA mask selecting source pixels.

        Examples:
            ```python
            from blanket import Image

            image = Image.new("RGB", (8, 8), (40, 100, 180))
            image.paste("red", (0, 0, 4, 4))
            ```
        """

        if isinstance(box, Image):
            if mask is not None:
                raise ValueError("If using second argument as mask, third argument must be None")
            mask, box = box, None
        box = (0, 0) if box is None else tuple(index(v) for v in box)
        if len(box) == 2:
            size = im.size if isinstance(im, Image) else mask.size if isinstance(mask, Image) else None
            if size is None:
                raise ValueError("cannot determine region size; use 4-item box")
            box += (box[0] + size[0], box[1] + size[1])
        if len(box) != 4:
            raise TypeError("box must be a 2- or 4-item sequence")
        size = (box[2] - box[0], box[3] - box[1])
        if min(size) < 0:
            raise ValueError("invalid paste region")
        self._sync_palette()
        if isinstance(im, Image):
            im.load()
            if im.size != size:
                raise ValueError("images do not match")
            depth = self.bit_depth if self.bit_depth > 8 and self.mode in ("L", "RGB", "RGBA") else None
            source = im if im.mode == self.mode else im.convert(self.mode, bit_depth=depth)
        else:
            if self.mode == "P" and isinstance(im, (str, tuple)) and not (isinstance(im, tuple) and len(im) == 1):
                raise TypeError("color must be int or single-element tuple")
            if self.bit_depth > 8 and self.mode in ("L", "RGB", "RGBA"):
                color = color_pixel(im, self.mode, self.bit_depth)
                raw = b"".join(value.to_bytes(2, "little") for value in color)
                source = Image(image_new(self.mode, size, raw, self.bit_depth))
            else:
                source = new(self.mode, size, im)
        # Snapshot aliases before acquiring the native mutable destination borrow.
        source = source.copy() if source._native is self._native else source
        if mask is not None and mask._native is self._native:
            mask = mask.copy()
        image_paste(self._native, source._native, box[:2], mask._native if mask is not None else None, not isinstance(im, Image))

    def alpha_composite(self, im: Image, dest: tuple[int, int] = (0, 0), source: tuple[int, ...] = (0, 0), *, linear_light: bool = False, transfer: str = "srgb") -> None:
        """Composite an RGBA source region onto this image in place.

        Args:
            im: RGBA source image.
            dest: Destination `(x, y)` position.
            source: Source `(x, y)` origin or `(left, top, right, bottom)` region.
            linear_light: Decode the selected transfer before source-over blending.
            transfer: "srgb" or "linear"; applies when linear_light is true.

        Examples:
            ```python
            from blanket import Image

            image = Image.new("RGB", (8, 8), (40, 100, 180))
            image = image.convert("RGBA")
            overlay = Image.new("RGBA", (4, 4), (255, 0, 0, 128))
            image.alpha_composite(overlay, dest=(2, 2))
            ```
        """
        if not isinstance(source, (list, tuple)) or len(source) not in (2, 4):
            raise ValueError("Source must be a sequence of length 2 or 4")
        if not isinstance(dest, (list, tuple)) or len(dest) != 2:
            raise ValueError("Destination must be a sequence of length 2")
        source = tuple(index(v) for v in source)
        dest = tuple(index(v) for v in dest)
        if min(source) < 0:
            raise ValueError("Source must be non-negative")
        region = source + im.size if len(source) == 2 else source
        overlay = im if region == (0, 0, *im.size) else im.crop(region)
        box = (*dest, dest[0] + overlay.width, dest[1] + overlay.height)
        if linear_light or self.mode == "RGBA;F":
            background = self.crop(box)
            self.paste(alpha_composite(background, overlay, linear_light=linear_light, transfer=transfer), box)
            return
        if box == (0, 0, *self.size):
            overlay = overlay.copy() if overlay._native is self._native else overlay
            image_alpha_composite_inplace(self._native, overlay._native)
            return
        background = self if box == (0, 0, *self.size) else self.crop(box)
        self.paste(alpha_composite(background, overlay), box)

    def putalpha(self, alpha: Image | int) -> None:
        """Replace alpha in place; RGB images become RGBA.

        Only RGB and RGBA are accepted. LA and PA images are not supported.

        Args:
            alpha: L image matching the image size, or a constant in the image's sample range. High-depth images accept matching-depth or scaled 8-bit alpha images.

        Examples:
            ```python
            from blanket import Image

            image = Image.new("RGB", (8, 8), (40, 100, 180))
            image.putalpha(128)
            ```
        """

        if self.mode not in ("RGB", "RGBA"):
            raise ValueError("putalpha requires RGB or RGBA; LA and PA modes are not supported")
        if isinstance(alpha, Image):
            band = alpha
        elif self.bit_depth > 8:
            value = max(0, min((1 << self.bit_depth) - 1, index(alpha)))
            band = Image(image_new("L", self.size, value.to_bytes(2, "little"), self.bit_depth))
        else:
            band = new("L", self.size, index(alpha))
        if band.mode != "L" or band.size != self.size:
            raise ValueError("illegal image mode or size for alpha")
        image_putalpha(self._native, band._native)
        self.palette = None
        self._bands = _BANDS["RGBA"]

    def split(self) -> tuple[Image, ...]:
        """Return independent single-band images in channel order.

        Examples:
            ```python
            from blanket import Image

            image = Image.new("RGB", (8, 8), (40, 100, 180))
            red, green, blue = image.split()
            ```
        """

        if self.mode in ("1", "I", "F", "I;16", "I;16L", "I;16B"):
            return (self.copy(),)
        bands = tuple(Image(native) for native in ops_split(self._native))
        info = self._metadata_info()
        for band in bands:
            band.info.update(info)
        return bands

    def getbands(self) -> tuple[str, ...]:
        """Return channel names in pixel order.

        Examples:
            ```python
            from blanket import Image

            image = Image.new("RGB", (8, 8), (40, 100, 180))
            bands = image.getbands()
            ```
        """
        return self._bands

    def getchannel(self, channel: int | str) -> Image:
        """Return an independent image for a channel name or index.

        Args:
            channel: Channel name, such as `R`, or zero-based channel index.

        Examples:
            ```python
            from blanket import Image

            image = Image.new("RGB", (8, 8), (40, 100, 180))
            red = image.getchannel("R")
            ```
        """
        self.load()
        if isinstance(channel, str):
            try:
                channel = self.getbands().index(channel)
            except ValueError as error:
                raise ValueError(f'The image has no channel "{channel}"') from error
        channel = index(channel)
        if not 0 <= channel < len(self.getbands()):
            raise ValueError("band index out of range")
        if self.mode in ("1", "I", "F", "I;16", "I;16L", "I;16B"):
            return self.copy()
        result = Image(self._native.getchannel(channel))
        result.info.update(self._metadata_info())
        return result

    def histogram(self, mask: Image | None = None, extrema: tuple[float, float] | None = None) -> list[int]:
        """Return 256 bins per band for 8-bit modes; extrema is ignored.

        Args:
            mask: Optional L mask; pixels with a zero mask value are excluded.
            extrema: Accepted for Pillow compatibility; ignored.

        Examples:
            ```python
            from blanket import Image

            image = Image.new("RGB", (8, 8), (40, 100, 180))
            counts = image.histogram()
            ```
        """

        self.load()
        if mask is not None:
            mask.load()
            if mask.mode != "L":
                raise ValueError("bad transparency mask")
        return ops_histogram(self._native, None if mask is None else mask._native)

    def getextrema(self) -> tuple[int | float, int | float] | tuple[tuple[int | float, int | float], ...] | None:
        """Return the minimum and maximum sample value of each band.

        Examples:
            ```python
            from blanket import Image

            image = Image.new("RGB", (8, 8), (40, 100, 180))
            extrema = image.getextrema()
            ```
        """
        if self.mode in ("RGB;F", "RGBA;F"):
            return tuple(cast("tuple[float, float]", self.getchannel(c).getextrema()) for c in self.getbands()) if self.width * self.height else None
        if self.mode in ("I", "F", "I;16", "I;16L", "I;16B"):
            return self._native.getextrema_scalar()
        ranges = self._native.getextrema()
        if not ranges:
            return None
        if len(ranges) == 1:
            return ranges[0]
        return tuple(ranges)

    def getbbox(self, *, alpha_only: bool = True) -> tuple[int, int, int, int] | None:
        """Return the nonzero bounding box, using alpha for LA, PA, and RGBA by default.

        Args:
            alpha_only: For LA, PA, and RGBA images, inspect only alpha when true; otherwise inspect all channels.

        Examples:
            ```python
            from blanket import Image

            image = Image.new("RGB", (8, 8), (40, 100, 180))
            box = image.getbbox()
            ```
        """
        return self._native.getbbox(alpha_only)

    def point(self, lut: Sequence[float] | Callable[[int], float], mode: str | None = None) -> Image:
        """Map 8-bit channels through a table or a function evaluated 256 times.

        Mode `F` accepts a callable applied to each floating-point sample.

        Args:
            lut: 256 entries per input channel, a callable evaluated for each 8-bit value, or a per-sample callable for mode `F`.
            mode: Optional output mode; must match the input mode.

        Examples:
            ```python
            from blanket import Image

            image = Image.new("RGB", (8, 8), (40, 100, 180))
            result = image.point(lambda value: 255 - value)
            ```
        """

        self.load()
        if self.mode == "F":
            if not callable(lut):
                raise ValueError("point operation not supported for this mode")
            result = Image(self._native.point_float(lut))
            result.info.update(self._metadata_info())
            return result
        if mode is not None and mode != self.mode:
            raise ValueError("point mode conversion is not supported")
        values = [lut(i) for i in range(256)] * len(self.getbands()) if callable(lut) else list(lut)
        if len(values) != 256 * len(self.getbands()):
            raise ValueError("wrong number of lut entries")
        result = Image(ops_lut(self._native, [max(0, min(255, round(v))) for v in values]))
        result.info.update(self._metadata_info())
        return result

    def thumbnail(self, size: tuple[float, float], resample: int = Resampling.BICUBIC, reducing_gap: float | None = 2.0) -> None:
        """Shrink in place to fit size, preserving aspect ratio without upscaling.

        Args:
            size: Output `(width, height)` in pixels.
            resample: Resampling filter from `Image.Resampling`; supported filters depend on the operation.
            reducing_gap: Optional pre-reduction optimization threshold; must be at least 1.0.

        Examples:
            ```python
            from blanket import Image

            image = Image.new("RGB", (8, 8), (40, 100, 180))
            image.thumbnail((4, 4))
            ```
        """
        self.load()
        x, y = (math.floor(v) for v in size)
        if x <= 0 or y <= 0:
            raise ValueError("height and width must be > 0")
        if x >= self.width and y >= self.height:
            return
        if not self.width or not self.height:
            return
        aspect = self.width / self.height
        if x / y >= aspect:
            value = y * aspect
            x = max(1, min(math.floor(value), math.ceil(value), key=lambda n: abs(aspect - n / y)))
        else:
            value = x / aspect
            y = max(1, min(math.floor(value), math.ceil(value), key=lambda n: 0 if n == 0 else abs(aspect - x / n)))
        resized = self.resize((x, y), resample, reducing_gap=reducing_gap)
        self._native = resized._native
        self.palette = resized.palette

    def reduce(self, factor: int | tuple[int, int], box: tuple[int, int, int, int] | None = None) -> Image:
        """Average integer blocks, rounding the output dimensions up.

        factor can specify horizontal and vertical factors separately.
        box selects a nonempty source rectangle within the image.
        Indexed `P` images are unsupported.

        Args:
            factor: Positive integer reduction factor, or separate `(x, y)` factors.
            box: Optional source `(left, top, right, bottom)` rectangle.

        Examples:
            ```python
            from blanket import Image

            image = Image.new("RGB", (8, 8), (40, 100, 180))
            result = image.reduce(2)
            ```
        """

        self.load()
        factor = factor if isinstance(factor, (list, tuple)) else (factor, factor)
        factor = tuple(index(value) for value in factor)
        if len(factor) != 2:
            raise TypeError("factor must contain two integers")
        if min(factor) < 1:
            raise ValueError("scale must be > 0")
        box = (0, 0, self.width, self.height) if box is None else tuple(index(v) for v in box)
        if len(box) != 4:
            raise TypeError("box must contain four integers")
        if factor == (1, 1) and box == (0, 0, self.width, self.height):
            return self.crop()
        if self.mode == "P":
            raise ValueError("image has wrong mode")
        if box[0] < 0 or box[1] < 0 or box[2] > self.width or box[3] > self.height:
            raise ValueError("box must be within the image")
        if box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError("box can't be empty")
        result = Image(ops_reduce(self._native, factor, box))
        result.info.update(self._metadata_info())
        return result

    def entropy(self, mask: Image | None = None, extrema: tuple[float, float] | None = None) -> float:
        """Return Shannon entropy over all channel histogram bins.

        An L mask selects pixels with nonzero values. Each channel uses 256
        bins. Extrema controls binning for high-bit-depth and wide scalar
        images; it is ignored for 8-bit modes.

        Args:
            mask: Optional L mask; pixels with a zero mask value are excluded.
            extrema: Optional lower and upper sample values for wide images.

        Examples:
            ```python
            from blanket import Image

            image = Image.new("RGB", (8, 8), (40, 100, 180))
            entropy = image.entropy()
            ```
        """

        self.load()
        if mask is not None:
            mask.load()
        return ops_entropy(self._native, None if mask is None else mask._native, extrema)

    def transpose(self, method: int) -> Image:
        """Return a flipped or right-angle rotated copy using ``Transpose``.

        Args:
            method: Flip or right-angle rotation from `Image.Transpose`.

        Examples:
            ```python
            from blanket import Image

            image = Image.new("RGB", (8, 8), (40, 100, 180))
            result = image.transpose(Image.Transpose.ROTATE_90)
            ```
        """

        method = index(method)
        if method not in range(7):
            raise ValueError("No such transpose operation")
        result = Image(ops_transpose(self._native, (2, 4, 8, 3, 6, 5, 7)[method]))
        result.info.update(self._metadata_info())
        return result

    def transform(
        self,
        size: tuple[int, int],
        method: int | ImageTransformHandler | SupportsGetData,
        data: Sequence[object] | None = None,
        resample: int = Resampling.NEAREST,
        fill: int = 1,
        fillcolor: str | int | tuple[int, ...] | None = None,
    ) -> Image:
        """Map source pixels to a new canvas using a ``Transform`` method.

        AFFINE and PERSPECTIVE use inverse mapping coefficients. EXTENT takes
        a source rectangle. QUAD takes NW, SW, SE, NE source corners; MESH
        takes (destination rectangle, source quad) pairs in drawing order.
        Supports NEAREST, BILINEAR and BICUBIC, plus optional fillcolor.

        Args:
            size: Output `(width, height)` in pixels.
            method: Method from `Image.Transform`, a getdata() object, or an ImageTransformHandler.
            data: Coefficients, source rectangle, quadrilateral, or mesh for the selected transform.
            resample: Resampling filter from `Image.Resampling`; supported filters depend on the operation.
            fill: Pillow-compatible fill flag.
            fillcolor: Color for pixels outside the source image.

        Examples:
            ```python
            from blanket import Image

            image = Image.new("RGB", (8, 8), (40, 100, 180))
            result = image.transform((8, 8), Image.Transform.AFFINE, (1, 0, 1, 0, 1, 0))
            ```
        """

        if isinstance(method, ImageTransformHandler):
            return method.transform(size, self, resample=resample, fill=fill)
        if hasattr(method, "getdata"):
            method, data = method.getdata()
        if data is None:
            raise ValueError("missing method data")
        if method not in Transform:
            raise ValueError("unknown transformation method")
        if resample not in (0, 2, 3):
            raise ValueError("transform supports NEAREST, BILINEAR and BICUBIC")
        if self.mode in ("1", "P", "PA"):
            resample = Resampling.NEAREST
        size = tuple(index(v) for v in size)
        if len(size) != 2:
            raise TypeError("size must contain two integers")
        if min(size) < 0:
            raise ValueError("width and height must be >= 0")
        self.load()
        color = list(struct.pack("<f", 0.0 if fillcolor is None else float(fillcolor))) if self.mode == "F" else color_pixel(fillcolor, self.mode)

        def coordinates(values: Sequence[object], count: int) -> tuple[float, ...]:
            values = tuple(float(v) for v in values[:count])
            if len(values) != count or not all(math.isfinite(v) for v in values):
                raise ValueError(f"transform requires {count} finite coordinates")
            return values

        if method in (Transform.AFFINE, Transform.EXTENT):
            if method == Transform.EXTENT:
                left, top, right, bottom = coordinates(data, 4)
                matrix = ((right - left) / size[0], 0, left, 0, (bottom - top) / size[1], top)
            else:
                matrix = coordinates(data, 6)
            native = ops_affine(self._native, size, matrix, resample, color, isinstance(fillcolor, str))
        else:
            if method == Transform.MESH:
                mesh = []
                for box, quad in data:
                    box = tuple(index(v) for v in box)
                    if len(box) != 4:
                        raise ValueError("mesh boxes require four coordinates")
                    mesh.append((box, coordinates(quad, 8)))
            else:
                mesh = [((0, 0, *size), coordinates(data, 8))]
            native = ops_warp(self._native, size, mesh, (resample, method == Transform.PERSPECTIVE), color if fillcolor is not None else None, isinstance(fillcolor, str))
        result = Image(native)
        result.info.update(self._metadata_info())
        return result

    def crop(self, box: tuple[float, float, float, float] | None = None) -> Image:
        """Return the rectangular region defined by ``box``.

        Coordinates are ``(left, upper, right, lower)``. Areas outside the
        source image are padded with zero-valued pixels, as in Pillow.

        Args:
            box: `(left, top, right, bottom)` rectangle, or None to copy the image; outside pixels are zero-filled.

        Examples:
            ```python
            from blanket import Image

            image = Image.new("RGB", (8, 8), (40, 100, 180))
            result = image.crop((1, 1, 7, 7))
            ```
        """

        if box is None:
            self.load()
            native = ops_transpose(self._native, 1)
        else:
            if box[2] < box[0]:
                raise ValueError("Coordinate 'right' is less than 'left'")
            if box[3] < box[1]:
                raise ValueError("Coordinate 'lower' is less than 'upper'")
            self.load()
            left, upper, right, lower = box
            if type(left) is not int or type(upper) is not int or type(right) is not int or type(lower) is not int:
                left, upper, right, lower = round(left), round(upper), round(right), round(lower)
            native = ops_canvas(self._native, (right - left, lower - upper), (-left, -upper), [0] * (len(self._bands) * (4 if self.mode in ("F", "RGB;F", "RGBA;F") else 1)))
        result = Image(native)
        result.info.update(self._metadata_info())
        return result

    def resize(
        self,
        size: tuple[int, int],
        resample: int | None = None,
        box: tuple[float, float, float, float] | None = None,
        reducing_gap: float | None = None,
        *,
        linear_light: bool = False,
        transfer: str = "srgb",
        premultiply_alpha: bool = True,
    ) -> Image:
        """Return a resized copy, using BICUBIC unless a filter is specified.

        ``box`` selects a source rectangle within the image. ``reducing_gap``
        optionally enables integer reduction before filtering (at least 1.0).

        Args:
            size: Output `(width, height)` in pixels.
            resample: Filter from `Image.Resampling`; None selects BICUBIC, or NEAREST for indexed images.
            box: Optional floating-point source rectangle.
            reducing_gap: Optional pre-reduction optimization threshold; must be at least 1.0.
            linear_light: Decode the selected transfer before filtering RGB/RGBA.
            transfer: "srgb" or "linear"; applies when linear_light is true.
            premultiply_alpha: Filter premultiplied color; applies to float or linear-light resizing.

        Examples:
            ```python
            from blanket import Image

            image = Image.new("RGB", (8, 8), (40, 100, 180))
            result = image.resize((16, 16), Image.Resampling.LANCZOS)
            ```
        """

        method = Resampling.BICUBIC if resample is None else resample
        if method not in range(6):
            raise ValueError(f"Unknown resampling filter ({method})")
        if self.mode in ("1", "P", "PA"):
            method = Resampling.NEAREST
        if reducing_gap is not None and reducing_gap < 1.0:
            raise ValueError("reducing_gap must be 1.0 or greater")
        size = tuple(index(value) for value in size)
        if len(size) != 2:
            raise TypeError("size must contain two integers")
        if min(size) <= 0:
            raise ValueError("height and width must be > 0")
        box = (0, 0, self.width, self.height) if box is None else tuple(box)
        if len(box) != 4:
            raise TypeError("box must contain four coordinates")
        if not all(math.isfinite(value) for value in box) or box[0] < 0 or box[1] < 0 or box[2] > self.width or box[3] > self.height or box[2] < box[0] or box[3] < box[1]:
            raise ValueError("invalid resize box")
        self.load()
        native = self._native
        if transfer not in ("srgb", "linear"):
            raise ValueError("transfer must be 'srgb' or 'linear'")
        if linear_light or self.mode in ("RGB;F", "RGBA;F"):
            result = Image(ops_color_resize(native, size, method, box, linear_light and transfer == "srgb", premultiply_alpha))
            result.info.update(self._metadata_info())
            return result
        if size == self.size and box == (0, 0, self.width, self.height):
            native = ops_transpose(native, 1)
        else:
            if reducing_gap is not None and method != Resampling.NEAREST and self.mode not in ("LA", "RGBA", "I", "F", "I;16", "I;16L", "I;16B"):
                fx = max(1, int((box[2] - box[0]) / size[0] / reducing_gap))
                fy = max(1, int((box[3] - box[1]) / size[1] / reducing_gap))
                if fx > 1 or fy > 1:
                    support = _RESIZE_SUPPORT[method] - 0.5
                    sx = support * (box[2] - box[0]) / size[0]
                    sy = support * (box[3] - box[1]) / size[1]
                    safe = (max(0, int(box[0] - sx)), max(0, int(box[1] - sy)), min(self.width, math.ceil(box[2] + sx)), min(self.height, math.ceil(box[3] + sy)))
                    native = ops_reduce(native, (fx, fy), safe)
                    box = ((box[0] - safe[0]) / fx, (box[1] - safe[1]) / fy, (box[2] - safe[0]) / fx, (box[3] - safe[1]) / fy)
            native = ops_resize(native, size, method, box, reducing_gap if self.mode in ("LA", "RGBA") else None)
        result = Image(native)
        result.info.update(self._metadata_info())
        return result

    def rotate(
        self,
        angle: float,
        resample: int = Resampling.NEAREST,
        expand: bool = False,
        center: tuple[float, float] | None = None,
        translate: tuple[float, float] | None = None,
        fillcolor: str | int | tuple[int, ...] | None = None,
    ) -> Image:
        """Return a copy rotated counterclockwise by an angle in degrees.

        Supports NEAREST (default), BILINEAR, and BICUBIC. The default center
        is the image midpoint; translate shifts the result after rotation.
        expand enlarges the canvas assuming the default center and no translation.
        fillcolor colors pixels outside the source image.

        Args:
            angle: Counterclockwise angle in degrees.
            resample: NEAREST, BILINEAR, or BICUBIC from `Image.Resampling`.
            expand: Expand the output canvas to contain the rotated image.
            center: Rotation center in pixel coordinates, or None for the image center.
            translate: Optional `(x, y)` translation after rotation.
            fillcolor: Color for pixels outside the source image.

        Examples:
            ```python
            from blanket import Image

            image = Image.new("RGB", (8, 8), (40, 100, 180))
            result = image.rotate(30, expand=True)
            ```
        """

        angle %= 360.0
        if not math.isfinite(angle):
            raise ValueError("angle must be finite")
        self.load()
        orientation = None
        if not (center or translate):
            if angle in (0, 180):
                orientation = 1 if angle == 0 else 3
            elif angle in (90, 270) and (expand or self.width == self.height):
                orientation = 8 if angle == 90 else 6
        if orientation is not None:
            result = Image(ops_transpose(self._native, orientation))
        else:
            if resample not in (Resampling.NEAREST, Resampling.BILINEAR, Resampling.BICUBIC):
                raise ValueError("rotate supports NEAREST, BILINEAR and BICUBIC")
            if self.mode in ("1", "P", "PA"):
                resample = Resampling.NEAREST
            w, h = self.size
            cx, cy = (w / 2, h / 2) if center is None else center
            tx, ty = (0, 0) if translate is None else translate
            if not all(math.isfinite(v) for v in (cx, cy, tx, ty)):
                raise ValueError("center and translate must contain finite coordinates")
            radians = -math.radians(angle)
            a, b = round(math.cos(radians), 15), round(math.sin(radians), 15)
            d, e = -b, a
            c = a * (-cx - tx) + b * (-cy - ty) + cx
            f = d * (-cx - tx) + e * (-cy - ty) + cy
            if expand:
                corners = [(a * x + b * y + c, d * x + e * y + f) for x, y in ((0, 0), (w, 0), (w, h), (0, h))]
                nw = math.ceil(max(x for x, _ in corners)) - math.floor(min(x for x, _ in corners))
                nh = math.ceil(max(y for _, y in corners)) - math.floor(min(y for _, y in corners))
                dx, dy = -(nw - w) / 2, -(nh - h) / 2
                c, f = a * dx + b * dy + c, d * dx + e * dy + f
                w, h = nw, nh
            fill = list(struct.pack("<f", 0.0 if fillcolor is None else float(fillcolor))) if self.mode == "F" else color_pixel(fillcolor, self.mode)
            result = Image(ops_affine(self._native, (w, h), (a, b, c, d, e, f), resample, fill, isinstance(fillcolor, str)))
        result.info.update(self._metadata_info())
        return result

    def to_pillow(self) -> object:
        """Return an equivalent Pillow image for interoperability.

        High-bit-depth `L`/`RGB`/`RGBA` images must be converted to 8 bits first.
        `I`, `F`, and `I;16` variants transfer at their native sample width.

        Examples:
            ```python
            from blanket import Image

            image = Image.new("RGB", (8, 8), (40, 100, 180))
            pillow_image = image.to_pillow()
            ```
        """
        if self.bit_depth != 8 and self.mode not in ("I", "F", "I;16", "I;16L", "I;16B"):
            raise ValueError("to_pillow requires 8-bit pixels; use convert(..., bit_depth=8) explicitly")

        try:
            from PIL import Image as PillowImage
        except ImportError as error:
            raise ImportError("Pillow is required for to_pillow(); install pyblanket[test] or Pillow") from error
        result = PillowImage.frombytes(self.mode, self.size, self.tobytes())
        if self.palette is not None:
            result.putpalette(self.palette.tobytes(), self.palette.mode)
        result.info.update(self._metadata_info())
        return result

    def save(self, fp: str | bytes | os.PathLike[str] | os.PathLike[bytes] | BinaryIO, format: str | None = None, **options: object) -> None:
        """Save in a supported raster format or PDF; see the formats guide.

        HSV cannot be saved. `F` and `LAB` write TIFF only. `YCbCr` writes
        JPEG only. CMYK writes JPEG, TIFF, or PDF. Integer modes
        convert to `L` for encoding.

        Pass a ``LosslessImageCompressor`` or ``LossyImageCompressor`` to
        ``compressor`` to optimize this save.
        Omitting it (or passing None) uses the normal encoder settings.
        Binary streams must return written byte counts from ``write()``;
        streams that make no progress raise ``BlockingIOError``.

        Args:
            fp: Filename, path-like object, or binary stream.
            format: Output format name; inferred from the filename when omitted.
            **options: Format-specific encoder options and optional `compressor`; see the formats guide.

        Examples:
            ```python
            from blanket import Image

            image = Image.new("RGB", (8, 8), (40, 100, 180))
            from io import BytesIO

            output = BytesIO()
            image.save(output, format="PNG")
            ```
        """

        compressor = options.pop("compressor", None)
        if compressor is not None and not isinstance(compressor, (LosslessImageCompressor, LossyImageCompressor)):
            raise TypeError("compressor must be a LosslessImageCompressor, LossyImageCompressor, or None")
        output_format = _output_format(fp, format)
        if {"save_all", "append_images", "duration", "loop", "disposal", "blend", "default_image"} & options.keys():
            from ._sequence import save_sequence

            if save_sequence(self, fp, output_format, options, compressor):
                return None
        if self.mode == "LAB":
            if output_format != "TIFF":
                raise OSError(f"cannot write mode LAB as {output_format}")
            if compressor is not None:
                raise ValueError("compression optimization is not supported for LAB images")
        if self.mode in ("CMYK", "YCbCr"):
            allowed = ("JPEG", "TIFF", "PDF") if self.mode == "CMYK" else ("JPEG",)
            if output_format not in allowed:
                raise OSError(f"cannot write mode {self.mode} as {output_format}")
            if self.mode == "CMYK" and compressor is not None:
                raise ValueError("compression optimization is not supported for CMYK images")
        if self.mode == "HSV":
            raise OSError(f"cannot write mode HSV as {output_format}")
        if self.mode == "F":
            if output_format != "TIFF":
                raise OSError(f"cannot write mode F as {output_format}")
            if compressor is not None:
                raise ValueError("compression optimization is not supported for F images")
        if self.mode in ("I", "I;16", "I;16L", "I;16B"):
            if self.mode == "I":
                depth = 16 if output_format in ("PNG", "TIFF", "JXL", "JPEG2000", "PPM", "PAM") else 8
                return self.convert("L", bit_depth=depth).save(fp, output_format, compressor=compressor, **options)
            if output_format not in ("PNG", "TIFF", "JXL", "JPEG2000", "PPM", "PAM"):
                return self.convert("L").save(fp, output_format, compressor=compressor, **options)
            return self.convert("L", bit_depth=16).save(fp, output_format, compressor=compressor, **options)
        self._sync_palette()
        metadata = save_metadata(self, output_format, options)
        values = _save_options(output_format, options)
        has_metadata = any(metadata)
        output = None if hasattr(fp, "write") else os.fspath(fp)
        encoded = _encode(
            self._native,
            output_format,
            **values,
            compressor=None if compressor is None else compressor._native,
            output=output,
            icc_profile=metadata[2],
            metadata_exif=metadata[0],
            metadata_xmp=metadata[1],
            attach_metadata=has_metadata,
            metadata_chunks=metadata[3],
        )
        if hasattr(fp, "write"):
            write_all(cast("BinaryIO", fp), encoded)
        return None

    def __enter__(self) -> Self:
        """Load the image and return it for use in a context manager.

        Examples:
            ```python
            from blanket import Image

            with Image.new("RGB", (8, 8), (40, 100, 180)) as image:
                print(image.size)
            ```
        """
        self.load()
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Close the image when leaving a context manager.

        Args:
            *exc_info: Exception type, value, and traceback supplied by the context manager.

        Examples:
            ```python
            from blanket import Image

            with Image.new("RGB", (8, 8), (40, 100, 180)) as image:
                result = image.copy()
            ```
        """
        self.close()

    def __repr__(self) -> str:
        """Return a representation containing the image mode and dimensions.

        Examples:
            ```python
            from blanket import Image

            image = Image.new("RGB", (8, 8), (40, 100, 180))
            print(repr(image))
            ```
        """
        return f"<blanket.Image.Image image mode={self.mode} size={self.width}x{self.height}>"


def new(mode: str, size: tuple[int, int], color: str | float | tuple[int, ...] | None = 0) -> Image:
    """Create a supported-mode image filled with color.

    Args:
        mode: Image mode: `1`, `L`, `LA`, `P`, `PA`, `RGB`, `RGBA`, `HSV`, `CMYK`, `YCbCr`, `LAB`, `I`, `F`, or an `I;16` variant.
        size: Output `(width, height)` in pixels.
        color: Fill value, channel tuple, or CSS color string.

    Examples:
        ```python
        from blanket import Image

        image = Image.new("RGB", (64, 64), "navy")
        ```
    """

    if not isinstance(size, (list, tuple)) or len(size) != 2:
        raise ValueError("Size must be a list or tuple of length 2")
    dimensions = size if type(size) is tuple and type(size[0]) is int and type(size[1]) is int else tuple(index(v) for v in size)
    if dimensions[0] < 0 or dimensions[1] < 0:
        raise ValueError("Width and height must be >= 0")
    if mode in ("I", "I;16", "I;16L", "I;16B"):
        value = 0 if color is None else index(color)
        if mode == "I":
            pixel = (value & 0xFFFFFFFF).to_bytes(4, "little")
        else:
            pixel = (value & 0xFFFF).to_bytes(2, "big" if mode == "I;16B" else "little")
        return Image(image_new(mode, dimensions, pixel))
    if mode == "F":
        value = 0.0 if color is None else float(color)
        return Image(image_new(mode, dimensions, struct.pack("<f", value)))
    if mode in ("RGB;F", "RGBA;F"):
        channels = 3 if mode == "RGB;F" else 4
        values = (0.0,) * channels if color is None else color if isinstance(color, tuple) else (float(color),) * channels
        if len(values) != channels:
            raise ValueError("wrong number of color components")
        return Image(image_new(mode, dimensions, struct.pack("<" + "f" * channels, *values)))
    native_mode = "L" if mode == "P" else mode
    if mode == "HSV" and isinstance(color, str):
        color = frombytes("RGB", (1, 1), bytes(color_pixel(color, "RGB"))).convert("HSV").getpixel((0, 0))
    if mode not in ("1", "L", "LA", "RGB", "RGBA", "P", "PA", "HSV", "CMYK", "YCbCr", "LAB"):
        raise ValueError(f"unsupported image mode {mode!r}")
    palette_color = mode == "P" and (isinstance(color, str) or (isinstance(color, tuple) and len(color) in (3, 4)))
    result = Image(image_new(native_mode, dimensions, color_pixel(0 if palette_color else color, native_mode)))
    if mode in ("P", "PA"):
        result.putpalette(bytes(color_pixel(color, "RGB")) if palette_color else bytes(768))
    return result


def merge(mode: str, bands: Sequence[Image]) -> Image:
    """Interleave L bands into an independent image.

    The destination mode may be `1`, `L`, `LA`, `PA`, `RGB`, `RGBA`, `HSV`,
    `CMYK`, `YCbCr`, or `LAB`.

    Args:
        mode: Destination mode with as many channels as supplied L bands.
        bands: Sequence of single-channel L images, one per output band.

    Examples:
        ```python
        from blanket import Image

        image = Image.new("RGB", (8, 8), (40, 100, 180))
        red, green, blue = image.split()
        result = Image.merge("RGB", (blue, green, red))
        ```
    """

    bands = tuple(bands)
    if any(band.mode != "L" for band in bands):
        raise ValueError("mode mismatch")
    return Image(filter_merge(mode, [band._native for band in bands]))


def blend(im1: Image, im2: Image, alpha: float) -> Image:
    """Interpolate equal-sized images of matching depth, clipping extrapolated values.

    Indexed `P` images are unsupported.

    Args:
        im1: First image; both inputs must have the same mode and dimensions.
        im2: Second image.
        alpha: Blending weight: 0 selects the first image and 1 the second; values outside this range extrapolate.

    Examples:
        ```python
        from blanket import Image

        image = Image.new("RGB", (8, 8), (40, 100, 180))
        other = Image.new("RGB", image.size, "white")
        result = Image.blend(image, other, 0.25)
        ```
    """

    im1.load()
    im2.load()
    if im1.mode == "P" or im2.mode == "P":
        raise ValueError("image has wrong mode")
    result = Image(enhance_blend(im1._native, im2._native, alpha))
    result.info.update(im1._metadata_info())
    return result


def composite(image1: Image, image2: Image, mask: Image) -> Image:
    """Select between images using an L or RGBA mask through native paste.

    Args:
        image1: First input image; its mode and dimensions must match the second image.
        image2: Second input image.
        mask: L or RGBA mask selecting pixels from the first image.

    Examples:
        ```python
        from blanket import Image

        image = Image.new("RGB", (8, 8), (40, 100, 180))
        other = Image.new("RGB", image.size, "white")
        mask = Image.new("L", image.size, 128)
        result = Image.composite(image, other, mask)
        ```
    """
    if image1.size != image2.size:
        raise ValueError("images do not match")
    result = image2.copy()
    result.paste(image1, (0, 0), mask)
    return result


def alpha_composite(im1: Image, im2: Image, *, linear_light: bool = False, transfer: str = "srgb") -> Image:
    """Return im2 composited over im1; both must be equal-sized RGBA of matching depth.

    High-depth inputs use straight-alpha source-over with nearest-integer
    rounding and retain their sample depth.

    Args:
        im1: First image; both inputs must have the same mode and dimensions.
        im2: Second image.
        linear_light: Decode the selected transfer before source-over blending.
        transfer: "srgb" or "linear"; applies when linear_light is true.

    Examples:
        ```python
        from blanket import Image

        background = Image.new("RGBA", (8, 8), "navy")
        overlay = Image.new("RGBA", (8, 8), (255, 0, 0, 128))
        result = Image.alpha_composite(background, overlay)
        ```
    """

    if transfer not in ("srgb", "linear"):
        raise ValueError("transfer must be 'srgb' or 'linear'")
    if linear_light or im1.mode == "RGBA;F":
        result = Image(ops_color_composite(im1._native, im2._native, linear_light and transfer == "srgb"))
    else:
        result = Image(image_alpha_composite(im1._native, im2._native))
    result.info.update(im1._metadata_info())
    return result


@dataclass(frozen=True)
class ImageHeader:
    """Image description and container metadata without a decoded raster.

    Mode uses the backend's header color conventions (some expose RGB/RGBA
    for converted or indexed storage). bit_depth reports header precision.
    Indexed PNG/GIF/XPM use P; packed PNG samples retain their encoded depth.
    For containers, describes the primary image, first TIFF page, GIF canvas,
    or largest ICO entry. Metadata uses the same keys as Image.info where
    available; frame timing and controls are not collected.

    Examples:
        ```python
        from blanket import Image

        header = Image.inspect("photo.png")
        print(header.size, header.mode, header.bit_depth)
        ```
    """

    format: str
    size: tuple[int, int]
    mode: str
    bit_depth: int
    info: dict[object, object]

    @property
    def width(self) -> int:
        """Image width in pixels."""
        return self.size[0]

    @property
    def height(self) -> int:
        """Image height in pixels."""
        return self.size[1]


def inspect(fp: str | bytes | os.PathLike[str] | os.PathLike[bytes] | BinaryIO, formats: list[str] | tuple[str, ...] | None = None) -> ImageHeader:
    """Read image headers and available metadata without decoding pixels.

    Accepts paths and binary streams, with an optional format allowlist.
    Reads the encoded file into memory and restores seekable stream positions.
    Pixel data and later frames are not validated. Decoded-pixel limits do not
    apply; codec header/metadata limits still apply. This does not make an
    otherwise unsupported image decodable with open().

    Args:
        fp: Image filename, path-like object, or binary stream.
        formats: Optional allowlist of format names to inspect.

    Examples:
        ```python
        from io import BytesIO
        from blanket import Image

        stream = BytesIO()
        Image.new("RGB", (8, 6)).save(stream, "PNG")
        header = Image.inspect(stream)
        print(header.size, header.mode, header.bit_depth)
        ```
    """
    if formats is not None and not isinstance(formats, (list, tuple)):
        raise TypeError("formats must be a list or tuple")
    position = None
    if hasattr(fp, "read"):
        with contextlib.suppress(AttributeError, OSError):
            position = fp.tell()
    try:
        data = _read_bytes(fp)
    finally:
        if position is not None:
            with contextlib.suppress(AttributeError, OSError):
                cast("BinaryIO", fp).seek(position)
    format, size, mode, bit_depth = _inspect(data, formats)
    info = read_metadata(data, "HEIF" if format == "AVIF" else format)
    return ImageHeader(format, size, mode, bit_depth, info)


def open(fp: str | bytes | os.PathLike[str] | os.PathLike[bytes] | BinaryIO, mode: str = "r", formats: list[str] | tuple[str, ...] | None = None) -> Image:
    """Open and eagerly decode a supported image.

    Args:
        fp: Image filename, path-like object, or binary stream. Decoding is eager.
        mode: Read mode; only `r` is supported.
        formats: Optional allowlist of format names to attempt.

    Examples:
        ```python
        from blanket import Image

        image = Image.new("RGB", (8, 8), (40, 100, 180))
        from io import BytesIO

        encoded = BytesIO()
        image.save(encoded, format="PNG")
        encoded.seek(0)
        with Image.open(encoded) as reopened:
            print(reopened.size)
        ```
    """
    if mode != "r":
        raise ValueError(f"bad mode {mode!r}")
    if formats is not None and not isinstance(formats, (list, tuple)):
        raise TypeError("formats must be a list or tuple")
    data = _read_bytes(fp)
    frames, controls, loop_count, default_image = _open_sequence(data, formats)
    image = Image(frames[0])
    if not hasattr(fp, "read"):
        image.filename = os.fspath(fp)

    metadata = read_tiff_metadata(data, len(frames)) if image.format == "TIFF" else None
    if metadata is None:
        image.info.update(read_metadata(data, image.format))
    decoded = [Image(frame) for frame in frames]
    for frame, (duration, disposal, blend) in zip(decoded, controls, strict=True):
        frame._info.update(next(metadata) if metadata is not None else image.info)
        if duration is not None:
            frame._info["duration"] = duration
        if loop_count is not None:
            frame._info["loop"] = loop_count
        if default_image:
            frame._info["default_image"] = True
        if image.format == "GIF":
            frame._info["disposal"] = disposal
        elif image.format == "PNG" and (len(frames) > 1 or loop_count is not None):
            frame._info.update(disposal=disposal, blend=blend)
    image._frames = decoded if len(decoded) > 1 else None
    image._info = decoded[0]._info
    return image


def open_region(
    fp: str | bytes | os.PathLike[str] | os.PathLike[bytes] | BinaryIO, box: tuple[int, int, int, int], *, fallback: str = "error", formats: list[str] | tuple[str, ...] | None = None
) -> Image:
    """Decode a nonempty region of the first image/page.

    TIFF unsigned 8/16-bit chunky L/RGB/straight RGBA reads only intersecting
    strips or tiles. Other layouts require fallback="full", which eagerly
    opens the source and crops frame zero. The default rejects fallback.
    Encoded input is read in full; normal source dimension limits still apply.
    """
    if fallback not in ("error", "full"):
        raise ValueError("fallback must be 'error' or 'full'")
    if formats is not None and not isinstance(formats, (list, tuple)):
        raise TypeError("formats must be a list or tuple")
    box = tuple(index(v) for v in box)
    if len(box) != 4:
        raise ValueError("box must contain four integers")
    data = _read_bytes(fp)
    header = inspect(BytesIO(data), formats)
    left, top, right, bottom = box
    if not (0 <= left < right <= header.size[0] and 0 <= top < bottom <= header.size[1]):
        raise ValueError("region must be a nonempty rectangle inside the image")
    native = _decode_region(data, box, formats)
    if native is None:
        if fallback == "error":
            raise NotImplementedError("region decoding unavailable; use fallback='full'")
        with open(BytesIO(data), formats=formats) as source:
            result = source.crop(box)
    else:
        result = Image(native)
        result.info.update(read_metadata(data, result.format))
    result.info["decode_region"] = box
    result.info["decode_fallback"] = native is None
    return result


def frombytes(mode: str, size: tuple[int, int], data: object, *, bit_depth: int = 8) -> Image:
    """Create an image from bytes; mode 1 uses row-packed bits.

    Args:
        mode: Image mode, including `1` (row-packed bits), indexed, integer, float, and 8-bit color modes.
        size: Output `(width, height)` in pixels.
        data: Contiguous packed pixel buffer. `I` uses little-endian int32, `F` uses little-endian float32, and `I;16B` uses big-endian uint16.
        bit_depth: Significant bits per channel: 8, 10, 12, or 16 for high-bit-depth storage, or 32 for `I` and `F`.

    Examples:
        ```python
        from blanket import Image

        image = Image.frombytes("RGB", (2, 1), bytes([255, 0, 0, 0, 255, 0]))
        ```
    """
    try:
        raw = bytes(data)
    except (TypeError, ValueError) as error:
        raise TypeError("data must be a bytes-like object") from error
    result = Image(_native_frombytes("L" if mode == "P" else mode, size, raw, bit_depth))
    if mode in ("P", "PA"):
        result.putpalette(_GRAY_PALETTE)
    return result


def fromarray(obj: object, mode: str | None = None, *, bit_depth: int | None = None) -> Image:
    """Create an image from uint8, uint16, int32, float32, or float64 array samples.

    Args:
        obj: Object exposing the array interface with uint8, uint16, int32, float32, or float64 samples.
        mode: Optional mode matching the array's channel layout.
        bit_depth: Significant bits per channel: 8, 10, 12, or 16. None retains or infers the depth. Integer and float arrays map to `I` and `F`.

    Examples:
        ```python
        import numpy as np
        from blanket import Image

        pixels = np.zeros((8, 8, 3), dtype=np.uint8)
        image = Image.fromarray(pixels)
        ```
    """
    return Image(_native_fromarray(obj, mode, bit_depth))


def _read_bytes(fp: str | bytes | os.PathLike[str] | os.PathLike[bytes] | BinaryIO) -> bytes:
    """Read encoded bytes from a path or stream, rewinding when supported."""
    if hasattr(fp, "read"):
        stream = fp
        with contextlib.suppress(AttributeError, OSError):
            stream.seek(0)
        data = stream.read()
        if not isinstance(data, bytes):
            raise ValueError("binary data must be used to open an image")
        return data
    return Path(os.fsdecode(os.fspath(fp))).read_bytes()


def _output_format(fp: object, requested: str | None) -> str:
    """Resolve a supported save format from an explicit name or filename extension."""
    if requested is not None:
        normalized = requested.upper().replace(" ", "")
        normalized = _FORMAT_ALIASES.get(normalized, normalized)
        if normalized not in _SAVE_OPTIONS:
            raise ValueError(f"unsupported image format {requested!r}")
        return normalized
    if hasattr(fp, "write"):
        raise ValueError("unknown file extension; specify format")
    extension = os.path.splitext(os.fsdecode(os.fspath(fp)))[1].lower()
    try:
        return _EXTENSIONS[extension]
    except KeyError as error:
        raise ValueError(f"unknown file extension: {extension}") from error


def _save_options(format: str, supplied: dict[str, object]) -> dict[str, object]:
    """Validate format-specific save options and supply native encoder defaults."""
    unknown = supplied.keys() - _SAVE_OPTIONS[format]
    if unknown:
        names = ", ".join(sorted(unknown))
        raise TypeError(f"unsupported {format} save option(s): {names}")

    lossless = supplied.get("lossless", False)
    if not isinstance(lossless, bool):
        raise TypeError("lossless must be a bool")
    quality = 100 if lossless else _bounded_int("quality", supplied.get("quality", 90), 1, 100)
    compress_level = _bounded_int("compress_level", supplied.get("compress_level", 6), 0, 9)
    effort = _bounded_int("effort", supplied.get("effort", 7), 1, 10)

    values = {"quality": quality, "compress_level": compress_level, "lossless": lossless, "effort": effort}
    if format == "TIFF":
        compression = supplied.get("compression", "raw")
        methods = {"raw": 0, "tiff_lzw": 1, "tiff_adobe_deflate": 2, "tiff_deflate": 2, "deflate": 2, "jpeg": 3}
        if not isinstance(compression, str) or compression not in methods:
            raise ValueError(f"unsupported TIFF compression {compression!r}")
        values["tiff_compression"] = methods[compression]
    return values


def _bounded_int(name: str, value: object, minimum: int, maximum: int) -> int:
    """Require a non-boolean integer within inclusive bounds."""
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an int")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


__all__ = [
    "AFFINE",
    "BICUBIC",
    "BILINEAR",
    "BOX",
    "EXTENT",
    "FLIP_LEFT_RIGHT",
    "FLIP_TOP_BOTTOM",
    "HAMMING",
    "LANCZOS",
    "MESH",
    "NEAREST",
    "PERSPECTIVE",
    "QUAD",
    "ROTATE_90",
    "ROTATE_180",
    "ROTATE_270",
    "TRANSPOSE",
    "TRANSVERSE",
    "Exif",
    "Image",
    "ImageTransformHandler",
    "Resampling",
    "SupportsGetData",
    "Transform",
    "Transpose",
    "blend",
    "composite",
    "fromarray",
    "frombytes",
    "open",
]
