"""Pillow-shaped image entry points backed by Blanket's Rust extension."""

from __future__ import annotations

import builtins
import contextlib
import math
import os
from enum import IntEnum
from operator import index
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO, Protocol, Self

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from .ImageFilter import Filter
    from .ImagePalette import ImagePalette

from ._blanket import _encode, _Image, fromarray as _native_fromarray, frombytes as _native_frombytes, open_bytes

_EXTENSIONS = {
    ".bmp": "BMP",
    ".gif": "GIF",
    ".ico": "ICO",
    ".png": "PNG",
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
}
_BANDS = {"1": ("1",), "L": ("L",), "LA": ("L", "A"), "P": ("P",), "PA": ("P", "A"), "RGB": ("R", "G", "B"), "RGBA": ("R", "G", "B", "A")}


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
    """An image whose pixels and supported operations live in Rust."""

    def __init__(self, native: _Image) -> None:
        self._native = native
        self._bands = _BANDS[native.mode]
        self.filename: str | bytes = ""
        self.palette: ImagePalette | None = None
        self._info: dict[object, object] = {}
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
        """Significant bits per channel (8, 10, 12, or 16).

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
        """Blanket exposes a single frame for every supported image.

        Examples:
            ```python
            from blanket import Image

            image = Image.new("RGB", (8, 8), (40, 100, 180))
            print(image.is_animated)
            ```
        """
        return False

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

    def convert(self, mode: str, *, bit_depth: int | None = None) -> Image:
        """Return a new image converted to a supported direct or indexed mode.

        Args:
            mode: Destination mode: `1`, `L`, `LA`, `RGB`, `RGBA`, `P`, or `PA`.
            bit_depth: Output sample depth, or None to retain the input depth.

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
                alpha = bytes(entries[slot * 4 + 3] if slot * 4 + 3 < len(entries) else 255 for slot in source.tobytes())
            else:
                alpha = bytes([255]) * (self.width * self.height)
            result = frombytes("PA", source.size, bytes(v for pair in zip(source.tobytes(), alpha, strict=True) for v in pair))
            if source.palette is not None:
                result.putpalette(source.palette)
            return result
        if mode == "PA" and self.mode == "PA":
            return self.copy()
        if mode == "P" and self.mode == "PA":
            result = frombytes("P", self.size, self.tobytes()[::2])
            if self.palette is not None:
                result.putpalette(self.palette)
            return result
        if mode == "P" and self.mode != "P":
            return (self.convert("RGB") if self.mode in ("1", "LA") else self).quantize()
        self._sync_palette()
        result = Image(self._native.convert(mode, bit_depth))
        result.info.update(self.info)
        return result

    def _sync_palette(self) -> None:
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
        result.info.update(self.info)
        return result

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
            return [v for i, v in enumerate(data) if i % 4 != 3]
        return [v for i in range(0, len(data), 3) for v in (*data[i : i + 3], 255)]

    def putpixel(self, xy: tuple[int, int], value: int | tuple[int, ...]) -> None:
        """Write an 8-bit pixel, accepting negative coordinates and clipping values.

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
        self._native.putpixel(tuple(index(v) for v in xy), value)

    def getdata(self, band: int | None = None) -> list[int | tuple[int, ...]]:
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

    def get_flattened_data(self, band: int | None = None) -> tuple[int | tuple[int, ...], ...]:
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
        return tuple(self.getdata(band))

    def putdata(self, data: Sequence[int | float | tuple[int, ...]], scale: float = 1.0, offset: float = 0.0) -> None:
        """Write 8-bit pixels in row order; scale/offset apply to single-band data.

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

    def getcolors(self, maxcolors: int = 256) -> list[tuple[int, int | tuple[int, ...]]] | None:
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
        """Attach an RGB or RGBA palette to an L or P image.

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
        from ._blanket import quantize

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
        result = Image(quantize(source._native, colors, method, kmeans, None if palette is None else palette._native, index(dither)))
        result.info.update(self.info)
        return result

    def tobytes(self) -> bytes:
        """Return packed pixels; high-depth samples use little-endian uint16 storage.

        Examples:
            ```python
            from blanket import Image

            image = Image.new("RGB", (8, 8), (40, 100, 180))
            pixels = image.tobytes()
            ```
        """
        return self._native.tobytes()

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
        from ._blanket import filter_merge
        from .ImageFilter import MultibandFilter

        self.load()
        if callable(filter):
            filter = filter()
        if not hasattr(filter, "filter"):
            raise TypeError("filter argument should be ImageFilter.Filter instance or class")
        if self.mode == "L" or isinstance(filter, MultibandFilter):
            result = Image(filter.filter(self._native))
            result.info.update(self.info)
            return result
        bands = [filter.filter(band._native) for band in self.split()]
        result = Image(filter_merge(self.mode, bands))
        result.info.update(self.info)
        return result

    def paste(self, im: Image | str | int | tuple[int, ...], box: Image | tuple[int, ...] | None = None, mask: Image | None = None) -> None:
        """Paste pixels or a color, optionally interpolating through an L/RGBA mask.

        Args:
            im: Source image or fill color.
            box: Destination origin or rectangle, or a mask image as the second positional argument.
            mask: Optional mask selecting pixels. Histogram operations require an L mask; compositing also accepts RGBA alpha.

        Examples:
            ```python
            from blanket import Image

            image = Image.new("RGB", (8, 8), (40, 100, 180))
            image.paste("red", (0, 0, 4, 4))
            ```
        """
        from ._blanket import image_paste

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
            source = im if im.mode == self.mode else im.convert(self.mode)
        else:
            if self.mode == "P" and isinstance(im, (str, tuple)) and not (isinstance(im, tuple) and len(im) == 1):
                raise TypeError("color must be int or single-element tuple")
            source = new(self.mode, size, im)
        # Snapshot aliases before acquiring the native mutable destination borrow.
        source = source.copy() if source._native is self._native else source
        if mask is not None and mask._native is self._native:
            mask = mask.copy()
        image_paste(self._native, source._native, box[:2], mask._native if mask is not None else None, not isinstance(im, Image))

    def alpha_composite(self, im: Image, dest: tuple[int, int] = (0, 0), source: tuple[int, ...] = (0, 0)) -> None:
        """Composite an RGBA source region onto this image in place.

        Args:
            im: RGBA source image.
            dest: Destination `(x, y)` position.
            source: Source `(x, y)` origin or `(left, top, right, bottom)` region.

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
        if box == (0, 0, *self.size):
            from ._blanket import image_alpha_composite_inplace

            overlay = overlay.copy() if overlay._native is self._native else overlay
            image_alpha_composite_inplace(self._native, overlay._native)
            return
        background = self if box == (0, 0, *self.size) else self.crop(box)
        self.paste(alpha_composite(background, overlay), box)

    def putalpha(self, alpha: Image | int) -> None:
        """Replace alpha in place; RGB images become RGBA.

        L and P inputs are unsupported because Blanket does not implement LA/PA.

        Args:
            alpha: L image matching the image size, or a constant alpha value from 0 through 255.

        Examples:
            ```python
            from blanket import Image

            image = Image.new("RGB", (8, 8), (40, 100, 180))
            image.putalpha(128)
            ```
        """
        from ._blanket import image_putalpha

        if self.mode not in ("RGB", "RGBA"):
            raise ValueError("putalpha requires RGB or RGBA; LA and PA modes are not supported")
        band = alpha if isinstance(alpha, Image) else new("L", self.size, index(alpha))
        if band.mode != "L" or band.size != self.size:
            raise ValueError("illegal image mode or size for alpha")
        image_putalpha(self._native, band._native)
        self.palette = None
        self._bands = _BANDS["RGBA"]

    def split(self) -> tuple[Image, ...]:
        """Return independent L images for each band, in channel order.

        Examples:
            ```python
            from blanket import Image

            image = Image.new("RGB", (8, 8), (40, 100, 180))
            red, green, blue = image.split()
            ```
        """
        from ._blanket import ops_split

        if self.mode == "1":
            return (self.copy(),)
        bands = tuple(Image(native) for native in ops_split(self._native))
        for band in bands:
            band.info.update(self.info)
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
        """Return an independent L image for a channel name or index.

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
        if self.mode == "1":
            return self.copy()
        result = Image(self._native.getchannel(channel))
        result.info.update(self.info)
        return result

    def histogram(self, mask: Image | None = None, extrema: tuple[float, float] | None = None) -> list[int]:
        """Return 256 bins per band for 8-bit pixels; extrema is ignored.

        Args:
            mask: Optional mask selecting pixels. Histogram operations require an L mask; compositing also accepts RGBA alpha.
            extrema: Accepted for Pillow compatibility; ignored.

        Examples:
            ```python
            from blanket import Image

            image = Image.new("RGB", (8, 8), (40, 100, 180))
            counts = image.histogram()
            ```
        """
        from ._blanket import ops_histogram

        self.load()
        if mask is not None:
            mask.load()
            if mask.mode != "L":
                raise ValueError("bad transparency mask")
        return ops_histogram(self._native, None if mask is None else mask._native)

    def getextrema(self) -> tuple[int, int] | tuple[tuple[int, int], ...] | None:
        """Return the minimum and maximum sample value of each band.

        Examples:
            ```python
            from blanket import Image

            image = Image.new("RGB", (8, 8), (40, 100, 180))
            extrema = image.getextrema()
            ```
        """
        ranges = self._native.getextrema()
        if not ranges:
            return None
        return ranges[0] if len(ranges) == 1 else tuple(ranges)

    def getbbox(self, *, alpha_only: bool = True) -> tuple[int, int, int, int] | None:
        """Return the nonzero bounding box, using RGBA alpha by default.

        Args:
            alpha_only: For RGBA images, inspect only alpha when true; otherwise inspect all channels.

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

        Args:
            lut: 256 entries per input channel, or a callable evaluated for each possible 8-bit value.
            mode: Optional output mode; must match the input mode.

        Examples:
            ```python
            from blanket import Image

            image = Image.new("RGB", (8, 8), (40, 100, 180))
            result = image.point(lambda value: 255 - value)
            ```
        """
        from ._blanket import ops_lut

        self.load()
        if mode is not None and mode != self.mode:
            raise ValueError("point mode conversion is not supported")
        values = [lut(i) for i in range(256)] * len(self.getbands()) if callable(lut) else list(lut)
        if len(values) != 256 * len(self.getbands()):
            raise ValueError("wrong number of lut entries")
        result = Image(ops_lut(self._native, [max(0, min(255, round(v))) for v in values]))
        result.info.update(self.info)
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
        from ._blanket import ops_reduce

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
        result.info.update(self.info)
        return result

    def entropy(self, mask: Image | None = None, extrema: tuple[float, float] | None = None) -> float:
        """Return Shannon entropy over all channel histogram bins.

        An L mask selects pixels with nonzero values. As in Pillow, extrema
        is ignored for the supported 8-bit modes.

        Args:
            mask: Optional mask selecting pixels. Histogram operations require an L mask; compositing also accepts RGBA alpha.
            extrema: Accepted for Pillow compatibility; ignored.

        Examples:
            ```python
            from blanket import Image

            image = Image.new("RGB", (8, 8), (40, 100, 180))
            entropy = image.entropy()
            ```
        """
        from ._blanket import ops_entropy

        self.load()
        if mask is not None:
            mask.load()
        return ops_entropy(self._native, None if mask is None else mask._native)

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
        from ._blanket import ops_transpose

        method = index(method)
        if method not in range(7):
            raise ValueError("No such transpose operation")
        result = Image(ops_transpose(self._native, (2, 4, 8, 3, 6, 5, 7)[method]))
        result.info.update(self.info)
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
        from ._blanket import ops_affine, ops_warp
        from ._color import color_pixel

        if isinstance(method, ImageTransformHandler):
            return method.transform(size, self, resample=resample, fill=fill)
        if hasattr(method, "getdata"):
            method, data = method.getdata()
        if data is None:
            raise ValueError("missing method data")
        if method not in tuple(Transform):
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
        color = color_pixel(fillcolor, self.mode)
        if self.mode == "RGBA" and resample != 0 and isinstance(fillcolor, str):
            color[3] = 255

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
            native = ops_affine(self._native, size, matrix, resample, color)
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
            native = ops_warp(self._native, size, mesh, (resample, method == Transform.PERSPECTIVE), color if fillcolor is not None else None)
        result = Image(native)
        result.info.update(self.info)
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
        from ._blanket import ops_canvas, ops_transpose

        if box is None:
            self.load()
            native = ops_transpose(self._native, 1)
        else:
            if box[2] < box[0]:
                raise ValueError("Coordinate 'right' is less than 'left'")
            if box[3] < box[1]:
                raise ValueError("Coordinate 'lower' is less than 'upper'")
            self.load()
            left, upper, right, lower = (round(value) for value in box)
            native = ops_canvas(self._native, (right - left, lower - upper), (-left, -upper), [0] * len(self.mode))
        result = Image(native)
        result.info.update(self.info)
        return result

    def resize(self, size: tuple[int, int], resample: int | None = None, box: tuple[float, float, float, float] | None = None, reducing_gap: float | None = None) -> Image:
        """Return a resized copy, using BICUBIC unless a filter is specified.

        ``box`` selects a source rectangle within the image. ``reducing_gap``
        optionally enables integer reduction before filtering (at least 1.0).

        Args:
            size: Output `(width, height)` in pixels.
            resample: Filter from `Image.Resampling`; None selects BICUBIC, or NEAREST for indexed images.
            box: Optional floating-point source rectangle.
            reducing_gap: Optional pre-reduction optimization threshold; must be at least 1.0.

        Examples:
            ```python
            from blanket import Image

            image = Image.new("RGB", (8, 8), (40, 100, 180))
            result = image.resize((16, 16), Image.Resampling.LANCZOS)
            ```
        """
        from ._blanket import ops_reduce, ops_resize, ops_transpose

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
        if size == self.size and box == (0, 0, self.width, self.height):
            native = ops_transpose(native, 1)
        else:
            # Pillow's alpha-aware path does not use integer reduction.
            if reducing_gap is not None and method != Resampling.NEAREST and self.mode != "RGBA":
                fx = max(1, int((box[2] - box[0]) / size[0] / reducing_gap))
                fy = max(1, int((box[3] - box[1]) / size[1] / reducing_gap))
                if fx > 1 or fy > 1:
                    support = {1: 3, 2: 1, 3: 2, 4: 0.5, 5: 1}[method] - 0.5
                    sx = support * (box[2] - box[0]) / size[0]
                    sy = support * (box[3] - box[1]) / size[1]
                    safe = (max(0, int(box[0] - sx)), max(0, int(box[1] - sy)), min(self.width, math.ceil(box[2] + sx)), min(self.height, math.ceil(box[3] + sy)))
                    native = ops_reduce(native, (fx, fy), safe)
                    box = ((box[0] - safe[0]) / fx, (box[1] - safe[1]) / fy, (box[2] - safe[0]) / fx, (box[3] - safe[1]) / fy)
            native = ops_resize(native, size, method, box)
        result = Image(native)
        result.info.update(self.info)
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
        from ._blanket import ops_affine, ops_transpose
        from ._color import color_pixel

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
            fill = color_pixel(fillcolor, self.mode)
            if self.mode == "RGBA" and resample != Resampling.NEAREST and isinstance(fillcolor, str):
                # Pillow parses strings in its intermediate RGBa mode as RGB.
                fill[3] = 255
            result = Image(ops_affine(self._native, (w, h), (a, b, c, d, e, f), resample, fill))
        result.info.update(self.info)
        return result

    def to_pillow(self) -> object:
        """Return an equivalent Pillow image for interoperability.

        Examples:
            ```python
            from blanket import Image

            image = Image.new("RGB", (8, 8), (40, 100, 180))
            pillow_image = image.to_pillow()
            ```
        """
        if self.bit_depth != 8:
            raise ValueError("to_pillow requires 8-bit pixels; use convert(..., bit_depth=8) explicitly")

        try:
            from PIL import Image as PillowImage
        except ImportError as error:
            raise ImportError("Pillow is required for to_pillow(); install blanket[test] or Pillow") from error
        result = PillowImage.frombytes(self.mode, self.size, self.tobytes())
        if self.palette is not None:
            result.putpalette(self.palette.tobytes(), self.palette.mode)
        result.info.update(self.info)
        return result

    def save(self, fp: str | bytes | os.PathLike[str] | os.PathLike[bytes] | BinaryIO, format: str | None = None, **options: object) -> None:
        """Save as PNG, JPEG, JPEG XL, TIFF, WebP, HEIF, AVIF, BMP, GIF, ICO, or PDF.

        Pass a ``LosslessImageCompressor`` or ``LossyImageCompressor`` to
        ``compressor`` to optimize this save.
        Omitting it (or passing None) uses the normal encoder settings.

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
        from .Compressor import LosslessImageCompressor, LossyImageCompressor

        compressor = options.pop("compressor", None)
        if compressor is not None and not isinstance(compressor, (LosslessImageCompressor, LossyImageCompressor)):
            raise TypeError("compressor must be a LosslessImageCompressor, LossyImageCompressor, or None")
        output_format = _output_format(fp, format)
        self._sync_palette()
        values = _save_options(output_format, options)
        encoded = _encode(self._native, output_format, **values, compressor=None if compressor is None else compressor._native)
        _write_bytes(fp, encoded)

    def __enter__(self) -> Self:
        self.load()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"<blanket.Image.Image image mode={self.mode} size={self.width}x{self.height}>"


def new(mode: str, size: tuple[int, int], color: str | int | tuple[int, ...] | None = 0) -> Image:
    """Create an 8-bit direct or indexed image filled with color.

    Args:
        mode: Image mode, such as `L`, `RGB`, or `RGBA`.
        size: Output `(width, height)` in pixels.
        color: Fill value, channel tuple, or CSS color string.

    Examples:
        ```python
        from blanket import Image

        image = Image.new("RGB", (64, 64), "navy")
        ```
    """
    from ._blanket import image_new
    from ._color import color_pixel

    if not isinstance(size, (list, tuple)) or len(size) != 2:
        raise ValueError("Size must be a list or tuple of length 2")
    dimensions = tuple(index(v) for v in size)
    if min(dimensions) < 0:
        raise ValueError("Width and height must be >= 0")
    native_mode = "L" if mode == "P" else mode
    if mode not in ("1", "L", "LA", "RGB", "RGBA", "P", "PA"):
        raise ValueError(f"unsupported image mode {mode!r}")
    palette_color = mode == "P" and (isinstance(color, str) or (isinstance(color, tuple) and len(color) in (3, 4)))
    result = Image(image_new(native_mode, dimensions, color_pixel(0 if palette_color else color, native_mode)))
    if mode in ("P", "PA"):
        result.putpalette(bytes(color_pixel(color, "RGB")) if palette_color else bytes(768))
    return result


def merge(mode: str, bands: Sequence[Image]) -> Image:
    """Interleave L bands into an independent L, RGB, or RGBA image.

    Args:
        mode: Image mode, such as `L`, `RGB`, or `RGBA`.
        bands: Sequence of single-channel L images, one per output band.

    Examples:
        ```python
        from blanket import Image

        image = Image.new("RGB", (8, 8), (40, 100, 180))
        red, green, blue = image.split()
        result = Image.merge("RGB", (blue, green, red))
        ```
    """
    from ._blanket import filter_merge

    bands = tuple(bands)
    if any(band.mode != "L" for band in bands):
        raise ValueError("mode mismatch")
    return Image(filter_merge(mode, [band._native for band in bands]))


def blend(im1: Image, im2: Image, alpha: float) -> Image:
    """Interpolate equal-sized 8-bit images, clipping extrapolated values.

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
    from ._blanket import enhance_blend

    im1.load()
    im2.load()
    if im1.mode == "P" or im2.mode == "P":
        raise ValueError("image has wrong mode")
    result = Image(enhance_blend(im1._native, im2._native, alpha))
    result.info.update(im1.info)
    return result


def composite(image1: Image, image2: Image, mask: Image) -> Image:
    """Select between images using an L or RGBA mask through native paste.

    Args:
        image1: First input image; its mode and dimensions must match the second image.
        image2: Second input image.
        mask: Optional mask selecting pixels. Histogram operations require an L mask; compositing also accepts RGBA alpha.

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


def alpha_composite(im1: Image, im2: Image) -> Image:
    """Return im2 composited over im1; both must be equal-sized 8-bit RGBA.

    Args:
        im1: First image; both inputs must have the same mode and dimensions.
        im2: Second image.

    Examples:
        ```python
        from blanket import Image

        background = Image.new("RGBA", (8, 8), "navy")
        overlay = Image.new("RGBA", (8, 8), (255, 0, 0, 128))
        result = Image.alpha_composite(background, overlay)
        ```
    """
    from ._blanket import image_alpha_composite

    result = Image(image_alpha_composite(im1._native, im2._native))
    result.info.update(im1.info)
    return result


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
    image = Image(open_bytes(data, formats))
    if not hasattr(fp, "read"):
        image.filename = os.fspath(fp)
    from ._exif import read_metadata

    image.info.update(read_metadata(data, image.format))
    return image


def frombytes(mode: str, size: tuple[int, int], data: object, *, bit_depth: int = 8) -> Image:
    """Create an image from bytes; mode 1 uses row-packed bits.

    Args:
        mode: Image mode, such as `L`, `RGB`, or `RGBA`.
        size: Output `(width, height)` in pixels.
        data: Contiguous packed pixel buffer; depths above 8 use little-endian uint16 samples.
        bit_depth: Significant bits per channel: 8, 10, 12, or 16. None retains or infers the depth where supported.

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
        result.putpalette(bytes(v for v in range(256) for _ in range(3)))
    return result


def fromarray(obj: object, mode: str | None = None, *, bit_depth: int | None = None) -> Image:
    """Create an image from uint8 or uint16 samples exposing the array interface.

    Args:
        obj: Object exposing the array interface with uint8 or uint16 samples.
        mode: Optional mode matching the array's channel layout.
        bit_depth: Significant bits per channel: 8, 10, 12, or 16. None retains or infers the depth where supported.

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
    if hasattr(fp, "read"):
        stream = fp
        with contextlib.suppress(AttributeError, OSError):
            stream.seek(0)
        data = stream.read()
        if not isinstance(data, bytes):
            raise ValueError("binary data must be used to open an image")
        return data
    return Path(os.fsdecode(os.fspath(fp))).read_bytes()


def _write_bytes(fp: str | bytes | os.PathLike[str] | os.PathLike[bytes] | BinaryIO, data: bytes) -> None:
    if hasattr(fp, "write"):
        fp.write(data)
        return
    with builtins.open(fp, "wb") as output:
        output.write(data)


def _output_format(fp: object, requested: str | None) -> str:
    if requested is not None:
        normalized = requested.upper().replace(" ", "")
        aliases = {"JPG": "JPEG", "JPEGXL": "JXL", "TIF": "TIFF", "HEIC": "HEIF"}
        normalized = aliases.get(normalized, normalized)
        if normalized not in {"PNG", "JPEG", "JXL", "TIFF", "WEBP", "HEIF", "AVIF", "PDF", "BMP", "GIF", "ICO"}:
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
    allowed = {
        "BMP": set(),
        "GIF": set(),
        "ICO": set(),
        "PNG": {"compress_level"},
        "JPEG": {"quality"},
        "JXL": {"quality", "lossless", "effort"},
        "TIFF": set(),
        "PDF": set(),
        "WEBP": {"quality", "lossless"},
        "HEIF": {"quality", "lossless"},
        "AVIF": {"quality", "effort"},
    }[format]
    unknown = supplied.keys() - allowed
    if unknown:
        names = ", ".join(sorted(unknown))
        raise TypeError(f"unsupported {format} save option(s): {names}")

    lossless = supplied.get("lossless", False)
    if not isinstance(lossless, bool):
        raise TypeError("lossless must be a bool")
    quality = 100 if lossless else _bounded_int("quality", supplied.get("quality", 90), 1, 100)
    compress_level = _bounded_int("compress_level", supplied.get("compress_level", 6), 0, 9)
    effort = _bounded_int("effort", supplied.get("effort", 7), 1, 10)

    return {"quality": quality, "compress_level": compress_level, "lossless": lossless, "effort": effort}


def _bounded_int(name: str, value: object, minimum: int, maximum: int) -> int:
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
