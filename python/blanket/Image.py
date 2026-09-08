"""Pillow-shaped image entry points backed by Blanket's Rust extension."""

from __future__ import annotations

import builtins
import math
import os
from collections.abc import Sequence
from enum import IntEnum
from operator import index
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO, Protocol, Self

if TYPE_CHECKING:
    from .ImageFilter import Filter
    from .ImagePalette import ImagePalette

from ._blanket import _Image, open_bytes
from ._blanket import fromarray as _native_fromarray
from ._blanket import frombytes as _native_frombytes

_EXTENSIONS = {".png": "PNG", ".jpg": "JPEG", ".jpeg": "JPEG", ".jxl": "JXL"}


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
    def getdata(self) -> tuple[int, Sequence[object]]: ...


class ImageTransformHandler:
    """Base class for custom transformation handlers."""

    def transform(self, size: tuple[int, int], image: Image, resample: int = Resampling.NEAREST, fill: int = 1) -> Image:
        raise NotImplementedError


class Image:
    """An 8-bit image whose pixels and supported operations live in Rust."""

    def __init__(self, native: _Image) -> None:
        self._native = native
        self.filename: str | bytes = ""
        self.palette: ImagePalette | None = None
        self._info: dict[object, object] = {}
        palette_data = native.palette_data()
        if palette_data is not None:
            from .ImagePalette import ImagePalette

            self.palette = ImagePalette(palette_data[0], bytes(palette_data[1]))

    @property
    def mode(self) -> str:
        return self._native.mode

    @property
    def size(self) -> tuple[int, int]:
        return self._native.size

    @property
    def width(self) -> int:
        return self._native.width

    @property
    def height(self) -> int:
        return self._native.height

    @property
    def format(self) -> str | None:
        return self._native.format

    @property
    def info(self) -> dict[object, object]:
        return self._info

    @property
    def is_animated(self) -> bool:
        """Blanket exposes a single frame for every supported image."""
        return False

    @property
    def has_transparency_data(self) -> bool:
        """Whether alpha or transparency metadata exists, even if opaque."""
        return self.mode == "RGBA" or "transparency" in self.info or (self.palette is not None and self.palette.mode == "RGBA")

    def load(self) -> None:
        """Validate that this eagerly loaded image remains open."""

        self._native.load()
        self._sync_palette()

    def close(self) -> None:
        self._native.close()

    def convert(self, mode: str) -> Image:
        """Return a new image converted to `L`, `RGB`, or `RGBA`."""

        if mode == "P" and self.mode != "P":
            return self.quantize()
        self._sync_palette()
        result = Image(self._native.convert(mode))
        result.info.update(self.info)
        return result

    def _sync_palette(self) -> None:
        if self.mode == "P" and self.palette is not None:
            self._native.set_palette(self.palette.mode, self.palette.tobytes())

    def copy(self) -> Image:
        """Return an independent copy of the pixels, palette, and metadata."""
        self._sync_palette()
        result = Image(self._native.copy())
        result.info.update(self.info)
        return result

    def getpixel(self, xy: tuple[int, int] | list[int]) -> int | tuple[int, ...]:
        """Return a pixel value, accepting negative coordinates as Pillow does."""
        self.load()
        if not isinstance(xy, (tuple, list)):
            raise TypeError("argument must be a sequence")
        if len(xy) != 2:
            raise TypeError("argument must be sequence of length 2")
        coordinates = tuple(int(v) if isinstance(v, float) else index(v) for v in xy)
        return self._native.getpixel(coordinates)

    def getpalette(self, rawmode: str | None = "RGB") -> list[int] | None:
        """Return interleaved palette entries, or None for non-palette images."""
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

    def putpalette(self, data: Sequence[int] | bytes | ImagePalette, rawmode: str = "RGB") -> None:
        """Attach an RGB or RGBA palette to an L or P image."""
        from .ImagePalette import ImagePalette

        self.load()
        if self.mode not in ("L", "P"):
            raise ValueError("illegal image mode")
        palette = data.copy() if isinstance(data, ImagePalette) else ImagePalette(rawmode, bytes(data))
        self._native.set_palette(palette.mode, palette.tobytes())
        self.palette = palette

    def quantize(self, colors: int = 256, method: int | None = None, kmeans: int = 0, palette: Image | None = None, dither: Dither = Dither.FLOYDSTEINBERG) -> Image:
        """Return an indexed P image using a generated or supplied palette.

        MEDIANCUT, MAXCOVERAGE, and FASTOCTREE run in the native backend.
        LIBIMAGEQUANT is unavailable in this build. Generated palette ordering
        and color choices may differ from Pillow's implementations.
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
        return self._native.tobytes()

    def filter(self, filter: Filter | type[Filter]) -> Image:
        """Return a filtered image, accepting a filter instance or class."""
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

    def split(self) -> tuple[Image, ...]:
        """Return independent L images for each band, in channel order."""
        from ._blanket import ops_split

        bands = tuple(Image(native) for native in ops_split(self._native))
        for band in bands:
            band.info.update(self.info)
        return bands

    def reduce(self, factor: int | tuple[int, int], box: tuple[int, int, int, int] | None = None) -> Image:
        """Average integer blocks, rounding the output dimensions up.

        factor can specify horizontal and vertical factors separately.
        box selects a nonempty source rectangle within the image.
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
        """
        from ._blanket import ops_entropy

        self.load()
        if mask is not None:
            mask.load()
        return ops_entropy(self._native, None if mask is None else mask._native)

    def transpose(self, method: int) -> Image:
        """Return a flipped or right-angle rotated copy using ``Transpose``."""
        from ._blanket import ops_transpose

        method = index(method)
        if method not in range(7):
            raise ValueError("No such transpose operation")
        result = Image(ops_transpose(self._native, (2, 4, 8, 3, 6, 5, 7)[method]))
        result.info.update(self.info)
        return result

    def transform(self, size: tuple[int, int], method: int | ImageTransformHandler | SupportsGetData, data: Sequence[object] | None = None, resample: int = Resampling.NEAREST, fill: int = 1, fillcolor: str | int | tuple[int, ...] | None = None) -> Image:
        """Map source pixels to a new canvas using a ``Transform`` method.

        AFFINE and PERSPECTIVE use inverse mapping coefficients. EXTENT takes
        a source rectangle. QUAD takes NW, SW, SE, NE source corners; MESH
        takes (destination rectangle, source quad) pairs in drawing order.
        Supports NEAREST, BILINEAR and BICUBIC, plus optional fillcolor.
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
        """
        from ._blanket import ops_reduce, ops_resize, ops_transpose

        method = Resampling.BICUBIC if resample is None else resample
        if method not in range(6):
            raise ValueError(f"Unknown resampling filter ({method})")
        if self.mode == "P":
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

    def rotate(self, angle: float, resample: int = Resampling.NEAREST, expand: bool = False, center: tuple[float, float] | None = None, translate: tuple[float, float] | None = None, fillcolor: str | int | tuple[int, ...] | None = None) -> Image:
        """Return a copy rotated counterclockwise by an angle in degrees.

        Supports NEAREST (default), BILINEAR, and BICUBIC. The default center
        is the image midpoint; translate shifts the result after rotation.
        expand enlarges the canvas assuming the default center and no translation.
        fillcolor colors pixels outside the source image.
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
        """Return an equivalent Pillow image when Pillow is installed."""

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
        """Save this image as PNG, JPEG, or JPEG XL."""

        output_format = _output_format(fp, format)
        self._sync_palette()
        values = _save_options(output_format, options)
        encoded = self._native._encode(output_format, **values)
        _write_bytes(fp, encoded)

    def __enter__(self) -> Self:
        self.load()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"<blanket.Image.Image image mode={self.mode} size={self.width}x{self.height}>"


def open(fp: str | bytes | os.PathLike[str] | os.PathLike[bytes] | BinaryIO, mode: str = "r", formats: list[str] | tuple[str, ...] | None = None) -> Image:
    """Open and eagerly decode a supported image."""

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


def frombytes(mode: str, size: tuple[int, int], data: object) -> Image:
    """Create an image from tightly packed 8-bit pixel data."""

    try:
        raw = bytes(data)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise TypeError("data must be a bytes-like object") from error
    result = Image(_native_frombytes("L" if mode == "P" else mode, size, raw))
    if mode == "P":
        result.putpalette(bytes(v for v in range(256) for _ in range(3)))
    return result


def fromarray(obj: object, mode: str | None = None) -> Image:
    """Create an image from an 8-bit object exposing the array interface."""

    return Image(_native_fromarray(obj, mode))


def _read_bytes(fp: str | bytes | os.PathLike[str] | os.PathLike[bytes] | BinaryIO) -> bytes:
    if hasattr(fp, "read"):
        stream = fp
        try:
            stream.seek(0)  # type: ignore[union-attr]
        except (AttributeError, OSError):
            pass
        data = stream.read()  # type: ignore[union-attr]
        if not isinstance(data, bytes):
            raise ValueError("binary data must be used to open an image")
        return data
    return Path(os.fsdecode(os.fspath(fp))).read_bytes()


def _write_bytes(fp: str | bytes | os.PathLike[str] | os.PathLike[bytes] | BinaryIO, data: bytes) -> None:
    if hasattr(fp, "write"):
        fp.write(data)  # type: ignore[union-attr]
        return
    with builtins.open(fp, "wb") as output:
        output.write(data)


def _output_format(fp: object, requested: str | None) -> str:
    if requested is not None:
        normalized = requested.upper().replace(" ", "")
        aliases = {"JPG": "JPEG", "JPEGXL": "JXL"}
        normalized = aliases.get(normalized, normalized)
        if normalized not in {"PNG", "JPEG", "JXL"}:
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
    allowed = {"PNG": {"compress_level"}, "JPEG": {"quality"}, "JXL": {"quality", "lossless", "effort"}}[format]
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
    "fromarray",
    "frombytes",
    "open",
]
