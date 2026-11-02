"""Pillow-shaped image operations, backed by native Blanket pixel kernels.

Geometry, canvas, and resampling work on the current Image modes, including
`1`, `LA`, `PA`, integer, float, and print/luma/chroma modes.
Histogram operations accept `L` and `RGB`. Invert, posterize, and solarize
also transform P palettes while preserving indices and palette alpha.
"""

from __future__ import annotations

__lazy_modules__ = {"collections.abc", "struct"}

import struct
from operator import index
from typing import TYPE_CHECKING, Protocol, overload

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Literal

    from ._blanket import _Image

from . import Image
from ._blanket import chops_invert, ops_autocontrast, ops_canvas, ops_colorize_stops, ops_equalize, ops_histogram, ops_lut, ops_mesh, ops_transpose
from ._color import _rgb, color_pixel
from ._exif import transpose_metadata

__all__ = [
    "SupportsGetMesh",
    "autocontrast",
    "colorize",
    "contain",
    "cover",
    "crop",
    "deform",
    "equalize",
    "exif_transpose",
    "expand",
    "fit",
    "flip",
    "grayscale",
    "invert",
    "mirror",
    "pad",
    "posterize",
    "scale",
    "solarize",
]

Color = str | int | tuple[int, ...]
Border = int | tuple[int, ...]


def _result(image: Image.Image, native: _Image) -> Image.Image:
    """Wrap a native result and copy metadata from the source image."""
    result = Image.Image(native)
    result.info.update(image._metadata_info())
    return result


def _copy(image: Image.Image) -> Image.Image:
    """Copy the native raster and source metadata."""
    return image.copy()


def _canvas_color(image: Image.Image, color: Color | None) -> list[int]:
    """Adapt scalar canvas fills without narrowing integer samples."""
    if image.mode == "P":
        if isinstance(color, str):
            color = _rgb(color)
        if isinstance(color, tuple) and len(color) in (3, 4):
            if image.palette is None:
                raise ValueError("image has no palette")
            color = image.palette.getcolor(color, image)
            image._sync_palette()
        return [0 if color is None else index(color)]
    if image.mode in ("I", "I;16", "I;16L", "I;16B", "I;16N"):
        if isinstance(color, tuple):
            if len(color) != 1:
                raise TypeError("color must be an integer or a single-element tuple")
            color = color[0]
        return [0 if color is None else index(color)]
    if image.mode == "F":
        return list(struct.pack("<f", 0.0 if color is None else float(color)))
    return color_pixel(color, image.mode)


def _lut(image: Image.Image, table: list[int], *, palette: bool = False) -> Image.Image:
    """Apply a clamped 8-bit lookup table to an L or RGB image."""
    if image.mode == "P":
        if not palette:
            raise NotImplementedError("mode P support coming soon")
        result = image.copy()
        if result.palette is None:
            raise ValueError("image has no palette")
        mode = result.palette.mode
        entries = result.palette.tobytes()
        colors = Image.frombytes(mode, (len(entries) // len(mode), 1), entries)
        tables = table if mode == "RGB" else table * 3 + list(range(256))
        mapped = Image.Image(ops_lut(colors._native, tables))
        result.putpalette(mapped.tobytes(), mode)
        return result
    if image.mode not in ("L", "RGB"):
        raise OSError(f"not supported for mode {image.mode}")
    return _result(image, ops_lut(image._native, [max(0, min(255, v)) for v in table]))


def _border(border: Border) -> tuple[int, int, int, int]:
    """Expand a border specification to left, top, right, and bottom widths."""
    if isinstance(border, tuple):
        if len(border) == 2:
            return border[0], border[1], border[0], border[1]
        if len(border) == 4:
            return border[0], border[1], border[2], border[3]
        raise ValueError("border must be an integer, or a tuple of two or four elements")
    return border, border, border, border


def _resize(image: Image.Image, size: tuple[int, int], method: int, box: tuple[float, float, float, float] | None = None) -> Image.Image:
    """Resize an image using the requested resampling method and source box."""
    return image.resize(size, method, box)


def autocontrast(image: Image.Image, cutoff: float | tuple[float, float] = 0, ignore: int | Sequence[int] | None = None, mask: Image.Image | None = None, preserve_tone: bool = False) -> Image.Image:
    """Stretch histogram endpoints with optional masks, ignored values and tails.

    ``cutoff`` is a percentage, or separate (low, high) percentages.
    ``preserve_tone`` uses one luminance histogram for all color channels.

    Args:
        image: Input image.
        cutoff: Percentage discarded from histogram tails, or separate `(low, high)` percentages.
        ignore: Pixel value or sequence of values excluded from the histogram.
        mask: Optional L mask; pixels with a zero mask value are excluded.
        preserve_tone: Use a shared luminance histogram to preserve relative channel tone.

    Examples:
        ```python
        from blanket import Image, ImageOps

        image = Image.new("RGB", (8, 8), (40, 100, 180))
        result = ImageOps.autocontrast(image, cutoff=1)
        ```
    """
    if image.mode == "P":
        raise NotImplementedError("mode P support coming soon")
    source = image.convert("L") if preserve_tone else image
    ignored = () if ignore is None else (ignore,) if isinstance(ignore, int) else ignore
    tails = cutoff if isinstance(cutoff, tuple) else (cutoff, cutoff)
    if (
        len(tails) == 2
        and all(isinstance(value, (int, float)) and 0 <= value <= 100 for value in tails)
        and isinstance(ignored, (tuple, list))
        and all(isinstance(value, int) and -256 <= value < 256 for value in ignored)
    ):
        return _result(image, ops_autocontrast(image._native, source._native, None if mask is None else mask._native, tails, ignored))
    # Retain Python's permissive cutoff and index semantics outside the native fast path.
    histogram = ops_histogram(source._native, None if mask is None else mask._native)
    table = []
    for offset in range(0, len(histogram), 256):
        counts = histogram[offset : offset + 256]
        for value in ignored:
            counts[value] = 0
        total = sum(counts)
        if cutoff:
            for percent, indices in zip(tails, (range(256), range(255, -1, -1)), strict=False):
                remaining = int(total * percent // 100)
                for value in indices:
                    removed = min(remaining, counts[value])
                    counts[value] -= removed
                    remaining -= removed
                    if remaining <= 0:
                        break
        occupied = [i for i, n in enumerate(counts) if n]
        if len(occupied) < 2:
            table.extend(range(256))
        else:
            low, high = occupied[0], occupied[-1]
            gain = 255.0 / (high - low)
            offset_value = -low * gain
            table.extend(int(i * gain + offset_value) for i in range(256))
    return _lut(image, table)


def colorize(image: Image.Image, black: Color, white: Color, mid: Color | None = None, blackpoint: int = 0, whitepoint: int = 255, midpoint: int = 127) -> Image.Image:
    """Map an L image to an RGB gradient with two or three color stops.

    Args:
        image: Input image.
        black: Color mapped to the black point.
        white: Color mapped to the white point.
        mid: Optional color at the midpoint for a three-color gradient.
        blackpoint: Input intensity mapped to black, in 0 through 255.
        whitepoint: Input intensity mapped to white, in 0 through 255.
        midpoint: Input intensity mapped to the middle color.

    Examples:
        ```python
        from blanket import Image, ImageOps

        image = Image.new("RGB", (8, 8), (40, 100, 180))
        result = ImageOps.colorize(image.convert("L"), "navy", "white")
        ```
    """
    if image.mode != "L":
        raise ValueError(f"mode must be L, not {image.mode}")
    if not 0 <= blackpoint <= whitepoint <= 255:
        raise ValueError("blackpoint and whitepoint must each be between 0 and 255, with blackpoint <= whitepoint")
    if mid is not None and not blackpoint <= midpoint <= whitepoint:
        raise ValueError("midpoint must be between or equal to blackpoint and whitepoint")
    stops = [(blackpoint, color_pixel(black, "RGB"))]
    if mid is not None:
        stops.append((midpoint, color_pixel(mid, "RGB")))
    stops.append((whitepoint, color_pixel(white, "RGB")))
    return _result(image, ops_colorize_stops(image._native, stops))


def contain(image: Image.Image, size: tuple[int, int], method: int = Image.Resampling.BICUBIC) -> Image.Image:
    """Resize to fit inside size while preserving the aspect ratio.

    Args:
        image: Input image.
        size: Output `(width, height)` in pixels.
        method: Resampling filter from `Image.Resampling`.

    Examples:
        ```python
        from blanket import Image, ImageOps

        image = Image.new("RGB", (8, 8), (40, 100, 180))
        result = ImageOps.contain(image, (4, 6))
        ```
    """
    if image.height == 0:
        return Image.new(image.mode, (size[0], 0))
    if size[1] == 0:
        return Image.new(image.mode, (0, 0))
    ratio = image.width / image.height
    target = size[0] / size[1]
    if ratio > target:
        size = (size[0], round(image.height / image.width * size[0]))
    elif ratio < target:
        size = (round(image.width / image.height * size[1]), size[1])
    if min(size) == 0:
        return Image.new(image.mode, size)
    return _resize(image, size, method)


def cover(image: Image.Image, size: tuple[int, int], method: int = Image.Resampling.BICUBIC) -> Image.Image:
    """Resize to cover size while preserving the aspect ratio.

    Args:
        image: Input image.
        size: Output `(width, height)` in pixels.
        method: Resampling filter from `Image.Resampling`.

    Examples:
        ```python
        from blanket import Image, ImageOps

        image = Image.new("RGB", (8, 8), (40, 100, 180))
        result = ImageOps.cover(image, (4, 6))
        ```
    """
    if image.height == 0:
        return Image.new(image.mode, (size[0], 0))
    if size[1] == 0:
        return Image.new(image.mode, (0, 0))
    ratio = image.width / image.height
    target = size[0] / size[1]
    if ratio < target:
        size = (size[0], round(image.height / image.width * size[0]))
    elif ratio > target:
        size = (round(image.width / image.height * size[1]), size[1])
    return _resize(image, size, method)


def pad(image: Image.Image, size: tuple[int, int], method: int = Image.Resampling.BICUBIC, color: Color | None = None, centering: tuple[float, float] = (0.5, 0.5)) -> Image.Image:
    """Contain the image and pad it to size with the selected background.

    Args:
        image: Input image.
        size: Output `(width, height)` in pixels.
        method: Resampling filter from `Image.Resampling`.
        color: Fill value, channel tuple, or CSS color string.
        centering: Horizontal and vertical alignment in 0 through 1; 0.5 centers each axis.

    Examples:
        ```python
        from blanket import Image, ImageOps

        image = Image.new("RGB", (8, 8), (40, 100, 180))
        result = ImageOps.pad(image, (12, 8), color="white")
        ```
    """
    resized = contain(image, size, method)
    if resized.size == tuple(size):
        return resized
    offset = tuple(round((target - actual) * max(0, min(position, 1))) for target, actual, position in zip(size, resized.size, centering, strict=False))
    if resized.mode == "P":
        resized = resized.copy()
    fill = _canvas_color(resized, color)
    return _result(resized, ops_canvas(resized._native, size, offset, fill))


def crop(image: Image.Image, border: int = 0) -> Image.Image:
    """Remove ``border`` pixels from each side of the image.

    Like Pillow, also accepts (horizontal, vertical) or (left, top, right,
    bottom) border tuples. Use image.crop(box) to select a rectangle instead.

    Args:
        image: Input image.
        border: Border width in pixels.

    Examples:
        ```python
        from blanket import Image, ImageOps

        image = Image.new("RGB", (8, 8), (40, 100, 180))
        result = ImageOps.crop(image, border=1)
        ```
    """
    left, top, right, bottom = _border(border)
    return image.crop((left, top, image.width - right, image.height - bottom))


def scale(image: Image.Image, factor: float, resample: int = Image.Resampling.BICUBIC) -> Image.Image:
    """Resize by a positive scale factor, rounding dimensions to pixels.

    Args:
        image: Input image.
        factor: Positive multiplier for both image dimensions.
        resample: Resampling filter from `Image.Resampling`; supported filters depend on the operation.

    Examples:
        ```python
        from blanket import Image, ImageOps

        image = Image.new("RGB", (8, 8), (40, 100, 180))
        result = ImageOps.scale(image, 0.5)
        ```
    """
    if factor == 1:
        return _copy(image)
    if factor <= 0:
        raise ValueError("the factor must be greater than 0")
    return _resize(image, (round(image.width * factor), round(image.height * factor)), resample)


class SupportsGetMesh(Protocol):
    """A deformer returning destination boxes and source TL, BL, BR, TR quads."""

    def getmesh(self, image: Image.Image) -> list[tuple[tuple[int, int, int, int], tuple[float, float, float, float, float, float, float, float]]]:
        """Describe destination rectangles and their source quadrilaterals.

        Args:
            image: Image being deformed, used to determine source dimensions.

        Returns:
            Pairs of destination (left, top, right, bottom) rectangles and source
            coordinates in top-left, bottom-left, bottom-right, top-right order.

        Examples:
            ```python
            from blanket import Image, ImageOps


            class IdentityMesh:
                def getmesh(self, image):
                    width, height = image.size
                    return [((0, 0, width, height), (0, 0, 0, height, width, height, width, 0))]


            image = Image.new("RGB", (8, 8), "navy")
            result = ImageOps.deform(image, IdentityMesh())
            ```
        """
        ...


def deform(image: Image.Image, deformer: SupportsGetMesh, resample: int = Image.Resampling.BILINEAR) -> Image.Image:
    """Warp using a deformer mesh with nearest, bilinear or bicubic sampling.

    Args:
        image: Input image.
        deformer: Object implementing `getmesh(image)` and returning destination boxes with source quadrilaterals.
        resample: Resampling filter from `Image.Resampling`; supported filters depend on the operation.

    Examples:
        ```python
        from blanket import Image, ImageOps

        image = Image.new("RGB", (8, 8), (40, 100, 180))


        class IdentityMesh:
            def getmesh(self, image):
                width, height = image.size
                return [((0, 0, width, height), (0, 0, 0, height, width, height, width, 0))]


        result = ImageOps.deform(image, IdentityMesh())
        ```
    """
    mesh = deformer.getmesh(image)
    if any(box[0] == box[2] or box[1] == box[3] for box, quad in mesh):
        raise ZeroDivisionError("division by zero")
    if resample not in (0, 2, 3):
        raise ValueError("mesh transforms support NEAREST, BILINEAR and BICUBIC")
    image.load()
    return _result(image, ops_mesh(image._native, mesh, resample))


def equalize(image: Image.Image, mask: Image.Image | None = None) -> Image.Image:
    """Equalize each channel using the histogram selected by an optional mask.

    Args:
        image: Input image.
        mask: Optional L mask; pixels with a zero mask value are excluded.

    Examples:
        ```python
        from blanket import Image, ImageOps

        image = Image.new("RGB", (8, 8), (40, 100, 180))
        result = ImageOps.equalize(image)
        ```
    """
    if image.mode == "P":
        image = image.convert("RGB")
    return _result(image, ops_equalize(image._native, None if mask is None else mask._native))


def expand(image: Image.Image, border: Border = 0, fill: Color = 0) -> Image.Image:
    """Add a border filled with a pixel value or CSS color.

    Args:
        image: Input image.
        border: Integer, `(horizontal, vertical)`, or `(left, top, right, bottom)` widths.
        fill: Border value, channel tuple, or CSS color.

    Examples:
        ```python
        from blanket import Image, ImageOps

        image = Image.new("RGB", (8, 8), (40, 100, 180))
        result = ImageOps.expand(image, border=2, fill="white")
        ```
    """
    left, top, right, bottom = _border(border)
    size = (image.width + left + right, image.height + top + bottom)
    if min(size) < 0:
        raise ValueError("Width and height must be >= 0")
    image.load()
    if image.mode == "P":
        image = image.copy()
    color = _canvas_color(image, fill)
    return _result(image, ops_canvas(image._native, size, (left, top), color))


def fit(image: Image.Image, size: tuple[int, int], method: int = Image.Resampling.BICUBIC, bleed: float = 0.0, centering: tuple[float, float] = (0.5, 0.5)) -> Image.Image:
    """Crop to the requested aspect ratio and resize, with optional edge bleed.

    Args:
        image: Input image.
        size: Output `(width, height)` in pixels.
        method: Resampling filter from `Image.Resampling`.
        bleed: Fraction to remove from each edge before fitting; must be less than 0.5.
        centering: Horizontal and vertical alignment in 0 through 1; 0.5 centers each axis.

    Examples:
        ```python
        from blanket import Image, ImageOps

        image = Image.new("RGB", (8, 8), (40, 100, 180))
        result = ImageOps.fit(image, (4, 6))
        ```
    """
    if min(size) == 0:
        return Image.new(image.mode, size)
    cx, cy = (v if 0 <= v <= 1 else 0.5 for v in centering)
    if not 0 <= bleed < 0.5:
        bleed = 0.0
    bx, by = image.width * bleed, image.height * bleed
    width, height = image.width - 2 * bx, image.height - 2 * by
    if height == 0:
        return Image.new(image.mode, size)
    ratio = size[0] / size[1]
    if width / height == ratio:
        cw, ch = width, height
    elif width / height > ratio:
        cw, ch = ratio * height, height
    else:
        cw, ch = width, width / ratio
    left, top = bx + (width - cw) * cx, by + (height - ch) * cy
    return _resize(image, size, method, (left, top, left + cw, top + ch))


def flip(image: Image.Image) -> Image.Image:
    """Flip vertically.

    Args:
        image: Input image.

    Examples:
        ```python
        from blanket import Image, ImageOps

        image = Image.new("RGB", (8, 8), (40, 100, 180))
        result = ImageOps.flip(image)
        ```
    """
    return image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)


def grayscale(image: Image.Image) -> Image.Image:
    """Convert to luminance mode `L`, retaining the input sample depth.

    Args:
        image: Input image.

    Examples:
        ```python
        from blanket import Image, ImageOps

        image = Image.new("RGB", (8, 8), (40, 100, 180))
        result = ImageOps.grayscale(image)
        ```
    """
    return _result(image, image.convert("L")._native)


def invert(image: Image.Image) -> Image.Image:
    """Subtract each channel value from the sample depth's maximum.

    Args:
        image: Input image.

    Examples:
        ```python
        from blanket import Image, ImageOps

        image = Image.new("RGB", (8, 8), (40, 100, 180))
        result = ImageOps.invert(image)
        ```
    """
    if image.bit_depth in (10, 12, 16) and image.mode in ("L", "RGB"):
        image.load()
        return _result(image, chops_invert(image._native))
    if image.mode == "1":
        return image.point(list(range(255, -1, -1)))
    return _lut(image, list(range(255, -1, -1)), palette=True)


def mirror(image: Image.Image) -> Image.Image:
    """Flip horizontally.

    Args:
        image: Input image.

    Examples:
        ```python
        from blanket import Image, ImageOps

        image = Image.new("RGB", (8, 8), (40, 100, 180))
        result = ImageOps.mirror(image)
        ```
    """
    return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)


def posterize(image: Image.Image, bits: int) -> Image.Image:
    """Keep the highest bits of each channel (1 through 8).

    Args:
        image: Input image.
        bits: Number of high bits to retain per channel, from 1 through 8.

    Examples:
        ```python
        from blanket import Image, ImageOps

        image = Image.new("RGB", (8, 8), (40, 100, 180))
        result = ImageOps.posterize(image, bits=4)
        ```
    """
    if not 1 <= bits <= 8:
        raise ValueError("bits must be between or equal to 1 and 8")
    mask = ~(2 ** (8 - bits) - 1)
    return _lut(image, [v & mask for v in range(256)], palette=True)


def solarize(image: Image.Image, threshold: int = 128) -> Image.Image:
    """Invert channel values greater than or equal to threshold.

    Args:
        image: Input image.
        threshold: Intensity threshold, on the 0 through 255 scale.

    Examples:
        ```python
        from blanket import Image, ImageOps

        image = Image.new("RGB", (8, 8), (40, 100, 180))
        result = ImageOps.solarize(image, threshold=128)
        ```
    """
    return _lut(image, [v if v < threshold else 255 - v for v in range(256)], palette=True)


@overload
def exif_transpose(image: Image.Image, *, in_place: Literal[True]) -> None: ...


@overload
def exif_transpose(image: Image.Image, *, in_place: Literal[False] = False) -> Image.Image: ...


def exif_transpose(image: Image.Image, *, in_place: bool = False) -> Image.Image | None:
    """Apply EXIF orientation and remove it from EXIF and XMP metadata.

    Return a copy by default, or update the original and return None when
    ``in_place=True``. Other EXIF entries and their offsets are preserved.

    Args:
        image: Input image.
        in_place: Modify the input and return None when true; otherwise return a transformed copy.

    Examples:
        ```python
        from blanket import Image, ImageOps

        image = Image.new("RGB", (8, 8), (40, 100, 180))
        result = ImageOps.exif_transpose(image)
        ```
    """
    image.load()
    orientation, info = transpose_metadata(image._metadata_info())
    if orientation in range(2, 9):
        native = ops_transpose(image._native, orientation)
        result = image if in_place else Image.Image(native)
        result._native = native
        result._info = info
        return None if in_place else result
    return None if in_place else _copy(image)
