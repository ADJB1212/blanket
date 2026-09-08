"""Pillow-shaped image operations, backed by native Blanket pixel kernels.

Geometry works on L, RGB and RGBA. Histogram and lookup-table operations
accept L and RGB, matching Pillow's restrictions for these modes.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise
from typing import Literal, Protocol, overload

from . import Image
from ._blanket import _Image, ops_canvas, ops_colorize, ops_histogram, ops_lut, ops_mesh, ops_transpose
from ._color import color_pixel
from ._exif import transpose_metadata

__all__ = ["SupportsGetMesh", "autocontrast", "colorize", "contain", "cover", "crop", "deform", "equalize", "exif_transpose", "expand", "fit", "flip", "grayscale", "invert", "mirror", "pad", "posterize", "scale", "solarize"]

Color = str | int | tuple[int, ...]
Border = int | tuple[int, ...]


def _result(image: Image.Image, native: _Image) -> Image.Image:
    result = Image.Image(native)
    result.info.update(image.info)
    return result


def _copy(image: Image.Image) -> Image.Image:
    return _result(image, ops_transpose(image._native, 1))


def _lut(image: Image.Image, table: list[int]) -> Image.Image:
    if image.mode not in ("L", "RGB"):
        raise OSError(f"not supported for mode {image.mode}")
    return _result(image, ops_lut(image._native, [max(0, min(255, v)) for v in table]))


def _border(border: Border) -> tuple[int, int, int, int]:
    if isinstance(border, tuple):
        if len(border) == 2:
            return border * 2
        if len(border) == 4:
            return border
        raise ValueError("border must be an integer, or a tuple of two or four elements")
    return (border,) * 4


def _resize(image: Image.Image, size: tuple[int, int], method: int, box: tuple[float, float, float, float] | None = None) -> Image.Image:
    return image.resize(size, method, box)


def autocontrast(image: Image.Image, cutoff: float | tuple[float, float] = 0, ignore: int | Sequence[int] | None = None, mask: Image.Image | None = None, preserve_tone: bool = False) -> Image.Image:
    """Stretch histogram endpoints with optional masks, ignored values and tails.

    ``cutoff`` is a percentage, or separate (low, high) percentages.
    ``preserve_tone`` uses one luminance histogram for all color channels.
    """
    source = image.convert("L") if preserve_tone else image
    histogram = ops_histogram(source._native, None if mask is None else mask._native)
    ignored = () if ignore is None else (ignore,) if isinstance(ignore, int) else ignore
    tails = cutoff if isinstance(cutoff, tuple) else (cutoff, cutoff)
    table = []
    for offset in range(0, len(histogram), 256):
        counts = histogram[offset : offset + 256]
        for value in ignored:
            counts[value] = 0
        total = sum(counts)
        if cutoff:
            for percent, indices in zip(tails, (range(256), range(255, -1, -1))):
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
    """Map an L image to an RGB gradient with two or three color stops."""
    assert image.mode == "L"
    assert 0 <= blackpoint <= whitepoint <= 255
    if mid is not None:
        assert blackpoint <= midpoint <= whitepoint
    stops = [(blackpoint, color_pixel(black, "RGB"))]
    if mid is not None:
        stops.append((midpoint, color_pixel(mid, "RGB")))
    stops.append((whitepoint, color_pixel(white, "RGB")))
    table = []
    for channel in range(3):
        ramp = [stops[0][1][channel]] * blackpoint
        for (start, first), (end, last) in pairwise(stops):
            ramp.extend(first[channel] + (i - start) * (last[channel] - first[channel]) // (end - start) for i in range(start, end))
        ramp.extend([stops[-1][1][channel]] * (256 - whitepoint))
        table.extend(ramp)
    return _result(image, ops_colorize(image._native, table))


def contain(image: Image.Image, size: tuple[int, int], method: int = Image.Resampling.BICUBIC) -> Image.Image:
    """Resize to fit inside size while preserving the aspect ratio."""
    ratio = image.width / image.height
    target = size[0] / size[1]
    if ratio > target:
        size = (size[0], round(image.height / image.width * size[0]))
    elif ratio < target:
        size = (round(image.width / image.height * size[1]), size[1])
    return _resize(image, size, method)


def cover(image: Image.Image, size: tuple[int, int], method: int = Image.Resampling.BICUBIC) -> Image.Image:
    """Resize to cover size while preserving the aspect ratio."""
    ratio = image.width / image.height
    target = size[0] / size[1]
    if ratio < target:
        size = (size[0], round(image.height / image.width * size[0]))
    elif ratio > target:
        size = (round(image.width / image.height * size[1]), size[1])
    return _resize(image, size, method)


def pad(image: Image.Image, size: tuple[int, int], method: int = Image.Resampling.BICUBIC, color: Color | None = None, centering: tuple[float, float] = (0.5, 0.5)) -> Image.Image:
    """Contain the image and pad it to size with the selected background."""
    resized = contain(image, size, method)
    if resized.size == tuple(size):
        return resized
    offset = tuple(round((target - actual) * max(0, min(position, 1))) for target, actual, position in zip(size, resized.size, centering))
    return Image.Image(ops_canvas(resized._native, size, offset, color_pixel(color, image.mode)))


def crop(image: Image.Image, border: int = 0) -> Image.Image:
    """Remove ``border`` pixels from each side of the image.

    Like Pillow, also accepts (horizontal, vertical) or (left, top, right,
    bottom) border tuples. Use image.crop(box) to select a rectangle instead.
    """
    left, top, right, bottom = _border(border)
    return image.crop((left, top, image.width - right, image.height - bottom))


def scale(image: Image.Image, factor: float, resample: int = Image.Resampling.BICUBIC) -> Image.Image:
    """Resize by a positive scale factor, rounding dimensions to pixels."""
    if factor == 1:
        return _copy(image)
    if factor <= 0:
        raise ValueError("the factor must be greater than 0")
    return _resize(image, (round(image.width * factor), round(image.height * factor)), resample)


class SupportsGetMesh(Protocol):
    """A deformer returning destination boxes and source TL, BL, BR, TR quads."""

    def getmesh(self, image: Image.Image) -> list[tuple[tuple[int, int, int, int], tuple[float, float, float, float, float, float, float, float]]]: ...


def deform(image: Image.Image, deformer: SupportsGetMesh, resample: int = Image.Resampling.BILINEAR) -> Image.Image:
    """Warp using a deformer mesh with nearest, bilinear or bicubic sampling."""
    mesh = deformer.getmesh(image)
    if any(box[0] == box[2] or box[1] == box[3] for box, quad in mesh):
        raise ZeroDivisionError("division by zero")
    if resample not in (0, 2, 3):
        raise ValueError("mesh transforms support NEAREST, BILINEAR and BICUBIC")
    return _result(image, ops_mesh(image._native, mesh, resample))


def equalize(image: Image.Image, mask: Image.Image | None = None) -> Image.Image:
    """Equalize each channel using the histogram selected by an optional mask."""
    histogram = ops_histogram(image._native, None if mask is None else mask._native)
    table = []
    for offset in range(0, len(histogram), 256):
        counts = histogram[offset : offset + 256]
        occupied = [n for n in counts if n]
        step = (sum(occupied) - occupied[-1]) // 255 if occupied else 0
        if not step:
            table.extend(range(256))
            continue
        cumulative = step // 2
        for count in counts:
            table.append(cumulative // step)
            cumulative += count
    return _lut(image, table)


def expand(image: Image.Image, border: Border = 0, fill: Color = 0) -> Image.Image:
    """Add a border filled with a pixel value or CSS color."""
    left, top, right, bottom = _border(border)
    size = (image.width + left + right, image.height + top + bottom)
    if min(size) < 0:
        raise ValueError("Width and height must be >= 0")
    return Image.Image(ops_canvas(image._native, size, (left, top), color_pixel(fill, image.mode)))


def fit(image: Image.Image, size: tuple[int, int], method: int = Image.Resampling.BICUBIC, bleed: float = 0.0, centering: tuple[float, float] = (0.5, 0.5)) -> Image.Image:
    """Crop to the requested aspect ratio and resize, with optional edge bleed."""
    cx, cy = (v if 0 <= v <= 1 else 0.5 for v in centering)
    if not 0 <= bleed < 0.5:
        bleed = 0.0
    bx, by = image.width * bleed, image.height * bleed
    width, height = image.width - 2 * bx, image.height - 2 * by
    ratio = size[0] / size[1]
    if width / height >= ratio:
        cw, ch = ratio * height, height
    else:
        cw, ch = width, width / ratio
    left, top = bx + (width - cw) * cx, by + (height - ch) * cy
    return _resize(image, size, method, (left, top, left + cw, top + ch))


def flip(image: Image.Image) -> Image.Image:
    """Flip vertically."""
    return _result(image, ops_transpose(image._native, 4))


def grayscale(image: Image.Image) -> Image.Image:
    """Convert to 8-bit luminance."""
    return _result(image, image.convert("L")._native)


def invert(image: Image.Image) -> Image.Image:
    """Replace each channel value with 255 minus that value."""
    return _lut(image, list(range(255, -1, -1)))


def mirror(image: Image.Image) -> Image.Image:
    """Flip horizontally."""
    return _result(image, ops_transpose(image._native, 2))


def posterize(image: Image.Image, bits: int) -> Image.Image:
    """Keep the highest bits of each channel (0 through 8)."""
    mask = ~(2 ** (8 - bits) - 1)
    return _lut(image, [v & mask for v in range(256)])


def solarize(image: Image.Image, threshold: int = 128) -> Image.Image:
    """Invert channel values greater than or equal to threshold."""
    return _lut(image, [v if v < threshold else 255 - v for v in range(256)])


@overload
def exif_transpose(image: Image.Image, *, in_place: Literal[True]) -> None: ...


@overload
def exif_transpose(image: Image.Image, *, in_place: Literal[False] = False) -> Image.Image: ...


def exif_transpose(image: Image.Image, *, in_place: bool = False) -> Image.Image | None:
    """Apply EXIF orientation and remove it from EXIF and XMP metadata.

    Return a copy by default, or update the original and return None when
    ``in_place=True``. Other EXIF entries and their offsets are preserved.
    """
    image.load()
    orientation, info = transpose_metadata(image.info)
    if orientation in range(2, 9):
        native = ops_transpose(image._native, orientation)
        result = image if in_place else Image.Image(native)
        result._native = native
        result._info = info
        return None if in_place else result
    return None if in_place else _copy(image)
