"""Pillow-shaped image operations, backed by native Blanket pixel kernels.

Geometry, canvas, and resampling work on the current Image modes, including
`1`, `LA`, `PA`, integer, float, and print/luma/chroma modes.
Equalize and autocontrast accept `L` and `RGB`; reference transfers also
accept unsigned grayscale/RGB with alpha and high sample depths.
Invert, posterize, and solarize also transform P palettes while preserving
indices and palette alpha.
"""

from __future__ import annotations

__lazy_modules__ = {"collections.abc", "operator"}

from dataclasses import dataclass
from operator import index
from typing import TYPE_CHECKING, Protocol, cast, overload

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Literal, SupportsIndex

    from ._blanket import _Image

from . import Image, ImageChops
from ._blanket import (
    ops_autocontrast,
    ops_autocontrast_lut,
    ops_canvas,
    ops_colorize_stops,
    ops_contain_size,
    ops_cover_size,
    ops_equalize,
    ops_fit_box,
    ops_histogram,
    ops_lut,
    ops_mesh,
    ops_pad_offset,
    ops_phase_correlate,
    ops_reference_transfer,
    ops_scale_size,
    ops_smart_crop_box,
    ops_smart_crop_size,
    ops_tone_lut,
    ops_transpose,
    palette_lut,
)
from ._color import _rgb, color_pixel
from ._exif import transpose_metadata

__all__ = [
    "Alignment",
    "SupportsGetMesh",
    "autocontrast",
    "color_transfer",
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
    "match_histogram",
    "mirror",
    "pad",
    "phase_correlate",
    "posterize",
    "scale",
    "smart_crop",
    "solarize",
]

Color = str | int | tuple[int, ...]
Border = int | tuple[int, ...]


@dataclass(frozen=True, slots=True)
class Alignment:
    """Correction to align the moving image to its reference.

    Rotate counterclockwise by ``rotation`` degrees about the image center,
    then translate by ``dx`` pixels right and ``dy`` pixels down.
    ``confidence`` is the normalized phase-correlation peak in [0, 1],
    not a probability. Flat inputs return zero offsets and confidence.
    """

    dx: float
    dy: float
    rotation: float
    confidence: float


def phase_correlate(reference: Image.Image, image: Image.Image, *, max_rotation: float = 0.0, rotation_step: float = 1.0, window: Literal["hann"] | None = "hann") -> Alignment:
    """Estimate translation and optionally a small rotation to a reference.

    Require equal dimensions, each at least 3, and at most 4,194,304 pixels.
    Accept L, LA, RGB, RGBA, I;16 variants and F/RGB;F/RGBA;F, at their
    supported depths. Color samples must be finite; alpha is ignored.
    Analyze encoded Rec.601 luma with mean removal and optional Hann window.
    No ICC or EXIF orientation is applied. Current frames, metadata and
    navigation are preserved. No runtime Python dependencies are required.

    Return the correction to apply to image: rotate about its center, then
    translate. Translation is cyclic in [-width/2, width/2) and
    [-height/2, height/2), with approximate subpixel parabolic refinement.
    A half-size displacement is reported negative. Large shifts can alias;
    repeated patterns, cropping, occlusion or unrelated images can be ambiguous.
    Confidence is the inverse cross-power peak divided by pixel count in
    [0, 1], with weak frequencies suppressed. Higher is better, not a
    calibrated probability. Inspect confidence before applying a correction.

    Args:
        reference: Fixed reference image.
        image: Moving image to align, without editing it.
        max_rotation: Search both directions up to 15 degrees; zero disables.
        rotation_step: Positive search interval in degrees, including endpoints
            and zero; at most 60 intervals per direction. Rotation uses bilinear
            sampling with zero fill after mean removal. Choose a smaller step
            for finer rotation estimates at greater computational cost.
        window: Use "hann" to reduce border effects, or None for cyclic data.
            Axes with only three samples keep uniform weights.

    Returns:
        The translation and optional rotation that aligns ``image`` to ``reference``.

    Raises:
        TypeError: An argument has the wrong type.
        ValueError: The dimensions, mode, or rotation settings are unsupported.

    Examples:
        ```python
        from blanket import Image, ImageOps

        reference = Image.frombytes("L", (32, 32), bytes((i * 37 + i // 32) % 256 for i in range(1024)))
        alignment = ImageOps.phase_correlate(reference, reference.copy())
        print(alignment.dx, alignment.dy, alignment.confidence)
        ```
    """
    if not isinstance(reference, Image.Image) or not isinstance(image, Image.Image):
        raise TypeError("reference and image must be Blanket Images")
    return Alignment(*ops_phase_correlate(reference._native, image._native, max_rotation, rotation_step, window))


def match_histogram(image: Image.Image, reference: Image.Image, *, mode: Literal["channels", "luminance"] = "channels") -> Image.Image:
    """Match the source's cumulative histogram to a reference image.

    Accept unsigned L, LA, RGB, RGBA and I;16 variants, at supported depths
    through 16 bits. Reference size and depth may differ. Channels mode
    requires matching color channel counts; luminance mode also permits a
    grayscale reference for RGB. Use the smallest reference sample whose CDF
    reaches the source CDF, scaling to the source depth and rounding half up.
    A deterministic lookup table cannot split equal source samples.

    Luminance is rounded encoded Rec.601 luma (0.299 R + 0.587 G + 0.114 B).
    Apply its change equally to RGB channels, clipping to the source range.
    Alpha is preserved; zero-alpha pixels are excluded and left unchanged,
    and all other pixels have equal weight. Color-key transparency is expanded
    to an alpha channel in L and RGB images. Wide grayscale uses RGBA so it
    retains its sample depth. Empty or fully transparent inputs
    raise ValueError. Geometry, depth and remaining source metadata are preserved.
    ICC profiles and EXIF orientation are not applied.

    Args:
        image: Source image, processed without editing it.
        reference: Image supplying the target distribution.
        mode: Match individual channels or luminance.

    Returns:
        A new image with the source geometry and matched sample distribution.

    Raises:
        ValueError: The modes are incompatible or an input has no opaque samples.

    Examples:
        ```python
        from blanket import Image, ImageOps

        image = Image.new("RGB", (8, 8), (40, 80, 120))
        reference = Image.new("L", (4, 4), 100)
        result = ImageOps.match_histogram(image, reference, mode="luminance")
        ```
    """
    return _reference_transfer(image, reference, mode, "histogram")


def color_transfer(image: Image.Image, reference: Image.Image, *, mode: Literal["channels", "luminance"] = "channels") -> Image.Image:
    """Transfer a reference image's mean and population standard deviation.

    Use the same modes, depth scaling, alpha policy and luminance adjustment
    as match_histogram. Transform encoded samples by
    ``(sample - source_mean) * reference_std / source_std + reference_mean``,
    then round half up and clip. A constant source maps to the reference mean;
    a constant reference removes source variation. Clipping and quantization
    can prevent the output from attaining the reference moments exactly.
    Inputs are not edited and geometry and source metadata are preserved.

    Args:
        image: Source image, processed without editing it.
        reference: Image supplying the target moments.
        mode: Transfer individual channel or luminance moments.

    Returns:
        A new image with the source geometry and transferred color moments.

    Raises:
        ValueError: The modes are incompatible or an input has no opaque samples.

    Examples:
        ```python
        from blanket import Image, ImageOps

        image = Image.new("RGB", (8, 8), (40, 80, 120))
        reference = Image.new("RGB", (4, 4), (60, 100, 140))
        result = ImageOps.color_transfer(image, reference)
        ```
    """
    return _reference_transfer(image, reference, mode, "moments")


def _reference_transfer(image: Image.Image, reference: Image.Image, mode: str, method: str) -> Image.Image:
    """Expand transparency before computing statistics or changing color samples."""
    if not isinstance(image, Image.Image) or not isinstance(reference, Image.Image):
        raise TypeError("image and reference must be Blanket Images")
    source_key = image.mode in ("L", "RGB") and image.info.get("transparency") is not None
    reference_key = reference.mode in ("L", "RGB") and reference.info.get("transparency") is not None
    source_wide_gray = source_key and image.mode == "L" and image.bit_depth > 8
    reference_wide_gray = reference_key and reference.mode == "L" and reference.bit_depth > 8
    if mode == "channels" and (source_wide_gray or reference_wide_gray):
        if image.mode in ("RGB", "RGBA") or reference.mode in ("RGB", "RGBA"):
            raise ValueError("channel matching requires the same number of color channels")
        mode = "luminance"
    if source_key:
        image = image.convert("LA" if image.mode == "L" and not source_wide_gray else "RGBA")
    if reference_key:
        reference = reference.convert("LA" if reference.mode == "L" and not reference_wide_gray else "RGBA")
    return _result(image, ops_reference_transfer(image._native, reference._native, mode, method))


def _result(image: Image.Image, native: _Image) -> Image.Image:
    """Wrap a native result and copy metadata from the source image."""
    result = Image.Image(native)
    result.info.update(image._metadata_info())
    return result


def _copy(image: Image.Image) -> Image.Image:
    """Copy the native raster and source metadata."""
    return image.copy()


def _canvas_color(image: Image.Image, color: str | float | tuple[float, ...] | None) -> list[int]:
    """Adapt scalar canvas fills without narrowing integer samples."""
    if image.mode == "P":
        if isinstance(color, str):
            color = _rgb(color)
        if isinstance(color, tuple) and len(color) in (3, 4):
            if image.palette is None:
                raise ValueError("image has no palette")
            color = image.palette.getcolor(cast("tuple[int, ...]", color), image)
            image._sync_palette()
        return [0 if color is None else index(cast("SupportsIndex", color))]
    if image.mode in ("I", "I;16", "I;16L", "I;16B", "I;16N"):
        if isinstance(color, tuple):
            if len(color) != 1:
                raise TypeError("color must be an integer or a single-element tuple")
            color = color[0]
        return [0 if color is None else index(cast("SupportsIndex", color))]
    if image.mode in ("F", "RGB;F", "RGBA;F"):
        return list(Image.new(image.mode, (1, 1), color).tobytes())
    return color_pixel(cast("Color | None", color), image.mode, image.bit_depth)


def _lut(image: Image.Image, table: bytes, *, palette: bool = False) -> Image.Image:
    """Apply a clamped 8-bit lookup table to an L or RGB image."""
    if image.mode == "P":
        if not palette:
            raise NotImplementedError("mode P support coming soon")
        result = image.copy()
        if result.palette is None:
            raise ValueError("image has no palette")
        mode = result.palette.mode
        entries = result.palette.tobytes()
        result.putpalette(palette_lut(entries, len(mode), table), mode)
        return result
    if image.mode not in ("L", "RGB"):
        raise OSError(f"not supported for mode {image.mode}")
    return _result(image, ops_lut(image._native, table))


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

    Returns:
        A new image with stretched histogram endpoints.

    Raises:
        NotImplementedError: The image mode is unsupported.
        ValueError: A cutoff or mask is invalid.

    Examples:
        ```python
        from blanket import Image, ImageOps

        image = Image.new("RGB", (8, 8), (40, 100, 180))
        result = ImageOps.autocontrast(image, cutoff=1)
        ```
    """
    if image.mode == "P":
        raise NotImplementedError("mode P support coming soon")
    ignored = () if ignore is None else (ignore,) if isinstance(ignore, int) else ignore
    tails = cutoff if isinstance(cutoff, tuple) else (cutoff, cutoff)
    if len(tails) == 2 and all(isinstance(value, (int, float)) and 0 <= value <= 100 for value in tails) and isinstance(ignored, (tuple, list)) and all(isinstance(value, int) and -256 <= value < 256 for value in ignored):
        native_tone = preserve_tone and image.mode == "RGB" and image.bit_depth == 8
        source = image.convert("L") if preserve_tone and image.mode != "L" and not native_tone else image
        return _result(image, ops_autocontrast(image._native, source._native, None if mask is None else mask._native, tails, ignored, native_tone))
    # Retain Python's permissive cutoff and index semantics outside the native fast path.
    source = image.convert("L") if preserve_tone and image.mode != "L" else image
    histogram = ops_histogram(source._native, None if mask is None else mask._native)
    table = b"".join(ops_autocontrast_lut(histogram[offset : offset + 256], tails[:2] if cutoff else (), [index(value) for value in ignored]) for offset in range(0, len(histogram), 256))
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

    Returns:
        A new color image mapped from the grayscale input.

    Raises:
        ValueError: The image is not grayscale or a color point is invalid.

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

    Returns:
        A new image that fits inside ``size``.

    Examples:
        ```python
        from blanket import Image, ImageOps

        image = Image.new("RGB", (8, 8), (40, 100, 180))
        result = ImageOps.contain(image, (4, 6))
        ```
    """
    size = ops_contain_size(image.size, size)
    if min(size) == 0:
        return image.crop((0, 0, *size))
    return _resize(image, size, method)


def cover(image: Image.Image, size: tuple[int, int], method: int = Image.Resampling.BICUBIC) -> Image.Image:
    """Resize to cover size while preserving the aspect ratio.

    Args:
        image: Input image.
        size: Output `(width, height)` in pixels.
        method: Resampling filter from `Image.Resampling`.

    Returns:
        A new image that covers ``size``. The aspect ratio is preserved.

    Examples:
        ```python
        from blanket import Image, ImageOps

        image = Image.new("RGB", (8, 8), (40, 100, 180))
        result = ImageOps.cover(image, (4, 6))
        ```
    """
    empty = image.height == 0 or size[1] == 0
    size = ops_cover_size(image.size, size)
    if empty:
        return image.crop((0, 0, *size))
    return _resize(image, size, method)


def pad(image: Image.Image, size: tuple[int, int], method: int = Image.Resampling.BICUBIC, color: str | float | tuple[float, ...] | None = None, centering: tuple[float, float] = (0.5, 0.5)) -> Image.Image:
    """Contain the image and pad it to size with the selected background.

    High-depth numeric colors use native sample units.

    Args:
        image: Input image.
        size: Output `(width, height)` in pixels.
        method: Resampling filter from `Image.Resampling`.
        color: Fill value, channel tuple, or CSS color string.
        centering: Horizontal and vertical alignment in 0 through 1; 0.5 centers each axis.

    Returns:
        A new image of exactly ``size``.

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
    offset = ops_pad_offset(resized.size, size, centering)
    fill = _canvas_color(resized, color)
    return _result(resized, ops_canvas(resized._native, size, offset, fill))


def crop(image: Image.Image, border: int = 0) -> Image.Image:
    """Remove ``border`` pixels from each side of the image.

    Like Pillow, also accepts (horizontal, vertical) or (left, top, right,
    bottom) border tuples. Use image.crop(box) to select a rectangle instead.

    Args:
        image: Input image.
        border: Border width in pixels.

    Returns:
        A new image with the border removed.

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

    Returns:
        A new image scaled by ``factor``.

    Raises:
        ValueError: ``factor`` is not positive.

    Examples:
        ```python
        from blanket import Image, ImageOps

        image = Image.new("RGB", (8, 8), (40, 100, 180))
        result = ImageOps.scale(image, 0.5)
        ```
    """
    if factor == 1:
        return _copy(image)
    return _resize(image, ops_scale_size(image.size, factor), resample)


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

    Returns:
        A new image warped by the mesh.

    Raises:
        ValueError: The resampling filter is unsupported.
        ZeroDivisionError: A destination box has zero width or height.

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

    Returns:
        A new image with equalized channels.

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


def expand(image: Image.Image, border: Border = 0, fill: str | float | tuple[float, ...] | None = 0) -> Image.Image:
    """Add a border filled with a pixel value or CSS color.

    High-depth numeric colors use native sample units.

    Args:
        image: Input image.
        border: Integer, `(horizontal, vertical)`, or `(left, top, right, bottom)` widths.
        fill: Border value, channel tuple, or CSS color.

    Returns:
        A new image enlarged by ``border``.

    Raises:
        ValueError: The resulting width or height is negative.

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

    Returns:
        A new image of exactly ``size``.

    Raises:
        ValueError: ``bleed`` is 0.5 or greater.

    Examples:
        ```python
        from blanket import Image, ImageOps

        image = Image.new("RGB", (8, 8), (40, 100, 180))
        result = ImageOps.fit(image, (4, 6))
        ```
    """
    box = ops_fit_box(image.size, size, bleed, centering)
    if box is None:
        return image.crop((0, 0, *size))
    return _resize(image, size, method, box)


def smart_crop(image: Image.Image, size: tuple[int, int], method: int = Image.Resampling.BICUBIC) -> Image.Image:
    """Crop around visual saliency and resize to exactly ``size``.

    Args:
        image: Nonempty input image; only the current frame is used.
        size: Positive integer output ``(width, height)`` in pixels.
        method: Resampling filter from ``Image.Resampling``.

    The largest rectangle with the requested aspect ratio is positioned using
    color contrast, edges, and visible alpha on an 8-bit RGBA thumbnail no
    larger than 128x128. Content near the crop center scores higher. Flat or
    fully transparent images use a centered crop. This heuristic does not
    detect faces or guarantee that every subject remains inside the crop.

    Output retains the source mode, depth, palette, and metadata. Analysis
    ignores ICC profiles and does not apply EXIF orientation; call
    ``exif_transpose`` first when needed. No optional dependencies are required.

    Returns:
        A new image of exactly ``size``.

    Raises:
        TypeError: ``image`` is not a Blanket image.
        ValueError: ``size`` is not two positive integers, or the image is empty.

    Examples:
        ```python
        from blanket import Image, ImageOps

        image = Image.new("RGB", (600, 400), "navy")
        thumbnail = ImageOps.smart_crop(image, (128, 128))
        ```
    """
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a Blanket Image")
    if len(size) != 2:
        raise ValueError("size must contain two positive integers")
    width, height = (index(value) for value in size)
    if width <= 0 or height <= 0:
        raise ValueError("size must contain two positive integers")
    if image.width == 0 or image.height == 0:
        raise ValueError("cannot smart crop an empty image")
    analysis_size = ops_smart_crop_size(image.size, (width, height))
    if analysis_size is None:
        return image.resize((width, height), method)
    source = image.convert("LA") if image.mode == "La" else image
    analysis = source.convert("RGBA", bit_depth=8).resize(analysis_size, Image.Resampling.BOX)
    box = ops_smart_crop_box(analysis._native, image.size, (width, height))
    return image.resize((width, height), method, box)


def flip(image: Image.Image) -> Image.Image:
    """Flip vertically.

    Args:
        image: Input image.

    Returns:
        A new vertically flipped image.

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

    Returns:
        A new mode ``L`` image.

    Examples:
        ```python
        from blanket import Image, ImageOps

        image = Image.new("RGB", (8, 8), (40, 100, 180))
        result = ImageOps.grayscale(image)
        ```
    """
    return image.convert("L")


def invert(image: Image.Image) -> Image.Image:
    """Subtract each channel value from the sample depth's maximum.

    Args:
        image: Input image.

    Returns:
        A new image with inverted channels.

    Examples:
        ```python
        from blanket import Image, ImageOps

        image = Image.new("RGB", (8, 8), (40, 100, 180))
        result = ImageOps.invert(image)
        ```
    """
    if image.mode in ("L", "RGB"):
        return ImageChops.invert(image)
    if image.mode == "1":
        return _result(image, ops_lut(image._native, ops_tone_lut("invert")))
    return _lut(image, ops_tone_lut("invert"), palette=True)


def mirror(image: Image.Image) -> Image.Image:
    """Flip horizontally.

    Args:
        image: Input image.

    Returns:
        A new horizontally flipped image.

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

    Returns:
        A new image using only the retained high bits.

    Raises:
        ValueError: ``bits`` is outside 1 through 8.

    Examples:
        ```python
        from blanket import Image, ImageOps

        image = Image.new("RGB", (8, 8), (40, 100, 180))
        result = ImageOps.posterize(image, bits=4)
        ```
    """
    if not 1 <= bits <= 8:
        raise ValueError("bits must be between or equal to 1 and 8")
    return _lut(image, ops_tone_lut("posterize", index(bits)), palette=True)


def solarize(image: Image.Image, threshold: int = 128) -> Image.Image:
    """Invert channel values greater than or equal to threshold.

    Args:
        image: Input image.
        threshold: Intensity threshold, on the 0 through 255 scale.

    Returns:
        A new image with samples at or above ``threshold`` inverted.

    Examples:
        ```python
        from blanket import Image, ImageOps

        image = Image.new("RGB", (8, 8), (40, 100, 180))
        result = ImageOps.solarize(image, threshold=128)
        ```
    """
    return _lut(image, ops_tone_lut("solarize", threshold), palette=True)


@overload
def exif_transpose(image: Image.Image, *, in_place: Literal[True]) -> None:
    """Apply EXIF orientation in-place and return None.

    Args:
        image: Input image.
        in_place: Must be true for this overload.

    Examples:
        ```python
        from blanket import Image, ImageOps

        image = Image.new("RGB", (8, 8), (40, 100, 180))
        ImageOps.exif_transpose(image, in_place=True)
        ```
    """


@overload
def exif_transpose(image: Image.Image, *, in_place: Literal[False] = False) -> Image.Image:
    """Apply EXIF orientation and return a transformed copy.

    Args:
        image: Input image.
        in_place: Must be false for this overload.

    Returns:
        A transformed copy with orientation metadata removed.

    Examples:
        ```python
        from blanket import Image, ImageOps

        image = Image.new("RGB", (8, 8), (40, 100, 180))
        result = ImageOps.exif_transpose(image)
        ```
    """


def exif_transpose(image: Image.Image, *, in_place: bool = False) -> Image.Image | None:
    """Apply EXIF orientation and remove it from EXIF and XMP metadata.

    Return a copy by default, or update the original and return None when
    ``in_place=True``. Other EXIF entries and their offsets are preserved.

    Args:
        image: Input image.
        in_place: Modify the input and return None when true; otherwise return a transformed copy.

    Returns:
        The transformed copy, or None when ``in_place`` is true.

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
