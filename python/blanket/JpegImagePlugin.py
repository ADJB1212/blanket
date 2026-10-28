"""Compressed-domain JPEG transforms without pixel decoding or recompression."""

from __future__ import annotations

from typing import TYPE_CHECKING

from . import Image
from ._blanket import _transform_jpeg

if TYPE_CHECKING:
    from typing import Literal

__all__ = ["transform"]


def transform(
    data: bytes | bytearray | memoryview,
    method: Image.Transpose | int | None = None,
    *,
    crop: tuple[int, int, int, int] | None = None,
    trim: bool = False,
    metadata: Literal["strip", "preserve"] = "strip",
) -> bytes:
    """Return transformed JPEG bytes, preserving quantized DCT coefficients.

    ``method`` uses Image.Transpose values (ROTATE_90 is counterclockwise).
    ``crop`` is a (left, top, right, bottom) box in source coordinates, applied
    before the transform. Its origin must align to the source JPEG's MCU grid
    (typically 16 x 16 for 4:2:0, 8 x 8 for grayscale or 4:4:4). The right and
    bottom boundaries may be unaligned. Out-of-bounds boxes are rejected.

    Imperfect transforms raise OSError unless ``trim=True`` allows discarding
    partial edge blocks. No pixels are silently left untransformed.

    ``metadata="strip"`` removes ancillary markers, including EXIF, XMP, ICC,
    and comments. ``"preserve"`` retains ancillary metadata; libjpeg may update
    EXIF pixel dimensions, but does not normalize metadata. Orientation,
    dimensions, density, and embedded thumbnails may be stale.
    EXIF orientation is never applied automatically. JPEG structural markers
    required to interpret the color data are retained in either mode.

    Input must be encoded JPEG bytes, not an opened Image. Malformed or
    unsupported JPEGs raise OSError. Output encoding and decoded pixels need
    not be byte-identical: IDCT rounding and chroma upsampling can differ after
    transforms even though no coefficient requantization occurs.
    """
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("data must be JPEG bytes")
    if method is not None:
        if isinstance(method, bool) or not isinstance(method, int):
            raise TypeError("method must be an Image.Transpose value or None")
        method = Image.Transpose(method)
    if not isinstance(trim, bool):
        raise TypeError("trim must be a bool")
    if metadata not in ("strip", "preserve"):
        raise ValueError("metadata must be 'strip' or 'preserve'")
    if crop is not None:
        if not isinstance(crop, tuple) or len(crop) != 4:
            raise TypeError("crop must be a tuple of four integers")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in crop):
            raise TypeError("crop coordinates must be integers")
        if any(value < 0 or value > 2**31 - 1 for value in crop):
            raise ValueError("crop coordinates must be between 0 and 2147483647")
    return _transform_jpeg(bytes(data), method, crop, trim, metadata == "preserve")
