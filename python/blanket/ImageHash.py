"""Perceptual image hashes for deduplication and similarity search.

The spatial algorithms convert to 8-bit L and resize with Lanczos. Color and
crop-resistant hashes use orderless histograms without resizing the image.
Alpha is discarded, orientation is not applied, and ICC profiles are not used.
Compare hashes produced with the same algorithm and settings. These are visual
fingerprints, not cryptographic hashes or guarantees of identical pixels.
"""

from __future__ import annotations

__lazy_modules__ = {"math", "string"}

import string
from contextlib import ExitStack
from dataclasses import dataclass
from math import isqrt
from typing import TYPE_CHECKING

from . import Image
from ._blanket import hash_group, hash_nearest_neighbors, image_hash_native, image_histogram_hash_native

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = ["ImageHash", "average_hash", "colorhash", "crop_resistant_hash", "dhash", "group_hashes", "hex_to_hash", "nearest_neighbors", "phash"]


def _size(hash_size: int) -> int:
    """Validate hash side length between 2 and 64."""
    if isinstance(hash_size, bool) or not isinstance(hash_size, int):
        raise TypeError("hash_size must be an integer")
    if not 2 <= hash_size <= 64:
        raise ValueError("hash_size must be between 2 and 64")
    return hash_size


@dataclass(frozen=True, slots=True)
class ImageHash:
    """Immutable square bit hash, stored as an integer and side length.

    Bits are ordered by row, with the top-left bit most significant.
    ``str(value)`` returns lowercase hex padded to fit ``hash_size ** 2`` bits.
    Subtraction returns Hamming distance; values can also be dictionary keys.
    The algorithm is not stored in the value.
    """

    value: int
    hash_size: int = 8

    def __post_init__(self) -> None:
        """Validate hash size and integer bit capacity."""
        _size(self.hash_size)
        if isinstance(self.value, bool) or not isinstance(self.value, int):
            raise TypeError("value must be an integer")
        if not 0 <= self.value < 1 << len(self):
            raise ValueError("value does not fit the hash size")

    def __len__(self) -> int:
        """Return the number of bits."""
        return self.hash_size**2

    def __str__(self) -> str:
        """Return the zero-padded hexadecimal representation."""
        return f"{self.value:0{(len(self) + 3) // 4}x}"

    def __sub__(self, other: ImageHash) -> int:
        """Return the number of differing bits; require matching sizes."""
        if not isinstance(other, ImageHash):
            return NotImplemented
        if self.hash_size != other.hash_size:
            raise ValueError("hash sizes must match")
        return (self.value ^ other.value).bit_count()


def hex_to_hash(hexstr: str, hash_size: int | None = None) -> ImageHash:
    """Restore a hash from hex, optionally specifying its square side length.

    If omitted, the side length is inferred from the number of hex digits.
    Leading zero digits must be retained. Nonzero padding bits are rejected.
    """
    if not isinstance(hexstr, str):
        raise TypeError("hexstr must be a string")
    if not hexstr or any(c not in string.hexdigits for c in hexstr):
        raise ValueError("expected hexadecimal digits")
    size = _size(isqrt(len(hexstr) * 4) if hash_size is None else hash_size)
    if len(hexstr) != (size**2 + 3) // 4:
        raise ValueError("hex length does not match hash size")
    return ImageHash(int(hexstr, 16), size)


def _validate_max_distance(max_distance: int) -> None:
    if isinstance(max_distance, bool) or not isinstance(max_distance, int):
        raise TypeError("max_distance must be an integer")
    if max_distance < 0:
        raise ValueError("max_distance must be nonnegative")


def _pack_hashes(hashes: Iterable[ImageHash], hash_size: int | None = None) -> tuple[bytes, int]:
    width = 0 if hash_size is None else (hash_size**2 + 7) // 8
    packed = bytearray()
    for fingerprint in hashes:
        if not isinstance(fingerprint, ImageHash):
            raise TypeError("hashes must contain ImageHash objects")
        if hash_size is None:
            hash_size = fingerprint.hash_size
            width = (hash_size**2 + 7) // 8
        elif fingerprint.hash_size != hash_size:
            raise ValueError("hash sizes must match")
        packed.extend(int.to_bytes(fingerprint.value, width, "big"))
    return bytes(packed), width


def nearest_neighbors(query: ImageHash, hashes: Iterable[ImageHash], max_distance: int) -> list[tuple[int, int]]:
    """Return ``(index, distance)`` pairs within the inclusive Hamming radius.

    Indices are zero-based positions in ``hashes``. Sort by distance, breaking
    ties by input order. Every occurrence of a matching hash is returned.
    Use hashes from the same algorithm and settings. All sizes must match
    the query, and ``max_distance`` must be a nonnegative integer.
    """
    _validate_max_distance(max_distance)
    if not isinstance(query, ImageHash):
        raise TypeError("query must be an ImageHash")
    packed, width = _pack_hashes(hashes, query.hash_size)
    query_bytes = int.to_bytes(query.value, width, "big")
    return hash_nearest_neighbors(query_bytes, packed, min(max_distance, width * 8))


def group_hashes(hashes: Iterable[ImageHash], max_distance: int) -> list[list[int]]:
    """Group zero-based input positions linked within the Hamming radius.

    Two positions share a group if a chain of pairs at distance at most
    ``max_distance`` connects them. Include singletons and duplicate positions.
    Order positions within each group by input order and groups by their first
    position. Use hashes from the same algorithm and settings. All sizes must
    match, and ``max_distance`` must be a nonnegative integer.

    Grouping may compare every pair of distinct hashes in the worst case.
    """
    _validate_max_distance(max_distance)
    packed, width = _pack_hashes(hashes)
    if not packed:
        return []
    return hash_group(packed, width, min(max_distance, width * 8))


def _hash(image: Image.Image, hash_size: int, side: int, algorithm: str) -> ImageHash:
    """Compute an image hash with the given algorithm and dimensions."""
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a Blanket Image")
    if image.width == 0 or image.height == 0:
        raise ValueError("cannot hash an empty image")
    width, height = (hash_size + 1, hash_size) if algorithm == "difference" else (side, side)
    grayscale = image if image.mode == "L" and image.bit_depth == 8 else image.convert("L", bit_depth=8)
    thumbnail = grayscale if grayscale.size == (width, height) else grayscale.resize((width, height), Image.Resampling.LANCZOS)
    packed = image_hash_native(thumbnail._native, hash_size, side, algorithm)
    return ImageHash(int.from_bytes(packed, "big"), hash_size)


def average_hash(image: Image.Image, hash_size: int = 8) -> ImageHash:
    """Return aHash: each bit is true when a sample exceeds the thumbnail mean.

    ``hash_size`` is a side length from 2 to 64; the default produces 64 bits.
    """
    size = _size(hash_size)
    return _hash(image, size, size, "average")


def _histogram_hash(image: Image.Image, hash_size: int, algorithm: str, mode: str) -> ImageHash:
    """Convert the current frame and project its native histogram to bits."""
    size = _size(hash_size)
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a Blanket Image")
    image.load()
    if not image.width or not image.height:
        raise ValueError("cannot hash an empty image")
    with ExitStack() as cleanup:
        source = image
        if mode == "L" and source.mode in ("RGB;F", "RGBA;F"):
            source = source.convert("RGB", bit_depth=8)
            cleanup.callback(source.close)
        converted = source if source.mode == mode and source.bit_depth == 8 else source.convert(mode, bit_depth=8)
        if converted is not source:
            cleanup.callback(converted.close)
        packed = image_histogram_hash_native(converted._native, size, algorithm)
        return ImageHash(int.from_bytes(packed, "big"), size)


def colorhash(image: Image.Image, hash_size: int = 8) -> ImageHash:
    """Hash the color distribution without retaining spatial arrangement.

    Convert the current frame to 8-bit RGB without resizing. Count a 4x4x4
    RGB histogram (each channel divided into four equal bins), take square
    roots of its counts, and project onto fixed deterministic sign vectors.
    Positive projections set bits; ties leave bits clear. Square-root weights
    reduce the influence of dominant colors. Pixel order does not matter.

    Alpha is discarded, including hidden colors. ICC and orientation are
    ignored. Quantization boundaries can change bits abruptly; different
    arrangements of the same colors collide. This is Blanket's algorithm,
    with no promise of bit compatibility with other color-hash libraries.

    Args:
        image: Nonempty source image; the current frame stays unchanged.
        hash_size: Square side from 2 to 64; the default produces 64 bits.
    """
    return _histogram_hash(image, hash_size, "color", "RGB")


def crop_resistant_hash(image: Image.Image, hash_size: int = 8) -> ImageHash:
    """Hash local grayscale texture while discarding spatial arrangement.

    Convert the current frame to 8-bit L without resizing (float color goes
    through 8-bit RGB first). For each interior
    pixel, compare its eight neighbors with the center (greater or equal),
    starting at northwest and proceeding clockwise, least significant bit
    first. Count the resulting 256 local binary patterns and project their
    square-root counts onto fixed deterministic sign vectors, setting bits
    for positive projections and clearing ties.

    Crops retaining similar texture distributions tend to have small Hamming
    distances. This does not match arbitrary crops, and resizing can change
    local texture. Spatially different scenes with the same texture can
    collide; all constant grayscale images share a hash. Alpha, ICC, and
    orientation are ignored. This is a single fixed-size Blanket fingerprint,
    not a segmented multi-hash compatible with other libraries.

    Args:
        image: Source at least 3x3 pixels; the current frame stays unchanged.
        hash_size: Square side from 2 to 64; the default produces 64 bits.
    """
    return _histogram_hash(image, hash_size, "crop-resistant", "L")


def dhash(image: Image.Image, hash_size: int = 8) -> ImageHash:
    """Return horizontal dHash: a bit is true when the right sample is brighter.

    The thumbnail is ``(hash_size + 1, hash_size)``. Side lengths are 2 to 64.
    """
    size = _size(hash_size)
    return _hash(image, size, size, "difference")


def phash(image: Image.Image, hash_size: int = 8, highfreq_factor: int = 4) -> ImageHash:
    """Return pHash using a separable, unnormalized two-dimensional DCT-II.

    Resize to ``hash_size * highfreq_factor`` square, then threshold the top-left
    ``hash_size`` square coefficients against their median, including DC.
    Near-zero AC coefficients are rounded to zero to suppress numerical noise.
    Side lengths are 2 to 64; the positive integer factor must keep the resized
    side at most 256. Default settings produce a 64-bit hash from 32x32 samples.
    """
    size = _size(hash_size)
    if isinstance(highfreq_factor, bool) or not isinstance(highfreq_factor, int):
        raise TypeError("highfreq_factor must be an integer")
    side = size * highfreq_factor
    if highfreq_factor < 1 or side > 256:
        raise ValueError("highfreq_factor must be positive and hash_size * highfreq_factor must not exceed 256")
    return _hash(image, size, side, "perceptual")
