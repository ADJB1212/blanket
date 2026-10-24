"""Perceptual image hashes for deduplication and similarity search.

All algorithms convert the current frame to 8-bit L and resize with Lanczos.
Alpha is discarded, orientation is not applied, and ICC profiles are not used.
Compare hashes produced with the same algorithm and settings. These are visual
fingerprints, not cryptographic hashes or guarantees of identical pixels.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isqrt

from . import Image
from ._blanket import image_hash

__all__ = ["ImageHash", "average_hash", "dhash", "hex_to_hash", "phash"]


def _size(hash_size: int) -> int:
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
    if not hexstr or any(c not in "0123456789abcdefABCDEF" for c in hexstr):
        raise ValueError("expected hexadecimal digits")
    size = _size(isqrt(len(hexstr) * 4) if hash_size is None else hash_size)
    if len(hexstr) != (size**2 + 3) // 4:
        raise ValueError("hex length does not match hash size")
    return ImageHash(int(hexstr, 16), size)


def _hash(image: Image.Image, hash_size: int, side: int, algorithm: str) -> ImageHash:
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a Blanket Image")
    if image.width == 0 or image.height == 0:
        raise ValueError("cannot hash an empty image")
    width, height = (hash_size + 1, hash_size) if algorithm == "difference" else (side, side)
    thumbnail = image.convert("L", bit_depth=8).resize((width, height), Image.Resampling.LANCZOS)
    packed = image_hash(thumbnail.tobytes(), hash_size, side, algorithm)
    return ImageHash(int.from_bytes(bytes(packed), "big"), hash_size)


def average_hash(image: Image.Image, hash_size: int = 8) -> ImageHash:
    """Return aHash: each bit is true when a sample exceeds the thumbnail mean.

    ``hash_size`` is a side length from 2 to 64; the default produces 64 bits.
    """
    size = _size(hash_size)
    return _hash(image, size, size, "average")


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
