"""Pillow-compatible channel operations with native Rust pixel processing.

Arithmetic supports 8-bit L, RGB, RGBA, and indexed P pixels. All channels,
including alpha, participate. Binary arithmetic crops to the top-left overlap.
Logical operations requiring Pillow's bilevel mode 1 are not supported.
"""

from __future__ import annotations

from . import Image
from ._blanket import chops_binary, chops_invert, chops_offset

__all__ = [
    "add", "add_modulo", "blend", "composite", "constant", "darker",
    "difference", "duplicate", "hard_light", "invert", "lighter", "multiply",
    "offset", "overlay", "screen", "soft_light", "subtract", "subtract_modulo",
]


def _binary(image1: Image.Image, image2: Image.Image, operation: str, scale: float = 1.0, offset: int = 0) -> Image.Image:
    image1.load()
    image2.load()
    result = Image.Image(chops_binary(image1._native, image2._native, operation, scale, offset))
    result.info.update(image1.info)
    return result


def difference(image1: Image.Image, image2: Image.Image) -> Image.Image:
    """Return the absolute difference between corresponding channel values."""
    return _binary(image1, image2, "difference")


def multiply(image1: Image.Image, image2: Image.Image) -> Image.Image:
    """Multiply corresponding channel values and divide by 255."""
    return _binary(image1, image2, "multiply")


def screen(image1: Image.Image, image2: Image.Image) -> Image.Image:
    """Invert, multiply, and invert again to lighten the inputs."""
    return _binary(image1, image2, "screen")


def lighter(image1: Image.Image, image2: Image.Image) -> Image.Image:
    """Return the larger value in each channel."""
    return _binary(image1, image2, "lighter")


def darker(image1: Image.Image, image2: Image.Image) -> Image.Image:
    """Return the smaller value in each channel."""
    return _binary(image1, image2, "darker")


def add(image1: Image.Image, image2: Image.Image, scale: float = 1.0, offset: int = 0) -> Image.Image:
    """Add channels, divide by scale, add offset, and clip to 0..255."""
    return _binary(image1, image2, "add", scale, offset)


def subtract(image1: Image.Image, image2: Image.Image, scale: float = 1.0, offset: int = 0) -> Image.Image:
    """Subtract channels, divide by scale, add offset, and clip to 0..255."""
    return _binary(image1, image2, "subtract", scale, offset)


def add_modulo(image1: Image.Image, image2: Image.Image) -> Image.Image:
    """Add channels with wraparound modulo 256."""
    return _binary(image1, image2, "add_modulo")


def subtract_modulo(image1: Image.Image, image2: Image.Image) -> Image.Image:
    """Subtract channels with wraparound modulo 256."""
    return _binary(image1, image2, "subtract_modulo")


def soft_light(image1: Image.Image, image2: Image.Image) -> Image.Image:
    """Combine images using Pillow's soft-light blend."""
    return _binary(image1, image2, "soft_light")


def hard_light(image1: Image.Image, image2: Image.Image) -> Image.Image:
    """Multiply or screen according to the second image's channel values."""
    return _binary(image1, image2, "hard_light")


def overlay(image1: Image.Image, image2: Image.Image) -> Image.Image:
    """Multiply or screen according to the first image's channel values."""
    return _binary(image1, image2, "overlay")


def invert(image: Image.Image) -> Image.Image:
    """Subtract every channel, including alpha, from 255."""
    image.load()
    result = Image.Image(chops_invert(image._native))
    result.info.update(image.info)
    return result


def offset(image: Image.Image, xoffset: int, yoffset: int | None = None) -> Image.Image:
    """Translate with wraparound; the vertical offset defaults to xoffset."""
    image.load()
    result = Image.Image(chops_offset(image._native, xoffset, xoffset if yoffset is None else yoffset))
    result.info.update(image.info)
    return result


def constant(image: Image.Image, value: int) -> Image.Image:
    """Return a constant L image with the input's dimensions."""
    return Image.new("L", image.size, value)


def duplicate(image: Image.Image) -> Image.Image:
    """Return an independent copy of an image."""
    return image.copy()


def blend(image1: Image.Image, image2: Image.Image, alpha: float) -> Image.Image:
    """Interpolate two images with a constant blending factor."""
    return Image.blend(image1, image2, alpha)


def composite(image1: Image.Image, image2: Image.Image, mask: Image.Image) -> Image.Image:
    """Select between images using an L or RGBA mask."""
    return Image.composite(image1, image2, mask)
