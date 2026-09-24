"""Pillow-compatible channel operations with native Rust pixel processing.

Arithmetic supports 8-bit L, RGB, RGBA, and indexed P pixels. All channels,
including alpha, participate. Binary arithmetic crops to the top-left overlap.
Logical operations accept bilevel mode 1 images.
"""

from __future__ import annotations

from . import Image
from ._blanket import chops_binary, chops_invert, chops_offset

__all__ = [
    "add",
    "add_modulo",
    "blend",
    "composite",
    "constant",
    "darker",
    "difference",
    "duplicate",
    "hard_light",
    "invert",
    "lighter",
    "logical_and",
    "logical_or",
    "logical_xor",
    "multiply",
    "offset",
    "overlay",
    "screen",
    "soft_light",
    "subtract",
    "subtract_modulo",
]


def _logical(image1: Image.Image, image2: Image.Image, operation: str) -> Image.Image:
    if image1.mode != "1" or image2.mode != "1":
        raise ValueError("image has wrong mode")
    width = min(image1.width, image2.width)
    height = min(image1.height, image2.height)
    first = image1.crop((0, 0, width, height)).tobytes()
    second = image2.crop((0, 0, width, height)).tobytes()
    if operation == "and":
        pixels = bytes(a & b for a, b in zip(first, second, strict=True))
    elif operation == "or":
        pixels = bytes(a | b for a, b in zip(first, second, strict=True))
    else:
        pixels = bytes(a ^ b for a, b in zip(first, second, strict=True))
    return Image.frombytes("1", (width, height), pixels)


def logical_and(image1: Image.Image, image2: Image.Image) -> Image.Image:
    """Return the bitwise AND of two bilevel images."""
    return _logical(image1, image2, "and")


def logical_or(image1: Image.Image, image2: Image.Image) -> Image.Image:
    """Return the bitwise OR of two bilevel images."""
    return _logical(image1, image2, "or")


def logical_xor(image1: Image.Image, image2: Image.Image) -> Image.Image:
    """Return the bitwise XOR of two bilevel images."""
    return _logical(image1, image2, "xor")


def _binary(image1: Image.Image, image2: Image.Image, operation: str, scale: float = 1.0, offset: int = 0) -> Image.Image:
    image1.load()
    image2.load()
    result = Image.Image(chops_binary(image1._native, image2._native, operation, scale, offset))
    result.info.update(image1.info)
    return result


def difference(image1: Image.Image, image2: Image.Image) -> Image.Image:
    """Return the absolute difference between corresponding channel values.

    Args:
        image1: First input image; its mode and dimensions must match the second image.
        image2: Second input image.

    Examples:
        ```python
        from blanket import Image, ImageChops

        image = Image.new("RGB", (8, 8), (40, 100, 180))
        other = Image.new("RGB", image.size, "white")
        result = ImageChops.difference(image, other)
        ```
    """
    return _binary(image1, image2, "difference")


def multiply(image1: Image.Image, image2: Image.Image) -> Image.Image:
    """Multiply corresponding channel values and divide by 255.

    Args:
        image1: First input image; its mode and dimensions must match the second image.
        image2: Second input image.

    Examples:
        ```python
        from blanket import Image, ImageChops

        image = Image.new("RGB", (8, 8), (40, 100, 180))
        other = Image.new("RGB", image.size, "white")
        result = ImageChops.multiply(image, other)
        ```
    """
    return _binary(image1, image2, "multiply")


def screen(image1: Image.Image, image2: Image.Image) -> Image.Image:
    """Invert, multiply, and invert again to lighten the inputs.

    Args:
        image1: First input image; its mode and dimensions must match the second image.
        image2: Second input image.

    Examples:
        ```python
        from blanket import Image, ImageChops

        image = Image.new("RGB", (8, 8), (40, 100, 180))
        other = Image.new("RGB", image.size, "white")
        result = ImageChops.screen(image, other)
        ```
    """
    return _binary(image1, image2, "screen")


def lighter(image1: Image.Image, image2: Image.Image) -> Image.Image:
    """Return the larger value in each channel.

    Args:
        image1: First input image; its mode and dimensions must match the second image.
        image2: Second input image.

    Examples:
        ```python
        from blanket import Image, ImageChops

        image = Image.new("RGB", (8, 8), (40, 100, 180))
        other = Image.new("RGB", image.size, "white")
        result = ImageChops.lighter(image, other)
        ```
    """
    return _binary(image1, image2, "lighter")


def darker(image1: Image.Image, image2: Image.Image) -> Image.Image:
    """Return the smaller value in each channel.

    Args:
        image1: First input image; its mode and dimensions must match the second image.
        image2: Second input image.

    Examples:
        ```python
        from blanket import Image, ImageChops

        image = Image.new("RGB", (8, 8), (40, 100, 180))
        other = Image.new("RGB", image.size, "white")
        result = ImageChops.darker(image, other)
        ```
    """
    return _binary(image1, image2, "darker")


def add(image1: Image.Image, image2: Image.Image, scale: float = 1.0, offset: int = 0) -> Image.Image:
    """Add channels, divide by scale, add offset, and clip to 0..255.

    Args:
        image1: First input image; its mode and dimensions must match the second image.
        image2: Second input image.
        scale: Divisor applied before the offset; must be nonzero.
        offset: Value added after scaling.

    Examples:
        ```python
        from blanket import Image, ImageChops

        image = Image.new("RGB", (8, 8), (40, 100, 180))
        other = Image.new("RGB", image.size, "white")
        result = ImageChops.add(image, other)
        ```
    """
    return _binary(image1, image2, "add", scale, offset)


def subtract(image1: Image.Image, image2: Image.Image, scale: float = 1.0, offset: int = 0) -> Image.Image:
    """Subtract channels, divide by scale, add offset, and clip to 0..255.

    Args:
        image1: First input image; its mode and dimensions must match the second image.
        image2: Second input image.
        scale: Divisor applied before the offset; must be nonzero.
        offset: Value added after scaling.

    Examples:
        ```python
        from blanket import Image, ImageChops

        image = Image.new("RGB", (8, 8), (40, 100, 180))
        other = Image.new("RGB", image.size, "white")
        result = ImageChops.subtract(image, other)
        ```
    """
    return _binary(image1, image2, "subtract", scale, offset)


def add_modulo(image1: Image.Image, image2: Image.Image) -> Image.Image:
    """Add channels with wraparound modulo 256.

    Args:
        image1: First input image; its mode and dimensions must match the second image.
        image2: Second input image.

    Examples:
        ```python
        from blanket import Image, ImageChops

        image = Image.new("RGB", (8, 8), (40, 100, 180))
        other = Image.new("RGB", image.size, "white")
        result = ImageChops.add_modulo(image, other)
        ```
    """
    return _binary(image1, image2, "add_modulo")


def subtract_modulo(image1: Image.Image, image2: Image.Image) -> Image.Image:
    """Subtract channels with wraparound modulo 256.

    Args:
        image1: First input image; its mode and dimensions must match the second image.
        image2: Second input image.

    Examples:
        ```python
        from blanket import Image, ImageChops

        image = Image.new("RGB", (8, 8), (40, 100, 180))
        other = Image.new("RGB", image.size, "white")
        result = ImageChops.subtract_modulo(image, other)
        ```
    """
    return _binary(image1, image2, "subtract_modulo")


def soft_light(image1: Image.Image, image2: Image.Image) -> Image.Image:
    """Combine images using Pillow's soft-light blend.

    Args:
        image1: First input image; its mode and dimensions must match the second image.
        image2: Second input image.

    Examples:
        ```python
        from blanket import Image, ImageChops

        image = Image.new("RGB", (8, 8), (40, 100, 180))
        other = Image.new("RGB", image.size, "white")
        result = ImageChops.soft_light(image, other)
        ```
    """
    return _binary(image1, image2, "soft_light")


def hard_light(image1: Image.Image, image2: Image.Image) -> Image.Image:
    """Multiply or screen according to the second image's channel values.

    Args:
        image1: First input image; its mode and dimensions must match the second image.
        image2: Second input image.

    Examples:
        ```python
        from blanket import Image, ImageChops

        image = Image.new("RGB", (8, 8), (40, 100, 180))
        other = Image.new("RGB", image.size, "white")
        result = ImageChops.hard_light(image, other)
        ```
    """
    return _binary(image1, image2, "hard_light")


def overlay(image1: Image.Image, image2: Image.Image) -> Image.Image:
    """Multiply or screen according to the first image's channel values.

    Args:
        image1: First input image; its mode and dimensions must match the second image.
        image2: Second input image.

    Examples:
        ```python
        from blanket import Image, ImageChops

        image = Image.new("RGB", (8, 8), (40, 100, 180))
        other = Image.new("RGB", image.size, "white")
        result = ImageChops.overlay(image, other)
        ```
    """
    return _binary(image1, image2, "overlay")


def invert(image: Image.Image) -> Image.Image:
    """Subtract every channel, including alpha, from 255.

    Args:
        image: Input image.

    Examples:
        ```python
        from blanket import Image, ImageChops

        image = Image.new("RGB", (8, 8), (40, 100, 180))
        result = ImageChops.invert(image)
        ```
    """
    image.load()
    result = Image.Image(chops_invert(image._native))
    result.info.update(image.info)
    return result


def offset(image: Image.Image, xoffset: int, yoffset: int | None = None) -> Image.Image:
    """Translate with wraparound; the vertical offset defaults to xoffset.

    Args:
        image: Input image.
        xoffset: Horizontal translation in pixels; pixels wrap around.
        yoffset: Vertical translation, or None to use the horizontal offset.

    Examples:
        ```python
        from blanket import Image, ImageChops

        image = Image.new("RGB", (8, 8), (40, 100, 180))
        result = ImageChops.offset(image, xoffset=2, yoffset=1)
        ```
    """
    image.load()
    result = Image.Image(chops_offset(image._native, xoffset, xoffset if yoffset is None else yoffset))
    result.info.update(image.info)
    return result


def constant(image: Image.Image, value: int) -> Image.Image:
    """Return a constant L image with the input's dimensions.

    Args:
        image: Input image.
        value: Pixel value or channel tuple.

    Examples:
        ```python
        from blanket import Image, ImageChops

        image = Image.new("RGB", (8, 8), (40, 100, 180))
        result = ImageChops.constant(image, value=128)
        ```
    """
    return Image.new("L", image.size, value)


def duplicate(image: Image.Image) -> Image.Image:
    """Return an independent copy of an image.

    Args:
        image: Input image.

    Examples:
        ```python
        from blanket import Image, ImageChops

        image = Image.new("RGB", (8, 8), (40, 100, 180))
        result = ImageChops.duplicate(image)
        ```
    """
    return image.copy()


def blend(image1: Image.Image, image2: Image.Image, alpha: float) -> Image.Image:
    """Interpolate two images with a constant blending factor.

    Args:
        image1: First input image; its mode and dimensions must match the second image.
        image2: Second input image.
        alpha: Blending weight: 0 selects the first image and 1 the second; values outside this range extrapolate.

    Examples:
        ```python
        from blanket import Image, ImageChops

        image = Image.new("RGB", (8, 8), (40, 100, 180))
        other = Image.new("RGB", image.size, "white")
        result = ImageChops.blend(image, other, alpha=0.5)
        ```
    """
    return Image.blend(image1, image2, alpha)


def composite(image1: Image.Image, image2: Image.Image, mask: Image.Image) -> Image.Image:
    """Select between images using an L or RGBA mask.

    Args:
        image1: First input image; its mode and dimensions must match the second image.
        image2: Second input image.
        mask: Optional mask selecting pixels. Histogram operations require an L mask; compositing also accepts RGBA alpha.

    Examples:
        ```python
        from blanket import Image, ImageChops

        image = Image.new("RGB", (8, 8), (40, 100, 180))
        other = Image.new("RGB", image.size, "white")
        mask = Image.new("L", image.size, 128)
        result = ImageChops.composite(image, other, mask)
        ```
    """
    return Image.composite(image1, image2, mask)
