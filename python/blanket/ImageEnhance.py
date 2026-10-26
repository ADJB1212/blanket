"""Pillow-compatible image enhancement classes backed by native kernels.

Color, Contrast, Brightness, and Sharpness operate on 8-bit images, including
`1`, `LA`, `PA`, `HSV`, `CMYK`, `YCbCr`, and `LAB`, as well as 10/12/16-bit
`L`/`RGB`/`RGBA` and the `I;16` variants. High-depth outputs retain their
sample range and storage mode. Finite enhancement factors preserve RGBA alpha.
"""

from __future__ import annotations

from . import Image
from ._blanket import enhance_blend, enhance_brightness, enhance_color, enhance_contrast, enhance_sharpness

__all__ = ["Brightness", "Color", "Contrast", "Sharpness"]


class _Enhance:
    """Share factor-based blending between an image and its degenerate form."""

    image: Image.Image
    degenerate: Image.Image

    def enhance(self, factor: float) -> Image.Image:
        """Return an enhanced copy, interpolated by an unrestricted factor.

        Args:
            factor: Enhancement factor: 1 retains the image, 0 gives the degenerate image, and values above 1 strengthen the effect.

        Examples:
            ```python
            from blanket import Image, ImageEnhance

            image = Image.new("RGB", (8, 8), (40, 100, 180))
            enhancer = ImageEnhance.Contrast(image)
            result = enhancer.enhance(1.5)
            ```
        """

        result = Image.Image(enhance_blend(self.degenerate._native, self.image._native, factor))
        result.info.update(self.degenerate.info)
        return result


class Color(_Enhance):
    """Adjust image color balance."""

    def __init__(self, image: Image.Image) -> None:
        """Configure Color.

        Args:
            image: Input image.

        Examples:
            ```python
            from blanket import Image, ImageEnhance

            image = Image.new("RGB", (8, 8), (40, 100, 180))
            enhancer = ImageEnhance.Color(image)
            result = enhancer.enhance(1.5)
            ```
        """
        self.image = image
        self.intermediate_mode = "LA" if image.mode == "RGBA" else "L"
        self.degenerate = image if image.mode == "L" else Image.Image(enhance_color(image._native))
        if self.degenerate is not image:
            self.degenerate.info.update(image.info)


class Contrast(_Enhance):
    """Adjust image contrast around the rounded mean luminance."""

    def __init__(self, image: Image.Image) -> None:
        """Configure Contrast.

        Args:
            image: Input image.

        Examples:
            ```python
            from blanket import Image, ImageEnhance

            image = Image.new("RGB", (8, 8), (40, 100, 180))
            enhancer = ImageEnhance.Contrast(image)
            result = enhancer.enhance(1.5)
            ```
        """
        self.image = image
        self.degenerate = Image.Image(enhance_contrast(image._native))


class Brightness(_Enhance):
    """Adjust image brightness relative to black."""

    def __init__(self, image: Image.Image) -> None:
        """Configure Brightness.

        Args:
            image: Input image.

        Examples:
            ```python
            from blanket import Image, ImageEnhance

            image = Image.new("RGB", (8, 8), (40, 100, 180))
            enhancer = ImageEnhance.Brightness(image)
            result = enhancer.enhance(1.5)
            ```
        """
        self.image = image
        self.degenerate = Image.Image(enhance_brightness(image._native))


class Sharpness(_Enhance):
    """Adjust sharpness relative to Pillow's 3x3 smooth filter."""

    def __init__(self, image: Image.Image) -> None:
        """Configure Sharpness.

        Args:
            image: Input image.

        Examples:
            ```python
            from blanket import Image, ImageEnhance

            image = Image.new("RGB", (8, 8), (40, 100, 180))
            enhancer = ImageEnhance.Sharpness(image)
            result = enhancer.enhance(1.5)
            ```
        """
        self.image = image
        self.degenerate = Image.Image(enhance_sharpness(image._native))
        self.degenerate.info.update(image.info)
