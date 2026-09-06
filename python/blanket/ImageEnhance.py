"""Pillow-compatible image enhancement classes backed by native kernels."""

# ruff: noqa: N999 -- Preserve Pillow's public submodule spelling.

from __future__ import annotations

from . import Image
from ._blanket import enhance_blend, enhance_brightness, enhance_color, enhance_contrast, enhance_sharpness

__all__ = ["Color", "Contrast", "Brightness", "Sharpness"]


class _Enhance:
    image: Image.Image
    degenerate: Image.Image

    def enhance(self, factor: float) -> Image.Image:
        """Return an enhanced copy, interpolated by an unrestricted factor."""

        result = Image.Image(enhance_blend(self.degenerate._native, self.image._native, factor))
        result.info.update(self.degenerate.info)
        return result


class Color(_Enhance):
    """Adjust image color balance."""

    def __init__(self, image: Image.Image) -> None:
        self.image = image
        self.intermediate_mode = "LA" if image.mode == "RGBA" else "L"
        self.degenerate = image if image.mode == "L" else Image.Image(enhance_color(image._native))
        if self.degenerate is not image:
            self.degenerate.info.update(image.info)


class Contrast(_Enhance):
    """Adjust image contrast around the rounded mean luminance."""

    def __init__(self, image: Image.Image) -> None:
        self.image = image
        self.degenerate = Image.Image(enhance_contrast(image._native))


class Brightness(_Enhance):
    """Adjust image brightness relative to black."""

    def __init__(self, image: Image.Image) -> None:
        self.image = image
        self.degenerate = Image.Image(enhance_brightness(image._native))


class Sharpness(_Enhance):
    """Adjust sharpness relative to Pillow's 3x3 smooth filter."""

    def __init__(self, image: Image.Image) -> None:
        self.image = image
        self.degenerate = Image.Image(enhance_sharpness(image._native))
        self.degenerate.info.update(image.info)
