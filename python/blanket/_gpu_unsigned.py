"""Unsigned resident processing using normalized float32 working samples."""

from __future__ import annotations

__lazy_modules__ = {"torch", "collections.abc"}

from typing import TYPE_CHECKING

from . import Image, _blanket

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch import Tensor

    from .ImageGpu import GpuImage, GpuReduction


class GpuUnsignedImage:
    """10/12/16-bit image; public results saturate and round half upward.

    Float32 working storage retains every unsigned input sample. Filters evaluate
    in float32 and quantize once per public operation, including all alpha bands.
    """

    def __init__(self, image: GpuImage, mode: str, depth: int) -> None:
        self._image = image
        self.mode = mode
        self.bit_depth = depth
        self.dtype = "uint16"

    @property
    def size(self) -> tuple[int, int]:
        return self._image.size

    @property
    def info(self) -> dict[str, object]:
        return self._image.info

    @property
    def runtime(self) -> str:
        return self._image.runtime

    @property
    def device(self) -> str:
        return self._image.device

    def _result(self, image: GpuImage, mode: str | None = None) -> GpuUnsignedImage:
        image = image._run(30, values=[float((1 << self.bit_depth) - 1)])
        return GpuUnsignedImage(image, self.mode if mode is None else mode, self.bit_depth)

    def _other(self, image: GpuUnsignedImage) -> GpuImage:
        if (image.mode, image.bit_depth) != (self.mode, self.bit_depth):
            raise ValueError("unsigned GPU images must match mode and depth")
        return image._image

    def to_image(self) -> Image.Image:
        working = self._image.to_image()
        data = _blanket._gpu_unsigned_bytes(working.tobytes(), self.bit_depth, True)
        result = Image.frombytes(self.mode, self.size, data, bit_depth=self.bit_depth)
        result.info.update(self.info)
        return result

    def synchronize(self) -> None:
        self._image.synchronize()

    def close(self) -> None:
        self._image.close()

    def copy(self) -> GpuUnsignedImage:
        return self._result(self._image._run(6))

    def resize(self, size: tuple[int, int], resample: int = Image.Resampling.BICUBIC, *, linear_light: bool = False) -> GpuUnsignedImage:
        return self._result(self._image.resize(size, resample, linear_light=linear_light))

    def crop(self, box: tuple[int, int, int, int]) -> GpuUnsignedImage:
        return self._result(self._image.crop(box))

    def transpose(self, method: int) -> GpuUnsignedImage:
        return self._result(self._image.transpose(method))

    def offset(self, x: int, y: int = 0) -> GpuUnsignedImage:
        return self._result(self._image.offset(x, y))

    def invert(self) -> GpuUnsignedImage:
        return self._result(self._image.invert())

    def brightness(self, factor: float) -> GpuUnsignedImage:
        return self._result(self._image.brightness(factor))

    def color(self, factor: float) -> GpuUnsignedImage:
        return self._result(self._image.color(factor))

    def contrast(self, factor: float) -> GpuUnsignedImage:
        return self._result(self._image.contrast(factor))

    def sharpness(self, factor: float) -> GpuUnsignedImage:
        return self._result(self._image.sharpness(factor))

    def box_blur(self, radius: float | tuple[float, float]) -> GpuUnsignedImage:
        return self._result(self._image.box_blur(radius))

    def gaussian_blur(self, radius: float | tuple[float, float]) -> GpuUnsignedImage:
        return self._result(self._image.gaussian_blur(radius))

    def rank_filter(self, size: int, rank: int) -> GpuUnsignedImage:
        return self._result(self._image.rank_filter(size, rank))

    def min_filter(self, size: int = 3) -> GpuUnsignedImage:
        return self.rank_filter(size, 0)

    def max_filter(self, size: int = 3) -> GpuUnsignedImage:
        return self.rank_filter(size, size * size - 1)

    def median_filter(self, size: int = 3) -> GpuUnsignedImage:
        return self.rank_filter(size, size * size // 2)

    def unsharp_mask(self, radius: float = 2, percent: int = 150, threshold: float = 3) -> GpuUnsignedImage:
        return self._result(self._image.unsharp_mask(radius, percent, threshold / ((1 << self.bit_depth) - 1)))

    def convolve(self, kernel: Sequence[float], *, scale: float | None = None, offset: float = 0) -> GpuUnsignedImage:
        return self._result(self._image.convolve(kernel, scale=scale, offset=offset / ((1 << self.bit_depth) - 1)))

    def blend(self, other: GpuUnsignedImage, alpha: float) -> GpuUnsignedImage:
        return self._result(self._image.blend(self._other(other), alpha))

    def alpha_composite(self, source: GpuUnsignedImage, *, linear_light: bool = False) -> GpuUnsignedImage:
        return self._result(self._image.alpha_composite(self._other(source), linear_light=linear_light))

    def paste(self, source: GpuUnsignedImage, box: tuple[int, int] = (0, 0), mask: GpuImage | None = None) -> GpuUnsignedImage:
        return self._result(self._image.paste(self._other(source), box, mask))

    def to_torch(self, *, layout: str = "HWC") -> Tensor:
        """Share normalized float32 working samples without downloading pixels."""
        return self._image.to_torch(layout=layout)

    def getchannel(self, channel: int | str) -> GpuUnsignedImage:
        return self._result(self._image.getchannel(channel), "L")

    def split(self) -> tuple[GpuUnsignedImage, ...]:
        return tuple(self._result(band, "L") for band in self._image.split())

    def transform(self, size: tuple[int, int], method: int, data: Sequence[float], resample: int = Image.Resampling.NEAREST) -> GpuUnsignedImage:
        return self._result(self._image.transform(size, method, data, resample))

    def rotate(self, angle: float, resample: int = Image.Resampling.NEAREST, *, expand: bool = False) -> GpuUnsignedImage:
        return self._result(self._image.rotate(angle, resample, expand=expand))

    def remap(self, coordinates: GpuImage, resample: int = Image.Resampling.NEAREST) -> GpuUnsignedImage:
        return self._result(self._image.remap(coordinates, resample))

    def statistics(self) -> GpuReduction:
        """Return resident statistics in normalized 0..1 working units."""
        return self._image.statistics()
