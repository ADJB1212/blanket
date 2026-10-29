"""Explicit resident GPU images; CPU transfers occur only at import/export."""

from __future__ import annotations

__lazy_modules__ = {"types", "torch", "collections.abc"}

import math
import struct
from functools import lru_cache
from operator import index
from typing import TYPE_CHECKING

from . import Image, _blanket

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import ModuleType
    from typing import Any

    from torch import Tensor

    from ._gpu_unsigned import GpuUnsignedImage
    from .ImageFilter import Color3DLUT

_MODES = {"L": (1, False), "LA": (2, False), "RGB": (3, False), "RGBA": (4, False), "F": (1, True), "RGB;F": (3, True), "RGBA;F": (4, True)}
_FORMATS = tuple(_MODES)
_OPERATIONS = (
    "resize",
    "convolve",
    "crop",
    "transpose",
    "offset",
    "brightness",
    "invert",
    "point",
    "solarize",
    "posterize",
    "difference",
    "multiply",
    "screen",
    "lighter",
    "darker",
    "add",
    "subtract",
    "blend",
    "alpha_composite",
    "convert",
    "grayscale",
    "getchannel",
    "split",
    "merge",
    "putalpha",
    "box_blur",
    "gaussian_blur",
    "unsharp_mask",
    "rank_filter",
    "min_filter",
    "max_filter",
    "median_filter",
    "color",
    "contrast",
    "sharpness",
    "color_lut",
    "linear_light_resize",
    "linear_light_composite",
    "transform",
    "rotate",
    "remap",
    "paste",
    "statistics",
    "histogram",
)


def _scalar(value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or abs(value) > 3.4028234663852886e38:
        raise ValueError("GPU scalar must be finite and representable as float32")
    return value


def capabilities(backend: str = "metal", *, device: int = 0) -> dict[str, object]:
    """Report backend support and driver diagnostics without importing PyTorch."""
    if backend not in ("metal", "cuda"):
        raise ValueError("unsupported GPU backend")
    compiled = hasattr(_blanket, "_gpu_weights") if backend == "metal" else hasattr(_blanket, "_gpu_cuda_fatbin")
    probe = getattr(_blanket, "_gpu_available", None)
    available = bool(probe and probe()) if backend == "metal" else False
    diagnostic = None
    if backend == "cuda":
        from . import _gpu_cuda

        try:
            _gpu_cuda.require(device)
            available = True
        except (RuntimeError, ValueError) as error:
            diagnostic = str(error)
    return {
        "compiled": compiled,
        "available": available,
        "backend": backend,
        "device": device,
        "diagnostic": diagnostic,
        "modes": tuple(_MODES),
        "operations": _OPERATIONS,
        "compiled_operations": _OPERATIONS if compiled else (),
        "unsigned_modes": ("L", "RGB", "RGBA"),
        "unsigned_depths": (10, 12, 16),
        "unsigned_operations": (
            "copy",
            "resize",
            "crop",
            "transpose",
            "offset",
            "invert",
            "brightness",
            "color",
            "contrast",
            "sharpness",
            "box_blur",
            "gaussian_blur",
            "rank_filter",
            "min_filter",
            "max_filter",
            "median_filter",
            "unsharp_mask",
            "convolve",
            "blend",
            "alpha_composite",
            "paste",
            "getchannel",
            "split",
            "transform",
            "rotate",
            "remap",
            "statistics",
        ),
        "histogram_modes": ("L", "LA", "RGB", "RGBA"),
        "torch_interop": "DLPack with CUDA stream ordering" if backend == "cuda" else "requires torch.mps.compile_shader",
    }


def _require(backend: str = "metal", device: int = 0) -> None:
    if backend == "cuda":
        from . import _gpu_cuda

        _gpu_cuda.require(device)
    elif not capabilities()["available"]:
        raise RuntimeError("Metal unavailable; build Blanket with the gpu-metal feature on macOS")


def _torch(backend: str = "metal") -> ModuleType:
    import torch

    if backend == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA interoperability requires a CUDA-enabled PyTorch installation and NVIDIA driver")
    elif not torch.backends.mps.is_available() or not hasattr(torch.mps, "compile_shader"):
        raise RuntimeError("PyTorch MPS with torch.mps.compile_shader is required")
    return torch


@lru_cache(maxsize=1)
def _library() -> object:
    return _torch().mps.compile_shader(_blanket._gpu_shader_source)


def _validate(size: tuple[int, int], mode: str) -> None:
    if mode not in _MODES:
        raise ValueError("unsupported GPU image mode")
    if len(size) != 2 or any(type(v) is not int or v <= 0 or v > 0xFFFFFFFF for v in size) or size[0] * size[1] > 178_956_970:
        raise ValueError("invalid GPU image dimensions")


def from_image(image: Image.Image, *, backend: str = "metal", runtime: str = "native", device: int = 0) -> GpuImage | GpuUnsignedImage:
    """Upload the current CPU frame, retaining its metadata."""
    if backend not in ("metal", "cuda") or runtime not in ("native", "torch"):
        raise ValueError("unsupported GPU backend or runtime")
    _require(backend, device)
    if backend == "metal" and device != 0:
        raise ValueError("Metal supports only the default device")
    _validate(image.size, image.mode)
    if image.mode in ("L", "RGB", "RGBA") and image.bit_depth in (10, 12, 16):
        from ._gpu_unsigned import GpuUnsignedImage

        mode = {"L": "F", "RGB": "RGB;F", "RGBA": "RGBA;F"}[image.mode]
        data = _blanket._gpu_unsigned_bytes(image.tobytes(), image.bit_depth, False)
        working = Image.frombytes(mode, image.size, data)
        working.info.update(image._metadata_info())
        resident = from_image(working, backend=backend, runtime=runtime, device=device)
        if not isinstance(resident, GpuImage):
            raise RuntimeError("invalid unsigned GPU working storage")
        return GpuUnsignedImage(resident, image.mode, image.bit_depth)
    if image.mode not in ("F", "RGB;F", "RGBA;F") and image.bit_depth != 8:
        raise ValueError("unsupported GPU sample depth")
    channels, floating = _MODES[image.mode]
    data = image.tobytes()
    if backend == "cuda":
        from . import _gpu_cuda

        resource = _gpu_cuda.upload(data, (image.height, image.width, channels), floating, device)
        runtime = "cuda"
    elif runtime == "native":
        resource = _blanket.MetalImage(*image.size, channels, floating, data)
    else:
        torch = _torch()
        resource = torch.frombuffer(bytearray(data), dtype=torch.float32 if floating else torch.uint8).reshape(image.height, image.width, channels).to("mps")
    return GpuImage(resource, image.size, image.mode, runtime, dict(image._metadata_info()))


def from_torch(tensor: Tensor, *, layout: str, mode: str | None = None, copy: bool = False, bit_depth: int | None = None) -> GpuImage | GpuUnsignedImage:
    """Borrow an MPS/CUDA tensor. Views are packed on the GPU before processing."""
    backend = "cuda" if getattr(getattr(tensor, "device", None), "type", None) == "cuda" else "metal"
    device_type = "cuda" if backend == "cuda" else "mps"
    _require(backend, tensor.device.index if backend == "cuda" else 0)
    torch = _torch(backend)
    if bit_depth is not None:
        from ._gpu_unsigned import GpuUnsignedImage

        if bit_depth not in (10, 12, 16) or not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.uint16 or tensor.device.type != device_type:
            raise ValueError("unsigned import requires an MPS/CUDA uint16 tensor and depth 10, 12, or 16")
        if mode not in (None, "L", "RGB", "RGBA"):
            raise ValueError("unsupported unsigned GPU mode")
        working_mode = None if mode is None else {"L": "F", "RGB": "RGB;F", "RGBA": "RGBA;F"}[mode]
        resident = from_torch(tensor.to(torch.float32) / ((1 << bit_depth) - 1), layout=layout, mode=working_mode)
        if not isinstance(resident, GpuImage):
            raise RuntimeError("invalid unsigned tensor working storage")
        return GpuUnsignedImage(resident._run(30, values=[float((1 << bit_depth) - 1)]), {"F": "L", "RGB;F": "RGB", "RGBA;F": "RGBA"}[resident.mode], bit_depth)
    if not isinstance(tensor, torch.Tensor) or tensor.device.type != device_type or tensor.dtype not in (torch.uint8, torch.float32) or tensor.requires_grad:
        raise ValueError("requires an MPS/CUDA uint8/float32 tensor without gradients")
    if layout == "HW" and tensor.ndim == 2:
        tensor = tensor.unsqueeze(-1)
    elif layout == "CHW" and tensor.ndim == 3:
        tensor = tensor.permute(1, 2, 0)
    elif layout != "HWC" or tensor.ndim != 3:
        raise ValueError("layout must match HW, HWC, or CHW tensor dimensions")
    height, width, channels = tensor.shape
    floating = tensor.dtype == torch.float32
    if mode is None:
        mode = ({1: "F", 3: "RGB;F", 4: "RGBA;F"} if floating else {1: "L", 2: "LA", 3: "RGB", 4: "RGBA"}).get(channels)
    if mode not in _MODES or _MODES[mode] != (channels, floating):
        raise ValueError("tensor channels/dtype do not match mode")
    _validate((width, height), mode)
    if backend == "cuda":
        from . import _gpu_cuda

        return GpuImage(_gpu_cuda.from_torch(tensor, copy=copy), (width, height), mode, "cuda", {})
    return GpuImage(tensor.clone() if copy else tensor, (width, height), mode, "torch", {})


def merge(mode: str, bands: Sequence[GpuImage]) -> GpuImage:
    """Merge independent resident single-channel images into a supported mode."""
    if mode not in _MODES or len(bands) != _MODES[mode][0]:
        raise ValueError("mode and number of bands do not match")
    first = bands[0]
    first._check()
    floating = _MODES[mode][1]
    for band in bands:
        band._check()
        if _MODES[band.mode] != (1, floating) or (band.size, band.runtime) != (first.size, first.runtime):
            raise ValueError("bands must match in size, dtype, and execution context")
    result = first.convert(mode)
    for channel, band in enumerate(bands):
        result = result._run(16, other=band, k=channel)
    return result


class GpuImage:
    """Resident image with independent operation results and explicit download."""

    def __init__(self, resource: object, size: tuple[int, int], mode: str, runtime: str, info: dict[str, Any]) -> None:
        self._resource: Any = resource
        self.size = size
        self.mode = mode
        self.runtime = runtime
        self.info = info
        self.device = f"cuda:{self._resource.device}" if runtime == "cuda" else "metal" if runtime == "native" else "mps"
        self.dtype = "float32" if _MODES[mode][1] else "uint8"

    def _check(self) -> None:
        if self._resource is None:
            raise ValueError("GPU image is closed")

    def _run(self, op: int, *, size: tuple[int, int] | None = None, mode: str | None = None, k: int = 0, axis: int = 0, values: list[float] | None = None, other: GpuImage | None = None) -> GpuImage:
        self._check()
        other = self if other is None else other
        other._check()
        compatible_mode = self.mode == other.mode or (op == 16 and _MODES[other.mode] == (1, _MODES[self.mode][1]))
        resident_mean = op == 22 and k == 1 and other.mode == "RGBA;F" and other.size == (1, 1)
        coordinate_map = op == 27 and other.mode == "RGB;F" and other.size == size
        mask_band = op == 28 and other.mode == "F"
        if (
            not (compatible_mode or resident_mean or coordinate_map or mask_band)
            or not (self.size == other.size or resident_mean or coordinate_map)
            or self.runtime != other.runtime
            or self.device != other.device
        ):
            raise ValueError("GPU images must have matching size, mode, and execution context")
        size = self.size if size is None else size
        mode = self.mode if mode is None else mode
        _validate(size, mode)
        channels, floating = _MODES[mode]
        parameters = [*self.size, _MODES[self.mode][0], *size, op, k, axis]
        values = [0.0] if values is None else values
        if self.runtime == "cuda":
            resource = self._resource.run(other._resource, parameters, values, channels, floating, _MODES[self.mode][1])
        elif self.runtime == "native":
            resource = self._resource.run(other._resource, parameters, values)
        else:
            torch = _torch()
            src = self._resource.contiguous()
            rhs = other._resource.contiguous()
            resource = torch.empty((size[1], size[0], channels), dtype=torch.float32 if floating else torch.uint8, device=src.device)
            weights = torch.tensor(values, dtype=torch.float32, device="mps")
            kernel_name = "image_f32" if floating else "image_u8"
            if op == 29:
                kernel_name = "image_f32" if _MODES[self.mode][1] else "image_u8"
            if op == 14:
                kernel_name = f"convert_{'f32' if _MODES[self.mode][1] else 'u8'}_{'f32' if floating else 'u8'}"
            elif op == 21:
                kernel_name = "reduce_f32" if _MODES[self.mode][1] else "reduce_u8"
            kernel = getattr(_library(), kernel_name)
            kernel(src, resource, rhs, parameters, weights, threads=size[0] * size[1] * channels, arg_casts={3: "int32"})
        return GpuImage(resource, size, mode, self.runtime, self.info.copy())

    def convert(self, mode: str) -> GpuImage:
        """Convert resident modes; F uses raw 0..255 units, color floats use 0..1."""
        self._check()
        _validate(self.size, mode)
        return self._run(6) if mode == self.mode else self._run(14, mode=mode, k=_FORMATS.index(mode))

    def statistics(self) -> GpuReduction:
        """Retain per-channel extrema and raw moments of orders 1..4 on the GPU."""
        return GpuReduction(self._run(29, size=(7, _MODES[self.mode][0]), mode="F"), "statistics")

    def histogram(self) -> GpuReduction:
        """Retain exact 256-bin uint8 channel histograms on the GPU."""
        if _MODES[self.mode][1]:
            raise ValueError("GPU histograms currently require uint8 samples")
        return GpuReduction(self._run(29, size=(256, _MODES[self.mode][0]), mode="F", k=1), "histogram")

    def transform(self, size: tuple[int, int], method: int, data: Sequence[float], resample: int = Image.Resampling.NEAREST, *, fillcolor: float | Sequence[float] = 0) -> GpuImage:
        """Apply inverse affine or perspective mapping with edge-clamped sampling."""
        self._check()
        if method not in (Image.Transform.AFFINE, Image.Transform.PERSPECTIVE):
            raise ValueError("GPU transform supports AFFINE and PERSPECTIVE")
        coefficients = [_scalar(v) for v in data]
        expected = 6 if method == Image.Transform.AFFINE else 8
        if len(coefficients) != expected:
            raise ValueError("incorrect transform coefficient count")
        if expected == 6:
            coefficients += [0.0, 0.0]
        if resample not in (Image.Resampling.NEAREST, Image.Resampling.BILINEAR, Image.Resampling.BICUBIC):
            raise ValueError("transform supports NEAREST, BILINEAR, and BICUBIC")
        channels = _MODES[self.mode][0]
        fill = [fillcolor] * channels if isinstance(fillcolor, (int, float)) else list(fillcolor)
        if len(fill) != channels:
            raise ValueError("fillcolor must match image channels")
        controls = coefficients + [_scalar(v) for v in fill]
        sampling = {Image.Resampling.NEAREST: 0, Image.Resampling.BILINEAR: 1, Image.Resampling.BICUBIC: 2}[resample]
        return self._run(26, size=size, k=sampling, values=controls)

    def rotate(self, angle: float, resample: int = Image.Resampling.NEAREST, *, expand: bool = False, fillcolor: float | Sequence[float] = 0) -> GpuImage:
        """Rotate counterclockwise about the image center."""
        angle = _scalar(angle) % 360
        width, height = self.size
        radians = -math.radians(angle)
        a, b = round(math.cos(radians), 15), round(math.sin(radians), 15)
        d, e = -b, a
        size = self.size
        if expand:
            cx = width / 2 - a * width / 2 - b * height / 2
            cy = height / 2 - d * width / 2 - e * height / 2
            corners = [(a * x + b * y + cx, d * x + e * y + cy) for x, y in ((0, 0), (width, 0), (0, height), (width, height))]
            size = (math.ceil(max(x for x, _ in corners)) - math.floor(min(x for x, _ in corners)), math.ceil(max(y for _, y in corners)) - math.floor(min(y for _, y in corners)))
        c = width / 2 - a * size[0] / 2 - b * size[1] / 2
        f = height / 2 - d * size[0] / 2 - e * size[1] / 2
        return self.transform(size, Image.Transform.AFFINE, (a, b, c, d, e, f), resample, fillcolor=fillcolor)

    def remap(self, coordinates: GpuImage, resample: int = Image.Resampling.NEAREST, *, fillcolor: float | Sequence[float] = 0) -> GpuImage:
        """Reuse an RGB;F coordinate map: red=x, green=y; blue is ignored.

        Coordinates address pixel centers at (x + 0.5, y + 0.5).
        """
        coordinates._check()
        if coordinates.mode != "RGB;F":
            raise ValueError("coordinate maps require RGB;F mode")
        if resample not in (Image.Resampling.NEAREST, Image.Resampling.BILINEAR, Image.Resampling.BICUBIC):
            raise ValueError("remap supports NEAREST, BILINEAR, and BICUBIC")
        channels = _MODES[self.mode][0]
        fill = [fillcolor] * channels if isinstance(fillcolor, (int, float)) else list(fillcolor)
        if len(fill) != channels:
            raise ValueError("fillcolor must match image channels")
        sampling = {Image.Resampling.NEAREST: 0, Image.Resampling.BILINEAR: 1, Image.Resampling.BICUBIC: 2}[resample]
        return self._run(27, size=coordinates.size, other=coordinates, k=sampling, values=[0.0] * 8 + [_scalar(v) for v in fill])

    def paste(self, source: GpuImage, box: tuple[int, int] = (0, 0), mask: GpuImage | None = None) -> GpuImage:
        """Return an independent image with source pasted through an L/F mask.

        L masks use 0..255 coverage; F masks use 0..1, with NaN treated as zero.
        All channels, including alpha, are interpolated independently.
        """
        self._check()
        source._check()
        if self.mode != source.mode or self.runtime != source.runtime:
            raise ValueError("paste source must match mode and execution context")
        if len(box) != 2:
            raise ValueError("paste box requires an (x, y) pair")
        x, y = (index(value) for value in box)
        working_mode = "F" if self.mode in ("L", "F") else "RGB;F" if self.mode in ("RGB", "RGB;F") else "RGBA;F"
        background, foreground = self.convert(working_mode), source.convert(working_mode)
        if mask is None:
            # The alpha band of an opaque resident conversion supplies full coverage.
            coverage = source.convert("RGBA;F").putalpha(1.0).getchannel("A")
            maximum = 1.0
        else:
            mask._check()
            if mask.mode not in ("L", "F") or mask.size != source.size or mask.runtime != source.runtime:
                raise ValueError("paste mask must be L/F and match source size and execution context")
            maximum = 255.0 if mask.mode == "L" else 1.0
            coverage = mask.convert("F")
        coefficients = (1, 0, -x, 0, 1, -y)
        foreground = foreground.transform(self.size, Image.Transform.AFFINE, coefficients)
        coverage = coverage.transform(self.size, Image.Transform.AFFINE, coefficients)
        foreground = foreground._run(28, other=coverage, values=[maximum])
        background = background._run(28, other=coverage, values=[maximum], axis=1)
        return background.add(foreground).convert(self.mode)

    def grayscale(self) -> GpuImage:
        """Return L for uint8 or raw-sample F for float32, discarding alpha."""
        return self.convert("F" if _MODES[self.mode][1] else "L")

    def getchannel(self, channel: int | str) -> GpuImage:
        """Extract an independent resident channel without rescaling samples."""
        self._check()
        if isinstance(channel, str):
            try:
                channel = Image._BANDS[self.mode].index(channel)
            except ValueError as error:
                raise ValueError("unknown image band") from error
        channel = index(channel)
        if not 0 <= channel < _MODES[self.mode][0]:
            raise ValueError("band index out of range")
        return self._run(15, mode="F" if _MODES[self.mode][1] else "L", k=channel)

    def split(self) -> tuple[GpuImage, ...]:
        """Return independent resident copies of all channels."""
        self._check()
        return tuple(self.getchannel(channel) for channel in range(_MODES[self.mode][0]))

    def putalpha(self, alpha: GpuImage | float) -> GpuImage:
        """Return a copy with replaced alpha; float grayscale promotes to RGBA;F."""
        self._check()
        if _MODES[self.mode][1]:
            mode = "RGBA;F"
        else:
            mode = "LA" if _MODES[self.mode][0] <= 2 else "RGBA"
        result = self.convert(mode)
        if isinstance(alpha, GpuImage):
            return result._run(16, other=alpha, k=_MODES[mode][0] - 1)
        value = min(1.0, max(0.0, _scalar(alpha))) if _MODES[mode][1] else float(min(255, max(0, index(alpha))))
        return result._run(17, values=[value])

    def _blur(self, radius: float | tuple[float, float], gaussian: bool) -> GpuImage:
        self._check()
        radii = radius if isinstance(radius, tuple) else (radius, radius)
        if len(radii) != 2:
            raise ValueError("blur radius must contain two values")
        controls = [_blanket._gpu_blur_parameters(_scalar(value), gaussian) for value in radii]
        result = self
        for axis, (integer, weights) in enumerate(controls):
            if radii[axis] == 0:
                continue
            for _ in range(3 if gaussian else 1):
                result = result._run(18, k=integer, axis=axis, values=list(weights))
        return self._run(6) if result is self else result

    def box_blur(self, radius: float | tuple[float, float] = 2.0) -> GpuImage:
        """Blur with a fractional box kernel and replicated edges, including alpha."""
        return self._blur(radius, False)

    def gaussian_blur(self, radius: float | tuple[float, float] = 2.0) -> GpuImage:
        """Approximate Gaussian blur with three fractional box passes per axis."""
        return self._blur(radius, True)

    def unsharp_mask(self, radius: float = 2.0, percent: int = 150, threshold: float = 3.0) -> GpuImage:
        """Sharpen against Gaussian blur; threshold is measured in sample units."""
        self._check()
        percent = index(percent)
        threshold = _scalar(threshold)
        if not -2147483648 <= percent <= 2147483647:
            raise ValueError("percent must fit a signed 32-bit integer")
        _blanket._gpu_blur_parameters(_scalar(radius), True)
        if percent == 0 or radius == 0 or (not _MODES[self.mode][1] and threshold >= 255):
            return self._run(6)
        return self._run(20, other=self.gaussian_blur(radius), k=percent & 0xFFFFFFFF, values=[threshold])

    def rank_filter(self, size: int, rank: int) -> GpuImage:
        """Select a rank from an odd 1x1 through 7x7 replicated-edge window."""
        size, rank = index(size), index(rank)
        if size not in (1, 3, 5, 7) or not 0 <= rank < size * size:
            raise ValueError("rank filters require an odd size from 1 to 7 and a valid rank")
        return self._run(19, k=size, axis=rank)

    def min_filter(self, size: int = 3) -> GpuImage:
        """Apply per-channel erosion with replicated edges; float NaNs propagate."""
        return self.rank_filter(size, 0)

    def max_filter(self, size: int = 3) -> GpuImage:
        """Apply per-channel dilation with replicated edges; float NaNs propagate."""
        size = index(size)
        return self.rank_filter(size, size * size - 1)

    def median_filter(self, size: int = 3) -> GpuImage:
        """Select each channel's median, including alpha; float NaNs propagate."""
        size = index(size)
        return self.rank_filter(size, size * size // 2)

    def _float_color(self) -> GpuImage:
        if self.mode not in ("RGB", "RGBA", "RGB;F", "RGBA;F"):
            raise ValueError("linear-light processing requires RGB or RGBA color")
        return self.convert("RGBA;F" if _MODES[self.mode][0] == 4 else "RGB;F")

    def resize(self, size: tuple[int, int], resample: int = Image.Resampling.BICUBIC, *, linear_light: bool = False, transfer: str = "srgb", premultiply_alpha: bool = True) -> GpuImage:
        """Resize with optional linear-light processing and float alpha selection."""
        self._check()
        _validate(size, self.mode)
        if resample not in range(6):
            raise ValueError("unsupported resampling filter")
        if transfer not in ("srgb", "linear"):
            raise ValueError("transfer must be 'srgb' or 'linear'")
        if linear_light:
            working = self._float_color()
            if transfer == "srgb":
                working = working._run(23, k=0)
            result = working.resize(size, resample, premultiply_alpha=premultiply_alpha)
            if transfer == "srgb":
                result = result._run(23, k=1)
            return result.convert(self.mode)
        if size == self.size and not _MODES[self.mode][1]:
            return self._run(6)
        floating = _MODES[self.mode][1]
        has_alpha = _MODES[self.mode][0] in (2, 4)
        alpha = has_alpha and (resample != 0 or floating) and (premultiply_alpha or not floating)
        clamp_alpha = has_alpha and floating and not premultiply_alpha
        result = self._run(4) if alpha else (self._run(24) if clamp_alpha else self)
        axes = (1, 0) if not _MODES[self.mode][1] and self.size[1] > self.size[0] * 100 and size[1] < self.size[1] else (0, 1)
        for axis in axes:
            target = (size[0], result.size[1]) if axis == 0 else (result.size[0], size[1])
            if result.size[axis] != target[axis]:
                stride, weights = _blanket._gpu_weights(result.size[axis], target[axis], resample)
                result = result._run(0, size=target, k=stride, axis=axis, values=weights)
        if result is self:
            return self._run(6)
        return result._run(5) if alpha else (result._run(24) if clamp_alpha else result)

    def convolve(self, kernel: list[float], *, size: int = 3, scale: float | None = None, offset: float = 0.0) -> GpuImage:
        """Apply a 3x3 or 5x5 kernel, retaining border samples."""
        if size not in (3, 5) or len(kernel) != size * size:
            raise ValueError("kernel must be 3x3 or 5x5")
        scale = (sum(kernel) or 1.0) if scale is None else scale
        if scale == 0 or not all(math.isfinite(v) for v in [*kernel, scale, offset]):
            raise ValueError("kernel parameters must be finite with nonzero scale")
        return self._run(1, k=size, values=[0.0, float(offset), *[float(v / scale) for v in kernel]])

    def crop(self, box: tuple[int, int, int, int]) -> GpuImage:
        """Crop integer bounds, filling pixels outside the source with zero."""
        self._check()
        if len(box) != 4:
            raise ValueError("crop box must contain four integers")
        left, top, right, bottom = (index(v) for v in box)
        if not all(-2147483648 <= v <= 2147483647 for v in (left, top, right, bottom)):
            raise ValueError("crop bounds must fit signed 32-bit integers")
        return self._run(7, size=(right - left, bottom - top), k=left & 0xFFFFFFFF, axis=top & 0xFFFFFFFF)

    def transpose(self, method: int) -> GpuImage:
        """Flip, rotate, or transpose using Image.Transpose identifiers."""
        self._check()
        method = index(method)
        if method not in range(7):
            raise ValueError("unsupported transpose method")
        size = self.size[::-1] if method in (2, 4, 5, 6) else self.size
        return self._run(8, size=size, k=method)

    def offset(self, xoffset: int, yoffset: int | None = None) -> GpuImage:
        """Translate with wraparound, matching ImageChops.offset."""
        self._check()
        xoffset = index(xoffset)
        yoffset = xoffset if yoffset is None else index(yoffset)
        return self._run(9, k=xoffset % self.size[0], axis=yoffset % self.size[1])

    def brightness(self, factor: float) -> GpuImage:
        """Scale channels, preserving RGBA alpha; float colors are not clipped."""
        return self._run(10, values=[_scalar(factor)])

    def color(self, factor: float) -> GpuImage:
        """Adjust saturation around luminance, preserving alpha and sample range."""
        self._check()
        factor = _scalar(factor)
        if factor == 1 or _MODES[self.mode][0] < 3:
            return self._run(6)
        return self._run(22, k=0, values=[factor])

    def contrast(self, factor: float) -> GpuImage:
        """Adjust contrast around mean luminance computed entirely on the GPU."""
        self._check()
        factor = _scalar(factor)
        if factor == 1:
            return self._run(6)
        floating = _MODES[self.mode][1]
        count = self.size[0] * self.size[1]
        partials = self._run(21, mode="RGBA;F", size=((count + 255) // 256, 1), k=int(floating))
        while partials.size != (1, 1):
            partials = partials._run(21, size=((partials.size[0] + 255) // 256, 1), k=3 if floating else 2)
        return self._run(22, other=partials, k=1, axis=count, values=[factor])

    def sharpness(self, factor: float) -> GpuImage:
        """Enhance against the CPU's 3x3 smooth kernel, retaining borders and RGBA alpha."""
        self._check()
        factor = _scalar(factor)
        if factor == 1 or min(self.size) < 3:
            return self._run(6)
        return self._run(22, k=2, values=[factor])

    def color_lut(self, lut: Color3DLUT) -> GpuImage:
        """Apply a Color3DLUT on resident RGB/RGBA pixels, retaining sample precision."""
        from .ImageFilter import Color3DLUT

        self._check()
        if not isinstance(lut, Color3DLUT):
            raise TypeError("color_lut requires ImageFilter.Color3DLUT")
        mode = self.mode if lut.mode is None else lut.mode
        _validate(self.size, mode)
        channels, floating = _MODES[self.mode]
        output_channels, output_float = _MODES[mode]
        table_channels = index(lut.channels)
        if channels < 3 or floating != output_float or output_channels < table_channels or (output_channels > table_channels and output_channels > channels):
            raise ValueError("3D LUT requires compatible RGB/RGBA input and output modes with matching dtype")
        controls = _blanket._gpu_lut_controls(lut.size, table_channels, list(lut.table), floating)
        return self._run(25, mode=mode, k=table_channels, axis=_FORMATS.index(mode), values=controls)

    def invert(self) -> GpuImage:
        """Invert every channel, including alpha, around 255 or float 1."""
        return self._run(11)

    def point(self, lut: list[float]) -> GpuImage:
        """Map uint8 channels through a shared or per-channel 256-entry LUT."""
        self._check()
        channels, floating = _MODES[self.mode]
        if floating:
            raise ValueError("point requires uint8 samples")
        if len(lut) not in (256, 256 * channels):
            raise ValueError("lookup table requires 256 values per channel")
        values = [float(max(0, min(255, round(_scalar(v))))) for v in lut]
        if len(values) == 256:
            values *= channels
        return self._run(12, values=values)

    def solarize(self, threshold: int = 128) -> GpuImage:
        """Invert uint8 channel values at or above the threshold, including alpha."""
        threshold = index(threshold)
        return self.point([v if v < threshold else 255 - v for v in range(256)])

    def posterize(self, bits: int) -> GpuImage:
        """Retain the requested high bits of every uint8 channel, including alpha."""
        bits = index(bits)
        if bits not in range(9):
            raise ValueError("bits must be between 0 and 8")
        mask = (255 << (8 - bits)) & 255
        return self.point([v & mask for v in range(256)])

    def _binary(self, other: GpuImage, operation: int, scale: float = 1.0, offset: float = 0.0) -> GpuImage:
        scale = _scalar(scale)
        if abs(scale) < 1.1754943508222875e-38:
            raise ValueError("scale must be a nonzero normal float32 value")
        return self._run(13, other=other, k=operation, values=[scale, _scalar(offset)])

    def difference(self, other: GpuImage) -> GpuImage:
        """Return the absolute difference in each channel, including alpha."""
        return self._binary(other, 0)

    def multiply(self, other: GpuImage) -> GpuImage:
        """Multiply channels, normalized by 255 for uint8 and 1 for float32."""
        return self._binary(other, 1)

    def screen(self, other: GpuImage) -> GpuImage:
        """Screen channels using 255 for uint8 and 1 for float32 as white."""
        return self._binary(other, 2)

    def lighter(self, other: GpuImage) -> GpuImage:
        """Return per-channel maxima; float NaNs propagate."""
        return self._binary(other, 3)

    def darker(self, other: GpuImage) -> GpuImage:
        """Return per-channel minima; float NaNs propagate."""
        return self._binary(other, 4)

    def add(self, other: GpuImage, scale: float = 1.0, offset: float = 0.0) -> GpuImage:
        """Add channels, divide by scale, add offset, and clip only uint8."""
        return self._binary(other, 5, scale, offset)

    def subtract(self, other: GpuImage, scale: float = 1.0, offset: float = 0.0) -> GpuImage:
        """Subtract channels, divide by scale, add offset, and clip only uint8."""
        return self._binary(other, 6, scale, offset)

    def blend(self, other: GpuImage, alpha: float) -> GpuImage:
        """Blend matching images using a finite scalar weight."""
        if not math.isfinite(alpha):
            raise ValueError("blend alpha must be finite")
        return self._run(2, other=other, values=[float(alpha)])

    def alpha_composite(self, source: GpuImage, *, linear_light: bool = False, transfer: str = "srgb") -> GpuImage:
        """Composite source over this image with optional linear-light processing."""
        self._check()
        source._check()
        if transfer not in ("srgb", "linear"):
            raise ValueError("transfer must be 'srgb' or 'linear'")
        if _MODES[self.mode][0] not in (2, 4):
            raise ValueError("source-over requires alpha")
        if (self.size, self.mode, self.runtime) != (source.size, source.mode, source.runtime):
            raise ValueError("GPU images must have matching size, mode, and execution context")
        if linear_light:
            background, foreground = self._float_color(), source._float_color()
            if transfer == "srgb":
                background, foreground = background._run(23, k=0), foreground._run(23, k=0)
            result = background._run(3, other=foreground)
            if transfer == "srgb":
                result = result._run(23, k=1)
            return result.convert(self.mode)
        return self._run(3, other=source)

    def synchronize(self) -> None:
        """Wait for pending execution and surface GPU errors."""
        self._check()
        if self.runtime in ("native", "cuda"):
            self._resource.synchronize()
        else:
            _torch().mps.synchronize()

    def to_image(self) -> Image.Image:
        """Download an independent CPU image."""
        self.synchronize()
        data = self._resource.download() if self.runtime in ("native", "cuda") else bytes(self._resource.contiguous().view(_torch().uint8).cpu().flatten().tolist())
        result = Image.frombytes(self.mode, self.size, data)
        result.info.update(self.info)
        return result

    def to_torch(self, *, layout: str = "HWC") -> Tensor:
        """Share a tensor-backed result without downloading pixels."""
        self._check()
        if self.runtime == "cuda":
            resource = self._resource.to_torch()
            if layout == "HWC":
                return resource
            if layout == "CHW":
                return resource.permute(2, 0, 1)
            if layout == "HW" and _MODES[self.mode][0] == 1:
                return resource.squeeze(-1)
            raise ValueError("invalid export layout")
        if self.runtime != "torch":
            raise ValueError("native Metal export unsupported; upload with runtime='torch'")
        if layout == "HWC":
            return self._resource
        if layout == "CHW":
            return self._resource.permute(2, 0, 1)
        if layout == "HW" and _MODES[self.mode][0] == 1:
            return self._resource.squeeze(-1)
        raise ValueError("invalid export layout")

    def close(self) -> None:
        """Release this handle; submitted work retains its resources."""
        self._resource = None


class GpuReduction:
    """Small resident result; to_cpu explicitly synchronizes and downloads."""

    def __init__(self, image: GpuImage, kind: str) -> None:
        self._image = image
        self.kind = kind

    def synchronize(self) -> None:
        self._image.synchronize()

    def close(self) -> None:
        self._image.close()

    def to_cpu(self) -> list[list[int]] | list[dict[str, object]]:
        data = self._image.to_image().tobytes()
        if self.kind == "histogram":
            counts = struct.unpack(f"{len(data) // 4}I", data)
            return [list(counts[start : start + 256]) for start in range(0, len(counts), 256)]
        result: list[dict[str, object]] = []
        for low, high, total, squared, cubed, fourth, count in struct.iter_unpack("6fI", data):
            mean = total / count
            variance = squared / count - mean * mean
            variance = variance if math.isnan(variance) else max(0.0, variance)
            result.append(
                {
                    "count": count,
                    "extrema": (low, high),
                    "sum": total,
                    "sum2": squared,
                    "moments": (mean, squared / count, cubed / count, fourth / count),
                    "mean": mean,
                    "variance": variance,
                    "stddev": math.sqrt(variance),
                }
            )
        return result
