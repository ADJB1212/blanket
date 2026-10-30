"""Explicit resident GPU images; CPU transfers occur only at import/export."""

from __future__ import annotations

__lazy_modules__ = {"types", "torch", "collections.abc", "concurrent.futures"}

import math
import struct
from functools import lru_cache
from operator import index
from typing import TYPE_CHECKING

from . import Image, _blanket
from ._gpu_async import Event, Stream, submit
from ._gpu_plan import _OPERATIONS as _PLAN_OPERATIONS, ProcessingPlan, cache_info, configure_cache

if TYPE_CHECKING:
    from collections.abc import Sequence
    from concurrent.futures import Future
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
    """Convert a control value to a finite number representable as float32."""
    value = float(value)
    if not math.isfinite(value) or abs(value) > 3.4028234663852886e38:
        raise ValueError("GPU scalar must be finite and representable as float32")
    return value


def capabilities(backend: str = "metal", *, device: int = 0) -> dict[str, object]:
    """Describe compiled features and availability for a GPU backend.

    Args:
        backend: "metal" or "cuda"; defaults to Metal.
        device: CUDA device index. Metal uses device 0.

    Returns:
        A dictionary containing compiled/available flags, diagnostics, supported
        modes, operations, sample depths, plans, and tensor interoperability.
        CUDA probing can import CuPy and initialize its driver; PyTorch is
        not imported. An unavailable CUDA backend is reported in diagnostic.

    Raises:
        ValueError: The backend name is unsupported.
    """
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
        "plan_runtimes": (("native",) if hasattr(_blanket, "MetalPlan") else ()) if backend == "metal" else (("cuda",) if compiled else ()),
        "plan_operations": tuple(sorted(_PLAN_OPERATIONS)) if (backend == "metal" and hasattr(_blanket, "MetalPlan")) or (backend == "cuda" and compiled) else (),
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
    """Raise RuntimeError unless the selected backend and device are usable."""
    if backend == "cuda":
        from . import _gpu_cuda

        _gpu_cuda.require(device)
    elif not capabilities()["available"]:
        raise RuntimeError("Metal unavailable; build Blanket with the gpu-metal feature on macOS")


def _torch(backend: str = "metal") -> ModuleType:
    """Import PyTorch and require CUDA or MPS shader interoperability."""
    import torch

    if backend == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA interoperability requires a CUDA-enabled PyTorch installation and NVIDIA driver")
    elif not torch.backends.mps.is_available() or not hasattr(torch.mps, "compile_shader"):
        raise RuntimeError("PyTorch MPS with torch.mps.compile_shader is required")
    return torch


@lru_cache(maxsize=1)
def _library() -> object:
    """Return the cached PyTorch MPS shader library."""
    return _torch().mps.compile_shader(_blanket._gpu_shader_source)


def _validate(size: tuple[int, int], mode: str) -> None:
    """Validate the mode, positive integer dimensions, and decoded-pixel limit."""
    if mode not in _MODES:
        raise ValueError("unsupported GPU image mode")
    if len(size) != 2 or any(type(v) is not int or v <= 0 or v > 0xFFFFFFFF for v in size) or size[0] * size[1] > 178_956_970:
        raise ValueError("invalid GPU image dimensions")


def _dispatch_threads(parameters: list[int], channels: int, floating: bool) -> int:
    """Return the grid size, accounting for byte blur's sliding windows."""
    if parameters[5] == 18 and not floating and parameters[6] >= 32:
        length, lines = parameters[:2] if parameters[7] == 0 else parameters[1::-1]
        return ((length + 15) // 16) * lines * channels
    return parameters[3] * parameters[4] * channels


def from_image(image: Image.Image, *, backend: str = "metal", runtime: str = "native", device: int = 0, unified_memory: bool = False) -> GpuImage | GpuUnsignedImage:
    """Upload the current CPU frame and copy its metadata.

    Args:
        image: An open Image in L, LA, RGB, RGBA, F, RGB;F, or RGBA;F mode.
            Byte modes require depth 8, except L/RGB/RGBA also allow 10/12/16.
        backend: "metal" or "cuda".
        runtime: "native" for native Metal or "torch" for MPS tensors.
            CUDA uses its CuPy runtime regardless of this selection.
        device: CUDA device index; Metal supports only 0.
        unified_memory: Share native Metal exports with CPU images. Requires
            8-bit or float32 storage; CPU mutations detach shared storage.

    Returns:
        A resident GpuImage, or GpuUnsignedImage for 10/12/16-bit input.
        Float grayscale F keeps raw sample units; float color modes use
        normalized color units. Upload does not navigate source frames.

    Raises:
        ValueError: The mode, depth, dimensions, runtime, or options are invalid.
        RuntimeError: The requested backend or runtime is unavailable.
    """
    if backend not in ("metal", "cuda") or runtime not in ("native", "torch"):
        raise ValueError("unsupported GPU backend or runtime")
    _require(backend, device)
    if unified_memory and (backend != "metal" or runtime != "native" or image.bit_depth not in (8, 32)):
        raise ValueError("unified memory requires native Metal uint8 or float32 images")
    if backend == "metal" and device != 0:
        raise ValueError("Metal supports only the default device")
    _validate(image.size, image.mode)
    if image.mode in ("L", "RGB", "RGBA") and image.bit_depth in (10, 12, 16):
        from ._gpu_unsigned import GpuUnsignedImage

        mode = {"L": "F", "RGB": "RGB;F", "RGBA": "RGBA;F"}[image.mode]
        if backend == "metal" and runtime == "native":
            resource = _blanket.MetalImage.from_unsigned_image(image._native)
            return GpuUnsignedImage(GpuImage(resource, image.size, mode, runtime, dict(image._metadata_info())), image.mode, image.bit_depth)
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
    if backend == "cuda":
        from . import _gpu_cuda

        resource = _gpu_cuda.upload(image.tobytes(), (image.height, image.width, channels), floating, device)
        runtime = "cuda"
    elif runtime == "native":
        resource = _blanket.MetalImage.shared_resource(image._native) if unified_memory else None
        if resource is None:
            resource = _blanket.MetalImage.from_image(image._native)
    else:
        torch = _torch()
        resource = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.float32 if floating else torch.uint8).reshape(image.height, image.width, channels).to("mps")
    return GpuImage(resource, image.size, image.mode, runtime, dict(image._metadata_info()), unified_memory=unified_memory)


def from_image_async(image: Image.Image, *, backend: str = "metal", runtime: str = "native", device: int = 0, unified_memory: bool = False) -> Future[GpuImage | GpuUnsignedImage]:
    """Snapshot the current frame and upload it on a transfer worker.

    Args:
        image: CPU image to copy immediately; subsequent edits do not affect
            the upload.
        backend: "metal" or "cuda".
        runtime: Must be "native"; PyTorch MPS transfers are unsupported here.
        device: CUDA device index; Metal requires 0.
        unified_memory: Enable native Metal sharing as in from_image.

    Returns:
        A Future whose result is a synchronized GpuImage or GpuUnsignedImage.
        Calling result() waits and re-raises upload errors. CUDA work uses
        the stream current when the transfer was submitted.
    """
    if runtime != "native":
        raise ValueError("asynchronous transfers require native Metal or CUDA")
    snapshot = image.copy()

    def upload() -> GpuImage | GpuUnsignedImage:
        """Upload the retained snapshot, synchronize it, and release the CPU copy."""
        try:
            result = from_image(snapshot, backend=backend, runtime=runtime, device=device, unified_memory=unified_memory)
            result.synchronize()
            return result
        finally:
            snapshot.close()

    return submit(upload, device=f"cuda:{device}" if backend == "cuda" else "metal")


def from_torch(tensor: Tensor, *, layout: str, mode: str | None = None, copy: bool = False, bit_depth: int | None = None) -> GpuImage | GpuUnsignedImage:
    """Import a GPU tensor without staging pixels through CPU memory.

    Args:
        tensor: MPS or CUDA tensor with uint8 or float32 samples and no gradients.
            Unsigned-depth import instead accepts uint16 storage.
        layout: "HW", "HWC", or "CHW", matching the tensor dimensions.
        mode: Matching image mode, or None to infer it from channels and dtype.
        copy: Copy ordinary byte/float storage instead of borrowing it.
            Noncontiguous views are packed on the GPU when processed.
        bit_depth: 10, 12, or 16 for unsigned uint16 input; otherwise None.
            Unsigned samples are converted to normalized float32 working storage.

    Returns:
        A tensor-backed GpuImage, or a GpuUnsignedImage for unsigned input.
        Borrowed storage must not be mutated while GPU operations consume it.
        F samples use raw units; RGB;F/RGBA;F use normalized color units.

    Raises:
        ValueError: Layout, dtype, channels, mode, or sample depth is unsupported.
        RuntimeError: The matching GPU backend or PyTorch support is unavailable.
    """
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
    """Combine resident single-channel bands into a new GPU image.

    Args:
        mode: Supported output mode determining the number and dtype of bands.
        bands: L bands for byte output or F bands for float output, in image
            band order. Sizes, runtimes, and devices must match.

    Returns:
        An independent resident image carrying metadata from the first band.
        Band sample values are copied without rescaling.
    """
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
    """An explicitly resident byte or float32 image.

    Create images with from_image or from_torch. Operations return independent
    GPU results and preserve source pixels and metadata unless stated otherwise.
    Work is normally submitted without waiting; synchronize and CPU exports wait
    for completion. Binary operations require matching size, mode, runtime,
    and device unless their method documents different input shapes.

    Supported modes are L/LA/RGB/RGBA (uint8) and F/RGB;F/RGBA;F (float32).
    F uses raw sample units; color floats conventionally use 0..1 and can retain
    values outside that range. close releases one handle; queued work and shared
    handles retain the storage they need.

    Example:
        >>> resident = ImageGpu.from_image(image)
        >>> result = resident.resize((320, 240)).brightness(1.2)
        >>> cpu_image = result.to_image()
    """

    def __init__(self, resource: object, size: tuple[int, int], mode: str, runtime: str, info: dict[str, Any], *, unified_memory: bool = False) -> None:
        """Wrap an existing backend resource with image metadata.

        Args:
            resource: Native Metal, MPS tensor, or CUDA image resource.
            size: Raster dimensions as (width, height).
            mode: Supported mode matching the resource's channels and sample dtype.
            runtime: "native", "torch", or "cuda".
            info: Metadata dictionary owned by this handle.
            unified_memory: Enable shared CPU exports for native Metal storage.

        Use from_image or from_torch for validated public construction.
        """
        if unified_memory and runtime != "native":
            raise ValueError("unified memory requires native Metal")
        self.unified_memory = unified_memory
        self._resource: Any = resource
        self.size = size
        self.mode = mode
        self.runtime = runtime
        self.info = info
        self.device = f"cuda:{self._resource.device}" if runtime == "cuda" else "metal" if runtime == "native" else "mps"
        self.dtype = "float32" if _MODES[mode][1] else "uint8"

    def _check(self) -> None:
        """Raise ValueError if this handle has been closed."""
        if self._resource is None:
            raise ValueError("GPU image is closed")

    def _run(self, op: int, *, size: tuple[int, int] | None = None, mode: str | None = None, k: int = 0, axis: int = 0, values: list[float] | None = None, other: GpuImage | None = None) -> GpuImage:
        """Validate inputs and dispatch a numeric operation into a new resident image."""
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
            kernel(src, resource, rhs, parameters, weights, threads=_dispatch_threads(parameters, channels, floating), arg_casts={3: "int32"})
        return GpuImage(resource, size, mode, self.runtime, self.info.copy(), unified_memory=self.unified_memory)

    def convert(self, mode: str) -> GpuImage:
        """Convert mode or copy resident samples without downloading them.

        Args:
            mode: L, LA, RGB, RGBA, F, RGB;F, or RGBA;F.

        Returns:
            An independent GPU image in the requested mode. Byte/color-float
            conversions rescale between 0..255 and 0..1. F uses raw grayscale
            units, and same-mode conversion copies the image.
        """
        self._check()
        _validate(self.size, mode)
        return self._run(6) if mode == self.mode else self._run(14, mode=mode, k=_FORMATS.index(mode))

    def statistics(self) -> GpuReduction:
        """Compute resident per-channel extrema and raw moments of orders 1..4.

        Returns:
            A GpuReduction. Its to_cpu() returns one dictionary per band with
            count, extrema, sum, sum2, moments, mean, variance, and stddev.
            Values use the source's sample units, including alpha. Power sums
            accumulate in float32 and can have rounding error; NaNs propagate.
        """
        return GpuReduction(self._run(29, size=(7, _MODES[self.mode][0]), mode="F"), "statistics")

    def histogram(self) -> GpuReduction:
        """Compute exact 256-bin histograms for byte channels, including alpha.

        Returns:
            A resident GpuReduction; to_cpu() returns one list of 256 integer
            counts per band. This call submits work without downloading counts.

        Raises:
            ValueError: The image is closed or has float32 samples.
        """
        if _MODES[self.mode][1]:
            raise ValueError("GPU histograms currently require uint8 samples")
        return GpuReduction(self._run(29, size=(256, _MODES[self.mode][0]), mode="F", k=1), "histogram")

    def transform(self, size: tuple[int, int], method: int, data: Sequence[float], resample: int = Image.Resampling.NEAREST, *, fillcolor: float | Sequence[float] = 0) -> GpuImage:
        """Apply an inverse affine or perspective map to a new resident image.

        Args:
            size: Positive output (width, height).
            method: Image.Transform.AFFINE or Image.Transform.PERSPECTIVE.
            data: Six affine coefficients (a, b, c, d, e, f), or eight perspective
                coefficients adding (g, h). Coordinates map output pixel centers
                to ((a*x+b*y+c)/(g*x+h*y+1), (d*x+e*y+f)/(g*x+h*y+1)).
            resample: NEAREST, BILINEAR, or BICUBIC.
            fillcolor: Finite scalar or one value per source channel, in sample
                units, for coordinates outside the source.

        Returns:
            An image with the requested size and original mode. Interpolation
            clamps neighbor reads to source edges; alpha is sampled independently.
        """
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
        """Rotate around the image center and return a resident result.

        Args:
            angle: Finite angle in degrees; positive angles are counterclockwise.
            resample: NEAREST, BILINEAR, or BICUBIC.
            expand: Enlarge the output canvas to contain the rotated image.
            fillcolor: Finite scalar or one value per band for uncovered pixels.

        Returns:
            A GPU image in the same mode; size is unchanged unless expand is True.
            Interpolation and alpha handling follow transform.
        """
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
        """Sample through a reusable resident coordinate map.

        Args:
            coordinates: RGB;F image on the same runtime/device. Red stores source
                x, green stores source y, and blue is ignored. Pixel centers are
                (x + 0.5, y + 0.5); map dimensions determine the output size.
            resample: NEAREST, BILINEAR, or BICUBIC.
            fillcolor: Finite scalar or one value per band, used for out-of-bounds
                or nonfinite map coordinates.

        Returns:
            A new GPU image in the source mode, with independently sampled alpha.
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
        """Paste a resident source through an optional mask into a new image.

        Args:
            source: Image with the same mode, runtime, and device; sizes may differ.
            box: Integer (x, y) destination for the source's top-left corner.
                Regions outside the destination are clipped.
            mask: Optional L or F image matching the source size and execution
                context. L coverage is divided by 255; F coverage is clamped to
                0..1 and NaNs are treated as zero. None means full coverage.

        Returns:
            An independent image of the original size and mode. Every channel,
            including alpha, is interpolated separately; source alpha does not
            implicitly supply coverage.
        """
        self._check()
        source._check()
        if self.mode != source.mode or self.runtime != source.runtime:
            raise ValueError("paste source must match mode and execution context")
        if len(box) != 2:
            raise ValueError("paste box requires an (x, y) pair")
        x, y = (index(value) for value in box)
        if mask is not None:
            mask._check()
            if mask.mode not in ("L", "F") or mask.size != source.size or mask.runtime != source.runtime or mask.device != source.device:
                raise ValueError("paste mask must be L/F and match source size and execution context")
        if self.device != source.device:
            raise ValueError("paste source must match mode and execution context")
        if self.runtime == "native":
            if x >= self.size[0] or y >= self.size[1] or x <= -source.size[0] or y <= -source.size[1]:
                return self._run(6)
            resource = self._resource.paste(source._resource, None if mask is None else mask._resource, (x, y))
            return GpuImage(resource, self.size, self.mode, self.runtime, self.info.copy(), unified_memory=self.unified_memory)
        working_mode = "F" if self.mode in ("L", "F") else "RGB;F" if self.mode in ("RGB", "RGB;F") else "RGBA;F"
        background, foreground = self.convert(working_mode), source.convert(working_mode)
        if mask is None:
            # The alpha band of an opaque resident conversion supplies full coverage.
            coverage = source.convert("RGBA;F").putalpha(1.0).getchannel("A")
            maximum = 1.0
        else:
            maximum = 255.0 if mask.mode == "L" else 1.0
            coverage = mask.convert("F")
        coefficients = (1, 0, -x, 0, 1, -y)
        foreground = foreground.transform(self.size, Image.Transform.AFFINE, coefficients)
        coverage = coverage.transform(self.size, Image.Transform.AFFINE, coefficients)
        foreground = foreground._run(28, other=coverage, values=[maximum])
        background = background._run(28, other=coverage, values=[maximum], axis=1)
        return background.add(foreground).convert(self.mode)

    def grayscale(self) -> GpuImage:
        """Return resident luminance, discarding any alpha channel.

        Returns:
            An independent L image for byte input, or F image for float input.
            Float color luminance is rescaled into raw grayscale sample units.
        """
        return self.convert("F" if _MODES[self.mode][1] else "L")

    def getchannel(self, channel: int | str) -> GpuImage:
        """Extract one channel without rescaling its samples.

        Args:
            channel: Zero-based band index or band name such as "R" or "A".

        Returns:
            An independent L or F GPU image of the same size. In particular,
            extracting a float color band into F retains its original 0..1 units.

        Raises:
            ValueError: The band is absent or its index is out of range.
        """
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
        """Copy all image bands into independent resident single-channel images.

        Returns:
            A tuple in image band order. Byte bands use L; float bands use F.
            Sample values are preserved without rescaling.
        """
        self._check()
        return tuple(self.getchannel(channel) for channel in range(_MODES[self.mode][0]))

    def putalpha(self, alpha: GpuImage | float) -> GpuImage:
        """Return a resident copy with a replaced or newly added alpha band.

        Args:
            alpha: Integer coverage clamped to 0..255 for byte images, or finite
                coverage clamped to 0..1 for floats. A resident L/F band may
                instead supply coverage and must match size, dtype, and context.

        Returns:
            LA for byte grayscale, RGBA for byte color, or RGBA;F for float input.
            Float grayscale is promoted to RGB color; input pixels are unchanged.
        """
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
        """Prepare fractional box weights and submit one or three passes per axis."""
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
        """Blur every channel, including alpha, using a fractional box kernel.

        Args:
            radius: Finite nonnegative radius, or (horizontal, vertical) radii.
                Zero on an axis leaves that axis unchanged.

        Returns:
            A new image with the same size and mode. Edges repeat the nearest
            sample; byte passes retain fixed-point rounding.
        """
        return self._blur(radius, False)

    def gaussian_blur(self, radius: float | tuple[float, float] = 2.0) -> GpuImage:
        """Approximate a Gaussian blur with three box passes per axis.

        Args:
            radius: Finite nonnegative Gaussian radius, or separate horizontal
                and vertical radii. A zero radius produces an independent copy.

        Returns:
            A resident image of the same size and mode. All channels, including
            alpha, are filtered with replicated edges.
        """
        return self._blur(radius, True)

    def unsharp_mask(self, radius: float = 2.0, percent: int = 150, threshold: float = 3.0) -> GpuImage:
        """Sharpen differences between the image and its Gaussian blur.

        Args:
            radius: Finite nonnegative Gaussian blur radius.
            percent: Signed 32-bit integer percentage of the difference to add.
            threshold: Finite difference threshold in source sample units.
                Only differences strictly greater than this value are sharpened.

        Returns:
            A new GPU image with the same size and mode. Alpha is processed as
            a channel; byte results saturate to 0..255.
        """
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
        """Select a sorted neighborhood sample independently in every channel.

        Args:
            size: Square window width: 1, 3, 5, or 7.
            rank: Zero-based index in the sorted size*size samples.

        Returns:
            A resident image with replicated edges and unchanged size/mode.
            Alpha is filtered too; any NaN in a float window propagates.
        """
        size, rank = index(size), index(rank)
        if size not in (1, 3, 5, 7) or not 0 <= rank < size * size:
            raise ValueError("rank filters require an odd size from 1 to 7 and a valid rank")
        return self._run(19, k=size, axis=rank)

    def min_filter(self, size: int = 3) -> GpuImage:
        """Return the minimum sample in each channel's square neighborhood.

        Args:
            size: Odd window width from 1 through 7.

        Returns:
            A resident image with replicated edges. Alpha is included and float
            NaNs propagate. Equivalent to rank_filter(size, 0).
        """
        return self.rank_filter(size, 0)

    def max_filter(self, size: int = 3) -> GpuImage:
        """Return the maximum sample in each channel's square neighborhood.

        Args:
            size: Odd window width from 1 through 7.

        Returns:
            A resident image with replicated edges. Alpha is included and float
            NaNs propagate. Equivalent to rank_filter(size, size*size - 1).
        """
        size = index(size)
        return self.rank_filter(size, size * size - 1)

    def median_filter(self, size: int = 3) -> GpuImage:
        """Return the median sample in each channel's square neighborhood.

        Args:
            size: Odd window width from 1 through 7.

        Returns:
            A resident image with replicated edges. Alpha is included and float
            NaNs propagate. Equivalent to rank_filter(size, size*size // 2).
        """
        size = index(size)
        return self.rank_filter(size, size * size // 2)

    def _float_color(self) -> GpuImage:
        """Convert RGB/RGBA input to float color storage for linear-light operations."""
        if self.mode not in ("RGB", "RGBA", "RGB;F", "RGBA;F"):
            raise ValueError("linear-light processing requires RGB or RGBA color")
        return self.convert("RGBA;F" if _MODES[self.mode][0] == 4 else "RGB;F")

    def resize(self, size: tuple[int, int], resample: int = Image.Resampling.BICUBIC, *, linear_light: bool = False, transfer: str = "srgb", premultiply_alpha: bool = True) -> GpuImage:
        """Resize resident pixels using a separable resampling filter.

        Args:
            size: Positive output (width, height).
            resample: An Image.Resampling filter: NEAREST, LANCZOS, BILINEAR,
                BICUBIC, BOX, or HAMMING.
            linear_light: Decode color before filtering and encode it afterward.
                Requires RGB/RGBA byte or float color input.
            transfer: "srgb" to decode/encode sRGB, or "linear" when color samples
                already represent linear light. Used when linear_light is True.
            premultiply_alpha: Premultiply float alpha during filtering. False
                samples float channels independently and clamps alpha. Byte
                alpha is always premultiplied for non-nearest resampling.

        Returns:
            An independent GPU image of the requested size and original mode.
            Byte results retain fixed-point rounding; float colors are not clipped.
        """
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
        """Apply a square convolution kernel while copying border samples.

        Args:
            kernel: Row-major list of finite coefficients, with size*size entries.
                Rows follow ImageFilter.Kernel's bottom-to-top sampling order.
            size: Kernel width, either 3 or 5.
            scale: Nonzero finite divisor for coefficients. None uses their sum,
                substituting 1 when the sum is zero.
            offset: Finite value added to each output sample in source units.

        Returns:
            A resident image of the same size/mode. All channels, including alpha,
            are processed; borders lacking a full window retain the source.
        """
        if size not in (3, 5) or len(kernel) != size * size:
            raise ValueError("kernel must be 3x3 or 5x5")
        scale = (sum(kernel) or 1.0) if scale is None else scale
        if scale == 0 or not all(math.isfinite(v) for v in [*kernel, scale, offset]):
            raise ValueError("kernel parameters must be finite with nonzero scale")
        return self._run(1, k=size, values=[0.0, float(offset), *[float(v / scale) for v in kernel]])

    def crop(self, box: tuple[int, int, int, int]) -> GpuImage:
        """Copy an integer rectangle, filling outside-source samples with zero.

        Args:
            box: (left, top, right, bottom), with right/bottom exclusive. Bounds
                must fit signed 32-bit integers and define positive dimensions.

        Returns:
            A resident image sized (right-left, bottom-top) in the source mode.
            Out-of-bounds alpha is zero as well.
        """
        self._check()
        if len(box) != 4:
            raise ValueError("crop box must contain four integers")
        left, top, right, bottom = (index(v) for v in box)
        if not all(-2147483648 <= v <= 2147483647 for v in (left, top, right, bottom)):
            raise ValueError("crop bounds must fit signed 32-bit integers")
        return self._run(7, size=(right - left, bottom - top), k=left & 0xFFFFFFFF, axis=top & 0xFFFFFFFF)

    def transpose(self, method: int) -> GpuImage:
        """Flip or rotate resident samples without interpolation.

        Args:
            method: An Image.Transpose constant for a flip, right-angle rotation,
                diagonal transpose, or transverse operation.

        Returns:
            An independent image in the original mode. Quarter-turn rotations
            and diagonal transposes swap width and height.
        """
        self._check()
        method = index(method)
        if method not in range(7):
            raise ValueError("unsupported transpose method")
        size = self.size[::-1] if method in (2, 4, 5, 6) else self.size
        return self._run(8, size=size, k=method)

    def offset(self, xoffset: int, yoffset: int | None = None) -> GpuImage:
        """Translate resident pixels with wraparound.

        Args:
            xoffset: Integer horizontal offset; positive values move right.
            yoffset: Integer vertical offset; positive values move down.
                None uses xoffset for both axes.

        Returns:
            An independent image with the same size and mode. Offsets are
            reduced modulo the image dimensions.
        """
        self._check()
        xoffset = index(xoffset)
        yoffset = xoffset if yoffset is None else index(yoffset)
        return self._run(9, k=xoffset % self.size[0], axis=yoffset % self.size[1])

    def brightness(self, factor: float) -> GpuImage:
        """Scale brightness without downloading pixels.

        Args:
            factor: Finite multiplier; 0 gives black color samples and 1 copies
                their values. Values outside 0..1 are allowed.

        Returns:
            A resident image with RGBA alpha preserved. LA alpha is scaled.
            Byte samples are truncated and clipped; float colors are not clipped.
        """
        return self._run(10, values=[_scalar(factor)])

    def color(self, factor: float) -> GpuImage:
        """Adjust color saturation around per-pixel luminance.

        Args:
            factor: Finite blend factor; 0 produces grayscale color, 1 preserves
                the original, and values above 1 increase saturation.

        Returns:
            A new resident image with alpha preserved. Grayscale inputs copy
            unchanged; byte color results clip, while float colors remain unbounded.
        """
        self._check()
        factor = _scalar(factor)
        if factor == 1 or _MODES[self.mode][0] < 3:
            return self._run(6)
        return self._run(22, k=0, values=[factor])

    def contrast(self, factor: float) -> GpuImage:
        """Adjust contrast around mean luminance computed on the GPU.

        Args:
            factor: Finite blend factor; 0 fills color bands with mean luminance,
                1 preserves the original, and larger values increase contrast.

        Returns:
            An independent resident image with LA/RGBA alpha preserved. The mean
            uses every pixel; byte results clip and float results remain unbounded.
        """
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
        """Blend with the 3x3 smooth kernel to adjust sharpness.

        Args:
            factor: Finite blend factor; 0 smooths, 1 preserves the original,
                and values above 1 sharpen.

        Returns:
            A resident image retaining borders and RGBA alpha. LA alpha is
            processed; images smaller than 3 pixels on either axis are copied.
        """
        self._check()
        factor = _scalar(factor)
        if factor == 1 or min(self.size) < 3:
            return self._run(6)
        return self._run(22, k=2, values=[factor])

    def color_lut(self, lut: Color3DLUT) -> GpuImage:
        """Apply a three-dimensional color lookup table on the GPU.

        Args:
            lut: ImageFilter.Color3DLUT with three or four output channels.
                Input must be RGB/RGBA byte or float color. An explicit lut.mode
                must be supported and retain the source sample dtype.

        Returns:
            A resident image using trilinear lookup. A three-channel table
            preserves an existing alpha band; four-channel tables supply alpha.
        """
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
        """Invert every sample, including alpha, into a new resident image.

        Returns:
            255 minus each byte sample, or 1 minus each float sample. Float
            values remain unbounded, including raw-unit F samples.
        """
        return self._run(11)

    def point(self, lut: list[float]) -> GpuImage:
        """Map byte samples through shared or per-channel lookup tables.

        Args:
            lut: 256 entries shared by every channel, or 256*bands entries
                concatenated in band order. Finite entries are rounded with
                Python's round and clamped to 0..255.

        Returns:
            A new byte GPU image with every channel, including alpha, mapped.

        Raises:
            ValueError: The source has float samples or the table is invalid.
        """
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
        """Invert byte samples at or above a threshold, including alpha.

        Args:
            threshold: Integer cutoff in byte sample units; values outside
                0..255 are accepted.

        Returns:
            An independent byte GPU image. Samples below threshold are copied;
            all remaining samples become 255 minus their original value.
        """
        threshold = index(threshold)
        return self.point([v if v < threshold else 255 - v for v in range(256)])

    def posterize(self, bits: int) -> GpuImage:
        """Keep only the requested high bits of each byte sample.

        Args:
            bits: Integer from 0 through 8; 0 clears every band and 8 copies it.

        Returns:
            A resident byte image with the mask applied to every band, including
            alpha. Float sources are unsupported.
        """
        bits = index(bits)
        if bits not in range(9):
            raise ValueError("bits must be between 0 and 8")
        mask = (255 << (8 - bits)) & 255
        return self.point([v & mask for v in range(256)])

    def _binary(self, other: GpuImage, operation: int, scale: float = 1.0, offset: float = 0.0) -> GpuImage:
        """Validate arithmetic controls and dispatch a matching-image channel operation."""
        scale = _scalar(scale)
        if abs(scale) < 1.1754943508222875e-38:
            raise ValueError("scale must be a nonzero normal float32 value")
        return self._run(13, other=other, k=operation, values=[scale, _scalar(offset)])

    def difference(self, other: GpuImage) -> GpuImage:
        """Return absolute channel differences, including alpha.

        Args:
            other: GPU image matching this image's size, mode, runtime, and device.

        Returns:
            A new resident image containing abs(self - other) per sample.
        """
        return self._binary(other, 0)

    def multiply(self, other: GpuImage) -> GpuImage:
        """Multiply corresponding samples, including alpha.

        Args:
            other: GPU image matching this image's size, mode, runtime, and device.

        Returns:
            A resident image containing self*other/255 for bytes, with integer
            truncation, or self*other for floats.
        """
        return self._binary(other, 1)

    def screen(self, other: GpuImage) -> GpuImage:
        """Screen corresponding samples, including alpha.

        Args:
            other: GPU image matching this image's size, mode, runtime, and device.

        Returns:
            A resident image containing white - (white-self)*(white-other)/white,
            where white is 255 for bytes and 1 for floats.
        """
        return self._binary(other, 2)

    def lighter(self, other: GpuImage) -> GpuImage:
        """Select the larger corresponding sample in every channel.

        Args:
            other: GPU image matching this image's size, mode, runtime, and device.

        Returns:
            A new resident image of channel maxima, including alpha.
            Float NaNs propagate.
        """
        return self._binary(other, 3)

    def darker(self, other: GpuImage) -> GpuImage:
        """Select the smaller corresponding sample in every channel.

        Args:
            other: GPU image matching this image's size, mode, runtime, and device.

        Returns:
            A new resident image of channel minima, including alpha.
            Float NaNs propagate.
        """
        return self._binary(other, 4)

    def add(self, other: GpuImage, scale: float = 1.0, offset: float = 0.0) -> GpuImage:
        """Add corresponding samples, including alpha, with scale and offset.

        Args:
            other: GPU image matching size, mode, runtime, and device.
            scale: Finite, nonzero, normal float32 divisor.
            offset: Finite amount to add after scaling, in source sample units.

        Returns:
            A new resident image containing (self+other)/scale + offset.
            Byte results are floored and clipped; float results are not clipped.
        """
        return self._binary(other, 5, scale, offset)

    def subtract(self, other: GpuImage, scale: float = 1.0, offset: float = 0.0) -> GpuImage:
        """Subtract corresponding samples, including alpha, with scale and offset.

        Args:
            other: GPU image matching size, mode, runtime, and device.
            scale: Finite, nonzero, normal float32 divisor.
            offset: Finite amount to add after scaling, in source sample units.

        Returns:
            A new resident image containing (self-other)/scale + offset.
            Byte results are floored and clipped; float results are not clipped.
        """
        return self._binary(other, 6, scale, offset)

    def blend(self, other: GpuImage, alpha: float) -> GpuImage:
        """Interpolate matching images using a constant weight.

        Args:
            other: GPU image matching size, mode, runtime, and device.
            alpha: Finite weight in self + (other-self)*alpha. Values outside
                0..1 extrapolate.

        Returns:
            A new resident image with every band, including alpha, blended.
            Byte samples are floored and clipped; float samples are not clipped.
        """
        if not math.isfinite(alpha):
            raise ValueError("blend alpha must be finite")
        return self._run(2, other=other, values=[float(alpha)])

    def alpha_composite(self, source: GpuImage, *, linear_light: bool = False, transfer: str = "srgb") -> GpuImage:
        """Composite a source image over this background using straight alpha.

        Args:
            source: Image matching size, mode, runtime, and device. Mode must
                have alpha: LA, RGBA, or RGBA;F.
            linear_light: Composite color in linear light; requires RGBA/RGBA;F.
            transfer: "srgb" to decode/encode sRGB, or "linear" for color already
                in linear light. Used when linear_light is True.

        Returns:
            A new resident image in the source mode. Float alpha is clamped to
            0..1 with NaNs treated as zero; float color remains unbounded.
        """
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
        """Wait for this image's pending GPU work and report execution errors.

        Returns:
            None. Pixels stay resident. PyTorch MPS waits for the device queue;
            native Metal and CUDA wait for this resource's completion point.
        """
        self._check()
        if self.runtime in ("native", "cuda"):
            self._resource.synchronize()
        else:
            _torch().mps.synchronize()

    def to_image(self) -> Image.Image:
        """Wait for GPU completion and export a CPU image with copied metadata.

        Returns:
            An Image with the same size, mode, and samples. Normally pixels are
            copied. With native Metal unified_memory, the CPU image instead
            shares immutable storage and detaches when modified. It remains
            valid after the GPU handle is closed.
        """
        if self.runtime == "native":
            self._check()
            native = self._resource.shared_image(self.mode) if self.unified_memory else self._resource.download_image(self.mode)
            result = Image.Image(native)
            result.info.update(self.info)
            return result
        self._check()
        if self.runtime == "cuda":
            data = self._resource.download()
        else:
            self.synchronize()
            data = bytes(self._resource.contiguous().view(_torch().uint8).cpu().flatten().tolist())
        result = Image.frombytes(self.mode, self.size, data)
        result.info.update(self.info)
        return result

    def share(self) -> GpuImage:
        """Create another independently closable handle without copying pixels.

        Returns:
            A GpuImage sharing the backend resource and pending work, with an
            independent metadata dictionary. This call does not synchronize.
        """
        self._check()
        return GpuImage(self._resource, self.size, self.mode, self.runtime, self.info.copy(), unified_memory=self.unified_memory)

    def record_event(self) -> Event:
        """Record a completion point without waiting on the CPU.

        Returns:
            An Event retaining the image resource even if this handle closes.
            Native Metal/CUDA track resource completion; PyTorch MPS records
            an event on its current stream.
        """
        self._check()
        return Event(self._resource, self.runtime, self.device)

    def to_image_async(self) -> Future[Image.Image]:
        """Export pixels and metadata on a transfer worker.

        Returns:
            A Future resolving to a CPU Image after completion. Storage is
            retained even if this GPU handle closes; result() reports errors.
            CUDA uses the stream current at submission.

        Raises:
            ValueError: This image uses the PyTorch MPS runtime.
        """
        if self.runtime == "torch":
            raise ValueError("asynchronous transfers require native Metal or CUDA")
        retained = self.share()
        return submit(retained.to_image, device=self.device)

    def to_memoryview(self) -> memoryview:
        """Wait for completion and borrow native Metal sample bytes.

        Returns:
            A read-only contiguous memoryview retaining the pixel resource after
            the image closes. Float samples are exposed as raw bytes.

        Raises:
            ValueError: The handle is closed or uses CUDA/PyTorch storage.
        """
        self._check()
        if self.runtime != "native":
            raise ValueError("shared mappings require native Metal storage")
        return memoryview(self._resource)

    def to_torch(self, *, layout: str = "HWC") -> Tensor:
        """Expose tensor-backed pixels without downloading them.

        Args:
            layout: "HWC", "CHW", or "HW"; HW requires one channel.

        Returns:
            A PyTorch tensor sharing the MPS or CUDA resource, with the image's
            dtype and sample values. CUDA export uses DLPack stream ordering.
            Treat shared storage as immutable while operations consume it.

        Raises:
            ValueError: The layout is invalid or storage uses native Metal.
                Import with runtime="torch" to enable MPS tensor export.
        """
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
        """Release this GPU handle without invalidating shared handles.

        Queued work, exported views, and shared CPU images retain their resources.
        Repeated calls are allowed; subsequent operations on this handle raise
        ValueError. Closing does not download pixels.
        """
        self._resource = None


class GpuReduction:
    """Resident histograms or per-band statistics with explicit CPU export.

    Use GpuImage.histogram or GpuImage.statistics to create a reduction.
    The result retains its storage independently of the source image.
    """

    def __init__(self, image: GpuImage, kind: str) -> None:
        """Wrap a resident result image with kind 'histogram' or 'statistics'."""
        self._image = image
        self.kind = kind

    def synchronize(self) -> None:
        """Wait for reduction completion without downloading its values."""
        self._image.synchronize()

    def record_event(self) -> Event:
        """Retain a nonblocking completion point for this reduction.

        Returns:
            An Event that remains valid after the reduction handle closes.
        """
        return self._image.record_event()

    def to_cpu_async(self) -> Future[list[list[int]] | list[dict[str, object]]]:
        """Download and decode the reduction on a transfer worker.

        Returns:
            A Future resolving to the same band-ordered lists as to_cpu().
            Storage survives closing this handle; result() reports transfer errors.

        Raises:
            ValueError: The reduction uses the PyTorch MPS runtime.
        """
        if self._image.runtime == "torch":
            raise ValueError("asynchronous transfers require native Metal or CUDA")
        retained = GpuReduction(self._image.share(), self.kind)
        return submit(retained.to_cpu, device=self._image.device)

    def close(self) -> None:
        """Release this reduction handle; retained events and transfers stay valid."""
        self._image.close()

    def to_cpu(self) -> list[list[int]] | list[dict[str, object]]:
        """Wait for completion, download, and decode band-ordered results.

        Returns:
            For histograms, one 256-element list of exact integer counts per band.
            For statistics, one dictionary per band containing count, extrema,
            sum, sum2, moments, mean, variance, and stddev. moments contains
            normalized raw moments of orders 1..4; variance is the population
            variance, clamped to zero for negative rounding error. Statistics use
            source sample units and float32 accumulation; NaNs propagate.
        """
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
