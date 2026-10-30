"""Reusable GPU plans with bounded working storage."""

from __future__ import annotations

__lazy_modules__ = {"collections.abc"}

from operator import index
from typing import TYPE_CHECKING

from . import _blanket

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    from ._gpu_unsigned import GpuUnsignedImage
    from .ImageGpu import GpuImage

_OPERATIONS = frozenset(("brightness", "invert", "point", "solarize", "posterize", "convolve", "box_blur", "gaussian_blur", "rank_filter", "min_filter", "max_filter", "median_filter"))


class _Recorder:
    """Capture validated unary dispatches while preparing a processing plan."""

    def __init__(self) -> None:
        """Initialize an empty ordered list of operation controls."""
        self.steps: list[tuple[int, int, int, list[float]]] = []

    def run(self, _other: object, parameters: list[int], values: list[float]) -> _Recorder:
        """Record a dispatch in place of execution and return this recorder."""
        op, kernel, axis = parameters[5:]
        self.steps.append((op, kernel, axis, [] if op in (6, 11, 19) else values))
        return self


class ProcessingPlan:
    """A reusable sequence of same-size unary operations for native Metal or CUDA.

    Plans validate operations up front, fuse compatible byte lookup tables, and
    tile execution to bound scratch storage. Inputs and results remain resident;
    each result preserves the corresponding input's metadata. Public operation
    rounding is retained, including per-operation unsigned-depth quantization.

    Supported operations are brightness, invert, point, solarize, posterize,
    convolve, box_blur, gaussian_blur, rank_filter, min_filter, max_filter, and
    median_filter. Unsigned-depth plans exclude lookup-table operations.

    Example:
        >>> from blanket import ImageGpu
        >>> plan = ImageGpu.ProcessingPlan((640, 480), "RGB", [("gaussian_blur", {"radius": 2}), ("brightness", {"factor": 1.2})])
        >>> result = plan.run(resident_image)
    """

    def __init__(
        self,
        size: tuple[int, int],
        mode: str,
        operations: Sequence[tuple[str, dict[str, Any]]],
        *,
        memory_limit: int = 64 * 1024 * 1024,
        bit_depth: int | None = None,
        backend: str = "metal",
        device: int = 0,
    ) -> None:
        """Validate and prepare a resident processing plan.

        Args:
            size: Positive input/output (width, height), shared by all operations.
            mode: Supported GPU image mode.
            operations: Ordered sequence of (method_name, keyword_arguments) pairs,
                containing at most 1024 supported same-size unary operations.
            memory_limit: Positive byte limit for the two tile scratch buffers.
                Input/output rasters and retained control buffers are separate.
            bit_depth: None for the mode's default 8/32-bit samples, or 10/12/16
                for unsigned L/RGB/RGBA images.
            backend: "metal" for native Metal or "cuda".
            device: CUDA device index; Metal supports only 0.

        Raises:
            ValueError: An operation, descriptor, or memory budget is unsupported,
                or the budget cannot hold one tile with its required filter halo.
            RuntimeError: The requested backend is unavailable.
        """
        from ._gpu_unsigned import GpuUnsignedImage
        from .ImageGpu import _MODES, GpuImage, _require, _validate

        _validate_backend(backend, device)

        size = tuple(index(value) for value in size)
        if len(size) != 2:
            raise ValueError("GPU plan size requires two dimensions")
        _validate(size, mode)
        channels, floating = _MODES[mode]
        bit_depth = (32 if floating else 8) if bit_depth is None else index(bit_depth)
        unsigned = bit_depth in (10, 12, 16) and mode in ("L", "RGB", "RGBA")
        if not unsigned and bit_depth != (32 if floating else 8):
            raise ValueError("unsupported GPU plan mode or sample depth")
        working_mode = {"L": "F", "RGB": "RGB;F", "RGBA": "RGBA;F"}[mode] if unsigned else mode
        memory_limit = index(memory_limit)
        if memory_limit <= 0:
            raise ValueError("GPU plan memory_limit must be positive")
        if len(operations) > 1024:
            raise ValueError("GPU plans support at most 1024 operations")
        recorder = _Recorder()
        template: GpuImage | GpuUnsignedImage = GpuImage(recorder, size, working_mode, "native", {})
        if unsigned:
            template = GpuUnsignedImage(template, mode, bit_depth)
        for name, arguments in operations:
            if name not in _OPERATIONS or (unsigned and name in ("point", "posterize", "solarize")):
                raise ValueError(f"unsupported GPU plan operation: {name}")
            template = getattr(template, name)(**arguments)
        _require(backend, device)
        channels, floating = _MODES[working_mode]
        self._resource: Any
        if backend == "cuda":
            from ._gpu_cuda_plan import CudaPlan

            self._resource = CudaPlan(*size, channels, floating, recorder.steps, memory_limit, device)
        else:
            self._resource = _blanket.MetalPlan(*size, channels, floating, recorder.steps, memory_limit)
        self.runtime = "cuda" if backend == "cuda" else "native"
        self.device = f"cuda:{device}" if backend == "cuda" else "metal"
        self.size = size
        self.mode = mode
        self.bit_depth = bit_depth
        self._working_mode = working_mode
        self.memory_limit = memory_limit

    def _check(self) -> None:
        """Raise ValueError if this processing plan has been closed."""
        if self._resource is None:
            raise ValueError("GPU plan is closed")

    def info(self) -> dict[str, object]:
        """Describe the prepared operations and bounded working geometry.

        Returns:
            A dictionary with steps (fused dispatch count), tile_size (width,
            height), halo (horizontal, vertical), and scratch_bytes (both scratch
            buffers together). Sizes and halos are measured in pixels.
        """
        self._check()
        steps, tile, halo, scratch_bytes = self._resource.info()
        return {"steps": steps, "tile_size": tuple(tile), "halo": tuple(halo), "scratch_bytes": scratch_bytes}

    def run(self, image: GpuImage | GpuUnsignedImage) -> GpuImage | GpuUnsignedImage:
        """Submit one resident image without waiting for GPU completion.

        Args:
            image: Open GpuImage or GpuUnsignedImage matching this plan's size,
                mode, depth, runtime, and device.

        Returns:
            An independent resident image of the same type and descriptor.
            The input remains unchanged; synchronize or export to wait.
        """
        return self.run_batch([image])[0]

    def run_batch(self, images: Sequence[GpuImage | GpuUnsignedImage]) -> list[GpuImage | GpuUnsignedImage]:
        """Submit a batch sharing the plan's scratch storage and completion point.

        Args:
            images: Up to 1024 open GPU images, each matching the prepared size,
                mode, depth, runtime, and device. Empty batches are allowed.

        Returns:
            Independent resident images in input order, with copied metadata.
            No CPU pixel transfer or completion wait is performed on success.

        Raises:
            TypeError: An input is not a supported GPU image type.
            ValueError: The plan is closed, the batch is too large, or an input
                descriptor/execution context differs.
        """
        from ._gpu_unsigned import GpuUnsignedImage
        from .ImageGpu import GpuImage

        self._check()
        images = list(images)
        if len(images) > 1024:
            raise ValueError("GPU plan batches support at most 1024 images")
        working = []
        for image in images:
            if not isinstance(image, (GpuImage, GpuUnsignedImage)):
                raise TypeError("GPU plans require GpuImage inputs")
            raster = image._image if isinstance(image, GpuUnsignedImage) else image
            depth = image.bit_depth if isinstance(image, GpuUnsignedImage) else (32 if raster.dtype == "float32" else 8)
            raster._check()
            if image.runtime != self.runtime or image.device != self.device or image.size != self.size or image.mode != self.mode or depth != self.bit_depth:
                raise ValueError("GPU plan input must match size, mode, sample depth, runtime, and device")
            working.append(raster)
        resources = self._resource.run_batch([image._resource for image in working])
        results: list[GpuImage | GpuUnsignedImage] = []
        for resource, image in zip(resources, working, strict=True):
            result = GpuImage(resource, self.size, self._working_mode, self.runtime, image.info.copy(), unified_memory=image.unified_memory)
            results.append(GpuUnsignedImage(result, self.mode, self.bit_depth) if self.bit_depth in (10, 12, 16) else result)
        return results

    def close(self) -> None:
        """Release this plan's prepared resources.

        Queued commands retain their controls and scratch storage. Repeated calls
        are allowed; further execution or info queries raise ValueError.
        """
        self._resource = None


def _validate_backend(backend: str, device: int) -> None:
    """Validate a Metal/CUDA backend and its nonnegative device index."""
    if backend not in ("metal", "cuda") or type(device) is not int or device < 0 or (backend == "metal" and device != 0):
        raise ValueError("unsupported GPU plan/cache backend or device")


def configure_cache(*, image_bytes: int = 128 * 1024 * 1024, control_bytes: int = 16 * 1024 * 1024, backend: str = "metal", device: int = 0) -> None:
    """Set retained-resource cache budgets and discard cached entries.

    Args:
        image_bytes: Nonnegative byte budget for reusable image/scratch buffers.
            Zero disables their retention.
        control_bytes: Nonnegative byte budget for immutable control buffers.
            Zero disables their retention.
        backend: "metal" or "cuda".
        device: CUDA device index; Metal supports only 0.

    Active images and submitted work remain valid. Budgets bound resources
    retained by Blanket's caches, not total process GPU memory.

    Raises:
        ValueError: A budget, backend, or device is invalid.
        RuntimeError: The requested backend is unavailable.
    """
    from .ImageGpu import _require

    image_bytes, control_bytes = index(image_bytes), index(control_bytes)
    if image_bytes < 0 or control_bytes < 0:
        raise ValueError("GPU cache budgets must be nonnegative")
    _validate_backend(backend, device)
    _require(backend, device)
    if backend == "cuda":
        from ._gpu_cuda import cupy
        from ._gpu_cuda_plan import cache

        with cupy().cuda.Device(device):
            cache(device).configure(image_bytes, control_bytes)
    else:
        _blanket._gpu_configure_cache(image_bytes, control_bytes)


def cache_info(*, backend: str = "metal", device: int = 0) -> dict[str, int]:
    """Report current retained-resource cache usage for a GPU device.

    Args:
        backend: "metal" or "cuda".
        device: CUDA device index; Metal supports only 0.

    Returns:
        A dictionary of integer byte counts: image_bytes, image_limit,
        control_bytes, and control_limit. Live images and resources retained
        by submitted work are not all represented by these cache totals.
    """
    from .ImageGpu import _require

    _validate_backend(backend, device)
    _require(backend, device)
    if backend == "cuda":
        from ._gpu_cuda_plan import cache

        image_bytes, image_limit, control_bytes, control_limit = cache(device).info()
    else:
        image_bytes, image_limit, control_bytes, control_limit = _blanket._gpu_cache_info()
    return {"image_bytes": image_bytes, "image_limit": image_limit, "control_bytes": control_bytes, "control_limit": control_limit}
