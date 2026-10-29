"""Reusable native Metal command batches with bounded working storage."""

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
    def __init__(self) -> None:
        self.steps: list[tuple[int, int, int, list[float]]] = []

    def run(self, _other: object, parameters: list[int], values: list[float]) -> _Recorder:
        op, kernel, axis = parameters[5:]
        self.steps.append((op, kernel, axis, [] if op in (6, 11, 19) else values))
        return self


class ProcessingPlan:
    """Prepare same-size unary operations for native Metal images.

    Each operation is a (method_name, keyword_arguments) pair. memory_limit
    bounds the two tile scratch buffers; input/output rasters and retained
    controls are separate. Calls preserve every public operation's rounding.
    """

    def __init__(self, size: tuple[int, int], mode: str, operations: Sequence[tuple[str, dict[str, Any]]], *, memory_limit: int = 64 * 1024 * 1024, bit_depth: int | None = None) -> None:
        from ._gpu_unsigned import GpuUnsignedImage
        from .ImageGpu import _MODES, GpuImage, _require, _validate

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
        _require()
        channels, floating = _MODES[working_mode]
        self._resource: Any = _blanket.MetalPlan(*size, channels, floating, recorder.steps, memory_limit)
        self.size = size
        self.mode = mode
        self.bit_depth = bit_depth
        self._working_mode = working_mode
        self.memory_limit = memory_limit

    def _check(self) -> None:
        if self._resource is None:
            raise ValueError("GPU plan is closed")

    def info(self) -> dict[str, object]:
        """Report fused dispatch count, tile geometry, and working bytes."""
        self._check()
        steps, tile, halo, scratch_bytes = self._resource.info()
        return {"steps": steps, "tile_size": tuple(tile), "halo": tuple(halo), "scratch_bytes": scratch_bytes}

    def run(self, image: GpuImage | GpuUnsignedImage) -> GpuImage | GpuUnsignedImage:
        """Submit one image without waiting or downloading pixels."""
        return self.run_batch([image])[0]

    def run_batch(self, images: Sequence[GpuImage | GpuUnsignedImage]) -> list[GpuImage | GpuUnsignedImage]:
        """Submit up to 1024 images in one ordered Metal command buffer."""
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
            if image.runtime != "native" or image.size != self.size or image.mode != self.mode or depth != self.bit_depth:
                raise ValueError("GPU plan input must match size, mode, sample depth, and native Metal runtime")
            working.append(raster)
        resources = self._resource.run_batch([image._resource for image in working])
        results: list[GpuImage | GpuUnsignedImage] = []
        for resource, image in zip(resources, working, strict=True):
            result = GpuImage(resource, self.size, self._working_mode, "native", image.info.copy(), unified_memory=image.unified_memory)
            results.append(GpuUnsignedImage(result, self.mode, self.bit_depth) if self.bit_depth in (10, 12, 16) else result)
        return results

    def close(self) -> None:
        """Release prepared controls; submitted commands retain their buffers."""
        self._resource = None


def configure_cache(*, image_bytes: int = 128 * 1024 * 1024, control_bytes: int = 16 * 1024 * 1024) -> None:
    """Set Metal cache budgets and discard previously cached entries."""
    from .ImageGpu import _require

    image_bytes, control_bytes = index(image_bytes), index(control_bytes)
    if image_bytes < 0 or control_bytes < 0:
        raise ValueError("GPU cache budgets must be nonnegative")
    _require()
    _blanket._gpu_configure_cache(image_bytes, control_bytes)


def cache_info() -> dict[str, int]:
    """Return retained cache bytes and configured limits for native Metal."""
    from .ImageGpu import _require

    _require()
    image_bytes, image_limit, control_bytes, control_limit = _blanket._gpu_cache_info()
    return {"image_bytes": image_bytes, "image_limit": image_limit, "control_bytes": control_bytes, "control_limit": control_limit}
