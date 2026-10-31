"""CUDA allocation, DLPack ownership, and stream adaptation for Rust kernels."""

from __future__ import annotations

__lazy_modules__ = {"cupy", "numpy", "torch", "collections.abc", "types"}

import atexit
from collections import deque
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import RLock
from typing import TYPE_CHECKING

from . import _blanket

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType
    from typing import Any

    from torch import Tensor


_PENDING: dict[tuple[int, int], deque[tuple[Any, tuple[object, ...]]]] = {}
_PENDING_LOCK = RLock()
_PENDING_REGISTERED = False


def _collect_pending(device: int, stream: int | None = None) -> None:
    """Release completed submissions without waiting for GPU work."""
    retired = []
    with _PENDING_LOCK:
        keys = tuple(_PENDING) if stream is None else ((device, stream),)
        for key in keys:
            if key[0] != device:
                continue
            queue = _PENDING.get(key)
            if queue is None:
                continue
            while queue and queue[0][0].query():
                retired.append(queue.popleft())
            if not queue:
                del _PENDING[key]
    retired.clear()


def _finish_pending() -> None:
    """Keep asynchronous storage alive until its device finishes at shutdown."""
    with _PENDING_LOCK:
        pending = list(_PENDING.items())
        _PENDING.clear()
    for (device, _), queue in pending:
        with cupy().cuda.Device(device):
            for event, _owners in queue:
                event.synchronize()


def cupy() -> ModuleType:
    """Import CuPy lazily, raising RuntimeError with CUDA dependency guidance."""
    try:
        import cupy
    except (ImportError, OSError) as error:
        raise RuntimeError("CUDA requires CuPy and the CUDA runtime; install 'cupy-cuda13x[ctk]>=14.2'") from error
    return cupy


def require(device: int) -> None:
    """Require compiled CUDA kernels and a usable NVIDIA device.

    Args:
        device: Nonnegative CUDA device index.

    Raises:
        ValueError: The index is invalid or outside the driver's device count.
        RuntimeError: CUDA support, dependencies, driver initialization, or
            kernel loading is unavailable.
    """
    if not hasattr(_blanket, "_gpu_cuda_fatbin"):
        raise RuntimeError("CUDA kernels unavailable; build Blanket with the gpu-cuda feature")
    if type(device) is not int or device < 0:
        raise ValueError("CUDA device must be a nonnegative integer")
    cp = cupy()
    try:
        count = cp.cuda.runtime.getDeviceCount()
    except Exception as error:
        raise RuntimeError(f"CUDA driver initialization failed: {error}") from error
    if count == 0:
        raise RuntimeError("CUDA unavailable: no NVIDIA GPUs detected")
    if device >= count:
        raise ValueError(f"CUDA device {device} unavailable; found {count} NVIDIA devices")
    try:
        library(device)
    except RuntimeError:
        raise
    except Exception as error:
        raise RuntimeError(f"CUDA context initialization failed on device {device}: {error}") from error


@lru_cache(maxsize=8)
def library(device: int) -> Callable[[str], Callable[..., None]]:
    """Load and cache the extension's precompiled CUDA module on one device.

    Args:
        device: CUDA device index used for module loading and entry resolution.

    Returns:
        A kernel-name lookup callable retaining the loaded module functions.
        No CUDA source compilation is performed at runtime.

    Raises:
        RuntimeError: Driver or GPU architecture compatibility prevents loading.
    """
    cp = cupy()
    with cp.cuda.Device(device):
        try:
            with TemporaryDirectory(prefix="blanket-cuda-") as directory:
                path = Path(directory) / "blanket.fatbin"
                path.write_bytes(_blanket._gpu_cuda_fatbin)
                module = cp.RawModule(path=str(path))
                # Resolve all entry points before removing the temporary file.
                names = ("image_u8", "image_f32", "reduce_u8", "reduce_f32", "convert_u8_u8", "convert_u8_f32", "convert_f32_u8", "convert_f32_f32", "tile_u8", "tile_f32", "finish_statistics")
                kernels = {name: module.get_function(name) for name in names}
        except Exception as error:
            raise RuntimeError(f"CUDA kernel loading failed on device {device}; check NVIDIA driver and GPU architecture compatibility: {error}") from error
    return kernels.__getitem__


class CudaImage:
    """Own a CUDA array and retain asynchronous input owners until completion.

    The array's device, submission stream, and completion event travel with the
    resource. Consumers order streams through events before reading its pixels.
    """

    def __init__(self, array: object, *, owners: tuple[object, ...] = (), completion: object | None = None) -> None:
        """Retain CUDA storage and its completion dependencies.

        Args:
            array: CuPy-compatible GPU array exposing its device identifier.
            owners: Objects that must outlive this resource's pending work.
            completion: Existing CUDA completion event, or None to record one
                on the current stream of the array's device.
        """
        global _PENDING_REGISTERED
        cp = cupy()
        self.array: Any = array
        self.device = int(self.array.device.id)
        with cp.cuda.Device(self.device):
            self.stream: Any = cp.cuda.get_current_stream()
            self.event: Any = cp.cuda.Event() if completion is None else completion
            if completion is None:
                self.event.record(self.stream)
            _collect_pending(self.device, int(self.stream.ptr))
            dependencies = tuple(owner.array if isinstance(owner, CudaImage) else owner for owner in owners)
            with _PENDING_LOCK:
                if not _PENDING_REGISTERED:
                    atexit.register(_finish_pending)
                    _PENDING_REGISTERED = True
                queue = _PENDING.setdefault((self.device, int(self.stream.ptr)), deque())
                queue.append((self.event, (self.array, *dependencies)))

    def synchronize(self) -> None:
        """Wait for this resource's event and wrap execution failures as RuntimeError."""
        try:
            with cupy().cuda.Device(self.device):
                self.event.synchronize()
                _collect_pending(self.device)
        except Exception as error:
            raise RuntimeError(f"CUDA execution failed on device {self.device}: {error}") from error

    def download(self) -> bytes:
        """Copy CUDA pixels through pinned host storage and wait for completion.

        Returns:
            Packed CPU bytes in the array's dtype and shape order. The current
            stream waits for this resource's event before starting the transfer.

        Raises:
            RuntimeError: Transfer or CUDA execution fails.
        """
        import numpy as np

        cp = cupy()
        with cp.cuda.Device(self.device):
            stream = cp.cuda.get_current_stream()
            stream.wait_event(self.event)
            memory = cp.cuda.alloc_pinned_memory(self.array.nbytes)
            host = np.frombuffer(memory, dtype=self.array.dtype, count=self.array.size).reshape(self.array.shape)
            try:
                self.array.get(out=host, stream=stream, blocking=False)
                completion = cp.cuda.Event()
                completion.record(stream)
                completion.synchronize()
                _collect_pending(self.device)
                return host.tobytes()
            except Exception as error:
                stream.synchronize()
                raise RuntimeError(f"CUDA download failed on device {self.device}: {error}") from error

    def to_torch(self) -> Tensor:
        """Export the CUDA array as a shared PyTorch tensor through DLPack.

        Returns:
            A tensor on the same device, with DLPack-managed storage ownership
            and stream ordering. No CPU pixel transfer is performed.
        """
        import torch
        from torch.utils import dlpack

        cp = cupy()
        with cp.cuda.Device(self.device), self.stream, torch.cuda.device(self.device):
            return dlpack.from_dlpack(self.array)

    def run(self, other: CudaImage, parameters: list[int], values: list[float], channels: int, floating: bool, source_float: bool) -> CudaImage:
        """Submit a validated image operation on the current CUDA stream.

        Args:
            other: Secondary resource on the same CUDA device.
            parameters: Eight protocol words describing input/output geometry
                and operation-specific selectors.
            values: Float32 operation controls.
            channels: Output channel count.
            floating: Whether output samples use float32 rather than uint8.
            source_float: Whether source samples use float32.

        Returns:
            A new CudaImage retaining inputs, packed views, controls, and its
            kernel until asynchronous execution completes.

        Raises:
            ValueError: Inputs use different CUDA devices.
            RuntimeError: Allocation, preparation, or dispatch fails.
        """
        from .ImageGpu import _dispatch_threads

        if self.device != other.device:
            raise ValueError("CUDA images must use the same device")
        cp = cupy()
        with cp.cuda.Device(self.device):
            stream = cp.cuda.get_current_stream()
            stream.wait_event(self.event)
            if other is not self:
                stream.wait_event(other.event)
            try:
                source = cp.ascontiguousarray(self.array)
                rhs = source if other is self else cp.ascontiguousarray(other.array)
                output = cp.empty((parameters[4], parameters[3], channels), dtype=cp.float32 if floating else cp.uint8)
                from ._gpu_cuda_plan import cache

                state = cache(self.device)
                op = parameters[5]
                blocks = min((parameters[0] * parameters[1] + 1023) // 1024, 4096) if op == 29 else 1
                if op == 29:
                    parameters = [*parameters[:7], blocks]
                parameter_owner = state.control(parameters, "uint32")
                weight_owner = state.control(values, "float32")
                controls, weights = parameter_owner.array, weight_owner.array
                kernel_name = "image_f32" if floating else "image_u8"
                if op == 14:
                    kernel_name = f"convert_{'f32' if source_float else 'u8'}_{'f32' if floating else 'u8'}"
                elif op == 21:
                    kernel_name = "reduce_f32" if source_float else "reduce_u8"
                elif op == 29:
                    kernel_name = "image_f32" if source_float else "image_u8"
                kernel = library(self.device)(kernel_name)
                # A moderate block size covers multiple warps without assuming GPU limits.
                threads = 128
                count = _dispatch_threads(parameters, channels, floating)
                partials = cp.empty((output.size * blocks,), dtype=cp.float32) if blocks > 1 else output
                if op == 29:
                    count = parameters[2] * blocks
                kernel(((count + threads - 1) // threads,), (threads,), (source, partials, rhs, controls, weights), stream=stream)
                finish = None
                if blocks > 1:
                    finish = library(self.device)("finish_statistics")
                    count = output.size if parameters[6] == 1 else parameters[2]
                    finish(((count + threads - 1) // threads,), (threads,), (partials, output, controls), stream=stream)
                return CudaImage(output, owners=(self, other, source, rhs, parameter_owner, weight_owner, kernel, partials, finish))
            except Exception as error:
                # Failed launches must finish before temporary input owners disappear.
                stream.synchronize()
                raise RuntimeError(f"CUDA dispatch failed on device {self.device}: {error}") from error


def upload(data: bytes, shape: tuple[int, int, int], floating: bool, device: int) -> CudaImage:
    """Upload packed CPU samples through pinned memory on the current stream.

    Args:
        data: Packed uint8 or float32 sample bytes matching shape.
        shape: Output array shape in (height, width, channels) order.
        floating: True for float32 samples, False for uint8.
        device: CUDA device index.

    Returns:
        A CudaImage retaining pinned input storage until transfer completes.

    Raises:
        RuntimeError: CUDA initialization, allocation, or upload fails.
    """
    import numpy as np

    require(device)
    cp = cupy()
    with cp.cuda.Device(device):
        try:
            memory = cp.cuda.alloc_pinned_memory(len(data))
            host = np.frombuffer(memory, dtype=np.float32 if floating else np.uint8).reshape(shape)
            host.view(np.uint8).reshape(-1)[:] = np.frombuffer(data, dtype=np.uint8)
            array = cp.asarray(host, blocking=False)
            return CudaImage(array, owners=(memory, host))
        except Exception as error:
            cp.cuda.get_current_stream().synchronize()
            raise RuntimeError(f"CUDA upload/allocation failed on device {device}: {error}") from error


def from_torch(tensor: Tensor, *, copy: bool = False) -> CudaImage:
    """Import CUDA tensor storage through DLPack without a CPU transfer.

    Args:
        tensor: Validated CUDA tensor whose storage and device are retained.
        copy: Copy into independent CUDA storage rather than borrowing it.

    Returns:
        A CudaImage recording completion on the current device stream.
        Layout, dtype, mode, and gradient checks belong to ImageGpu.from_torch.
    """
    device = tensor.device.index
    require(device)
    cp = cupy()
    with cp.cuda.Device(device):
        array = cp.from_dlpack(tensor)
        if copy:
            array = array.copy()
        return CudaImage(array, owners=(tensor,))
