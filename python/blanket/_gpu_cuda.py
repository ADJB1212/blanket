"""CUDA allocation, DLPack ownership, and stream adaptation for Rust kernels."""

from __future__ import annotations

__lazy_modules__ = {"cupy", "numpy", "torch", "collections.abc", "types"}

from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

from . import _blanket

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType
    from typing import Any

    from torch import Tensor


def cupy() -> ModuleType:
    try:
        import cupy
    except ImportError as error:
        raise RuntimeError("CUDA requires the optional gpu-cuda dependency group (CuPy and NVIDIA CUDA runtime)") from error
    return cupy


def require(device: int) -> None:
    if not hasattr(_blanket, "_gpu_cuda_fatbin"):
        raise RuntimeError("CUDA kernels unavailable; build Blanket with the gpu-cuda feature")
    if type(device) is not int or device < 0:
        raise ValueError("CUDA device must be a nonnegative integer")
    try:
        count = cupy().cuda.runtime.getDeviceCount()
        if device >= count:
            raise ValueError(f"CUDA device {device} unavailable; found {count} NVIDIA devices")
    except ValueError:
        raise
    except Exception as error:
        raise RuntimeError(f"CUDA driver initialization failed: {error}") from error
    library(device)


@lru_cache(maxsize=8)
def library(device: int) -> Callable[[str], Callable[..., None]]:
    cp = cupy()
    with cp.cuda.Device(device):
        try:
            with TemporaryDirectory(prefix="blanket-cuda-") as directory:
                path = Path(directory) / "blanket.fatbin"
                path.write_bytes(_blanket._gpu_cuda_fatbin)
                module = cp.RawModule(path=str(path))
                # Resolve all entry points before removing the temporary file.
                names = ("image_u8", "image_f32", "reduce_u8", "reduce_f32", "convert_u8_u8", "convert_u8_f32", "convert_f32_u8", "convert_f32_f32", "tile_u8", "tile_f32")
                kernels = {name: module.get_function(name) for name in names}
        except Exception as error:
            raise RuntimeError(f"CUDA kernel loading failed on device {device}; check NVIDIA driver and GPU architecture compatibility: {error}") from error
    return kernels.__getitem__


class CudaImage:
    """Own an array and retain all asynchronous inputs until completion."""

    def __init__(self, array: object, *, owners: tuple[object, ...] = (), completion: object | None = None) -> None:
        cp = cupy()
        self.array: Any = array
        self.device = int(self.array.device.id)
        self.owners = owners
        with cp.cuda.Device(self.device):
            self.stream: Any = cp.cuda.get_current_stream()
            self.event: Any = cp.cuda.Event() if completion is None else completion
            if completion is None:
                self.event.record(self.stream)

    def __del__(self) -> None:
        # Keep owners alive while the last handle waits for its submitted work.
        if hasattr(self, "event"):
            with cupy().cuda.Device(self.device):
                self.event.synchronize()

    def synchronize(self) -> None:
        try:
            with cupy().cuda.Device(self.device):
                self.event.synchronize()
        except Exception as error:
            raise RuntimeError(f"CUDA execution failed on device {self.device}: {error}") from error

    def download(self) -> bytes:
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
                return host.tobytes()
            except Exception as error:
                stream.synchronize()
                raise RuntimeError(f"CUDA download failed on device {self.device}: {error}") from error

    def to_torch(self) -> Tensor:
        import torch
        from torch.utils import dlpack

        cp = cupy()
        with cp.cuda.Device(self.device), self.stream, torch.cuda.device(self.device):
            return dlpack.from_dlpack(self.array)

    def run(self, other: CudaImage, parameters: list[int], values: list[float], channels: int, floating: bool, source_float: bool) -> CudaImage:
        from .ImageGpu import _dispatch_threads

        if self.device != other.device:
            raise ValueError("CUDA images must use the same device")
        cp = cupy()
        with cp.cuda.Device(self.device):
            stream = cp.cuda.get_current_stream()
            stream.wait_event(self.event)
            stream.wait_event(other.event)
            try:
                source = cp.ascontiguousarray(self.array)
                rhs = cp.ascontiguousarray(other.array)
                output = cp.empty((parameters[4], parameters[3], channels), dtype=cp.float32 if floating else cp.uint8)
                from ._gpu_cuda_plan import cache

                state = cache(self.device)
                parameter_owner = state.control(parameters, "uint32")
                weight_owner = state.control(values, "float32")
                controls, weights = parameter_owner.array, weight_owner.array
                op = parameters[5]
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
                kernel(((count + threads - 1) // threads,), (threads,), (source, output, rhs, controls, weights), stream=stream)
                return CudaImage(output, owners=(self, other, source, rhs, parameter_owner, weight_owner, kernel))
            except Exception as error:
                # Failed launches must finish before temporary input owners disappear.
                stream.synchronize()
                raise RuntimeError(f"CUDA dispatch failed on device {self.device}: {error}") from error


def upload(data: bytes, shape: tuple[int, int, int], floating: bool, device: int) -> CudaImage:
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
    device = tensor.device.index
    require(device)
    cp = cupy()
    with cp.cuda.Device(device):
        array = cp.from_dlpack(tensor)
        if copy:
            array = array.copy()
        return CudaImage(array, owners=(tensor,))
