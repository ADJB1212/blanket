"""CUDA allocation, DLPack ownership, and stream adaptation for Rust kernels."""

from __future__ import annotations

__lazy_modules__ = {"cupy", "numpy", "torch", "collections.abc", "types"}

from functools import lru_cache
from typing import TYPE_CHECKING, Any

from . import _blanket

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

    from torch import Tensor


def cupy() -> ModuleType:
    try:
        import cupy
    except ImportError as error:
        raise RuntimeError("CUDA requires the optional gpu-cuda dependency group (CuPy and NVIDIA CUDA runtime)") from error
    return cupy


def require(device: int) -> None:
    if not hasattr(_blanket, "_gpu_cuda_source"):
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
            module = cp.RawModule(code=_blanket._gpu_cuda_source, options=("--std=c++17", "--fmad=false"), backend="nvrtc")
            module.compile()
        except Exception as error:
            raise RuntimeError(f"CUDA kernel compilation failed on device {device}; check NVRTC/runtime installation and GPU architecture: {error}") from error
    return module.get_function


class CudaImage:
    """Own an array and retain all asynchronous inputs until completion."""

    def __init__(self, array: object, *, owners: tuple[object, ...] = ()) -> None:
        cp = cupy()
        self.array: Any = array
        self.device = int(self.array.device.id)
        self.owners = owners
        with cp.cuda.Device(self.device):
            self.stream: Any = cp.cuda.get_current_stream()
            self.event: Any = cp.cuda.Event()
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
        cp = cupy()
        self.synchronize()
        with cp.cuda.Device(self.device), self.stream:
            return self.array.get().tobytes()

    def to_torch(self) -> Tensor:
        import torch
        from torch.utils import dlpack

        cp = cupy()
        with cp.cuda.Device(self.device), self.stream, torch.cuda.device(self.device):
            return dlpack.from_dlpack(self.array)

    def run(self, other: CudaImage, parameters: list[int], values: list[float], channels: int, floating: bool, source_float: bool) -> CudaImage:
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
                controls = cp.asarray(parameters, dtype=cp.uint32)
                weights = cp.asarray(values, dtype=cp.float32)
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
                count = parameters[3] * parameters[4] * channels
                kernel(((count + threads - 1) // threads,), (threads,), (source, output, rhs, controls, weights), stream=stream)
                return CudaImage(output, owners=(self, other, source, rhs, controls, weights, kernel))
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
            host = np.frombuffer(data, dtype=np.float32 if floating else np.uint8).reshape(shape)
            array = cp.asarray(host, blocking=True)
            return CudaImage(array)
        except Exception as error:
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
