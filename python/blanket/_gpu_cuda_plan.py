"""CUDA plan scheduling and bounded per-device resource caches."""

from __future__ import annotations

from collections import OrderedDict
from threading import RLock
from typing import Any

from . import _blanket, _gpu_cuda
from .ImageGpu import _dispatch_threads


class _Cache:
    def __init__(self, device: int) -> None:
        self.device = device
        self.lock = RLock()
        self.image_limit = 128 * 1024 * 1024
        self.control_limit = 16 * 1024 * 1024
        self.scratch: list[Any] = []
        self.controls: OrderedDict[tuple[str, bytes], Any] = OrderedDict()

    def info(self) -> tuple[int, int, int, int]:
        with self.lock:
            return (sum(array.nbytes for array in self.scratch), self.image_limit, sum(len(key[1]) + value.array.nbytes for key, value in self.controls.items()), self.control_limit)

    def configure(self, image_bytes: int, control_bytes: int) -> None:
        with self.lock:
            self.image_limit, self.control_limit = image_bytes, control_bytes
            self.scratch.clear()
            self.controls.clear()

    def take(self, size: int) -> object:
        with self.lock:
            for i, array in enumerate(self.scratch):
                if array.nbytes == size:
                    return self.scratch.pop(i)
        cp = _gpu_cuda.cupy()
        # Direct allocations keep this cache independent of CuPy's global pool.
        memory = cp.cuda.MemoryPointer(cp.cuda.Memory(size), 0)
        return cp.ndarray((size,), dtype=cp.uint8, memptr=memory)

    def put(self, arrays: tuple[Any, Any]) -> None:
        with self.lock:
            for array in arrays:
                if array.nbytes > self.image_limit:
                    continue
                while self.scratch and (len(self.scratch) >= 16 or sum(a.nbytes for a in self.scratch) + array.nbytes > self.image_limit):
                    self.scratch.pop(0)
                self.scratch.append(array)

    def control(self, values: list[int] | list[float], dtype: str) -> _gpu_cuda.CudaImage:
        import numpy as np

        cp = _gpu_cuda.cupy()
        key = (dtype, np.asarray(values, dtype=dtype).tobytes())
        with self.lock:
            if key in self.controls:
                value = self.controls[key]
                self.controls.move_to_end(key)
                cp.cuda.get_current_stream().wait_event(value.event)
                return value
            memory = cp.cuda.MemoryPointer(cp.cuda.Memory(len(key[1])), 0)
            array = cp.ndarray((len(values),), dtype=dtype, memptr=memory)
            host = np.frombuffer(key[1], dtype=dtype)
            array.set(host, stream=cp.cuda.get_current_stream())
            value = _gpu_cuda.CudaImage(array, owners=(key, host))
            if len(key[1]) * 2 <= self.control_limit:
                while self.controls and (len(self.controls) >= 64 or self.info()[2] + len(key[1]) * 2 > self.control_limit):
                    self.controls.popitem(last=False)
                self.controls[key] = value
            return value


_CACHES: dict[int, _Cache] = {}
_LOCK = RLock()


def cache(device: int) -> _Cache:
    with _LOCK:
        if device not in _CACHES:
            _CACHES[device] = _Cache(device)
        return _CACHES[device]


class _ScratchLease:
    def __init__(self, state: _Cache, scratch: tuple[Any, Any], event: object) -> None:
        self.state, self.scratch = state, scratch
        self.event: Any = event

    def __del__(self) -> None:
        with _gpu_cuda.cupy().cuda.Device(self.state.device):
            try:
                self.event.synchronize()
            except Exception:
                return
            self.state.put(self.scratch)


class CudaPlan:
    def __init__(self, width: int, height: int, channels: int, floating: bool, steps: list[tuple[int, int, int, list[float]]], memory_limit: int, device: int) -> None:
        steps, self._info = _blanket._gpu_prepare_plan(width, height, channels, floating, steps, memory_limit)
        self.device = device
        self.shape = (height, width, channels)
        self.floating = floating
        self.state = cache(device)
        self.schedule: list[tuple[int, int, int]] = []
        parameters: list[int] = []
        _, tile, halo, _ = self._info
        tiles = ((width + tile[0] - 1) // tile[0]) * ((height + tile[1] - 1) // tile[1])
        if tiles * (len(steps) + 2) > 65_536:
            raise ValueError("CUDA plan exceeds 65536 dispatches; increase memory_limit or reduce steps")
        for y in range(0, height, tile[1]):
            for x in range(0, width, tile[0]):
                w, h = min(tile[0], width - x), min(tile[1], height - y)
                left, top = max(0, x - halo[0]), max(0, y - halo[1])
                tw, th = min(width, x + w + halo[0]) - left, min(height, y + h + halo[1]) - top
                self.schedule.append((-1, 0, -1))
                parameters.extend([width, height, channels, tw, th, left, top, 0, 0, tw, th])
                for i, (op, kernel, axis, _) in enumerate(steps):
                    self.schedule.append((i % 2, (i + 1) % 2, i))
                    parameters.extend([tw, th, channels, tw, th, op, kernel, axis, 0, 0, 0])
                self.schedule.append((len(steps) % 2, -1, -1))
                parameters.extend([tw, th, channels, width, height, x - left, y - top, x, y, w, h])
        self.counts = [
            parameters[i + 9] * parameters[i + 10] * channels if entry[2] == -1 else _dispatch_threads(parameters[i : i + 8], channels, floating)
            for i, entry in zip(range(0, len(parameters), 11), self.schedule, strict=True)
        ]
        cp = _gpu_cuda.cupy()
        with cp.cuda.Device(device):
            self.parameters = self.state.control(parameters, "uint32")
            self.rows = self.parameters.array.reshape((-1, 11))
            self.controls = [self.state.control(values or [0.0], "float32") for _, _, _, values in steps]
            self.kernel = _gpu_cuda.library(device)("image_f32" if floating else "image_u8")
            self.tile_kernel = _gpu_cuda.library(device)("tile_f32" if floating else "tile_u8")

    def info(self) -> tuple[int, list[int], list[int], int]:
        return self._info

    def run_batch(self, images: list[Any]) -> list[Any]:
        if not images:
            return []
        if len(images) > 1024 or any(image.device != self.device or image.array.shape != self.shape for image in images):
            raise ValueError("CUDA plan inputs must match device and shape (at most 1024 images)")
        cp = _gpu_cuda.cupy()
        with cp.cuda.Device(self.device):
            stream = cp.cuda.get_current_stream()
            scratch: tuple[Any, Any] | None = None
            try:
                for image in [*images, self.parameters, *self.controls]:
                    stream.wait_event(image.event)
                dtype = cp.float32 if self.floating else cp.uint8
                if any(image.array.dtype != dtype for image in images):
                    raise ValueError("CUDA plan input sample type must match its prepared mode")
                inputs = [cp.ascontiguousarray(image.array) for image in images]
                outputs = [cp.empty(self.shape, dtype=dtype) for _ in images]
                scratch = (self.state.take(self._info[3] // 2), self.state.take(self._info[3] // 2))
                buffers = (scratch[0].view(dtype), scratch[1].view(dtype))
                for source, output in zip(inputs, outputs, strict=True):
                    for row, (src, dst, step), count in zip(self.rows, self.schedule, self.counts, strict=True):
                        source_buffer = source if src == -1 else buffers[src]
                        destination = output if dst == -1 else buffers[dst]
                        if step == -1:
                            self.tile_kernel(((count + 127) // 128,), (128,), (source_buffer, destination, row), stream=stream)
                        else:
                            self.kernel(((count + 127) // 128,), (128,), (source_buffer, destination, source_buffer, row, self.controls[step].array), stream=stream)
                event = cp.cuda.Event()
                event.record(stream)
                lease = _ScratchLease(self.state, scratch, event)
                scratch = None
                owners = (self, lease, *images, *inputs)
                return [_gpu_cuda.CudaImage(output, owners=owners, completion=event) for output in outputs]
            except Exception as error:
                stream.synchronize()
                if scratch is not None:
                    self.state.put(scratch)
                raise RuntimeError(f"CUDA plan execution failed on device {self.device}: {error}") from error
