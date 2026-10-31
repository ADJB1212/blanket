"""CUDA plan scheduling and bounded per-device resource caches."""

from __future__ import annotations

from collections import OrderedDict
from threading import RLock
from typing import Any

from . import _blanket, _gpu_cuda
from .ImageGpu import _dispatch_threads


class _Cache:
    """Manage bounded scratch and immutable control buffers for one CUDA device."""

    def __init__(self, device: int) -> None:
        """Initialize per-device locks, byte limits, and empty resource caches."""
        self.device = device
        self.lock = RLock()
        self.image_limit = 128 * 1024 * 1024
        self.control_limit = 16 * 1024 * 1024
        self.scratch: list[Any] = []
        self.controls: OrderedDict[tuple[str, bytes], Any] = OrderedDict()
        self.scratch_bytes = 0
        self.control_bytes = 0

    def info(self) -> tuple[int, int, int, int]:
        """Return retained scratch bytes/limit and control bytes/limit, in that order."""
        with _gpu_cuda.cupy().cuda.Device(self.device):
            _gpu_cuda._collect_pending(self.device)
        with self.lock:
            return (self.scratch_bytes, self.image_limit, self.control_bytes, self.control_limit)

    def configure(self, image_bytes: int, control_bytes: int) -> None:
        """Replace byte limits and discard cached scratch and control entries."""
        with _gpu_cuda.cupy().cuda.Device(self.device):
            _gpu_cuda._collect_pending(self.device)
        with self.lock:
            self.image_limit, self.control_limit = image_bytes, control_bytes
            self.scratch.clear()
            self.controls.clear()
            self.scratch_bytes = self.control_bytes = 0

    def take(self, size: int) -> object:
        """Remove an exact-size scratch buffer from the cache or allocate new storage."""
        with self.lock:
            for i, array in enumerate(self.scratch):
                if array.nbytes == size:
                    self.scratch_bytes -= size
                    return self.scratch.pop(i)
        cp = _gpu_cuda.cupy()
        # Direct allocations keep this cache independent of CuPy's global pool.
        memory = cp.cuda.MemoryPointer(cp.cuda.Memory(size), 0)
        return cp.ndarray((size,), dtype=cp.uint8, memptr=memory)

    def put(self, arrays: tuple[Any, ...]) -> None:
        """Retain completed scratch buffers while respecting byte and entry limits."""
        with self.lock:
            for array in arrays:
                if array.nbytes > self.image_limit:
                    continue
                while self.scratch and (len(self.scratch) >= 16 or self.scratch_bytes + array.nbytes > self.image_limit):
                    self.scratch_bytes -= self.scratch.pop(0).nbytes
                self.scratch.append(array)
                self.scratch_bytes += array.nbytes

    def control(self, values: list[int] | list[float], dtype: str) -> _gpu_cuda.CudaImage:
        """Reuse or upload an immutable CUDA control array.

        Args:
            values: Packed numeric control values used to construct the cache key.
            dtype: NumPy/CuPy sample dtype name, such as uint32 or float32.

        Returns:
            A CudaImage holding the uploaded controls. Cache hits order the
            current stream after the control resource's upload event.
        """
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
                while self.controls and (len(self.controls) >= 64 or self.control_bytes + len(key[1]) * 2 > self.control_limit):
                    old_key, old_value = self.controls.popitem(last=False)
                    self.control_bytes -= len(old_key[1]) + old_value.array.nbytes
                self.controls[key] = value
                self.control_bytes += len(key[1]) + value.array.nbytes
            return value


_CACHES: dict[int, _Cache] = {}
_LOCK = RLock()


def cache(device: int) -> _Cache:
    """Return the shared bounded cache for a CUDA device, creating it if needed."""
    with _LOCK:
        if device not in _CACHES:
            _CACHES[device] = _Cache(device)
        return _CACHES[device]


class _ScratchLease:
    """Hold plan scratch buffers until the associated completion event finishes."""

    def __init__(self, state: _Cache, scratch: tuple[Any, ...]) -> None:
        """Retain scratch buffers and their cache until completion tracking releases them."""
        self.state, self.scratch = state, scratch

    def __del__(self) -> None:
        """Return scratch after the completion queue releases this lease."""
        self.state.put(self.scratch)


class CudaPlan:
    """Execute a prepared tiled unary pipeline using cached CUDA controls and scratch."""

    def __init__(self, width: int, height: int, channels: int, floating: bool, steps: list[tuple[int, int, int, list[float]]], memory_limit: int, device: int) -> None:
        """Prepare a same-size tiled CUDA execution schedule.

        Args:
            width: Input/output width in pixels.
            height: Input/output height in pixels.
            channels: Interleaved channel count.
            floating: True for float32 working samples, False for uint8.
            steps: Validated protocol steps as (operation, kernel, axis, values).
            memory_limit: Byte budget for the pair of tile scratch buffers.
            device: CUDA device index for kernels, parameters, and controls.

        Raises:
            ValueError: Plan validation fails or its schedule exceeds 65536
                dispatches; a larger tile memory budget can reduce dispatch count.
        """
        steps, self._info = _blanket._gpu_prepare_plan(width, height, channels, floating, steps, memory_limit)
        self.device = device
        self.shape = (height, width, channels)
        self.floating = floating
        self.state = cache(device)
        self.schedule: list[tuple[int, int, int]] = []
        parameters: list[int] = []
        _, tile, halo, _ = self._info
        direct = tuple(tile) == (width, height)
        self.scratch_count = int(len(steps) > 1) if direct else 2
        tiles = ((width + tile[0] - 1) // tile[0]) * ((height + tile[1] - 1) // tile[1])
        if tiles * (len(steps) + 2) > 65_536:
            raise ValueError("CUDA plan exceeds 65536 dispatches; increase memory_limit or reduce steps")
        for y in range(0, height, tile[1]):
            for x in range(0, width, tile[0]):
                w, h = min(tile[0], width - x), min(tile[1], height - y)
                left, top = max(0, x - halo[0]), max(0, y - halo[1])
                tw, th = min(width, x + w + halo[0]) - left, min(height, y + h + halo[1]) - top
                if not direct or not steps:
                    self.schedule.append((-1, -1 if direct else 0, -1))
                    parameters.extend([width, height, channels, tw, th, left, top, 0, 0, tw, th])
                for i, (op, kernel, axis, _) in enumerate(steps):
                    if direct:
                        last = (len(steps) - i) % 2 == 1
                        src = -1 if i == 0 else (0 if last else -2)
                        self.schedule.append((src, -1 if last else 0, i))
                    else:
                        self.schedule.append((i % 2, (i + 1) % 2, i))
                    parameters.extend([tw, th, channels, tw, th, op, kernel, axis, 0, 0, 0])
                if not direct:
                    self.schedule.append((len(steps) % 2, -1, -1))
                    parameters.extend([tw, th, channels, width, height, x - left, y - top, x, y, w, h])
        self.counts = [
            parameters[i + 9] * parameters[i + 10] * channels if entry[2] == -1 else _dispatch_threads(parameters[i : i + 8], channels, floating)
            for i, entry in zip(range(0, len(parameters), 11), self.schedule, strict=True)
        ]
        cp = _gpu_cuda.cupy()
        with cp.cuda.Device(device):
            self.parameters = self.state.control(parameters, "uint32")
            self.rows = tuple(self.parameters.array.reshape((-1, 11)))
            self.controls = [self.state.control(values or [0.0], "float32") for _, _, _, values in steps]
            self.kernel = _gpu_cuda.library(device)("image_f32" if floating else "image_u8")
            self.tile_kernel = _gpu_cuda.library(device)("tile_f32" if floating else "tile_u8")

    def info(self) -> tuple[int, list[int], list[int], int]:
        """Return fused step count, tile dimensions, halo dimensions, and scratch bytes."""
        steps, tile, halo, scratch_bytes = self._info
        return steps, tile, halo, scratch_bytes // 2 * self.scratch_count

    def run_batch(self, images: list[Any]) -> list[Any]:
        """Submit resident CUDA inputs through the prepared tile schedule.

        Args:
            images: Up to 1024 CUDA resources matching the plan's device, shape,
                and sample dtype; an empty list returns an empty list.

        Returns:
            Independent CudaImage results in input order, sharing a completion
            event and a scratch lease. Successful submission does not wait.

        Raises:
            ValueError: A batch is too large or resources differ in device/shape.
            RuntimeError: Execution preparation or dispatch fails; the stream
                is synchronized before temporary resources are released.
        """
        if not images:
            return []
        if len(images) > 1024 or any(image.device != self.device or image.array.shape != self.shape for image in images):
            raise ValueError("CUDA plan inputs must match device and shape (at most 1024 images)")
        cp = _gpu_cuda.cupy()
        with cp.cuda.Device(self.device):
            stream = cp.cuda.get_current_stream()
            scratch: tuple[Any, ...] | None = None
            try:
                for image in [*images, self.parameters, *self.controls]:
                    stream.wait_event(image.event)
                dtype = cp.float32 if self.floating else cp.uint8
                if any(image.array.dtype != dtype for image in images):
                    raise ValueError("CUDA plan input sample type must match its prepared mode")
                inputs = [cp.ascontiguousarray(image.array) for image in images]
                outputs = [cp.empty(self.shape, dtype=dtype) for _ in images]
                scratch = tuple(self.state.take(self._info[3] // 2) for _ in range(self.scratch_count))
                buffers = tuple(array.view(dtype) for array in scratch)
                for source, output in zip(inputs, outputs, strict=True):
                    for row, (src, dst, step), count in zip(self.rows, self.schedule, self.counts, strict=True):
                        source_buffer = source if src == -1 else (output if src == -2 else buffers[src])
                        destination = output if dst == -1 else buffers[dst]
                        if step == -1:
                            self.tile_kernel(((count + 127) // 128,), (128,), (source_buffer, destination, row), stream=stream)
                        else:
                            self.kernel(((count + 127) // 128,), (128,), (source_buffer, destination, source_buffer, row, self.controls[step].array), stream=stream)
                event = cp.cuda.Event()
                event.record(stream)
                lease = _ScratchLease(self.state, scratch)
                scratch = None
                owners = (self, lease, *images, *inputs)
                return [_gpu_cuda.CudaImage(output, owners=owners, completion=event) for output in outputs]
            except Exception as error:
                stream.synchronize()
                if scratch is not None:
                    self.state.put(scratch)
                raise RuntimeError(f"CUDA plan execution failed on device {self.device}: {error}") from error
