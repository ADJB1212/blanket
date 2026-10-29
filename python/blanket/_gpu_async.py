"""Completion events, stream ordering, and transfer worker ownership."""

from __future__ import annotations

__lazy_modules__ = {"types", "collections.abc"}

from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import ExitStack
from functools import lru_cache
from threading import local
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import TracebackType
    from typing import Any, Self


@lru_cache(maxsize=1)
def _executor() -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=2, thread_name_prefix="blanket-transfer")


def submit[T](operation: Callable[[], T], *, device: str) -> Future[T]:
    stream = None
    if device.startswith("cuda:"):
        from ._gpu_cuda import cupy

        cp = cupy()
        index = int(device.split(":")[1])
        with cp.cuda.Device(index):
            stream = cp.cuda.get_current_stream()

    def run() -> T:
        if stream is None:
            return operation()
        with cp.cuda.Device(index), stream:
            return operation()

    return _executor().submit(run)


class Event:
    """Retain a completion point and its inputs independently of image handles."""

    def __init__(self, resource: object, runtime: str, device: str) -> None:
        self._resource: Any = resource
        self.runtime = runtime
        self.device = device
        self._event = None
        if runtime == "torch":
            from .ImageGpu import _torch

            self._event = _torch().mps.Event()
            self._event.record()

    def done(self) -> bool:
        if self.runtime == "native":
            return self._resource.done()
        if self.runtime == "torch":
            return self._event.query()
        from ._gpu_cuda import cupy

        with cupy().cuda.Device(self._resource.device):
            return self._resource.event.query()

    def wait(self) -> None:
        if self.runtime == "torch":
            self._event.synchronize()
        else:
            self._resource.synchronize()


class Stream:
    """CUDA execution stream, or the native Metal ordered queue."""

    def __init__(self, backend: str = "metal", *, device: int = 0) -> None:
        from .ImageGpu import _require

        if backend not in ("metal", "cuda") or type(device) is not int or device < 0 or (backend == "metal" and device != 0):
            raise ValueError("invalid GPU stream backend or device")
        _require(backend, device)
        self.backend = backend
        self.device = device
        self._stream: Any = None
        self._local = local()
        if backend == "cuda":
            from ._gpu_cuda import cupy

            with cupy().cuda.Device(device):
                self._stream = cupy().cuda.Stream(non_blocking=True)

    def __enter__(self) -> Self:
        if self._stream is not None:
            from ._gpu_cuda import cupy

            stack = ExitStack()
            try:
                stack.enter_context(cupy().cuda.Device(self.device))
                stack.enter_context(self._stream)
            except BaseException:
                stack.close()
                raise
            if not hasattr(self._local, "stacks"):
                self._local.stacks = []
            self._local.stacks.append(stack)
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: TracebackType | None) -> None:
        if self._stream is not None:
            self._local.stacks.pop().__exit__(exc_type, exc, traceback)

    def wait_event(self, event: Event) -> None:
        expected = f"cuda:{self.device}" if self.backend == "cuda" else "metal"
        if event.device != expected or (self.backend == "metal" and event.runtime != "native"):
            raise ValueError("stream and event must use the same execution context")
        if self._stream is not None:
            from ._gpu_cuda import cupy

            with cupy().cuda.Device(self.device):
                self._stream.wait_event(event._resource.event)
        # Native images are already committed to the same ordered Metal queue.

    def synchronize(self) -> None:
        if self._stream is not None:
            from ._gpu_cuda import cupy

            with cupy().cuda.Device(self.device):
                self._stream.synchronize()
        else:
            from . import _blanket

            _blanket._gpu_synchronize()
