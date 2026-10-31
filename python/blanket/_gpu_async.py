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
    """Return the shared two-worker pool used for CPU/GPU transfers."""
    return ThreadPoolExecutor(max_workers=2, thread_name_prefix="blanket-transfer")


def submit[T](operation: Callable[[], T], *, device: str) -> Future[T]:
    """Schedule an operation while retaining the caller's CUDA stream context.

    Args:
        operation: Callable executed on a transfer worker.
        device: "metal", "mps", or "cuda:<index>"; CUDA captures its current
            device stream before submission.

    Returns:
        A Future propagating the callable's result or exception.
    """
    stream = None
    if device.startswith("cuda:"):
        from ._gpu_cuda import cupy

        cp = cupy()
        index = int(device.split(":")[1])
        with cp.cuda.Device(index):
            stream = cp.cuda.get_current_stream()

    def run() -> T:
        """Execute the operation under the captured CUDA device and stream, if any."""
        if stream is None:
            return operation()
        with cp.cuda.Device(index), stream:
            return operation()

    return _executor().submit(run)


class Event:
    """A retained GPU completion point independent of an image handle.

    Use an image or reduction's record_event method to create events. done
    polls without waiting; wait blocks the CPU; Stream.wait_event orders later
    work in a compatible execution context.
    """

    def __init__(self, resource: object, runtime: str, device: str) -> None:
        """Retain a backend resource and capture its completion point.

        Args:
            resource: Native Metal image, MPS tensor, or CUDA image resource.
            runtime: "native", "torch", or "cuda".
            device: Identifier of the resource's execution device.

        PyTorch MPS records an event on the current stream; other runtimes use
        the retained resource's existing completion point.
        """
        self._resource: Any = resource
        self.runtime = runtime
        self.device = device
        self._event = None
        if runtime == "torch":
            from .ImageGpu import _torch

            self._event = _torch().mps.Event()
            self._event.record()

    def done(self) -> bool:
        """Check completion without blocking the CPU.

        Returns:
            True once the retained GPU work has completed, otherwise False.
        """
        if self.runtime == "native":
            return self._resource.done()
        if self.runtime == "torch":
            return self._event.query()
        from ._gpu_cuda import cupy

        with cupy().cuda.Device(self._resource.device):
            return self._resource.event.query()

    def wait(self) -> None:
        """Block the CPU until this completion point finishes.

        Returns:
            None. Pixels remain resident; backend execution errors propagate.
        """
        if self.runtime == "torch":
            self._event.synchronize()
        else:
            self._resource.synchronize()


class Stream:
    """A CUDA stream context or the shared native Metal ordered queue.

    CUDA contexts select a nonblocking stream on the requested device and
    restore the caller's context on exit. Metal contexts use the existing
    ordered native queue. Exiting a context does not synchronize it.
    """

    def __init__(self, backend: str = "metal", *, device: int = 0) -> None:
        """Create an execution context for resident work.

        Args:
            backend: "metal" or "cuda".
            device: Nonnegative CUDA device index; Metal supports only 0.

        Raises:
            ValueError: The backend or device is invalid.
            RuntimeError: The requested backend is unavailable.
        """
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
        """Select this CUDA stream/device, or retain the native Metal queue context."""
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
        """Restore the previous CUDA stream/device without waiting for completion."""
        if self._stream is not None:
            self._local.stacks.pop().__exit__(exc_type, exc, traceback)

    def wait_event(self, event: Event) -> None:
        """Order subsequent stream work after a completion event.

        Args:
            event: Event from the same device and compatible runtime. Native
                Metal requires a native event on its shared ordered queue.

        Returns:
            None. CUDA enqueues a GPU-side wait without blocking the CPU; native
            Metal already guarantees the required queue order.

        Raises:
            ValueError: Stream and event execution contexts differ.
        """
        expected = f"cuda:{self.device}" if self.backend == "cuda" else "metal"
        if event.device != expected or (self.backend == "metal" and event.runtime != "native"):
            raise ValueError("stream and event must use the same execution context")
        if self._stream is not None:
            from ._gpu_cuda import cupy

            with cupy().cuda.Device(self.device):
                self._stream.wait_event(event._resource.event)
        # Native images are already committed to the same ordered Metal queue.

    def synchronize(self) -> None:
        """Block the CPU until the stream's previously submitted work completes.

        Returns:
            None. Metal waits for its shared native queue; CUDA waits for this
            stream. Backend execution errors propagate.
        """
        if self._stream is not None:
            from ._gpu_cuda import cupy

            with cupy().cuda.Device(self.device):
                self._stream.synchronize()
        else:
            from . import _blanket

            _blanket._gpu_synchronize()
