"""Measure per-operation peak allocations separately from timed runs."""

from __future__ import annotations

import gc
import sys
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType


def memory_profiler() -> ModuleType:
    """Load the development profiler without adding a runtime dependency."""
    if sys.platform not in ("darwin", "linux"):
        raise OSError(f"allocation tracking is unavailable on {sys.platform}; Memray supports macOS and Linux")
    try:
        import memray
    except ImportError as error:
        raise OSError("allocation tracking requires Memray; run uv sync --group test") from error
    return memray


def measure_memory(operation: Callable[[], object], iterations: int) -> dict[str, float]:
    """Return median, p25, and p75 peak allocated bytes over independent runs.

    Track Python and native allocations, including returned data and transient
    buffers. Input fixtures and allocations made before each run are excluded.
    """
    if iterations < 1:
        raise ValueError("iterations must be positive")
    profiler = memory_profiler()
    samples: list[int] = []
    with TemporaryDirectory(prefix="blanket-memory-") as directory:
        destination = profiler.FileDestination(Path(directory) / "allocations.bin", overwrite=True, compress_on_exit=False)
        for _ in range(iterations):
            gc.collect()
            with profiler.Tracker(destination=destination, trace_python_allocators=True, file_format=profiler.FileFormat.AGGREGATED_ALLOCATIONS):
                result = operation()
            del result
            with profiler.FileReader(destination.path) as reader:
                samples.append(reader.metadata.peak_memory)
    samples.sort()
    return {"median_bytes": median(samples), "p25_bytes": samples[len(samples) // 4], "p75_bytes": samples[(len(samples) * 3) // 4]}
