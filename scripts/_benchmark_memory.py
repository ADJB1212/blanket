"""Sample native and Python resident memory separately from timed runs."""

from __future__ import annotations

import ctypes
import gc
import os
import sys
from threading import Event, Thread
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

SAMPLE_INTERVAL = 0.001


def rss_reader() -> Callable[[], int]:
    """Return a process RSS reader in bytes for the current platform."""
    if sys.platform == "darwin":
        proc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        proc.proc_pidinfo.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_int]
        proc.proc_pidinfo.restype = ctypes.c_int

        def read_mac() -> int:
            # proc_taskinfo: six uint64 counters followed by twelve int32 fields.
            info = (ctypes.c_uint64 * 12)()
            if proc.proc_pidinfo(os.getpid(), 4, 0, info, ctypes.sizeof(info)) != ctypes.sizeof(info):
                raise OSError(ctypes.get_errno(), "cannot read process RSS")
            return info[1]

        return read_mac
    if sys.platform.startswith("linux"):
        page_size = os.sysconf("SC_PAGE_SIZE")

        def read_linux() -> int:
            with open("/proc/self/statm", encoding="ascii") as status:
                return int(status.read().split()[1]) * page_size

        return read_linux
    if sys.platform == "win32":

        class Counters(ctypes.Structure):
            _fields_ = [("size", ctypes.c_ulong), ("faults", ctypes.c_ulong), ("peak", ctypes.c_size_t), ("rss", ctypes.c_size_t), ("remaining", ctypes.c_size_t * 6)]

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        process = kernel.GetCurrentProcess()
        api = ctypes.WinDLL("psapi", use_last_error=True)
        api.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong]
        api.GetProcessMemoryInfo.restype = ctypes.c_int

        def read_windows() -> int:
            info = Counters()
            info.size = ctypes.sizeof(info)
            if not api.GetProcessMemoryInfo(process, ctypes.byref(info), info.size):
                raise ctypes.WinError(ctypes.get_last_error())
            return info.rss

        return read_windows
    raise OSError(f"RSS measurement is unavailable on {sys.platform}")


def measure_memory(operation: Callable[[], object]) -> dict[str, int]:
    """Return sampled peak RSS and its increase over the pre-operation RSS.

    Sampling can miss brief temporaries. Allocator reuse and resident fixtures
    affect the increase; it is not an allocation count or an isolated peak.
    """
    read_rss = rss_reader()
    gc.collect()
    ready, stopped = Event(), Event()
    sampled_peak = 0
    errors: list[Exception] = []

    def sample() -> None:
        nonlocal sampled_peak
        try:
            sampled_peak = read_rss()
            ready.set()
            while not stopped.wait(SAMPLE_INTERVAL):
                sampled_peak = max(sampled_peak, read_rss())
        except Exception as error:
            errors.append(error)
            ready.set()

    sampler = Thread(target=sample, daemon=True)
    sampler.start()
    ready.wait()
    try:
        if errors:
            raise errors[0]
        baseline = read_rss()
        result = operation()
        final_rss = read_rss()
    finally:
        stopped.set()
        sampler.join()
    if errors:
        raise errors[0]
    peak = max(baseline, sampled_peak, final_rss)
    del result
    return {"peak_rss_bytes": peak, "rss_increase_bytes": peak - baseline}
