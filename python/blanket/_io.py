"""Binary stream output helpers."""

from __future__ import annotations

import errno
from typing import BinaryIO


def write_all(stream: BinaryIO, data: bytes) -> None:
    """Write every byte, rejecting invalid counts and streams that would block."""
    view = memoryview(data)
    offset = 0
    pending = data
    while pending:
        count = stream.write(pending)
        if count is None:
            raise BlockingIOError(errno.EAGAIN, "image output write made no progress")
        if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= len(pending):
            raise OSError("write() returned an invalid byte count")
        if count == 0:
            raise BlockingIOError(errno.EAGAIN, "image output write made no progress")
        offset += count
        pending = bytes(view[offset : offset + 64 * 1024])
