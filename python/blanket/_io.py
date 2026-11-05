"""Binary image file and stream helpers."""

from __future__ import annotations

__lazy_modules__ = {"errno", "contextlib", "os", "pathlib"}

import contextlib
import errno
import os
from pathlib import Path
from typing import TYPE_CHECKING, cast

from ._blanket import UnidentifiedImageError

if TYPE_CHECKING:
    from typing import BinaryIO


def read_bytes(fp: str | bytes | os.PathLike[str] | os.PathLike[bytes] | BinaryIO, *, restore_position: bool = False) -> bytes:
    """Read encoded bytes, rewinding streams and optionally restoring position."""
    if not hasattr(fp, "read"):
        return Path(os.fsdecode(os.fspath(fp))).read_bytes()
    stream = cast("BinaryIO", fp)
    position = None
    if restore_position:
        with contextlib.suppress(AttributeError, OSError):
            position = stream.tell()
    try:
        return _read_stream(stream)
    finally:
        if position is not None:
            with contextlib.suppress(AttributeError, OSError):
                stream.seek(position)


def _read_stream(stream: BinaryIO) -> bytes:
    """Read binary stream contents, including streams without sized reads."""
    try:
        stream.seek(0)
    except (AttributeError, OSError):
        rewind = False
    else:
        rewind = True
    try:
        head = stream.read(16)
    except TypeError:
        data = stream.read()
        if not isinstance(data, bytes):
            raise ValueError("binary data must be used to open an image") from None
        return data
    if not isinstance(head, bytes):
        raise ValueError("binary data must be used to open an image")
    if head.startswith(b"\xff\xff"):
        raise UnidentifiedImageError("cannot identify image file")
    if rewind:
        try:
            stream.seek(0)
        except (AttributeError, OSError):
            pass
        else:
            data = stream.read()
            if not isinstance(data, bytes):
                raise ValueError("binary data must be used to open an image")
            return data
    rest = stream.read()
    if not isinstance(rest, bytes):
        raise ValueError("binary data must be used to open an image")
    return head + rest


def write_bytes(fp: str | bytes | os.PathLike[str] | os.PathLike[bytes] | BinaryIO, data: bytes) -> None:
    """Write encoded bytes to a path or caller-owned stream."""
    if hasattr(fp, "write"):
        write_all(cast("BinaryIO", fp), data)
    else:
        with open(fp, "wb") as output:
            output.write(data)


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
