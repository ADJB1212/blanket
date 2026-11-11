"""Describe Python conversion and binary output contracts for API adapters."""

from collections.abc import Buffer, Iterable
from typing import Protocol, SupportsBytes, SupportsFloat, SupportsIndex

type FloatInput = str | Buffer | SupportsFloat | SupportsIndex
type BytesInput = Iterable[SupportsIndex] | SupportsIndex | SupportsBytes | Buffer


class Writable(Protocol):
    """Describe the binary write operation used by image encoders."""

    def write(self, data: bytes, /) -> int | None:
        """Write bytes and report progress, or None when the stream would block."""
        ...
