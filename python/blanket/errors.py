"""Structured failures from image codecs.

File access errors and ordinary API argument errors retain their usual Python
types. Unrecognized input retains UnidentifiedImageError. Recognized codec
failures expose their format and operation without requiring message parsing.
"""

from __future__ import annotations

from ._blanket import UnidentifiedImageError

__all__ = ["CodecError", "CodecResourceLimitError", "CorruptImageError", "MissingBackendError", "UnsupportedCodecError"]


class CodecError(OSError):
    """A codec failure, with an optional format and operation name."""

    def __init__(self, message: str, *, format: str | None = None, operation: str | None = None) -> None:
        """Initialize a codec error with an optional format and operation identifier."""
        super().__init__(message)
        self.format = format
        self.operation = operation


class UnsupportedCodecError(CodecError, ValueError):
    """The codec does not support the requested feature or image layout."""


class CorruptImageError(CodecError, UnidentifiedImageError):
    """Recognized image data is malformed or truncated."""


class MissingBackendError(CodecError, UnidentifiedImageError):
    """A required codec library, plugin, or executable is unavailable."""


class CodecResourceLimitError(CodecError, UnidentifiedImageError):
    """Decoding would exceed a pixel, memory, or codec resource limit."""
