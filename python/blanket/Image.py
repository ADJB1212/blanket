"""Pillow-shaped image entry points backed by Blanket's Rust extension."""

from __future__ import annotations

import builtins
import os
from enum import IntEnum
from pathlib import Path
from typing import BinaryIO

from ._blanket import _Image, open_bytes
from ._blanket import fromarray as _native_fromarray
from ._blanket import frombytes as _native_frombytes

_EXTENSIONS = {".png": "PNG", ".jpg": "JPEG", ".jpeg": "JPEG", ".jxl": "JXL"}


class Resampling(IntEnum):
    """Pillow-compatible resampling filter identifiers."""

    NEAREST = 0
    LANCZOS = 1
    BILINEAR = 2
    BICUBIC = 3
    BOX = 4
    HAMMING = 5


NEAREST, LANCZOS, BILINEAR, BICUBIC, BOX, HAMMING = Resampling


class Image:
    """An 8-bit image whose pixels and supported operations live in Rust."""

    def __init__(self, native: _Image) -> None:
        self._native = native
        self._info: dict[object, object] = {}

    @property
    def mode(self) -> str:
        return self._native.mode

    @property
    def size(self) -> tuple[int, int]:
        return self._native.size

    @property
    def width(self) -> int:
        return self._native.width

    @property
    def height(self) -> int:
        return self._native.height

    @property
    def format(self) -> str | None:
        return self._native.format

    @property
    def info(self) -> dict[object, object]:
        return self._info

    def load(self) -> None:
        """Validate that this eagerly loaded image remains open."""

        self._native.load()

    def close(self) -> None:
        self._native.close()

    def convert(self, mode: str) -> Image:
        """Return a new image converted to `L`, `RGB`, or `RGBA`."""

        return Image(self._native.convert(mode))

    def tobytes(self) -> bytes:
        return self._native.tobytes()

    def to_pillow(self) -> object:
        """Return an equivalent Pillow image when Pillow is installed."""

        try:
            from PIL import Image as PillowImage
        except ImportError as error:
            raise ImportError("Pillow is required for to_pillow(); install blanket[test] or Pillow") from error
        return PillowImage.frombytes(self.mode, self.size, self.tobytes())

    def save(self, fp: str | bytes | os.PathLike[str] | os.PathLike[bytes] | BinaryIO, format: str | None = None, **options: object) -> None:
        """Save this image as PNG, JPEG, or JPEG XL."""

        output_format = _output_format(fp, format)
        values = _save_options(output_format, options)
        encoded = self._native._encode(output_format, **values)
        _write_bytes(fp, encoded)

    def __enter__(self) -> Image:
        self.load()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"<blanket.Image.Image image mode={self.mode} size={self.width}x{self.height}>"


def open(fp: str | bytes | os.PathLike[str] | os.PathLike[bytes] | BinaryIO, mode: str = "r", formats: list[str] | tuple[str, ...] | None = None) -> Image:
    """Open and eagerly decode a supported image."""

    if mode != "r":
        raise ValueError(f"bad mode {mode!r}")
    if formats is not None and not isinstance(formats, (list, tuple)):
        raise TypeError("formats must be a list or tuple")
    data = _read_bytes(fp)
    image = Image(open_bytes(data, formats))
    from ._exif import read_metadata

    image.info.update(read_metadata(data, image.format))
    return image


def frombytes(mode: str, size: tuple[int, int], data: object) -> Image:
    """Create an image from tightly packed 8-bit pixel data."""

    try:
        raw = bytes(data)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise TypeError("data must be a bytes-like object") from error
    return Image(_native_frombytes(mode, size, raw))


def fromarray(obj: object, mode: str | None = None) -> Image:
    """Create an image from an 8-bit object exposing the array interface."""

    return Image(_native_fromarray(obj, mode))


def _read_bytes(fp: str | bytes | os.PathLike[str] | os.PathLike[bytes] | BinaryIO) -> bytes:
    if hasattr(fp, "read"):
        stream = fp
        try:
            stream.seek(0)  # type: ignore[union-attr]
        except (AttributeError, OSError):
            pass
        data = stream.read()  # type: ignore[union-attr]
        if not isinstance(data, bytes):
            raise ValueError("binary data must be used to open an image")
        return data
    return Path(os.fsdecode(os.fspath(fp))).read_bytes()


def _write_bytes(fp: str | bytes | os.PathLike[str] | os.PathLike[bytes] | BinaryIO, data: bytes) -> None:
    if hasattr(fp, "write"):
        fp.write(data)  # type: ignore[union-attr]
        return
    with builtins.open(fp, "wb") as output:
        output.write(data)


def _output_format(fp: object, requested: str | None) -> str:
    if requested is not None:
        normalized = requested.upper().replace(" ", "")
        aliases = {"JPG": "JPEG", "JPEGXL": "JXL"}
        normalized = aliases.get(normalized, normalized)
        if normalized not in {"PNG", "JPEG", "JXL"}:
            raise ValueError(f"unsupported image format {requested!r}")
        return normalized
    if hasattr(fp, "write"):
        raise ValueError("unknown file extension; specify format")
    extension = os.path.splitext(os.fsdecode(os.fspath(fp)))[1].lower()
    try:
        return _EXTENSIONS[extension]
    except KeyError as error:
        raise ValueError(f"unknown file extension: {extension}") from error


def _save_options(format: str, supplied: dict[str, object]) -> dict[str, object]:
    allowed = {"PNG": {"compress_level"}, "JPEG": {"quality"}, "JXL": {"quality", "lossless", "effort"}}[format]
    unknown = supplied.keys() - allowed
    if unknown:
        names = ", ".join(sorted(unknown))
        raise TypeError(f"unsupported {format} save option(s): {names}")

    lossless = supplied.get("lossless", False)
    if not isinstance(lossless, bool):
        raise TypeError("lossless must be a bool")
    quality = 100 if lossless else _bounded_int("quality", supplied.get("quality", 90), 1, 100)
    compress_level = _bounded_int("compress_level", supplied.get("compress_level", 6), 0, 9)
    effort = _bounded_int("effort", supplied.get("effort", 7), 1, 10)

    return {"quality": quality, "compress_level": compress_level, "lossless": lossless, "effort": effort}


def _bounded_int(name: str, value: object, minimum: int, maximum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an int")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


__all__ = ["Image", "Resampling", "NEAREST", "LANCZOS", "BILINEAR", "BICUBIC", "BOX", "HAMMING", "fromarray", "frombytes", "open"]
