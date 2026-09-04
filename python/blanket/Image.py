"""Pillow-shaped image entry points backed by Blanket's Rust extension."""

from ._blanket import Image, frombytes as _frombytes


def frombytes(mode: str, size: tuple[int, int], data: object) -> Image:
    """Create an image from tightly packed 8-bit pixel data."""

    try:
        raw = bytes(data)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise TypeError("data must be a bytes-like object") from error
    return _frombytes(mode, size, raw)


__all__ = ["Image", "frombytes"]
