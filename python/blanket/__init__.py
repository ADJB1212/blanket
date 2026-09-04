"""A focused, Rust-backed subset of Pillow's image API."""

from . import Image
from ._blanket import UnidentifiedImageError, __version__

__all__ = ["Image", "UnidentifiedImageError", "__version__"]
