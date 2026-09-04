"""A focused, Rust-backed subset of Pillow's image API."""

from . import Image
from ._blanket import __version__

__all__ = ["Image", "__version__"]
