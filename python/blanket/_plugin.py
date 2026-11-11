"""Format-specific constructors over the native image decoder."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._blanket import UnidentifiedImageError
from .Image import Image, open

if TYPE_CHECKING:
    import os
    from typing import BinaryIO


class PluginImageMeta(type):
    """Recognize native images by format and construct them through open()."""

    _format: str

    def __instancecheck__(cls, instance: object) -> bool:
        """Check if instance is an Image with matching format."""
        return isinstance(instance, Image) and instance.format == cls._format

    def __call__(cls, fp: str | bytes | os.PathLike[str] | os.PathLike[bytes] | BinaryIO, filename: str | bytes | None = None) -> Image:
        """Open an image using the class format."""
        try:
            image = open(fp, formats=[cls._format])
        except UnidentifiedImageError as error:
            raise SyntaxError(getattr(cls, "_error_message", f"not a {cls._format} file")) from error
        if filename is not None:
            image.filename = filename
        return image


class PluginImage(Image, metaclass=PluginImageMeta):
    """Base for format-specific image constructors."""

    _format = ""
