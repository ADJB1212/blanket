"""Runtime capabilities and backend versions for the installed Blanket build."""

from __future__ import annotations

import platform
import sys
import warnings
from typing import TYPE_CHECKING

from ._blanket import _feature_info

if TYPE_CHECKING:
    from typing import TextIO

__all__ = ["blanketinfo", "check", "get_supported", "version"]

_FEATURES = ("blanket", "png", "jpg", "jxl", "libtiff", "webp", "avif", "libheif", "heif", "heif_decoder", "heif_encoder", "bmp", "gif", "ico", "pdf", "littlecms2", "libjpeg_turbo")
_ALIASES = {"jpeg": "jpg", "jpegxl": "jxl", "tiff": "libtiff", "heic": "heif"}
_FORMATS = (
    ("BMP", "bmp"),
    ("GIF", "gif"),
    ("ICO", "ico"),
    ("PNG", "png"),
    ("JPEG", "jpg"),
    ("JXL", "jxl"),
    ("TIFF", "libtiff"),
    ("WEBP", "webp"),
    ("AVIF", "avif"),
    ("HEIF/HEIC", "heif"),
    ("PDF", "pdf"),
)


def _info(feature: str) -> tuple[bool, str | None] | None:
    if not isinstance(feature, str):
        raise TypeError("feature must be a string")
    return _feature_info(_ALIASES.get(feature, feature))


def check(feature: str) -> bool:
    """Check whether the installed build supports a feature.

    Args:
        feature: A feature name, such as ``jpg``, ``jxl``, or ``heif``.

    Returns:
        Whether the feature is available. HEIF requires both an HEVC decoder
        and encoder; query ``heif_decoder`` or ``heif_encoder`` separately
        for one direction. PDF availability means writing only. Unknown
        names emit ``UserWarning`` and return False.

    Examples:
        >>> from blanket import features
        >>> features.check("png")
        True
    """
    info = _info(feature)
    if info is None:
        warnings.warn(f"Unknown feature '{feature}'.", stacklevel=2)
        return False
    return info[0]


def version(feature: str) -> str | None:
    """Return the installed backend's version, if available.

    Args:
        feature: A name accepted by :func:`check`.

    Returns:
        A native library version for JXL, WebP, libheif, or LittleCMS;
        a labeled Rust backend crate version for the other codecs; or
        None for unavailable, unknown, or unversioned features. JPEG reports
        the turbojpeg Rust wrapper, not the underlying libjpeg-turbo version.

    Examples:
        >>> from blanket import features
        >>> features.version("blanket") is not None
        True
    """
    info = _info(feature)
    return info[1] if info is not None and info[0] else None


def get_supported() -> list[str]:
    """List the available canonical feature names.

    Returns:
        Available feature names, without aliases.

    Examples:
        >>> from blanket import features
        >>> "png" in features.get_supported()
        True
    """
    return [feature for feature in _FEATURES if check(feature)]


def blanketinfo(out: TextIO | None = None, supported_formats: bool = True) -> None:
    """Print build, backend, and format diagnostics.

    Args:
        out: Text stream to receive the report. Defaults to current stdout.
        supported_formats: Include format read/write capabilities.

    Examples:
        >>> from io import StringIO
        >>> from blanket import features
        >>> report = StringIO()
        >>> features.blanketinfo(report, supported_formats=False)
        >>> "Blanket" in report.getvalue()
        True
    """
    if out is None:
        out = sys.stdout
    info = {feature: _info(feature) or (False, None) for feature in _FEATURES}
    print(f"Blanket {info['blanket'][1]}", file=out)
    print(f"Python {platform.python_version()}", file=out)
    print(f"Platform {platform.platform()}", file=out)
    print("Backend support:", file=out)
    for feature, (available, backend_version) in info.items():
        status = "available" if available else "unavailable"
        suffix = f" ({backend_version})" if backend_version is not None else ""
        print(f"  {feature}: {status}{suffix}", file=out)
    if supported_formats:
        print("Formats (single image; metadata is not preserved on save):", file=out)
        for label, feature in _FORMATS:
            read = write = info[feature][0]
            if feature == "heif":
                read = info["heif_decoder"][0]
                write = info["heif_encoder"][0]
            elif feature == "pdf":
                read = False
            capabilities = [name for name, available in (("open", read), ("save", write)) if available]
            print(f"  {label}: {', '.join(capabilities) or 'unavailable'}", file=out)
