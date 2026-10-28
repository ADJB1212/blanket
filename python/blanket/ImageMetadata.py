"""Metadata editing, retention, physical units, and embedded previews."""

from __future__ import annotations

__lazy_modules__ = {"contextlib", "os", "io", "pathlib", "collections.abc"}

import contextlib
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from typing import BinaryIO, Literal

from . import Image
from ._blanket import _container_previews, _inspect, _rewrite_metadata
from ._exif import read_metadata
from ._ifd import Exif
from ._io import write_all
from ._metadata_policy import MetadataPolicy
from ._resolution import Resolution, parse_resolution
from .PngImagePlugin import PngInfo

__all__ = ["EmbeddedPreview", "MetadataPolicy", "Resolution", "get_previews", "get_resolution", "rewrite"]

type _File = str | bytes | os.PathLike[str] | os.PathLike[bytes] | BinaryIO
type _Bytes = bytes | bytearray | memoryview
_TEXT_ALIASES = {"XML:com.adobe.xmp", "Raw profile type exif"}


class _Unchanged:
    """Sentinel indicating an unchanged metadata category in rewrite()."""


_UNCHANGED = _Unchanged()


def _read(fp: _File) -> bytes:
    """Read encoded file bytes, restoring seekable stream positions."""
    position = None
    if hasattr(fp, "read"):
        with contextlib.suppress(AttributeError, OSError):
            position = fp.tell()
    try:
        return Image._read_bytes(fp)
    finally:
        if position is not None:
            with contextlib.suppress(AttributeError, OSError):
                cast("BinaryIO", fp).seek(position)


def _load(fp: _File) -> tuple[bytes, str, dict[object, object]]:
    """Inspect input bytes to determine format and read container metadata."""
    data = _read(fp)
    format, _, _, _ = _inspect(data, None)
    info = read_metadata(data, "HEIF" if format == "AVIF" else format)
    raw_exif = info.get("Raw profile type exif")
    if "exif" not in info and isinstance(raw_exif, str):
        with contextlib.suppress(ValueError):
            info["exif"] = bytes.fromhex("".join(raw_exif.split("\n")[3:]) if "\n" in raw_exif else raw_exif)
    return data, format, info


def _bytes(value: object, name: str) -> bytes:
    """Validate and convert metadata values to raw bytes."""
    if value is None:
        return b""
    if name == "exif" and isinstance(value, Exif):
        return value.tobytes()
    if name == "xmp" and isinstance(value, str):
        return value.encode("utf-8")
    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise TypeError(f"{name} must be bytes" + (", text, or None" if name == "xmp" else ", an Exif object, or None" if name == "exif" else " or None"))
    return bytes(value)


def _text(info: Mapping[object, object]) -> list[tuple[bytes, bytes, bool]]:
    """Convert text metadata mapping to PNG ancillary text chunks."""
    png = PngInfo()
    for key, value in info.items():
        if isinstance(key, str) and isinstance(value, str) and key not in _TEXT_ALIASES:
            png.add_text(key, value)
    return png.chunks


def _dpi(value: object) -> tuple[float, float] | None:
    """Validate and normalize a pair of positive finite numbers for DPI."""
    if value is None:
        return None
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise TypeError("dpi must contain two positive finite numbers or be None")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in value):
        raise TypeError("dpi must contain two positive finite numbers")
    result = float(value[0]), float(value[1])
    if any(not math.isfinite(v) or v <= 0 for v in result):
        raise ValueError("dpi must contain two positive finite numbers")
    return result


def rewrite(
    fp: _File,
    output: _File | None = None,
    *,
    exif: Exif | _Bytes | _Unchanged | None = _UNCHANGED,
    xmp: str | _Bytes | _Unchanged | None = _UNCHANGED,
    icc_profile: _Bytes | _Unchanged | None = _UNCHANGED,
    text: Mapping[str, str] | _Unchanged | None = _UNCHANGED,
    dpi: tuple[float, float] | _Unchanged | None = _UNCHANGED,
    policy: MetadataPolicy | None = None,
) -> bytes:
    """Rewrite JPEG, PNG/APNG, WebP, or JPEG XL metadata without decoding pixels.

    Omitted categories are preserved; None removes a category. text replaces
    PNG text (XMP remains separate). dpi synchronizes EXIF and JPEG/PNG density;
    None removes physical density. Policy filtering applies after explicit edits.
    JPEG XL ICC changes require re-encoding and are rejected. TIFF/HEIF/AVIF
    rewrites are unsupported. Unknown ancillary data is retained.

    Reads paths or streams, restoring seekable source positions. Returns encoded
    bytes and optionally writes them to a path or stream after all edits succeed.
    Input bytes denote a path; wrap encoded bytes in BytesIO.

    Args:
        fp: Source path or binary stream.
        output: Optional destination path or stream (distinct from the input stream).
        exif: Replacement EXIF bytes or Exif object, or None to remove.
        xmp: Replacement XMP bytes/text, or None to remove.
        icc_profile: Replacement profile, or None to remove.
        text: Replacement PNG text mapping, or None to remove text.
        dpi: Physical resolution in dots per inch, or None to remove.
        policy: Optional retention policy, applied after explicit replacements.

    Examples:
        ```python
        from blanket import ImageMetadata

        ImageMetadata.rewrite("photo.jpg", "public.jpg", policy=ImageMetadata.MetadataPolicy(location=False))
        ```
    """
    if policy is not None and not isinstance(policy, MetadataPolicy):
        raise TypeError("policy must be a MetadataPolicy or None")
    if output is fp and hasattr(fp, "read"):
        raise ValueError("input and output streams must be distinct")
    data, format, info = _load(fp)
    original_icc = info.get("icc_profile")
    if format not in ("JPEG", "PNG", "WEBP", "JXL"):
        raise ValueError(f"metadata-only rewriting is not supported for {format}")
    if not isinstance(text, _Unchanged) and format != "PNG":
        raise ValueError("text replacement is supported only for PNG")
    edits = {"exif": exif, "xmp": xmp, "icc_profile": icc_profile}
    for key, value in edits.items():
        if not isinstance(value, _Unchanged):
            info[key] = _bytes(value, key)
    replace_text = not isinstance(text, _Unchanged)
    if replace_text:
        info = {k: v for k, v in info.items() if not (isinstance(v, str) and k not in _TEXT_ALIASES)}
        if text is not None:
            if not isinstance(text, Mapping) or any(not isinstance(k, str) or not isinstance(v, str) or k in _TEXT_ALIASES | {"xmp", "exif", "icc_profile"} for k, v in text.items()):
                raise TypeError("text must map PNG keywords to strings; EXIF and XMP use separate arguments")
            info.update(text)
    new_dpi = None
    remove_resolution = False
    if not isinstance(dpi, _Unchanged):
        new_dpi = _dpi(dpi)
        remove_resolution = dpi is None
        values = Exif()
        raw = info.get("exif")
        if isinstance(raw, bytes) and raw:
            values.load(raw)
        if new_dpi is None:
            for tag in (282, 283, 296):
                values.pop(tag, None)
        else:
            values[282], values[283], values[296] = new_dpi[0], new_dpi[1], 2
        info["exif"] = values.tobytes() if values or values.thumbnail is not None else b""
    if policy is not None:
        info = policy.apply(info)
    exif_changed = (
        not isinstance(exif, _Unchanged) or not isinstance(dpi, _Unchanged) or (policy is not None and (not policy.exif or not policy.location or not policy.previews or policy.exif_tags is not None))
    )
    xmp_changed = not isinstance(xmp, _Unchanged) or (policy is not None and (not policy.xmp or not policy.location or policy.xmp_fields is not None))
    icc_changed = not isinstance(icc_profile, _Unchanged) or (policy is not None and not policy.icc_profile)
    replace_text |= policy is not None and (not policy.text or not policy.location or policy.text_keys is not None)
    if new_dpi is not None and format in ("WEBP", "JXL") and policy is not None and not policy.exif:
        raise ValueError(f"{format} DPI requires retaining EXIF")
    exif_value = _bytes(info.get("exif"), "exif") if exif_changed else None
    xmp_value = _bytes(info.get("xmp", info.get("XML:com.adobe.xmp")), "xmp") if xmp_changed else None
    icc_value = _bytes(info.get("icc_profile"), "icc_profile") if icc_changed else None
    if format == "JXL" and icc_changed and icc_value == original_icc:
        icc_value = None
    encoded = _rewrite_metadata(data, format, exif_value, xmp_value, icc_value, _text(info) if replace_text else None, new_dpi, remove_resolution)
    if output is not None:
        if hasattr(output, "write"):
            write_all(cast("BinaryIO", output), encoded)
        else:
            Path(os.fsdecode(os.fspath(output))).write_bytes(encoded)
    return encoded


def get_resolution(fp: _File, *, conflict: Literal["container", "exif", "error"] = "container") -> Resolution:
    """Normalize JPEG JFIF, PNG pHYs, BMP, and EXIF/TIFF physical units.

    DPI always means samples per inch; cm densities multiply by 2.54 and
    meter densities by 0.0254. Pixel aspect is physical pixel width/height
    (vertical density divided by horizontal density). Unitless densities supply
    aspect only. Physical densities take priority over unitless densities.
    Conflicts beyond 0.1 percent prefer container or EXIF, or raise on 'error'.
    Missing/invalid densities return dpi None and square pixels.

    Args:
        fp: Source path or binary stream.
        conflict: Conflict rule: container, exif, or error.

    Examples:
        ```python
        from blanket import ImageMetadata

        resolution = ImageMetadata.get_resolution("photo.jpg")
        print(resolution.dpi, resolution.pixel_aspect)
        ```
    """
    data, format, info = _load(fp)
    return parse_resolution(data, format, info, conflict)


@dataclass(frozen=True)
class EmbeddedPreview:
    """Independent encoded preview bytes with their format, size, and origin.

    HEIF/AVIF thumbnails are decoded independently and returned as lossless PNG;
    other previews retain their encoded bytes. WMF previews have no parsed size.
    """

    data: bytes
    format: str
    size: tuple[int, int] | None
    source: str

    def open(self) -> Image.Image:
        """Decode this preview as an independent image.

        Examples:
            ```python
            from blanket import ImageMetadata

            previews = ImageMetadata.get_previews("photo.jpg")
            if previews:
                image = previews[0].open()
            ```
        """
        return Image.open(BytesIO(self.data))


def get_previews(fp: _File) -> tuple[EmbeddedPreview, ...]:
    """Extract EXIF JPEG thumbnails, HEIF/AVIF thumbnails, and EPS previews.

    Primary image pixels are never decoded. Only HEIF/AVIF thumbnail pixels are
    decoded, under the per-image and cumulative pixel budgets. Missing previews
    return an empty tuple. Malformed referenced previews raise an error.

    Args:
        fp: Source path or binary stream.

    Examples:
        ```python
        from blanket import ImageMetadata

        for preview in ImageMetadata.get_previews("photo.jpg"):
            print(preview.source, preview.size)
        ```
    """
    data, format, info = _load(fp)
    result = []
    raw = info.get("exif")
    if isinstance(raw, bytes) and raw:
        exif = Exif()
        exif.load(raw)
        if exif.thumbnail:
            header = Image.inspect(BytesIO(exif.thumbnail))
            result.append(EmbeddedPreview(exif.thumbnail, header.format, header.size, "exif"))
        elif exif.get_ifd(1).get(514):
            raise ValueError("invalid EXIF thumbnail bounds")
    for name, encoded in _container_previews(data, format):
        size = None if name == "WMF" else Image.inspect(BytesIO(encoded)).size
        result.append(EmbeddedPreview(encoded, name, size, "container"))
    return tuple(result)
