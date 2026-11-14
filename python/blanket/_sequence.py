"""Validation and native adaptation for multi-frame saves."""

from __future__ import annotations

__lazy_modules__ = {"os", "collections.abc", "math", "itertools"}

import math
import os
from collections.abc import Iterable
from itertools import islice, repeat
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._typing import Writable

from ._blanket import _encode_sequence, _png_frame_plan
from ._exif import save_metadata
from ._io import write_bytes
from .errors import UnsupportedCodecError
from .Image import Image, _bounded_int, _save_options


def _png_regions(
    frames: list[Image], durations: list[float | int], disposals: list[float | int], blends: list[float | int], default_image: bool, duration_known: bool
) -> tuple[list[Image], list[float], list[int], list[int], tuple[tuple[int, int], list[tuple[int, int]]]]:
    """Adapt APNG deltas and repeated frames using native raster operations."""
    for frame in frames:
        frame._sync_palette()
    canvas, records = _png_frame_plan([frame._native for frame in frames], durations, disposals, blends, [frame.info.get("transparency") for frame in frames], default_image, duration_known)
    prepared = frames[: int(default_image)]
    offsets = [(0, 0)] * int(default_image)
    for position, bbox, _, _, _ in records:
        frame = frames[position]
        prepared.append(frame if bbox is None or bbox == (0, 0, *frame.size) else frame.crop(bbox))
        offsets.append((0, 0) if bbox is None else bbox[:2])
    if prepared[0].size != canvas:
        prepared[0] = prepared[0].crop((0, 0, *canvas))
    return (prepared, [0.0] * int(default_image) + [record[2] for record in records], [0] * int(default_image) + [record[3] for record in records], [0] * int(default_image) + [record[4] for record in records], (canvas, offsets))


def save_sequence(image: Image, fp: str | bytes | os.PathLike[str] | os.PathLike[bytes] | Writable | None, format: str, options: dict[str, object], compressor: object) -> bytes | None:
    """Encode or save frames, returning None for ordinary saves."""
    save_all = options.pop("save_all", False)
    if not isinstance(save_all, bool):
        raise TypeError("save_all must be a bool")
    appended = options.pop("append_images", ())
    if not save_all and not {"duration", "loop", "disposal", "blend", "default_image"} & options.keys():
        return None
    if format not in ("GIF", "WEBP", "PNG", "PDF"):
        raise UnsupportedCodecError(f"multi-frame saving is not supported for {format}", format=format, operation="encode")
    if compressor is not None:
        raise UnsupportedCodecError("compression optimization is not supported for multi-frame saves", format=format, operation="encode")
    allowed = {"GIF": {"duration", "loop", "disposal"}, "WEBP": {"duration", "loop"}, "PNG": {"duration", "loop", "disposal", "blend", "default_image"}, "PDF": set()}[format]
    animation_keys = {"duration", "loop", "disposal", "blend", "default_image"}
    unsupported = options.keys() & (animation_keys - allowed)
    if unsupported:
        raise TypeError(f"unsupported {format} save option(s): {', '.join(sorted(unsupported))}")
    controls = {key: options.pop(key) for key in allowed if key in options}
    metadata = save_metadata(image, format, options)
    values = _save_options(format, options)
    sources = [image]
    if save_all:
        if not isinstance(appended, Iterable):
            raise TypeError("append_images must be an iterable of images")
        sources.extend(appended)
        if any(not isinstance(source, Image) for source in sources):
            raise TypeError("append_images must contain images")
        frames: list[Image] = []
        for source in sources:
            source._native.load()
            source._sync_palette()
            if source._lazy_frames is not None:
                from .ImageSequence import all_frames

                frames.extend(all_frames(source))
            elif source._frames is None:
                frames.append(source)
            else:
                current = source._frames[source._frame]
                current._native, current.palette, current._info = source._native, source.palette, source._info
                current._bands = source._bands
                frames.extend(source._frames)
    else:
        frames = [image]
    default_image = controls.get("default_image", image.info.get("default_image", False)) if format == "PNG" else False
    if not isinstance(default_image, bool):
        raise TypeError("default_image must be a bool")
    start = int(default_image)
    count = len(frames) - start
    if count <= 0:
        raise ValueError("default_image requires at least one animation frame")

    def array(name: str, default: int, maximum: int | None = None) -> list[float | int]:
        """Normalize a frame control, padding a separate APNG default image with zero."""
        supplied = controls.get(name, image.info.get(name) if format == "PNG" and name != "duration" else None)
        if isinstance(supplied, (list, tuple)):
            if len(supplied) != count:
                raise ValueError(f"{name} must have one value per animation frame")
            result = supplied
        elif supplied is not None:
            result = repeat(supplied, count)
        elif name == "duration":
            result = (frame.info.get("duration", default) for frame in islice(frames, start, None))
        else:
            result = repeat(default, count)
        normalized: list[float | int] = [0] * start
        for value in result:
            if maximum is not None:
                normalized.append(_bounded_int(name, value, 0, maximum))
            else:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise TypeError(f"{name} must contain numbers")
                if not math.isfinite(value) or value < 0:
                    raise ValueError(f"{name} must be finite and nonnegative")
                normalized.append(float(value))
        return normalized

    durations = array("duration", 0)
    disposals = array("disposal", 2 if format == "GIF" else 0, 3 if format == "GIF" else 2)
    blends = array("blend", 0, 1)
    loop = controls.get("loop", image.info.get("loop"))
    if loop is not None:
        loop = _bounded_int("loop", loop, 0, 65535 if format in ("GIF", "WEBP") else 0x7FFFFFFF)
    prepared: list[Image] = []
    png_mode = "RGBA"
    png_palette = None
    if format == "PNG" and all(frame.bit_depth == 8 for frame in frames):
        modes = {frame.mode for frame in frames}
        png_mode = next((mode for mode in ("RGBA", "RGB", "P", "LA", "L", "1") if mode in modes), "RGBA")
        indexed = [frame for frame in frames if frame.mode == "P"]
        if indexed:
            png_palette = indexed[0].palette
            palette_values = indexed[0].getpalette()
            if any(frame.getpalette() != palette_values for frame in indexed[1:]):
                png_mode = "RGBA" if any(frame.palette is not None and frame.palette.mode == "RGBA" for frame in indexed) else "RGB"
    for source in frames:
        frame = source
        if frame.mode in ("HSV", "LAB", "F", "YCbCr", "La", "RGBa", "RGBX") or (frame.mode == "CMYK" and format != "PDF"):
            raise UnsupportedCodecError(f"cannot write mode {frame.mode} as {format}", format=format, operation="encode")
        if format == "PNG" and frame.mode in ("I", "I;16", "I;16L", "I;16B", "I;16N"):
            frame = frame.convert("L", bit_depth=16)
        if format == "PDF":
            target = frame.mode if frame.mode in ("L", "RGB", "RGBA", "CMYK") else "RGBA" if frame.has_transparency_data else "L" if frame.mode in ("1", "I", "I;16", "I;16L", "I;16B", "I;16N") else "RGB"
        elif format == "PNG":
            target = png_mode
        else:
            target = "RGBA"
        if frame.bit_depth != 8 and format != "PNG" and frame.mode not in ("I", "I;16", "I;16L", "I;16B", "I;16N"):
            raise UnsupportedCodecError("high-bit-depth images require explicit conversion for this format", format=format, operation="encode")
        prepared.append(frame.convert(target) if frame.mode != target else frame)
    regions = None
    if format == "PNG" and prepared[0].bit_depth == 8:
        prepared, durations, disposals, blends, regions = _png_regions(prepared, durations, disposals, blends, default_image, "duration" in controls or any("duration" in frame.info for frame in frames))
        if png_mode == "P" and png_palette is not None:
            prepared[0] = prepared[0].copy()
            prepared[0].putpalette(png_palette)
            transparency = image.info.get("transparency")
            if isinstance(transparency, int):
                prepared[0]._native.set_palette_alpha(index=transparency)
            elif isinstance(transparency, bytes):
                prepared[0]._native.set_palette_alpha(alpha=transparency)
    output = None if fp is None or hasattr(fp, "write") else os.fspath(fp)
    encoded = _encode_sequence(
        [frame._native for frame in prepared],
        format,
        durations,
        disposals,
        blends,
        loop,
        default_image,
        values["quality"],
        values["compress_level"],
        values["lossless"],
        regions,
        values.get("pdf_dpi", 72.0),
        output=output,
        metadata_exif=metadata[0],
        metadata_xmp=metadata[1],
        icc_profile=metadata[2],
        metadata_chunks=metadata[3],
    )
    if encoded is None:
        return b""
    if fp is None:
        return encoded
    write_bytes(fp, encoded)
    return encoded
