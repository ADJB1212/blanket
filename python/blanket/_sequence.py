"""Validation and native adaptation for multi-frame saves."""

from __future__ import annotations

__lazy_modules__ = {"os", "collections.abc", "math", "fractions", "itertools"}

import math
import os
from collections.abc import Iterable
from dataclasses import dataclass
from fractions import Fraction
from itertools import repeat
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import BinaryIO

from ._blanket import _difference_bbox, _encode_sequence
from ._exif import save_metadata
from ._io import write_bytes
from .errors import UnsupportedCodecError
from .Image import Image, _bounded_int, _save_options


@dataclass
class _PngFrame:
    image: Image
    bbox: tuple[int, int, int, int] | None
    duration: float
    disposal: int
    blend: int


def _png_regions(
    frames: list[Image], durations: list[float | int], disposals: list[float | int], blends: list[float | int], default_image: bool, duration_known: bool
) -> tuple[list[Image], list[float], list[int], list[int], tuple[tuple[int, int], list[tuple[int, int]]]]:
    """Adapt APNG deltas and repeated frames using native raster operations."""
    canvas = (max(frame.width for frame in frames), max(frame.height for frame in frames))
    records: list[_PngFrame] = []
    for position in range(int(default_image), len(frames)):
        frame = frames[position]
        bbox = None
        duration = float(durations[position])
        disposal, blend = int(disposals[position]), int(blends[position])
        if records:
            previous = records[-1]
            previous_disposal = 1 if previous.disposal == 2 and len(records) < 2 else previous.disposal
            base = records[-2].image if previous_disposal == 2 else previous.image
            clear = (previous.bbox or (0, 0, *frames[0].size)) if previous_disposal == 1 else None
            if frame.mode == base.mode and frame.mode in ("1", "L", "LA", "RGB", "RGBA") and frame.info.get("transparency") is None and base.info.get("transparency") is None:
                bbox = _difference_bbox(frame._native, base._native, clear)
            else:
                if clear is not None and base.mode != "RGBA":
                    base = base.copy()
                    base.paste(0, clear)
                    clear = None
                current = frame if frame.mode == "RGBA" else frame.convert("RGBA")
                comparison = base if base.mode == "RGBA" else base.convert("RGBA")
                bbox = _difference_bbox(current._native, comparison._native, clear)
            combined = previous.duration + duration
            if bbox is None and previous_disposal == disposal and previous.blend == blend and duration_known and math.isfinite(combined) and Fraction(combined / 1000).limit_denominator(65535).numerator <= 65535:
                previous.duration = combined
                continue
        records.append(_PngFrame(frame, bbox, duration, disposal, blend))
    prepared = frames[: int(default_image)]
    offsets = [(0, 0)] * int(default_image)
    for record in records:
        prepared.append(record.image if record.bbox is None or record.bbox == (0, 0, *record.image.size) else record.image.crop(record.bbox))
        offsets.append((0, 0) if record.bbox is None else record.bbox[:2])
    if prepared[0].size != canvas:
        prepared[0] = prepared[0].crop((0, 0, *canvas))
    return (prepared, [0.0] * int(default_image) + [record.duration for record in records], [0] * int(default_image) + [record.disposal for record in records], [0] * int(default_image) + [record.blend for record in records], (canvas, offsets))


def save_sequence(image: Image, fp: str | bytes | os.PathLike[str] | os.PathLike[bytes] | BinaryIO | None, format: str, options: dict[str, object], compressor: object) -> bytes | None:
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
            result = (frames[i].info.get("duration", default) for i in range(start, len(frames)))
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
    for frame in frames:
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
