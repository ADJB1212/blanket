"""Validation and native adaptation for multi-frame saves."""

from __future__ import annotations

__lazy_modules__ = {"os", "collections.abc"}

import math
from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import os
    from typing import BinaryIO

from ._blanket import _encode_sequence, _write_metadata
from ._exif import save_metadata
from ._io import write_all
from .errors import UnsupportedCodecError
from .Image import Image, _bounded_int, _save_options


def save_sequence(image: Image, fp: str | bytes | os.PathLike[str] | os.PathLike[bytes] | BinaryIO, format: str, options: dict[str, object], compressor: object) -> bool:
    """Consume sequence options and save frames, returning False for ordinary saves."""
    save_all = options.pop("save_all", False)
    if not isinstance(save_all, bool):
        raise TypeError("save_all must be a bool")
    appended = options.pop("append_images", ())
    if not save_all and not {"duration", "loop", "disposal", "blend", "default_image"} & options.keys():
        return False
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
        supplied = controls.get(name)
        if isinstance(supplied, (list, tuple)):
            if len(supplied) != count:
                raise ValueError(f"{name} must have one value per animation frame")
            result = list(supplied)
        elif supplied is not None:
            result = [supplied] * count
        else:
            result = [frame.info.get("duration", default) if name == "duration" else default for frame in frames[start:]]
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
    for frame in frames:
        if frame.mode in ("HSV", "LAB", "F", "YCbCr") or (frame.mode == "CMYK" and format != "PDF"):
            raise UnsupportedCodecError(f"cannot write mode {frame.mode} as {format}", format=format, operation="encode")
        if format == "PNG" and frame.mode in ("I", "I;16", "I;16L", "I;16B"):
            frame = frame.convert("L", bit_depth=16)
        if format == "PDF":
            target = frame.mode if frame.mode in ("L", "RGB", "RGBA", "CMYK") else "RGBA" if frame.has_transparency_data else "L" if frame.mode in ("1", "I", "I;16", "I;16L", "I;16B") else "RGB"
        else:
            target = "RGBA"
        if frame.bit_depth != 8 and format != "PNG" and frame.mode not in ("I", "I;16", "I;16L", "I;16B"):
            raise UnsupportedCodecError("high-bit-depth images require explicit conversion for this format", format=format, operation="encode")
        prepared.append(frame.convert(target) if frame.mode != target else frame)
    encoded = bytes(_encode_sequence([frame._native for frame in prepared], format, durations, disposals, blends, loop, default_image, values["quality"], values["compress_level"], values["lossless"]))
    if any(metadata):
        encoded = _write_metadata(encoded, format, *metadata)
    if hasattr(fp, "write"):
        write_all(fp, encoded)
    else:
        with open(fp, "wb") as output:
            output.write(encoded)
    return True
