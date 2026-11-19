"""Keep mapped buffers live while native operations use owned snapshots."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from ._blanket import _Image, frombuffer_copy as _native_frombuffer_copy, mapped_getpixel
from ._progress import validate_progress
from .Image import Image, PixelAccess, _pixel_coordinates

if TYPE_CHECKING:
    from collections.abc import Buffer

    import numpy as np

    from ._typing import ProgressCallback
    from .ImagePalette import ImagePalette


@dataclass(frozen=True)
class BufferMap:
    """Retain external pixel storage and the layout needed for native reads."""

    data: memoryview
    mode: str
    size: tuple[int, int]
    stride: int
    orientation: int
    bit_depth: int = 8

    def snapshot(self, palette: ImagePalette | None) -> _Image:
        """Create an owned native raster snapshot from the mapped buffer."""
        native = _native_frombuffer_copy("L" if self.mode == "P" else self.mode, self.size, self.data, self.bit_depth, self.mode, self.stride, self.orientation)
        if palette is not None:
            native.set_palette(palette.mode, palette.tobytes())
        return native


class BufferPixelAccess(PixelAccess):
    """Pixel access interface for images backed by a mapped buffer."""

    def __init__(self, mapping: BufferMap, palette: ImagePalette | None) -> None:
        """Initialize pixel access for a mapped buffer."""
        self._mapping = mapping
        self._palette = palette

    @property
    def _native(self) -> _Image:
        """Create a native raster snapshot for reading pixels."""
        return self._mapping.snapshot(self._palette)

    def __setitem__(self, xy: tuple[int, int], value: float | tuple[int | float, ...]) -> None:
        """Raise ValueError because mapped buffer pixel access is readonly."""
        raise ValueError("image is readonly")

    def __getitem__(self, xy: tuple[int, int]) -> int | float | tuple[int | float, ...]:
        """Read one mapped pixel without taking a raster snapshot."""
        if not isinstance(xy, tuple):
            raise TypeError("argument must be sequence of length 2")
        mapping = self._mapping
        return mapped_getpixel(mapping.data, mapping.mode, mapping.size, mapping.stride, mapping.orientation, _pixel_coordinates(xy), mapping.bit_depth)


class MappedImage(Image):
    """An image backed by a mapped memory buffer."""

    def __init__(self, mapping: BufferMap) -> None:
        """Initialize a mapped image backed by a buffer."""
        self._mapping: BufferMap | None = None
        depth = mapping.bit_depth if mapping.bit_depth != 8 else None
        super().__init__(_Image._mapped_layout(mapping.mode, mapping.size, depth))
        self._mapping = mapping
        self.readonly = 1

    @property
    def mode(self) -> str:
        """Return the pixel mode without copying mapped pixels."""
        if self._mapping is not None:
            if self.palette is not None:
                return "P"
            if self._mapping.mode == "P":
                return "L"
        return self._raster.mode

    @property
    def bit_depth(self) -> int:
        """Return sample depth without copying mapped pixels."""
        return self._raster.bit_depth

    @property
    def size(self) -> tuple[int, int]:
        """Return dimensions without copying mapped pixels."""
        return self._raster.size

    @property
    def width(self) -> int:
        """Return width without copying mapped pixels."""
        return self._raster.width

    @property
    def height(self) -> int:
        """Return height without copying mapped pixels."""
        return self._raster.height

    @property
    def format(self) -> str | None:
        """Return format without copying mapped pixels."""
        return self._raster.format

    @property
    def _native(self) -> _Image:
        """Return the current native raster snapshot."""
        if self._mapping is not None:
            return self._mapping.snapshot(self.palette)
        return self._raster

    @_native.setter
    def _native(self, native: _Image) -> None:
        """Detach buffer mapping and set a new native raster."""
        self._mapping = None
        self._raster = native

    def _ensure_mutable(self) -> None:
        """Detach from buffer mapping and ensure raster is mutable."""
        if self._mapping is not None:
            self._native = self._native
            self.readonly = 0
        super()._ensure_mutable()

    def _check_open(self) -> None:
        """Check mapped raster lifetime without taking a pixel snapshot."""
        if self._mapping is None:
            self._raster.load()

    def load(self, *, progress: ProgressCallback | None = None, show_progress: bool = False) -> PixelAccess:
        """Load pixel access interface for the mapped buffer."""
        if self._mapping is not None:
            validate_progress(progress, show_progress=show_progress)
            return BufferPixelAccess(self._mapping, self.palette)
        return super().load(progress=progress, show_progress=show_progress)

    def _sync_palette(self) -> None:
        if self._mapping is None:
            super()._sync_palette()

    def getpixel(self, xy: tuple[int, int] | list[int]) -> int | tuple[int, ...]:
        """Read a pixel directly from a live buffer mapping."""
        mapping = self._mapping
        if mapping is None:
            return super().getpixel(xy)
        return mapped_getpixel(mapping.data, mapping.mode, mapping.size, mapping.stride, mapping.orientation, _pixel_coordinates(xy), mapping.bit_depth)

    def buffer_view(self) -> memoryview:
        """Return a read-only view retaining physical mapped rows and padding."""
        if self._mapping is None:
            return super().buffer_view()
        return self._mapping.data.toreadonly()

    def array_view(self) -> np.ndarray:
        """Return mapped samples with their row padding and orientation retained."""
        if self._mapping is None:
            return super().array_view()
        import numpy as np

        mapping = self._mapping
        shape, typestr = self._array_layout()
        sample_bytes = np.dtype(typestr).itemsize
        pixel_bytes = sample_bytes * len(self.getbands())
        row_stride = mapping.stride * mapping.orientation
        strides = (row_stride, pixel_bytes) if len(shape) == 2 else (row_stride, pixel_bytes, sample_bytes)
        offset = (self.height - 1) * mapping.stride if mapping.orientation == -1 and self.height else 0
        array = np.ndarray(shape, dtype=typestr, buffer=self.buffer_view(), offset=offset, strides=strides)
        # Keep the read-only buffer export as the base even if the original owner is writable.
        return np.asarray(memoryview(array))

    def close(self) -> None:
        """Release buffer mapping and close native raster."""
        self._mapping = None
        super().close()
        self._raster.close()

    def frombytes(self, data: object, decoder_name: str = "raw", *args: object) -> None:
        """Load pixel data from bytes into the mapped buffer or native raster."""
        mapping = self._mapping
        if mapping is None or mapping.data.readonly:
            super().frombytes(data, decoder_name, *args)
            return
        from .Image import frombytes

        replacement = frombytes(self.mode, self.size, data, decoder_name, *args, bit_depth=self.bit_depth)
        replacement._native._write_mapped_rows(mapping.data, mapping.mode, mapping.stride, mapping.orientation)


def from_array_view(obj: object, mode: str | None, bit_depth: int | None) -> Image:
    """Map a compatible contiguous array without conversion or pixel allocation."""
    if bit_depth is not None and (isinstance(bit_depth, bool) or not isinstance(bit_depth, int)):
        raise TypeError("bit_depth must be an integer")
    view = memoryview(cast("Buffer", obj))
    shape = view.shape
    if not view.c_contiguous or view.ndim not in (2, 3) or shape is None or min(shape) <= 0:
        raise ValueError("copy=False requires a nonempty C-contiguous HW or HWC buffer")
    channels = 1 if view.ndim == 2 else shape[2]
    code = view.format.lstrip("@=<>!")
    big_endian = view.format.startswith((">", "!")) or (not view.format.startswith("<") and sys.byteorder == "big")
    if code == "B" and view.itemsize == 1:
        inferred = {1: "L", 2: "LA", 3: "RGB", 4: "RGBA"}.get(channels)
        depth = 8
    elif code == "H" and view.itemsize == 2:
        if channels == 1 and bit_depth is None:
            inferred = "I;16B" if big_endian else "I;16L"
        else:
            inferred = {1: "L", 3: "RGB", 4: "RGBA"}.get(channels)
        depth = 16 if bit_depth is None else bit_depth
    elif code in ("i", "f") and view.itemsize == 4 and not big_endian:
        if code == "i":
            inferred = "I" if channels == 1 else None
        else:
            inferred = {1: "F", 3: "RGB;F", 4: "RGBA;F"}.get(channels)
        depth = 32
    else:
        raise ValueError("copy=False requires uint8, uint16, little-endian int32, or little-endian float32 samples")
    mode = inferred if mode is None else mode
    compatible = {"B": {"L": 1, "LA": 2, "RGB": 3, "RGBA": 4, "RGBX": 4, "CMYK": 4, "P": 1}, "H": {"L": 1, "RGB": 3, "RGBA": 4, "I;16": 1, "I;16L": 1, "I;16B": 1, "I;16N": 1}, "i": {"I": 1}, "f": {"F": 1, "RGB;F": 3, "RGBA;F": 4}}
    if mode is None or compatible[code].get(mode) != channels:
        raise ValueError("copy=False mode must match the array sample type and channel count")
    if (bit_depth is not None and bit_depth != depth) or (code == "H" and depth not in (10, 12, 16)):
        raise ValueError("bit_depth does not match the array sample type")
    if code == "H":
        target_big = mode == "I;16B" or (mode == "I;16N" and sys.byteorder == "big")
        if big_endian != target_big or (mode.startswith("I;") and depth != 16):
            raise ValueError("copy=False requires matching sample byte order and depth")
    width, height = shape[1], shape[0]
    raw = view.cast("B")
    mapping = BufferMap(raw, mode, (width, height), width * channels * view.itemsize, 1, depth)
    return MappedImage(mapping)
