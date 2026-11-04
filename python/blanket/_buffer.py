"""Keep mapped buffers live while native operations use owned snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from operator import index
from typing import TYPE_CHECKING

from ._blanket import frombytes as _native_frombytes, mapped_getpixel
from .Image import Image, PixelAccess

if TYPE_CHECKING:
    from ._blanket import _Image
    from .ImagePalette import ImagePalette


@dataclass(frozen=True)
class BufferMap:
    data: memoryview
    mode: str
    size: tuple[int, int]
    stride: int
    orientation: int

    def snapshot(self, palette: ImagePalette | None) -> _Image:
        """Create an owned native raster snapshot from the mapped buffer."""
        native = _native_frombytes("L" if self.mode == "P" else self.mode, self.size, self.data.tobytes(), 8, self.mode, self.stride, self.orientation)
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
        mapping = self._mapping
        return mapped_getpixel(mapping.data, mapping.mode, mapping.size, mapping.stride, mapping.orientation, self._coordinates(xy))


class MappedImage(Image):
    """An image backed by a mapped memory buffer."""

    def __init__(self, native: _Image, mapping: BufferMap) -> None:
        """Initialize a mapped image backed by a buffer."""
        self._mapping: BufferMap | None = None
        super().__init__(native)
        self._mapping = mapping
        self.readonly = 1

    @property
    def _native(self) -> _Image:
        """Return the current native raster snapshot."""
        if self._mapping is not None:
            self._raster = self._mapping.snapshot(self.palette)
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
        super()._ensure_mutable()

    def load(self) -> PixelAccess:
        """Load pixel access interface for the mapped buffer."""
        if self._mapping is not None:
            self._native.load()
            return BufferPixelAccess(self._mapping, self.palette)
        return super().load()

    def getpixel(self, xy: tuple[int, int] | list[int]) -> int | tuple[int, ...]:
        """Read a pixel directly from a live buffer mapping."""
        mapping = self._mapping
        if mapping is None:
            return super().getpixel(xy)
        if not isinstance(xy, (tuple, list)):
            raise TypeError("argument must be a sequence")
        if len(xy) != 2:
            raise TypeError("argument must be sequence of length 2")
        coordinates = tuple(int(value) if isinstance(value, float) else index(value) for value in xy)
        return mapped_getpixel(mapping.data, mapping.mode, mapping.size, mapping.stride, mapping.orientation, coordinates)

    def close(self) -> None:
        """Release buffer mapping and close native raster."""
        self._mapping = None
        super().close()

    def frombytes(self, data: object, decoder_name: str = "raw", *args: object) -> None:
        """Load pixel data from bytes into the mapped buffer or native raster."""
        mapping = self._mapping
        if mapping is None or mapping.data.readonly:
            super().frombytes(data, decoder_name, *args)
            return
        from .Image import frombytes

        replacement = frombytes(self.mode, self.size, data, decoder_name, *args, bit_depth=self.bit_depth)
        encoded = replacement.tobytes("raw", mapping.mode, 0, mapping.orientation)
        row_bytes = len(encoded) // mapping.size[1] if mapping.size[1] else 0
        for row in range(mapping.size[1]):
            mapping.data[row * mapping.stride : row * mapping.stride + row_bytes] = encoded[row * row_bytes : (row + 1) * row_bytes]
