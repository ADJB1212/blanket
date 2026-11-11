"""Keep mapped buffers live while native operations use owned snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ._blanket import _Image, frombuffer_copy as _native_frombuffer_copy, mapped_getpixel
from .Image import Image, PixelAccess, _pixel_coordinates

if TYPE_CHECKING:
    from .ImagePalette import ImagePalette


@dataclass(frozen=True)
class BufferMap:
    """Retain external pixel storage and the layout needed for native reads."""

    data: memoryview
    mode: str
    size: tuple[int, int]
    stride: int
    orientation: int

    def snapshot(self, palette: ImagePalette | None) -> _Image:
        """Create an owned native raster snapshot from the mapped buffer."""
        native = _native_frombuffer_copy("L" if self.mode == "P" else self.mode, self.size, self.data, 8, self.mode, self.stride, self.orientation)
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

    def __init__(self, mapping: BufferMap) -> None:
        """Initialize a mapped image backed by a buffer."""
        self._mapping: BufferMap | None = None
        super().__init__(_Image._mapped_layout(mapping.mode, mapping.size))
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

    def load(self) -> PixelAccess:
        """Load pixel access interface for the mapped buffer."""
        if self._mapping is not None:
            return BufferPixelAccess(self._mapping, self.palette)
        return super().load()

    def _sync_palette(self) -> None:
        if self._mapping is None:
            super()._sync_palette()

    def getpixel(self, xy: tuple[int, int] | list[int]) -> int | tuple[int, ...]:
        """Read a pixel directly from a live buffer mapping."""
        mapping = self._mapping
        if mapping is None:
            return super().getpixel(xy)
        return mapped_getpixel(mapping.data, mapping.mode, mapping.size, mapping.stride, mapping.orientation, _pixel_coordinates(xy))

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
        replacement._native._write_mapped_rows(mapping.data, mapping.mode, mapping.stride, mapping.orientation)
