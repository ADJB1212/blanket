"""Keep mapped buffers live while native operations use owned snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ._blanket import frombytes as _native_frombytes
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
        native = _native_frombytes("L" if self.mode == "P" else self.mode, self.size, self.data.tobytes(), 8, self.mode, self.stride, self.orientation)
        if palette is not None:
            native.set_palette(palette.mode, palette.tobytes())
        return native


class BufferPixelAccess(PixelAccess):
    def __init__(self, mapping: BufferMap, palette: ImagePalette | None) -> None:
        self._mapping = mapping
        self._palette = palette

    @property
    def _native(self) -> _Image:
        return self._mapping.snapshot(self._palette)

    def __setitem__(self, xy: tuple[int, int], value: float | tuple[int | float, ...]) -> None:
        raise ValueError("image is readonly")


class MappedImage(Image):
    def __init__(self, native: _Image, mapping: BufferMap) -> None:
        self._mapping: BufferMap | None = None
        super().__init__(native)
        self._mapping = mapping
        self.readonly = 1

    @property
    def _native(self) -> _Image:
        if self._mapping is not None:
            self._raster = self._mapping.snapshot(self.palette)
        return self._raster

    @_native.setter
    def _native(self, native: _Image) -> None:
        self._mapping = None
        self._raster = native

    def _ensure_mutable(self) -> None:
        if self._mapping is not None:
            self._native = self._native
        super()._ensure_mutable()

    def load(self) -> PixelAccess:
        if self._mapping is not None:
            self._native.load()
            return BufferPixelAccess(self._mapping, self.palette)
        return super().load()

    def close(self) -> None:
        self._mapping = None
        super().close()

    def frombytes(self, data: object, decoder_name: str = "raw", *args: object) -> None:
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
