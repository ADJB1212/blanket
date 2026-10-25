"""Progressive JPEG XL input and incremental PNG row output.

Streaming consumes encoded chunks without retaining the encoded file. Decoders
still allocate a full raster and codec working memory. PNG output requires only
row buffers and compression state, plus the data supplied to each write.
"""

from __future__ import annotations

__lazy_modules__ = {"contextlib", "types", "os"}

from contextlib import ExitStack
from typing import TYPE_CHECKING

from ._blanket import PngStreamWriter, StreamingDecoder, UnidentifiedImageError
from .Image import Image

if TYPE_CHECKING:
    import os
    from types import TracebackType
    from typing import BinaryIO, Self


class Parser:
    """Feed JPEG XL chunks and close to obtain the decoded image.

    The image property returns an independent snapshot once a progressive pass
    is available. Container metadata is not retained. Instances
    must be used on the thread where they were created.
    """

    def __init__(self) -> None:
        """Create an empty parser ready to receive JPEG XL chunks."""
        self._prefix = bytearray()
        self._decoder: StreamingDecoder | None = None
        self._closed = False
        self._image: Image | None = None

    @property
    def image(self) -> Image | None:
        """Return the latest available raster snapshot, or None before a pass."""
        if self._closed or self._decoder is None:
            return self._image
        native = self._decoder.image
        return Image(native) if native is not None else None

    def feed(self, data: bytes | bytearray | memoryview) -> None:
        """Consume the next input chunk, raising OSError for malformed data.

        Args:
            data: The next bytes of a JPEG XL image, in file order.
        """
        if self._closed:
            raise ValueError("closed image parser")
        view = memoryview(data).cast("B")
        try:
            if self._decoder is not None and isinstance(data, bytes):
                if data:
                    self._decoder.feed(data)
                return
            if self._decoder is None:
                count = min(12 - len(self._prefix), len(view))
                self._prefix.extend(view[:count])
                view = view[count:]
                prefix = bytes(self._prefix)
                if prefix.startswith(b"\xff\x0a") or prefix == b"\x00\x00\x00\x0cJXL \x0d\x0a\x87\x0a":
                    self._decoder = StreamingDecoder("JXL")
                elif prefix.startswith(b"\xff\xd8"):
                    raise UnidentifiedImageError("streaming decode supports JXL")
                elif len(prefix) < 12:
                    return
                else:
                    raise UnidentifiedImageError("streaming decode supports JXL")
                self._decoder.feed(prefix)
                self._prefix.clear()
            if view:
                self._decoder.feed(view.tobytes())
        except Exception:
            self._closed = True
            self._decoder = None
            self._prefix.clear()
            raise

    def close(self) -> Image:
        """Finish decoding; raise OSError if the image is incomplete."""
        if self._closed:
            if self._image is None:
                raise ValueError("closed image parser")
            return self._image
        self._closed = True
        decoder, self._decoder = self._decoder, None
        self._prefix.clear()
        if decoder is None:
            raise OSError("incomplete image header")
        self._image = Image(decoder.close())
        return self._image


class PngWriter:
    """Encode complete PNG rows directly to a path or binary output stream.

    Supported modes are 1, L, LA, RGB, and RGBA at 8 bits, with 10/12/16-bit
    samples supported for L, RGB, and RGBA. Mode 1 uses packed rows; wide samples
    use little-endian unsigned integers, matching Image.frombytes. Metadata and
    animation are not supported. Call close or use a context manager to finish.
    """

    def __init__(self, fp: str | bytes | os.PathLike[str] | os.PathLike[bytes] | BinaryIO, mode: str, size: tuple[int, int], *, bit_depth: int = 8, compress_level: int = 6) -> None:
        """Start a PNG stream with the declared mode, size, and sample depth.

        Args:
            fp: Filename, path, or binary stream with write().
            mode: Image mode of the supplied rows.
            size: Width and height in pixels.
            bit_depth: Significant sample bits; mode 1 uses the default of 8.
            compress_level: PNG deflate level from 0 through 9.
        """
        width, height = size
        if width <= 0 or height <= 0:
            raise ValueError("PNG dimensions must be positive")
        if bit_depth not in (8, 10, 12, 16) or not 0 <= compress_level <= 9:
            raise ValueError("invalid bit depth or compression level")
        PngStreamWriter.validate(mode, size, bit_depth, compress_level)
        self._stack = ExitStack()
        self._fp = self._stack.enter_context(open(fp, "wb")) if not hasattr(fp, "write") else fp  # noqa: SIM115
        self._writer: PngStreamWriter | None = None
        try:
            self._writer = PngStreamWriter(self._fp, mode, size, bit_depth, compress_level)
        except Exception:
            self._stack.close()
            raise

    def write(self, data: bytes | bytearray | memoryview) -> None:
        """Encode one or more complete rows, without exceeding the image height.

        Args:
            data: Raw rows in the declared mode and sample depth.
        """
        if self._writer is None:
            raise ValueError("closed PNG writer")
        self._writer.write(data if isinstance(data, bytes) else memoryview(data).tobytes())

    def close(self) -> None:
        """Finish PNG output and close owned files; reject missing rows."""
        writer, self._writer = self._writer, None
        try:
            if writer is not None:
                writer.close()
        finally:
            self._stack.close()

    def __enter__(self) -> Self:
        """Return this writer for a with block."""
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: TracebackType | None) -> None:
        """Finish successful output and release owned files on any exit."""
        if exc_type is None:
            self.close()
        else:
            writer, self._writer = self._writer, None
            try:
                if writer is not None:
                    writer.abort()
            finally:
                self._stack.close()


__all__ = ["Parser", "PngWriter"]
