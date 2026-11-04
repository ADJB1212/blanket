from __future__ import annotations

import os
import sys
from io import BytesIO
from typing import TYPE_CHECKING

import pytest
from helpers import assert_pixels_equal, assert_raises, image_buffer
from PIL import Image as PillowImage

from blanket import Image, ImageMetadata
from blanket.Compressor import LosslessImageCompressor, LossyImageCompressor

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


@pytest.mark.parametrize("operation", [Image.inspect, ImageMetadata.rewrite, ImageMetadata.get_resolution, ImageMetadata.get_previews])
def test_read_failures_restore_source_position(operation) -> None:
    class FailingStream(BytesIO):
        def read(self, size: int = -1) -> bytes:
            if size == -1:
                raise OSError("read failed")
            return super().read(size)

    stream = FailingStream(Image.new("L", (3, 2), 7).encode("PNG"))
    stream.seek(5)
    assert_raises(OSError, lambda: operation(stream), match="read failed")
    assert stream.tell() == 5
    assert not stream.closed


@pytest.mark.parametrize("operation", [Image.inspect, ImageMetadata.rewrite, ImageMetadata.get_resolution, ImageMetadata.get_previews])
def test_unsized_reads_restore_source_position(operation) -> None:
    class UnsizedStream(BytesIO):
        def read(self) -> bytes:
            return super().read()

    encoded = Image.new("L", (3, 2), 7).encode("PNG")
    stream = UnsizedStream(encoded)
    stream.seek(5)
    result = operation(stream)
    if operation is Image.inspect:
        assert result.size == (3, 2)
    elif operation is ImageMetadata.rewrite:
        assert result == encoded
    elif operation is ImageMetadata.get_resolution:
        assert result.dpi is None
    else:
        assert result == ()
    assert stream.tell() == 5
    assert not stream.closed


@pytest.mark.parametrize(
    ("mode", "depth", "format"),
    [
        ("1", 8, "PNG"),
        ("L", 8, "PNG"),
        ("LA", 8, "PNG"),
        ("P", 8, "PNG"),
        ("PA", 8, "PNG"),
        ("RGB", 8, "PNG"),
        ("RGBA", 8, "PNG"),
        ("RGB", 16, "PNG"),
        ("RGB", 8, "BMP"),
        ("RGB", 8, "JPEG"),
        ("LA", 8, "TIFF"),
        ("P", 8, "TIFF"),
        ("CMYK", 8, "TIFF"),
        ("LAB", 8, "TIFF"),
        ("F", 32, "TIFF"),
        ("I;16B", 16, "PNG"),
    ],
)
@pytest.mark.parametrize("path_type", ["path", "str", "bytes", "bytes_pathlike"])
def test_file_save_matches_stream(tmp_path: Path, mode: str, depth: int, format: str, path_type: str) -> None:
    if depth == 16 and mode == "RGB":
        source = Image.frombytes(mode, (7, 5), bytes(range(210)), bit_depth=depth)
    else:
        source = Image.new(mode, (7, 5), 1)
    output = image_buffer(source, format)
    path = tmp_path / "image"

    class BytesPath:
        def __fspath__(self) -> bytes:
            return os.fsencode(path)

    target = {"path": path, "str": str(path), "bytes": os.fsencode(path), "bytes_pathlike": BytesPath()}[path_type]
    assert source.save(target, format) is None
    assert path.read_bytes() == output.getvalue()
    with PillowImage.open(path) as actual, PillowImage.open(BytesIO(output.getvalue())) as expected:
        actual.load()
        expected.load()
        assert_pixels_equal(actual, expected)


def test_file_save_encoding_failure_preserves_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "image.ico"
    path.write_bytes(b"original")
    assert_raises(ValueError, lambda: Image.new("RGB", (257, 1)).save(path), match="ICO dimensions")
    assert path.read_bytes() == b"original"


@pytest.mark.parametrize("format", ["PNG", "JPEG", "JXL"])
@pytest.mark.parametrize("path_type", [str, os.fsencode])
def test_metadata_file_save_matches_stream_and_preserves_file_on_error(tmp_path: Path, format: str, path_type: Callable[[Path], str | bytes]) -> None:
    source = Image.new("RGB", (17, 9), (31, 73, 127))
    exif = Image.Exif()
    exif[305] = "Blanket metadata writer"
    options = {"exif": exif, "xmp": b"test packet"}
    stream = image_buffer(source, format, **options)
    path = tmp_path / "image"
    source.save(path_type(path), format, **options)
    assert path.read_bytes() == stream.getvalue()
    decoded = Image.open(path)
    assert decoded.getexif()[305] == exif[305]
    assert decoded.info["xmp"] == options["xmp"]
    if format == "JPEG":
        assert_raises(ValueError, lambda: source.save(path_type(path), format, xmp=b"x" * 65536), match="XMP data is too long")
        assert path.read_bytes() == stream.getvalue()


@pytest.mark.parametrize("path_type", [str, os.fsencode])
def test_file_save_errors_include_filename(tmp_path: Path, path_type: Callable[[Path], str | bytes]) -> None:
    path = path_type(tmp_path / "missing" / "image.png")
    with pytest.raises(FileNotFoundError) as error:
        Image.new("RGB", (1, 1)).save(path)
    assert error.value.filename == path


def test_file_save_rejects_embedded_null() -> None:
    assert_raises(ValueError, lambda: Image.new("RGB", (1, 1)).save("image\0.png"), match="embedded null")


@pytest.mark.skipif(os.name == "nt" or sys.platform == "darwin", reason="filesystem requires Unicode filenames")
def test_file_save_with_non_utf8_filename(tmp_path: Path) -> None:
    path = os.fsencode(tmp_path) + b"/image-\xff.png"
    source = Image.new("RGB", (3, 5), (23, 57, 91))
    source.save(path)
    assert Image.open(path).tobytes() == source.tobytes()


@pytest.mark.parametrize("path_type", [str, os.fsencode])
def test_file_save_with_unicode_filename(tmp_path: Path, path_type: Callable[[Path], str | bytes]) -> None:
    path = path_type(tmp_path / "image-\N{LATIN SMALL LETTER E WITH ACUTE}.png")
    source = Image.new("RGB", (3, 5), (23, 57, 91))
    source.save(path)
    assert Image.open(path).tobytes() == source.tobytes()


@pytest.mark.parametrize("compressor", [LosslessImageCompressor(effort=1), LossyImageCompressor(effort=1)])
def test_file_save_with_compressor_matches_stream(tmp_path: Path, compressor: LosslessImageCompressor | LossyImageCompressor) -> None:
    source = Image.new("RGB", (17, 13), (23, 57, 91))
    output = image_buffer(source, "JPEG", compressor=compressor)
    path = tmp_path / "image.jpg"
    source.save(path, compressor=compressor)
    assert path.read_bytes() == output.getvalue()


def test_stream_save_receives_bytes() -> None:
    class Output:
        data: bytes

        def write(self, data: bytes) -> int:
            assert isinstance(data, bytes)
            self.data = data
            return len(data)

    output = Output()
    Image.new("RGB", (1, 1)).save(output, "PNG")
    assert Image.open(BytesIO(output.data)).size == (1, 1)


@pytest.mark.parametrize("sequence", [False, True])
@pytest.mark.parametrize("metadata", [False, True])
def test_stream_save_retries_partial_writes(sequence: bool, metadata: bool) -> None:
    class PartialOutput(BytesIO):
        def write(self, data: bytes) -> int:
            assert isinstance(data, bytes)
            return super().write(data[:7])

    source = Image.new("RGB", (17, 13), (23, 57, 91))
    options = {"save_all": True, "append_images": [source.copy()]} if sequence else {}
    if metadata:
        options["xmp"] = b"test packet"
    expected, actual = BytesIO(), PartialOutput()
    source.save(expected, "PNG", **options)
    source.save(actual, "PNG", **options)
    assert actual.getvalue() == expected.getvalue()
    assert Image.open(BytesIO(actual.getvalue())).n_frames == (2 if sequence else 1)


@pytest.mark.parametrize("sequence", [False, True])
@pytest.mark.parametrize("count", [None, 0, -1, 1.5, True, 100000])
def test_stream_save_rejects_no_progress_and_invalid_counts(sequence: bool, count: object) -> None:
    class Output:
        def write(self, data: bytes) -> object:
            return count

    assert_raises(OSError, lambda: Image.new("RGB", (1, 1)).save(Output(), "PNG", save_all=sequence))


@pytest.mark.parametrize("sequence", [False, True])
def test_stream_save_propagates_write_errors(sequence: bool) -> None:
    class Output:
        calls = 0

        def write(self, data: bytes) -> int:
            self.calls += 1
            if self.calls == 1:
                return 7
            raise BlockingIOError("output would block")

    assert_raises(BlockingIOError, lambda: Image.new("RGB", (1, 1)).save(Output(), "PNG", save_all=sequence), match="output would block")
