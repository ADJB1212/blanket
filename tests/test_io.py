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


@pytest.mark.parametrize("seekable", [False, True])
def test_stream_reads_preserve_binary_data_and_avoid_seekable_copy(seekable):
    from blanket._io import read_bytes

    class Stream(BytesIO):
        def seek(self, offset, whence=0):
            if not seekable:
                raise OSError("not seekable")
            return super().seek(offset, whence)

    payload = bytes(range(256)) * 1024
    stream = Stream(payload)
    data = read_bytes(stream)
    assert data == payload
    assert stream.tell() == len(payload)
    if seekable:
        assert data is payload


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
    ("mode", "depth", "format", "path_type"),
    [
        (*case, path_type)
        for case in [
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
        ]
        for path_type in ("path", "str", "bytes", "bytes_pathlike")
        if path_type == "path" or case == ("RGB", 8, "PNG")
    ],
)
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


@pytest.mark.parametrize("mode,key,other", [("1", 1, 0), ("L", 42, 9), ("RGB", (4, 5, 6), (7, 8, 9))])
@pytest.mark.parametrize("explicit", [False, True])
def test_png_save_preserves_transparent_color_key(mode, key, other, explicit):
    image = Image.new(mode, (2, 1), key)
    image.putpixel((1, 0), other)
    if not explicit:
        image.info["transparency"] = key
    data = image.encode("PNG", **({"transparency": key} if explicit else {}))
    for decoder in (Image, PillowImage):
        with decoder.open(BytesIO(data)) as decoded:
            rgba = decoded.convert("RGBA")
            assert rgba.getpixel((0, 0))[3] == 0
            assert rgba.getpixel((1, 0))[3] == 255
    assert image.info.get("transparency") == (None if explicit else key)


@pytest.mark.parametrize("depth", [10, 12, 16])
def test_png_save_scales_wide_transparency_key_with_samples(depth):
    key = 343
    image = Image.frombytes("L", (2, 1), key.to_bytes(2, "little") + (key + 1).to_bytes(2, "little"), bit_depth=depth)
    data = image.encode("PNG", transparency=key)
    decoded = Image.decode(data).convert("RGBA")
    assert decoded.getpixel((0, 0))[3] == 0
    assert decoded.getpixel((1, 0))[3] == (1 << decoded.bit_depth) - 1


def test_png_save_after_depth_conversion_uses_rescaled_key():
    image = Image.frombytes("L", (2, 1), bytes((42, 9)))
    image.info["transparency"] = 42
    converted = image.convert("L", bit_depth=12)
    assert converted.info["transparency"] == converted.getpixel((0, 0))
    decoded = Image.decode(converted.encode("PNG")).convert("RGBA")
    assert decoded.getpixel((0, 0))[3] == 0
    assert decoded.getpixel((1, 0))[3] > 0


@pytest.mark.parametrize("mode", ["L", "RGB"])
@pytest.mark.parametrize("source_depth,out_depth", [(8, 12), (12, 12), (16, 8)])
def test_keyed_conversion_to_rgba_honors_depth(mode, source_depth, out_depth):
    image = Image.new(mode, (2, 1), 42).convert(mode, bit_depth=source_depth)
    image.putpixel((1, 0), 9 if mode == "L" else (9, 10, 11))
    image.info["transparency"] = image.getpixel((0, 0))
    converted = image.convert("RGBA", bit_depth=out_depth)
    assert converted.bit_depth == out_depth
    assert converted.getpixel((0, 0))[3] == 0
    assert converted.getpixel((1, 0))[3] == (1 << out_depth) - 1
    assert "transparency" not in converted.info


def test_png_save_after_discarding_decoded_alpha_does_not_restore_color_key():
    original = Image.new("RGB", (2, 1), (4, 5, 6))
    rgba = Image.decode(original.encode("PNG", transparency=(4, 5, 6)))
    grayscale = rgba.convert("L")
    assert "transparency" not in grayscale.info
    assert Image.decode(grayscale.encode("PNG")).mode == "L"


@pytest.mark.parametrize("compressor", [LosslessImageCompressor(), LossyImageCompressor(max_rmse=255)])
def test_png_compressor_keeps_transparent_color_key(compressor):
    source = Image.frombytes("RGB", (2, 1), bytes((4, 5, 6, 7, 8, 9)))
    source.info["transparency"] = (4, 5, 6)
    encoded = source.encode("PNG", compressor=compressor)
    decoded = Image.decode(encoded).convert("RGBA")
    assert decoded.getdata() == [(4, 5, 6, 0), (7, 8, 9, 255)]
    assert len(encoded) <= len(source.encode("PNG"))


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
