from __future__ import annotations

import random
from io import BytesIO
from typing import TYPE_CHECKING

import pytest
from PIL import Image as PillowImage

from blanket import Image, ImageFile, UnidentifiedImageError

if TYPE_CHECKING:
    from pathlib import Path


def encoded(mode: str) -> bytes:
    channels = {"L": 1, "RGB": 3, "RGBA": 4}[mode]
    raw = random.Random(74).randbytes(33 * 19 * channels)
    output = BytesIO()
    Image.frombytes(mode, (33, 19), raw).save(output, "JXL", lossless=True)
    return output.getvalue()


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
@pytest.mark.parametrize("chunk_size", [1, 2, 7, 64, 65536])
def test_parser_chunk_boundaries_match_eager(mode: str, chunk_size: int) -> None:
    data = encoded(mode)
    parser = ImageFile.Parser()
    assert parser.image is None
    parser.feed(b"")
    for start in range(0, len(data), chunk_size):
        parser.feed(data[start : start + chunk_size])
        parser.feed(b"")
    result = parser.close()
    expected = Image.open(BytesIO(data))
    assert (result.mode, result.size, result.bit_depth, result.format) == (expected.mode, expected.size, expected.bit_depth, "JXL")
    assert result.tobytes() == expected.tobytes()
    assert parser.close() is result
    assert parser.image is result
    with pytest.raises(ValueError, match="closed"):
        parser.feed(b"")


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
def test_jxl_stream_retains_wide_samples(mode: str) -> None:
    channels = len(Image.new(mode, (1, 1)).getbands())
    raw = b"".join(((i * 1973) % 65536).to_bytes(2, "little") for i in range(17 * 9 * channels))
    output = BytesIO()
    Image.frombytes(mode, (17, 9), raw, bit_depth=16).save(output, "JXL", lossless=True)
    parser = ImageFile.Parser()
    data = output.getvalue()
    for i in range(0, len(data), 13):
        parser.feed(memoryview(data)[i : i + 13])
    result = parser.close()
    assert result.bit_depth == 16
    assert result.tobytes() == raw


@pytest.mark.parametrize("cut", [0, 1, 12, -2])
def test_parser_rejects_truncated_input(cut: int) -> None:
    parser = ImageFile.Parser()
    parser.feed(encoded("RGB")[:cut])
    with pytest.raises(OSError, match=r"incomplete|truncated"):
        parser.close()


def test_parser_rejects_unsupported_and_corrupt_input() -> None:
    parser = ImageFile.Parser()
    with pytest.raises(UnidentifiedImageError, match="supports JXL"):
        parser.feed(b"not an image" * 100)
    with pytest.raises(ValueError, match="closed"):
        parser.feed(b"")
    parser = ImageFile.Parser()
    with pytest.raises(OSError):
        parser.feed(b"\xff\x0a" + b"\xff" * 128)


@pytest.mark.parametrize("chunk_size", [1, 7, 65536])
def test_parser_rejects_jpeg_and_preserves_eager_decode(chunk_size: int) -> None:
    output = BytesIO()
    source = Image.new("RGB", (3, 2), "red")
    source.save(output, "JPEG")
    data = output.getvalue()
    parser = ImageFile.Parser()
    with pytest.raises(UnidentifiedImageError, match="supports JXL"):
        for i in range(0, len(data), chunk_size):
            parser.feed(data[i : i + chunk_size])
    with pytest.raises(ValueError, match="closed"):
        parser.feed(b"")
    assert Image.open(BytesIO(data)).tobytes() == PillowImage.open(BytesIO(data)).tobytes()


@pytest.mark.parametrize("chunk_size", [1, 7, 17, 4096])
def test_jxl_container_signature_and_payload_boundaries(chunk_size: int) -> None:
    data = encoded("RGBA")
    signature = b"\x00\x00\x00\x0cJXL \x0d\x0a\x87\x0a"
    ftyp = b"\x00\x00\x00\x14ftypjxl \x00\x00\x00\x00jxl "
    container = signature + ftyp + (len(data) + 8).to_bytes(4, "big") + b"jxlc" + data
    parser = ImageFile.Parser()
    for i in range(0, len(container), chunk_size):
        parser.feed(container[i : i + chunk_size])
    assert parser.close().tobytes() == Image.open(BytesIO(data)).tobytes()


@pytest.mark.parametrize("mode", ["1", "L", "LA", "RGB", "RGBA"])
@pytest.mark.parametrize("compress_level", [0, 6, 9])
@pytest.mark.parametrize("buffer_type", [bytes, bytearray, memoryview])
def test_png_rows_match_pillow_and_eager(mode: str, compress_level: int, buffer_type: type) -> None:
    size = (33, 19)
    channels = len(Image.new(mode, (1, 1)).getbands())
    stride = (size[0] + 7) // 8 if mode == "1" else size[0] * channels
    raw = random.Random(19).randbytes(stride * size[1])
    expected = Image.frombytes(mode, size, raw)
    output = BytesIO()
    with ImageFile.PngWriter(output, mode, size, compress_level=compress_level) as writer:
        assert output.getvalue().startswith(b"\x89PNG\r\n\x1a\n")
        for i in range(0, len(raw), stride):
            writer.write(buffer_type(raw[i : i + stride]))
    assert not output.closed
    result = Image.open(BytesIO(output.getvalue()))
    assert (result.mode, result.size, result.bit_depth) == (expected.mode, size, 8)
    assert result.tobytes() == expected.tobytes()
    assert PillowImage.open(BytesIO(output.getvalue())).tobytes() == expected.tobytes()


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
@pytest.mark.parametrize("depth", [10, 12, 16])
def test_png_wide_rows(mode: str, depth: int) -> None:
    channels = len(Image.new(mode, (1, 1)).getbands())
    maximum = (1 << depth) - 1
    raw = b"".join(value.to_bytes(2, "little") for i in range(17 * 9 * channels) for value in [(i * 1973) & maximum])
    output = BytesIO()
    with ImageFile.PngWriter(output, mode, (17, 9), bit_depth=depth) as writer:
        writer.write(memoryview(raw))
    result = Image.open(BytesIO(output.getvalue()))
    assert (result.mode, result.bit_depth) == (mode, depth)
    assert result.tobytes() == raw
    reference = BytesIO()
    Image.frombytes(mode, (17, 9), raw, bit_depth=depth).save(reference, "PNG")
    assert PillowImage.open(BytesIO(output.getvalue())).tobytes() == PillowImage.open(reference).tobytes()


def test_png_writer_validates_rows_and_close(tmp_path: Path) -> None:
    path = tmp_path / "rows.png"
    writer = ImageFile.PngWriter(path, "L", (3, 2))
    with pytest.raises(ValueError, match="complete rows"):
        writer.write(b"ab")
    with pytest.raises(ValueError, match="complete rows"):
        writer.write(b"abcdefghi")
    writer.write(b"abc")
    with pytest.raises(OSError, match="incomplete"):
        writer.close()
    with pytest.raises(ValueError, match="closed"):
        writer.write(b"def")
    with ImageFile.PngWriter(path, "L", (3, 2)) as writer:
        writer.write(b"abcdef")
    assert Image.open(path).tobytes() == b"abcdef"


@pytest.mark.parametrize(("mode", "size", "options"), [("P", (1, 1), {}), ("L", (0, 1), {}), ("L", (178956971, 1), {}), ("L", (1, 1), {"bit_depth": 9}), ("L", (1, 1), {"compress_level": 10})])
def test_png_writer_rejects_invalid_headers(mode: str, size: tuple[int, int], options: dict[str, int]) -> None:
    output = BytesIO()
    with pytest.raises(ValueError):
        ImageFile.PngWriter(output, mode, size, **options)
    assert output.getvalue() == b""


def test_png_writer_rejects_out_of_range_wide_samples() -> None:
    output = BytesIO()
    writer = ImageFile.PngWriter(output, "L", (1, 1), bit_depth=10)
    with pytest.raises(ValueError, match="sample exceeds"):
        writer.write(b"\x00\x04")
    writer.write(b"\xff\x03")
    writer.close()
    assert Image.open(BytesIO(output.getvalue())).tobytes() == b"\xff\x03"


def test_png_writer_short_writes_and_sink_failures() -> None:
    class ShortSink(BytesIO):
        def write(self, data: bytes) -> int:
            return super().write(data[:3])

    output = ShortSink()
    with ImageFile.PngWriter(output, "L", (3, 2)) as writer:
        writer.write(b"abcdef")
    assert Image.open(BytesIO(output.getvalue())).tobytes() == b"abcdef"

    class BrokenSink(BytesIO):
        def write(self, data: bytes) -> int:
            raise RuntimeError("sink failure")

    with pytest.raises(RuntimeError, match="sink failure"):
        ImageFile.PngWriter(BrokenSink(), "L", (3, 2))


def test_png_writer_context_preserves_exception_and_borrowed_stream() -> None:
    output = BytesIO()
    with pytest.raises(RuntimeError, match="producer failed"), ImageFile.PngWriter(output, "L", (3, 2)):
        raise RuntimeError("producer failed")
    assert not output.closed


def test_png_writer_emits_data_incrementally() -> None:
    output = BytesIO()
    raw = random.Random(19).randbytes(257 * 67)
    with ImageFile.PngWriter(output, "L", (257, 67)) as writer:
        for i in range(0, len(raw) - 257, 257):
            writer.write(raw[i : i + 257])
        assert b"IDAT" in output.getvalue()
        assert b"IEND" not in output.getvalue()
        writer.write(raw[-257:])
    assert Image.open(BytesIO(output.getvalue())).tobytes() == raw


@pytest.mark.parametrize("during_write", [False, True])
def test_png_writer_preserves_late_sink_error(during_write: bool) -> None:
    class FailingSink(BytesIO):
        failed = False

        def write(self, data: bytes) -> int:
            if self.failed:
                raise RuntimeError("late sink failure")
            return super().write(data)

    output = FailingSink()
    writer = ImageFile.PngWriter(output, "L", (8192, 32), compress_level=0)
    if during_write:
        output.failed = True
        with pytest.raises(RuntimeError, match="late sink failure"):
            writer.write(bytes(8192 * 32))
        with pytest.raises(ValueError, match="closed"):
            writer.write(bytes(8192))
    else:
        writer.write(bytes(8192 * 32))
        output.failed = True
        with pytest.raises(RuntimeError, match="late sink failure"):
            writer.close()


def test_png_writer_invalid_headers_preserve_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "existing.png"
    path.write_bytes(b"existing output")
    with pytest.raises(ValueError):
        ImageFile.PngWriter(path, "P", (1, 1))
    assert path.read_bytes() == b"existing output"
