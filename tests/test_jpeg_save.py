from __future__ import annotations

from io import BytesIO

import pytest
from helpers import assert_raises, encode_image
from PIL import Image as PillowImage

from blanket import Image, ImageFile
from blanket._jpeg import segments
from blanket.Compressor import LosslessImageCompressor


def source(mode: str = "RGB", size: tuple[int, int] = (17, 9)) -> tuple[Image.Image, PillowImage.Image]:
    channels = len(PillowImage.new(mode, size).getbands())
    length = ((size[0] + 7) // 8 if mode == "1" else size[0] * channels) * size[1]
    raw = bytes((i * 137 + i // 7) % 256 for i in range(length))
    return Image.frombytes(mode, size, raw), PillowImage.frombytes(mode, size, raw)


def save(image: Image.Image | PillowImage.Image, **options: object) -> bytes:
    return encode_image(image, "JPEG", **options)


def compare(actual: bytes, expected: bytes, tolerance: int = 6) -> None:
    with PillowImage.open(BytesIO(actual)) as decoded, PillowImage.open(BytesIO(expected)) as reference:
        assert (decoded.mode, decoded.size) == (reference.mode, reference.size)
        assert decoded.layer == reference.layer
        assert decoded.quantization == reference.quantization
        assert max(abs(a - b) for a, b in zip(decoded.tobytes(), reference.tobytes(), strict=True)) <= tolerance
        assert Image.decode(actual).tobytes() == decoded.tobytes()


@pytest.mark.parametrize("mode,count", [("1", 1), ("L", 1), ("RGB", 1), ("RGB", 2), ("RGB", 3), ("YCbCr", 3), ("CMYK", 1), ("CMYK", 2), ("CMYK", 3), ("CMYK", 4)])
@pytest.mark.parametrize("wide", [False, True])
@pytest.mark.parametrize("progressive", [False, True])
def test_qtables_modes_and_precision(mode: str, count: int, wide: bool, progressive: bool) -> None:
    tables = [[(300 if wide else 1) + (i * 7 + n * 11) % 31 for i in range(64)] for n in range(count)]
    original = [table[:] for table in tables]
    actual, expected = source(mode)
    encoded = save(actual, qtables=tables, progressive=progressive)
    compare(encoded, save(expected, qtables=tables, progressive=progressive))
    assert Image.decode(encoded).quantization == dict(enumerate(tables))
    assert tables == original
    compare(save(Image.decode(encoded), qtables="keep"), save(PillowImage.open(BytesIO(encoded)), qtables="keep"))


@pytest.mark.parametrize("quality", [0, 1, 50, 75, 100, 101])
def test_qtables_quality_scaling(quality: int) -> None:
    tables = [[(i * 17 + 11) % 65536 for i in range(64)], [65535] * 64]
    actual, expected = source()
    compare(save(actual, qtables=tables, quality=quality), save(expected, qtables=tables, quality=quality))


@pytest.mark.parametrize(
    "tables,error",
    [
        ("a", ValueError),
        ("1 2 3", ValueError),
        ([], ValueError),
        ({}, ValueError),
        ([[1] * 64] * 5, ValueError),
        ([1], ValueError),
        ([[1] * 63], ValueError),
        ([[1.5] * 64], ValueError),
        ([[-1] * 64], OverflowError),
        ([[65536] * 64], OverflowError),
        ("keep", ValueError),
    ],
)
def test_qtables_validation(tables: object, error: type[Exception]) -> None:
    for image in source():
        output = BytesIO(b"unchanged")
        assert_raises(error, lambda: image.save(output, "JPEG", qtables=tables))
        assert output.getvalue() == b"unchanged"


@pytest.mark.parametrize("blocks,rows,count", [(0, 0, 0), (1, 0, 15), (3, 0, 5), (8, 0, 1), (0, 1, 3), (0, 2, 1), (1, 2, 1)])
def test_restart_marker_counts_and_rows_override(blocks: int, rows: int, count: int) -> None:
    actual, expected = source(size=(32, 32))
    options = {"restart_marker_blocks": blocks, "restart_marker_rows": rows, "subsampling": 0}
    data = save(actual, **options)
    reference = save(expected, **options)
    markers = [data[i + 1] for i in range(len(data) - 1) if data[i] == 255 and 208 <= data[i + 1] <= 215]
    assert markers == [208 + i % 8 for i in range(count)]
    assert [bytes(payload) for marker, payload in segments(data) if marker == 221] == [bytes(payload) for marker, payload in segments(reference) if marker == 221]
    compare(data, reference)


@pytest.mark.parametrize("mode", ["L", "RGB", "CMYK", "YCbCr"])
@pytest.mark.parametrize("options", [{}, {"progressive": True}, {"optimize": True}, {"smooth": 100}, {"qtables": [[300] * 64]}])
def test_restart_markers_with_encoder_options(mode: str, options: dict[str, object]) -> None:
    actual, expected = source(mode, (33, 19))
    data = save(actual, restart_marker_blocks=2, **options)
    compare(data, save(expected, restart_marker_blocks=2, **options))
    parser = ImageFile.Parser()
    for position in range(0, len(data), 7):
        parser.feed(data[position : position + 7])
    assert parser.close().tobytes() == Image.decode(data).tobytes()


@pytest.mark.parametrize("options", [{}, {"keep_rgb": True}, {"smooth": 100}, {"qtables": [[300] * 64]}, {"progressive": True}])
def test_rgbx_save_ignores_padding(options: dict[str, object]) -> None:
    actual, expected = source("RGBX")
    original = actual.tobytes()
    compare(save(actual, **options), save(expected, **options))
    assert save(actual, **options) == save(actual.convert("RGB"), **options)
    assert actual.mode == "RGBX" and actual.tobytes() == original


def test_jpeg_xmp_requires_explicit_save_option() -> None:
    actual, expected = source()
    for image in (actual, expected):
        for packet in (b"XMP test", b"1" * 65504):
            data = save(image, xmp=packet)
            decoded = Image.decode(data)
            assert decoded.info["xmp"] == packet
            assert PillowImage.open(BytesIO(data)).info["xmp"] == packet
            assert "xmp" not in Image.decode(save(decoded)).info
            assert Image.decode(save(decoded, xmp=decoded.info["xmp"])).info["xmp"] == packet
        output = BytesIO(b"unchanged")
        assert_raises(ValueError, lambda: image.save(output, "JPEG", xmp=b"1" * 65505))
        assert output.getvalue() == b"unchanged"


@pytest.mark.parametrize("mode", ["L", "RGB", "CMYK"])
@pytest.mark.parametrize("options", [{}, {"progressive": True}, {"optimize": True}, {"qtables": [[300] * 64]}])
def test_separate_streams_interoperate(mode: str, options: dict[str, object]) -> None:
    actual, expected = source(mode)
    for table_image, body_image in ((actual, actual), (actual, expected), (expected, actual)):
        tables = save(table_image, streamtype=1, **options)
        body = save(body_image, streamtype=2, **options)
        assert tables.startswith(b"\xff\xd8") and tables.endswith(b"\xff\xd9")
        assert body.startswith(b"\xff\xd8") and body.endswith(b"\xff\xd9")
        assert 219 in [marker for marker, _ in segments(tables)]
        assert 219 not in [marker for marker, _ in segments(body)]
        assert not any(marker in (192, 193, 194, 224) for marker, _ in segments(tables))
        if not options.get("optimize") and not options.get("progressive"):
            assert 196 not in [marker for marker, _ in segments(body)]
        combined = tables + body
        decoded = Image.decode(combined)
        reference = PillowImage.open(BytesIO(combined))
        assert decoded.tobytes() == reference.tobytes()
        assert decoded.layer == reference.layer
        assert decoded.quantization == reference.quantization
        assert Image.inspect(BytesIO(combined)).size == actual.size
        assert decoded.tobytes() == Image.decode(save(body_image, **options)).tobytes()
        parser = ImageFile.Parser()
        for position in range(0, len(combined), 7):
            parser.feed(combined[position : position + 7])
        assert parser.close().tobytes() == decoded.tobytes()


@pytest.mark.parametrize("streamtype", [1, 2])
def test_separate_streams_reject_compressors(streamtype: int) -> None:
    assert_raises(ValueError, lambda: save(source()[0], streamtype=streamtype, compressor=LosslessImageCompressor()), match="complete JPEG stream")


def test_image_only_stream_cannot_reuse_tables_from_previous_decodes() -> None:
    image = source()[0]
    data = save(image)
    body = save(image, streamtype=2)
    for preceding in (data, save(image, qtables=[[300] * 64]), data):
        Image.decode(preceding).load()
        assert_raises(OSError, lambda: Image.decode(body).load())
    assert Image.decode(save(image, streamtype=1) + body).tobytes() == Image.decode(data).tobytes()


@pytest.mark.parametrize("mode", ["L", "RGB", "CMYK"])
@pytest.mark.parametrize("progressive", [False, True])
def test_restart_rows_respect_sampling_and_scan_width(mode: str, progressive: bool) -> None:
    actual, expected = source(mode, (33, 19))
    options = {"restart_marker_rows": 1, "subsampling": 2, "progressive": progressive}
    data = save(actual, **options)
    reference = save(expected, **options)
    compare(data, reference)
    assert sum(data.count(bytes((255, marker))) for marker in range(208, 216)) == sum(reference.count(bytes((255, marker))) for marker in range(208, 216))


@pytest.mark.parametrize("value", [0, 255, 256, 32767, 32768, 65535])
def test_qtable_coefficient_bounds(value: int) -> None:
    actual, expected = source()
    tables = [[value] * 64]
    compare(save(actual, qtables=tables), save(expected, qtables=tables))


@pytest.mark.parametrize("chunk_size", [1, 64, 65536])
def test_separate_stream_chunk_boundaries_and_truncation(chunk_size: int) -> None:
    image = source()[0]
    tables = save(image, streamtype=1)
    body = save(image, streamtype=2)
    combined = tables + body
    parser = ImageFile.Parser()
    for position in range(0, len(combined), chunk_size):
        parser.feed(combined[position : position + chunk_size])
    assert parser.close().tobytes() == Image.decode(combined).tobytes()
    for incomplete in (tables, tables + body[:5], tables + body[:-20], b"\xff\xd8\xff\xdb\x00\x01\xff\xd9" + body):
        assert_raises(OSError, lambda: Image.decode(incomplete).load())
        parser = ImageFile.Parser()
        with pytest.raises(OSError):
            parser.feed(incomplete)
            parser.close()


@pytest.mark.parametrize("mode", ["RGB", "L", "YCbCr"])
def test_separate_stream_draft(mode: str) -> None:
    image = source(size=(63, 47))[0]
    data = save(image, streamtype=1) + save(image, streamtype=2)
    actual = Image.decode(data)
    expected = PillowImage.open(BytesIO(data))
    assert actual.draft(mode, (8, 7)) == expected.draft(mode, (8, 7))
    assert actual.tobytes() == expected.tobytes()


def test_jpeg_decode_stops_at_eof_and_handles_wide_scanlines() -> None:
    actual, _ = source(size=(65500, 1))
    data = save(actual)
    assert Image.decode(data).tobytes() == PillowImage.open(BytesIO(data)).tobytes()
    for cut in (2, 20, data.index(b"\xff\xda") + 5):
        assert_raises(OSError, lambda: Image.decode(data[:cut]).load())
    parser = ImageFile.Parser()
    parser.feed(data[:-20])
    assert_raises(OSError, lambda: parser.close(), match=r"incomplete|truncated")
