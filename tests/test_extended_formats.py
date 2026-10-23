from __future__ import annotations

import shutil
import struct
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image as PillowImage

from blanket import Image, UnidentifiedImageError, features


@pytest.mark.parametrize(
    ("format", "mode"),
    [
        ("PPM", "1"),
        ("PPM", "L"),
        ("PPM", "RGB"),
        ("TGA", "L"),
        ("TGA", "LA"),
        ("TGA", "RGB"),
        ("TGA", "RGBA"),
        ("QOI", "RGB"),
        ("QOI", "RGBA"),
        ("JPEG2000", "L"),
        ("JPEG2000", "LA"),
        ("JPEG2000", "RGB"),
        ("JPEG2000", "RGBA"),
    ],
)
def test_extended_interoperability(format: str, mode: str) -> None:
    size = (13, 7)
    channels = len(mode) if mode != "1" else 1
    source = Image.frombytes(mode, size, bytes((i * 37 + 13) % 256 for i in range(size[0] * size[1] * channels))) if mode != "1" else Image.new("1", size)
    if mode == "1":
        source.putdata([255 if i % 3 else 0 for i in range(size[0] * size[1])])
    for writer in (source, source.to_pillow()):
        stream = BytesIO()
        writer.save(stream, format)
        actual = Image.open(stream, formats=[format])
        expected = PillowImage.open(stream)
        assert (actual.format, actual.size, actual.mode) == (format, size, mode)
        assert actual.tobytes() == source.tobytes() == expected.tobytes()
        with pytest.raises(UnidentifiedImageError, match="not in formats"):
            Image.open(stream, formats=["PNG"])


@pytest.mark.parametrize("mode", ["L", "LA", "RGB", "RGBA"])
def test_pam_roundtrip(mode: str) -> None:
    channels = len(mode)
    raw = bytes((i * 19) % 256 for i in range(5 * 3 * channels))
    source = Image.frombytes(mode, (5, 3), raw)
    stream = BytesIO()
    source.save(stream, "PAM")
    assert stream.getvalue().startswith(b"P7\n")
    for formats in (None, ["PAM"], ["PPM"]):
        actual = Image.open(stream, formats=formats)
        assert (actual.format, actual.mode, actual.size) == ("PPM", mode, source.size)
        assert actual.tobytes() == raw


@pytest.mark.parametrize(
    ("data", "mode", "raw"),
    [(b"P1\n# comment\n3 2\n0 1 0 1 0 1\n", "1", b"\xff\0\xff\0\xff\0"), (b"P2\n3 1\n255\n0 127 255\n", "L", b"\0\x7f\xff"), (b"P3\n1 1\n255\n5 25 250\n", "RGB", b"\x05\x19\xfa")],
)
def test_ascii_netpbm(data: bytes, mode: str, raw: bytes) -> None:
    actual = Image.open(BytesIO(data))
    assert actual.mode == mode
    assert (actual.convert("L").tobytes() if mode == "1" else actual.tobytes()) == raw


@pytest.mark.parametrize("depth", [10, 12, 16])
@pytest.mark.parametrize(("format", "mode"), [(format, mode) for format in ("JPEG2000", "PAM", "PPM") for mode in ("L", "RGB", "RGBA") if format != "PPM" or mode != "RGBA"])
def test_extended_wide_roundtrip(mode: str, depth: int, format: str) -> None:
    channels = len(mode)
    values = [(i * 719) % (1 << depth) for i in range(9 * 5 * channels)]
    source = Image.frombytes(mode, (9, 5), struct.pack("<" + "H" * len(values), *values), bit_depth=depth)
    stream = BytesIO()
    source.save(stream, format)
    actual = Image.open(stream)
    assert (actual.mode, actual.bit_depth) == (mode, 16)
    maximum = (1 << depth) - 1
    expected = [(value * 65535 + maximum // 2) // maximum for value in values]
    assert actual.tobytes() == struct.pack("<" + "H" * len(expected), *expected)


@pytest.mark.parametrize("raw_codestream", [False, True])
def test_external_jpeg2000_16bit(raw_codestream: bool) -> None:
    raw = struct.pack("<6H", 0, 1, 255, 256, 32768, 65535)
    stream = BytesIO()
    PillowImage.frombytes("I;16", (3, 2), raw).save(stream, "JPEG2000", no_jp2=raw_codestream)
    actual = Image.open(stream)
    assert actual.bit_depth == 16
    assert actual.tobytes() == raw


@pytest.mark.parametrize("mode", ["I;16", "I;16L", "I;16B"])
@pytest.mark.parametrize("format", ["PPM", "PAM", "JPEG2000"])
def test_extended_integer_save(mode: str, format: str) -> None:
    values = [0, 1, 255, 256, 32768, 65535]
    raw = struct.pack((">" if mode == "I;16B" else "<") + "6H", *values)
    stream = BytesIO()
    Image.frombytes(mode, (3, 2), raw).save(stream, format)
    actual = Image.open(stream)
    assert actual.bit_depth == 16
    assert actual.tobytes() == struct.pack("<6H", *values)


@pytest.mark.parametrize("mode", ["L", "LA", "RGB", "RGBA", "P"])
@pytest.mark.parametrize("compression", [None, "tga_rle"])
def test_external_tga_orientation_and_palette(mode: str, compression: str | None) -> None:
    source = PillowImage.frombytes(mode, (7, 5), bytes((i * 37) % 256 for i in range(35 * len(mode))))
    if mode == "P":
        source.putpalette(bytes((i * 17) % 256 for i in range(768)))
    stream = BytesIO()
    source.save(stream, "TGA", compression=compression)
    actual = Image.open(stream)
    assert actual.convert("RGBA").tobytes() == source.convert("RGBA").tobytes()


@pytest.mark.parametrize("mode", ["RGB", "RGBA"])
@pytest.mark.parametrize("width", [1, 15, 16, 17, 33])
@pytest.mark.parametrize("top_origin", [False, True])
def test_tga_raw_rows_with_image_id(mode: str, width: int, top_origin: bool) -> None:
    size = (width, 3)
    raw = bytes((i * 37) % 256 for i in range(width * 3 * len(mode)))
    source = PillowImage.frombytes(mode, size, raw)
    stream = BytesIO()
    source.save(stream, "TGA", orientation=1 if top_origin else -1, id_section=b"test image id")
    actual = Image.open(stream)
    assert (actual.mode, actual.size, actual.tobytes()) == (mode, size, raw)
    assert actual.tobytes() == PillowImage.open(stream).tobytes()


@pytest.mark.parametrize("width", [1, 7, 8, 9, 15, 16, 17])
def test_binary_pbm_row_padding(width: int) -> None:
    packed = bytes((i * 37 + 0x95) % 256 for i in range((width + 7) // 8 * 3))
    stream = BytesIO(f"P4\n{width} 3\n".encode() + packed)
    actual = Image.open(stream)
    assert (actual.mode, actual.size) == ("1", (width, 3))
    assert actual.tobytes() == PillowImage.open(stream).tobytes()


@pytest.mark.parametrize(("magic", "mode", "channels"), [("P5", "L", 1), ("P6", "RGB", 3), ("P7", "RGBA", 4)])
def test_binary_netpbm_16bit_byte_order(magic: str, mode: str, channels: int) -> None:
    samples = [0, 1, 255, 256, 32768, 65535] * channels
    header = f"{magic}\n3 2\n65535\n" if magic != "P7" else "P7\nWIDTH 3\nHEIGHT 2\nDEPTH 4\nMAXVAL 65535\nTUPLTYPE RGB_ALPHA\nENDHDR\n"
    payload = header.encode() + struct.pack(">" + "H" * len(samples), *samples)
    actual = Image.open(BytesIO(payload))
    assert (actual.mode, actual.size, actual.bit_depth) == (mode, (3, 2), 16)
    assert actual.tobytes() == struct.pack("<" + "H" * len(samples), *samples)
    with pytest.raises(UnidentifiedImageError):
        Image.open(BytesIO(payload[:-1]))


def test_lossy_jpeg2000_decode() -> None:
    source = PillowImage.frombytes("RGB", (13, 7), bytes((i * 37) % 256 for i in range(13 * 7 * 3)))
    stream = BytesIO()
    source.save(stream, "JPEG2000", irreversible=True)
    assert Image.open(stream).tobytes() == PillowImage.open(stream).tobytes()


@pytest.mark.parametrize(("mode", "depth"), [(mode, depth) for mode in ("L", "RGB", "RGBA") for depth in (8, 10, 12, 16)] + [("LA", 8)])
def test_jpeg2000_parallel_codec_preserves_serial_output(monkeypatch: pytest.MonkeyPatch, mode: str, depth: int) -> None:
    size = (513, 513)
    count = size[0] * size[1] * len(mode)
    values = [(i * 719) % (1 << depth) for i in range(count)]
    raw = bytes(values) if depth == 8 else struct.pack("<" + "H" * count, *values)
    source = Image.frombytes(mode, size, raw, bit_depth=depth)
    monkeypatch.setenv("OPJ_NUM_THREADS", "1")
    serial = BytesIO()
    source.save(serial, "JPEG2000")
    expected = Image.open(serial)
    restored = expected if depth in (8, 16) else expected.convert(mode, bit_depth=depth)
    assert restored.tobytes() == raw
    monkeypatch.delenv("OPJ_NUM_THREADS")
    parallel = BytesIO()
    source.save(parallel, "JPEG2000")
    assert parallel.getvalue() == serial.getvalue()
    actual = Image.open(parallel)
    assert (actual.mode, actual.size, actual.bit_depth) == (expected.mode, expected.size, expected.bit_depth)
    assert actual.tobytes() == expected.tobytes()


def test_jpeg2000_parallel_lossy_decode_matches_serial_and_pillow(monkeypatch: pytest.MonkeyPatch) -> None:
    size = (513, 513)
    raw = bytes((i * 37 + i // 13) % 256 for i in range(size[0] * size[1] * 3))
    stream = BytesIO()
    monkeypatch.setenv("OPJ_NUM_THREADS", "1")
    PillowImage.frombytes("RGB", size, raw).save(stream, "JPEG2000", irreversible=True)
    expected = Image.open(stream).tobytes()
    assert expected == PillowImage.open(stream).tobytes()
    monkeypatch.delenv("OPJ_NUM_THREADS")
    assert Image.open(stream).tobytes() == expected


@pytest.mark.parametrize(
    ("extension", "format", "mode"),
    [
        ("ppm", "PPM", "RGB"),
        ("pgm", "PPM", "L"),
        ("pbm", "PPM", "1"),
        ("pnm", "PPM", "RGB"),
        ("pam", "PPM", "RGBA"),
        ("tga", "TGA", "RGBA"),
        ("qoi", "QOI", "RGB"),
        ("jp2", "JPEG2000", "RGB"),
        ("j2k", "JPEG2000", "RGB"),
    ],
)
def test_extended_extension_inference(tmp_path: Path, extension: str, format: str, mode: str) -> None:
    path = tmp_path / f"image.{extension}"
    Image.new(mode, (3, 2)).save(path)
    actual = Image.open(path)
    assert (actual.format, actual.size) == (format, (3, 2))


@pytest.mark.parametrize("format", ["DDS", "EPS"])
def test_readonly_save_rejected(format: str) -> None:
    with pytest.raises(ValueError, match="unsupported image format"):
        Image.new("RGB", (1, 1)).save(BytesIO(), format)


@pytest.mark.parametrize("fourcc", [b"DXT1", b"DXT3", b"DXT5"])
def test_dds_block_compression(fourcc: bytes) -> None:
    header = bytearray(128)
    header[:4] = b"DDS "
    struct.pack_into("<7I", header, 4, 124, 0x81007, 4, 4, 8, 0, 0)
    struct.pack_into("<II4s", header, 76, 32, 4, fourcc)
    struct.pack_into("<I", header, 108, 0x1000)
    colors = struct.pack("<HHI", 0xF800, 0, 0)
    alpha = b"" if fourcc == b"DXT1" else b"\xff" * 8 if fourcc == b"DXT3" else b"\xff\0" + bytes(6)
    stream = BytesIO(bytes(header) + alpha + colors)
    actual = Image.open(stream, formats=["DDS"])
    assert (actual.format, actual.size) == ("DDS", (4, 4))
    assert actual.convert("RGBA").tobytes() == PillowImage.open(stream).convert("RGBA").tobytes() == bytes([255, 0, 0, 255]) * 16


EPS = b"%!PS-Adobe-3.0 EPSF-3.0\n%%BoundingBox: 0 0 4 3\n%%EndComments\n1 0 0 setrgbcolor\n0 0 4 3 rectfill\nshowpage\n%%EOF\n"


def test_eps_missing_ghostscript(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", "")
    assert features.check("eps") is False
    with pytest.raises(UnidentifiedImageError, match="requires Ghostscript"):
        Image.open(BytesIO(EPS))


@pytest.mark.skipif(shutil.which("gs") is None, reason="Ghostscript is optional")
@pytest.mark.parametrize("preview", [False, True])
def test_eps_rasterization(preview: bool) -> None:
    data = struct.pack("<7I", 0xC6D3D0C5, 30, len(EPS), 0, 0, 0, 0) + b"\xff\xff" + EPS if preview else EPS
    actual = Image.open(BytesIO(data), formats=["EPS"])
    assert (actual.format, actual.mode, actual.size) == ("EPS", "RGB", (4, 3))
    assert actual.tobytes() == b"\xff\0\0" * 12


@pytest.mark.parametrize(
    "data",
    [
        b"P6\n99999999 99999999\n255\n",
        b"P7\n",
        b"qoif",
        b"DDS ",
        b"%!PS-Adobe\n",
        b"%!PS-Adobe\n%%BoundingBox: 0 0 9999999 9999999\n",
        b"\xc5\xd0\xd3\xc6",
        b"\0\0\0\x0cjP  \r\n\x87\n",
        b"\xff\x4f\xff\x51",
        bytes([0, 0, 2] + [0] * 9 + [2, 0, 2, 0, 24, 0]),
    ],
)
def test_malformed_extended_inputs(data: bytes) -> None:
    with pytest.raises(UnidentifiedImageError):
        Image.open(BytesIO(data))
