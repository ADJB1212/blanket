from __future__ import annotations

import struct
import zlib
from io import BytesIO

import pytest

from blanket import Image, ImageOps
from blanket._exif import _png_metadata, read_metadata, transpose_metadata


def png_chunk(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload))


@pytest.mark.parametrize("kind", [b"tEXt", b"zTXt", b"iTXt", b"iCCP", b"eXIf"])
def test_png_metadata_ignores_chunks_after_iend(kind: bytes) -> None:
    output = BytesIO()
    Image.new("L", (1, 1)).save(output, "PNG")
    payload = {
        b"tEXt": b"key\0value",
        b"zTXt": b"key\0\0" + zlib.compress(b"value"),
        b"iTXt": b"key\0\1\0\0\0" + zlib.compress(b"value"),
        b"iCCP": b"key\0\0" + zlib.compress(b"value"),
        b"eXIf": b"II*\0\x08\0\0\0" + bytes(6),
    }[kind]
    data = output.getvalue() + png_chunk(kind, payload)
    assert read_metadata(data, "PNG") == {}
    assert Image.open(BytesIO(data)).info == {}


@pytest.mark.parametrize("kind", [b"tEXt", b"zTXt", b"iTXt", b"iCCP", b"eXIf"])
@pytest.mark.parametrize("repeated", [False, True])
def test_png_metadata_has_a_cumulative_budget(monkeypatch, kind: bytes, repeated: bool) -> None:
    monkeypatch.setattr("blanket._exif._MAX_PNG_METADATA_BYTES", 100)
    output = BytesIO()
    Image.new("L", (1, 1)).save(output, "PNG")
    chunks = []
    for i in range(10):
        key = b"key" if repeated else f"key{i}".encode()
        value = bytes([65 + i]) * 40
        payload = {
            b"tEXt": key + b"\0" + value,
            b"zTXt": key + b"\0\0" + zlib.compress(value),
            b"iTXt": key + b"\0\1\0\0\0" + zlib.compress(value),
            b"iCCP": key + b"\0\0" + zlib.compress(value),
            b"eXIf": value,
        }[kind]
        chunks.append(png_chunk(kind, payload))
    data = output.getvalue()
    data = data[:33] + b"".join(chunks) + data[33:]
    info = read_metadata(data, "PNG")
    assert info
    assert sum(len(value) for value in info.values()) <= 100
    assert all(b"C" not in value if isinstance(value, bytes) else "C" not in value for value in info.values())


@pytest.mark.parametrize("kind", [b"zTXt", b"iTXt", b"iCCP"])
def test_png_metadata_retains_per_chunk_expansion_limit(kind: bytes) -> None:
    raw = zlib.compress(b"x" * (1024 * 1024 + 1))
    prefix = b"key\0\1\0\0\0" if kind == b"iTXt" else b"key\0\0"
    info = {}
    assert _png_metadata(info, kind, prefix + raw) == 0
    assert info == {}


@pytest.mark.parametrize(
    ("data", "format"),
    [
        (b"\xff\xd8\xff\xff", "JPEG"),
        (b"\xff\xd8\xff\xff\xff\xff", "JPEG"),
        (b"\xff\xd8\xff\xff\xe1\0", "JPEG"),
        (b"\xff\xd8\xff\xff\xd9\0", "JPEG"),
        (b"\xff\xd8\xff\x01\xff", "JPEG"),
        (b"\xff\xd8\xff\xd0\xff\xe1\0", "JPEG"),
        (b"\xff\xd8\xff\xe1\0\x01", "JPEG"),
        (b"\xff\xd8\xff\xe1\0\x10", "JPEG"),
        (b"\x89PNG\r\n\x1a\n\0\0\0\x20tEXt" + bytes(4), "PNG"),
        (b"\0\0\0\x0cJXL \r\n\x87\n\0\0\0\x01xml ", "JXL"),
        (b"\0\0\0\x0cJXL \r\n\x87\n\0\0\0\x04xml ", "JXL"),
        (b"RIFF\0\0\0\0WEBPEXIF\xff\xff\xff\xff", "WEBP"),
    ],
)
def test_truncated_metadata_containers(data: bytes, format: str) -> None:
    assert read_metadata(data, format) == {}


def test_jxl_extended_and_final_metadata_boxes() -> None:
    signature = b"\0\0\0\x0cJXL \r\n\x87\n"
    extended = struct.pack(">I4sQ", 1, b"xml ", 19) + b"xmp"
    assert read_metadata(signature + extended, "JXL") == {"xmp": b"xmp"}
    assert read_metadata(signature + struct.pack(">I4s", 0, b"xml ") + b"xmp", "JXL") == {"xmp": b"xmp"}


@pytest.mark.parametrize(("kind", "payload"), [(b"tEXt", b"\0value"), (b"zTXt", b"key\0\x01data"), (b"iTXt", b"key\0\x02\0\0\0data")])
def test_invalid_png_text_fields(kind: bytes, payload: bytes) -> None:
    info = {}
    _png_metadata(info, kind, payload)
    assert info == {}


def test_raw_exif_profile_and_xmp_orientation() -> None:
    exif = Image.Exif()
    exif.load(b"")
    exif[274] = 6
    image = Image.new("L", (2, 1))
    image.info["Raw profile type exif"] = exif.tobytes().hex()
    assert image.getexif()[274] == 6
    assert ImageOps.exif_transpose(image).size == (1, 2)
    for value in ("invalid hex", b"", (b"<tiff:Orientation>6</tiff:Orientation>",)):
        info = {"Raw profile type exif": "invalid hex", "xmp": value}
        orientation, _ = transpose_metadata(info)
        assert orientation == (6 if isinstance(value, tuple) else 1)


@pytest.mark.parametrize("raw", [b"II\0\0\x08\0\0\0\0\0\0\0\0\0", b"II*\0\x08\0\0\0\x01\0\0\0\0\0"])
def test_orientation_ignores_invalid_tiff_headers(raw: bytes) -> None:
    assert transpose_metadata({"exif": raw}) == (1, {"exif": raw})


@pytest.mark.parametrize(("format", "options"), [("GIF", {"exif": b""}), ("JPEG", {"pnginfo": None})])
def test_metadata_options_reject_unsupported_containers(format: str, options: dict) -> None:
    with pytest.raises(TypeError):
        Image.new("RGB", (1, 1)).save(BytesIO(), format, **options)


def test_exif_mapping_stream_loading_and_signed_rationals() -> None:
    exif = Image.Exif()
    exif[274] = 1
    del exif[274]
    assert list(exif) == []
    exif[65000] = -0.5
    exif[65001] = b"opaque"
    data = exif.tobytes()[6:]
    loaded = Image.Exif()
    loaded.load_from_fp(BytesIO(data))
    assert loaded[65000] == -0.5
    loaded.load_from_fp(BytesIO(data), int.from_bytes(data[4:8], "big"))
    assert loaded[65000] == -0.5


@pytest.mark.parametrize(("kind", "length", "field", "expected"), [(0, 1, 0, {}), (7, 10, 1000, None)])
def test_exif_unknown_types_and_truncated_values(kind, length, field, expected) -> None:
    raw = b"II*\0\x08\0\0\0" + struct.pack("<HHHII", 1, 65000, kind, length, field) + bytes(4)
    exif = Image.Exif()
    if expected is None:
        with pytest.raises(ValueError, match="truncated EXIF value"):
            exif.load(raw)
    else:
        exif.load(raw)
        assert dict(exif) == expected


def test_exif_rejects_cycles_and_excessive_tags() -> None:
    exif = Image.Exif()
    nested = {}
    nested[34665] = nested
    exif[34665] = nested
    with pytest.raises(ValueError, match="cyclic"):
        exif.tobytes()
    exif = Image.Exif()
    exif.update(dict.fromkeys(range(65536), 0))
    with pytest.raises(ValueError, match="too many"):
        exif.tobytes()


@pytest.mark.parametrize("options", [{"append_images": 1}, {"default_image": 1}, {"default_image": True}, {"duration": "20"}])
def test_sequence_control_types(options: dict) -> None:
    with pytest.raises((TypeError, ValueError)):
        Image.new("RGB", (1, 1)).save(BytesIO(), "PNG", save_all=True, **options)


def test_sequence_rejects_unsupported_mode() -> None:
    with pytest.raises(OSError, match="cannot write mode HSV"):
        Image.new("HSV", (1, 1)).save(BytesIO(), "PNG", save_all=True)
    with pytest.raises(ValueError, match="high-bit-depth"):
        Image.frombytes("L", (1, 1), b"\x01\0", bit_depth=16).save(BytesIO(), "GIF", save_all=True)


def test_single_frame_animation_controls() -> None:
    output = BytesIO()
    Image.new("RGB", (1, 1), "red").save(output, "GIF", duration=40)
    image = Image.open(output)
    assert image.n_frames == 1
    assert image.info["duration"] == 40
