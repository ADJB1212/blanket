from __future__ import annotations

import tracemalloc

import pytest

from blanket import Image, ImageOps
from blanket._exif import read_metadata


def test_jpeg_metadata_joins_exif_fragments_in_marker_order():
    exif = Image.Exif()
    exif[305] = "fragmented" * 256
    raw = exif.tobytes()
    fragments = [raw[:6] + raw[i : i + 17] for i in range(6, len(raw), 17)]
    data = b"\xff\xd8" + b"".join(b"\xff\xe1" + (len(part) + 2).to_bytes(2, "big") + part for part in fragments) + b"\xff\xd9"
    assert read_metadata(data, "JPEG")["exif"] == raw


@pytest.mark.parametrize("parts", [[(2, 3), (1, 3), (3, 3)], [(1, 3), (3, 3)], [(1, 2), (1, 2)], [(1, 2), (2, 3)], [(0, 1)]])
def test_jpeg_icc_fragments_require_a_complete_consistent_set(parts):
    data = b"\xff\xd8"
    for part, count in parts:
        payload = b"ICC_PROFILE\0" + bytes((part, count)) + bytes((part,))
        data += b"\xff\xe2" + (len(payload) + 2).to_bytes(2, "big") + payload
    info = read_metadata(data + b"\xff\xd9", "JPEG")
    if parts == [(2, 3), (1, 3), (3, 3)]:
        assert info["icc_profile"] == b"\x01\x02\x03"
    else:
        assert "icc_profile" not in info


def test_exif_transpose_reload_uses_updated_metadata() -> None:
    image = Image.new("RGB", (3, 2), "red")
    image.info["xmp"] = b'<rdf:Description tiff:Orientation="3"/>'
    assert image.getexif()[274] == 3
    result = ImageOps.exif_transpose(image)
    result._reload_exif()
    assert 274 not in result.getexif()


@pytest.mark.parametrize("format", ["JPEG", "PNG", "JXL"])
def test_metadata_skips_large_pixel_payloads(format: str) -> None:
    exif = b"Exif\0\0II*\0\x08\0\0\0\0\0"
    if format == "JPEG":
        payload = b"x" * 60000
        segment = b"\xff\xe2" + (len(payload) + 2).to_bytes(2, "big") + payload
        data = b"\xff\xd8" + segment * 32 + b"\xff\xe1" + (len(exif) + 2).to_bytes(2, "big") + exif + b"\xff\xd9"
    elif format == "PNG":
        payload = b"x" * (2 * 1024 * 1024)
        data = b"\x89PNG\r\n\x1a\n" + len(payload).to_bytes(4, "big") + b"IDAT" + payload + bytes(4)
        data += (len(exif) - 6).to_bytes(4, "big") + b"eXIf" + exif[6:] + bytes(4)
    else:
        payload = b"x" * (2 * 1024 * 1024)
        data = b"\0\0\0\x0cJXL \r\n\x87\n" + (len(payload) + 8).to_bytes(4, "big") + b"jxlc" + payload
        data += (len(exif) + 6).to_bytes(4, "big") + b"Exif" + bytes(4) + exif[6:]
    tracemalloc.start()
    try:
        info = read_metadata(data, format)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert info == ({"exif": exif, "dpi": (72, 72)} if format == "JPEG" else {"exif": exif})
    assert peak < 32 * 1024


@pytest.mark.parametrize("kind", [b"eXIf", b"tEXt"])
def test_png_metadata_skips_oversized_uncompressed_payloads_before_copying(kind: bytes, monkeypatch) -> None:
    monkeypatch.setattr("blanket._exif._MAX_PNG_METADATA_BYTES", 100)
    payload = b"key\0" + b"x" * (2 * 1024 * 1024)
    data = b"\x89PNG\r\n\x1a\n" + len(payload).to_bytes(4, "big") + kind + payload + bytes(4)
    data += (8).to_bytes(4, "big") + b"tEXt" + b"kept\0yes" + bytes(4)
    tracemalloc.start()
    try:
        info = read_metadata(data, "PNG")
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert info == {"kept": "yes"}
    assert peak < 32 * 1024
