from __future__ import annotations

import tracemalloc

import pytest

from blanket._exif import read_metadata


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
    assert info == {"exif": exif}
    assert peak < 32 * 1024
