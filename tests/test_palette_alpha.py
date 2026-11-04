from __future__ import annotations

import struct
import tracemalloc

import pytest
from PIL import Image as PillowImage

from blanket import Image


@pytest.mark.parametrize("mode", ["PA", "LA", "RGBA"])
def test_indexed_alpha_conversion_keeps_full_image_buffers_native(mode: str) -> None:
    image = Image.new("P", (1024, 1024))
    image.putpalette(bytes(range(256)) * 3)
    image.info["transparency"] = b"\x00\x7f"
    tracemalloc.start()
    try:
        result = image.convert(mode)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert result.mode == mode and result.size == image.size
    assert peak < 64 * 1024


@pytest.mark.parametrize("palette_mode", ["RGB", "RGBA"])
@pytest.mark.parametrize("transparency", [None, 1, 300, b"\x00\x7f"])
@pytest.mark.parametrize("count", [0, 1, 17, 16385, 131073])
def test_indexed_to_pa_preserves_indices_palette_and_alpha(palette_mode: str, transparency: int | bytes | None, count: int) -> None:
    indices = (bytes(range(256)) * (count // 256 + 1))[:count]
    image = Image.frombytes("P", (count, 1), indices)
    entries = bytes((i * 71 + 13) % 256 for i in range(7 * len(palette_mode)))
    image.putpalette(entries, palette_mode)
    image.info["custom"] = "retained"
    if transparency is not None:
        image.info["transparency"] = transparency
    result = image.convert("PA")
    assert result.mode == "PA" and result.size == image.size
    raw = result.tobytes()
    assert raw[::2] == indices
    table = [entries[i * 4 + 3] if palette_mode == "RGBA" and i < 7 else 255 for i in range(256)]
    if isinstance(transparency, int):
        table = [0 if i == transparency else 255 for i in range(256)]
    elif isinstance(transparency, bytes):
        table = list(transparency.ljust(256, b"\xff"))
    assert raw[1::2] == bytes(table[i] for i in indices)
    assert result.getpalette(None) == list(entries)
    assert result.palette is not image.palette
    assert result.info == {"custom": "retained"}
    assert image.tobytes() == indices


@pytest.mark.parametrize("mode", ["LA", "RGBA"])
@pytest.mark.parametrize("count", [0, 1, 17, 16385])
def test_pa_conversion_reads_original_alpha(mode: str, count: int) -> None:
    channels = len(mode)
    raw = bytes((i * 37 + 13) % 256 for i in range(count * channels))
    image = Image.frombytes(mode, (count, 1), raw)
    indexed = image.convert("RGB").quantize()
    result = image.convert("PA")
    assert result.tobytes()[::2] == indexed.tobytes()
    assert result.tobytes()[1::2] == raw[channels - 1 :: channels]
    assert result.getpalette() == indexed.getpalette()
    assert image.tobytes() == raw


@pytest.mark.parametrize("mode", ["LA", "RGBA"])
@pytest.mark.parametrize("palette_mode", ["RGB", "RGBA"])
@pytest.mark.parametrize("transparency", [1, b"\x00\x7f"])
@pytest.mark.parametrize("count", [0, 17, 131073])
def test_indexed_transparency_expansion_matches_explicit_table(mode: str, palette_mode: str, transparency: int | bytes, count: int) -> None:
    indices = (bytes(range(256)) * (count // 256 + 1))[:count]
    image = Image.frombytes("P", (count, 1), indices)
    entries = bytes((i * 71 + 13) % 256 for i in range(7 * len(palette_mode)))
    image.putpalette(entries, palette_mode)
    image.info["transparency"] = transparency
    colors = bytearray()
    for i in range(256):
        rgb = entries[i * len(palette_mode) : i * len(palette_mode) + 3] if i < 7 else bytes(3)
        original_alpha = entries[i * 4 + 3] if palette_mode == "RGBA" and i < 7 else 255
        alpha = (0 if i == transparency else original_alpha) if isinstance(transparency, int) else (transparency[i] if i < len(transparency) else 255)
        colors.extend((*rgb, alpha))
    table = PillowImage.frombytes("RGBA", (256, 1), bytes(colors)).convert(mode).tobytes()
    result = image.convert(mode)
    expected = b"".join(table[i * len(mode) : (i + 1) * len(mode)] for i in indices)
    assert result.tobytes() == expected
    assert "transparency" not in result.info and image.info["transparency"] == transparency
    assert image.tobytes() == indices and image.getpalette(None) == list(entries)
    if mode == "RGBA":
        wide = image.convert(mode, bit_depth=16)
        assert wide.bit_depth == 16
        assert wide.tobytes() == struct.pack(f"<{len(expected)}H", *(value * 257 for value in expected))


def test_indexed_alpha_conversion_rejects_invalid_tables_and_closed_images() -> None:
    image = Image.new("P", (1, 1))
    image.info["transparency"] = bytes(257)
    for mode in ("PA", "LA", "RGBA"):
        with pytest.raises(ValueError):
            image.convert(mode)
    image.close()
    with pytest.raises(ValueError, match="closed"):
        image.convert("PA")
