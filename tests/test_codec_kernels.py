from __future__ import annotations

import random
import struct
import zlib
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import pytest
from PIL import Image as PillowImage

from blanket import Image, UnidentifiedImageError


def png_chunks(width: int, height: int, depth: int, color: int, pixels: list[tuple[int, ...]], interlaced: bool, extra: list[tuple[bytes, bytes]]) -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    passes = [(0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8), (2, 0, 4, 4), (0, 2, 2, 4), (1, 0, 2, 2), (0, 1, 1, 2)] if interlaced else [(0, 0, 1, 1)]
    rows = bytearray()
    for left, top, dx, dy in passes:
        if left >= width:
            continue
        for y in range(top, height, dy):
            values = [value for x in range(left, width, dx) for value in pixels[y * width + x]]
            rows.append(0)
            if depth == 16:
                rows.extend(struct.pack(">" + "H" * len(values), *values))
            elif depth == 8:
                rows.extend(values)
            else:
                packed = bytearray((len(values) * depth + 7) // 8)
                for i, value in enumerate(values):
                    packed[i * depth // 8] |= value << (8 - depth - i * depth % 8)
                rows.extend(packed)
    header = struct.pack(">IIBBBBB", width, height, depth, color, 0, 0, int(interlaced))
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + b"".join(chunk(kind, data) for kind, data in extra) + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b"")


@pytest.mark.parametrize(("color", "depth"), [(0, depth) for depth in (1, 2, 4, 8, 16)] + [(3, depth) for depth in (1, 2, 4, 8)] + [(color, depth) for color in (2, 4, 6) for depth in (8, 16)])
@pytest.mark.parametrize("interlaced", [False, True])
@pytest.mark.parametrize("transparent", [False, True])
@pytest.mark.parametrize("size", [(1, 1), (17, 9)])
def test_png_single_decoder_preserves_layout_and_transparency(color: int, depth: int, interlaced: bool, transparent: bool, size: tuple[int, int]) -> None:
    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[color]
    maximum = (1 << depth) - 1
    pixels = [tuple((i * 73 + i // 5 + c * 17) & maximum for c in range(channels)) for i in range(size[0] * size[1])]
    extra = []
    if color == 3:
        extra.append((b"PLTE", bytes(value for i in range(maximum + 1) for value in (i, i ^ 127, i ^ 255))))
        if transparent:
            extra.append((b"tRNS", bytes(i % 256 for i in range(maximum + 1))))
    elif transparent and color in (0, 2):
        extra.append((b"tRNS", struct.pack(">" + "H" * channels, *pixels[0])))
    encoded = png_chunks(*size, depth, color, pixels, interlaced, extra)
    actual = Image.open(BytesIO(encoded))
    external = PillowImage.open(BytesIO(encoded))
    assert actual.size == size
    assert actual.format == "PNG"
    if depth < 16:
        assert actual.bit_depth == 8
        assert actual.convert("RGBA").tobytes() == external.convert("RGBA").tobytes()
        modes = {0: "LA" if transparent else "1" if depth == 1 else "L", 2: "RGBA" if transparent else "RGB", 3: "RGBA" if transparent else "RGB", 4: "LA", 6: "RGBA"}
        assert actual.mode == modes[color]
    else:
        expected = []
        for pixel in pixels:
            if color == 4:
                expected.extend((pixel[0], pixel[0], pixel[0], pixel[1]))
            elif transparent and color == 0:
                expected.extend((pixel[0], pixel[0], pixel[0], 0 if pixel == pixels[0] else maximum))
            elif transparent and color == 2:
                expected.extend((*pixel, 0 if pixel == pixels[0] else maximum))
            else:
                expected.extend(pixel)
        assert actual.mode == ("RGBA" if color in (4, 6) or transparent else "L" if color == 0 else "RGB")
        assert actual.bit_depth == 16
        assert actual.tobytes() == struct.pack("<" + "H" * len(expected), *expected)


@pytest.mark.parametrize(("color", "channels"), [(0, 1), (2, 3), (4, 2), (6, 4)])
@pytest.mark.parametrize("bits", [8, 10, 12, 16, None])
def test_png_sbit_preserves_supported_depth_and_grayscale_alpha(color: int, channels: int, bits: int | None) -> None:
    pixels = [tuple((i * 137 + c * 911) % 65536 for c in range(channels)) for i in range(15)]
    significant = bytes([bits] * channels) if bits is not None else bytes(range(10, 10 + channels))
    image = Image.open(BytesIO(png_chunks(5, 3, 16, color, pixels, False, [(b"sBIT", significant)])))
    depth = significant[0] if color != 4 and len(set(significant)) == 1 and significant[0] in (10, 12) else 16
    expected = []
    for pixel in pixels:
        expanded = (pixel[0], pixel[0], pixel[0], pixel[1]) if color == 4 else pixel
        expected.extend(value >> (16 - depth) for value in expanded)
    assert image.bit_depth == depth
    assert image.tobytes() == struct.pack("<" + "H" * len(expected), *expected)


@pytest.mark.parametrize("default_image", [False, True])
def test_png_decoder_keeps_first_image_for_animated_containers(default_image: bool) -> None:
    first = PillowImage.new("RGBA", (17, 9), (100, 30, 20, 127))
    second = PillowImage.new("RGBA", first.size, (20, 30, 100, 255))
    encoded = BytesIO()
    first.save(encoded, "PNG", save_all=True, append_images=[second], default_image=default_image)
    actual = Image.open(encoded)
    assert actual.size == first.size
    assert actual.mode == first.mode
    assert actual.tobytes() == first.tobytes()


@pytest.mark.parametrize("format", ["PNG", "TIFF"])
@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
@pytest.mark.parametrize("depth", [10, 12, 16])
def test_wide_encoding_reuses_sample_storage_with_exact_byte_order(format: str, mode: str, depth: int) -> None:
    size = (37, 3)
    maximum = (1 << depth) - 1
    values = [(i * 137 + i // 7) & maximum for i in range(size[0] * size[1] * len(mode))]
    raw = struct.pack("<" + "H" * len(values), *values)
    image = Image.frombytes(mode, size, raw, bit_depth=depth)
    for level in (0, 6) if format == "PNG" else (None,):
        encoded = BytesIO()
        image.save(encoded, format, **({"compress_level": level} if level is not None else {}))
        actual = Image.open(encoded)
        scaled = [(value * 65535 + maximum // 2) // maximum for value in values]
        expected = raw if format == "PNG" else struct.pack("<" + "H" * len(scaled), *scaled)
        assert actual.tobytes() == expected
        assert actual.bit_depth == (depth if format == "PNG" else 16)
        external = PillowImage.open(encoded)
        if mode == "L":
            assert list(external.get_flattened_data()) == scaled
        else:
            assert external.tobytes() == bytes(value >> 8 for value in scaled)
        assert image.tobytes() == raw


@pytest.mark.parametrize("mode", ["1", "L", "RGB", "RGBA"])
@pytest.mark.parametrize("level", range(10))
@pytest.mark.parametrize("random_pixels", [False, True])
def test_png_deflate_levels(mode: str, level: int, random_pixels: bool) -> None:
    size = (129, 65)
    count = ((size[0] + 7) // 8) * size[1] if mode == "1" else size[0] * size[1] * len(mode)
    raw = random.Random(61).randbytes(count) if random_pixels else bytes(i % 251 for i in range(count))
    image = Image.frombytes(mode, size, raw)
    raw = image.tobytes()
    encoded = BytesIO()
    image.save(encoded, "PNG", compress_level=level)
    for decoder in (Image.open, PillowImage.open):
        decoded = decoder(BytesIO(encoded.getvalue()))
        assert decoded.mode == mode
        assert decoded.size == size
        assert decoded.tobytes() == raw


def test_jxl_reused_runners_are_independent() -> None:
    cases = []
    for size in ((17, 13), (513, 129), (65, 513)):
        for mode in ("L", "RGB", "RGBA"):
            raw = random.Random(71).randbytes(size[0] * size[1] * len(mode))
            encoded = BytesIO()
            Image.frombytes(mode, size, raw).save(encoded, "JXL", lossless=True, effort=1)
            cases.append((encoded.getvalue(), mode, size, raw))

    def decode_cases(_: int) -> None:
        for payload, mode, size, raw in cases + cases[::-1]:
            with pytest.raises(UnidentifiedImageError):
                Image.open(BytesIO(b"\xff\x0a\0"))
            decoded = Image.open(BytesIO(payload))
            assert decoded.mode == mode
            assert decoded.size == size
            assert decoded.tobytes() == raw

    decode_cases(0)
    with ThreadPoolExecutor(max_workers=3) as executor:
        list(executor.map(decode_cases, range(3)))
