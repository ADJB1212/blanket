from __future__ import annotations

import random
import struct
import zlib
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from itertools import starmap

import pytest
from helpers import assert_image_properties, assert_raises, image_buffer, png_chunk
from PIL import Image as PillowImage

from blanket import Image, UnidentifiedImageError


def png_chunks(width: int, height: int, depth: int, color: int, pixels: list[tuple[int, ...]], interlaced: bool, extra: list[tuple[bytes, bytes]]) -> bytes:
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
    return b"\x89PNG\r\n\x1a\n" + png_chunk(b"IHDR", header) + b"".join(starmap(png_chunk, extra)) + png_chunk(b"IDAT", zlib.compress(rows)) + png_chunk(b"IEND", b"")


def test_png_raster_decode_preserves_separately_parsed_compressed_metadata() -> None:
    profile = b"profile data" * 65536
    text = b"image description" * 1024
    pixels = [(i * 17 % 256, i * 37 % 256, i * 53 % 256) for i in range(15)]
    payload = png_chunks(5, 3, 8, 2, pixels, False, [(b"iCCP", b"profile\0\0" + zlib.compress(profile)), (b"zTXt", b"description\0\0" + zlib.compress(text))])
    decoded = Image.open(BytesIO(payload))
    assert decoded.tobytes() == bytes(value for pixel in pixels for value in pixel)
    assert decoded.info["icc_profile"] == profile
    assert decoded.info["description"] == text.decode("latin-1")


@pytest.mark.parametrize(("color", "depth"), [(0, depth) for depth in (1, 2, 4, 8, 16)] + [(3, depth) for depth in (1, 2, 4, 8)] + [(color, depth) for color in (2, 4, 6) for depth in (8, 16)])
def test_png_single_decoder_preserves_layout_and_transparency(color: int, depth: int, subtests: pytest.Subtests) -> None:
    for interlaced in [False, True]:
        for transparent in [False, True]:
            for size in [(1, 1), (17, 9)]:
                with subtests.test(color=color, depth=depth, interlaced=interlaced, transparent=transparent, size=size):
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
                    assert_image_properties(actual, size=size, format="PNG")
                    if depth < 16:
                        assert actual.bit_depth == 8
                        assert actual.convert("RGBA").tobytes() == external.convert("RGBA").tobytes()
                        modes = {0: "LA" if transparent else "1" if depth == 1 else "L", 2: "RGBA" if transparent else "RGB", 3: "P", 4: "LA", 6: "RGBA"}
                        assert actual.mode == modes[color]
                        if color == 3:
                            assert actual.tobytes() == external.tobytes()
                            assert actual.getpalette() == external.getpalette()
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
                        assert_image_properties(actual, mode="RGBA" if color in (4, 6) or transparent else "L" if color == 0 else "RGB", bit_depth=16, pixels=struct.pack("<" + "H" * len(expected), *expected))


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
    assert_image_properties(image, bit_depth=depth, pixels=struct.pack("<" + "H" * len(expected), *expected))


@pytest.mark.parametrize("default_image", [False, True])
def test_png_decoder_keeps_first_image_for_animated_containers(default_image: bool) -> None:
    first = PillowImage.new("RGBA", (17, 9), (100, 30, 20, 127))
    second = PillowImage.new("RGBA", first.size, (20, 30, 100, 255))
    encoded = image_buffer(first, "PNG", save_all=True, append_images=[second], default_image=default_image)
    actual = Image.open(encoded)
    assert_image_properties(actual, size=first.size, mode=first.mode, pixels=first.tobytes())


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
        encoded = image_buffer(image, format, **({"compress_level": level} if level is not None else {}))
        actual = Image.open(encoded)
        scaled = [(value * 65535 + maximum // 2) // maximum for value in values]
        expected = raw if format == "PNG" else struct.pack("<" + "H" * len(scaled), *scaled)
        assert_image_properties(actual, pixels=expected, bit_depth=depth if format == "PNG" else 16)
        external = PillowImage.open(encoded)
        if mode == "L":
            assert list(external.get_flattened_data()) == scaled
        else:
            assert external.tobytes() == bytes(value >> 8 for value in scaled)
        assert image.tobytes() == raw


@pytest.mark.parametrize("values", [[0], [255], [3, 17, 249], list(range(256))])
@pytest.mark.parametrize("count", [1, 17, 4097])
def test_grayscale_gif_preserves_exact_samples(values: list[int], count: int) -> None:
    raw = bytes(values[i % len(values)] for i in range(count))
    image = Image.frombytes("L", (count, 1), raw)
    encoded = image_buffer(image, "GIF")
    assert Image.open(encoded).convert("L").tobytes() == raw
    assert PillowImage.open(encoded).convert("L").tobytes() == raw
    assert image.tobytes() == raw


@pytest.mark.parametrize("mode", ["1", "L", "RGB", "RGBA"])
def test_png_deflate_levels(mode: str, subtests: pytest.Subtests) -> None:
    for level in range(10):
        for random_pixels in [False, True]:
            with subtests.test(mode=mode, level=level, random_pixels=random_pixels):
                size = (129, 65)
                count = ((size[0] + 7) // 8) * size[1] if mode == "1" else size[0] * size[1] * len(mode)
                raw = random.Random(61).randbytes(count) if random_pixels else bytes(i % 251 for i in range(count))
                image = Image.frombytes(mode, size, raw)
                raw = image.tobytes()
                encoded = image_buffer(image, "PNG", compress_level=level)
                for decoder in (Image.open, PillowImage.open):
                    decoded = decoder(BytesIO(encoded.getvalue()))
                    assert_image_properties(decoded, mode=mode, size=size, pixels=raw)


def test_jxl_reused_runners_are_independent() -> None:
    cases = []
    for size in ((17, 13), (513, 129), (65, 513)):
        for mode in ("L", "RGB", "RGBA"):
            raw = random.Random(71).randbytes(size[0] * size[1] * len(mode))
            encoded = image_buffer(Image.frombytes(mode, size, raw), "JXL", lossless=True, effort=1)
            cases.append((encoded.getvalue(), mode, size, raw))

    def decode_cases(_: int) -> None:
        for payload, mode, size, raw in cases + cases[::-1]:
            assert_raises(UnidentifiedImageError, lambda: Image.open(BytesIO(b"\xff\x0a\0")))
            decoded = Image.open(BytesIO(payload))
            assert_image_properties(decoded, mode=mode, size=size, pixels=raw)

    decode_cases(0)
    with ThreadPoolExecutor(max_workers=3) as executor:
        list(executor.map(decode_cases, range(3)))
