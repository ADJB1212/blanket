from __future__ import annotations

import io
import os
import random
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image as PillowImage

from blanket import Image
from blanket.Compressor import LosslessImageCompressor


@pytest.mark.parametrize("effort", [1, 7])
@pytest.mark.parametrize("gray", [False, True])
def test_png_reduces_rgba_channels(effort: int, gray: bool) -> None:
    rng = random.Random(48)
    size = (129, 65)
    pixels = bytearray()
    for i in range(size[0] * size[1]):
        if gray:
            value = rng.randrange(256)
            pixels.extend((value, value, value, i % 256))
        else:
            pixels.extend((*rng.randbytes(3), 255))
    original = Image.frombytes("RGBA", size, bytes(pixels))
    baseline, output = io.BytesIO(), io.BytesIO()
    original.save(baseline, "PNG", compress_level=9)
    original.save(output, "PNG", compress_level=9, compressor=LosslessImageCompressor(effort=effort))
    encoded = output.getvalue()
    assert encoded[25] == (4 if gray else 2)  # IHDR color type: LA or RGB
    assert len(encoded) < len(baseline.getvalue())
    assert Image.open(io.BytesIO(encoded)).convert("RGBA").tobytes() == bytes(pixels)
    with PillowImage.open(io.BytesIO(encoded)) as decoded:
        assert decoded.convert("RGBA").tobytes() == bytes(pixels)
    assert original.tobytes() == bytes(pixels)


def test_compressor_output_is_identical_across_worker_counts() -> None:
    code = """
import hashlib, io, random
from blanket import Image
from blanket.Compressor import LosslessImageCompressor
size = (513, 257)
palette = [bytes((i, i * 71 % 256, i * 133 % 256, i % 3 * 127)) for i in range(256)]
rng = random.Random(91)
images = [Image.frombytes('RGBA', size, b''.join(palette[i] for i in rng.randbytes(size[0] * size[1]))),
          Image.frombytes('RGB', size, rng.randbytes(size[0] * size[1] * 3))]
for image, fmt in [(images[0], 'PNG'), (images[1], 'PNG'), (images[1], 'JPEG')]:
    output = io.BytesIO()
    image.save(output, fmt, compressor=LosslessImageCompressor(effort=7))
    print(hashlib.sha256(output.getvalue()).hexdigest())
"""
    results = []
    for workers in (1, 4):
        process = subprocess.run([sys.executable, "-c", code], env={**os.environ, "RAYON_NUM_THREADS": str(workers)}, capture_output=True, text=True, check=True)
        results.append(process.stdout)
    assert results[0] == results[1]


@pytest.mark.parametrize("effort", [1, 4, 6, 7, 10])
@pytest.mark.parametrize("mode,kind", [("L", "gray"), ("RGB", "gray"), ("RGBA", "gray"), ("RGBA", "opaque"), ("RGBA", "transparent")])
def test_png_optimization_roundtrip(effort: int, mode: str, kind: str) -> None:
    pixels = bytearray()
    for i in range(64 * 64):
        value = (i * 37 + i // 64 * 19) % 256
        if mode == "L":
            pixels.append(value)
        elif mode == "RGB":
            pixels.extend([value] * 3)
        elif kind == "gray":
            pixels.extend([value, value, value, i % 256])
        else:
            pixels.extend([value, (value + 71) % 256, (value + 133) % 256, 255 if kind == "opaque" else 0])
    original = Image.frombytes(mode, (64, 64), bytes(pixels))
    original.info["custom"] = b"metadata"
    baseline, output = io.BytesIO(), io.BytesIO()
    original.save(baseline, "PNG", compress_level=1)
    compressor = LosslessImageCompressor(effort=effort)
    assert compressor.effort == effort
    original.save(output, "PNG", compress_level=1, compressor=compressor)
    assert len(output.getvalue()) <= len(baseline.getvalue())
    assert original.mode == mode
    assert original.format is None
    assert original.tobytes() == bytes(pixels)
    assert original.info == {"custom": b"metadata"}
    with PillowImage.open(io.BytesIO(output.getvalue())) as decoded:
        assert decoded.convert(mode).tobytes() == bytes(pixels)
    assert Image.open(io.BytesIO(output.getvalue())).convert(mode).tobytes() == bytes(pixels)
    unchanged = io.BytesIO()
    original.save(unchanged, "PNG", compress_level=1, compressor=None)
    assert unchanged.getvalue() == baseline.getvalue()
    repeated = io.BytesIO()
    original.save(repeated, "PNG", compress_level=1, compressor=compressor)
    assert repeated.getvalue() == output.getvalue()


def test_output_format_controls_optimization(tmp_path: Path) -> None:
    source = io.BytesIO()
    Image.new("RGB", (128, 128), (71, 71, 71)).save(source, "TIFF")
    original = Image.open(io.BytesIO(source.getvalue()))
    output = tmp_path / "optimized.png"
    baseline = io.BytesIO()
    original.save(baseline, "PNG", compress_level=0)
    original.save(output, compressor=LosslessImageCompressor(), compress_level=0)
    assert output.stat().st_size < len(baseline.getvalue()) // 2
    with PillowImage.open(output) as decoded:
        assert decoded.convert("RGB").tobytes() == original.tobytes()


@pytest.mark.parametrize("colors,depth", [(2, 1), (4, 2), (16, 4), (256, 8)])
@pytest.mark.parametrize("mode", ["RGB", "RGBA"])
def test_exact_palette_png(colors: int, depth: int, mode: str) -> None:
    rng = random.Random(42)
    palette = [bytes((i, (i * 71) % 256, (i * 133) % 256, i % 3 * 127))[: len(mode)] for i in range(colors)]
    # Odd widths exercise byte padding independently at every row.
    pixels = b"".join(palette[rng.randrange(colors)] for _ in range(129 * 127))
    original = Image.frombytes(mode, (129, 127), pixels)
    baseline, output = io.BytesIO(), io.BytesIO()
    original.save(baseline, "PNG")
    original.save(output, "PNG", compressor=LosslessImageCompressor())
    encoded = output.getvalue()
    assert encoded[24:26] == bytes((depth, 3))
    assert len(encoded) < len(baseline.getvalue())
    with PillowImage.open(io.BytesIO(encoded)) as decoded:
        assert decoded.convert(mode).tobytes() == pixels
    assert Image.open(io.BytesIO(encoded)).convert(mode).tobytes() == pixels
    assert original.tobytes() == pixels
    repeated = io.BytesIO()
    original.save(repeated, "PNG", compressor=LosslessImageCompressor())
    assert repeated.getvalue() == encoded


def test_palette_overflow_keeps_all_colors() -> None:
    pixels = b"".join(bytes((i % 256, i // 256, 19, 0)) for i in range(257))
    original = Image.frombytes("RGBA", (257, 1), pixels)
    baseline, output = io.BytesIO(), io.BytesIO()
    original.save(baseline, "PNG")
    original.save(output, "PNG", compressor=LosslessImageCompressor())
    assert len(output.getvalue()) <= len(baseline.getvalue())
    with PillowImage.open(io.BytesIO(output.getvalue())) as decoded:
        assert decoded.convert("RGBA").tobytes() == pixels


@pytest.mark.parametrize("width", [1, 7, 17, 257])
@pytest.mark.parametrize("compress_level", [0, 6, 9])
def test_png_high_effort_preserves_palette_alpha_and_row_tails(width: int, compress_level: int) -> None:
    rng = random.Random(491)
    # Include identical RGB with distinct alpha, and distinct hidden colors.
    palette = [bytes((19, 73, 151, a)) for a in (0, 1, 127, 255)]
    palette += [bytes((i * 17, 255 - i * 17, 37, 0)) for i in range(12)]
    pixels = b"".join(palette[rng.randrange(len(palette))] for _ in range(width * 19))
    source = Image.frombytes("RGBA", (width, 19), pixels)
    baseline, output = io.BytesIO(), io.BytesIO()
    source.save(baseline, "PNG", compress_level=compress_level)
    source.save(output, "PNG", compress_level=compress_level, compressor=LosslessImageCompressor(effort=8))
    assert len(output.getvalue()) <= len(baseline.getvalue())
    assert Image.open(io.BytesIO(output.getvalue())).convert("RGBA").tobytes() == pixels
    with PillowImage.open(output) as decoded:
        assert decoded.convert("RGBA").tobytes() == pixels
    assert source.tobytes() == pixels


@pytest.mark.parametrize("colors", [4, 16])
def test_png_wider_palette_indices_can_compress_better(colors: int) -> None:
    rng = random.Random(71)
    palette = [bytes((i * 71 % 256, i * 133 % 256, i * 37 % 256)) for i in range(colors)]
    samples = []
    value = 0
    for _ in range(129 * 127):
        if rng.randrange(16) == 0:
            value = rng.randrange(colors)
        samples.append(palette[value])
    pixels = b"".join(samples)
    source = Image.frombytes("RGB", (129, 127), pixels)
    packed, wider = io.BytesIO(), io.BytesIO()
    # Hold DEFLATE level fixed to isolate the palette depth search.
    source.save(packed, "PNG", compress_level=9, compressor=LosslessImageCompressor(effort=7))
    source.save(wider, "PNG", compress_level=9, compressor=LosslessImageCompressor(effort=8))
    encoded = wider.getvalue()
    assert encoded[24:26] == bytes((8, 3))
    assert len(encoded) < len(packed.getvalue())
    assert Image.open(io.BytesIO(encoded)).convert("RGB").tobytes() == pixels
    with PillowImage.open(wider) as decoded:
        assert decoded.convert("RGB").tobytes() == pixels


@pytest.mark.parametrize("bits", [1, 2, 4])
@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
def test_png_packed_grayscale(bits: int, mode: str) -> None:
    rng = random.Random(73)
    values = [rng.randrange(1 << bits) * (255 // ((1 << bits) - 1)) for _ in range(129 * 127)]
    pixels = b"".join(bytes([v] if mode == "L" else [v, v, v] + ([255] if mode == "RGBA" else [])) for v in values)
    original = Image.frombytes(mode, (129, 127), pixels)
    baseline, output = io.BytesIO(), io.BytesIO()
    original.save(baseline, "PNG")
    original.save(output, "PNG", compressor=LosslessImageCompressor())
    encoded = output.getvalue()
    assert encoded[24:26] == bytes((bits, 0))
    assert len(encoded) < len(baseline.getvalue())
    with PillowImage.open(output) as decoded:
        assert decoded.convert(mode).tobytes() == pixels
    assert Image.open(io.BytesIO(encoded)).convert(mode).tobytes() == pixels


@pytest.mark.parametrize("grayscale", [False, True])
@pytest.mark.parametrize("key", [0, 85, 255])
def test_png_transparency_key(grayscale: bool, key: int) -> None:
    rng = random.Random(93)
    pixels = bytearray()
    for i in range(129 * 127):
        if i % 5 == 0:
            pixels.extend((key, key, key, 0))
        elif grayscale:
            value = rng.choice([v for v in (0, 85, 170, 255) if v != key])
            pixels.extend((value, value, value, 255))
        else:
            red = (key + rng.randrange(1, 256)) % 256
            pixels.extend((red, rng.randrange(256), rng.randrange(256), 255))
    source = Image.frombytes("RGBA", (129, 127), bytes(pixels))
    baseline, output = io.BytesIO(), io.BytesIO()
    source.save(baseline, "PNG")
    source.save(output, "PNG", compressor=LosslessImageCompressor())
    encoded = output.getvalue()
    if grayscale and key == 0:
        # A wider exact representation can compress better after filtering.
        assert encoded[24] in (2, 4, 8)
        assert encoded[25] == 0
    elif not grayscale:
        assert encoded[24:26] == bytes((8, 2))
    assert len(encoded) < len(baseline.getvalue())
    with PillowImage.open(output) as decoded:
        assert decoded.convert("RGBA").tobytes() == bytes(pixels)
    assert Image.open(io.BytesIO(encoded)).convert("RGBA").tobytes() == bytes(pixels)


@pytest.mark.parametrize("special", [bytes((0, 0, 0, 255)), bytes((1, 2, 3, 0)), bytes((4, 5, 6, 127))])
def test_png_transparency_key_rejects_inexact_reductions(special: bytes) -> None:
    rng = random.Random(87)
    pixels = bytes((0, 0, 0, 0)) + special + b"".join(rng.randbytes(3) + b"\xff" for _ in range(257))
    source = Image.frombytes("RGBA", (259, 1), pixels)
    output = io.BytesIO()
    source.save(output, "PNG", compressor=LosslessImageCompressor())
    with PillowImage.open(output) as decoded:
        assert decoded.convert("RGBA").tobytes() == pixels


def test_png_one_bit_white_transparency_key() -> None:
    rng = random.Random(81)
    pixels = b"".join(bytes((255, 255, 255, 0) if rng.randrange(2) else (0, 0, 0, 255)) for _ in range(65 * 63))
    output = io.BytesIO()
    Image.frombytes("RGBA", (65, 63), pixels).save(output, "PNG", compressor=LosslessImageCompressor())
    assert output.getvalue()[24:26] == bytes((1, 0))
    with PillowImage.open(output) as decoded:
        assert decoded.convert("RGBA").tobytes() == pixels
    assert Image.open(io.BytesIO(output.getvalue())).convert("RGBA").tobytes() == pixels


def test_png_palette_reordering_shortens_transparency() -> None:
    palette = [bytes((i, (i * 71) % 256, (i * 133) % 256, 0 if i == 255 else 255)) for i in range(256)]
    rng = random.Random(17)
    pixels = b"".join(palette[i] for i in list(range(256)) + [rng.randrange(256) for _ in range(128 * 128 - 256)])
    source = Image.frombytes("RGBA", (128, 128), pixels)
    output = io.BytesIO()
    source.save(output, "PNG", compressor=LosslessImageCompressor(effort=5))
    encoded = output.getvalue()
    assert encoded[25] == 3
    position = 8
    transparency = None
    while position < len(encoded):
        length = int.from_bytes(encoded[position : position + 4], "big")
        if encoded[position + 4 : position + 8] == b"tRNS":
            transparency = encoded[position + 8 : position + 8 + length]
        position += length + 12
    assert transparency == b"\x00"
    with PillowImage.open(output) as decoded:
        assert decoded.convert("RGBA").tobytes() == pixels
    assert Image.open(io.BytesIO(encoded)).convert("RGBA").tobytes() == pixels


@pytest.mark.parametrize("format", ["TIFF", "PNG"])
def test_normal_encoding_fallback(format: str) -> None:
    original = Image.new("RGB", (4, 4), (10, 20, 30))
    if format == "PNG":
        original = original.convert("RGB", bit_depth=16)
    baseline, output = io.BytesIO(), io.BytesIO()
    original.save(baseline, format)
    original.save(output, format, compressor=LosslessImageCompressor())
    assert output.getvalue() == baseline.getvalue()


@pytest.mark.parametrize("mode", ["L", "RGB"])
@pytest.mark.parametrize("quality", [1, 75, 100])
@pytest.mark.parametrize("effort", [1, 3, 10])
def test_jpeg_optimization(mode: str, quality: int, effort: int) -> None:
    size = (67, 35)  # Partial MCUs must not be cropped or modified.
    pixels = random.Random(41).randbytes(size[0] * size[1] * len(mode))
    original = Image.frombytes(mode, size, pixels)
    baseline, output = io.BytesIO(), io.BytesIO()
    original.save(baseline, "JPEG", quality=quality)
    original.save(output, "JPEG", quality=quality, compressor=LosslessImageCompressor(effort=effort))
    assert len(output.getvalue()) <= len(baseline.getvalue())
    with PillowImage.open(baseline) as expected, PillowImage.open(output) as actual:
        assert actual.size == expected.size == size
        assert actual.quantization == expected.quantization
        assert actual.layer == expected.layer
        assert actual.tobytes() == expected.tobytes()
    assert original.tobytes() == pixels


def test_jpeg_optimization_reduces_size() -> None:
    original = Image.new("RGB", (256, 256), (12, 31, 79))
    baseline, output = io.BytesIO(), io.BytesIO()
    original.save(baseline, "JPEG")
    original.save(output, "JPEG", compressor=LosslessImageCompressor())
    assert len(output.getvalue()) < len(baseline.getvalue())


@pytest.mark.parametrize("format", ["JXL", "HEIF", "HEIC"])
@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
@pytest.mark.parametrize("depth", [8, 10, 12])
@pytest.mark.parametrize("lossless", [False, True])
def test_sample_codec_optimization(format: str, mode: str, depth: int, lossless: bool) -> None:
    size = (33, 35)
    rng = random.Random(79)
    samples = [rng.randrange(1 << depth) for _ in range(size[0] * size[1] * len(mode))]
    if mode == "RGBA":
        samples[3::4] = [0 if i % 2 else (1 << depth) - 1 for i in range(size[0] * size[1])]
    raw = b"".join(value.to_bytes(1 if depth == 8 else 2, "little") for value in samples)
    original = Image.frombytes(mode, size, raw, bit_depth=depth)
    baseline, output = io.BytesIO(), io.BytesIO()
    options = {"lossless": lossless, "quality": 71}
    original.save(baseline, format, **options)
    original.save(output, format, compressor=LosslessImageCompressor(effort=2), **options)
    assert len(output.getvalue()) <= len(baseline.getvalue())
    expected, actual = Image.open(baseline), Image.open(output)
    assert (actual.size, actual.mode, actual.bit_depth) == (expected.size, expected.mode, expected.bit_depth)
    assert actual.tobytes() == expected.tobytes()
    assert original.tobytes() == raw
    assert original.bit_depth == depth
    if format == "JXL" and lossless:
        assert actual.convert(mode, bit_depth=depth).tobytes() == raw
    if format in {"HEIF", "HEIC"}:
        import pillow_heif

        external = pillow_heif.open_heif(output.getvalue(), convert_hdr_to_8bit=False, hdr_to_16bit=False)
        external_baseline = pillow_heif.open_heif(baseline.getvalue(), convert_hdr_to_8bit=False, hdr_to_16bit=False)
        assert bytes(external.data) == bytes(external_baseline.data)


@pytest.mark.parametrize("format", ["JXL", "HEIF", "HEIC"])
def test_sample_codec_reuse_and_palette(format: str, tmp_path: Path) -> None:
    original = Image.new("RGB", (32, 32), (10, 20, 30)).quantize()
    compressor = LosslessImageCompressor(effort=1)
    path = tmp_path / f"optimized.{format.lower()}"
    original.save(path, lossless=True, compressor=compressor)
    output = io.BytesIO()
    original.save(output, format, lossless=True, compressor=compressor)
    assert path.read_bytes() == output.getvalue()
    baseline = io.BytesIO()
    original.save(baseline, format, lossless=True)
    assert len(output.getvalue()) <= len(baseline.getvalue())
    assert Image.open(output).tobytes() == Image.open(baseline).tobytes()


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
@pytest.mark.parametrize("depth", [8, 16])
def test_jxl_compressor_reuses_runner_across_frame_sizes(mode: str, depth: int) -> None:
    compressor = LosslessImageCompressor(effort=2)
    for size in [(17, 9), (513, 257), (19, 11)]:
        samples = size[0] * size[1] * len(mode)
        raw = b"".join(((i * 71) % (1 << depth)).to_bytes(depth // 8, "little") for i in range(samples))
        source = Image.frombytes(mode, size, raw, bit_depth=depth)
        baseline, output = io.BytesIO(), io.BytesIO()
        source.save(baseline, "JXL", lossless=True, effort=1)
        source.save(output, "JXL", lossless=True, effort=1, compressor=compressor)
        assert len(output.getvalue()) <= len(baseline.getvalue())
        decoded = Image.open(output)
        assert decoded.size == size
        assert decoded.convert(mode, bit_depth=depth).tobytes() == raw
        assert source.tobytes() == raw


@pytest.mark.parametrize("format", ["JXL", "HEIF"])
def test_sample_codec_optimization_reduces_size(format: str) -> None:
    original = Image.new("RGB", (128, 128), (12, 31, 79))
    baseline, output = io.BytesIO(), io.BytesIO()
    options = {"effort": 1} if format == "JXL" else {}
    original.save(baseline, format, lossless=True, **options)
    original.save(output, format, lossless=True, compressor=LosslessImageCompressor(effort=7), **options)
    assert len(output.getvalue()) < len(baseline.getvalue())


@pytest.mark.parametrize("format", ["JXL", "HEIC"])
@pytest.mark.parametrize("effort", [1, 4, 7, 10])
def test_sample_codec_effort_range(format: str, effort: int) -> None:
    source = Image.frombytes("L", (32, 32), bytes(i % 256 for i in range(1024)))
    baseline, output = io.BytesIO(), io.BytesIO()
    source.save(baseline, format, lossless=True)
    source.save(output, format, lossless=True, compressor=LosslessImageCompressor(effort=effort))
    assert len(output.getvalue()) <= len(baseline.getvalue())
    assert Image.open(output).convert("L").tobytes() == source.tobytes()


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
def test_jxl_sixteen_bit_optimization(mode: str) -> None:
    raw = random.Random(35).randbytes(17 * 19 * len(mode) * 2)
    source = Image.frombytes(mode, (17, 19), raw, bit_depth=16)
    baseline, output = io.BytesIO(), io.BytesIO()
    source.save(baseline, "JXL", lossless=True)
    source.save(output, "JXL", lossless=True, compressor=LosslessImageCompressor(effort=3))
    assert len(output.getvalue()) <= len(baseline.getvalue())
    assert Image.open(output).tobytes() == raw


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
def test_optimized_jxl_external_decode(mode: str) -> None:
    pytest.importorskip("pillow_jxl")
    raw = random.Random(37).randbytes(33 * 35 * len(mode))
    output = io.BytesIO()
    Image.frombytes(mode, (33, 35), raw).save(output, "JXL", lossless=True, compressor=LosslessImageCompressor(effort=3))
    with PillowImage.open(output) as decoded:
        assert decoded.tobytes() == raw


@pytest.mark.parametrize("format,mode,depth,error", [("JPEG", "RGBA", 8, OSError), ("JPEG", "RGB", 16, ValueError), ("HEIC", "RGB", 16, ValueError)])
def test_compressor_preserves_codec_validation(format: str, mode: str, depth: int, error: type[Exception]) -> None:
    source = Image.new(mode, (32, 32)).convert(mode, bit_depth=depth)
    output = io.BytesIO()
    with pytest.raises(error):
        source.save(output, format, compressor=LosslessImageCompressor())
    assert output.getvalue() == b""


def test_indexed_png_preserves_palette() -> None:
    original = Image.new("RGB", (4, 4), (10, 20, 30)).quantize()
    baseline, output = io.BytesIO(), io.BytesIO()
    original.save(baseline, "PNG")
    original.save(output, "PNG", compressor=LosslessImageCompressor())
    assert output.getvalue() == baseline.getvalue()
    with PillowImage.open(io.BytesIO(output.getvalue())) as decoded:
        assert decoded.mode == "P"
        assert decoded.convert("RGB").tobytes() == original.convert("RGB").tobytes()


@pytest.mark.parametrize("use_compressor", [False, True])
def test_pillow_save_call_compatibility(tmp_path: Path, use_compressor: bool) -> None:
    options = {"compress_level": 6}
    if use_compressor:
        options["compressor"] = LosslessImageCompressor()
    pixels = bytes(range(192))
    for backend, name in [(Image, "blanket"), (PillowImage, "pillow")]:
        image = backend.frombytes("RGB", (8, 8), pixels)
        stream = io.BytesIO()
        image.save(stream, "PNG", **options)
        with PillowImage.open(io.BytesIO(stream.getvalue())) as decoded:
            assert decoded.convert("RGB").tobytes() == pixels
        path = tmp_path / f"{name}.png"
        image.save(path, **options)
        with PillowImage.open(path) as decoded:
            assert decoded.convert("RGB").tobytes() == pixels


@pytest.mark.parametrize("effort", [0, 11, -1, 1000])
def test_invalid_effort_range(effort: int) -> None:
    with pytest.raises(ValueError, match="effort"):
        LosslessImageCompressor(effort=effort)


@pytest.mark.parametrize("effort", [True, 1.5, "7", None])
def test_invalid_effort_type(effort: object) -> None:
    with pytest.raises(TypeError, match="effort"):
        LosslessImageCompressor(effort=effort)


def test_compression_is_only_available_through_save() -> None:
    compressor = LosslessImageCompressor()
    assert not callable(compressor)
    assert not callable(compressor._native)
    image = Image.new("L", (1, 1))
    output = io.BytesIO()
    with pytest.raises(TypeError, match="compressor"):
        image.save(output, "PNG", compressor=object())
    assert output.getvalue() == b""
    image.close()
    with pytest.raises(ValueError, match="closed"):
        image.save(output, "PNG", compressor=compressor)
