"""Pixel access and native indexed-color quantization behavior."""

from __future__ import annotations

from io import BytesIO

import pytest
from blanket import Image
from PIL import Image as PillowImage


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
@pytest.mark.parametrize("xy", [(0, 0), (3, 2), (-1, -1), (-4, -3), [2, 1], (1.9, 0.1)])
def test_getpixel_matches_pillow(mode: str, xy: object) -> None:
    raw = bytes(range(12 * len(mode)))
    actual = Image.frombytes(mode, (4, 3), raw)
    expected = PillowImage.frombytes(mode, (4, 3), raw)
    assert actual.getpixel(xy) == expected.getpixel(xy)


@pytest.mark.parametrize("xy", [(4, 0), (0, 3), (-5, 0), (0, -4), (), (0,), (0, 0, 0), "00", None, ("0", 0)])
def test_getpixel_errors(xy: object) -> None:
    actual = Image.frombytes("L", (4, 3), bytes(12))
    expected = PillowImage.new("L", (4, 3))
    with pytest.raises(Exception) as error:
        expected.getpixel(xy)
    with pytest.raises(type(error.value)):
        actual.getpixel(xy)


@pytest.mark.parametrize("mode,methods", [("L", [0, 1, 2]), ("RGB", [0, 1, 2]), ("RGBA", [2])])
def test_quantize_lossless_small_palette(mode: str, methods: list[int]) -> None:
    colors = {"L": [b"\x00", b"\xff"], "RGB": [b"\xff\x00\x00", b"\x00\xff\x00"], "RGBA": [b"\xff\x00\x00\xff", b"\x00\xff\x00\x80"]}[mode]
    raw = b"".join(colors * 4)
    image = Image.frombytes(mode, (4, 2), raw)
    image.info["custom"] = 42
    for method in methods:
        result = image.quantize(colors=2, method=method)
        assert (result.mode, result.size, len(result.tobytes())) == ("P", (4, 2), 8)
        assert len(set(result.tobytes())) == 2
        assert result.convert(mode).tobytes() == raw
        assert result.info == image.info
        assert isinstance(result.getpixel((0, 0)), int)
        assert result.palette is not None
        assert result.palette.mode == ("RGBA" if mode == "RGBA" else "RGB")
        assert result.to_pillow().convert(mode).tobytes() == raw


@pytest.mark.parametrize("method", [0, 1, 2])
@pytest.mark.parametrize("colors", [1, 2, 16, 256])
def test_quantize_color_budget_and_determinism(method: int, colors: int) -> None:
    raw = bytes((i * 37 + i // 11) % 256 for i in range(31 * 17 * 3))
    image = Image.frombytes("RGB", (31, 17), raw)
    result = image.quantize(colors, method, kmeans=1)
    repeated = image.quantize(colors, method, kmeans=1)
    assert result.tobytes() == repeated.tobytes()
    assert result.getpalette() == repeated.getpalette()
    assert len(set(result.tobytes())) <= colors
    assert len(result.getpalette()) <= colors * 3
    assert image.tobytes() == raw


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
def test_empty_quantize(mode: str) -> None:
    result = Image.frombytes(mode, (0, 0), b"").quantize()
    assert result.mode == "P"
    assert result.tobytes() == b""
    assert result.getpalette() == []


@pytest.mark.parametrize("dither", [Image.Dither.NONE, Image.Dither.FLOYDSTEINBERG])
@pytest.mark.parametrize("mode", ["L", "RGB"])
def test_fixed_palette(mode: str, dither: int) -> None:
    palette = Image.frombytes("P", (1, 1), b"\x00")
    palette.putpalette([255, 0, 0, 0, 255, 0])
    reference_palette = PillowImage.new("P", (1, 1))
    reference_palette.putpalette(palette.getpalette())
    raw = bytes([0, 1, 120, 255]) if mode == "L" else bytes([255, 0, 0, 0, 255, 0, 0, 0, 255, 128, 128, 128])
    image = Image.frombytes(mode, (4, 1), raw)
    reference = PillowImage.frombytes(mode, (4, 1), raw)
    result = image.quantize(colors=-1, palette=palette, dither=dither)
    expected = reference.quantize(colors=-1, palette=reference_palette, dither=dither)
    assert result.tobytes() == expected.tobytes()
    assert result.getpalette() == expected.getpalette()
    assert result.palette is not palette.palette


@pytest.mark.parametrize("kwargs,exception", [({"colors": 0}, ValueError), ({"colors": 257}, ValueError), ({"colors": 1.5}, TypeError), ({"method": -1}, ValueError), ({"method": 4}, ValueError), ({"method": 3}, RuntimeError), ({"kmeans": -1}, ValueError)])
def test_quantize_validation(kwargs: dict[str, object], exception: type[Exception]) -> None:
    with pytest.raises(exception):
        Image.frombytes("RGB", (1, 1), bytes(3)).quantize(**kwargs)


@pytest.mark.parametrize("method", [0, 1])
def test_rgba_rejects_rgb_methods(method: int) -> None:
    with pytest.raises(ValueError):
        Image.frombytes("RGBA", (1, 1), bytes(4)).quantize(method=method)


def test_palette_roundtrip_and_geometry() -> None:
    image = Image.frombytes("RGBA", (2, 1), bytes([255, 0, 0, 128, 0, 255, 0, 255])).quantize(colors=2)
    reference = image.to_pillow()
    for result, expected in [(image.copy(), reference.copy()), (image.crop((0, 0, 1, 1)), reference.crop((0, 0, 1, 1))), (image.resize((4, 3)), reference.resize((4, 3))), (image.transpose(Image.FLIP_LEFT_RIGHT), reference.transpose(PillowImage.Transpose.FLIP_LEFT_RIGHT))]:
        assert result.mode == "P"
        assert result.convert("RGBA").tobytes() == expected.convert("RGBA").tobytes()
        assert result.getpalette("RGBA") == expected.getpalette("RGBA")
    output = BytesIO()
    image.save(output, "PNG")
    decoded = PillowImage.open(BytesIO(output.getvalue()))
    assert decoded.mode == "P"
    assert decoded.convert("RGBA").tobytes() == image.convert("RGBA").tobytes()
    assert Image.open(BytesIO(output.getvalue())).tobytes() == image.convert("RGBA").tobytes()
    with pytest.raises(ValueError):
        image.reduce(2)
    assert image.reduce(1).mode == "P"


@pytest.mark.parametrize("dither", [0, 3])
@pytest.mark.parametrize("size", [(1, 17), (19, 17)])
def test_fixed_palette_random_grid(dither: int, size: tuple[int, int]) -> None:
    import random

    rng = random.Random(812)
    image = Image.frombytes("RGB", size, rng.randbytes(size[0] * size[1] * 3))
    palette = Image.frombytes("P", (1, 1), b"\x00")
    palette.putpalette(rng.randbytes(16 * 3))
    expected = image.to_pillow().quantize(palette=palette.to_pillow(), dither=dither)
    actual = image.quantize(palette=palette, dither=dither)
    assert actual.tobytes() == expected.tobytes()


def test_quantize_palette_validation() -> None:
    rgb = Image.frombytes("RGB", (1, 1), bytes(3))
    with pytest.raises(ValueError, match="palette"):
        rgb.quantize(palette=rgb)
    palette = rgb.quantize()
    with pytest.raises(ValueError, match="only RGB or L"):
        rgb.convert("RGBA").quantize(palette=palette)
    assert palette.quantize().convert("RGB").tobytes() == rgb.tobytes()


def test_empty_supplied_palette() -> None:
    palette = Image.frombytes("P", (1, 1), b"\x00")
    palette.putpalette([])
    image = Image.frombytes("RGB", (1, 1), b"\xff\x80\x40")
    result = image.quantize(palette=palette)
    assert result.tobytes() == b"\x00"
    assert result.getpalette() == []
    assert result.convert("RGB").tobytes() == bytes(3)


@pytest.mark.parametrize("mode,method", [("L", 0), ("RGB", 0), ("RGB", 1), ("RGB", 2), ("RGBA", 2)])
def test_parallel_quantize_preserves_serial_palette(mode: str, method: int) -> None:
    # Tiling preserves color counts in proportion. Palette decisions and
    # first-entry ties must survive worker-local histogram merging.
    raw = bytes((i * 37 + i // 11) % 256 for i in range(1024 * len(mode)))
    small = Image.frombytes(mode, (32, 32), raw).quantize(16, method)
    large = Image.frombytes(mode, (1024, 1024), raw * 1024).quantize(16, method)
    assert small.getpalette(None) == large.getpalette(None)
    assert large.tobytes() == small.tobytes() * 1024


def test_parallel_fixed_palette_matches_pillow() -> None:
    raw = bytes((i * 37 + i // 13) % 256 for i in range(513 * 517 * 3))
    image = Image.frombytes("RGB", (513, 517), raw)
    palette = Image.frombytes("P", (1, 1), b"\x00")
    palette.putpalette(bytes((i * 71) % 256 for i in range(16 * 3)))
    actual = image.quantize(palette=palette, dither=0)
    expected = image.to_pillow().quantize(palette=palette.to_pillow(), dither=0)
    assert actual.tobytes() == expected.tobytes()


def _photo(mode: str, size: tuple[int, int]) -> bytes:
    # Smooth gradients plus noise produce hundreds of thousands of distinct colors.
    import random

    rng = random.Random(97)
    w, h = size
    channels = len(mode)
    out = bytearray()
    for y in range(h):
        for x in range(w):
            base = (x * 255 // w, y * 255 // h, (x + y) * 255 // (w + h), 200 if (x + y) % 7 else 30)
            out.extend(min(255, max(0, v + rng.randint(-12, 12))) for v in base[:channels])
    return bytes(out)


def _error(image: bytes, reconstructed: bytes) -> int:
    return sum((a - b) ** 2 for a, b in zip(image, reconstructed, strict=True))


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
def test_fast_octree_matches_pillow_cells(mode: str) -> None:
    # FASTOCTREE mirrors Pillow's fixed color cubes. Pillow's qsort orders
    # equal cell populations unpredictably, so compare the reconstruction
    # rather than entry order; grayscale has too few cells to tie.
    size = (211, 97)
    raw = _photo(mode, size)
    image = Image.frombytes(mode, size, raw)
    actual = image.quantize(256, Image.Quantize.FASTOCTREE)
    expected = PillowImage.frombytes(mode, size, raw).quantize(256, PillowImage.Quantize.FASTOCTREE)
    assert len(actual.getpalette(None)) == len(expected.getpalette(None))
    actual_error = _error(raw, actual.convert(mode).tobytes())
    expected_error = _error(raw, expected.convert(mode).tobytes())
    assert abs(actual_error - expected_error) <= expected_error // 50
    if mode == "L":
        assert sorted(actual.getpalette()) == sorted(expected.getpalette())
        assert actual.convert(mode).tobytes() == expected.convert(mode).tobytes()


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
def test_fast_octree_small_palettes_keep_populous_cells(mode: str) -> None:
    size = (211, 97)
    raw = _photo(mode, size)
    image = Image.frombytes(mode, size, raw)
    for colors in (1, 2, 16, 64):
        result = image.quantize(colors, Image.Quantize.FASTOCTREE)
        assert len(result.getpalette(None)) == colors * len(result.palette.mode)
        assert len(set(result.tobytes())) <= colors
        assert result.convert(mode).tobytes() == result.to_pillow().convert(mode).tobytes()


@pytest.mark.parametrize("method", [0, 1])
def test_exact_methods_handle_many_colors(method: int) -> None:
    size = (157, 61)
    raw = _photo("RGB", size)
    image = Image.frombytes("RGB", size, raw)
    result = image.quantize(64, method, kmeans=1)
    assert len(set(result.tobytes())) <= 64
    assert result.convert("RGB").tobytes() == result.to_pillow().convert("RGB").tobytes()
    # Every pixel maps to the palette entry nearest to its color.
    palette = result.getpalette()
    entries = [tuple(palette[i : i + 3]) for i in range(0, len(palette), 3)]
    for pixel, index in zip(zip(*[iter(raw)] * 3), result.tobytes(), strict=True):
        best = min(sum((a - b) ** 2 for a, b in zip(pixel, entry)) for entry in entries)
        assert sum((a - b) ** 2 for a, b in zip(pixel, entries[index])) == best


def test_wide_dither_matches_pillow() -> None:
    # Wide rows carry the diffusion error across many pixels and rows.
    size = (301, 23)
    raw = _photo("RGB", size)
    image = Image.frombytes("RGB", size, raw)
    palette = image.quantize(16)
    expected = image.to_pillow().quantize(palette=palette.to_pillow(), dither=3)
    assert image.quantize(palette=palette, dither=3).tobytes() == expected.tobytes()


def test_palette_convert_matches_pillow_for_all_targets() -> None:
    size = (37, 11)
    raw = _photo("RGBA", size)
    quantized = Image.frombytes("RGBA", size, raw).quantize(64)
    reference = quantized.to_pillow()
    for mode in ("L", "RGB", "RGBA"):
        assert quantized.convert(mode).tobytes() == reference.convert(mode).tobytes()
    # Indices beyond a short palette read as opaque black.
    short = Image.frombytes("P", (3, 1), bytes([0, 1, 200]))
    short.putpalette([10, 20, 30, 40, 50, 60])
    assert short.convert("RGB").tobytes() == bytes([10, 20, 30, 40, 50, 60, 0, 0, 0])
    assert short.convert("RGBA").tobytes() == bytes([10, 20, 30, 255, 40, 50, 60, 255, 0, 0, 0, 255])


def test_closed_pixel_quantize_reduce() -> None:
    image = Image.frombytes("L", (1, 1), b"\x00")
    image.close()
    for operation in (lambda: image.getpixel((0, 0)), image.quantize, lambda: image.reduce(2)):
        with pytest.raises(ValueError, match="closed"):
            operation()
