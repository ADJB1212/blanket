from __future__ import annotations

import math
import struct
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image as PIL

from blanket import Image


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA", "P"])
def test_statistics_channels_and_point(mode: str) -> None:
    data = bytes((i * 37) % 256 for i in range(63 * len(mode)))
    actual = Image.frombytes(mode, (9, 7), data)
    expected = PIL.frombytes(mode, (9, 7), data)
    assert actual.getbands() == expected.getbands()
    assert actual.histogram() == expected.histogram()
    assert actual.getextrema() == expected.getextrema()
    for channel in (*range(len(mode)), *mode):
        assert actual.getchannel(channel).tobytes() == expected.getchannel(channel).tobytes()
    for lut in (lambda v: v / 2, list(range(255, -1, -1)) * len(mode)):
        assert actual.point(lut).tobytes() == expected.point(lut).tobytes()
    mask_data = bytes(i % 3 for i in range(63))
    assert actual.histogram(Image.frombytes("L", (9, 7), mask_data)) == expected.histogram(PIL.frombytes("L", (9, 7), mask_data))


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA", "P"])
@pytest.mark.parametrize("size", [(13, 11), (1, 8), (8, 1), (100, 100), (7.9, 5.2)])
def test_thumbnail(mode: str, size: tuple[float, float]) -> None:
    data = bytes((i * 19) % 256 for i in range(31 * 23 * len(mode)))
    actual = Image.frombytes(mode, (31, 23), data)
    expected = PIL.frombytes(mode, (31, 23), data)
    actual.info["custom"] = 42
    assert actual.thumbnail(size) is None
    expected.thumbnail(size)
    assert actual.size == expected.size
    assert actual.tobytes() == expected.tobytes()
    assert actual.info["custom"] == 42


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA", "P"])
@pytest.mark.parametrize("alpha_only", [True, False])
def test_bbox(mode: str, alpha_only: bool) -> None:
    actual = Image.new(mode, (7, 5))
    expected = PIL.new(mode, (7, 5))
    assert actual.getbbox(alpha_only=alpha_only) is None
    color = (10, 20, 30, 0) if mode == "RGBA" else 12
    actual.paste(color, (2, 1, 5, 4))
    expected.paste(color, (2, 1, 5, 4))
    assert actual.getbbox(alpha_only=alpha_only) == expected.getbbox(alpha_only=alpha_only)


def test_validation_and_empty_images() -> None:
    image = Image.new("RGB", (2, 2))
    for channel in (-1, 3, "A"):
        with pytest.raises(ValueError):
            image.getchannel(channel)
    with pytest.raises(ValueError):
        image.point([0] * 256)
    with pytest.raises(ValueError):
        image.histogram(Image.new("L", (1, 1)))
    assert Image.new("L", (0, 0)).getextrema() is None
    assert Image.new("RGBA", (0, 0)).getbbox() is None
    image.close()
    for operation in (image.getbbox, image.histogram, image.getextrema):
        with pytest.raises(ValueError):
            operation()


def test_palette_and_independent_channel() -> None:
    image = Image.frombytes("P", (2, 1), b"\x01\x02")
    image.putpalette(bytes(range(30)))
    image.info["custom"] = 42
    mapped = image.point(lambda v: 255 - v)
    assert mapped.getpalette() == image.getpalette()
    assert mapped.info == image.info
    channel = image.getchannel("P")
    channel.paste(99, (0, 0, 2, 1))
    assert image.tobytes() == b"\x01\x02"


@pytest.mark.parametrize("depth", [10, 12, 16])
def test_high_depth_bbox(depth: int) -> None:
    image = Image.frombytes("RGBA", (2, 1), b"\x00" * 8 + b"\x00\x01" * 4, bit_depth=depth)
    assert image.getbbox() == (1, 0, 2, 1)
    assert image.getchannel("A").getpixel((1, 0)) == 256
    with pytest.raises(ValueError, match="8-bit"):
        image.histogram()


def test_getbands_tracks_in_place_mode_changes() -> None:
    image = Image.new("L", (2, 2))
    assert image.getbands() == ("L",)
    image.putpalette(bytes(range(256)) * 3)
    assert image.getbands() == ("P",)
    image = Image.new("RGB", (2, 2))
    assert image.getbands() == ("R", "G", "B")
    image.putalpha(127)
    assert image.getbands() == ("R", "G", "B", "A")


@pytest.mark.parametrize("width", [1, 17, 1024])
def test_l_bbox_scans_empty_edge_columns(width: int) -> None:
    raw = bytes(width) + (b"\0" + bytes([7]) * (width - 1)) * 31 + bytes(width)
    image = Image.frombytes("L", (width, 33), raw)
    reference = PIL.frombytes("L", (width, 33), raw)
    assert image.getbbox() == reference.getbbox()


def test_remap_palette_matches_pillow_and_preserves_transparency() -> None:
    image = Image.frombytes("P", (4, 1), b"\x00\x01\x02\x01")
    reference = PIL.frombytes("P", image.size, image.tobytes())
    palette = bytes([10, 20, 30, 40, 50, 60, 70, 80, 90])
    image.putpalette(palette)
    reference.putpalette(palette)
    image.info["transparency"] = reference.info["transparency"] = 1
    mapping = [2, 0, 1]
    actual = image.remap_palette(mapping)
    expected = reference.remap_palette(mapping)
    assert actual.mode == expected.mode == "P"
    assert actual.tobytes() == expected.tobytes()
    assert actual.getpalette() == expected.getpalette()
    assert actual.info["transparency"] == expected.info["transparency"] == 2
    assert image.tobytes() == b"\x00\x01\x02\x01"


@pytest.mark.parametrize("mode", ["L", "P"])
@pytest.mark.parametrize("palette_mode", ["RGB", "RGBA"])
def test_remap_palette_with_source_palette(mode: str, palette_mode: str) -> None:
    raw = b"\x00\x01\x02"
    palette = bytes(i % 256 for i in range(256 * len(palette_mode)))
    image = Image.frombytes(mode, (3, 1), raw)
    reference = PIL.frombytes(mode, (3, 1), raw)
    actual = image.remap_palette([2, 0, 1], source_palette=palette)
    expected = reference.remap_palette([2, 0, 1], source_palette=palette)
    assert actual.tobytes() == expected.tobytes()
    assert actual.getpalette(palette_mode) == expected.getpalette(palette_mode)


def test_xbm_bitmap_matches_pillow() -> None:
    image = Image.new("1", (9, 2))
    reference = PIL.new("1", image.size)
    for point in ((0, 0), (7, 0), (8, 1)):
        image.putpixel(point, 255)
        reference.putpixel(point, 255)
    assert image.tobitmap("sample") == reference.tobitmap("sample")
    with pytest.raises(ValueError, match="not a bitmap"):
        Image.new("L", (1, 1)).tobitmap()


@pytest.mark.parametrize("width", [0, 1, 7, 8, 9, 31, 32, 33, 257])
def test_native_xbm_preserves_partial_rows_and_ascii_names(width: int) -> None:
    from blanket import _blanket

    image = Image.new("1", (width, 3))
    for y in range(3):
        for x in range(width):
            if (x * 7 + y) % 11 < 5:
                image.putpixel((x, y), 255)
    body = ",".join(f"0x{value:02x}" for value in _blanket.ops_xbm_bits(image._native))
    expected = f"#define sample_width {width}\n#define sample_height 3\nstatic char sample_bits[] = {{\n{body}\n}};".encode("ascii")
    assert image.tobitmap("sample") == expected
    with pytest.raises(UnicodeEncodeError):
        image.tobitmap("non-ascii-\u00e9")


def test_eager_verify_draft_and_core_shim() -> None:
    stream = BytesIO()
    Image.new("RGB", (3, 2), (10, 20, 30)).save(stream, "PNG")
    image = Image.open(BytesIO(stream.getvalue()))
    assert image.verify() is None
    assert image.draft("L", (1, 1)) is None
    assert image.getim() is image.im
    assert image.size == (3, 2)
    image.close()
    with pytest.raises(ValueError, match="closed"):
        image.verify()


@pytest.mark.parametrize(
    ("mode", "depth", "sample_bytes"),
    [(mode, 8, channels) for mode, channels in (("1", 1), ("L", 1), ("LA", 2), ("P", 1), ("PA", 2), ("RGB", 3), ("RGBA", 4), ("HSV", 3), ("CMYK", 4), ("YCbCr", 3), ("LAB", 3))]
    + [(mode, 32 if mode in ("I", "F") else 16, 4 if mode in ("I", "F") else 2) for mode in ("I", "F", "I;16", "I;16L", "I;16B")]
    + [(mode, depth, channels * 2) for mode, channels in (("L", 1), ("RGB", 3), ("RGBA", 4)) for depth in (10, 12, 16)],
)
@pytest.mark.parametrize("size", [(1, 1), (1, 17), (17, 1), (7, 5), (0, 3)])
def test_spread_keeps_complete_samples_within_neighborhood(mode: str, depth: int, sample_bytes: int, size: tuple[int, int]) -> None:
    count = size[0] * size[1]
    if mode == "1":
        image = Image.new(mode, size)
        image.putdata([i % 2 * 255 for i in range(count)])
    elif mode in ("L", "RGB", "RGBA") and depth > 8:
        samples = [(i * 73 + i // 7) % (1 << depth) for i in range(count * len(mode))]
        image = Image.frombytes(mode, size, struct.pack("<" + "H" * len(samples), *samples), bit_depth=depth)
    else:
        raw = bytes((i * 73 + i // 7) % 256 for i in range(count * sample_bytes))
        image = Image.frombytes(mode, size, raw)
    if mode == "P":
        image.putpalette(bytes(range(256)) * 3)
    pixels = bytes(image.getdata()) if mode == "1" else image.tobytes()
    for distance in (0, 1, 2, -1, -(2**63)):
        result = image.effect_spread(distance)
        actual = bytes(result.getdata()) if mode == "1" else result.tobytes()
        assert (result.mode, result.bit_depth, result.size) == (image.mode, image.bit_depth, size)
        assert result.getpalette() == image.getpalette()
        if distance == 0:
            assert actual == pixels
        radius = abs(distance)
        for i in range(count):
            x, y = i % size[0], i // size[0]
            neighbors = {
                pixels[(yy * size[0] + xx) * sample_bytes : (yy * size[0] + xx + 1) * sample_bytes]
                for yy in range(max(0, y - radius), min(size[1], y + radius + 1))
                for xx in range(max(0, x - radius), min(size[0], x + radius + 1))
            }
            assert actual[i * sample_bytes : (i + 1) * sample_bytes] in neighbors


@pytest.mark.parametrize(("mode", "data"), [("I", [0, 10, 20, 30]), ("F", [0.0, 0.5, 1.0, 2.0])])
def test_wide_entropy_matches_pillow(mode: str, data: list[int | float]) -> None:
    image = Image.new(mode, (4, 1))
    reference = PIL.new(mode, image.size)
    image.putdata(data)
    reference.putdata(data)
    for extrema in (None, (0, 40)):
        assert image.entropy(extrema=extrema) == pytest.approx(reference.entropy(extrema=extrema))
    mask = Image.frombytes("L", (4, 1), b"\x01\x00\x01\x00")
    assert image.entropy(mask) == pytest.approx(1.0)


@pytest.mark.parametrize(("mode", "data"), [("I", [-10, 0, 1, 2, 255, 256, 1000]), ("F", [-1.0, 0.0, 0.1, 0.5, 1.0, 2.0, 10.0])])
def test_wide_entropy_extrema_binning_matches_pillow(mode: str, data: list[int | float]) -> None:
    image = Image.new(mode, (len(data), 1))
    reference = PIL.new(mode, image.size)
    image.putdata(data)
    reference.putdata(data)
    for extrema in (None, (0, 1), (0, 256), (-10, 1000)):
        assert image.entropy(extrema=extrema) == pytest.approx(reference.entropy(extrema=extrema))
    constant = Image.new(mode, (2, 1), 5)
    assert math.isnan(constant.entropy())


def test_high_depth_entropy() -> None:
    image = Image.frombytes("L", (4, 1), b"\x00\x00\x00\x01\x00\x02\x00\x03", bit_depth=16)
    assert image.entropy() == pytest.approx(2.0)
    assert math.isnan(Image.new("I", (0, 0)).entropy())


@pytest.mark.parametrize("mode", ["I", "I;16", "I;16L", "I;16B", "F"])
def test_wide_entropy_streams_samples_with_masks(mode: str) -> None:
    code = "i" if mode == "I" else "f" if mode == "F" else "H"
    raw = struct.pack((">" if mode == "I;16B" else "<") + "4" + code, 0, 256, 512, 768)
    image = Image.frombytes(mode, (4, 1), raw)
    mask = Image.frombytes("L", image.size, b"\x01\x00\xff\x00")
    for extrema in (None, (0, 768)):
        assert image.entropy(extrema=extrema) == pytest.approx(2.0)
        assert image.entropy(mask, extrema=extrema) == pytest.approx(1.0)
    assert math.isnan(image.entropy(Image.new("L", image.size)))
    assert image.tobytes() == raw


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
@pytest.mark.parametrize("depth", [10, 12, 16])
def test_high_depth_entropy_preserves_band_and_mask_counts(mode: str, depth: int) -> None:
    maximum = (1 << depth) - 1
    values = [value for value in (0, maximum // 3, maximum * 2 // 3, maximum) for _ in mode]
    raw = struct.pack("<" + "H" * len(values), *values)
    image = Image.frombytes(mode, (4, 1), raw, bit_depth=depth)
    mask = Image.frombytes("L", image.size, b"\x01\x00\xff\x00")
    for extrema in (None, (0, maximum)):
        assert image.entropy(extrema=extrema) == pytest.approx(math.log2(4 * len(mode)))
        assert image.entropy(mask, extrema=extrema) == pytest.approx(math.log2(2 * len(mode)))


def test_float_entropy_ignores_nonfinite_samples_and_validates_extrema() -> None:
    raw = struct.pack("<5f", float("nan"), float("inf"), -float("inf"), -1, 1)
    image = Image.frombytes("F", (5, 1), raw)
    assert image.entropy() == pytest.approx(1.0)
    assert image.entropy(extrema=(-1, 1)) == pytest.approx(1.0)
    for extrema in ((1, -1), (float("nan"), 1), (0, float("inf"))):
        with pytest.raises(ValueError, match="invalid extrema"):
            image.entropy(extrema=extrema)
    with pytest.raises(ValueError, match="invalid extrema"):
        Image.frombytes("F", (3, 1), raw[:12]).entropy()


@pytest.mark.parametrize("mode", ["RGB", "CMYK"])
def test_show_launches_configured_viewer(monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    launched: list[list[str]] = []
    monkeypatch.setenv("BLANKET_IMAGE_VIEWER", "image-viewer --single")
    monkeypatch.setattr(Image.subprocess, "Popen", lambda command, **kwargs: launched.append(command))
    color = (255, 0, 0) if mode == "RGB" else (0, 255, 255, 0)
    Image.new(mode, (1, 1), color).show()
    assert launched[0][:2] == ["image-viewer", "--single"]
    path = Path(launched[0][-1])
    assert path.read_bytes().startswith(b"\x89PNG")
    path.unlink()


@pytest.mark.parametrize("size", [(800, 600), (1024, 1024), (803, 607)])
def test_l_thumbnail_integer_reduction_matches_pillow(size: tuple[int, int]) -> None:
    raw = bytes((i * 37 + i // 11) % 256 for i in range(size[0] * size[1]))
    image = Image.frombytes("L", size, raw)
    reference = PIL.frombytes("L", size, raw)
    target = (size[0] // 8, size[1] // 8)
    image.thumbnail(target)
    reference.thumbnail(target)
    assert image.size == reference.size
    assert image.tobytes() == reference.tobytes()


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
@pytest.mark.parametrize("depth", [8, 16])
@pytest.mark.parametrize("size", [(0, 3), (1, 1031), (1031, 1), (37, 41), (1025, 1027)])
def test_copy_preserves_pixels_metadata_and_independent_storage(mode: str, depth: int, size: tuple[int, int]) -> None:
    raw = bytes(i % 256 for i in range(size[0] * size[1] * len(mode) * (depth // 8)))
    image = Image.frombytes(mode, size, raw, bit_depth=depth)
    image.info["note"] = "retained"
    result = image.copy()
    assert (result.mode, result.size, result.bit_depth) == (mode, size, depth)
    assert result.tobytes() == raw
    assert result.info == image.info
    result.info["note"] = "changed"
    if depth == 8 and size[0] and size[1]:
        result.putpixel((0, 0), 0)
        assert image.tobytes() == raw
    image.close()
    assert result.tobytes() is not None
    assert image.info["note"] == "retained"
