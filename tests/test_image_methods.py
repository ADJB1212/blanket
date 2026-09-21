from __future__ import annotations

import pytest
from blanket import Image
from PIL import Image as PIL


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
