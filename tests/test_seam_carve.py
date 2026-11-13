"""Content-aware retargeting preserves salient edges and sample values."""

from itertools import pairwise, product

import pytest
from helpers import image_buffer

from blanket import Image


@pytest.mark.parametrize("mode", ["L", "LA", "RGB", "RGBA"])
@pytest.mark.parametrize("size", [(1, 1), (5, 4), (2, 3), (8, 7), (2, 9), (11, 2)])
def test_uniform_retargeting(mode: str, size: tuple[int, int]) -> None:
    color = {"L": 73, "LA": (73, 127), "RGB": (12, 73, 199), "RGBA": (12, 73, 199, 127)}[mode]
    image = Image.new(mode, (5, 4), color)
    result = image.seam_carve(size)
    assert result.size == size
    assert result.mode == mode
    assert result.tobytes() == Image.new(mode, size, color).tobytes()
    assert image.size == (5, 4)


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
@pytest.mark.parametrize("depth", [10, 12, 16])
@pytest.mark.parametrize("size", [(2, 3), (9, 7)])
def test_high_depth_samples(mode: str, depth: int, size: tuple[int, int]) -> None:
    maximum = (1 << depth) - 1
    channels = len(mode)
    values = [(i * 137 + 259) & maximum for i in range(5 * 4 * channels)]
    raw = b"".join(value.to_bytes(2, "little") for value in values)
    image = Image.frombytes(mode, (5, 4), raw, bit_depth=depth)
    source_pixels = set(image.get_flattened_data())
    result = image.seam_carve(size)
    assert result.size == size
    assert result.mode == mode
    assert result.bit_depth == depth
    assert set(result.get_flattened_data()) <= source_pixels
    assert image.tobytes() == raw


@pytest.mark.parametrize("size", [(9, 5), (6, 5), (15, 5)])
def test_preserves_salient_stripe(size: tuple[int, int]) -> None:
    image = Image.new("RGB", (12, 5))
    for y in range(5):
        image.putpixel((5, y), (255, 30, 70))
        image.putpixel((6, y), (255, 30, 70))
    result = image.seam_carve(size)
    for y in range(5):
        assert sum(result.getpixel((x, y)) == (255, 30, 70) for x in range(size[0])) == 2


def test_preserves_alpha_edges_and_ignores_hidden_color() -> None:
    image = Image.new("RGBA", (9, 4), (255, 0, 255, 0))
    for y in range(4):
        image.putpixel((4, y), (0, 0, 0, 255))
    result = image.seam_carve((5, 4))
    for y in range(4):
        assert sum(result.getpixel((x, y))[3] == 255 for x in range(5)) == 1


def test_minimum_energy_connected_seam() -> None:
    rows = [(17, 91, 42, 230), (130, 83, 29, 215), (240, 78, 65, 142), (192, 5, 60, 177), (235, 23, 52, 118)]
    image = Image.frombytes("L", (4, 5), bytes(value for row in rows for value in row))
    result = image.seam_carve((3, 5))
    removed = []
    for y, row in enumerate(rows):
        output_row = tuple(result.getpixel((x, y)) for x in range(3))
        removed.append(next(x for x in range(4) if row[:x] + row[x + 1 :] == output_row))
    assert all(abs(a - b) <= 1 for a, b in pairwise(removed))

    def energy(seam: tuple[int, ...] | list[int]) -> int:
        return sum(abs(rows[y][x] - rows[ny][nx]) for y, x in enumerate(seam) for ny, nx in [(y, max(0, x - 1)), (y, min(3, x + 1)), (max(0, y - 1), x), (min(4, y + 1), x)])

    candidates = [seam for seam in product(range(4), repeat=5) if all(abs(a - b) <= 1 for a, b in pairwise(seam))]
    assert energy(removed) == min(energy(seam) for seam in candidates)


def test_enlargement_selects_distinct_seams_and_ignores_hidden_colors() -> None:
    raw = bytes([10, 20, 30, 0, 40, 50, 60, 0, 70, 80, 90, 0, 100, 110, 120, 0])
    image = Image.frombytes("RGBA", (4, 1), raw)
    result = image.seam_carve((7, 1))
    assert result.tobytes() == b"".join(raw[x * 4 : x * 4 + 4] for x in [0, 0, 1, 1, 2, 2, 3])
    assert image.seam_carve((2, 1)).tobytes() == raw[8:]


@pytest.mark.parametrize("lazy", [False, True])
def test_current_frame_and_navigation(lazy: bool) -> None:
    first = Image.new("RGB", (5, 4), (200, 20, 30))
    second = Image.new("RGB", (5, 4), (40, 190, 60))
    stream = image_buffer(first, "PNG", save_all=True, append_images=[second], duration=[20, 40])
    with Image.open(stream, lazy=lazy) as image:
        image.seek(1)
        result = image.seam_carve((3, 6))
        assert result.tobytes() == Image.new("RGB", (3, 6), (40, 190, 60)).tobytes()
        assert result.n_frames == 1
        assert result.info["duration"] == 40
        assert image.tell() == 1
        assert image.size == (5, 4)
        image.seek(0)
        assert image.tobytes() == first.tobytes()


@pytest.mark.parametrize("width", [1, 3, 9])
def test_horizontal_seams_match_transposed_vertical_seams(width: int) -> None:
    image = Image.frombytes("L", (7, 5), bytes((i * 43) % 256 for i in range(35)))
    transposed = image.transpose(Image.Transpose.TRANSPOSE)
    expected = transposed.seam_carve((width, 7)).transpose(Image.Transpose.TRANSPOSE)
    assert image.seam_carve((7, width)).tobytes() == expected.tobytes()


@pytest.mark.parametrize("source_size", [(1, 1), (1, 4), (4, 1)])
def test_single_pixel_axes_can_grow_and_shrink(source_size: tuple[int, int]) -> None:
    image = Image.new("RGBA", source_size, (5, 9, 30, 127))
    assert image.seam_carve((7, 8)).seam_carve((1, 1)).getpixel((0, 0)) == (5, 9, 30, 127)


def test_noop_copy_and_metadata() -> None:
    image = Image.new("L", (3, 2), 7)
    image.info["icc_profile"] = b"profile"
    result = image.seam_carve(image.size)
    assert result.tobytes() == image.tobytes()
    assert result.info["icc_profile"] == b"profile"
    result.putpixel((0, 0), 99)
    result.info["icc_profile"] = b"other"
    assert image.getpixel((0, 0)) == 7
    assert image.info["icc_profile"] == b"profile"


@pytest.mark.parametrize("mode", ["1", "P", "PA", "I", "I;16", "F", "RGB;F", "RGBA;F", "La", "RGBa", "RGBX", "CMYK", "HSV", "LAB", "YCbCr"])
def test_unsupported_modes(mode: str) -> None:
    with pytest.raises(ValueError, match="seam carving supports"):
        Image.new(mode, (3, 2)).seam_carve((2, 2))


@pytest.mark.parametrize("size", [(0, 2), (-1, 2), (2, 0)])
def test_invalid_dimensions(size: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match="height and width"):
        Image.new("L", (3, 2)).seam_carve(size)


@pytest.mark.parametrize("size", [(2,), (2, 3, 4), (2.5, 2)])
def test_invalid_size_type(size: tuple[float, ...]) -> None:
    with pytest.raises(TypeError):
        Image.new("L", (3, 2)).seam_carve(size)


def test_empty_and_closed_sources() -> None:
    with pytest.raises(ValueError, match="positive image dimensions"):
        Image.new("L", (0, 2)).seam_carve((1, 1))
    image = Image.new("L", (2, 2))
    image.close()
    with pytest.raises(ValueError):
        image.seam_carve((1, 1))


def test_oversized_dimensions_fail_before_processing() -> None:
    image = Image.new("RGBA", (1, 1))
    with pytest.raises(ValueError, match="dimensions are too large"):
        image.seam_carve(((1 << 32) - 1, (1 << 32) - 1))
    with pytest.raises(OverflowError):
        image.seam_carve((1 << 32, 1))
