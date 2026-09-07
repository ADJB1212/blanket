from __future__ import annotations

import random
from array import array
from io import StringIO
from pathlib import Path

import pytest
from blanket import Image, ImagePalette
from PIL import Image as PillowImage
from PIL import ImagePalette as PillowPalette


def assert_palette(actual: ImagePalette.ImagePalette, expected: PillowPalette.ImagePalette) -> None:
    assert (actual.mode, actual.rawmode, actual.dirty) == (expected.mode, expected.rawmode, expected.dirty)
    assert type(actual.palette) is type(expected.palette)
    assert actual.palette == expected.palette
    assert actual.colors == expected.colors
    assert actual.getdata() == expected.getdata()


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
@pytest.mark.parametrize("storage", [bytes, bytearray, list, tuple, lambda values: array("B", values)])
def test_storage_copy_and_cache(mode: str, storage: type) -> None:
    data = [1, 2, 3, 4, 1, 2, 3, 4, 1, 2]
    actual = ImagePalette.ImagePalette(mode, storage(data))
    expected = PillowPalette.ImagePalette(mode, storage(data))
    assert_palette(actual, expected)
    assert actual.colors is actual.colors
    assert actual.tostring() == expected.tostring() == bytes(data)
    actual.dirty = expected.dirty = 1
    copied = actual.copy()
    assert_palette(copied, expected.copy())
    assert copied.colors is not actual.colors
    actual.palette = expected.palette = bytearray([7] * len(mode))
    assert_palette(actual, expected)
    assert copied.tobytes() == bytes(data)
    custom = {(2, 4, 6): 17}
    actual.colors = custom
    assert actual.colors is custom


@pytest.mark.parametrize("data", [None, [], b"", bytearray()])
def test_empty_constructor(data: object) -> None:
    assert_palette(ImagePalette.ImagePalette(palette=data), PillowPalette.ImagePalette(palette=data))


@pytest.mark.parametrize("mode", ["RGB", "RGBA"])
def test_color_allocation(mode: str) -> None:
    actual = ImagePalette.ImagePalette(mode)
    expected = PillowPalette.ImagePalette(mode)
    for color in [(1, 2, 3), (1, 2, 3, 255), (4, 5, 6), (1, 2, 3)]:
        assert actual.getcolor(color) == expected.getcolor(color)
        assert_palette(actual, expected)


@pytest.mark.parametrize("color", ["red", [1, 2, 3], 0, None, (1, 2, 3, 0), (-1, 2, 3), (256, 2, 3), (1.0, 2, 3)])
def test_invalid_color(color: object) -> None:
    with pytest.raises(Exception) as reference:
        PillowPalette.ImagePalette().getcolor(color)
    with pytest.raises(type(reference.value)) as actual:
        ImagePalette.ImagePalette().getcolor(color)
    assert str(actual.value) == str(reference.value)


@pytest.mark.parametrize("reserved", [{}, {"background": 0}, {"background": 0, "transparency": 1}, {"transparency": (0, 0, 0)}])
def test_reserved_indices(reserved: dict[str, object]) -> None:
    actual = ImagePalette.ImagePalette()
    expected = PillowPalette.ImagePalette()
    image = PillowImage.new("P", (1, 1))
    image.info.update(reserved)
    for color in [(2, 4, 6), (7, 8, 9)]:
        assert actual.getcolor(color, image) == expected.getcolor(color, image)
        assert_palette(actual, expected)


def test_full_palette_and_reuse() -> None:
    actual = ImagePalette.wedge()
    expected = PillowPalette.wedge()
    color = (2, 4, 6)
    for palette in (actual, expected):
        with pytest.raises(ValueError, match="cannot allocate more than 256 colors"):
            palette.getcolor(color)
    image = PillowImage.new("P", (1, 1))
    image.info["transparency"] = 255
    actual_data, expected_data = actual.palette, expected.palette
    assert actual.getcolor(color, image) == expected.getcolor(color, image) == 254
    assert actual_data == expected_data
    assert_palette(actual, expected)
    full = PillowImage.frombytes("P", (256, 1), bytes(range(256)))
    for palette in (actual, expected):
        with pytest.raises(ValueError, match="cannot allocate more than 256 colors"):
            palette.getcolor((5, 6, 7), full)
    # Blanket images can provide the same histogram/info allocation context.
    gray = Image.frombytes("L", (1, 1), b"\0")
    assert ImagePalette.wedge().getcolor(color, gray) == 255


@pytest.mark.parametrize("rawmode", ["RGB;L", "RGBA", "BGR"])
def test_raw_palette(rawmode: str) -> None:
    data = bytearray(range(24))
    actual, expected = ImagePalette.raw(rawmode, data), PillowPalette.raw(rawmode, data)
    assert_palette(actual, expected)
    assert actual.getdata()[1] is data
    assert_palette(actual.copy(), expected.copy())
    for operation in (lambda: actual.tobytes(), lambda: actual.tostring(), lambda: actual.getcolor((1, 2, 3)), lambda: actual.save(StringIO())):
        with pytest.raises(ValueError, match="palette contains raw palette data"):
            operation()


@pytest.mark.parametrize("factory", ["wedge", "negative", "random"])
@pytest.mark.parametrize("mode", ["", "L", "RGB", "RGBA", "custom"])
def test_factories(factory: str, mode: str) -> None:
    state = random.getstate()
    try:
        random.seed(17)
        actual = getattr(ImagePalette, factory)(mode)
        actual_state = random.getstate()
        random.seed(17)
        expected = getattr(PillowPalette, factory)(mode)
        assert random.getstate() == actual_state
        if mode:
            assert_palette(actual, expected)
        else:
            assert actual.palette == expected.palette
    finally:
        random.setstate(state)


@pytest.mark.parametrize("white", ["#fff0c0", "red", "rebeccapurple", "#123", "#1234", "rgb(10, 20, 30)", "hsl(120, 50%, 50%)"])
def test_sepia(white: str) -> None:
    assert_palette(ImagePalette.sepia(white), PillowPalette.sepia(white))


@pytest.mark.parametrize("white", [-100, 0, 1, 128, 255, 256, 12.5, 0.1, 10**30])
def test_linear_lut(white: float) -> None:
    assert ImagePalette.make_linear_lut(0, white) == PillowPalette.make_linear_lut(0, white)
    with pytest.raises(NotImplementedError, match="unavailable when black is non-zero"):
        ImagePalette.make_linear_lut(1, white)


def test_linear_lut_all_byte_endpoints() -> None:
    for white in range(256):
        assert ImagePalette.make_linear_lut(0, white) == PillowPalette.make_linear_lut(0, white)


@pytest.mark.parametrize("white", [-(2**63), 2**63 - 1, -1, 256])
def test_native_linear_lut_wide_fallback(white: int) -> None:
    from blanket._blanket import palette_linear, palette_sepia

    assert palette_linear(white) == [white * value // 255 for value in range(256)]
    channels = (white, 240, 0)
    assert palette_sepia(channels) == [channel * value // 255 for value in range(256) for channel in channels]


@pytest.mark.parametrize("exp", [0, 0.01, 0.5, 1, 1.8, 2.2, 10, float("inf"), float("nan"), -1])
def test_gamma_lut(exp: float) -> None:
    try:
        expected = PillowPalette.make_gamma_lut(exp)
    except Exception as error:
        with pytest.raises(type(error)):
            ImagePalette.make_gamma_lut(exp)
    else:
        assert ImagePalette.make_gamma_lut(exp) == expected


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
def test_save(mode: str, tmp_path: Path) -> None:
    actual = ImagePalette.ImagePalette(mode, [1, 2, 3, 4])
    expected = PillowPalette.ImagePalette(mode, [1, 2, 3, 4])
    stream, reference = StringIO(), StringIO()
    actual.save(stream)
    expected.save(reference)
    assert not stream.closed
    assert stream.getvalue() == reference.getvalue()
    path = tmp_path / "palette.pal"
    actual.save(str(path))
    assert path.read_text() == reference.getvalue()
    if mode != "RGBA":
        assert ImagePalette.load(str(path)) == PillowPalette.load(str(path))


@pytest.mark.parametrize(
    "payload",
    [
        b"",
        b"# comment\n",
        b"0 255 0 0\n17 10\n255 1 2 3\n256 9\n-1 0\n",
        b"GIMP Palette\nName: Example\nColumns: 2\n#\n255 0 0 Red\n0 255 0 Green\n",
        b"GIMP Palette\n",
        b"GIMP Palette\n" + b"1 2 3\n" * 270,
        b"garbage",
        b"\n",
        b"0 256 0 0\n",
        b"GIMP Palette\n1 2\n",
        b"1" * 101,
        b"GIMP Gradient\n0\n",
        b"GIMP Gradient\n1\n0 .5 1 0 0 0 1 1 1 1 1 0 1\n",
    ],
)
def test_load_formats_and_errors(payload: bytes, tmp_path: Path) -> None:
    path = tmp_path / "palette"
    path.write_bytes(payload)
    try:
        expected = PillowPalette.load(str(path))
    except Exception as error:
        with pytest.raises(type(error)) as actual:
            ImagePalette.load(str(path))
        assert str(actual.value) == str(error)
    else:
        assert ImagePalette.load(str(path)) == expected


@pytest.mark.parametrize("kind", range(5))
@pytest.mark.parametrize("middle", [0, 0.2, 0.5, 0.9])
@pytest.mark.parametrize("named", [False, True])
def test_gradient_blends(kind: int, middle: float, named: bool, tmp_path: Path) -> None:
    name = "Name: Sample\n" if named else ""
    payload = f"GIMP Gradient\n{name}2\n0 {middle * 0.5} .5 0 .2 .8 0 1 .7 .3 1 {kind} 0\n.5 .8 1 1 .7 .3 1 .1 .9 .4 .2 {kind} 0\n"
    path = tmp_path / "gradient.ggr"
    path.write_text(payload)
    assert ImagePalette.load(str(path)) == PillowPalette.load(str(path))
