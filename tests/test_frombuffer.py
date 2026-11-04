from __future__ import annotations

import gc
import tracemalloc
import weakref

import pytest
from PIL import Image as PillowImage, ImageDraw as PillowDraw

from blanket import Image, ImageDraw

MAPPED_MODES = ["L", "P", "RGBX", "RGBA", "CMYK", "I;16", "I;16L", "I;16B"]


@pytest.mark.parametrize("mapped", [False, True])
@pytest.mark.parametrize("xy", [(1.9, 0.2), [-1.2, -1.0], [True, False]])
def test_getpixel_coordinate_conversion_matches_owned_images(mapped, xy) -> None:
    data = bytearray(range(6))
    image = Image.frombuffer("L", (3, 2), data) if mapped else Image.frombytes("L", (3, 2), data)
    reference = PillowImage.frombytes("L", (3, 2), data)
    assert image.getpixel(xy) == reference.getpixel(xy)
    data[1] = 99
    if mapped:
        assert image.getpixel((1.9, 0.2)) == 99


@pytest.mark.parametrize("mapped", [False, True])
@pytest.mark.parametrize(("xy", "message"), [(None, "argument must be a sequence"), ("00", "argument must be a sequence"), ([0], "argument must be sequence of length 2")])
def test_getpixel_coordinate_errors_match_owned_images(mapped, xy, message) -> None:
    data = bytearray(range(6))
    image = Image.frombuffer("L", (3, 2), data) if mapped else Image.frombytes("L", (3, 2), data)
    with pytest.raises(TypeError, match=message):
        image.getpixel(xy)


@pytest.mark.parametrize("mode", MAPPED_MODES)
@pytest.mark.parametrize("orientation", [1, -1])
def test_mapped_pixel_reads_use_bounded_memory(mode: str, orientation: int) -> None:
    size = (1024, 1024)
    channels = 2 if mode.startswith("I;16") else PillowImage.getmodebands(mode)
    stride = size[0] * channels + 3
    data = bytearray(stride * size[1])
    image = Image.frombuffer(mode, size, data, "raw", mode, stride, orientation)
    reference = PillowImage.frombuffer(mode, size, data, "raw", mode, stride, orientation)
    pixels = image.load()
    tracemalloc.start()
    try:
        for value in (19, 137, 251):
            data[0] = value
            data[-4] = 255 - value
            for xy in [(0, 0), (-1, -1), (1023, 0), (0, 1023)]:
                assert image.getpixel(xy) == reference.getpixel(xy)
                assert pixels[xy] == reference.getpixel(xy)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 32 * 1024
    for xy in [(1024, 0), (-1025, 0), (0, 1024), (0, -1025)]:
        with pytest.raises(IndexError):
            image.getpixel(xy)
        with pytest.raises(IndexError):
            pixels[xy]
    assert image.getpixel([0.5, 0]) == reference.getpixel((0, 0))
    image.close()
    with pytest.raises(ValueError, match="closed"):
        image.getpixel((0, 0))
    assert pixels[0, 0] == reference.getpixel((0, 0))


@pytest.mark.parametrize("mode", MAPPED_MODES)
@pytest.mark.parametrize("orientation", [1, -1])
@pytest.mark.parametrize("padding", [0, 3])
def test_frombuffer_observes_source_changes(mode: str, orientation: int, padding: int) -> None:
    size = (5, 3)
    channels = 2 if mode.startswith("I;16") else PillowImage.getmodebands(mode)
    stride = size[0] * channels + padding
    data = bytearray((i * 37 + i // 7) % 256 for i in range(stride * size[1]))
    actual = Image.frombuffer(mode, size, data, "raw", mode, stride, orientation)
    expected = PillowImage.frombuffer(mode, size, data, "raw", mode, stride, orientation)
    access, reference = actual.load(), expected.load()
    assert actual.readonly == expected.readonly == 1
    for value in (19, 137, 251):
        data[0] = value
        data[-1] = 255 - value
        assert actual.mode == expected.mode
        assert actual.tobytes() == expected.tobytes()
        assert actual.getdata() == list(expected.get_flattened_data())
        assert actual.getpixel((0, 0)) == expected.getpixel((0, 0))
        assert access[0, 0] == reference[0, 0]
        assert actual.transpose(Image.Transpose.FLIP_LEFT_RIGHT).tobytes() == expected.transpose(PillowImage.Transpose.FLIP_LEFT_RIGHT).tobytes()
    for pixels in (access, reference):
        with pytest.raises(ValueError, match="readonly"):
            pixels[0, 0] = 0


@pytest.mark.parametrize("mode", MAPPED_MODES)
@pytest.mark.parametrize("operation", ["putpixel", "putdata", "paste", "frombytes", "draw"])
def test_frombuffer_mutations_detach(mode: str, operation: str) -> None:
    size = (5, 3)
    channels = 2 if mode.startswith("I;16") else PillowImage.getmodebands(mode)
    data = bytearray((i * 37 + i // 7) % 256 for i in range(size[0] * size[1] * channels))
    actual = Image.frombuffer(mode, size, data, "raw", (mode, 0, 1))
    expected_data = bytearray(data)
    expected = PillowImage.frombuffer(mode, size, expected_data, "raw", (mode, 0, 1))
    before = bytes(data)
    access = actual.load()
    for image, drawing in ((actual, ImageDraw), (expected, PillowDraw)):
        if operation == "putpixel":
            image.putpixel((0, 0), 71)
        elif operation == "putdata":
            image.putdata([71] * 15)
        elif operation == "paste":
            image.paste(71, (0, 0, 2, 2))
        elif operation == "frombytes":
            image.frombytes(bytes(len(data)))
        else:
            drawing.Draw(image).rectangle((0, 0, 2, 2), fill=71)
    assert actual.tobytes() == expected.tobytes()
    assert actual.readonly == expected.readonly == (1 if operation == "frombytes" else 0)
    assert data == expected_data
    assert bytes(data) == (bytes(len(data)) if operation == "frombytes" else before)
    data[:] = bytes([199]) * len(data)
    expected_data[:] = data
    assert actual.tobytes() == expected.tobytes()
    assert access[0, 0] == PillowImage.frombuffer(mode, size, data).getpixel((0, 0))


@pytest.mark.parametrize("mode,rawmode,bits", [("RGB", "RGB", 24), ("RGB", "BGR", 24), ("P", "P;2L", 2), ("P", "P;4L", 4)])
def test_frombuffer_decodes_other_layouts_independently(mode: str, rawmode: str, bits: int) -> None:
    size = (9, 3)
    row = ((size[0] + 7) // 8 * bits) if rawmode in ("P;2L", "P;4L") else size[0] * bits // 8
    data = bytearray((i * 37 + i // 7) % 256 for i in range(row * size[1]))
    actual = Image.frombuffer(mode, size, data, "raw", rawmode, 0, 1)
    expected = PillowImage.frombuffer(mode, size, data, "raw", rawmode, 0, 1)
    data[:] = bytes(len(data))
    assert actual.tobytes() == expected.tobytes()
    assert actual.readonly == expected.readonly == 0


def test_frombuffer_keeps_export_alive_and_releases_on_close() -> None:
    class Storage(bytearray):
        pass

    data = Storage(range(12))
    reference = weakref.ref(data)
    image = Image.frombuffer("L", (4, 3), data)
    del data
    gc.collect()
    assert reference() is not None
    assert image.getpixel((3, 2)) == 11
    image.close()
    gc.collect()
    assert reference() is None
    with pytest.raises(ValueError, match="closed"):
        image.tobytes()


@pytest.mark.parametrize("length", [6, 7])
def test_mapped_buffer_requires_final_row_padding(length: int) -> None:
    for module in (Image, PillowImage):
        with pytest.raises(ValueError, match="buffer is not large enough"):
            module.frombuffer("L", (3, 2), bytes(length), "raw", "L", 4, 1)


def test_frombuffer_preserves_palette_and_copies() -> None:
    data = bytearray([0, 1, 2, 0, 1, 2])
    actual = Image.frombuffer("P", (3, 2), data)
    expected = PillowImage.frombuffer("P", (3, 2), data)
    for image in (actual, expected):
        image.putpalette(bytes([10, 20, 30, 40, 50, 60, 70, 80, 90]))
        image.info["custom"] = "retained"
    copy = actual.copy()
    reference = expected.copy()
    data[0] = 2
    assert actual.convert("RGB").tobytes() == expected.convert("RGB").tobytes()
    assert copy.tobytes() == reference.tobytes()
    assert copy.getpalette() == reference.getpalette()
    assert copy.info == reference.info
    assert copy.readonly == 0


def test_frombuffer_aliasing_and_alpha_writes() -> None:
    for operation in ("paste", "alpha_composite", "putalpha"):
        data = bytearray(range(24))
        actual = Image.frombuffer("RGBA", (3, 2), data)
        expected = PillowImage.frombuffer("RGBA", (3, 2), data)
        for image in (actual, expected):
            if operation == "paste":
                image.paste(image, (0, 0))
            elif operation == "alpha_composite":
                image.alpha_composite(image)
            else:
                image.putalpha(123)
        assert actual.tobytes() == expected.tobytes()
        assert data == bytearray(range(24))


def test_frombuffer_readonly_bytes_and_mode_selection() -> None:
    for module in (Image, PillowImage):
        image = module.frombuffer("RGB", (2, 1), bytes(range(8)), "raw", "RGBA", 0, 1)
        assert image.mode == "RGBA"
        assert image.readonly == 1
        assert image.tobytes() == bytes(range(8))


def test_frombuffer_immutable_source_detaches_on_replace() -> None:
    data = bytes(range(8))
    image = Image.frombuffer("L", (4, 2), data)
    image.frombytes(bytes([71]) * 8)
    assert image.tobytes() == bytes([71]) * 8
    assert image.readonly == 0
    assert data == bytes(range(8))


@pytest.mark.parametrize("size", [(0, 0), (0, 3), (3, 0)])
def test_frombuffer_empty_images(size: tuple[int, int]) -> None:
    actual = Image.frombuffer("RGBA", size, b"")
    expected = PillowImage.frombuffer("RGBA", size, b"")
    assert actual.size == expected.size
    assert actual.tobytes() == expected.tobytes() == b""


def test_frombuffer_rejects_noncontiguous_views() -> None:
    data = memoryview(bytearray(range(16)))[::2]
    for module in (Image, PillowImage):
        with pytest.raises((BufferError, TypeError, ValueError)):
            module.frombuffer("L", (4, 2), data)
