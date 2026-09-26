from __future__ import annotations

import random

import pytest
from PIL import Image as PIL

from blanket import Image


def assert_same(actual: Image.Image, expected: PIL.Image) -> None:
    assert (actual.mode, actual.size, actual.tobytes()) == (expected.mode, expected.size, expected.tobytes())


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA", "P"])
@pytest.mark.parametrize("size", [(0, 0), (0, 4), (7, 5)])
@pytest.mark.parametrize("color", [0, 128, None, "red"])
def test_new(mode, size, color):
    actual, expected = Image.new(mode, size, color), PIL.new(mode, size, color)
    if color is not None:
        assert_same(actual, expected)
        if mode == "P":
            assert_same(actual.convert("RGB"), expected.convert("RGB"))
    assert actual.size == size


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
@pytest.mark.parametrize("source_mode", ["L", "RGB", "RGBA"])
@pytest.mark.parametrize("mask_mode", [None, "L", "RGBA"])
@pytest.mark.parametrize("position", [(0, 0), (-2, -1), (5, 4), (20, 20)])
def test_paste(mode, source_mode, mask_mode, position):
    rng = random.Random(71)
    background = PIL.frombytes(mode, (7, 6), rng.randbytes(42 * len(mode)))
    foreground = PIL.frombytes(source_mode, (4, 3), rng.randbytes(12 * len(source_mode)))
    mask = PIL.frombytes(mask_mode, (4, 3), rng.randbytes(12 * len(mask_mode))) if mask_mode else None
    actual = Image.frombytes(mode, background.size, background.tobytes())
    source = Image.frombytes(source_mode, foreground.size, foreground.tobytes())
    bmask = Image.frombytes(mask_mode, mask.size, mask.tobytes()) if mask else None
    assert actual.paste(source, position, bmask) is None
    background.paste(foreground, position, mask)
    assert_same(actual, background)


@pytest.mark.parametrize("mode,color", [("L", 99), ("RGB", "red"), ("RGBA", (21, 73, 92, 111)), ("P", 3)])
def test_paste_fill_and_alias(mode, color):
    actual, expected = Image.new(mode, (6, 5)), PIL.new(mode, (6, 5))
    actual.paste(color, (-1, 1, 4, 3))
    expected.paste(color, (-1, 1, 4, 3))
    actual.paste(actual, (1, 0))
    expected.paste(expected.copy(), (1, 0))
    assert_same(actual, expected)
    amask, pmask = Image.new("L", (6, 5), 127), PIL.new("L", (6, 5), 127)
    actual.paste(color, amask)
    expected.paste(color, pmask)
    assert_same(actual, expected)


def test_alpha_composite_rounding():
    rng = random.Random(819)
    a, b = [PIL.frombytes("RGBA", (257, 257), rng.randbytes(257 * 257 * 4)) for _ in range(2)]
    ba, bb = [Image.frombytes("RGBA", im.size, im.tobytes()) for im in (a, b)]
    original = ba.tobytes()
    assert_same(Image.alpha_composite(ba, bb), PIL.alpha_composite(a, b))
    assert ba.tobytes() == original
    for dest, source in [((0, 0), (0, 0)), ((-3, 2), (1, 2, 20, 30)), ((250, 250), (1, 1, 12, 12)), ((0, 0), (250, 250, 260, 260))]:
        ac, ec = ba.copy(), a.copy()
        assert ac.alpha_composite(bb, dest, source) is None
        ec.alpha_composite(b, dest, source)
        assert_same(ac, ec)
    ba.alpha_composite(ba)
    a.alpha_composite(a.copy())
    assert_same(ba, a)


@pytest.mark.parametrize("mode", ["RGB", "RGBA"])
@pytest.mark.parametrize("alpha", [0, 128, 255, -1, 300, "band"])
def test_putalpha(mode, alpha):
    actual, expected = Image.new(mode, (7, 5), "red"), PIL.new(mode, (7, 5), "red")
    actual.info["custom"] = "kept"
    if alpha == "band":
        raw = random.Random(16).randbytes(35)
        aa, pa = Image.frombytes("L", (7, 5), raw), PIL.frombytes("L", (7, 5), raw)
    else:
        aa = pa = alpha
    assert actual.putalpha(aa) is None
    expected.putalpha(pa)
    assert_same(actual, expected)
    assert actual.info["custom"] == "kept"


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
def test_merge(mode):
    raw = random.Random(92).randbytes(35 * len(mode))
    actual, expected = Image.frombytes(mode, (7, 5), raw), PIL.frombytes(mode, (7, 5), raw)
    bands = actual.split()
    merged = Image.merge(mode, bands)
    assert_same(merged, PIL.merge(mode, expected.split()))
    bands[0].paste(0, (0, 0, 7, 5))
    assert merged.tobytes() == raw


def test_validation_and_closed_images():
    image = Image.new("RGBA", (3, 2))
    before = image.tobytes()
    for operation in [
        lambda: Image.new("unsupported", (3, 2)),
        lambda: Image.new("RGB", (-1, 2)),
        lambda: Image.merge("RGB", [Image.new("L", (3, 2))]),
        lambda: Image.merge("L", [Image.new("P", (3, 2))]),
        lambda: Image.merge("RGB", [Image.new("L", size) for size in [(3, 2), (2, 2), (3, 2)]]),
        lambda: image.paste(1),
        lambda: image.paste(image, (0, 0, 1, 1)),
        lambda: image.paste(image, mask=Image.new("RGB", image.size)),
        lambda: image.paste(image, mask=Image.new("L", (1, 1))),
        lambda: image.putalpha(Image.new("RGB", image.size)),
        lambda: image.putalpha(image),
        lambda: Image.new("L", image.size).putalpha(255),
        lambda: Image.new("P", image.size).putalpha(255),
        lambda: Image.alpha_composite(image, Image.new("RGB", image.size)),
        lambda: Image.alpha_composite(image, Image.new("RGBA", (1, 1))),
        lambda: image.alpha_composite(image, source=(-1, 0)),
    ]:
        with pytest.raises((ValueError, TypeError)):
            operation()
        assert image.tobytes() == before
    image.close()
    for operation in [lambda: image.paste(1, (0, 0, 3, 2)), lambda: image.putalpha(1), lambda: Image.alpha_composite(image, image)]:
        with pytest.raises(ValueError, match="closed"):
            operation()


def test_high_depth_rejected():
    image = Image.frombytes("RGBA", (1, 1), bytes(8), bit_depth=16)
    for operation in [lambda: image.paste(Image.new("RGBA", (1, 1))), lambda: image.putalpha(255), lambda: Image.alpha_composite(image, image)]:
        with pytest.raises(ValueError, match="8-bit"):
            operation()


@pytest.mark.parametrize("mode,color", [("L", (42,)), ("RGB", (20, 30, 40)), ("RGBA", (20, 30, 40)), ("RGBA", (20, 30, 40, 50)), ("P", (3,)), ("P", (20, 30, 40))])
def test_new_tuple_color(mode, color):
    assert_same(Image.new(mode, (3, 2), color), PIL.new(mode, (3, 2), color))


@pytest.mark.parametrize("alpha", [0, 1, 127, 254, 255])
def test_transparent_fill_and_composite(alpha):
    actual = Image.new("RGBA", (2, 2), (19, 29, 39, 0))
    expected = PIL.new("RGBA", (2, 2), (19, 29, 39, 0))
    actual.paste((73, 83, 93, 113), mask=Image.new("L", actual.size, alpha))
    expected.paste((73, 83, 93, 113), mask=PIL.new("L", expected.size, alpha))
    assert_same(actual, expected)
    overlay, reference = Image.new("RGBA", actual.size, (3, 7, 11, alpha)), PIL.new("RGBA", expected.size, (3, 7, 11, alpha))
    assert_same(Image.alpha_composite(actual, overlay), PIL.alpha_composite(expected, reference))


def test_empty_and_alias_alpha():
    for size in [(0, 0), (0, 4), (4, 0)]:
        image = Image.new("RGBA", size)
        image.putalpha(128)
        image.alpha_composite(image)
        assert Image.merge("RGBA", image.split()).tobytes() == b""
    alpha = Image.new("L", (2, 3), 42)
    image = Image.merge("RGBA", [alpha] * 4)
    image.putalpha(alpha)
    alpha.paste(0, (0, 0, 2, 3))
    assert image.getpixel((0, 0)) == (42, 42, 42, 42)


@pytest.mark.parametrize("width", [15, 16, 17, 31, 32, 33, 257])
@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
@pytest.mark.parametrize("mask_mode", ["L", "RGBA"])
def test_masked_paste_vector_boundaries(width, mode, mask_mode):
    rng = random.Random(492)
    size = (width, 5)
    source = PIL.frombytes(mode, size, rng.randbytes(width * 5 * len(mode)))
    background = PIL.frombytes(mode, size, rng.randbytes(width * 5 * len(mode)))
    mask = PIL.frombytes(mask_mode, size, rng.randbytes(width * 5 * len(mask_mode)))
    actual = Image.frombytes(mode, size, background.tobytes())
    bsource = Image.frombytes(mode, size, source.tobytes())
    bmask = Image.frombytes(mask_mode, size, mask.tobytes())
    for position in [(0, 0), (-1, 1)]:
        actual.paste(bsource, position, bmask)
        background.paste(source, position, mask)
        assert_same(actual, background)
    # Test the distinct color-fill rule on transparent RGBA pixels in vectors
    # and tails, including zero mask values.
    actual = Image.new(mode, size)
    background = PIL.new(mode, size)
    actual.paste("red", mask=bmask)
    background.paste("red", mask=mask)
    assert_same(actual, background)


def test_paste_mask_alias_and_failed_mutation():
    rng = random.Random(20)
    raw = rng.randbytes(33 * 4 * 3)
    actual = Image.frombytes("RGBA", (33, 3), raw)
    expected = PIL.frombytes("RGBA", (33, 3), raw)
    source = Image.new("RGBA", actual.size, "red")
    reference = PIL.new("RGBA", expected.size, "red")
    actual.paste(source, mask=actual)
    expected.paste(reference, mask=expected.copy())
    assert_same(actual, expected)
    before = actual.tobytes()
    bad_mask = Image.new("L", actual.size)
    bad_mask.close()
    with pytest.raises(ValueError, match="closed"):
        actual.paste(source, mask=bad_mask)
    assert actual.tobytes() == before


@pytest.mark.parametrize("size", [(17, 3), (513, 513)])
def test_merge_and_alpha_kernel_boundaries(size):
    rng = random.Random(745)
    for mode in ("RGB", "RGBA"):
        raw = rng.randbytes(size[0] * size[1] * len(mode))
        actual = Image.frombytes(mode, size, raw)
        expected = PIL.frombytes(mode, size, raw)
        assert_same(Image.merge(mode, actual.split()), PIL.merge(mode, expected.split()))
        alpha_raw = rng.randbytes(size[0] * size[1])
        actual.putalpha(Image.frombytes("L", size, alpha_raw))
        expected.putalpha(PIL.frombytes("L", size, alpha_raw))
        assert_same(actual, expected)
