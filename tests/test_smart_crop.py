"""Content-aware crop selection and source preservation."""

from io import BytesIO

import pytest

from blanket import Image, ImageOps


@pytest.mark.parametrize("vertical", [False, True])
@pytest.mark.parametrize("side", [0, 1])
def test_off_center_subject(vertical, side):
    size = (40, 120) if vertical else (120, 40)
    image = Image.new("RGB", size)
    start = 100 if side else 0
    box = (0, start, 40, start + 20) if vertical else (start, 0, start + 20, 40)
    image.paste((255, 0, 0), box)
    result = ImageOps.smart_crop(image, (40, 40), Image.Resampling.NEAREST)
    assert result.size == (40, 40)
    point = (20, 30 if side else 10) if vertical else (30 if side else 10, 20)
    assert result.getpixel(point) == (255, 0, 0)
    assert result.tobytes().count(bytes([255])) >= 20 * 40
    assert ImageOps.fit(image, (40, 40)).getbbox() is None
    assert image.size == size


@pytest.mark.parametrize("mode,color", [("L", 80), ("RGB", (80, 80, 80)), ("RGBA", (80, 80, 80, 255)), ("RGBA", (80, 80, 80, 128)), ("RGBA", (0, 0, 0, 0))])
def test_flat_center_fallback(mode, color):
    image = Image.new(mode, (91, 33), color)
    result = ImageOps.smart_crop(image, (17, 25))
    assert result.tobytes() == ImageOps.fit(image, (17, 25)).tobytes()


def test_transparent_rgb_does_not_affect_placement():
    image = Image.new("RGBA", (120, 40), (0, 0, 0, 0))
    image.paste((255, 0, 0, 255), (0, 0, 20, 40))
    hidden = image.copy()
    hidden.paste((255, 255, 255, 0), (40, 0, 120, 40))
    actual = ImageOps.smart_crop(image, (40, 40), Image.Resampling.NEAREST)
    assert actual.tobytes() == ImageOps.smart_crop(hidden, (40, 40), Image.Resampling.NEAREST).tobytes()
    assert actual.getpixel((10, 20)) == (255, 0, 0, 255)


@pytest.mark.parametrize("mode", ["1", "L", "LA", "P", "PA", "I", "I;16", "I;16B", "F", "RGB", "RGBA", "RGBa", "La", "RGBX", "CMYK", "YCbCr", "LAB", "HSV", "RGB;F", "RGBA;F"])
def test_modes_and_metadata(mode):
    image = Image.new(mode, (60, 20))
    image.info["test"] = b"preserved"
    original = image.tobytes()
    result = ImageOps.smart_crop(image, (10, 10), Image.Resampling.NEAREST)
    assert (result.mode, result.bit_depth, result.size) == (image.mode, image.bit_depth, (10, 10))
    assert result.info["test"] == b"preserved"
    assert image.tobytes() == original
    if image.palette is not None:
        assert result.palette is not None
        assert result.palette.tobytes() == image.palette.tobytes()


@pytest.mark.parametrize("depth", [10, 12, 16])
@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
def test_high_depth_preservation(mode, depth):
    color = (1 << depth) - 1
    image = Image.new(mode, (60, 20), 255 if mode == "L" else (255,) * len(mode)).convert(mode, bit_depth=depth)
    result = ImageOps.smart_crop(image, (10, 10))
    assert result.bit_depth == depth
    assert result.getpixel((5, 5)) == image.getpixel((30, 10))
    assert result.getpixel((5, 5)) == (color if mode == "L" else (color,) * len(mode))


@pytest.mark.parametrize("lazy", [False, True])
def test_current_frame_and_source_navigation(lazy):
    first = Image.new("RGB", (120, 40))
    first.paste((255, 0, 0), (0, 0, 20, 40))
    second = Image.new("RGB", (120, 40))
    second.paste((0, 0, 255), (100, 0, 120, 40))
    stream = BytesIO()
    first.save(stream, "PNG", save_all=True, append_images=[second], duration=[20, 40])
    stream.seek(0)
    with Image.open(stream, lazy=lazy) as image:
        image.seek(1)
        image.info["custom"] = b"current frame"
        before = image.tobytes()
        result = ImageOps.smart_crop(image, (40, 40), Image.Resampling.NEAREST)
        assert result.getpixel((30, 20)) == (0, 0, 255)
        assert result.info["custom"] == b"current frame"
        assert image.tell() == 1
        assert image.tobytes() == before
        image.seek(0)
        assert image.tobytes() == first.tobytes()
    assert result.getpixel((30, 20)) == (0, 0, 255)


@pytest.mark.parametrize("source,target", [((1, 31), (10, 10)), ((31, 1), (10, 10)), ((1, 1), (1, 40)), ((2000, 1), (1, 100)), ((2000, 999), (19, 17))])
def test_small_and_large_geometry(source, target):
    image = Image.new("RGB", source, (255, 128, 64))
    result = ImageOps.smart_crop(image, target)
    assert result.size == target
    assert result.getpixel((0, 0)) == (255, 128, 64)


def test_same_aspect_and_independent_output():
    image = Image.new("L", (20, 10), 77)
    result = ImageOps.smart_crop(image, image.size)
    assert result.tobytes() == image.tobytes()
    result.putpixel((0, 0), 1)
    assert image.getpixel((0, 0)) == 77


@pytest.mark.parametrize("size", [(0, 10), (-1, 10), (10, 0), (10,), (10, 10, 10)])
def test_invalid_size(size):
    with pytest.raises(ValueError, match="size"):
        ImageOps.smart_crop(Image.new("RGB", (20, 10)), size)


def test_invalid_inputs():
    with pytest.raises(TypeError, match="integer"):
        ImageOps.smart_crop(Image.new("RGB", (20, 10)), (2.5, 10))
    with pytest.raises(ValueError, match="empty"):
        ImageOps.smart_crop(Image.new("RGB", (0, 10)), (10, 10))
    with pytest.raises(TypeError, match="Blanket Image"):
        ImageOps.smart_crop(None, (10, 10))
    with pytest.raises(ValueError):
        ImageOps.smart_crop(Image.new("RGB", (20, 10)), (10, 10), -1)
