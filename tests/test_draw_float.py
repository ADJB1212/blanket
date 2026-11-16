import pytest

from blanket import Image, ImageDraw


@pytest.mark.parametrize("mode,channels", [("RGB;F", 3), ("RGBA;F", 4)])
@pytest.mark.parametrize("shape", ["point", "line", "rectangle", "ellipse", "polygon", "bitmap"])
@pytest.mark.parametrize("values", [(-0.25, 1.5, 3.25, 0.5), (1, 2, 3, 4)])
def test_float_color_drawing_preserves_numeric_inks(mode, channels, shape, values):
    image = Image.new(mode, (7, 7))
    ink = values[:channels]
    draw = ImageDraw.Draw(image)
    if shape == "point":
        draw.point((3, 3), fill=ink)
    elif shape in ("line", "polygon"):
        getattr(draw, shape)([(1, 1), (5, 1), (3, 5)], fill=ink)
    elif shape == "bitmap":
        draw.bitmap((1, 1), Image.new("1", (2, 2), 1), fill=ink)
    else:
        getattr(draw, shape)((1, 1, 5, 5), fill=ink)
    pixel = Image.new(mode, (1, 1), ink).tobytes()
    raw = image.tobytes()
    stride = channels * 4
    samples = [raw[i : i + stride] for i in range(0, len(raw), stride)]
    assert pixel in samples
    assert all(sample in (bytes(stride), pixel) for sample in samples)


@pytest.mark.parametrize("mode,channels", [("RGB;F", 3), ("RGBA;F", 4)])
def test_float_color_drawing_default_ink(mode, channels):
    image = Image.new(mode, (1, 1))
    ImageDraw.Draw(image).point((0, 0))
    assert image.getpixel((0, 0)) == (1.0,) * channels


@pytest.mark.parametrize("mode", ["RGB;F", "RGBA;F"])
def test_float_color_drawing_rejects_named_inks(mode):
    image = Image.new(mode, (1, 1))
    with pytest.raises(ValueError, match="numeric ink"):
        ImageDraw.Draw(image).point((0, 0), fill="white")


@pytest.mark.parametrize("mode,channels", [("F", 1), ("RGB;F", 3), ("RGBA;F", 4)])
def test_float_floodfill_uses_numeric_channel_distances(mode, channels):
    def pixel(value):
        return value if channels == 1 else (value,) * channels

    image = Image.new(mode, (4, 1))
    image.putdata([pixel(1.0), pixel(1.125), pixel(2.0), pixel(1.0)])
    ImageDraw.floodfill(image, (0, 0), pixel(10.0), thresh=0.5)
    assert list(image.getdata()) == [pixel(10.0), pixel(10.0), pixel(2.0), pixel(1.0)]


@pytest.mark.parametrize("mode,channels", [("F", 1), ("RGB;F", 3), ("RGBA;F", 4)])
def test_float_floodfill_treats_signed_zero_as_equal(mode, channels):
    def pixel(value):
        return value if channels == 1 else (value,) * channels

    image = Image.new(mode, (2, 1))
    image.putdata([pixel(0.0), pixel(-0.0)])
    ImageDraw.floodfill(image, (0, 0), pixel(1.0))
    assert list(image.getdata()) == [pixel(1.0)] * 2
