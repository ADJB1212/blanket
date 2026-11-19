"""Reference histogram and mean/variance transfer contracts."""

from io import BytesIO

import pytest

from blanket import Image, ImageOps

OPERATIONS = [ImageOps.match_histogram, ImageOps.color_transfer]


def gray(values, mode="L", depth=8):
    image = Image.new(mode, (len(values), 1))
    if image.bit_depth != depth:
        image = image.convert(mode, bit_depth=depth)
    image.putdata(values)
    return image


def test_histogram_uses_normalized_cdf_and_smallest_reaching_reference():
    source = gray([0, 0, 100, 200])
    reference = gray([10, 20, 30, 40, 50, 60, 70, 80])
    assert list(ImageOps.match_histogram(source, reference).getdata()) == [40, 40, 60, 80]
    assert list(source.getdata()) == [0, 0, 100, 200]
    assert list(reference.getdata()) == [10, 20, 30, 40, 50, 60, 70, 80]


@pytest.mark.parametrize("operation", OPERATIONS)
def test_independent_color_channels(operation):
    source = Image.frombytes("RGB", (2, 1), bytes([10, 100, 200, 30, 140, 220]))
    reference = Image.frombytes("RGB", (2, 1), bytes([50, 20, 80, 90, 40, 160]))
    assert operation(source, reference).tobytes() == reference.tobytes()


def test_mean_variance_population_scaling_and_clipping():
    assert list(ImageOps.color_transfer(gray([10, 20, 30]), gray([80, 100, 120])).getdata()) == [80, 100, 120]
    assert list(ImageOps.color_transfer(gray([50, 50]), gray([0, 255])).getdata()) == [128, 128]
    assert list(ImageOps.color_transfer(gray([0, 100, 200]), gray([70])).getdata()) == [70, 70, 70]
    assert list(ImageOps.color_transfer(gray([0, 100, 100]), gray([0, 255])).getdata()) == [0, 218, 218]


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("mode", ["L", "LA", "RGB", "RGBA", "I;16", "I;16L", "I;16B", "I;16N"])
def test_identity_metadata_and_independent_storage(operation, mode):
    source = gray([1, 300, 65535], mode=mode, depth=16) if mode.startswith("I;") else Image.new(mode, (3, 2), 90 if mode == "L" else (90,) * len(mode))
    source.info["custom"] = b"source"
    before = source.tobytes()
    result = operation(source, source)
    assert result.tobytes() == before
    assert (result.mode, result.bit_depth, result.size) == (source.mode, source.bit_depth, source.size)
    assert result.info["custom"] == b"source"
    result.paste(0, (0, 0, *result.size))
    assert source.tobytes() == before


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("depth", [8, 10, 12, 16])
@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
def test_depth_scaling(operation, depth, mode):
    maximum = (1 << depth) - 1
    source = Image.new(mode, (2, 1)).convert(mode, bit_depth=depth)
    source.putdata([0, maximum] if mode == "L" else [(0,) * 3 + ((maximum,) if mode == "RGBA" else ()), (maximum,) * len(mode)])
    reference = gray([64, 192])
    if mode != "L":
        reference = reference.convert("RGB")
    result = operation(source, reference)
    expected = [int(value * maximum / 255 + 0.5) for value in [64, 192]]
    assert list(result.getdata()) == (expected if mode == "L" else [(value,) * 3 + ((maximum,) if mode == "RGBA" else ()) for value in expected])
    assert result.bit_depth == depth


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("mode", ["LA", "RGBA"])
def test_alpha_is_preserved_and_hidden_pixels_are_ignored(operation, mode):
    colors = len(mode) - 1
    source = Image.new(mode, (3, 1))
    source.putdata([(200,) * colors + (0,), (10,) * colors + (1,), (30,) * colors + (180,)])
    reference = Image.new(mode, (3, 1))
    reference.putdata([(250,) * colors + (0,), (50,) * colors + (255,), (90,) * colors + (10,)])
    result = operation(source, reference)
    assert list(result.getdata()) == [(200,) * colors + (0,), (50,) * colors + (1,), (90,) * colors + (180,)]


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("mode", ["L", "RGB"])
@pytest.mark.parametrize("depth", [8, 16])
def test_color_keys_exclude_hidden_pixels_from_both_distributions(operation, mode, depth):
    def color(value):
        return value if mode == "L" else (value,) * 3

    source = gray([color(100), color(10)], mode=mode, depth=depth)
    source.info["transparency"] = color(100)
    reference = gray([color(200), color(100)], mode=mode, depth=depth)
    reference.info["transparency"] = color(200)
    result = operation(source, reference)
    assert result.mode == ("LA" if mode == "L" and depth == 8 else "RGBA")
    assert result.bit_depth == depth
    assert list(result.getdata()) == [(100,) * (len(result.mode) - 1) + (0,), (100,) * (len(result.mode) - 1) + ((1 << depth) - 1,)]
    assert "transparency" not in result.info
    assert source.info["transparency"] == color(100)


@pytest.mark.parametrize("operation", OPERATIONS)
def test_luminance_adjusts_color_differences_and_clips(operation):
    source = Image.new("RGB", (1, 1), (40, 80, 120))
    assert operation(source, gray([100]), mode="luminance").getpixel((0, 0)) == (67, 107, 147)
    assert operation(source, gray([0]), mode="luminance").getpixel((0, 0)) == (0, 7, 47)


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("mode", ["1", "P", "PA", "I", "F", "RGBa", "La", "RGBX", "CMYK", "YCbCr", "LAB", "HSV", "RGB;F"])
def test_unsupported_modes(operation, mode):
    image = Image.new(mode, (1, 1))
    with pytest.raises(ValueError, match="unsigned grayscale or RGB"):
        operation(image, gray([10]))
    with pytest.raises(ValueError, match="unsigned grayscale or RGB"):
        operation(gray([10]), image)


@pytest.mark.parametrize("operation", OPERATIONS)
def test_invalid_inputs(operation):
    with pytest.raises(ValueError, match="mode must be"):
        operation(gray([1]), gray([2]), mode="invalid")
    with pytest.raises(ValueError, match="same number"):
        operation(gray([1]), Image.new("RGB", (1, 1)))
    for bad in [Image.new("L", (0, 1)), Image.new("RGBA", (2, 2))]:
        with pytest.raises(ValueError):
            operation(bad, gray([1]), mode="luminance")
        with pytest.raises(ValueError):
            operation(gray([1]), bad, mode="luminance")


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("lazy", [False, True])
def test_sequence_current_frame_and_navigation(operation, lazy):
    stream = BytesIO()
    Image.new("RGB", (3, 2), (10, 20, 30)).save(stream, "PNG", save_all=True, append_images=[Image.new("RGB", (3, 2), (80, 90, 100))], duration=20)
    stream.seek(0)
    with Image.open(stream, lazy=lazy) as source:
        source.seek(1)
        before = source.tobytes()
        result = operation(source, Image.new("RGB", (1, 1), (50, 60, 70)))
        assert result.getpixel((0, 0)) == (50, 60, 70)
        assert source.tell() == 1
        assert source.tobytes() == before
