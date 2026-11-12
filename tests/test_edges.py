"""Normalized gradients and Canny edge detection contracts."""

import math
import struct
from io import BytesIO

import pytest

from blanket import Image, ImageFilter


@pytest.mark.parametrize("operator,outer,middle,scale", [(ImageFilter.sobel, 1, 2, 8), (ImageFilter.scharr, 3, 10, 32)])
@pytest.mark.parametrize("size", [(1, 1), (1, 17), (19, 1), (9, 7), (17, 13), (33, 21)])
def test_gradients_against_convolution(operator, outer, middle, scale, size):
    width, height = size
    values = [(i * 719 + i // width * 313) % 256 for i in range(width * height)]
    image = Image.frombytes("L", size, bytes(values))
    gx, gy = operator(image)
    kernel = ((-outer, 0, outer), (-middle, 0, middle), (-outer, 0, outer))
    for output, transpose in [(gx, False), (gy, True)]:
        expected = []
        for y in range(height):
            for x in range(width):
                total = 0
                for ky in range(3):
                    for kx in range(3):
                        nx = max(0, min(width - 1, x + kx - 1))
                        ny = max(0, min(height - 1, y + ky - 1))
                        total += values[ny * width + nx] * (kernel[kx][ky] if transpose else kernel[ky][kx])
                expected.append(total / (255 * scale))
        assert output.mode == "F"
        assert output.size == size
        assert output.getdata() == pytest.approx(expected, abs=1e-7)


@pytest.mark.parametrize("operator", [ImageFilter.sobel, ImageFilter.scharr])
def test_gradient_sign_and_border(operator):
    image = Image.frombytes("L", (17, 5), bytes(x * 10 for _ in range(5) for x in range(17)))
    gx, gy = operator(image)
    assert gx.getpixel((8, 2)) == pytest.approx(10 / 255)
    assert gx.getpixel((0, 2)) == pytest.approx(5 / 255)
    assert gx.getpixel((16, 2)) == pytest.approx(5 / 255)
    assert all(value == 0 for value in gy.getdata())
    falling = Image.frombytes("L", (17, 5), bytes(255 - value for value in image.tobytes()))
    assert operator(falling)[0].getpixel((8, 2)) == pytest.approx(-10 / 255)


@pytest.mark.parametrize("mode", ["L", "LA", "RGB", "RGBA", "I;16", "I;16L", "I;16B", "I;16N", "F", "RGB;F", "RGBA;F"])
def test_modes_depth_normalization_and_alpha(mode):
    image = Image.new(mode, (17, 9))
    maximum = 1.0 if mode in ("F", "RGB;F", "RGBA;F") else 65535 if mode.startswith("I;") else 255
    color = maximum if len(image.getbands()) == 1 else (maximum,) * len(image.getbands())
    image.paste(color, (8, 0, 17, 9))
    for operator in [ImageFilter.sobel, ImageFilter.scharr]:
        gx, gy = operator(image)
        assert gx.getpixel((7, 4)) == pytest.approx(0.5)
        assert gy.getpixel((7, 4)) == 0.0
    edges = ImageFilter.canny(image, 0.1, 0.4, sigma=0)
    assert edges.mode == "L"
    assert edges.size == image.size
    assert set(edges.tobytes()) == {0, 255}
    assert edges.getpixel((7, 4)) == 255
    if image.has_transparency_data and not mode.endswith(";F"):
        image.putalpha(0)
        assert ImageFilter.canny(image, sigma=0).tobytes() == edges.tobytes()


@pytest.mark.parametrize("depth", [10, 12, 16])
@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
def test_high_depth_retains_sub_byte_edges(depth, mode):
    channels = len(mode)
    maximum = (1 << depth) - 1
    values = [maximum // 2 + (x >= 8) for _y in range(9) for x in range(17) for _c in range(channels)]
    data = struct.pack(f"<{len(values)}H", *values)
    image = Image.frombytes(mode, (17, 9), data, bit_depth=depth)
    gx, _ = ImageFilter.sobel(image)
    assert gx.getpixel((7, 4)) == pytest.approx(0.5 / maximum, rel=0.01)
    edges = ImageFilter.canny(image, 0, 0, sigma=0)
    assert edges.getpixel((7, 4)) == 255


@pytest.mark.parametrize("vertical", [False, True])
def test_canny_thins_steps_and_zeros_border(vertical):
    image = Image.new("L", (17, 17))
    image.paste(255, (0, 8, 17, 17) if vertical else (8, 0, 17, 17))
    edges = ImageFilter.canny(image, 0.5, 0.5, sigma=0)
    expected = Image.new("L", image.size)
    expected.paste(255, (1, 7, 16, 8) if vertical else (7, 1, 8, 16))
    assert edges.tobytes() == expected.tobytes()
    assert ImageFilter.canny(image, 0.51, 0.51, sigma=0).getbbox() is None
    assert ImageFilter.canny(image, 0.5 + 1e-9, 0.5 + 1e-9, sigma=0).getbbox() is None


@pytest.mark.parametrize("falling", [False, True])
def test_canny_diagonal_edges(falling):
    width = 33
    image = Image.frombytes("L", (width, width), bytes(255 if (x + y >= width if falling else x >= y) else 0 for y in range(width) for x in range(width)))
    edges = ImageFilter.canny(image, 0.1, 0.2, sigma=0)
    assert edges.getbbox() is not None
    for y in range(width):
        for x in range(width):
            if edges.getpixel((x, y)):
                assert abs(x + y - width if falling else x - y) <= 1


def test_float_alpha_is_ignored_even_when_nonfinite():
    image = Image.new("RGBA;F", (17, 9), (0.0, 0.0, 0.0, math.nan))
    image.paste((1.0, 1.0, 1.0, math.inf), (8, 0, 17, 9))
    assert ImageFilter.sobel(image)[0].getpixel((7, 4)) == pytest.approx(0.5)
    assert ImageFilter.canny(image, sigma=0).getpixel((7, 4)) == 255


def test_canny_gaussian_suppresses_step_gradient():
    image = Image.new("L", (33, 33))
    image.paste(255, (16, 0, 33, 33))
    assert ImageFilter.canny(image, 0.3, 0.3, sigma=0).getbbox() is not None
    assert ImageFilter.canny(image, 0.3, 0.3, sigma=2).getbbox() is None
    assert ImageFilter.canny(image, 0.05, 0.1, sigma=2).getbbox() is not None


@pytest.mark.parametrize("size", [(1, 1), (1, 17), (19, 1), (2, 2), (19, 17)])
@pytest.mark.parametrize("sigma", [0, 1, 32])
def test_constant_images_have_no_edges_even_with_zero_thresholds(size, sigma):
    image = Image.new("L", size, 137)
    assert ImageFilter.canny(image, 0, 0, sigma=sigma).getbbox() is None


def test_color_luminance_and_alpha_policy():
    image = Image.new("RGBA", (17, 9), (0, 0, 0, 0))
    image.paste((255, 0, 0, 0), (8, 0, 17, 9))
    gx, _ = ImageFilter.sobel(image)
    assert gx.getpixel((7, 4)) == pytest.approx(0.299 * 0.5)
    assert ImageFilter.canny(image, 0.1, 0.1, sigma=0).getbbox() is not None
    assert ImageFilter.canny(image, 0.2, 0.2, sigma=0).getbbox() is None


@pytest.mark.parametrize("lazy", [False, True])
def test_current_frame_metadata_and_output_lifetime(lazy):
    first = Image.new("L", (17, 9))
    second = first.copy()
    second.paste(255, (8, 0, 17, 9))
    stream = BytesIO()
    first.save(stream, "PNG", save_all=True, append_images=[second], duration=[20, 40])
    stream.seek(0)
    with Image.open(stream, lazy=lazy) as image:
        image.seek(1)
        image.info["test"] = "metadata"
        before = image.tobytes()
        gx, gy = ImageFilter.scharr(image)
        edges = ImageFilter.canny(image, sigma=0)
        for output in [gx, gy, edges]:
            assert output.info["test"] == "metadata"
            assert output.n_frames == 1
        assert image.tell() == 1
        assert image.tobytes() == before
        image.seek(0)
        assert ImageFilter.canny(image, sigma=0).getbbox() is None
    assert gx.getpixel((7, 4)) == pytest.approx(0.5)
    assert edges.getpixel((7, 4)) == 255


@pytest.mark.parametrize("operator", [ImageFilter.sobel, ImageFilter.scharr, ImageFilter.canny])
def test_invalid_images(operator):
    with pytest.raises(TypeError, match="Blanket Image"):
        operator(None)
    with pytest.raises(ValueError, match="nonempty"):
        operator(Image.new("L", (0, 5)))
    for mode in ["P", "PA", "1", "I", "CMYK", "HSV", "La", "RGBa"]:
        with pytest.raises(ValueError, match="convert explicitly"):
            operator(Image.new(mode, (5, 5)))
    for value in [math.nan, math.inf, -math.inf, -0.1, 1.1]:
        with pytest.raises(ValueError, match="finite and in"):
            operator(Image.new("F", (5, 5), value))


@pytest.mark.parametrize("low,high", [(-0.1, 0.2), (0.3, 0.2), (0.1, 1.1), (math.nan, 0.2), (0.1, math.inf), (0, -1), (0, 1.000000001)])
def test_invalid_thresholds(low, high):
    with pytest.raises(ValueError, match="thresholds"):
        ImageFilter.canny(Image.new("L", (5, 5)), low, high)


@pytest.mark.parametrize("sigma", [-1, 33, math.nan, math.inf])
def test_invalid_sigma(sigma):
    with pytest.raises(ValueError, match="sigma"):
        ImageFilter.canny(Image.new("L", (5, 5)), sigma=sigma)
