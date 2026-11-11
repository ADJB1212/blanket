import math
import struct

import pytest
from gpu_helpers import CHANNELS, MODES, require_metal, shader_runtime
from helpers import assert_raises, float_samples

from blanket import Image, ImageEnhance, ImageFilter, ImageGpu

runtime = shader_runtime


def source(mode, size=(17, 11), seed=0):
    count = size[0] * size[1] * CHANNELS[mode]
    samples = [(i * 37 + i // 11 + seed) % 256 for i in range(count)]
    data = struct.pack(f"{count}f", *[(v - 40) / 180 for v in samples]) if "F" in mode else bytes(samples)
    image = Image.frombytes(mode, size, data)
    image.info["description"] = "gpu color"
    return image


@pytest.mark.parametrize("mode", ["RGB", "RGBA", "RGB;F", "RGBA;F"])
@pytest.mark.parametrize("channels", [3, 4])
def test_native_color_lut_keeps_controls_in_rust(mode, channels, monkeypatch):
    require_metal()
    gpu = ImageGpu.from_image(source(mode), unified_memory=True)
    output_mode = ("RGBA;F" if "F" in mode else "RGBA") if channels == 4 else mode
    lut = ImageFilter.Color3DLUT.generate((3, 4, 5), lambda r, g, b: (r * 0.8, g * 0.7, b * 0.9, 0.25)[:channels], channels=channels, target_mode=output_mode)
    controls = ImageGpu._blanket._gpu_lut_controls(lut.size, channels, lut.table, "F" in mode)
    expected = gpu._run(25, mode=output_mode, k=channels, axis=ImageGpu._FORMATS.index(output_mode), values=controls).to_image().tobytes()

    def unexpected(*args, **kwargs):
        raise AssertionError("native LUT controls must not pass back through Python")

    with monkeypatch.context() as patch:
        patch.setattr(ImageGpu._blanket, "_gpu_lut_controls", unexpected)
        actual = gpu.color_lut(lut)
    assert actual.info == gpu.info and actual.unified_memory
    gpu.close()
    assert actual.to_image().tobytes() == expected


def assert_images_equal(actual, expected):
    assert actual.mode == expected.mode and actual.size == expected.size
    if "F" in expected.mode:
        assert float_samples(actual) == pytest.approx(float_samples(expected), rel=1e-5, abs=2e-6, nan_ok=True)
    else:
        assert max(abs(a - b) for a, b in zip(actual.tobytes(), expected.tobytes(), strict=True)) <= 1


@pytest.mark.parametrize("mode,factor", [(mode, factor) for mode in MODES[:4] for factor in (0.0, 1.0, 1.75)] + [("RGB", factor) for factor in (-1.0, 0.3, 3.0)])
@pytest.mark.parametrize(("method", "enhancer"), [("color", ImageEnhance.Color), ("contrast", ImageEnhance.Contrast), ("sharpness", ImageEnhance.Sharpness)])
def test_enhancement_cpu_parity(runtime, mode, method, enhancer, factor):
    image = source(mode)
    actual = getattr(ImageGpu.from_image(image, runtime=runtime), method)(factor).to_image()
    assert_images_equal(actual, enhancer(image).enhance(factor))
    assert actual.info["description"] == image.info["description"]


@pytest.mark.parametrize("mode,size", [(mode, (257, 257)) for mode in MODES[:4]] + [("RGB", size) for size in ((1, 1), (257, 1), (17, 16))])
def test_contrast_reduction_boundaries(runtime, mode, size):
    image = source(mode, size)
    gpu = ImageGpu.from_image(image, runtime=runtime)
    assert_images_equal(gpu.contrast(0).to_image(), ImageEnhance.Contrast(image).enhance(0))
    assert_images_equal(gpu.contrast(1.75).to_image(), ImageEnhance.Contrast(image).enhance(1.75))


def test_contrast_sum_exceeds_uint32(runtime):
    gpu = ImageGpu.from_image(Image.new("L", (4105, 4105), 255), runtime=runtime)
    result = gpu.contrast(0).crop((0, 0, 1, 1))
    gpu.close()
    assert result.to_image().getpixel((0, 0)) == 255


@pytest.mark.parametrize("mode", MODES[4:])
def test_float_enhancements(runtime, mode):
    channels = CHANNELS[mode]
    pixels = ([10.0], [30.0]) if mode == "F" else ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    if channels == 4:
        pixels = (pixels[0] + [0.25], pixels[1] + [0.75])
    image = Image.frombytes(mode, (2, 1), struct.pack(f"{channels * 2}f", *pixels[0], *pixels[1]))
    gpu = ImageGpu.from_image(image, runtime=runtime)
    expected = [0.0, 40.0] if mode == "F" else [-0.5, -0.5, -0.5, 1.5, 1.5, 1.5]
    if channels == 4:
        expected = [*expected[:3], 0.25, *expected[3:], 0.75]
    assert float_samples(gpu.contrast(2).to_image()) == pytest.approx(expected)
    assert gpu.sharpness(0).to_image().tobytes() == image.tobytes()
    if channels >= 3:
        red = Image.frombytes(mode, (1, 1), struct.pack(f"{channels}f", *([1, 0, 0] + ([-0.25] if channels == 4 else []))))
        expected = [0.299] * 3 + ([-0.25] if channels == 4 else [])
        assert float_samples(ImageGpu.from_image(red, runtime=runtime).color(0).to_image()) == pytest.approx(expected)
    impulse = [0.0] * (9 * channels)
    for channel in range(channels):
        impulse[4 * channels + channel] = 1.0
    image = Image.frombytes(mode, (3, 3), struct.pack(f"{len(impulse)}f", *impulse))
    actual = float_samples(ImageGpu.from_image(image, runtime=runtime).sharpness(2).to_image())
    expected = impulse.copy()
    for channel in range(3 if channels == 4 else channels):
        expected[4 * channels + channel] = 21 / 13
    assert actual == pytest.approx(expected, rel=1e-5, abs=1e-6)


@pytest.mark.parametrize(("mode", "table_channels", "target"), [("RGB", 3, None), ("RGBA", 3, None), ("RGB", 4, "RGBA"), ("RGBA", 4, "RGBA"), ("RGBA", 3, "RGB")])
@pytest.mark.parametrize("size", [2, (3, 5, 4), 17])
def test_color_lut_cpu_parity(runtime, mode, table_channels, target, size):
    def callback(r, g, b):
        color = (-0.2 + r * 1.3, g * 0.7, 1 - b)
        return (*color, r * 0.25 + b * 0.75) if table_channels == 4 else color

    lut = ImageFilter.Color3DLUT.generate(size, callback, channels=table_channels, target_mode=target)
    image = source(mode)
    actual = ImageGpu.from_image(image, runtime=runtime).color_lut(lut).to_image()
    assert actual.tobytes() == image.filter(lut).tobytes()


def test_largest_lut(runtime):
    lut = ImageFilter.Color3DLUT.generate(65, lambda r, g, b: (r, g, b))
    image = Image.new("RGB", (1, 1), (255, 128, 0))
    assert ImageGpu.from_image(image, runtime=runtime).color_lut(lut).to_image().tobytes() == image.filter(lut).tobytes()


@pytest.mark.parametrize("mode", ["RGB;F", "RGBA;F"])
def test_float_color_lut(runtime, mode):
    image = source(mode)
    gpu = ImageGpu.from_image(image, runtime=runtime)
    lut = ImageFilter.Color3DLUT.generate((3, 4, 5), lambda r, g, b: (r * 0.5 - 0.25, g * 2, 1 - b))
    samples = float_samples(image)
    expected = []
    channels = CHANNELS[mode]
    for i in range(0, len(samples), channels):
        r, g, b = (max(0, min(1, v)) for v in samples[i : i + 3])
        expected.extend((r * 0.5 - 0.25, g * 2, 1 - b))
        if channels == 4:
            expected.append(samples[i + 3])
    assert float_samples(gpu.color_lut(lut).to_image()) == pytest.approx(expected, rel=1e-5, abs=1e-6)
    rgba = ImageFilter.Color3DLUT.generate(2, lambda r, g, b: (r, g, b, 1.5), channels=4, target_mode="RGBA;F")
    actual = gpu.color_lut(rgba)
    assert actual.mode == "RGBA;F"
    assert float_samples(actual.getchannel("A").to_image()) == (1.5,) * (image.width * image.height)


@pytest.mark.parametrize("mode", ["RGB", "RGBA", "RGB;F", "RGBA;F"])
@pytest.mark.parametrize("method,transfer", [(method, "srgb") for method in range(6)] + [(Image.Resampling.BILINEAR, "linear")])
def test_linear_resize_cpu_parity(runtime, mode, method, transfer):
    image = source(mode)
    actual = ImageGpu.from_image(image, runtime=runtime).resize((11, 19), method, linear_light=True, transfer=transfer).to_image()
    expected = image.resize((11, 19), method, linear_light=True, transfer=transfer)
    assert_images_equal(actual, expected)


@pytest.mark.parametrize("linear", [False, True])
def test_resize_without_alpha_premultiplication(runtime, linear):
    image = source("RGBA;F")
    actual = ImageGpu.from_image(image, runtime=runtime).resize((11, 19), linear_light=linear, premultiply_alpha=False).to_image()
    expected = image.resize((11, 19), linear_light=linear, premultiply_alpha=False)
    assert_images_equal(actual, expected)


@pytest.mark.parametrize("mode", ["RGB", "RGBA", "RGB;F", "RGBA;F"])
@pytest.mark.parametrize("method", [Image.Resampling.BILINEAR, Image.Resampling.BICUBIC])
def test_resize_vertical_first_cpu_parity(runtime, mode, method):
    image = source(mode, size=(17, 31))
    actual = ImageGpu.from_image(image, runtime=runtime).resize((13, 9), method).to_image()
    expected = image.resize((13, 9), method)
    assert_images_equal(actual, expected)


@pytest.mark.parametrize("mode", ["RGBA", "RGBA;F"])
@pytest.mark.parametrize("transfer", ["srgb", "linear"])
def test_linear_composite_cpu_parity(runtime, mode, transfer):
    image, foreground = source(mode), source(mode, seed=83)
    gpu, other = (ImageGpu.from_image(im, runtime=runtime) for im in (image, foreground))
    actual = gpu.alpha_composite(other, linear_light=True, transfer=transfer).to_image()
    expected = Image.alpha_composite(image, foreground, linear_light=True, transfer=transfer)
    assert_images_equal(actual, expected)
    assert actual.info["description"] == image.info["description"]


def test_linear_processing_differs_from_encoded_processing(runtime):
    image = Image.frombytes("RGB", (2, 1), bytes((0, 0, 0, 255, 255, 255)))
    gpu = ImageGpu.from_image(image, runtime=runtime)
    assert gpu.resize((1, 1), Image.Resampling.BOX).to_image().getpixel((0, 0)) == (128, 128, 128)
    assert gpu.resize((1, 1), Image.Resampling.BOX, linear_light=True).to_image().getpixel((0, 0)) == (188, 188, 188)
    background = ImageGpu.from_image(Image.new("RGBA", (1, 1), (0, 0, 0, 255)), runtime=runtime)
    foreground = ImageGpu.from_image(Image.new("RGBA", (1, 1), (255, 255, 255, 128)), runtime=runtime)
    assert background.alpha_composite(foreground, linear_light=True).to_image().getpixel((0, 0)) == (188, 188, 188, 255)


def test_nonfinite_enhancements_and_transfer(runtime):
    image = Image.frombytes("RGBA;F", (2, 1), struct.pack("8f", float("inf"), -float("inf"), float("nan"), 0.0, -0.5, 2.0, 0.25, float("nan")))
    gpu = ImageGpu.from_image(image, runtime=runtime)
    for method in ("color", "contrast", "sharpness"):
        assert getattr(gpu, method)(1).to_image().tobytes() == image.tobytes()
    actual = gpu.resize((3, 1), linear_light=True).to_image()
    assert float_samples(actual) == (0.0,) * 12
    mean = float_samples(gpu.contrast(0).to_image())
    assert math.isnan(mean[0]) and math.isnan(mean[4])
    identity = ImageFilter.Color3DLUT.generate(2, lambda r, g, b: (r, g, b))
    output = float_samples(gpu.color_lut(identity).to_image())
    assert output[:4] == (1.0, 0.0, 0.0, 0.0)
    assert math.isnan(output[7])


def test_validation(runtime):
    gpu = ImageGpu.from_image(source("RGB"), runtime=runtime)
    for method in ("color", "contrast", "sharpness"):
        for invalid in (float("nan"), float("inf"), 1e40):
            assert_raises(ValueError, lambda: getattr(gpu, method)(invalid))
    assert_raises(ValueError, lambda: gpu.resize((1, 1), transfer="bad"))
    assert_raises(ValueError, lambda: ImageGpu.from_image(source("L"), runtime=runtime).resize((1, 1), linear_light=True))
    lut = ImageFilter.Color3DLUT.generate(2, lambda r, g, b: (r, g, b))
    lut.table[0] = float("nan")
    assert_raises(ValueError, lambda: gpu.color_lut(lut))
    lut.table[0] = 0
    lut.size = (1, 2, 2)
    assert_raises(ValueError, lambda: gpu.color_lut(lut))
    lut.size = (2, 2, 2)
    lut.mode = "RGB;F"
    assert_raises(ValueError, lambda: gpu.color_lut(lut))
    gpu.close()
    assert_raises(ValueError, lambda: gpu.contrast(0), match="closed")


def test_resident_mps_color_pipeline(monkeypatch):
    require_metal()
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("MPS required")
    tensor = torch.arange(4 * 31 * 29, dtype=torch.float32, device="mps").reshape(4, 31, 29) / 3595
    original = tensor.clone()
    gpu = ImageGpu.from_torch(tensor[:, 1:30, 2:28], layout="CHW")
    lut = ImageFilter.Color3DLUT.generate(3, lambda r, g, b: (b, r, g))

    def no_staging(*args, **kwargs):
        raise AssertionError("unexpected CPU transfer")

    with monkeypatch.context() as context:
        for name in ("cpu", "numpy", "tolist", "item", "__float__"):
            context.setattr(torch.Tensor, name, no_staging)
        result = gpu.color(0.75).contrast(1.2).sharpness(1.4).color_lut(lut)
        result = result.resize((17, 19), linear_light=True)
        result = result.alpha_composite(result, linear_light=True)
        gpu.close()
        result.synchronize()
        assert result.to_torch().device.type == "mps"
    assert torch.equal(tensor, original)
