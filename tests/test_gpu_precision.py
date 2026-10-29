import math
import struct

import pytest

from blanket import Image, ImageFilter, ImageGpu

MODES = ("L", "LA", "RGB", "RGBA", "F", "RGB;F", "RGBA;F")
CHANNELS = dict(zip(MODES, (1, 2, 3, 4, 1, 3, 4), strict=True))
CPU_CONVERSIONS = sorted({(a, b) for a in MODES[:5] for b in MODES[:5]} | {(a, b) for a in ("RGB", "RGBA", "RGB;F", "RGBA;F") for b in ("RGB", "RGBA", "RGB;F", "RGBA;F")})


@pytest.fixture(params=("native", "torch"))
def runtime(request):
    if not ImageGpu.capabilities()["available"]:
        pytest.skip("Metal feature and device required")
    if request.param == "torch":
        torch = pytest.importorskip("torch")
        if not torch.backends.mps.is_available() or not hasattr(torch.mps, "compile_shader"):
            pytest.skip("PyTorch MPS shader support required")
    return request.param


def source(mode, size=(17, 11)):
    count = size[0] * size[1] * CHANNELS[mode]
    samples = [(i * 37 + i // 11) % 256 for i in range(count)]
    if "F" in mode:
        samples = [(v - 80) / 127 for v in samples]
        data = struct.pack(f"{count}f", *samples)
    else:
        data = bytes(samples)
    image = Image.frombytes(mode, size, data)
    image.info["description"] = "gpu precision"
    return image


def floats(image):
    data = image.tobytes()
    return struct.unpack(f"{len(data) // 4}f", data)


@pytest.mark.parametrize(("src", "dst"), CPU_CONVERSIONS)
def test_mode_conversion_cpu_parity(runtime, src, dst):
    image = source(src)
    actual = ImageGpu.from_image(image, runtime=runtime).convert(dst).to_image()
    expected = image.convert(dst)
    assert actual.mode == dst
    assert actual.size == image.size
    assert actual.info["description"] == "gpu precision"
    if "F" in dst:
        assert floats(actual) == pytest.approx(floats(expected), rel=1e-5, abs=1e-6)
    else:
        assert max(abs(a - b) for a, b in zip(actual.tobytes(), expected.tobytes(), strict=True)) <= 1


def test_extended_float_grayscale_conversion(runtime):
    gray = ImageGpu.from_image(Image.new("L", (1, 1), 64), runtime=runtime)
    assert floats(gray.convert("RGBA;F").to_image()) == pytest.approx((64 / 255, 64 / 255, 64 / 255, 1))
    raw = ImageGpu.from_image(Image.new("F", (1, 1), 128.5), runtime=runtime)
    assert floats(raw.convert("RGB;F").to_image()) == pytest.approx((128.5 / 255,) * 3)
    assert raw.convert("RGB").to_image().getpixel((0, 0)) == (128, 128, 128)
    color = Image.frombytes("RGBA;F", (1, 1), struct.pack("4f", 1, 0.5, 0, 0.25))
    gpu = ImageGpu.from_image(color, runtime=runtime)
    assert floats(gpu.grayscale().to_image()) == pytest.approx((151.0875,))
    assert gpu.convert("LA").to_image().getpixel((0, 0)) == (151, 64)


@pytest.mark.parametrize("mode", MODES)
def test_channels_merge_and_alpha(runtime, mode):
    image = source(mode)
    gpu = ImageGpu.from_image(image, runtime=runtime)
    bands = gpu.split()
    for channel, (band, name) in enumerate(zip(bands, image.getbands(), strict=True)):
        expected = image.getchannel(channel)
        assert band.to_image().tobytes() == expected.tobytes()
        assert gpu.getchannel(name).to_image().tobytes() == expected.tobytes()
    restored = ImageGpu.merge(mode, bands)
    assert restored.to_image().tobytes() == image.tobytes()
    new_alpha = ImageGpu.from_image(Image.new("F" if "F" in mode else "L", image.size, 0.25 if "F" in mode else 91), runtime=runtime)
    result = gpu.putalpha(new_alpha)
    assert result.getchannel("A").to_image().tobytes() == new_alpha.to_image().tobytes()
    assert gpu.putalpha(0.5 if "F" in mode else 83).getchannel("A").to_image().tobytes() == Image.new(new_alpha.mode, image.size, 0.5 if "F" in mode else 83).tobytes()
    if "F" not in mode:
        if mode in ("L", "LA"):
            expected = Image.merge("LA", (image.getchannel(0), Image.new("L", image.size, 83)))
        else:
            expected = image.copy()
            expected.putalpha(83)
        assert gpu.putalpha(83).to_image().tobytes() == expected.tobytes()
        if mode in ("L", "LA"):
            expected = Image.merge("LA", (image.getchannel(0), new_alpha.to_image()))
        else:
            expected.putalpha(new_alpha.to_image())
        assert result.to_image().tobytes() == expected.tobytes()
        assert gpu.putalpha(10**1000).getchannel("A").to_image().tobytes() == Image.new("L", image.size, 255).tobytes()
    gpu.close()
    assert ImageGpu.merge(mode, bands).to_image().tobytes() == image.tobytes()
    for band in bands:
        band.close()
    assert restored.to_image().tobytes() == image.tobytes()


@pytest.mark.parametrize("mode", MODES[:4])
@pytest.mark.parametrize("gaussian", [False, True])
@pytest.mark.parametrize("radius", [0, 0.25, 1, 1.5, 2.2, 25, (0, 1.5), (2, 0.5)])
def test_blur_cpu_parity(runtime, mode, gaussian, radius):
    image = source(mode)
    gpu = ImageGpu.from_image(image, runtime=runtime)
    actual = (gpu.gaussian_blur(radius) if gaussian else gpu.box_blur(radius)).to_image()
    expected = image.filter(ImageFilter.GaussianBlur(radius) if gaussian else ImageFilter.BoxBlur(radius))
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("size", [(1, 1), (1, 7), (9, 1)])
def test_tiny_blur_and_large_radius(runtime, size):
    image = source("RGBA", size)
    gpu = ImageGpu.from_image(image, runtime=runtime)
    for radius in (0.5, 5, (20, 0.5)):
        assert gpu.box_blur(radius).to_image().tobytes() == image.filter(ImageFilter.BoxBlur(radius)).tobytes()
        assert gpu.gaussian_blur(radius).to_image().tobytes() == image.filter(ImageFilter.GaussianBlur(radius)).tobytes()


@pytest.mark.parametrize("mode", MODES[:4])
@pytest.mark.parametrize("size", [1, 3, 5, 7])
@pytest.mark.parametrize(("method", "filter_class"), [("min_filter", ImageFilter.MinFilter), ("max_filter", ImageFilter.MaxFilter), ("median_filter", ImageFilter.MedianFilter)])
def test_rank_cpu_parity(runtime, mode, size, method, filter_class):
    image = source(mode, (9, 7))
    actual = getattr(ImageGpu.from_image(image, runtime=runtime), method)(size).to_image()
    assert actual.tobytes() == image.filter(filter_class(size)).tobytes()


@pytest.mark.parametrize("mode", MODES[:4])
@pytest.mark.parametrize(("radius", "percent", "threshold"), [(0, 150, 3), (0.5, 150, 0), (2, 0, 3), (2, 150, 3), (1.5, -75, 3), (3, 500, 255)])
def test_unsharp_cpu_parity(runtime, mode, radius, percent, threshold):
    image = source(mode)
    actual = ImageGpu.from_image(image, runtime=runtime).unsharp_mask(radius, percent, threshold).to_image()
    expected = image.filter(ImageFilter.UnsharpMask(radius, percent, threshold))
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("mode", MODES[4:])
def test_float_filters(runtime, mode):
    channels = CHANNELS[mode]
    samples = [0.0] * channels + [1.0] * channels + [0.0] * channels
    image = Image.frombytes(mode, (3, 1), struct.pack(f"{len(samples)}f", *samples))
    gpu = ImageGpu.from_image(image, runtime=runtime)
    assert floats(gpu.box_blur((1, 0)).to_image()) == pytest.approx([1 / 3] * len(samples), rel=1e-5, abs=1e-6)
    assert floats(gpu.gaussian_blur(1).to_image()) == pytest.approx([7 / 24] * channels + [5 / 12] * channels + [7 / 24] * channels, rel=1e-5, abs=1e-6)
    assert floats(gpu.unsharp_mask(1, 150, 0.01).to_image()) == pytest.approx([-7 / 16] * channels + [15 / 8] * channels + [-7 / 16] * channels, rel=1e-5, abs=1e-6)
    assert floats(gpu.min_filter().to_image()) == (0.0,) * len(samples)
    assert floats(gpu.max_filter().to_image()) == (1.0,) * len(samples)
    assert floats(gpu.median_filter().to_image()) == (0.0,) * len(samples)


def test_nonfinite_samples(runtime):
    image = Image.frombytes("F", (3, 1), struct.pack("3f", float("nan"), float("inf"), -float("inf")))
    gpu = ImageGpu.from_image(image, runtime=runtime)
    assert gpu.getchannel(0).to_image().tobytes() == image.tobytes()
    assert gpu.convert("L").to_image().tobytes() == bytes((0, 255, 0))
    for method in ("min_filter", "max_filter", "median_filter"):
        actual = floats(getattr(gpu, method)().to_image())
        assert math.isnan(actual[0]) and math.isnan(actual[1])
    assert gpu.box_blur(0).to_image().tobytes() == image.tobytes()
    assert gpu.unsharp_mask(percent=0).to_image().tobytes() == image.tobytes()


@pytest.mark.parametrize("mode", MODES[:4])
def test_rank_window_larger_than_image(runtime, mode):
    image = source(mode, (2, 1))
    gpu = ImageGpu.from_image(image, runtime=runtime)
    for rank in (0, 24, 48):
        assert gpu.rank_filter(7, rank).to_image().tobytes() == image.filter(ImageFilter.RankFilter(7, rank)).tobytes()


@pytest.mark.parametrize("mode", ["L", "F"])
@pytest.mark.parametrize("window", [3, 7])
def test_rank_filter_every_rank(runtime, mode, window):
    width, height = 5, 3
    samples = [float((i * 7) % 11 - 5) for i in range(width * height)]
    if mode == "L":
        samples = [int(value + 5) for value in samples]
    else:
        samples[0], samples[1], samples[2] = -math.inf, math.inf, -0.0
    data = bytes(samples) if mode == "L" else struct.pack(f"{len(samples)}f", *samples)
    gpu = ImageGpu.from_image(Image.frombytes(mode, (width, height), data), runtime=runtime)
    radius = window // 2
    neighborhoods = [
        sorted(samples[min(max(y + dy, 0), height - 1) * width + min(max(x + dx, 0), width - 1)] for dy in range(-radius, radius + 1) for dx in range(-radius, radius + 1))
        for y in range(height)
        for x in range(width)
    ]
    for rank in range(window * window):
        actual = gpu.rank_filter(window, rank).to_image()
        expected = [values[rank] for values in neighborhoods]
        assert (list(actual.tobytes()) if mode == "L" else list(floats(actual))) == expected


def test_validation(runtime):
    gpu = ImageGpu.from_image(source("RGB"), runtime=runtime)
    for mode in ("P", "I;16", "CMYK"):
        with pytest.raises(ValueError):
            gpu.convert(mode)
    for channel in (-1, 3, "A"):
        with pytest.raises(ValueError):
            gpu.getchannel(channel)
    for radius in (-1, float("nan"), float("inf"), 1e30, (1, -1)):
        with pytest.raises(ValueError):
            gpu.box_blur(radius)
        with pytest.raises(ValueError):
            gpu.gaussian_blur(radius)
    for size in (0, 2, 9):
        with pytest.raises(ValueError):
            gpu.median_filter(size)
    with pytest.raises(ValueError):
        gpu.rank_filter(3, 9)
    with pytest.raises(ValueError):
        gpu.unsharp_mask(percent=2**40)
    with pytest.raises(ValueError):
        ImageGpu.merge("RGB", gpu.split()[:2])
    with pytest.raises(ValueError):
        ImageGpu.merge("RGB;F", gpu.split())
    with pytest.raises(ValueError):
        gpu.putalpha(ImageGpu.from_image(source("RGB"), runtime=runtime))
    with pytest.raises(ValueError):
        gpu.putalpha(ImageGpu.from_image(source("F"), runtime=runtime))
    with pytest.raises(ValueError):
        gpu.putalpha(ImageGpu.from_image(source("L", (1, 1)), runtime=runtime))
    gpu.close()
    with pytest.raises(ValueError, match="closed"):
        gpu.box_blur()


def test_new_operations_remain_resident(monkeypatch):
    if not ImageGpu.capabilities()["available"]:
        pytest.skip("Metal feature required")
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("MPS required")
    tensor = torch.arange(4 * 11 * 17, dtype=torch.float32, device="mps").reshape(4, 11, 17) / 747
    original = tensor.clone()
    gpu = ImageGpu.from_torch(tensor[:, 1:10, 2:16], layout="CHW")

    def no_staging(*args, **kwargs):
        raise AssertionError("unexpected CPU pixel staging")

    with monkeypatch.context() as context:
        for name in ("cpu", "numpy", "tolist"):
            context.setattr(torch.Tensor, name, no_staging)
        result = ImageGpu.merge("RGBA;F", gpu.split()).convert("RGBA").grayscale().putalpha(123)
        result = result.box_blur(1.5).gaussian_blur((0.5, 1)).unsharp_mask().median_filter().max_filter().min_filter()
        output = result.to_torch()
        result.synchronize()
        assert tuple(output.shape) == (9, 14, 2)
        assert output.device.type == "mps"
    assert torch.equal(tensor, original)
