import math
import struct

import pytest

from blanket import Image, ImageGpu


@pytest.fixture(params=["native", "torch"])
def runtime(request):
    if not ImageGpu.capabilities()["available"]:
        pytest.skip("Metal required")
    if request.param == "torch":
        torch = pytest.importorskip("torch")
        if not torch.backends.mps.is_available():
            pytest.skip("MPS required")
    return request.param


@pytest.mark.parametrize("mode", ["L", "LA", "RGB", "RGBA", "F", "RGB;F", "RGBA;F"])
@pytest.mark.parametrize("resample", [0, 2, 3])
def test_identity_transform(runtime, mode, resample):
    image = Image.new(mode, (9, 7))
    gpu = ImageGpu.from_image(image, runtime=runtime)
    assert gpu.transform(image.size, Image.Transform.AFFINE, (1, 0, 0, 0, 1, 0), resample).to_image().tobytes() == image.tobytes()


@pytest.mark.parametrize("resample", [0, 2, 3])
@pytest.mark.parametrize("method, coefficients", [(Image.Transform.AFFINE, (1, 0.1, 1, -0.1, 1, 0)), (Image.Transform.PERSPECTIVE, (1, 0, 0, 0, 1, 0, 0.01, 0.02))])
def test_transform_cpu_parity(runtime, resample, method, coefficients):
    image = Image.frombytes("L", (9, 7), bytes((i * 37) % 256 for i in range(63)))
    gpu = ImageGpu.from_image(image, runtime=runtime)
    actual = gpu.transform((11, 8), method, coefficients, resample, fillcolor=71).to_image()
    expected = image.transform((11, 8), method, coefficients, resample, fillcolor=71)
    assert max(abs(a - b) for a, b in zip(actual.tobytes(), expected.tobytes(), strict=True)) <= 2


def test_reusable_map_and_lifetime(runtime):
    image = Image.frombytes("L", (3, 2), bytes(range(6)))
    coordinates = Image.frombytes("RGB;F", (2, 2), struct.pack("12f", 2.5, 0.5, 0, 0.5, 1.5, 0, math.nan, 0, 0, 1.5, 1.5, 0))
    source = ImageGpu.from_image(image, runtime=runtime)
    mapping = ImageGpu.from_image(coordinates, runtime=runtime)
    first = source.remap(mapping, fillcolor=77)
    second = source.invert().remap(mapping, fillcolor=77)
    mapping.close()
    source.close()
    assert first.to_image().tobytes() == bytes((2, 3, 77, 4))
    assert second.to_image().tobytes() == bytes((253, 252, 77, 251))


@pytest.mark.parametrize("angle", [0, 45, 90, 180, 270])
def test_rotation(runtime, angle):
    image = Image.frombytes("L", (9, 7), bytes(range(63)))
    actual = ImageGpu.from_image(image, runtime=runtime).rotate(angle, expand=True).to_image()
    expected = image.rotate(angle, expand=True)
    assert actual.size == expected.size
    assert actual.tobytes() == expected.tobytes()


def test_geometry_validation(runtime):
    source = ImageGpu.from_image(Image.new("L", (2, 2)), runtime=runtime)
    for coefficients in [(1, 0), (1, 0, math.inf, 0, 1, 0)]:
        with pytest.raises(ValueError):
            source.transform((2, 2), Image.Transform.AFFINE, coefficients)
    with pytest.raises(ValueError):
        source.remap(source)
    with pytest.raises(ValueError):
        source.rotate(math.nan)


@pytest.mark.parametrize("mode", ["L", "LA", "RGB", "RGBA"])
def test_masked_paste(runtime, mode):
    background = Image.new(mode, (7, 5), 12 if mode == "L" else tuple([12] * len(Image.new(mode, (1, 1)).getbands())))
    source = Image.new(mode, (3, 4), 180 if mode == "L" else tuple([180] * len(background.getbands())))
    mask = Image.frombytes("L", source.size, bytes((i * 23) % 256 for i in range(12)))
    expected = background.copy()
    expected.paste(source, (-1, 2), mask)
    result = ImageGpu.from_image(background, runtime=runtime).paste(ImageGpu.from_image(source, runtime=runtime), (-1, 2), ImageGpu.from_image(mask, runtime=runtime))
    assert max(abs(a - b) for a, b in zip(result.to_image().tobytes(), expected.tobytes(), strict=True)) <= 1


def test_resident_reductions(runtime):
    image = Image.frombytes("RGB", (9, 7), bytes((i * 13) % 256 for i in range(189)))
    gpu = ImageGpu.from_image(image, runtime=runtime)
    statistics = gpu.statistics()
    histogram = gpu.histogram()
    gpu.close()
    assert [value for band in histogram.to_cpu() for value in band] == image.histogram()
    for channel, actual in enumerate(statistics.to_cpu()):
        values = list(image.tobytes()[channel::3])
        assert actual["count"] == len(values)
        assert actual["extrema"] == (min(values), max(values))
        assert actual["sum"] == sum(values)
        assert actual["mean"] == pytest.approx(sum(values) / len(values))
    statistics.close()
    with pytest.raises(ValueError, match="closed"):
        statistics.to_cpu()


def test_geometry_and_reductions_remain_resident(monkeypatch):
    if not ImageGpu.capabilities()["available"]:
        pytest.skip("Metal required")
    torch = pytest.importorskip("torch")
    source = ImageGpu.from_torch(torch.ones((3, 5, 3), device="mps"), layout="HWC")
    mask = ImageGpu.from_torch(torch.ones((3, 5), device="mps"), layout="HW")

    def no_staging(*args, **kwargs):
        raise AssertionError("unexpected CPU pixel staging")

    with monkeypatch.context() as context:
        for name in ("cpu", "numpy", "tolist", "item"):
            context.setattr(torch.Tensor, name, no_staging)
        result = source.rotate(30).paste(source, mask=mask)
        reduction = result.statistics()
        result.synchronize()
        reduction.synchronize()
    assert reduction.to_cpu()[0]["mean"] == pytest.approx(1.0)


@pytest.mark.parametrize("mode,channels", [("F", 1), ("RGB;F", 3), ("RGBA;F", 4)])
def test_float_geometry_samples_and_statistics(runtime, mode, channels):
    samples = [float(i - 4) / 3 for i in range(6 * channels)]
    image = Image.frombytes(mode, (3, 2), struct.pack(f"{len(samples)}f", *samples))
    gpu = ImageGpu.from_image(image, runtime=runtime)
    actual = gpu.transform((3, 2), Image.Transform.AFFINE, (1, 0, 0, 0, 1, 0)).to_image()
    assert actual.tobytes() == image.tobytes()
    for band, values in enumerate(gpu.statistics().to_cpu()):
        channel = samples[band::channels]
        assert values["extrema"] == pytest.approx((min(channel), max(channel)))
        assert values["moments"] == pytest.approx([sum(value**order for value in channel) / 6 for order in range(1, 5)], rel=1e-5, abs=1e-6)
    singular = gpu.transform((1, 1), Image.Transform.PERSPECTIVE, (1, 0, 0, 0, 1, 0, -2, 0), fillcolor=-3)
    assert struct.unpack(f"{channels}f", singular.to_image().tobytes()) == (-3.0,) * channels


def test_nonfinite_statistics_and_mask_edges(runtime):
    image = Image.frombytes("F", (2, 1), struct.pack("2f", math.nan, math.inf))
    gpu = ImageGpu.from_image(image, runtime=runtime)
    values = gpu.statistics().to_cpu()[0]
    assert math.isnan(values["mean"])
    assert math.isnan(values["variance"])
    assert all(math.isnan(value) for value in values["extrema"])
    background = ImageGpu.from_image(Image.new("F", (2, 1), 7.0), runtime=runtime)
    mask = ImageGpu.from_image(Image.frombytes("F", (2, 1), struct.pack("2f", 0.0, 1.0)), runtime=runtime)
    actual = struct.unpack("2f", background.paste(gpu, mask=mask).to_image().tobytes())
    assert actual == (7.0, math.inf)
