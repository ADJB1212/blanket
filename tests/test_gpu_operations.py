import math
import struct

import pytest

from blanket import Image, ImageChops, ImageEnhance, ImageGpu

MODES = ("L", "LA", "RGB", "RGBA", "F", "RGB;F", "RGBA;F")
CHANNELS = dict(zip(MODES, (1, 2, 3, 4, 1, 3, 4), strict=True))


@pytest.fixture(params=("native", "torch"))
def runtime(request):
    if not ImageGpu.capabilities()["available"]:
        pytest.skip("Metal feature and device required")
    if request.param == "torch":
        torch = pytest.importorskip("torch")
        if not torch.backends.mps.is_available() or not hasattr(torch.mps, "compile_shader"):
            pytest.skip("PyTorch MPS shader support required")
    return request.param


def source(mode, seed=0, size=(17, 11)):
    count = size[0] * size[1] * CHANNELS[mode]
    values = [(i * 31 + i // 7 + seed) % 256 for i in range(count)]
    data = struct.pack(f"{count}f", *[(v - 80) / 127 for v in values]) if "F" in mode else bytes(values)
    image = Image.frombytes(mode, size, data)
    image.info["description"] = "resident operations"
    return image


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("method", range(7))
@pytest.mark.parametrize("size", [(17, 11), (31, 33)])
def test_transpose(runtime, mode, method, size):
    image = source(mode, size=size)
    gpu = ImageGpu.from_image(image, runtime=runtime)
    actual = gpu.transpose(method).to_image()
    expected = image.transpose(method)
    assert actual.size == expected.size
    assert actual.tobytes() == expected.tobytes()
    assert actual.info["description"] == image.info["description"]
    assert gpu.to_image().tobytes() == image.tobytes()


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("box", [(2, 3, 11, 8), (-3, -2, 20, 14), (20, 0, 24, 5), (0, 0, 17, 11)])
def test_crop(runtime, mode, box):
    image = source(mode)
    actual = ImageGpu.from_image(image, runtime=runtime).crop(box).to_image()
    expected = image.crop(box)
    assert actual.size == expected.size
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("offset", [(0, 0), (3, -5), (-36, 25), (10**30, -(10**30))])
def test_offset(runtime, mode, offset):
    image = source(mode)
    actual = ImageGpu.from_image(image, runtime=runtime).offset(*offset).to_image()
    expected = ImageChops.offset(image, offset[0] % image.width, offset[1] % image.height)
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("size", [(1, 1), (1, 33), (33, 1), (31, 33)])
def test_offset_wrap_boundaries(runtime, mode, size):
    image = source(mode, size=size)
    gpu = ImageGpu.from_image(image, runtime=runtime)
    width, height = size
    for x, y in [(0, 0), (width - 1, height - 1), (width, height), (-width - 1, -height - 1), (2**63, -(2**63))]:
        actual = gpu.offset(x, y).to_image()
        expected = ImageChops.offset(image, x % width, y % height)
        assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("mode", MODES[:4])
@pytest.mark.parametrize("factor", [-1.0, 0.0, 0.5, 1.0, 1.75])
def test_brightness(runtime, mode, factor):
    image = source(mode)
    actual = ImageGpu.from_image(image, runtime=runtime).brightness(factor).to_image()
    expected = ImageEnhance.Brightness(image).enhance(factor)
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("mode", MODES[:4])
def test_lookup_operations(runtime, mode):
    image = source(mode)
    gpu = ImageGpu.from_image(image, runtime=runtime)
    assert gpu.invert().to_image().tobytes() == ImageChops.invert(image).tobytes()
    lut = [(i * 7) % 256 for i in range(256 * CHANNELS[mode])]
    assert gpu.point(lut).to_image().tobytes() == image.point(lut).tobytes()
    assert gpu.point(list(range(256))).to_image().tobytes() == image.tobytes()
    for threshold in (-1, 0, 128, 256, 300):
        expected = image.point([v if v < threshold else 255 - v for v in range(256)] * CHANNELS[mode])
        assert gpu.solarize(threshold).to_image().tobytes() == expected.tobytes()
    for bits in (0, 1, 4, 8):
        expected = image.point([v & ((255 << (8 - bits)) & 255) for v in range(256)] * CHANNELS[mode])
        assert gpu.posterize(bits).to_image().tobytes() == expected.tobytes()


@pytest.mark.parametrize("mode", MODES[:4])
@pytest.mark.parametrize("operation", ["difference", "multiply", "screen", "lighter", "darker", "add", "subtract"])
def test_binary(runtime, mode, operation):
    image, other = source(mode), source(mode, 83)
    gpu, rhs = (ImageGpu.from_image(im, runtime=runtime) for im in (image, other))
    arguments = (2.5, -13) if operation in ("add", "subtract") else ()
    actual = getattr(gpu, operation)(rhs, *arguments).to_image()
    expected = getattr(ImageChops, operation)(image, other, *arguments)
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("mode", MODES[4:])
def test_float_arithmetic(runtime, mode):
    image, other = source(mode), source(mode, 83)
    gpu, rhs = (ImageGpu.from_image(im, runtime=runtime) for im in (image, other))
    count = image.width * image.height * CHANNELS[mode]
    a = struct.unpack(f"{count}f", image.tobytes())
    b = struct.unpack(f"{count}f", other.tobytes())
    operations = {
        "difference": lambda x, y: abs(x - y),
        "multiply": lambda x, y: x * y,
        "screen": lambda x, y: 1 - (1 - x) * (1 - y),
        "lighter": max,
        "darker": min,
        "add": lambda x, y: x + y,
        "subtract": lambda x, y: x - y,
    }
    for name, operation in operations.items():
        actual = struct.unpack(f"{count}f", getattr(gpu, name)(rhs).to_image().tobytes())
        expected = [operation(x, y) for x, y in zip(a, b, strict=True)]
        assert actual == pytest.approx(expected, rel=1e-5, abs=1e-6)
    inverted = struct.unpack(f"{count}f", gpu.invert().to_image().tobytes())
    assert inverted == pytest.approx([1 - v for v in a], rel=1e-5, abs=1e-6)
    bright = struct.unpack(f"{count}f", gpu.brightness(1.5).to_image().tobytes())
    expected = [v if mode == "RGBA;F" and i % 4 == 3 else v * 1.5 for i, v in enumerate(a)]
    assert bright == pytest.approx(expected, rel=1e-5, abs=1e-6)


def test_nonfinite_geometry_and_arithmetic(runtime):
    image = Image.frombytes("F", (3, 1), struct.pack("3f", float("inf"), -float("inf"), float("nan")))
    gpu = ImageGpu.from_image(image, runtime=runtime)
    assert gpu.transpose(0).crop((0, 0, 3, 1)).offset(3).to_image().tobytes() == image.transpose(0).tobytes()
    other = ImageGpu.from_image(Image.new("F", image.size, 1), runtime=runtime)
    for name in ("lighter", "darker", "multiply", "difference", "add", "subtract"):
        output = struct.unpack("3f", getattr(gpu, name)(other).to_image().tobytes())
        assert math.isnan(output[2])


def test_validation_and_lifetimes(runtime):
    gpu = ImageGpu.from_image(source("L"), runtime=runtime)
    for box in ((0, 0, 0, 1), (2, 0, 1, 1), (0, 0, 2**32, 1)):
        with pytest.raises(ValueError):
            gpu.crop(box)
    with pytest.raises(TypeError):
        gpu.crop((0.1, 0, 1, 1))
    with pytest.raises(ValueError):
        gpu.transpose(7)
    for value in (float("inf"), float("nan"), 1e40):
        with pytest.raises(ValueError):
            gpu.brightness(value)
    with pytest.raises(ValueError):
        gpu.point([1])
    with pytest.raises(ValueError):
        gpu.posterize(9)
    with pytest.raises(ValueError):
        gpu.add(gpu, scale=0)
    with pytest.raises(ValueError):
        gpu.add(ImageGpu.from_image(source("RGB"), runtime=runtime))
    result = gpu.transpose(2).crop((-2, -2, 9, 7)).invert().offset(-7)
    gpu.close()
    assert result.to_image().size == (11, 9)
    with pytest.raises(ValueError, match="closed"):
        gpu.invert()


def test_torch_pipeline_does_not_stage_pixels(monkeypatch):
    if not ImageGpu.capabilities()["available"]:
        pytest.skip("Metal required")
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("MPS required")
    tensor = torch.arange(3 * 11 * 17, dtype=torch.float32, device="mps").reshape(3, 11, 17)
    gpu = ImageGpu.from_torch(tensor[:, 1:10, 2:16], layout="CHW")

    def no_transfer(*args, **kwargs):
        raise AssertionError("unexpected CPU staging")

    with monkeypatch.context() as context:
        context.setattr(torch.Tensor, "cpu", no_transfer)
        context.setattr(torch.Tensor, "numpy", no_transfer)
        context.setattr(torch.Tensor, "tolist", no_transfer)
        result = gpu.crop((-2, 1, 11, 7)).transpose(2).offset(-2, 3).brightness(0.5).invert()
        result = result.multiply(result).difference(result).add(result, scale=2)
        output = result.to_torch()
        result.synchronize()
        assert tuple(output.shape) == (13, 6, 3)
        assert output.device.type == "mps"
    assert gpu.to_torch(layout="CHW").data_ptr() == tensor[:, 1:10, 2:16].data_ptr()
