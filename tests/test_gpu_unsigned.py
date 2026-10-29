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


@pytest.mark.parametrize("depth", [10, 12, 16])
@pytest.mark.parametrize("mode,channels", [("L", 1), ("RGB", 3), ("RGBA", 4)])
def test_unsigned_precision_roundtrip_and_geometry(runtime, depth, mode, channels):
    maximum = (1 << depth) - 1
    samples = [0, 1, maximum // 2, maximum - 1, maximum] * channels
    image = Image.frombytes(mode, (5, 1), struct.pack(f"{len(samples)}H", *samples), bit_depth=depth)
    gpu = ImageGpu.from_image(image, runtime=runtime)
    assert gpu.bit_depth == depth
    assert gpu.mode == mode
    assert gpu.to_image().tobytes() == image.tobytes()
    assert gpu.copy().to_image().tobytes() == image.tobytes()
    assert gpu.invert().invert().to_image().tobytes() == image.tobytes()
    result = gpu.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    gpu.close()
    assert result.to_image().tobytes() == image.transpose(Image.Transpose.FLIP_LEFT_RIGHT).tobytes()
    assert result.to_image().bit_depth == depth


@pytest.mark.parametrize("depth", [10, 12, 16])
def test_every_unsigned_sample_retains_precision(runtime, depth):
    samples = list(range(1 << depth))
    data = struct.pack(f"{len(samples)}H", *samples)
    image = Image.frombytes("L", (len(samples), 1), data, bit_depth=depth)
    gpu = ImageGpu.from_image(image, runtime=runtime)
    assert gpu.copy().to_image().tobytes() == data
    assert gpu.invert().to_image().tobytes() == struct.pack(f"{len(samples)}H", *reversed(samples))


@pytest.mark.parametrize("depth", [10, 12, 16])
def test_unsigned_filters_rounding_and_saturation(runtime, depth):
    maximum = (1 << depth) - 1
    image = Image.frombytes("L", (3, 1), struct.pack("3H", 0, maximum, 0), bit_depth=depth)
    gpu = ImageGpu.from_image(image, runtime=runtime)
    assert struct.unpack("3H", gpu.box_blur((1, 0)).to_image().tobytes()) == (round(maximum / 3),) * 3
    assert struct.unpack("3H", gpu.rank_filter(3, 0).to_image().tobytes()) == (0, 0, 0)
    assert struct.unpack("3H", gpu.brightness(3).to_image().tobytes()) == (0, maximum, 0)
    assert struct.unpack("3H", gpu.blend(gpu.invert(), 0.5).to_image().tobytes()) == ((maximum + 1) // 2,) * 3
    assert gpu.resize((7, 3)).to_image().bit_depth == depth


def test_unsigned_transfer_nonfinite_and_malformed():
    from blanket import _blanket

    values = struct.pack("4f", float("nan"), float("inf"), -float("inf"), 0.5)
    assert struct.unpack("4H", _blanket._gpu_unsigned_bytes(values, 16, True)) == (0, 65535, 0, 32768)
    for depth, data in [(8, b""), (16, b"x")]:
        with pytest.raises(ValueError):
            _blanket._gpu_unsigned_bytes(data, depth, False)


@pytest.mark.parametrize("depth", [10, 12, 16])
def test_unsigned_alpha(runtime, depth):
    maximum = (1 << depth) - 1
    background = Image.frombytes("RGBA", (1, 1), struct.pack("4H", maximum, 0, 0, maximum), bit_depth=depth)
    foreground = Image.frombytes("RGBA", (1, 1), struct.pack("4H", 0, maximum, 0, maximum // 2), bit_depth=depth)
    result = ImageGpu.from_image(background, runtime=runtime).alpha_composite(ImageGpu.from_image(foreground, runtime=runtime)).to_image()
    actual = struct.unpack("4H", result.tobytes())
    assert actual == (maximum - maximum // 2, maximum // 2, 0, maximum)


def test_unsigned_tensor_import_stays_resident():
    if not ImageGpu.capabilities()["available"]:
        pytest.skip("Metal required")
    torch = pytest.importorskip("torch")
    tensor = torch.arange(12, dtype=torch.float32, device="mps").reshape(2, 2, 3).to(torch.uint16)
    gpu = ImageGpu.from_torch(tensor, layout="HWC", mode="RGB", bit_depth=12)
    assert gpu.to_image().tobytes() == struct.pack("12H", *range(12))
    assert gpu.to_torch().device.type == "mps"
