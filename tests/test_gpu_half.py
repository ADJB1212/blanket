import asyncio
import math
import struct

import pytest
from gpu_helpers import CHANNELS, require_metal

from blanket import Image, ImageFilter, ImageGpu


@pytest.fixture(params=("metal", "cuda"))
def backend(request):
    if not ImageGpu.capabilities(request.param)["available"]:
        pytest.skip(f"{request.param} feature and device required")
    return request.param


@pytest.mark.parametrize("mode", ["F", "RGB;F", "RGBA;F"])
def test_half_upload_storage_and_async_roundtrip(backend, mode):
    values = [0.0, -0.0, 1.00048828125, 1.00146484375, 2**-24, -(2**-24), -7.123, 65504, 65520, -65520, float("inf"), -float("inf"), float("nan")]
    samples = values * CHANNELS[mode]
    count = len(samples)
    source = Image.frombytes(mode, (len(values), 1), struct.pack(f"<{count}f", *samples))
    source.info["description"] = "HDR"
    packed = struct.pack(f"<{count}e", *(math.copysign(math.inf, v) if abs(v) >= 65520 else v for v in samples))
    expected = struct.pack(f"<{count}f", *struct.unpack(f"<{count}e", packed))

    async def roundtrip():
        half = await ImageGpu.from_image_async(source, backend=backend, dtype="float16")
        shared = half.share()
        assert half.dtype == shared.dtype == "float16"
        assert half.mode == mode
        if backend == "metal":
            view = half.to_memoryview()
            assert len(view) == len(packed)
            for start in range(0, len(packed), len(values) * 2):
                end = start + (len(values) - 1) * 2
                assert view[start:end].tobytes() == packed[start:end]
        half.close()
        pending = shared.to_image_async()
        shared.close()
        return await pending

    restored = asyncio.run(asyncio.wait_for(roundtrip(), timeout=10))
    assert restored.bit_depth == 32 and restored.mode == mode
    assert restored.info["description"] == "HDR"
    actual = restored.tobytes()
    for start in range(0, len(expected), len(values) * 4):
        end = start + (len(values) - 1) * 4
        assert actual[start:end] == expected[start:end]
        assert math.isnan(struct.unpack_from("<f", actual, end)[0])


def test_half_processing_and_plans_promote_resident_inputs(backend, monkeypatch):
    size = (9, 7)
    image = Image.frombytes("RGB;F", size, struct.pack("<189f", *([0.37, 2.13, -0.17] * 63)))
    full = ImageGpu.from_image(image, backend=backend)
    half = full.astype("float16")
    mask = ImageGpu.from_image(Image.new("F", size, 0.5), backend=backend, dtype="float16")
    lut = ImageFilter.Color3DLUT.generate(3, lambda r, g, b: (r * 0.7, g, b + 0.2))
    plan = ImageGpu.ProcessingPlan(size, "RGB;F", [("invert", {}), ("brightness", {"factor": 1.3})], backend=backend)

    def unexpected(*args, **kwargs):
        raise AssertionError("processing should stay on the GPU")

    def process(image, mask):
        return [image.add(image), full.add(image), image.resize((5, 3)), image.gaussian_blur(0.7), image.paste(full, mask=mask), image.color_lut(lut), *plan.run_batch([image, full])]

    with monkeypatch.context() as patch:
        patch.setattr(ImageGpu.GpuImage, "to_image", unexpected)
        patch.setattr(ImageGpu, "from_image", unexpected)
        actual = process(half, mask)
        expected = process(half.astype("float32"), mask.astype("float32"))
        assert half.dtype == mask.dtype == "float16"
        half.close()
        mask.close()
        full.close()
        plan.close()

    for result, reference in zip(actual, expected, strict=True):
        assert result.dtype == "float32"
        assert result.to_image().tobytes() == reference.to_image().tobytes()


def test_half_upload_validation():
    require_metal()
    with pytest.raises(ValueError, match="float-mode"):
        ImageGpu.from_image(Image.new("L", (1, 1)), dtype="float16")
    with pytest.raises(ValueError, match="unified memory"):
        ImageGpu.from_image(Image.new("F", (1, 1)), dtype="float16", unified_memory=True)
    image = ImageGpu.from_image(Image.new("F", (1, 1)), dtype="float32", unified_memory=True)
    half = image.astype("float16")
    assert not half.unified_memory
    with pytest.raises(ValueError, match="GPU dtype conversion"):
        half.astype("uint8")
