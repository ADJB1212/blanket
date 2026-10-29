import gc
import os
import struct

import pytest

from blanket import Image, ImageGpu, _blanket, _gpu_cuda as cuda_runtime


@pytest.fixture
def cuda():
    capability = ImageGpu.capabilities("cuda")
    if not capability["available"]:
        if os.environ.get("BLANKET_REQUIRE_CUDA") == "1":
            pytest.fail(str(capability["diagnostic"]))
        pytest.skip(str(capability["diagnostic"]))
    ImageGpu.configure_cache(backend="cuda")
    yield cuda_runtime.cupy()
    gc.collect()
    ImageGpu.configure_cache(backend="cuda")


@pytest.mark.parametrize(
    "mode,channels,depth",
    [
        ("L", 1, 8),
        ("LA", 2, 8),
        ("RGB", 3, 8),
        ("RGBA", 4, 8),
        ("F", 1, 32),
        ("RGB;F", 3, 32),
        ("RGBA;F", 4, 32),
        *[(mode, channels, depth) for mode, channels in [("L", 1), ("RGB", 3), ("RGBA", 4)] for depth in (10, 12, 16)],
    ],
)
def test_cuda_tiled_plans_match_eager_rounding(cuda, mode, channels, depth):
    size = (37, 29)
    count = size[0] * size[1] * channels
    maximum = (1 << depth) - 1
    values = [(i * 137 + i // 7) % (maximum + 1) for i in range(count)]
    if depth == 32:
        data = struct.pack(f"<{count}f", *(v / 131 - 2 for v in values))
    elif depth == 8:
        data = bytes(values)
    else:
        data = struct.pack(f"<{count}H", *values)
    cpu = Image.frombytes(mode, size, data, **({"bit_depth": depth} if depth in (10, 12, 16) else {}))
    image = ImageGpu.from_image(cpu, backend="cuda")
    operations = [("brightness", {"factor": 0.731}), ("convolve", {"kernel": [1] * 9, "offset": 2.3}), ("gaussian_blur", {"radius": 0.7}), ("median_filter", {"size": 3}), ("invert", {})]
    if depth == 8:
        operations += [("solarize", {"threshold": 90}), ("posterize", {"bits": 5})]
    expected = image
    for name, arguments in operations:
        expected = getattr(expected, name)(**arguments)
    plan = ImageGpu.ProcessingPlan(size, mode, operations, bit_depth=depth, backend="cuda", memory_limit=16_384)
    assert plan.info()["scratch_bytes"] <= 16_384
    assert plan.run(image).to_image().tobytes() == expected.to_image().tobytes()


def test_cuda_plan_stream_ordering_ownership_and_caches(cuda):
    cp = cuda
    ImageGpu.configure_cache(backend="cuda", image_bytes=4_096, control_bytes=16_384)
    producer, consumer = cp.cuda.Stream(non_blocking=True), cp.cuda.Stream(non_blocking=True)
    with producer:
        image = ImageGpu.from_image(Image.new("RGBA", (31, 19), (30, 60, 90, 120)), backend="cuda")
        plan = ImageGpu.ProcessingPlan(image.size, image.mode, [("invert", {})], backend="cuda", memory_limit=2_048)
    with consumer:
        outputs = plan.run_batch([image, image])
    image.close()
    plan.close()
    expected = Image.new("RGBA", (31, 19), (225, 195, 165, 135)).tobytes()
    assert all(output.to_image().tobytes() == expected for output in outputs)
    for output in outputs:
        output.close()
    outputs.clear()
    gc.collect()
    info = ImageGpu.cache_info(backend="cuda")
    assert 0 < info["image_bytes"] <= info["image_limit"] == 4_096
    assert 0 < info["control_bytes"] <= info["control_limit"] == 16_384
    ImageGpu.configure_cache(backend="cuda", image_bytes=0, control_bytes=0)
    assert ImageGpu.cache_info(backend="cuda")["image_bytes"] == 0
    assert ImageGpu.cache_info(backend="cuda")["control_bytes"] == 0


def test_cuda_plan_noncontiguous_tensor_and_export_lifetime(cuda):
    torch = pytest.importorskip("torch")
    original = torch.arange(17 * 19 * 4, dtype=torch.float32, device="cuda").reshape(17, 19, 4)
    tensor = original[1::2, 2::2, :3]
    image = ImageGpu.from_torch(tensor, layout="HWC", mode="RGB;F")
    plan = ImageGpu.ProcessingPlan(image.size, image.mode, [("invert", {})], backend="cuda", memory_limit=1_024)
    result = plan.run(image)
    exported = result.to_torch(layout="HWC")
    result.close()
    image.close()
    for _ in range(3):
        temporary = ImageGpu.from_image(Image.new("RGB", (9, 8), (1, 2, 3)), backend="cuda")
        temporary.invert().synchronize()
    torch.testing.assert_close(exported, 1 - tensor)


def test_common_plan_preparation_is_backend_independent():
    if not hasattr(_blanket, "_gpu_prepare_plan"):
        pytest.skip("GPU preparation bindings required")
    steps, info = _blanket._gpu_prepare_plan(37, 29, 4, False, [(11, 0, 0, []), (11, 0, 0, [])], 512)
    assert info[0] == 1
    assert info[3] <= 512
    assert steps[0][3] == list(range(256)) * 4
    with pytest.raises(ValueError, match="halos"):
        _blanket._gpu_prepare_plan(37, 29, 4, True, [(19, 7, 0, [])], 16)


def test_cuda_float_fusion_preserves_nonfinite_values(cuda):
    data = struct.pack("<7f", float("nan"), float("inf"), -float("inf"), -1e20, 0.3, -0.0, 1.0)
    image = ImageGpu.from_image(Image.frombytes("F", (7, 1), data), backend="cuda")
    plan = ImageGpu.ProcessingPlan(image.size, "F", [("invert", {}), ("invert", {})], backend="cuda", memory_limit=32)
    assert plan.info()["steps"] == 2
    assert plan.run(image).to_image().tobytes() == image.invert().invert().to_image().tobytes()


@pytest.mark.parametrize("backend,device", [("bad", 0), ("cuda", -1), ("cuda", True), ("metal", 1)])
def test_plan_and_cache_backend_validation(backend, device):
    with pytest.raises(ValueError):
        ImageGpu.ProcessingPlan((1, 1), "L", [], backend=backend, device=device)
    with pytest.raises(ValueError):
        ImageGpu.configure_cache(backend=backend, device=device)
    with pytest.raises(ValueError):
        ImageGpu.cache_info(backend=backend, device=device)
