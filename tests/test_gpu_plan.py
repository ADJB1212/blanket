import math
import struct
from concurrent.futures import ThreadPoolExecutor

import pytest

from blanket import Image, ImageGpu


@pytest.fixture
def metal():
    if not ImageGpu.capabilities()["available"]:
        pytest.skip("native Metal required")
    ImageGpu.configure_cache()
    yield
    ImageGpu.configure_cache()


def source(mode, size=(37, 29)):
    channels = {"L": 1, "LA": 2, "RGB": 3, "RGBA": 4, "F": 1, "RGB;F": 3, "RGBA;F": 4}[mode]
    samples = size[0] * size[1] * channels
    values = [(i * 37 + i // 11) % 256 for i in range(samples)]
    data = struct.pack(f"<{samples}f", *(v / 31 - 2 for v in values)) if "F" in mode else bytes(values)
    return Image.frombytes(mode, size, data)


@pytest.mark.parametrize("mode", ["L", "LA", "RGB", "RGBA", "F", "RGB;F", "RGBA;F"])
@pytest.mark.parametrize("memory_limit", [8_192, 1_000_000])
def test_plan_tiles_match_sequential_filters_and_rounding(metal, mode, memory_limit):
    operations = [
        ("brightness", {"factor": 1.37}),
        ("convolve", {"kernel": [1, -1, 2, -1, 5, -1, 2, -1, 1], "scale": 7, "offset": 0.7}),
        ("box_blur", {"radius": (1.2, 0.7)}),
        ("median_filter", {"size": 3}),
        ("invert", {}),
        ("brightness", {"factor": 0.73}),
    ]
    image = ImageGpu.from_image(source(mode))
    expected = image
    for name, arguments in operations:
        expected = getattr(expected, name)(**arguments)
    plan = ImageGpu.ProcessingPlan(image.size, mode, operations, memory_limit=memory_limit)
    result = plan.run(image)
    assert plan.info()["scratch_bytes"] <= memory_limit
    assert result.to_image().tobytes() == expected.to_image().tobytes()


@pytest.mark.parametrize("mode", ["L", "LA", "RGB", "RGBA", "F", "RGB;F", "RGBA;F"])
@pytest.mark.parametrize("count", range(5))
def test_full_image_plan_preserves_each_intermediate_rounding(metal, mode, count):
    image = ImageGpu.from_image(source(mode))
    operations = [("brightness", {"factor": factor}) for factor in (0.731, 1.37, 0.81, 1.19)[:count]]
    expected = image
    for name, arguments in operations:
        expected = getattr(expected, name)(**arguments)
    plan = ImageGpu.ProcessingPlan(image.size, mode, operations)
    assert plan.info()["tile_size"] == image.size
    assert plan.info()["scratch_bytes"] == len(image.to_image().tobytes()) * int(count > 1)
    result = plan.run(image)
    assert result.to_image().tobytes() == expected.to_image().tobytes()
    assert image.to_image().tobytes() == source(mode).tobytes()


@pytest.mark.parametrize("mode", ["L", "LA", "RGB", "RGBA"])
def test_plan_fuses_only_exact_uint8_lookup_steps(metal, mode):
    operations = [("invert", {}), ("posterize", {"bits": 5}), ("solarize", {"threshold": 90}), ("brightness", {"factor": 1.3}), ("invert", {})]
    image = ImageGpu.from_image(source(mode, (257, 3)))
    expected = image.invert().posterize(5).solarize(90).brightness(1.3).invert()
    plan = ImageGpu.ProcessingPlan(image.size, mode, operations, memory_limit=4_096)
    assert plan.info()["steps"] == 3
    assert plan.run(image).to_image().tobytes() == expected.to_image().tobytes()


def test_float_plan_preserves_nonfinite_values_and_evaluation_order(metal):
    values = [float("nan"), float("inf"), -float("inf"), -1e20, 0.3, -0.0, 1.0]
    image = ImageGpu.from_image(Image.frombytes("F", (7, 1), struct.pack("<7f", *values)))
    plan = ImageGpu.ProcessingPlan(image.size, "F", [("invert", {}), ("invert", {})], memory_limit=32)
    assert plan.info()["steps"] == 2
    expected = struct.unpack("<7f", image.invert().invert().to_image().tobytes())
    actual = struct.unpack("<7f", plan.run(image).to_image().tobytes())
    assert all(math.isnan(a) if math.isnan(b) else a == b for a, b in zip(actual, expected, strict=True))


@pytest.mark.parametrize("size", [(1, 17), (19, 1), (2, 2), (17, 13)])
def test_gaussian_halos_and_small_image_boundaries(metal, size):
    image = ImageGpu.from_image(source("RGBA", size))
    plan = ImageGpu.ProcessingPlan(size, "RGBA", [("gaussian_blur", {"radius": 1.7}), ("max_filter", {"size": 5})], memory_limit=4_096)
    assert plan.run(image).to_image().tobytes() == image.gaussian_blur(1.7).max_filter(5).to_image().tobytes()


@pytest.mark.parametrize("mode,channels", [("L", 1), ("LA", 2), ("RGB", 3), ("RGBA", 4)])
def test_sliding_blur_in_tiled_plan(metal, mode, channels):
    image = ImageGpu.from_image(source(mode, (257, 129)))
    operations = [("invert", {}), ("box_blur", {"radius": (32.5, 40)}), ("brightness", {"factor": 0.73})]
    plan = ImageGpu.ProcessingPlan(image.size, mode, operations, memory_limit=50_000 * channels)
    assert plan.info()["tile_size"] != image.size
    expected = image.invert().box_blur((32.5, 40)).brightness(0.73).to_image().tobytes()
    assert plan.run(image).to_image().tobytes() == expected


def test_batch_owns_inputs_controls_and_outputs_after_close(metal):
    images = [ImageGpu.from_image(source("RGBA", (21, 17))) for _ in range(3)]
    images[0].info["test"] = "first"
    plan = ImageGpu.ProcessingPlan(images[0].size, "RGBA", [("invert", {}), ("median_filter", {"size": 3})], memory_limit=2_048)
    expected = images[0].invert().median_filter().to_image().tobytes()
    outputs = plan.run_batch(images)
    assert outputs[0].info == {"test": "first"}
    assert outputs[1].info == {}
    event = outputs[0].record_event()
    for image in images:
        image.close()
    plan.close()
    ImageGpu.configure_cache(image_bytes=0, control_bytes=0)
    event.wait()
    assert all(output.to_image().tobytes() == expected for output in outputs)
    outputs[0].close()
    assert outputs[1].to_image().tobytes() == expected


def test_cache_budgets_eviction_and_reentrant_plan(metal):
    ImageGpu.configure_cache(image_bytes=2_048, control_bytes=2_048)
    image = ImageGpu.from_image(source("RGBA", (19, 13)))
    plan = ImageGpu.ProcessingPlan(image.size, "RGBA", [("invert", {})], memory_limit=1_024)
    expected = image.invert().to_image().tobytes()
    with ThreadPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(lambda _: plan.run(image).to_image().tobytes(), range(6)))
    assert all(result == expected for result in results)
    for value in range(5):
        image.point([value] * 256).synchronize()
    info = ImageGpu.cache_info()
    assert info["image_bytes"] <= info["image_limit"] == 2_048
    assert info["control_bytes"] <= info["control_limit"] == 2_048
    assert plan.run(image).to_image().tobytes() == expected
    ImageGpu.configure_cache(image_bytes=0, control_bytes=0)
    assert ImageGpu.cache_info()["image_bytes"] == ImageGpu.cache_info()["control_bytes"] == 0


def test_plan_validation_closed_inputs_and_empty_batches(metal):
    image = ImageGpu.from_image(source("L", (5, 3)))
    plan = ImageGpu.ProcessingPlan(image.size, "L", [], memory_limit=32)
    assert plan.run_batch([]) == []
    assert plan.run(image).to_image().tobytes() == image.to_image().tobytes()
    with pytest.raises(ValueError, match="match"):
        plan.run(ImageGpu.from_image(source("L", (3, 5))))
    with pytest.raises(TypeError):
        plan.run(source("L"))
    image.close()
    with pytest.raises(ValueError, match="closed"):
        plan.run(image)
    plan.close()
    with pytest.raises(ValueError, match="closed"):
        plan.run_batch([])
    with pytest.raises(ValueError, match="halos"):
        ImageGpu.ProcessingPlan((29, 31), "RGBA;F", [("median_filter", {"size": 7})], memory_limit=100)


@pytest.mark.parametrize("operations", [[("resize", {"size": (2, 2)})], [("brightness", {"factor": float("nan")})], [("rank_filter", {"size": 8, "rank": 0})]])
def test_plan_rejects_invalid_operations_before_hardware_access(operations):
    with pytest.raises(ValueError):
        ImageGpu.ProcessingPlan((3, 5), "RGBA", operations)


def test_cache_rejects_negative_budgets():
    with pytest.raises(ValueError):
        ImageGpu.configure_cache(image_bytes=-1)


@pytest.mark.parametrize("mode,channels", [("L", 1), ("RGB", 3), ("RGBA", 4)])
@pytest.mark.parametrize("depth", [10, 12, 16])
def test_unsigned_plans_quantize_each_public_operation(metal, mode, channels, depth):
    size = (31, 19)
    maximum = (1 << depth) - 1
    samples = [(i * 137 + i // 5) % (maximum + 1) for i in range(size[0] * size[1] * channels)]
    cpu = Image.frombytes(mode, size, struct.pack(f"<{len(samples)}H", *samples), bit_depth=depth)
    image = ImageGpu.from_image(cpu)
    operations = [("brightness", {"factor": 0.731}), ("invert", {}), ("gaussian_blur", {"radius": 0.8}), ("convolve", {"kernel": [1] * 9, "offset": 2.7}), ("min_filter", {"size": 3})]
    expected = image
    for name, arguments in operations:
        expected = getattr(expected, name)(**arguments)
    plan = ImageGpu.ProcessingPlan(size, mode, operations, bit_depth=depth, memory_limit=12_000)
    assert plan.info()["steps"] == 15
    result = plan.run(image)
    assert result.bit_depth == depth
    assert result.to_image().tobytes() == expected.to_image().tobytes()
    with pytest.raises(ValueError, match="depth"):
        ImageGpu.ProcessingPlan(size, mode, [], bit_depth=8).run(image)


def test_control_and_coefficient_caches_reuse_and_evict(metal):
    ImageGpu.configure_cache(image_bytes=0, control_bytes=16_384)
    image = ImageGpu.from_image(source("RGBA"))
    plan = ImageGpu.ProcessingPlan(image.size, "RGBA", [("invert", {})])
    retained = ImageGpu.cache_info()["control_bytes"]
    assert retained == 8_192
    other = ImageGpu.ProcessingPlan(image.size, "RGBA", [("invert", {})])
    assert ImageGpu.cache_info()["control_bytes"] == retained
    for value in range(3):
        ImageGpu.ProcessingPlan(image.size, "RGBA", [("point", {"lut": [value] * 256})])
    assert ImageGpu.cache_info()["control_bytes"] == 16_384
    assert plan.run(image).to_image().tobytes() == other.run(image).to_image().tobytes()
    ImageGpu.configure_cache(image_bytes=0, control_bytes=1_000_000)
    image.resize((193, 137)).synchronize()
    coefficients = ImageGpu.cache_info()["control_bytes"]
    assert coefficients > 0
    image.resize((193, 137)).synchronize()
    assert ImageGpu.cache_info()["control_bytes"] == coefficients


def test_plan_execution_keeps_pixels_resident(metal, monkeypatch):
    image = ImageGpu.from_image(source("L"))
    expected = image.box_blur(1.3).invert().to_image().tobytes()
    plan = ImageGpu.ProcessingPlan(image.size, "L", [("box_blur", {"radius": 1.3}), ("invert", {})], memory_limit=512)

    def forbidden(*args, **kwargs):
        raise AssertionError("unexpected CPU transfer")

    with monkeypatch.context() as context:
        context.setattr(ImageGpu.GpuImage, "to_image", forbidden)
        context.setattr(ImageGpu, "from_image", forbidden)
        result = plan.run(image)
        result.synchronize()
    assert result.to_image().tobytes() == expected
