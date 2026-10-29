import pytest

from blanket import Image, ImageFilter, ImageGpu


def require_gpu():
    if not ImageGpu.capabilities()["available"]:
        pytest.skip("Metal feature and device required")


@pytest.mark.parametrize("mode", ["L", "LA", "RGB", "RGBA", "F", "RGB;F", "RGBA;F"])
def test_unified_memory_automatic_roundtrip_and_copy_on_write(mode: str) -> None:
    from blanket import ImageDraw

    require_gpu()
    channels = len(Image.new(mode, (1, 1)).getbands())
    value = 0.25 if "F" in mode else 57
    image = Image.new(mode, (17, 9), value if channels == 1 else (value,) * channels)
    expected = image.tobytes()
    gpu = ImageGpu.from_image(image, unified_memory=True)
    result = gpu.convert(mode)
    assert result.unified_memory and result.share().unified_memory
    cpu = result.to_image()
    sibling = result.to_image()
    restored = ImageGpu.from_image(cpu, unified_memory=True)
    assert restored.to_memoryview().obj is result.to_memoryview().obj
    result.close()
    gpu.close()
    image.close()
    assert cpu.tobytes() == expected
    cpu.putpixel((0, 0), 0 if channels == 1 else (0,) * channels)
    assert sibling.tobytes() == expected
    assert restored.to_image().tobytes() == expected
    detached = ImageGpu.from_image(cpu, unified_memory=True)
    assert detached.to_image().tobytes() == cpu.tobytes()
    assert detached.to_memoryview().obj is not restored.to_memoryview().obj
    if mode in ("L", "RGB", "RGBA"):
        ImageDraw.Draw(sibling).rectangle((0, 0, 3, 3), fill=0)
        assert restored.to_image().tobytes() == expected
    cpu.close()
    assert restored.to_image().tobytes() == expected


def test_unified_memory_paste_and_shared_image_close_on_worker() -> None:
    from concurrent.futures import ThreadPoolExecutor

    require_gpu()
    original = Image.new("RGBA", (17, 9), (50, 100, 150, 200))
    gpu = ImageGpu.from_image(original, unified_memory=True)
    cpu = gpu.to_image()
    last_shared = gpu.to_image()
    copied = cpu.copy()
    cpu.paste(Image.new("RGBA", (3, 3), "red"), (0, 0))
    assert gpu.to_image().tobytes() == original.tobytes()
    assert copied.tobytes() == original.tobytes()
    gpu.close()
    with ThreadPoolExecutor(max_workers=1) as pool:
        assert pool.submit(copied.tobytes).result() == original.tobytes()
        pool.submit(copied.close).result()
        pool.submit(cpu.close).result()
        pool.submit(last_shared.close).result()


@pytest.mark.parametrize("operation", ["putdata", "putalpha", "alpha_composite", "paste"])
def test_unified_memory_mutations_preserve_gpu_pixels(operation: str) -> None:
    import io

    require_gpu()
    original = Image.new("RGBA", (17, 9), (50, 100, 150, 200))
    gpu = ImageGpu.from_image(original, unified_memory=True)
    cpu = gpu.to_image()
    if operation == "putdata":
        cpu.putdata([(1, 2, 3, 4)] * (17 * 9))
    elif operation == "putalpha":
        cpu.putalpha(0)
    elif operation == "alpha_composite":
        cpu.alpha_composite(Image.new("RGBA", cpu.size, "red"))
    else:
        cpu.paste(Image.new("RGBA", cpu.size, "red"))
    assert cpu.tobytes() != original.tobytes()
    assert gpu.to_image().tobytes() == original.tobytes()
    output = io.BytesIO()
    shared = gpu.to_image()
    shared.save(output, format="PNG")
    assert Image.open(io.BytesIO(output.getvalue())).tobytes() == original.tobytes()


@pytest.mark.parametrize("mode", ["L", "LA", "RGB", "RGBA", "F", "RGB;F", "RGBA;F"])
def test_shared_metal_mapping_lifetime(mode: str) -> None:
    require_gpu()
    channels = len(Image.new(mode, (1, 1)).getbands())
    value = 0.25 if "F" in mode else 57
    image = Image.new(mode, (65, 17), value if channels == 1 else (value,) * channels)
    expected = image.tobytes()
    source = ImageGpu.from_image(image, runtime="native")
    result = source.convert(mode)
    result.info["description"] = "shared image"
    view = result.to_memoryview()
    assert view.readonly and view.c_contiguous
    assert bytes(view) == expected
    with pytest.raises(TypeError):
        view[0] = 1
    shared = result.share()
    assert shared.to_memoryview().obj is view.obj
    assert shared.info == result.info
    shared.info["description"] = "independent metadata"
    assert result.info["description"] == "shared image"
    result.close()
    source.close()
    image.close()
    for _ in range(20):
        temporary = ImageGpu.from_image(Image.new(mode, (65, 17)), runtime="native")
        temporary.close()
    assert bytes(view) == expected
    view.release()
    assert shared.to_image().tobytes() == expected
    last_view = shared.to_memoryview()
    shared.close()
    replacement = ImageGpu.from_image(Image.new(mode, (65, 17)), runtime="native")
    assert bytes(last_view) == expected
    replacement.close()
    last_view.release()


def test_shared_metal_handles_reject_closed_images() -> None:
    require_gpu()
    gpu = ImageGpu.from_image(Image.new("RGBA", (3, 5)), runtime="native")
    gpu.close()
    with pytest.raises(ValueError, match="closed"):
        gpu.to_memoryview()
    with pytest.raises(ValueError, match="closed"):
        gpu.share()


@pytest.mark.parametrize("mode", ["L", "LA", "RGB", "RGBA", "F", "RGB;F", "RGBA;F"])
def test_native_transfers_bypass_python_bytes(mode: str, monkeypatch: pytest.MonkeyPatch) -> None:
    require_gpu()
    image = Image.new(mode, (65, 17))
    expected = image.tobytes()

    def unexpected(*args: object, **kwargs: object) -> None:
        raise AssertionError("native transfers must not stage pixels in Python bytes")

    with monkeypatch.context() as patch:
        patch.setattr(Image.Image, "tobytes", unexpected)
        patch.setattr(Image, "frombytes", unexpected)
        resident = ImageGpu.from_image(image, runtime="native")
        result = resident.to_image()
        image.close()
        resident.close()
    assert result.tobytes() == expected


@pytest.mark.parametrize("mode", ["L", "LA", "RGB", "RGBA"])
@pytest.mark.parametrize("width", [1, 3, 63, 64, 65, 257])
def test_native_packed_operations_and_tails(mode: str, width: int) -> None:
    from blanket import ImageChops, ImageEnhance

    require_gpu()
    channels = len(Image.new(mode, (1, 1)).getbands())
    size = (width, 3)
    image = Image.frombytes(mode, size, bytes((i * 37 + 13) % 256 for i in range(width * 3 * channels)))
    other = Image.frombytes(mode, size, bytes((i * 19 + 71) % 256 for i in range(width * 3 * channels)))
    gpu = ImageGpu.from_image(image, runtime="native")
    rhs = ImageGpu.from_image(other, runtime="native")
    assert gpu.invert().to_image().tobytes() == ImageChops.invert(image).tobytes()
    assert gpu.convert(mode).to_image().tobytes() == image.tobytes()
    for factor in (-0.5, 0.0, 0.4, 1.0, 1.75):
        assert gpu.brightness(factor).to_image().tobytes() == ImageEnhance.Brightness(image).enhance(factor).tobytes()
        assert gpu.blend(rhs, factor).to_image().tobytes() == Image.blend(image, other, factor).tobytes()


@pytest.mark.parametrize("mode", ["L", "RGBA", "F", "RGBA;F"])
def test_native_closed_sources_survive_buffer_reuse(mode: str) -> None:
    require_gpu()
    pending = []
    for value in range(24):
        color = value if mode in ("L", "F") else (value,) * 4
        image = Image.new(mode, (257, 129), color)
        source = ImageGpu.from_image(image, runtime="native")
        copied = source.convert(mode)
        result = source.blend(copied, 0.5)
        source.close()
        copied.close()
        # New uploads must not overwrite buffers still read by queued consumers.
        replacement = ImageGpu.from_image(Image.new(mode, image.size), runtime="native")
        replacement.close()
        pending.append((result, image.tobytes()))
    for result, expected in reversed(pending):
        assert result.to_image().tobytes() == expected
        result.close()


@pytest.mark.parametrize("mode", ["L", "LA", "RGB", "RGBA", "F", "RGB;F", "RGBA;F"])
def test_upload_roundtrip_and_lifetime(mode):
    require_gpu()
    image = Image.new(mode, (17, 9))
    image.info["description"] = "gpu"
    resident = ImageGpu.from_image(image)
    result = resident.resize((7, 13)).convolve([0, 0, 0, 0, 1, 0, 0, 0, 0])
    resident.close()
    assert result.to_image().tobytes() == Image.new(mode, (7, 13)).tobytes()
    assert result.to_image().info["description"] == "gpu"
    with pytest.raises(ValueError, match="closed"):
        resident.to_image()


@pytest.mark.parametrize("method", range(6))
@pytest.mark.parametrize("mode", ["L", "LA", "RGB", "RGBA"])
def test_resize_cpu_parity(mode, method):
    require_gpu()
    channels = len(Image.new(mode, (1, 1)).getbands())
    data = bytes((i * 31 + i // 11) % 256 for i in range(23 * 19 * channels))
    image = Image.frombytes(mode, (23, 19), data)
    actual = ImageGpu.from_image(image).resize((11, 27), method).to_image()
    expected = image.resize((11, 27), method)
    assert max(abs(a - b) for a, b in zip(actual.tobytes(), expected.tobytes(), strict=True)) <= 1


def test_compositing_and_convolution():
    require_gpu()
    a = Image.new("RGBA", (13, 17), (40, 150, 70, 127))
    b = Image.new("RGBA", a.size, (220, 30, 60, 83))
    ga, gb = ImageGpu.from_image(a), ImageGpu.from_image(b)
    expected = Image.alpha_composite(a, b)
    actual = ga.alpha_composite(gb).to_image()
    assert max(abs(x - y) for x, y in zip(actual.tobytes(), expected.tobytes(), strict=True)) <= 1
    kernel = [1, 2, 1, 2, 4, 2, 1, 2, 1]
    expected = a.filter(ImageFilter.Kernel((3, 3), kernel, scale=16))
    assert ga.convolve(kernel, scale=16).to_image().tobytes() == expected.tobytes()
    assert ga.blend(gb, 0.5).to_image().tobytes() == Image.blend(a, b, 0.5).tobytes()


def test_tensor_residency_views_and_sharing(monkeypatch):
    require_gpu()
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available() or not hasattr(torch.mps, "compile_shader"):
        pytest.skip("PyTorch MPS shader support required")
    source = torch.arange(3 * 19 * 23, dtype=torch.float32, device="mps").reshape(3, 19, 23) / 2048
    view = source[:, 1:18, 2:21]
    resident = ImageGpu.from_torch(view, layout="CHW")
    assert resident.to_torch(layout="CHW").data_ptr() == view.data_ptr()

    def no_cpu(*args, **kwargs):
        raise AssertionError("unexpected CPU transfer")

    with monkeypatch.context() as context:
        context.setattr(torch.Tensor, "cpu", no_cpu)
        result = resident.resize((7, 11)).convolve([0, 0, 0, 0, 1, 0, 0, 0, 0])
        resident.close()
        output = result.to_torch()
        assert output.device.type == "mps"
        assert tuple(output.shape) == (11, 7, 3)
        result.synchronize()
    assert result.to_image().mode == "RGB;F"


def test_validation():
    require_gpu()
    with pytest.raises(ValueError):
        ImageGpu.from_image(Image.new("P", (2, 2)))
    resident = ImageGpu.from_image(Image.new("L", (2, 2)))
    with pytest.raises(ValueError):
        resident.resize((0, 2))
    with pytest.raises(ValueError):
        resident.convolve([1] * 9, scale=0)
    with pytest.raises(ValueError):
        resident.alpha_composite(resident)


@pytest.mark.parametrize("mode", ["F", "RGB;F", "RGBA;F"])
@pytest.mark.parametrize("method", range(6))
def test_float_resize(mode, method):
    import struct

    require_gpu()
    channels = {"F": 1, "RGB;F": 3, "RGBA;F": 4}[mode]
    samples = [((i * 31) % 251) / 250 for i in range(13 * 17 * channels)]
    image = Image.frombytes(mode, (13, 17), struct.pack(f"{len(samples)}f", *samples))
    actual = ImageGpu.from_image(image).resize((7, 23), method).to_image().tobytes()
    expected = image.resize((7, 23), method).tobytes()
    a = struct.unpack(f"{len(actual) // 4}f", actual)
    b = struct.unpack(f"{len(expected) // 4}f", expected)
    assert all(abs(x - y) <= 1e-6 + 1e-5 * abs(y) for x, y in zip(a, b, strict=True))


def test_torch_upload_copy_export_and_validation():
    require_gpu()
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("MPS required")
    image = Image.new("RGBA", (7, 11), (13, 27, 42, 255))
    gpu = ImageGpu.from_image(image, runtime="torch")
    assert gpu.to_image().tobytes() == image.tobytes()
    tensor = gpu.to_torch()
    copied = ImageGpu.from_torch(tensor, layout="HWC", copy=True)
    assert copied.to_torch().data_ptr() != tensor.data_ptr()
    assert copied.blend(gpu, 0.3).to_image().tobytes() == image.tobytes()
    native = ImageGpu.from_image(image)
    with pytest.raises(ValueError, match="execution context"):
        gpu.blend(native, 0.5)
    with pytest.raises(ValueError, match="runtime"):
        native.to_torch()
    with pytest.raises(ValueError):
        ImageGpu.from_torch(torch.zeros(2, 2), layout="HW")
    with pytest.raises(ValueError):
        ImageGpu.from_torch(torch.zeros(2, 2, device="mps", requires_grad=True), layout="HW")


@pytest.mark.parametrize("mode", ["L", "LA", "RGB", "RGBA"])
def test_box_half_and_same_size(mode):
    require_gpu()
    channels = len(Image.new(mode, (1, 1)).getbands())
    image = Image.frombytes(mode, (16, 18), bytes((i * 37 + i // 7) % 256 for i in range(16 * 18 * channels)))
    gpu = ImageGpu.from_image(image)
    assert gpu.resize(image.size).to_image().tobytes() == image.tobytes()
    assert gpu.resize((8, 9), Image.Resampling.BOX).to_image().tobytes() == image.resize((8, 9), Image.Resampling.BOX).tobytes()


def test_same_size_float_preserves_nonfinite():
    import math
    import struct

    require_gpu()
    image = Image.frombytes("F", (3, 1), struct.pack("3f", float("inf"), -float("inf"), float("nan")))
    output = ImageGpu.from_image(image).resize(image.size).to_image().tobytes()
    a, b, c = struct.unpack("3f", output)
    assert a == float("inf") and b == -float("inf") and math.isnan(c)


@pytest.mark.parametrize("floating", [False, True])
def test_native_download_owns_completed_pixels(floating):
    import struct

    from blanket import _blanket

    require_gpu()
    samples = [i % 251 for i in range(17 * 9 * 4)]
    data = struct.pack(f"{len(samples)}f", *samples) if floating else bytes(samples)
    source = _blanket.MetalImage(17, 9, 4, floating, data)
    result = source.run(source, [17, 9, 4, 17, 9, 6, 0, 0], [])
    del source
    first = result.download()
    second = result.download()
    del result
    assert isinstance(first, bytes)
    assert first == second == data


@pytest.mark.parametrize("floating", [False, True])
@pytest.mark.parametrize(("index", "value"), [(0, float("nan")), (3, float("inf")), (6, float("nan")), (7, float("inf"))])
def test_native_lut_rejects_nonfinite_controls(floating, index, value):
    from blanket import _blanket

    require_gpu()
    source = _blanket.MetalImage(1, 1, 3, floating, bytes(12 if floating else 3))
    controls = _blanket._gpu_lut_controls([2, 2, 2], 3, [0.0] * 24, floating)
    controls[index] = value
    with pytest.raises(ValueError, match="invalid GPU dispatch"):
        source.run(source, [1, 1, 3, 1, 1, 25, 3, 5 if floating else 2], controls)


@pytest.mark.parametrize("floating", [False, True])
@pytest.mark.parametrize("control_samples", [1023, 1024, 1025])
def test_native_control_storage_boundary(floating, control_samples):
    import struct

    from blanket import _blanket

    require_gpu()
    data = struct.pack("3f", 0.0, 0.25, 1.0) if floating else bytes((0, 34, 255))
    source = _blanket.MetalImage(3, 1, 1, floating, data)
    # Pad resize controls to immediately below, at, and above Metal's 4 KiB limit.
    controls = [1.0, 1.0, 1.0] + [0.0] * (control_samples - 3)
    parameters = [3, 1, 1, 1, 1, 0, control_samples, 0]
    first = source.run(source, parameters, controls)
    controls[2] = 2.0
    second = source.run(source, parameters, controls)
    del source, controls
    assert first.download() == (struct.pack("f", 0.25) if floating else bytes((34,)))
    assert second.download() == (struct.pack("f", 0.5) if floating else bytes((68,)))
