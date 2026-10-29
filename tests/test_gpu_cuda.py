import inspect
import itertools
import os
from pathlib import Path

import pytest
import test_gpu_color as color_contracts
import test_gpu_geometry as geometry_contracts
import test_gpu_operations as operation_contracts
import test_gpu_precision as precision_contracts
import test_gpu_unsigned as unsigned_contracts

from blanket import Image, ImageGpu, _gpu_cuda as cuda_runtime


def contract_cases():
    """Run the same numeric contracts on CUDA as on native Metal and MPS."""
    for module in (color_contracts, geometry_contracts, operation_contracts, precision_contracts, unsigned_contracts):
        for name, function in vars(module).items():
            if not name.startswith("test_") or not callable(function) or "runtime" not in inspect.signature(function).parameters:
                continue
            marks = [mark for mark in getattr(function, "pytestmark", []) if mark.name == "parametrize"]
            combinations = itertools.product(*(mark.args[1] for mark in marks))
            for index, combination in enumerate(combinations):
                arguments = {}
                for mark, values in zip(marks, combination, strict=True):
                    names = mark.args[0]
                    names = [name.strip() for name in names.split(",")] if isinstance(names, str) else list(names)
                    values = (values,) if len(names) == 1 else values
                    arguments.update(zip(names, values, strict=True))
                yield pytest.param(function, arguments, id=f"{module.__name__}/{name}/{index}")


@pytest.fixture
def cuda(monkeypatch):
    capability = ImageGpu.capabilities("cuda")
    if not capability["available"]:
        if os.environ.get("BLANKET_REQUIRE_CUDA") == "1":
            pytest.fail(f"NVIDIA hardware CI requires CUDA: {capability['diagnostic']}")
        pytest.skip(str(capability["diagnostic"]))
    original = ImageGpu.from_image

    def upload(image, *, runtime="native", backend="cuda", device=0):
        return original(image, backend="cuda", device=device)

    monkeypatch.setattr(ImageGpu, "from_image", upload)
    return cuda_runtime.cupy()


@pytest.mark.parametrize(("function", "arguments"), list(contract_cases()))
def test_shared_metal_contracts_on_cuda(cuda, function, arguments):
    function(runtime="cuda", **arguments)


def test_capability_diagnostics_without_cuda(monkeypatch):
    from blanket import _blanket

    monkeypatch.delattr(_blanket, "_gpu_cuda_fatbin", raising=False)
    capability = ImageGpu.capabilities("cuda")
    assert not capability["compiled"]
    assert not capability["available"]
    assert capability["compiled_operations"] == ()
    assert "gpu-cuda feature" in capability["diagnostic"]
    with pytest.raises(RuntimeError, match="gpu-cuda feature"):
        ImageGpu.from_image(Image.new("L", (1, 1)), backend="cuda")


def test_driver_diagnostics(monkeypatch):
    from blanket import _blanket

    monkeypatch.setattr(_blanket, "_gpu_cuda_fatbin", b"", raising=False)

    class Runtime:
        @staticmethod
        def getDeviceCount():
            raise RuntimeError("driver not found")

    class Cuda:
        runtime = Runtime

    class CuPy:
        cuda = Cuda

    monkeypatch.setattr(cuda_runtime, "cupy", lambda: CuPy)
    result = ImageGpu.capabilities("cuda")
    assert not result["available"]
    assert "driver not found" in result["diagnostic"]
    with pytest.raises(ValueError, match="nonnegative"):
        cuda_runtime.require(-1)


def test_precompiled_module_loads_without_source_compilation(monkeypatch):
    from blanket import _blanket

    monkeypatch.setattr(_blanket, "_gpu_cuda_fatbin", b"test fatbinary", raising=False)
    paths = []
    names = []

    class Device:
        def __init__(self, device):
            assert device == 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    class Module:
        def __init__(self, *, path):
            self.path = Path(path)
            assert self.path.read_bytes() == b"test fatbinary"
            paths.append(self.path)

        def get_function(self, name):
            assert self.path.exists()
            names.append(name)
            return lambda: name

    class Cuda:
        pass

    Cuda.Device = Device

    class CuPy:
        cuda = Cuda
        RawModule = Module

    monkeypatch.setattr(cuda_runtime, "cupy", lambda: CuPy)
    cuda_runtime.library.cache_clear()
    try:
        resolve = cuda_runtime.library(0)
        assert len(names) == 8
        assert resolve("image_f32")() == "image_f32"
        assert resolve("convert_u8_f32")() == "convert_u8_f32"
        assert all(not path.exists() for path in paths)
    finally:
        cuda_runtime.library.cache_clear()


def test_dlpack_views_ownership_and_streams(cuda, monkeypatch):
    import torch

    producer = torch.cuda.Stream()
    consumer = torch.cuda.Stream()
    processing = cuda.cuda.Stream(non_blocking=True)
    with torch.cuda.stream(producer):
        tensor = torch.arange(4 * 19 * 23, dtype=torch.float32, device="cuda").reshape(4, 19, 23) / 2048
        view = tensor[:, 1:18, 2:21]
        original = tensor.clone()
        with processing:
            image = ImageGpu.from_torch(view, layout="CHW")
    assert image.to_torch(layout="CHW").data_ptr() == view.data_ptr()

    def no_staging(*args, **kwargs):
        raise AssertionError("unexpected CPU pixel staging")

    with monkeypatch.context() as patch:
        for name in ("cpu", "numpy", "tolist", "item", "__float__"):
            patch.setattr(torch.Tensor, name, no_staging)
        patch.setattr(cuda_runtime.CudaImage, "download", no_staging)
        patch.setattr(cuda, "asnumpy", no_staging)
        # Consume on a different CuPy stream, then export on a PyTorch stream.
        with cuda.cuda.Stream(non_blocking=True):
            result = image.resize((11, 13)).color(0.8).contrast(1.2).box_blur(1).rotate(12)
            reduction = result.statistics()
        with torch.cuda.stream(consumer):
            output = result.to_torch(layout="CHW").clone()
        image.close()
        del tensor, view
        result.synchronize()
        reduction.synchronize()
    consumer.synchronize()
    assert output.device.type == "cuda"
    assert tuple(output.shape) == (4, 13, 11)
    assert torch.isfinite(output).all()
    assert torch.equal(original, torch.arange(4 * 19 * 23, dtype=torch.float32, device="cuda").reshape(4, 19, 23) / 2048)


def test_dlpack_copy_and_export_lifetime(cuda):
    import torch

    tensor = torch.zeros((7, 9, 3), dtype=torch.uint8, device="cuda")
    borrowed = ImageGpu.from_torch(tensor, layout="HWC")
    copied = ImageGpu.from_torch(tensor, layout="HWC", copy=True)
    borrowed.synchronize()
    copied.synchronize()
    tensor.fill_(17)
    torch.cuda.current_stream().synchronize()
    assert borrowed.to_torch().data_ptr() == tensor.data_ptr()
    assert copied.to_torch().data_ptr() != tensor.data_ptr()
    output = borrowed.invert().to_torch()
    borrowed.close()
    del tensor
    assert torch.all(output == 238)
    assert torch.all(copied.to_torch() == 0)


def test_explicit_device_and_cross_device_rejection(cuda):
    source = ImageGpu.from_image(Image.new("RGB", (3, 2), (1, 2, 3)), device=0)
    assert source.device == "cuda:0"
    with pytest.raises(ValueError, match="unavailable"):
        cuda_runtime.require(cuda.cuda.runtime.getDeviceCount())
    if cuda.cuda.runtime.getDeviceCount() > 1:
        other = ImageGpu.from_image(Image.new("RGB", (3, 2)), device=1)
        with pytest.raises(ValueError, match="execution context"):
            source.blend(other, 0.5)


def test_cuda_allocation_error_has_device_context(cuda, monkeypatch):
    image = ImageGpu.from_image(Image.new("L", (3, 2)))

    def fail_allocation(*args, **kwargs):
        raise RuntimeError("out of device memory")

    monkeypatch.setattr(cuda, "empty", fail_allocation)
    with pytest.raises(RuntimeError, match=r"CUDA dispatch failed on device 0.*out of device memory"):
        image.invert()
