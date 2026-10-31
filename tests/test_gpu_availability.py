from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from blanket import Image, ImageChops, ImageGpu, _blanket, _gpu_cuda


@pytest.mark.parametrize("backend", ["metal", "cuda"])
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("operation", ["image", "stream", "plan", "async"])
def test_unavailable_gpu_requests_raise_without_cpu_fallback(backend: str, compiled: bool, operation: str, monkeypatch: pytest.MonkeyPatch) -> None:
    if backend == "metal":
        if compiled:
            monkeypatch.setattr(_blanket, "MetalImage", object, raising=False)
        else:
            monkeypatch.delattr(_blanket, "MetalImage", raising=False)
        monkeypatch.setattr(_blanket, "_gpu_available", lambda: False, raising=False)
        monkeypatch.setattr(_blanket, "_gpu_diagnostic", lambda: "unsupported Metal GPU", raising=False)
    else:
        if compiled:
            monkeypatch.setattr(_blanket, "_gpu_cuda_fatbin", b"mock", raising=False)
        else:
            monkeypatch.delattr(_blanket, "_gpu_cuda_fatbin", raising=False)
        monkeypatch.setattr(_gpu_cuda, "cupy", lambda: SimpleNamespace(cuda=SimpleNamespace(runtime=SimpleNamespace(getDeviceCount=lambda: 0))))

    capability = ImageGpu.capabilities(backend)
    assert capability["compiled"] is compiled
    assert capability["available"] is False
    assert capability["diagnostic"]
    image = Image.new("L", (2, 3), 17)
    with pytest.raises(RuntimeError, match=r"Metal|CUDA"):
        if operation == "image":
            ImageGpu.from_image(image, backend=backend)
        elif operation == "stream":
            ImageGpu.Stream(backend)
        elif operation == "plan":
            ImageGpu.ProcessingPlan(image.size, image.mode, [], backend=backend)
        else:
            ImageGpu.from_image_async(image, backend=backend).result(timeout=10)
    assert ImageChops.invert(image).tobytes() == bytes([238] * 6)


def test_cuda_missing_dependency_reports_install_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_blanket, "_gpu_cuda_fatbin", b"mock", raising=False)
    monkeypatch.setitem(sys.modules, "cupy", None)
    capability = ImageGpu.capabilities("cuda")
    assert capability["compiled"] is True
    assert capability["available"] is False
    assert "cupy-cuda13x[ctk]" in str(capability["diagnostic"])
    with pytest.raises(RuntimeError, match="cupy-cuda13x"):
        ImageGpu.from_image(Image.new("L", (1, 1)), backend="cuda")


def test_missing_torch_reports_runtime_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(RuntimeError, match="install 'torch"):
        ImageGpu._torch()


def test_cuda_context_failure_reports_runtime_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_blanket, "_gpu_cuda_fatbin", b"mock", raising=False)
    monkeypatch.setattr(_gpu_cuda, "cupy", lambda: SimpleNamespace(cuda=SimpleNamespace(runtime=SimpleNamespace(getDeviceCount=lambda: 1))))

    def fail_context(device: int) -> None:
        raise OSError(f"context denied for device {device}")

    monkeypatch.setattr(_gpu_cuda, "library", fail_context)
    capability = ImageGpu.capabilities("cuda")
    assert capability["available"] is False
    assert "context denied" in str(capability["diagnostic"])
    with pytest.raises(RuntimeError, match="context denied"):
        ImageGpu.from_image(Image.new("L", (1, 1)), backend="cuda")


@pytest.mark.parametrize("backend,device", [("metal", 1), ("cuda", -1), ("cuda", True)])
def test_invalid_gpu_device_is_a_value_error(backend: str, device: int) -> None:
    with pytest.raises(ValueError, match="device index"):
        ImageGpu.capabilities(backend, device=device)
