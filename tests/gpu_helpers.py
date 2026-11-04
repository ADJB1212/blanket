"""GPU capability checks and runtime fixtures shared by numeric contracts."""

from __future__ import annotations

import pytest

from blanket import ImageGpu

MODES = ("L", "LA", "RGB", "RGBA", "F", "RGB;F", "RGBA;F")
CHANNELS = dict(zip(MODES, (1, 2, 3, 4, 1, 3, 4), strict=True))


def require_metal() -> None:
    if not ImageGpu.capabilities()["available"]:
        pytest.skip("Metal feature and device required")


def require_mps(*, shader: bool = False) -> None:
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("MPS required")
    if shader and not hasattr(torch.mps, "compile_shader"):
        pytest.skip("PyTorch MPS shader support required")


@pytest.fixture(params=("native", "torch"))
def runtime(request: pytest.FixtureRequest) -> str:
    require_metal()
    if request.param == "torch":
        require_mps()
    return request.param


@pytest.fixture(params=("native", "torch"))
def shader_runtime(request: pytest.FixtureRequest) -> str:
    require_metal()
    if request.param == "torch":
        require_mps(shader=True)
    return request.param
