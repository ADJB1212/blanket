from __future__ import annotations

import gc
import struct
from typing import NoReturn

import pytest

from blanket import Image, ImageGpu, _blanket

torch = pytest.importorskip("torch")
MODES = {"L": 1, "LA": 2, "RGB": 3, "RGBA": 4, "F": 1, "RGB;F": 3, "RGBA;F": 4}


@pytest.fixture(autouse=True)
def require_mps() -> None:
    if not ImageGpu.capabilities()["available"] or not torch.backends.mps.is_available():
        pytest.skip("native Metal and PyTorch MPS required")


def unexpected_transfer(*args: object, **kwargs: object) -> NoReturn:
    raise AssertionError("Metal exchange must not stage pixels through the CPU")


@pytest.mark.parametrize(("mode", "layout"), [(mode, layout) for mode, channels in MODES.items() for layout in ("HWC", "CHW", "HW") if layout != "HW" or channels == 1])
def test_native_exchange_layouts_precision_and_independence(mode: str, layout: str, monkeypatch: pytest.MonkeyPatch) -> None:
    channels = MODES[mode]
    floating = "F" in mode
    dtype = torch.float32 if floating else torch.uint8
    original = torch.arange(7 * 19 * channels, dtype=torch.float32).reshape(7, 19, channels)
    original = (original / 127 - 1).to(dtype) if floating else original.to(dtype)
    expected = 1 - original if floating else 255 - original
    source = original.to("mps")
    if layout == "CHW":
        source = source.permute(2, 0, 1)
    elif layout == "HW":
        source = source.squeeze(-1)
    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "cpu", unexpected_transfer)
        patch.setattr(torch.Tensor, "numpy", unexpected_transfer)
        patch.setattr(torch.Tensor, "tolist", unexpected_transfer)
        patch.setattr(Image.Image, "tobytes", unexpected_transfer)
        patch.setattr(ImageGpu.GpuImage, "to_image", unexpected_transfer)
        native = ImageGpu.from_torch(source, layout=layout, runtime="native")
        assert native.runtime == "native" and native.mode == mode
        result = native.invert()
        output = result.to_torch(layout=layout)
        restored = ImageGpu.from_torch(output, layout=layout, runtime="native")
        second_export = result.to_torch()
    assert second_export.data_ptr() != output.data_ptr()
    source.zero_()
    native.close()
    result.close()
    del source
    gc.collect()
    actual = output.permute(1, 2, 0) if layout == "CHW" else output.unsqueeze(-1) if layout == "HW" else output
    assert torch.equal(actual.cpu(), expected)
    output.zero_()
    assert torch.equal(restored.to_torch().cpu(), expected)
    assert torch.equal(second_export.cpu(), expected)


@pytest.mark.parametrize(("floating", "view"), [(floating, view) for floating in (False, True) for view in ("offset", "transpose", "slice", "expand", "negative") if view != "negative" or floating])
def test_native_import_views_and_storage_offsets(floating: bool, view: str) -> None:
    dtype = torch.float32 if floating else torch.uint8
    source = torch.arange(1024, dtype=torch.float32, device="mps").to(dtype)
    if view == "offset":
        source = source[1 : 1 + 7 * 19 * 3].reshape(7, 19, 3)
        assert source.storage_offset() == 1
    elif view == "transpose":
        source = source[: 7 * 19 * 3].reshape(19, 7, 3).transpose(0, 1)
    elif view == "slice":
        source = source[: 7 * 19 * 6].reshape(7, 19, 6)[..., ::2]
    elif view == "expand":
        source = source[:3].reshape(1, 1, 3).expand(7, 19, 3)
    else:
        source = torch._neg_view(source[: 7 * 19 * 3].reshape(7, 19, 3))
    expected = source.cpu().clone()
    native = ImageGpu.from_torch(source, layout="HWC", runtime="native")
    del source
    gc.collect()
    assert torch.equal(native.to_torch().cpu(), expected)


def test_native_exchange_preserves_nonfinite_sample_bits() -> None:
    bits = [0, 0x80000000, 0x7FC01234, 0x7F800000, 0xFF800000, 0xBF000000]
    data = struct.pack("6I", *bits)
    native = ImageGpu.from_image(Image.frombytes("F", (6, 1), data))
    tensor = native.to_torch(layout="HW")
    native.close()
    assert tensor.cpu().view(torch.uint8).numpy().tobytes() == data
    restored = ImageGpu.from_torch(tensor, layout="HW", runtime="native")
    tensor.zero_()
    assert restored.to_image().tobytes() == data


@pytest.mark.parametrize("depth", [10, 12, 16])
@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
def test_native_unsigned_exchange(depth: int, mode: str) -> None:
    channels = MODES[mode]
    maximum = (1 << depth) - 1
    samples = [0, 1, maximum // 2, maximum - 1, maximum] * channels
    data = struct.pack(f"{len(samples)}H", *samples)
    native = ImageGpu.from_image(Image.frombytes(mode, (5, 1), data, bit_depth=depth))
    working = native.to_torch()
    assert working.dtype == torch.float32
    assert torch.equal(working.cpu(), torch.tensor(samples, dtype=torch.float32).reshape(1, 5, channels) / maximum)
    tensor = torch.tensor(samples, dtype=torch.uint16).reshape(1, 5, channels).to("mps")
    restored = ImageGpu.from_torch(tensor, layout="HWC", bit_depth=depth, runtime="native")
    assert restored.runtime == "native" and restored.bit_depth == depth
    tensor.zero_()
    native.close()
    assert restored.to_image().tobytes() == data


def test_native_exchange_validation() -> None:
    caps = ImageGpu.capabilities()
    assert caps["native_torch_exchange"]
    assert caps["native_torch_exchange_policy"] == "synchronized GPU copy"
    native = ImageGpu.from_image(Image.new("RGB", (3, 7)))
    for layout in ("HW", "invalid"):
        with pytest.raises(ValueError, match="layout"):
            native.to_torch(layout=layout)
    native.close()
    with pytest.raises(ValueError, match="closed"):
        native.to_torch()
    with pytest.raises(ValueError, match="runtime"):
        ImageGpu.from_torch(torch.zeros((7, 3), device="mps"), layout="HW", runtime="invalid")
    for tensor in (torch.zeros((7, 3)), torch.zeros((7, 3), device="mps", requires_grad=True), torch.zeros((7, 3), dtype=torch.int32, device="mps")):
        with pytest.raises(ValueError):
            ImageGpu.from_torch(tensor, layout="HW", runtime="native")
    for tensor in (torch.zeros((1, 2, 3)), torch.zeros((1, 2, 3), device="mps", requires_grad=True), torch.zeros((1, 2, 3), dtype=torch.int32, device="mps")):
        with pytest.raises(ValueError):
            _blanket.MetalImage.from_torch(tensor)
    native = ImageGpu.from_image(Image.new("L", (3, 7)))
    with pytest.raises(ValueError, match="match"):
        native._resource.copy_to_torch(torch.empty((7, 3, 3), dtype=torch.uint8, device="mps"))


def test_native_export_to_offset_tensor_preserves_guards() -> None:
    image = Image.new("L", (7, 3), 57)
    native = ImageGpu.from_image(image).invert()
    storage = torch.full((27,), 99, dtype=torch.uint8, device="mps")
    destination = storage[1:22].reshape(3, 7, 1)
    native._resource.copy_to_torch(destination)
    assert storage.cpu().tolist() == [99] + [198] * 21 + [99] * 5


def test_native_exchange_survives_buffer_reuse() -> None:
    source = torch.full((17, 257, 4), 57, dtype=torch.uint8, device="mps")
    native = ImageGpu.from_torch(source, layout="HWC", runtime="native")
    tensor = native.invert().to_torch()
    native.close()
    del source
    gc.collect()
    for value in range(16):
        temporary = ImageGpu.from_image(Image.new("RGBA", (257, 17), (value,) * 4)).invert()
        temporary.close()
        torch.full((17, 257, 4), value, dtype=torch.uint8, device="mps")
    assert torch.equal(tensor.cpu(), torch.full((17, 257, 4), 198, dtype=torch.uint8))


def test_native_exchange_orders_mps_streams() -> None:
    if not hasattr(torch.mps, "Stream"):
        pytest.skip("PyTorch MPS streams required")
    stream = torch.mps.Stream()
    with torch.mps.stream(stream):
        source = torch.arange(257 * 17, dtype=torch.float32, device="mps").reshape(17, 257)
        native = ImageGpu.from_torch(source * 0.5, layout="HW", runtime="native")
    with torch.mps.stream(torch.mps.Stream()):
        actual = (native.invert().to_torch(layout="HW") * 2).cpu()
    expected = 2 - torch.arange(257 * 17, dtype=torch.float32).reshape(17, 257)
    assert torch.equal(actual, expected)
