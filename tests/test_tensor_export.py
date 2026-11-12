from __future__ import annotations

import builtins
import gc
from io import BytesIO

import numpy as np
import pytest
from PIL import Image as PillowImage

from blanket import Image


@pytest.mark.parametrize("layout", ["HWC", "CHW"])
@pytest.mark.parametrize("channels,mode", [(1, "L"), (2, "LA"), (3, "RGB"), (4, "RGBA")])
@pytest.mark.parametrize("width", [0, 1, 15, 16, 17, 33, 16385])
@pytest.mark.parametrize("normalize", [False, True])
def test_fused_export(layout, channels, mode, width, normalize):
    source = (np.arange(2 * width * channels) * 31 % 256).astype(np.uint8).reshape(2, width, channels)
    image = Image.frombytes(mode, (width, 2), source.tobytes())
    mean = np.linspace(-0.5, 0.5, channels, dtype=np.float32)
    std = np.linspace(0.25, 2, channels, dtype=np.float32)
    expected = source.astype(np.float32)
    if normalize:
        expected /= 255
    expected = (expected - mean) / std
    if layout == "CHW":
        expected = expected.transpose(2, 0, 1)
    actual = image.to_numpy(layout=layout, normalize=normalize, mean=mean, std=std)
    assert actual.dtype == np.float32
    assert actual.flags.c_contiguous and actual.flags.writeable
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("mode,channels", [("L", 1), ("RGB", 3), ("RGBA", 4)])
@pytest.mark.parametrize("depth", [10, 12, 16])
@pytest.mark.parametrize("layout", ["HWC", "CHW"])
def test_unsigned_depth(mode, channels, depth, layout):
    maximum = (1 << depth) - 1
    source = np.linspace(0, maximum, 17 * channels, dtype="<u2").reshape(1, 17, channels)
    image = Image.frombytes(mode, (17, 1), source.tobytes(), bit_depth=depth)
    expected = source.astype(np.float32) / maximum
    if layout == "CHW":
        expected = expected.transpose(2, 0, 1)
    np.testing.assert_array_equal(image.to_numpy(layout=layout), expected)


@pytest.mark.parametrize("mode,dtype", [("I;16", "<u2"), ("I;16L", "<u2"), ("I;16B", ">u2"), ("I;16N", "=u2"), ("I", "<i4"), ("F", "<f4"), ("RGB;F", "<f4"), ("RGBA;F", "<f4")])
def test_wide_samples(mode, dtype):
    channels = 4 if mode == "RGBA;F" else 3 if mode == "RGB;F" else 1
    source = np.arange(33 * channels, dtype=np.float32).reshape(1, 33, channels)
    if mode == "I":
        source -= 100
    if "F" in mode:
        source -= 2
        source.flat[:3] = [np.nan, np.inf, -np.inf]
    source = source.astype(dtype)
    image = Image.frombytes(mode, (33, 1), source.tobytes())
    expected = source.astype(np.float32)
    if mode.startswith("I;16"):
        expected /= 65535
    np.testing.assert_array_equal(image.to_numpy(), expected)
    np.testing.assert_array_equal(image.to_numpy(normalize=False), source.astype(np.float32))


@pytest.mark.parametrize("mode", ["1", "P", "PA", "La", "RGBa", "RGBX", "CMYK", "HSV", "YCbCr", "LAB"])
def test_other_bands(mode):
    image = Image.new(mode, (17, 2))
    image.putdata([i % 2 if mode == "1" else i * 19 % 256 if mode == "P" else tuple((i * 19 + c * 53) % 256 for c in range(len(image.getbands()))) for i in range(34)])
    expected = np.asarray(image).astype(np.float32)
    if mode != "1":
        expected /= 255
    if expected.ndim == 2:
        expected = expected[..., None]
    np.testing.assert_array_equal(image.to_numpy(), expected)


@pytest.mark.parametrize("export", ["to_numpy", "to_torch"])
def test_ownership(export):
    image = Image.new("RGB", (3, 2), (10, 20, 30))
    result = getattr(image, export)(layout="HWC", normalize=False)
    result[0, 0, 0] = 100
    assert image.getpixel((0, 0)) == (10, 20, 30)
    other = getattr(image, export)(layout="HWC", normalize=False)
    image.putpixel((0, 0), (40, 50, 60))
    image.close()
    del image
    gc.collect()
    assert result[0, 0, 0] == 100
    assert other[0, 0, 0] == 10


@pytest.mark.parametrize("layout", ["HW", "HWC", "CHW"])
@pytest.mark.parametrize("size", [(17, 2), (0, 2), (2, 0)])
def test_torch_and_single_band(layout, size):
    import torch

    image = Image.frombytes("L", size, bytes(range(size[0] * size[1])))
    expected = image.to_numpy(layout=layout, mean=[0.5], std=[0.25])
    result = image.to_torch(layout=layout, mean=[0.5], std=[0.25])
    assert result.dtype == torch.float32 and result.device.type == "cpu"
    assert result.is_contiguous()
    np.testing.assert_array_equal(result.numpy(), expected)


def test_defaults_and_torch_without_numpy(monkeypatch):
    import torch

    original = builtins.__import__

    def restricted(name, *args, **kwargs):
        if name == "numpy" or name.startswith("numpy."):
            raise AssertionError("unexpected NumPy import")
        return original(name, *args, **kwargs)

    image = Image.new("RGB", (2, 1), (255, 128, 0))
    assert image.to_numpy().shape == (1, 2, 3)
    with monkeypatch.context() as patch:
        patch.setattr(builtins, "__import__", restricted)
        result = image.to_torch()
        assert result[:, 0, 0].tolist() == pytest.approx([1, 128 / 255, 0])
    assert result.shape == (3, 1, 2)


@pytest.mark.parametrize("normalize", [False, True])
def test_bilevel_nonzero_samples(normalize):
    image = Image.new("1", (5, 1))
    image.putdata([0, 1, 2, 128, 255])
    result = image.to_numpy(layout="HW", normalize=normalize)
    np.testing.assert_array_equal(result, [[0, *([1 if normalize else 255] * 4)]])


@pytest.mark.parametrize("mode,dtype", [("L", "u1"), ("I;16L", "<u2"), ("I;16B", ">u2")])
def test_unaligned_mapped_input(mode, dtype):
    values = np.arange(33, dtype=dtype).reshape(1, 33)
    storage = bytearray(b"x" + values.tobytes())
    image = Image.frombuffer(mode, (33, 1), memoryview(storage)[1:], "raw", mode, 0, 1)
    result = image.to_numpy(layout="HW", normalize=False)
    np.testing.assert_array_equal(result, values.astype(np.float32))
    storage[1:] = bytes(values.nbytes)
    image.close()
    np.testing.assert_array_equal(result, values.astype(np.float32))


@pytest.mark.parametrize("kwargs", [{"layout": "bad"}, {"layout": "HW"}, {"mean": []}, {"mean": [0]}, {"std": [1]}, {"mean": [0, np.nan, 0]}, {"std": [1, np.inf, 1]}, {"std": [1, 0, 1]}, {"std": [1, -1, 1]}, {"mean": [0, 1e100, 0]}])
@pytest.mark.parametrize("export", ["to_numpy", "to_torch"])
def test_invalid_parameters(kwargs, export):
    with pytest.raises(ValueError):
        getattr(Image.new("RGB", (1, 1)), export)(**kwargs)


@pytest.mark.parametrize("export", ["to_numpy", "to_torch"])
def test_closed(export):
    image = Image.new("L", (1, 1))
    image.close()
    with pytest.raises(ValueError, match="closed"):
        getattr(image, export)()


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("export", ["to_numpy", "to_torch"])
def test_current_frame(lazy, export):
    stream = BytesIO()
    PillowImage.new("RGB", (3, 2), "red").save(stream, "TIFF", save_all=True, append_images=[PillowImage.new("L", (2, 4), 42)])
    image = Image.open(stream, lazy=lazy)
    first = getattr(image, export)(layout="CHW")
    image.seek(1)
    second = getattr(image, export)(layout="HW", normalize=False)
    image.close()
    assert first.shape == (3, 2, 3)
    assert second.shape == (4, 2)
    assert first[0, 0, 0] == 1
    assert second[0, 0] == 42
