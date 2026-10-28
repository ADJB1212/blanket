from __future__ import annotations

import struct
from io import BytesIO

import numpy as np
import pytest
from PIL import Image as PillowImage

from blanket import Image


@pytest.mark.parametrize("mode", ["1", "L", "LA", "P", "PA", "RGB", "RGBA", "HSV", "CMYK", "YCbCr", "LAB", "I", "I;16", "I;16L", "I;16B", "F"])
def test_array_conversion_matches_pillow(mode):
    expected = PillowImage.new(mode, (9, 3))
    channels = len(expected.getbands())
    values = [i * 9 % 256 if channels == 1 else tuple((i * 9 + c * 47) % 256 for c in range(channels)) for i in range(27)]
    if mode == "I":
        values = [-(1 << 20), 1 << 20, -1] * 9
    elif mode == "F":
        values = [-1.5, float("nan"), float("inf")] * 9
    elif mode.startswith("I;16"):
        values = [0, 32768, 65535] * 9
    expected.putdata(values)
    actual = Image.frombytes(mode, expected.size, expected.tobytes())
    result = np.asarray(actual)
    reference = np.asarray(expected)
    assert result.shape == reference.shape
    assert result.dtype == reference.dtype
    np.testing.assert_array_equal(result, reference)
    assert not result.flags.writeable


@pytest.mark.parametrize("mode,channels", [("L", 1), ("RGB", 3), ("RGBA", 4)])
@pytest.mark.parametrize("depth", [10, 12, 16])
def test_high_depth_array_conversion_retains_samples(mode, channels, depth):
    maximum = (1 << depth) - 1
    values = np.arange(6 * channels, dtype="<u2").reshape((2, 3) if channels == 1 else (2, 3, channels))
    values.flat[-1] = maximum
    image = Image.frombytes(mode, (3, 2), values.tobytes(), bit_depth=depth)
    result = np.asarray(image)
    assert result.dtype == np.dtype("<u2")
    np.testing.assert_array_equal(result, values)


@pytest.mark.parametrize("mode,channels", [("RGB;F", 3), ("RGBA;F", 4)])
def test_float_color_array_conversion(mode, channels):
    values = np.linspace(-2, 3, 6 * channels, dtype="<f4").reshape(2, 3, channels)
    values[0, 0, :3] = [float("nan"), float("inf"), -float("inf")]
    image = Image.frombytes(mode, (3, 2), values.tobytes(), bit_depth=32)
    result = np.asarray(image)
    assert result.dtype == np.dtype("<f4")
    np.testing.assert_array_equal(result, values)


def test_array_snapshots_survive_mutation_and_close():
    image = Image.new("RGB", (3, 2), (10, 20, 30))
    snapshot = np.asarray(image)
    copied = np.array(image)
    converted = np.asarray(image, dtype=np.float32)
    assert copied.flags.writeable
    assert converted.dtype == np.float32
    copied[0, 0] = 0
    image.load()[0, 0] = (40, 50, 60)
    np.testing.assert_array_equal(np.asarray(image)[0, 0], [40, 50, 60])
    image.close()
    np.testing.assert_array_equal(snapshot, np.full((2, 3, 3), [10, 20, 30], dtype=np.uint8))
    np.testing.assert_array_equal(converted, snapshot)
    with pytest.raises(ValueError, match="closed"):
        np.asarray(image)


@pytest.mark.parametrize("mode", ["1", "RGB", "I;16B", "RGB;F"])
@pytest.mark.parametrize("size", [(0, 3), (3, 0)])
def test_empty_array_conversion(mode, size):
    image = Image.frombytes(mode, size, b"", bit_depth=32 if mode == "RGB;F" else 8)
    result = np.asarray(image)
    shape = (size[1], size[0], 3) if mode in ("RGB", "RGB;F") else (size[1], size[0])
    assert result.shape == shape
    assert result.size == 0


def test_array_interface_exposes_immutable_bytes():
    image = Image.frombytes("I;16B", (2, 1), struct.pack(">2H", 1, 65535))
    interface = image.__array_interface__
    assert interface == {"version": 3, "shape": (1, 2), "typestr": ">u2", "data": b"\x00\x01\xff\xff"}


@pytest.mark.parametrize("lazy", [False, True])
def test_array_conversion_uses_current_frame(lazy):
    stream = BytesIO()
    first = PillowImage.new("RGB", (3, 2), "red")
    first.save(stream, "TIFF", save_all=True, append_images=[PillowImage.new("L", (2, 4), 42)])
    image = Image.open(stream, lazy=lazy)
    first_array = np.asarray(image)
    image.seek(1)
    second_array = np.asarray(image)
    assert second_array.shape == (4, 2)
    np.testing.assert_array_equal(second_array, np.full((4, 2), 42, dtype=np.uint8))
    image.close()
    np.testing.assert_array_equal(first_array, np.full((2, 3, 3), [255, 0, 0], dtype=np.uint8))
