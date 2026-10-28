from __future__ import annotations

import struct

import pytest
from PIL import Image as PillowImage

from blanket import Image


@pytest.mark.parametrize("mode,rawmode", [("RGB", "RGB"), ("RGB", "BGR"), ("RGB", "BGRX"), ("RGBA", "RGBA"), ("RGBA", "BGRA")])
@pytest.mark.parametrize("width", [1, 3, 15, 16, 17, 33])
@pytest.mark.parametrize("padding", [0, 1, 5])
@pytest.mark.parametrize("orientation", [1, -1])
def test_raw_buffers_match_pillow(mode, rawmode, width, padding, orientation):
    size = (width, 3)
    raw = bytes((i * 37 + i // 7) % 256 for i in range(width * 3 * len(mode)))
    actual = Image.frombytes(mode, size, raw)
    expected = PillowImage.frombytes(mode, size, raw)
    stride = width * len(rawmode) + padding
    args = (rawmode, stride, orientation)
    encoded = actual.tobytes("raw", *args)
    assert encoded == expected.tobytes("raw", *args)
    assert actual.tobytes("raw", args) == encoded
    assert Image.frombytes(mode, size, encoded, "raw", *args).tobytes() == raw
    assert Image.frombytes(mode, size, encoded, "raw", args).tobytes() == raw
    assert PillowImage.frombytes(mode, size, encoded, "raw", *args).tobytes() == raw
    assert actual.tobytes() == raw


@pytest.mark.parametrize("mode", ["1", "L", "LA", "P", "PA", "HSV", "CMYK", "YCbCr", "LAB", "I", "I;16", "I;16L", "I;16B", "F"])
@pytest.mark.parametrize("orientation", [1, -1])
def test_native_raw_modes_preserve_stride_and_orientation(mode, orientation):
    reference = PillowImage.new(mode, (9, 3))
    channels = len(reference.getbands())
    reference.putdata([i * 7 if channels == 1 else tuple((i * 7 + c * 31) % 256 for c in range(channels)) for i in range(27)])
    raw = reference.tobytes()
    actual = Image.frombytes(mode, reference.size, raw)
    stride = len(raw) // 3 + 3
    args = (mode, stride, orientation)
    encoded = actual.tobytes("raw", *args)
    assert encoded == reference.tobytes("raw", *args)
    assert Image.frombytes(mode, actual.size, encoded, "raw", *args).tobytes() == raw
    assert PillowImage.frombytes(mode, actual.size, encoded, "raw", *args).tobytes() == raw


@pytest.mark.parametrize("mode,channels,depth", [("L", 1, 10), ("RGB", 3, 12), ("RGBA", 4, 16), ("RGB;F", 3, 32), ("RGBA;F", 4, 32)])
def test_wide_raw_samples_are_not_reinterpreted(mode, channels, depth):
    values = [i / 4 - 3 if depth == 32 else (i * 71) % (1 << depth) for i in range(2 * 3 * channels)]
    raw = struct.pack(f"<{len(values)}{'f' if depth == 32 else 'H'}", *values)
    image = Image.frombytes(mode, (3, 2), raw, bit_depth=depth)
    row = len(raw) // 2
    stride = row + 5
    encoded = image.tobytes("raw", mode, stride, -1)
    assert encoded == raw[row:] + bytes(5) + raw[:row] + bytes(5)
    restored = Image.frombytes(mode, image.size, encoded, "raw", mode, stride, -1, bit_depth=depth)
    assert restored.tobytes() == raw
    assert restored.bit_depth == depth


def test_raw_input_ignores_padding_and_does_not_require_final_padding():
    data = bytearray([1, 2, 3, 99, 4, 5, 6])
    image = Image.frombytes("RGB", (1, 2), data, "raw", "BGR", 4)
    assert image.tobytes() == bytes([3, 2, 1, 6, 5, 4])
    data[:] = bytes(7)
    assert image.tobytes() == bytes([3, 2, 1, 6, 5, 4])
    with pytest.raises(ValueError, match="not enough image data"):
        Image.frombytes("RGB", (1, 2), bytes(6), "raw", "BGR", 4)


@pytest.mark.parametrize(
    "args,error",
    [
        (("RGB", -1), ValueError),
        (("RGB", 5), ValueError),
        (("RGB", 0, 0), ValueError),
        (("RGB", 0, 2), ValueError),
        (("RGB", 0, -2), ValueError),
        (("RGB", 1.5), TypeError),
        (("RGB", 0, 1.5), TypeError),
        ((None,), TypeError),
        (("unsupported",), ValueError),
        (("BGRA",), ValueError),
        (("RGB", 0, 1, 0), TypeError),
    ],
)
def test_invalid_raw_layouts_fail(args, error):
    image = Image.new("RGB", (2, 2))
    with pytest.raises(error):
        image.tobytes("raw", *args)
    with pytest.raises(error):
        Image.frombytes("RGB", (2, 2), bytes(12), "raw", *args)


@pytest.mark.parametrize("codec", ["png", "jpeg", "hex", ""])
def test_raw_methods_reject_other_codecs(codec):
    with pytest.raises(ValueError, match="only the raw"):
        Image.new("L", (1, 1)).tobytes(codec)
    with pytest.raises(ValueError, match="only the raw"):
        Image.frombytes("L", (1, 1), b"\0", codec)


def test_raw_rejects_size_overflow_and_closed_images():
    image = Image.new("RGB", (2, 2))
    with pytest.raises(ValueError, match="overflow"):
        image.tobytes("raw", "RGB", (1 << 64) - 1)
    image.close()
    with pytest.raises(ValueError, match="closed"):
        image.tobytes("raw", "BGR", 8, -1)


def test_bgr_layout_requires_8_bit_samples():
    image = Image.frombytes("RGB", (1, 1), bytes(6), bit_depth=16)
    with pytest.raises(ValueError, match="unsupported raw mode"):
        image.tobytes("raw", "BGR")
    with pytest.raises(ValueError, match="unsupported raw mode"):
        Image.frombytes("RGB", (1, 1), bytes(6), "raw", "BGR", bit_depth=16)
