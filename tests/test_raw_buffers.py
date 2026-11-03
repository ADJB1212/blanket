from __future__ import annotations

import struct

import pytest
from PIL import Image as PillowImage

from blanket import Image


@pytest.mark.parametrize("width", [1, 7, 8, 9, 16, 17, 33])
@pytest.mark.parametrize(
    "mode,rawmode",
    [
        ("1", "1;IR"),
        ("1", "L"),
        ("L", "L;16"),
        ("L", "L;16B"),
        ("LA", "LA;L"),
        ("P", "P;1"),
        ("P", "P;2"),
        ("P", "P;4"),
        ("PA", "PA;L"),
        ("RGB", "RGBX"),
        ("RGB", "XRGB"),
        ("RGB", "XBGR"),
        ("RGB", "RGB;L"),
        ("RGB", "G"),
        ("RGBA", "RGBA;L"),
        ("RGBA", "RGB"),
        ("RGBA", "ABGR"),
        ("RGBA", "BGRa"),
        ("RGBa", "BGRa"),
        ("RGBa", "aBGR"),
        ("RGBX", "RGBX;L"),
        ("RGBX", "BGRX"),
        ("CMYK", "CMYK;I"),
        ("CMYK", "CMYK;L"),
        ("YCbCr", "YCbCrX"),
        ("YCbCr", "YCbCrK"),
        ("LAB", "A"),
        ("HSV", "V"),
        ("I", "I;16B"),
        ("I", "I;32NS"),
        ("F", "F;32NF"),
    ],
)
def test_extended_raw_encoders_preserve_rows(mode: str, rawmode: str, width: int) -> None:
    expected = PillowImage.new(mode, (width, 3))
    channels = len(expected.getbands())
    expected.putdata([i * 7 if channels == 1 else tuple((i * 7 + c * 31) % 256 for c in range(channels)) for i in range(width * 3)])
    actual = Image.frombytes(mode, expected.size, expected.tobytes())
    row = len(expected.tobytes("raw", rawmode)) // 3
    for orientation in (1, -1):
        assert actual.tobytes("raw", rawmode, row + 3, orientation) == expected.tobytes("raw", rawmode, row + 3, orientation)


@pytest.mark.parametrize("width", [1, 7, 8, 9, 16, 17, 33])
@pytest.mark.parametrize(
    "mode,rawmode,bits",
    [
        ("1", "1;8", 8),
        ("L", "L;2IR", 2),
        ("L", "L;4R", 4),
        ("P", "P;2L", 2),
        ("P", "P;4L", 4),
        ("RGB", "RGB;15", 16),
        ("RGB", "BGR;16", 16),
        ("RGB", "RGB;4B", 16),
        ("RGB", "YCC;P", 24),
        ("RGB", "CMYK", 32),
        ("RGBA", "LA;16B", 32),
        ("RGBA", "YCCA;P", 32),
        ("RGBA", "RGBa;16L", 64),
        ("RGBX", "RGBXXX", 48),
        ("RGBX", "XRGB", 32),
        ("LAB", "B", 8),
        ("I", "I;8S", 8),
        ("I", "I;16BS", 16),
        ("F", "F;32BS", 32),
        ("F", "F;64BF", 64),
        ("I;16", "I;12", 12),
    ],
)
def test_extended_raw_decoders_clip_to_rows(mode: str, rawmode: str, bits: int, width: int) -> None:
    row = ((width + 7) // 8 * bits) if rawmode in ("P;2L", "P;4L") else (width * bits + 7) // 8
    stride = row + 3
    raw = bytes((i * 37 + i // 7) % 256 for i in range(stride * 3 - 3))
    for orientation in (1, -1):
        expected = PillowImage.frombytes(mode, (width, 3), raw, "raw", rawmode, stride, orientation)
        actual = Image.frombytes(mode, expected.size, raw, "raw", rawmode, stride, orientation)
        assert actual.tobytes() == expected.tobytes()
        with pytest.raises(ValueError, match="not enough image data"):
            Image.frombytes(mode, expected.size, raw[:-1], "raw", rawmode, stride, orientation)


@pytest.mark.parametrize("decoder,data", [("hex", b"01 02\n03ff"), ("packbits", bytes([2, 1, 2, 3, 0, 255]))])
def test_raw_byte_decoders(decoder: str, data: bytes) -> None:
    expected = PillowImage.frombytes("L", (4, 1), data, decoder, "L")
    actual = Image.frombytes("L", expected.size, data, decoder, "L")
    assert actual.tobytes() == expected.tobytes()
    actual.frombytes(data, decoder, "L")
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("decoder,data", [("hex", b"010203"), ("packbits", bytes([2, 1, 2, 3]))])
def test_layered_raw_decoders_reject_truncated_planes(decoder: str, data: bytes) -> None:
    image = Image.new("P", (9, 1))
    with pytest.raises(ValueError, match="not enough image data"):
        image.frombytes(data, decoder, "P;2L")
    assert image.tobytes() == bytes(9)


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


@pytest.mark.parametrize("codec", ["png", "jpeg", ""])
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
