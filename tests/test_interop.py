"""Pillow sharing, native capsule lifetimes, and half-precision exchange."""

from __future__ import annotations

import builtins
import ctypes
import gc
from io import BytesIO

import numpy as np
import pytest
from helpers import image_buffer
from PIL import Image as PillowImage

from blanket import Image


@pytest.mark.parametrize("mode", ["1", "L", "LA", "La", "P", "PA", "RGB", "RGBA", "RGBa", "RGBX", "CMYK", "YCbCr", "LAB", "HSV", "I", "F", "I;16", "I;16L", "I;16B", "I;16N"])
@pytest.mark.parametrize("copy", [False, True])
def test_from_pillow_modes(mode, copy):
    source = PillowImage.new(mode, (7, 3))
    if mode in ("P", "PA"):
        source.putpalette(bytes(range(256)) * 3)
    source.info["icc_profile"] = b"profile"
    expected = source.tobytes()
    result = Image.from_pillow(source, copy=copy)
    assert (result.mode, result.size) == (source.mode, source.size)
    assert result.tobytes() == expected
    if source.palette is not None:
        assert result.palette.tobytes() == source.palette.tobytes()
    assert result.info == source.info
    assert result.info is not source.info
    source.close()
    del source
    gc.collect()
    assert result.tobytes() == expected


@pytest.mark.parametrize("mode,color,new_color", [("L", 41, 99), ("RGBA", (1, 2, 3, 4), (5, 6, 7, 8)), ("F", 1.5, -2.5), ("I;16", 1234, 4321)])
def test_from_pillow_shares_and_detaches(mode, color, new_color):
    source = PillowImage.new(mode, (8, 2), color)
    result = Image.from_pillow(source)
    assert result.readonly == 1
    view = result.buffer_view()
    source.putpixel((0, 0), new_color)
    assert result.getpixel((0, 0)) == new_color
    result.putpixel((0, 0), color)
    assert source.getpixel((0, 0)) == new_color
    assert result.getpixel((0, 0)) == color
    source.close()
    assert len(view) == 8 * 2 * (4 if mode in ("RGBA", "F") else 2 if mode == "I;16" else 1)


def test_from_pillow_copy_and_stride_fallback():
    data = bytearray(range(24))
    source = PillowImage.frombuffer("L", (3, 3), data, "raw", "L", 8, -1)
    result = Image.from_pillow(source)
    copied = Image.from_pillow(source, copy=True)
    expected = source.tobytes()
    data[:] = bytes(24)
    assert result.tobytes() == copied.tobytes() == expected
    assert result.readonly == 0


def test_from_pillow_palette_without_pixel_copy(monkeypatch):
    from blanket._buffer import BufferMap

    def unexpected_snapshot(*args):
        raise AssertionError("unexpected pixel copy")

    source = PillowImage.new("P", (8, 2), 1)
    source.putpalette(bytes(range(256)) * 3)
    with monkeypatch.context() as patch:
        patch.setattr(BufferMap, "snapshot", unexpected_snapshot)
        result = Image.from_pillow(source)
    assert result.readonly == 1
    assert result.getbands() == ("P",)
    assert result.convert("RGB").tobytes() == source.convert("RGB").tobytes()


def test_from_pillow_current_frame():
    stream = image_buffer(PillowImage.new("RGB", (4, 2), "red"), "TIFF", save_all=True, append_images=[PillowImage.new("RGB", (3, 5), "blue")])
    with PillowImage.open(stream) as source:
        source.seek(1)
        result = Image.from_pillow(source)
        assert result.tobytes() == source.tobytes()
        assert source.tell() == 1
    assert result.size == (3, 5)
    assert result.n_frames == 1


def test_from_pillow_validation():
    with pytest.raises(TypeError, match="Pillow Image"):
        Image.from_pillow(object())
    with pytest.raises(TypeError, match="boolean"):
        Image.from_pillow(PillowImage.new("L", (1, 1)), copy=1)


def test_from_pillow_without_pillow(monkeypatch):
    original = builtins.__import__

    def restricted(name, *args, **kwargs):
        if name == "PIL":
            raise ImportError("Pillow unavailable")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", restricted)
    with pytest.raises(ImportError, match="Pillow is required"):
        Image.from_pillow(object())


class Descriptor(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("width", ctypes.c_uint32),
        ("height", ctypes.c_uint32),
        ("bit_depth", ctypes.c_uint32),
        ("channels", ctypes.c_uint32),
        ("data", ctypes.c_void_p),
        ("len", ctypes.c_size_t),
        ("row_stride", ctypes.c_size_t),
        ("mode", ctypes.c_char_p),
        ("palette", ctypes.c_void_p),
        ("palette_len", ctypes.c_size_t),
        ("palette_channels", ctypes.c_uint32),
    ]


def descriptor(capsule):
    get_pointer = ctypes.pythonapi.PyCapsule_GetPointer
    get_pointer.restype = ctypes.c_void_p
    get_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
    return Descriptor.from_address(get_pointer(capsule, b"blanket.image.v1"))


@pytest.mark.parametrize("mode,depth", [("1", 8), ("L", 8), ("P", 8), ("PA", 8), ("RGB", 8), ("RGB", 16), ("LAB", 8), ("I;16B", 16), ("F", 32), ("RGBA;F", 32)])
@pytest.mark.parametrize("copy", [False, True])
def test_capsule_lifetime_and_layout(mode, depth, copy):
    image = Image.fromarray(np.zeros((3, 9, 3), dtype=np.uint16), bit_depth=depth) if mode == "RGB" and depth == 16 else Image.new(mode, (9, 3))
    capsule = image.to_capsule(copy=copy)
    info = descriptor(capsule)
    assert info.abi_version == 1
    assert (info.width, info.height, info.bit_depth) == (9, 3, depth)
    assert info.channels == len(image.getbands())
    assert info.len == info.row_stride * info.height
    assert info.mode.decode() == ("L" if mode == "P" else mode)
    if mode in ("P", "PA"):
        assert info.palette_channels == 3
        assert ctypes.string_at(info.palette, info.palette_len) == image.palette.tobytes()
    expected = ctypes.string_at(info.data, info.len)
    image.putpixel((0, 0), 1 if info.channels == 1 else (1,) * info.channels)
    image.close()
    del image
    gc.collect()
    assert ctypes.string_at(info.data, info.len) == expected


def test_capsule_copy_out_and_mapped_snapshot():
    values = np.arange(12, dtype=np.uint8).reshape(3, 4)
    image = Image.fromarray(values, copy=False)
    capsule = image.to_capsule()
    info = descriptor(capsule)
    values[:] = 0
    restored = Image.frombytes("L", (info.width, info.height), ctypes.string_at(info.data, info.len))
    assert restored.tobytes() == bytes(range(12))
    with pytest.raises(TypeError, match="boolean"):
        image.to_capsule(copy=1)
    image.close()
    with pytest.raises(ValueError, match="closed"):
        image.to_capsule()


def test_capsule_storage_sharing():
    image = Image.new("RGB", (3, 2), (11, 22, 33))
    shared = image.to_capsule()
    second = image.to_capsule()
    copied = image.to_capsule(copy=True)
    assert descriptor(shared).data == descriptor(second).data
    assert descriptor(shared).data != descriptor(copied).data
    assert ctypes.string_at(descriptor(copied).data, 18) == image.tobytes()


@pytest.mark.parametrize("channels,mode", [(1, "F"), (3, "RGB;F"), (4, "RGBA;F")])
@pytest.mark.parametrize("dtype", ["<f2", ">f2"])
@pytest.mark.parametrize("strided", [False, True])
def test_float16_array_exchange(channels, mode, dtype, strided):
    values = np.array([0.0, -0.0, 1.5, -2.25, 65504, 2**-24, np.inf, -np.inf, np.nan], dtype=dtype)
    values = np.tile(values, 4 * channels).reshape(6, 6, channels)
    if channels == 1:
        values = values[:, :, 0]
    if strided:
        values = values[::-1, ::2]
    image = Image.fromarray(values)
    assert (image.mode, image.bit_depth) == (mode, 32)
    np.testing.assert_equal(np.asarray(image), values.astype(np.float32))
    result = image.to_numpy(dtype="float16", layout="HW" if channels == 1 else "HWC")
    assert result.dtype == np.float16
    assert result.flags.writeable
    np.testing.assert_equal(result, values)
    np.testing.assert_equal(np.signbit(result), np.signbit(values))
    result[:] = 0
    np.testing.assert_equal(np.asarray(image), values.astype(np.float32))
    with pytest.raises(ValueError, match="copy=False"):
        Image.fromarray(values, copy=False)


def test_float16_rounding_and_export_validation():
    values = np.array([[1 + 2**-11, 1 + 3 * 2**-11, 2**-25, -(2**-25), 65520]], dtype=np.float32)
    with np.errstate(over="ignore"):
        expected = values.astype(np.float16)
    result = Image.fromarray(values).to_numpy(dtype="float16", layout="HW")
    np.testing.assert_equal(result, expected)
    with pytest.raises(ValueError, match="dtype"):
        Image.new("RGB", (2, 2)).to_numpy(dtype="uint16")
    with pytest.raises(ValueError, match="bit_depth"):
        Image.fromarray(np.zeros((2, 2), dtype=np.float16), bit_depth=16)


@pytest.mark.parametrize("layout", ["HWC", "CHW"])
def test_float16_export_normalization(layout):
    image = Image.frombytes("RGB", (5, 3), bytes(range(45)))
    options = {"layout": layout, "mean": (0.1, 0.2, 0.3), "std": (0.5, 0.7, 0.9)}
    result = image.to_numpy(dtype="float16", **options)
    np.testing.assert_equal(result, image.to_numpy(**options).astype(np.float16))
    assert result.flags.c_contiguous and result.flags.writeable
    image.close()
    assert result.size == 45


def test_float16_all_encodings():
    values = np.arange(65536, dtype=np.uint16).view(np.float16).reshape(256, 256)
    result = Image.fromarray(values).to_numpy(dtype="float16", layout="HW")
    finite = np.isfinite(values)
    np.testing.assert_equal(result.view(np.uint16)[finite], values.view(np.uint16)[finite])
    np.testing.assert_equal(np.isnan(result), np.isnan(values))


def test_float16_import_without_buffer_or_numpy(monkeypatch):
    class HalfArray:
        @property
        def __array_interface__(self):
            return {"shape": (1, 2), "typestr": "<f2", "version": 3}

        def tobytes(self):
            return b"\x00\x3e\x00\xc0"

    original = builtins.__import__

    def restricted(name, *args, **kwargs):
        if name == "numpy" or name.startswith("numpy."):
            raise AssertionError("unexpected NumPy import")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", restricted)
    assert Image.fromarray(HalfArray()).getdata() == [1.5, -2.0]


@pytest.mark.parametrize("size", [(0, 2), (2, 0)])
def test_empty_interop(size):
    result = Image.from_pillow(PillowImage.new("L", size))
    assert result.size == size
    assert result.tobytes() == b""
    assert result.to_numpy(dtype="float16", layout="HW").shape == (size[1], size[0])
    capsule = result.to_capsule()
    assert descriptor(capsule).len == 0
