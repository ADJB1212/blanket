"""Opt-in views share storage with explicit lifetime and write rules."""

from __future__ import annotations

import builtins
import gc
import weakref
from io import BytesIO

import numpy as np
import pytest
from PIL import Image as PillowImage

from blanket import Image


@pytest.mark.parametrize(
    "mode,dtype,channels,depth",
    [
        ("L", "u1", 1, 8),
        ("LA", "u1", 2, 8),
        ("RGB", "u1", 3, 8),
        ("RGBA", "u1", 4, 8),
        ("P", "u1", 1, 8),
        ("PA", "u1", 2, 8),
        ("I", "<i4", 1, 32),
        ("I;16B", ">u2", 1, 16),
        ("RGB", "<u2", 3, 10),
        ("RGBA", "<u2", 4, 16),
        ("F", "<f4", 1, 32),
        ("RGB;F", "<f4", 3, 32),
        ("RGBA;F", "<f4", 4, 32),
    ],
)
def test_owned_views_share_allocation_and_preserve_layout(mode, dtype, channels, depth):
    shape = (3, 5) if channels == 1 else (3, 5, channels)
    values = np.arange(15 * channels, dtype=dtype).reshape(shape)
    source = Image.frombytes(mode, (5, 3), values.tobytes(), bit_depth=depth)
    raw = source.buffer_view()
    first = source.array_view()
    second = source.array_view()
    assert raw.readonly and raw.format == "B" and raw.ndim == 1
    assert raw.tobytes() == values.tobytes()
    assert first.dtype == values.dtype
    np.testing.assert_array_equal(first, values)
    assert np.shares_memory(first, second)
    assert not first.flags.writeable
    with pytest.raises(ValueError):
        first.setflags(write=True)
    with pytest.raises(TypeError):
        raw[0] = 1
    source.close()
    raw.release()
    del source
    gc.collect()
    np.testing.assert_array_equal(first, values)


@pytest.mark.parametrize("mutation", ["putpixel", "paste", "frombytes", "pixel_access"])
def test_native_writes_detach_exported_allocation(mutation):
    source = Image.new("RGB", (3, 2), (10, 20, 30))
    first = source.array_view()
    before = first.copy()
    if mutation == "putpixel":
        source.putpixel((0, 0), (40, 50, 60))
    elif mutation == "paste":
        source.paste((40, 50, 60), (0, 0, 3, 2))
    elif mutation == "frombytes":
        source.frombytes(bytes([40, 50, 60] * 6))
    else:
        source.load()[0, 0] = (40, 50, 60)
    np.testing.assert_array_equal(first, before)
    assert not np.shares_memory(first, source.array_view())
    assert source.getpixel((0, 0)) == (40, 50, 60)


@pytest.mark.parametrize(
    "dtype,shape,mode,depth",
    [
        ("u1", (3, 5), "L", 8),
        ("u1", (3, 5, 2), "LA", 8),
        ("u1", (3, 5, 3), "RGB", 8),
        ("u1", (3, 5, 4), "RGBA", 8),
        ("<u2", (3, 5), "I;16L", 16),
        (">u2", (3, 5), "I;16B", 16),
        ("<u2", (3, 5, 3), "RGB", 16),
        ("<i4", (3, 5), "I", 32),
        ("<f4", (3, 5), "F", 32),
        ("<f4", (3, 5, 3), "RGB;F", 32),
        ("<f4", (3, 5, 4), "RGBA;F", 32),
    ],
)
def test_fromarray_without_copy_retains_source_changes_and_owner(dtype, shape, mode, depth):
    values = np.arange(np.prod(shape), dtype=dtype).reshape(shape).copy()
    reference = weakref.ref(values)
    source = Image.fromarray(values, copy=False)
    assert source.mode == mode
    assert source.bit_depth == depth
    assert source.readonly == 1
    exported = source.array_view()
    assert np.shares_memory(values, exported)
    with pytest.raises(ValueError):
        exported.setflags(write=True)
    values[...] = 7
    np.testing.assert_array_equal(exported, values)
    pixel = 7 if len(shape) == 2 else (7,) * shape[-1]
    assert source.getpixel((1, 1)) == pixel
    assert source.load()[1, 1] == pixel
    assert source.tobytes() == values.tobytes()
    del values
    gc.collect()
    assert reference() is not None
    source.close()
    np.testing.assert_array_equal(exported, np.full(shape, 7, dtype=dtype))
    del exported
    gc.collect()
    assert reference() is None
    with pytest.raises(ValueError, match="closed"):
        source.buffer_view()


def test_array_import_detaches_for_image_writes_and_closes():
    values = np.zeros((3, 5, 3), dtype="u1")
    source = Image.fromarray(values, copy=False)
    shared = source.array_view()
    source.putpixel((0, 0), (10, 20, 30))
    assert source.readonly == 0
    assert source.getpixel((0, 0)) == (10, 20, 30)
    assert values[0, 0].tolist() == [0, 0, 0]
    assert shared[0, 0].tolist() == [0, 0, 0]
    source.close()
    with pytest.raises(ValueError, match="closed"):
        source.getpixel((0, 0))


@pytest.mark.parametrize("readonly", [False, True])
def test_mapped_frombytes_write_rules(readonly):
    values = np.zeros((2, 3, 3), dtype="u1")
    if readonly:
        values.setflags(write=False)
    source = Image.fromarray(values, copy=False)
    exported = source.array_view()
    source.frombytes(bytes([7] * 18))
    assert source.tobytes() == bytes([7] * 18)
    assert np.all(values == (0 if readonly else 7))
    assert np.all(exported == (0 if readonly else 7))


@pytest.mark.parametrize("orientation", [1, -1])
def test_mapped_export_retains_padding_orientation_and_lifetime(orientation):
    data = bytearray(range(24))
    source = Image.frombuffer("RGBA", (2, 2), data, "raw", "RGBA", 12, orientation)
    view = source.array_view()
    assert view.strides == (12 * orientation, 4, 1)
    assert source.buffer_view().nbytes == 24
    assert not view.flags.writeable
    with pytest.raises(ValueError):
        view.setflags(write=True)
    np.testing.assert_array_equal(view, np.asarray(source))
    data[0] = 99
    assert view[0 if orientation == 1 else 1, 0, 0] == 99
    source.close()
    assert view[0 if orientation == 1 else 1, 0, 0] == 99


@pytest.mark.parametrize("depth", [10, 12, 16])
def test_high_depth_mapped_range_validation_and_detachment(depth):
    values = np.full((3, 5, 3), (1 << depth) - 1, dtype="<u2")
    source = Image.fromarray(values, bit_depth=depth, copy=False)
    assert source.bit_depth == depth
    assert source.getpixel((0, 0)) == ((1 << depth) - 1,) * 3
    assert source.copy().tobytes() == values.tobytes()
    if depth < 16:
        values[0, 0, 0] = 1 << depth
        with pytest.raises(ValueError, match="bit_depth"):
            source.getpixel((0, 0))
        with pytest.raises(ValueError):
            source.copy()
        values[0, 0, 0] = 0
    source.putpixel((0, 0), (1, 2, 3))
    assert values[0, 0].tolist() != [1, 2, 3]


@pytest.mark.parametrize("mode", ["1", "LAB"])
def test_incompatible_public_storage_requires_explicit_conversion(mode):
    source = Image.new(mode, (3, 2))
    for export in (source.buffer_view, source.array_view):
        with pytest.raises(ValueError, match="convert explicitly"):
            export()


@pytest.mark.parametrize(
    "values", [np.zeros((3, 4), dtype="f8"), np.zeros((3, 4), dtype=">f4"), np.zeros((3, 4), dtype="i2"), np.zeros((3, 4), dtype="u1")[:, ::2], np.zeros((3, 4), dtype="u1")[::-1], np.zeros((0, 4), dtype="u1"), np.zeros(4, dtype="u1")]
)
def test_copy_false_rejects_incompatible_arrays(values):
    with pytest.raises(ValueError, match="copy=False"):
        Image.fromarray(values, copy=False)


@pytest.mark.parametrize("options", [{"mode": "RGB"}, {"bit_depth": 10}, {"copy": None}])
def test_array_option_validation(options):
    with pytest.raises((TypeError, ValueError)):
        Image.fromarray(np.zeros((3, 4), dtype="u1"), **({"copy": False} | options))


@pytest.mark.parametrize("lazy", [False, True])
def test_views_retain_current_frame_after_navigation_and_close(lazy):
    stream = BytesIO()
    first = PillowImage.new("RGB", (3, 2), "red")
    first.save(stream, "TIFF", save_all=True, append_images=[PillowImage.new("L", (2, 4), 42)])
    with Image.open(stream, lazy=lazy) as image:
        first_view = image.array_view()
        image.seek(1)
        second_view = image.array_view()
        assert image.tell() == 1
    np.testing.assert_array_equal(first_view, np.full((2, 3, 3), [255, 0, 0], dtype="u1"))
    np.testing.assert_array_equal(second_view, np.full((4, 2), 42, dtype="u1"))


def test_empty_native_export_and_optional_dependencies():
    source = Image.new("RGB", (0, 3))
    assert source.buffer_view().nbytes == 0
    assert source.array_view().shape == (3, 0, 3)


def test_views_do_not_need_numpy_or_tensor_imports(monkeypatch):
    original_import = builtins.__import__

    def import_module(name, *args, **kwargs):
        if name.partition(".")[0] in {"numpy", "torch", "cupy"}:
            pytest.fail("buffer exchange imported an optional dependency")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_module)
    source = Image.new("RGB", (3, 2), (1, 2, 3))
    assert source.buffer_view().tobytes() == bytes([1, 2, 3] * 6)
    values = memoryview(bytearray(range(18))).cast("B", (2, 3, 3))
    mapped = Image.fromarray(values, copy=False)
    assert mapped.getpixel((0, 0)) == (0, 1, 2)


def test_views_preserve_nonfinite_float_bits():
    raw = bytes.fromhex("0100c07f0000807f000080ff00000080")
    source = Image.frombytes("F", (4, 1), raw)
    assert source.buffer_view().tobytes() == raw
    assert source.array_view().tobytes() == raw


def test_view_reimport_retains_native_owner_and_detaches():
    source = Image.new("RGB", (3, 2), (1, 2, 3))
    values = source.array_view()
    imported = Image.fromarray(values, copy=False)
    source.close()
    del values
    gc.collect()
    assert imported.getpixel((0, 0)) == (1, 2, 3)
    imported.putpixel((0, 0), (4, 5, 6))
    assert imported.getpixel((0, 0)) == (4, 5, 6)
