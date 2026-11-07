from __future__ import annotations

import struct

import pytest

from blanket import Image, ImageHash, _blanket


@pytest.mark.parametrize(
    "mode,size,data",
    [
        ("1", (9, 2), b"\xa5\x80\x5a\x00"),
        ("RGB", (3, 2), bytes(range(18))),
        ("LAB", (3, 2), bytes(range(18))),
        ("I;16B", (3, 2), struct.pack(">6H", 0, 1, 255, 256, 32768, 65535)),
        ("F", (3, 2), struct.pack("<6f", -1, 0, 0.5, 1, float("inf"), float("nan"))),
    ],
)
@pytest.mark.parametrize("strided", [False, True])
def test_buffer_imports_snapshot_samples_without_changing_the_source(mode, size, data, strided):
    source = bytearray(len(data) * 2 if strided else len(data))
    view = memoryview(source)[::2] if strided else source
    view[:] = data
    image = Image.frombytes(mode, size, view)
    assert image.tobytes() == data
    view[:] = bytes(len(data))
    assert image.tobytes() == data


def test_buffer_import_preserves_bytes_overrides_and_released_view_errors():
    class CustomBytes(bytearray):
        def __bytes__(self):
            return b"\x7f"

    assert Image.frombytes("L", (1, 1), CustomBytes(b"\x00")).getpixel((0, 0)) == 127
    view = memoryview(b"\x00")
    view.release()
    with pytest.raises(TypeError, match="bytes-like"):
        Image.frombytes("L", (1, 1), view)


@pytest.mark.parametrize("mode", ["1", "L", "LA", "P", "PA", "RGB", "RGBA", "LAB", "I", "F", "I;16B", "RGB;F"])
def test_equality_uses_native_pixels_and_preserves_comparison_semantics(mode, monkeypatch):
    image = Image.new("RGB", (17, 3), (17, 31, 89)).convert(mode)
    same = image.copy()
    different = image.copy()
    scalar = 0 if image.getpixel((16, 2)) != 0 else 1
    different.putpixel((16, 2), scalar if len(image.getbands()) == 1 else (0,) * len(image.getbands()))

    def no_export(*args, **kwargs):
        pytest.fail("comparison exported pixels to Python")

    monkeypatch.setattr(Image.Image, "tobytes", no_export)
    assert image == same
    assert image != different
    same.info["changed"] = True
    assert image != same
    same.info.clear()
    same.close()
    with pytest.raises(ValueError, match="closed"):
        _ = image == same


def test_equality_preserves_float_bit_patterns():
    nan = struct.pack("<I", 0x7FC00001)
    assert Image.frombytes("F", (1, 1), nan) == Image.frombytes("F", (1, 1), nan)
    assert Image.new("F", (1, 1), 0.0) != Image.new("F", (1, 1), -0.0)
    assert Image.new("1", (1, 1), 1) == Image.new("1", (1, 1), 255)


@pytest.mark.parametrize("hasher,size", [(ImageHash.average_hash, (8, 8)), (ImageHash.dhash, (9, 8)), (ImageHash.phash, (32, 32))])
def test_hashing_prepared_thumbnails_needs_no_raster_copies(hasher, size, monkeypatch):
    image = Image.frombytes("L", size, bytes((i * 37) % 256 for i in range(size[0] * size[1])))
    expected = hasher(image)

    def no_copy(*args, **kwargs):
        pytest.fail("prepared thumbnail was copied or exported")

    for name in ("convert", "resize", "tobytes"):
        monkeypatch.setattr(Image.Image, name, no_copy)
    assert hasher(image) == expected


@pytest.mark.parametrize("depth", [10, 12, 16])
@pytest.mark.parametrize("mode,channels", [("L", 1), ("RGB", 3), ("RGBA", 4)])
def test_unsigned_native_staging_matches_byte_staging(depth, mode, channels):
    if not hasattr(_blanket, "_gpu_unsigned_image"):
        pytest.skip("GPU transfer bindings were not compiled")
    maximum = (1 << depth) - 1
    samples = [0, 1, maximum // 2, maximum - 1, maximum] * channels
    data = struct.pack(f"<{len(samples)}H", *samples)
    image = Image.frombytes(mode, (5, 1), data, bit_depth=depth)
    working = Image.Image(_blanket._gpu_unsigned_image(image._native, depth, False))
    assert working.tobytes() == _blanket._gpu_unsigned_bytes(data, depth, False)
    result = Image.Image(_blanket._gpu_unsigned_image(working._native, depth, True))
    assert (result.mode, result.size, result.bit_depth) == (mode, image.size, depth)
    assert result.tobytes() == data
    working.close()
    image.close()
    assert result.tobytes() == data


def test_unsigned_native_staging_preserves_nonfinite_rules_and_validation():
    if not hasattr(_blanket, "_gpu_unsigned_image"):
        pytest.skip("GPU transfer bindings were not compiled")
    image = Image.frombytes("F", (5, 1), struct.pack("<5f", float("nan"), float("inf"), -float("inf"), 0.5, -1.0))
    result = Image.Image(_blanket._gpu_unsigned_image(image._native, 16, True))
    assert struct.unpack("<5H", result.tobytes()) == (0, 65535, 0, 32768, 0)
    for depth, encode in [(8, True), (16, False)]:
        with pytest.raises(ValueError, match="transfer"):
            _blanket._gpu_unsigned_image(image._native, depth, encode)
    image.close()
    with pytest.raises(ValueError, match="closed"):
        _blanket._gpu_unsigned_image(image._native, 16, True)


@pytest.mark.parametrize("mode", ["LA", "PA"])
@pytest.mark.parametrize("count", [0, 1, 15, 16, 17, 16383, 16384, 16385, 262145])
def test_putalpha_reuses_luma_without_changing_it(mode, count):
    image = Image.frombytes(mode, (count, 1), bytes((i * 37) % 256 for i in range(count * 2)))
    before = image.getchannel(0).tobytes()
    alpha = Image.frombytes("L", image.size, bytes(i % 256 for i in range(count)))
    image.putalpha(alpha)
    assert image.mode == mode
    assert image.getchannel(0).tobytes() == before
    assert image.getchannel(1).tobytes() == alpha.tobytes()
