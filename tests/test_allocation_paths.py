from __future__ import annotations

import pickle
import struct

import pytest

from blanket import Image, ImageDraw, ImageHash, ImageOps, ImageStat, _blanket
from blanket._sequence import _png_regions


@pytest.mark.parametrize("mode", ["L", "LA", "RGB", "RGBA", "CMYK", "HSV", "LAB"])
def test_point_callable_transfers_one_shared_table(mode, monkeypatch):
    image = Image.frombytes(mode, (17, 3), bytes((i * 37) % 256 for i in range(51 * len(mode))))
    values = [value * 1.5 - 20.5 for value in range(256)]
    expected = image.point(values * len(mode))
    native_lut = Image.ops_lut
    calls = []

    def lookup(value):
        calls.append(value)
        return values[value]

    def shared_table(native, table):
        assert len(table) == 256
        return native_lut(native, table)

    monkeypatch.setattr(Image, "ops_lut", shared_table)
    assert image.point(lookup).tobytes() == expected.tobytes()
    assert calls == list(range(256))


def test_apng_full_frame_deltas_reuse_prepared_images(monkeypatch):
    first = Image.new("RGBA", (17, 3), (17, 31, 89, 255))
    second = Image.new("RGBA", first.size, (71, 113, 173, 149))

    def no_crop(*args, **kwargs):
        pytest.fail("full-frame APNG delta copied the raster")

    monkeypatch.setattr(Image.Image, "crop", no_crop)
    frames, durations, disposals, blends, regions = _png_regions([first, second], [10, 20], [0, 0], [0, 0], False, True)
    assert frames[0] is first and frames[1] is second
    assert (durations, disposals, blends, regions) == ([10, 20], [0, 0], [0, 0], ((17, 3), [(0, 0), (0, 0)]))


def test_apng_background_disposal_compares_without_copying(monkeypatch):
    first = Image.new("RGBA", (17, 3), (17, 31, 89, 255))
    second = first.copy()
    second.putpixel((5, 1), (71, 113, 173, 149))
    third = second.copy()
    third.putpixel((5, 1), (0, 0, 0, 0))
    third.putpixel((8, 2), (1, 2, 3, 4))

    def no_copy(*args, **kwargs):
        pytest.fail("APNG disposal copied the raster")

    monkeypatch.setattr(Image.Image, "copy", no_copy)
    frames, _, _, _, regions = _png_regions([first, second, third], [10, 20, 30], [0, 1, 0], [0, 0, 0], False, True)
    assert regions == ((17, 3), [(0, 0), (5, 1), (8, 2)])
    assert frames[2].getdata() == [(1, 2, 3, 4)]


@pytest.mark.parametrize("size", [(0, 1), (17, 3), (17003, 5)])
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("cutoff,ignore", [(0, None), ((1, 3), [0, -1]), (100, 17)])
def test_tone_preserving_autocontrast_needs_no_grayscale_image(size, masked, cutoff, ignore, monkeypatch):
    image = Image.frombytes("RGB", size, bytes((i * 37 + i // 7) % 256 for i in range(size[0] * size[1] * 3)))
    mask = Image.frombytes("L", size, bytes(i % 3 for i in range(size[0] * size[1]))) if masked else None
    ignored = () if ignore is None else (ignore,) if isinstance(ignore, int) else ignore
    tails = cutoff if isinstance(cutoff, tuple) else (cutoff, cutoff)
    expected = Image.Image(_blanket.ops_autocontrast(image._native, image.convert("L")._native, None if mask is None else mask._native, tails, ignored))

    def no_conversion(*args, **kwargs):
        pytest.fail("autocontrast allocated a grayscale image")

    monkeypatch.setattr(Image.Image, "convert", no_conversion)
    assert ImageOps.autocontrast(image, cutoff, ignore, mask, preserve_tone=True) == expected


@pytest.mark.parametrize("dtype", [">f4", "<f8", ">f8", "<i4", ">u2"])
@pytest.mark.parametrize("layout", ["contiguous", "strided", "reversed", "transposed"])
def test_array_snapshots_preserve_samples_and_source(dtype, layout):
    np = pytest.importorskip("numpy")
    source = np.arange(17 * 6, dtype=np.float64).astype(dtype).reshape(6, 17)
    if "f" in dtype:
        source[0, :5] = [-0.0, -1.25, float("nan"), float("inf"), -float("inf")]
    view = {"contiguous": source, "strided": source[::2, ::2], "reversed": source[::-1, ::-1], "transposed": source.T}[layout]
    before = source.tobytes()
    expected = view.astype("<f4").tobytes() if "f" in dtype else view.tobytes()
    image = Image.fromarray(view)
    assert image.size == (view.shape[1], view.shape[0])
    assert image.tobytes() == expected
    assert source.tobytes() == before
    source.fill(0)
    assert image.tobytes() == expected


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
@pytest.mark.parametrize("count", [0, 1, 16, 17, 16384, 262145])
def test_putalpha_reuses_luma_without_changing_it(mode, count):
    image = Image.frombytes(mode, (count, 1), bytes((i * 37) % 256 for i in range(count * 2)))
    before = image.getchannel(0).tobytes()
    alpha = Image.frombytes("L", image.size, bytes(i % 256 for i in range(count)))
    image.putalpha(alpha)
    assert image.mode == mode
    assert image.getchannel(0).tobytes() == before
    assert image.getchannel(1).tobytes() == alpha.tobytes()


@pytest.mark.parametrize("mode", ["1", "L", "LA", "P", "PA", "RGB", "RGBA", "RGBa", "La", "RGBX"])
@pytest.mark.parametrize("size", [(0, 1), (16385, 1)])
def test_constant_alpha_needs_no_temporary_band(mode, size, monkeypatch):
    image = Image.new("RGB", size, (17, 31, 89)).convert(mode)
    expected = image.copy()
    expected.putalpha(Image.new("L", size, 149))

    def no_band(*args, **kwargs):
        pytest.fail("constant alpha allocated a temporary image")

    monkeypatch.setattr(Image, "new", no_band)
    image.putalpha(149)
    assert image == expected


@pytest.mark.parametrize("depth", [10, 12, 16])
@pytest.mark.parametrize("mode", ["RGB", "RGBA"])
def test_high_depth_constant_alpha_matches_band(depth, mode):
    samples = [0, 1, (1 << depth) - 1] * len(mode) * 17
    image = Image.frombytes(mode, (51, 1), struct.pack(f"<{len(samples)}H", *samples), bit_depth=depth)
    expected = image.copy()
    value = (1 << depth) - 17
    band = Image.frombytes("L", image.size, struct.pack("<H", value) * 51, bit_depth=depth)
    expected.putalpha(band)
    image.putalpha(value)
    assert image == expected


@pytest.mark.parametrize("mode", ["1", "L"])
def test_bitmap_borrows_native_mask(mode, monkeypatch):
    mask = Image.new(mode, (19, 7), 1 if mode == "1" else 149)
    image = Image.new("RGB", (23, 11), (17, 31, 89))
    expected = image.copy()
    ImageDraw.Draw(expected).bitmap((-3, 2), mask.convert("L"), fill=(71, 113, 173))

    def no_export(*args, **kwargs):
        pytest.fail("bitmap exported or converted its mask")

    monkeypatch.setattr(Image.Image, "tobytes", no_export)
    monkeypatch.setattr(Image.Image, "convert", no_export)
    ImageDraw.Draw(image).bitmap((-3, 2), mask, fill=(71, 113, 173))
    assert image == expected


@pytest.mark.parametrize("mode", ["1", "L", "LA", "RGB", "RGBA", "CMYK", "I", "I;16B"])
def test_bitmap_normalizes_all_nonzero_bilevel_values(mode):
    mask = Image.new("1", (19, 7))
    mask.putdata(([0, 1, 2, 127, 255] * 27)[:133])
    image = Image.new(mode, (23, 11))
    expected = image.copy()
    ImageDraw.Draw(expected).bitmap((-3, 2), mask.convert("L"), fill=71)
    ImageDraw.Draw(image).bitmap((-3, 2), mask, fill=71)
    assert image == expected


@pytest.mark.parametrize("depth", [8, 10, 12, 16])
def test_stat_retains_native_histogram_until_public_access(depth):
    image = Image.new("RGB", (19, 7), (17, 31, 89))
    if depth != 8:
        image = image.convert("RGB", bit_depth=depth)
    statistics = ImageStat.Stat(image)
    reference = ImageStat.Stat(statistics.h.copy()) if depth == 8 else ImageStat.Stat(image)
    statistics = ImageStat.Stat(image)
    image.close()
    for name in ("count", "extrema", "sum", "sum2", "mean", "median", "rms", "var", "stddev"):
        assert getattr(statistics, name) == getattr(reference, name)
        assert not isinstance(statistics._histogram, list)
    assert isinstance(statistics.h, list)
    del statistics.sum
    statistics.h[:] = [0] * len(statistics.h)
    assert statistics.sum == [0, 0, 0]


def test_native_stat_histogram_can_be_pickled_without_materializing_original():
    statistics = ImageStat.Stat(Image.new("RGB", (19, 7), (17, 31, 89)))
    assert statistics.sum == [2261, 4123, 11837]
    restored = pickle.loads(pickle.dumps(statistics))
    assert not isinstance(statistics._histogram, list)
    assert restored.h == statistics.h
    assert restored.sum == statistics.sum
    assert restored.mean == statistics.mean


@pytest.mark.parametrize("operation,args", [(ImageOps.invert, ()), (ImageOps.posterize, (4,)), (ImageOps.solarize, (127,))])
@pytest.mark.parametrize("mode", ["RGB", "RGBA"])
def test_palette_lut_needs_no_temporary_image(operation, args, mode, monkeypatch):
    image = Image.new("P", (19, 7), 31)
    image.putpalette(bytes((i * 37) % 256 for i in range(256 * len(mode))), mode)
    expected = operation(image, *args)

    def no_import(*args, **kwargs):
        pytest.fail("palette transform allocated a temporary image")

    monkeypatch.setattr(Image, "frombytes", no_import)
    assert operation(image, *args) == expected


@pytest.mark.parametrize("mode", ["1", "L", "LA", "P", "PA", "RGB", "RGBA", "LAB", "I", "F", "I;16B", "RGB;F", "RGBA;F"])
@pytest.mark.parametrize("size", [(0, 1), (17, 3)])
def test_band_exports_need_no_channel_raster(mode, size, monkeypatch):
    image = Image.new("RGB", size, (17, 31, 89)).convert(mode)
    bands = [image.getchannel(c).getdata() for c in range(len(image.getbands()))]

    def no_channel(*args, **kwargs):
        pytest.fail("band export allocated a channel image")

    monkeypatch.setattr(Image.Image, "getchannel", no_channel)
    for c, expected in enumerate(bands):
        assert image.getdata(c) == expected
        assert image.get_flattened_data(c) == tuple(expected)


@pytest.mark.parametrize(
    "mode,color,box",
    [
        (mode, color, (-3, 2, 19, 11))
        for mode, color in [("1", 1), ("L", 71), ("LA", (71, 149)), ("P", 71), ("PA", (71, 149)), ("RGB", (71, 113, 173)), ("RGBA", (71, 113, 173, 149)), ("I", -1024), ("I;16B", 1024), ("F", -12.5), ("RGB;F", (0.1, 0.2, 0.3))]
    ]
    + [("RGBA", (71, 113, 173, 149), box) for box in [(1, -2, 5, 7), (0, 0, 23, 13), (1, 1, 1, 1), (25, 17, 31, 21)]],
)
def test_paste_color_uses_one_pixel_source(mode, color, box, monkeypatch):
    image = Image.new(mode, (23, 13))
    expected = image.copy()
    size = box[2] - box[0], box[3] - box[1]
    source = Image.new(mode, size, color)
    _blanket.image_paste(expected._native, source._native, box[:2], None, True)
    original_new = Image.new

    def small_source(mode, size, *args, **kwargs):
        assert size == (1, 1)
        return original_new(mode, size, *args, **kwargs)

    monkeypatch.setattr(Image, "new", small_source)
    image.paste(color, box)
    assert image == expected


@pytest.mark.parametrize(("mode", "color"), [("L", 71), ("LA", (71, 149)), ("RGB", (71, 113, 173)), ("RGBA", (71, 113, 173, 149)), ("RGBX", (71, 113, 173, 149))])
@pytest.mark.parametrize("mask_mode", ["L", "RGBA"])
def test_paste_masked_color_uses_one_pixel_source(mode, color, mask_mode, monkeypatch):
    box = (-3, -1, 805, 4)
    size = (808, 5)
    mask = Image.frombytes(mask_mode, size, bytes(i % 256 for i in range(size[0] * size[1] * len(mask_mode))))
    image = Image.new(mode, (801, 3))
    expected = image.copy()
    source = Image.new(mode, size, color)
    _blanket.image_paste(expected._native, source._native, box[:2], mask._native, True)
    original_new = Image.new

    def small_source(mode, size, *args, **kwargs):
        assert size == (1, 1)
        return original_new(mode, size, *args, **kwargs)

    monkeypatch.setattr(Image, "new", small_source)
    image.paste(color, box, mask)
    assert image == expected


def test_frombytes_replaces_storage_without_paste(monkeypatch):
    image = Image.new("P", (2, 1))
    image.putpalette([0, 0, 0, 71, 113, 173])
    image.info["transparency"] = 0
    access = image.load()

    def no_paste(*args, **kwargs):
        pytest.fail("frombytes copied the replacement raster")

    monkeypatch.setattr(Image.Image, "paste", no_paste)
    image.frombytes(b"\x01\x00")
    assert access[0, 0] == 1
    assert image.getpalette() == [0, 0, 0, 71, 113, 173]
    assert image.info["transparency"] == 0


def test_full_linear_composite_replaces_storage_without_crop_or_paste(monkeypatch):
    image = Image.new("RGBA", (257, 3), (11, 29, 101, 255))
    overlay = Image.new("RGBA", image.size, (173, 113, 71, 149))
    expected = Image.alpha_composite(image, overlay, linear_light=True)
    access = image.load()

    def no_copy(*args, **kwargs):
        pytest.fail("full-image linear composite copied a raster")

    monkeypatch.setattr(Image.Image, "crop", no_copy)
    monkeypatch.setattr(Image.Image, "paste", no_copy)
    image.alpha_composite(overlay, linear_light=True)
    assert image == expected
    assert access[0, 0] == expected.getpixel((0, 0))


@pytest.mark.parametrize("pixel", [(-0.25, 0.5, 3.0), [-0.25, 0.5, 3.0]])
def test_scaled_float_color_putdata_writes_native_samples(pixel):
    image = Image.new("RGB;F", (2, 1), 4.0)
    image.putdata([pixel], scale=2.0, offset=0.25)
    assert image.tobytes() == struct.pack("<6f", -0.25, 1.25, 6.25, 4, 4, 4)
