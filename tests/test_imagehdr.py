from __future__ import annotations

import math
import struct
from io import BytesIO

import numpy as np
import pytest
from PIL import Image as PillowImage

from blanket import Image, ImageHdr, PngImagePlugin
from blanket._exif import read_metadata

MASTERING = (0.708, 0.292, 0.170, 0.797, 0.131, 0.046, 0.3127, 0.3290, 1000.0, 0.0005)


def curve(value, operator):
    if operator == "reinhard":
        return value / (1 + value)
    if operator == "aces":
        return min(1, value * (2.51 * value + 0.03) / (value * (2.43 * value + 0.59) + 0.14))

    def hable(x):
        return (x * (0.15 * x + 0.05) + 0.004) / (x * (0.15 * x + 0.5) + 0.06) - 1 / 15

    return min(1, hable(value) / hable(11.2))


def encoded(value, depth=8):
    value = 12.92 * value if value <= 0.0031308 else 1.055 * value ** (1 / 2.4) - 0.055
    return math.floor(value * ((1 << depth) - 1) + 0.5)


@pytest.mark.parametrize("operator", ["reinhard", "aces", "filmic"])
@pytest.mark.parametrize("depth", [8, 16])
def test_float_scene_values_match_reference(operator, depth):
    values = [0, 0.001, 0.25, 1, 4, 11.2, 100]
    image = Image.frombytes("F", (len(values), 1), struct.pack(f"<{len(values)}f", *values))
    result = ImageHdr.tone_map(image, operator, exposure=1, bit_depth=depth)
    expected = [encoded(curve(v * 2, operator), depth) for v in struct.unpack(f"<{len(values)}f", image.tobytes())]
    assert list(result.getdata()) == expected
    assert (result.mode, result.bit_depth) == ("L", depth)
    assert image.tobytes() == struct.pack(f"<{len(values)}f", *values)


@pytest.mark.parametrize("depth", [8, 10, 12, 16])
@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
def test_integer_normalization_and_alpha(mode, depth):
    channels = len(mode) if mode != "L" else 1
    maximum = (1 << depth) - 1
    samples = np.array([0, 1, maximum // 2, maximum], dtype="u1" if depth == 8 else "<u2")
    samples = np.repeat(samples[:, None], channels, axis=1).reshape(1, 4, channels)
    image = Image.fromarray(samples[:, :, 0] if mode == "L" else samples, bit_depth=depth)
    result = ImageHdr.tone_map(image, bit_depth=16)
    for x in range(4):
        source = int(samples[0, x, 0])
        value = encoded(curve(source / maximum, "reinhard"), 16)
        expected = value if mode == "L" else (value,) * 3 + ((round(source * 65535 / maximum),) if mode == "RGBA" else ())
        assert result.getpixel((x, 0)) == expected


@pytest.mark.parametrize("mode,layout", [("I;16", "<"), ("I;16L", "<"), ("I;16B", ">")])
def test_scalar_endianness(mode, layout):
    image = Image.frombytes(mode, (3, 1), struct.pack(layout + "3H", 0, 32768, 65535))
    assert list(ImageHdr.tone_map(image).getdata()) == [0, encoded(curve(32768 / 65535, "reinhard")), encoded(0.5)]


def test_la_alpha_and_nonfinite_float():
    image = Image.frombytes("LA", (2, 1), bytes([255, 71, 128, 0]))
    assert list(ImageHdr.tone_map(image).getdata()) == [(encoded(0.5), 71), (encoded(curve(128 / 255, "reinhard")), 0)]
    with pytest.raises(ValueError):
        ImageHdr.tone_map(image, bit_depth=16)
    image = Image.frombytes("F", (5, 1), struct.pack("<5f", -1, math.nan, -math.inf, math.inf, 3.4e38))
    assert list(ImageHdr.tone_map(image).getdata()) == [0, 0, 0, 255, 255]


@pytest.mark.parametrize("transfer,code", [("pq", 16), ("hlg", 18)])
def test_hdr_transfer_metadata_and_white(transfer, code):
    image = Image.new("RGB", (1, 1), (255, 255, 255))
    image.info.update(cicp=(9, code, 0, 1), mastering_display=MASTERING, content_light_level=(1000, 250), icc_profile=b"stale", gamma=0.45, label="keep")
    original = image.info.copy()
    automatic = ImageHdr.tone_map(image, reference_white=100)
    explicit = ImageHdr.tone_map(image, transfer=transfer, primaries="bt2020", reference_white=100)
    assert automatic.tobytes() == explicit.tobytes()
    assert automatic.getpixel((0, 0)) == (encoded(curve(100 if transfer == "pq" else 10, "reinhard")),) * 3
    assert automatic.info == {"cicp": (1, 13, 0, 1), "label": "keep"}
    assert image.info == original


def test_pq_100_nits_and_hlg_midpoint():
    pq = 0.508078421517399
    image = Image.frombytes("F", (1, 1), struct.pack("<f", pq))
    assert ImageHdr.tone_map(image, transfer="pq", reference_white=100).getpixel((0, 0)) == encoded(0.5)
    image = Image.frombytes("F", (1, 1), struct.pack("<f", 0.5))
    linear = (1 / 12) ** 1.2 * 1000 / 100
    assert ImageHdr.tone_map(image, transfer="hlg", reference_white=100).getpixel((0, 0)) == encoded(curve(linear, "reinhard"))


def test_bt2020_primaries_and_srgb_input():
    image = Image.new("RGB", (1, 1), (255, 0, 0))
    result = ImageHdr.tone_map(image, primaries="bt2020")
    assert result.getpixel((0, 0)) == (encoded(curve(1.660491, "reinhard")), 0, 0)
    image = Image.new("L", (1, 1), 128)
    linear = ((128 / 255 + 0.055) / 1.055) ** 2.4
    assert ImageHdr.tone_map(image, transfer="srgb").getpixel((0, 0)) == encoded(curve(linear, "reinhard"))


@pytest.mark.parametrize(
    "options",
    [
        {"operator": "invalid"},
        {"transfer": ""},
        {"primaries": "p3"},
        {"bit_depth": 12},
        {"exposure": math.nan},
        {"exposure": 2000},
        {"reference_white": 0},
        {"reference_white": math.inf},
        {"peak_luminance": 0},
        {"peak_luminance": math.nan},
    ],
)
def test_invalid_options(options):
    with pytest.raises(ValueError):
        ImageHdr.tone_map(Image.new("RGB", (1, 1)), **options)


@pytest.mark.parametrize("mode", ["1", "P", "PA", "I", "CMYK", "HSV", "LAB", "YCbCr"])
def test_unsupported_modes(mode):
    with pytest.raises(ValueError):
        ImageHdr.tone_map(Image.new(mode, (1, 1)))


def test_closed_and_unsupported_metadata():
    image = Image.new("RGB", (1, 1))
    for cicp in [(9, 16, 9, 0), (12, 16, 0, 1), (9, 2, 0, 1), (9,)]:
        image.info["cicp"] = cicp
        with pytest.raises(ValueError):
            ImageHdr.tone_map(image)
    image.info.clear()
    image.close()
    with pytest.raises(ValueError):
        ImageHdr.tone_map(image)


@pytest.mark.parametrize("code", [16, 18])
def test_png_hdr_metadata_roundtrip_and_pillow_interop(code):
    image = Image.fromarray(np.tile(np.array([11, 91, 203], dtype="<u2"), (2, 3, 1)), bit_depth=16)
    metadata = {"cicp": (9, code, 0, 1), "mastering_display": MASTERING, "content_light_level": (1000.0, 250.0)}
    image.info.update(metadata)
    for _ in range(2):
        output = BytesIO()
        image.save(output, "PNG")
        data = output.getvalue()
        assert data.index(b"cICP") < data.index(b"IDAT")
        assert data.index(b"mDCV") < data.index(b"IDAT")
        assert data.index(b"cLLI") < data.index(b"IDAT")
        assert data[data.index(b"mDCV") + 4 : data.index(b"mDCV") + 28] == struct.pack(">8H2I", 35400, 14600, 8500, 39850, 6550, 2300, 15635, 16450, 10000000, 5)
        reopened = Image.open(BytesIO(data))
        for key, value in metadata.items():
            assert reopened.info[key] == value
        assert reopened.tobytes() == image.tobytes()
        assert PillowImage.open(BytesIO(data)).size == image.size
        image = reopened


@pytest.mark.parametrize(
    "key,value",
    [
        ("cicp", (9, 16, 1, 1)),
        ("cicp", (9, 16, 0)),
        ("cicp", (9, 16, 0, 256)),
        ("mastering_display", (*MASTERING[:8], 1, 2)),
        ("mastering_display", (2, *MASTERING[1:])),
        ("content_light_level", (math.nan, 250)),
        ("content_light_level", (-1, 1)),
        ("content_light_level", (1e10, 0)),
    ],
)
def test_invalid_png_hdr_metadata(key, value):
    image = Image.new("RGB", (1, 1))
    with pytest.raises(ValueError):
        image.save(BytesIO(), "PNG", **{key: value})


def test_metadata_overrides_removal_and_raw_chunk_validation():
    image = Image.new("RGB", (1, 1))
    image.info["cicp"] = (9, 16, 0, 1)
    output = BytesIO()
    image.save(output, "PNG", cicp=None)
    assert "cicp" not in Image.open(output).info
    with pytest.raises(TypeError):
        image.save(BytesIO(), "JPEG", cicp=(9, 16, 0, 1))
    with pytest.raises(ValueError):
        Image.new("RGB", (1, 1)).save(BytesIO(), "PNG", mastering_display=MASTERING)
    for payload, after in [(b"bad", False), (bytes([9, 16, 0, 1]), True), (bytes([9, 16, 0, 1]), False)]:
        pnginfo = PngImagePlugin.PngInfo()
        pnginfo.add(b"cICP", payload, after)
        with pytest.raises(ValueError):
            image.save(BytesIO(), "PNG", pnginfo=pnginfo)
    pnginfo = PngImagePlugin.PngInfo()
    pnginfo.add(b"cICP", b"bad")
    # Malformed ancillary metadata is ignored without changing image pixels.
    data = b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 3) + b"cICPbad" + bytes(4)
    assert read_metadata(data, "PNG") == {}


def test_current_frame_and_parallel_chunk_boundaries():
    frames = [Image.new("RGB", (2, 1), (value,) * 3) for value in [64, 192]]
    output = BytesIO()
    frames[0].save(output, "PNG", save_all=True, append_images=frames[1:])
    image = Image.open(output)
    image.seek(1)
    expected = ImageHdr.tone_map(frames[1].convert(image.mode))
    assert ImageHdr.tone_map(image).tobytes() == expected.tobytes()
    assert image.tell() == 1
    # Cross native parallel chunk boundaries, including the final partial chunk.
    samples = (np.arange(513 * 1025 * 4, dtype=np.uint32) % 4096).astype("<u2").reshape(513, 1025, 4)
    image = Image.fromarray(samples, bit_depth=12)
    result = ImageHdr.tone_map(image, bit_depth=16)
    expected = samples.astype(np.float64) / 4095
    color = expected[:, :, :3]
    color = color / (1 + color)
    expected[:, :, :3] = np.where(color <= 0.0031308, color * 12.92, 1.055 * color ** (1 / 2.4) - 0.055)
    assert result.tobytes() == np.floor(expected * 65535 + 0.5).astype("<u2").tobytes()


@pytest.mark.parametrize("transfer", ["pq", "hlg"])
def test_extreme_reference_white_stays_finite(transfer):
    for value, expected in [(0, 0), (255, 255)]:
        result = ImageHdr.tone_map(Image.new("RGB", (1, 1), (value,) * 3), transfer=transfer, primaries="bt2020", reference_white=1e-320)
        assert result.getpixel((0, 0)) == (expected,) * 3
