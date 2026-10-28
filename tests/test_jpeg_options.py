from __future__ import annotations

from io import BytesIO

import pytest
from PIL import Image as PillowImage, JpegImagePlugin

from blanket import Image
from blanket.Compressor import LosslessImageCompressor, LossyImageCompressor


def encode(image, **options):
    stream = BytesIO()
    image.save(stream, "JPEG", quality=85, **options)
    return stream.getvalue()


@pytest.mark.parametrize("mode", ["L", "RGB", "CMYK", "YCbCr"])
@pytest.mark.parametrize("subsampling", [-1, 0, 1, 2])
@pytest.mark.parametrize("progressive,optimize", [(False, False), (False, True), (True, False), (True, True)])
def test_jpeg_controls_match_pillow(mode, subsampling, progressive, optimize):
    size = (17, 9)
    channels = len(PillowImage.new(mode, size).getbands())
    raw = bytes((i * 37 + i // 7) % 256 for i in range(size[0] * size[1] * channels))
    options = {"subsampling": subsampling, "progressive": progressive, "optimize": optimize}
    actual = encode(Image.frombytes(mode, size, raw), **options)
    reference_options = {**options, "subsampling": 0} if mode == "CMYK" else options
    expected = encode(PillowImage.frombytes(mode, size, raw), **reference_options)
    with PillowImage.open(BytesIO(actual)) as decoded, PillowImage.open(BytesIO(expected)) as reference:
        assert decoded.mode == reference.mode == ("RGB" if mode == "YCbCr" else mode)
        assert decoded.size == size
        assert bool(decoded.info.get("progressive")) == progressive
        assert JpegImagePlugin.get_sampling(decoded) == JpegImagePlugin.get_sampling(reference)
        assert decoded.tobytes() == reference.tobytes()
        assert Image.open(BytesIO(actual)).tobytes() == decoded.tobytes()


@pytest.mark.parametrize("width", [1, 3, 17, 33, 257])
@pytest.mark.parametrize("subsampling", [0, 1, 2])
def test_ycbcr_subsampling_preserves_odd_edges(width, subsampling):
    size = (width, 3)
    raw = bytes((i * 137 + i // 7) % 256 for i in range(width * 3 * 3))
    actual = encode(Image.frombytes("YCbCr", size, raw), subsampling=subsampling)
    expected = encode(PillowImage.frombytes("YCbCr", size, raw), subsampling=subsampling)
    assert PillowImage.open(BytesIO(actual)).tobytes() == PillowImage.open(BytesIO(expected)).tobytes()


@pytest.mark.parametrize("subsampling", [0, 1, 2])
def test_ycbcr_even_edge_padding_interoperates(subsampling):
    size = (2, 3)
    raw = bytes((i * 137 + i // 7) % 256 for i in range(18))
    encoded = encode(Image.frombytes("YCbCr", size, raw), subsampling=subsampling)
    reference = PillowImage.open(BytesIO(encoded))
    actual = Image.open(BytesIO(encoded))
    assert reference.size == actual.size == size
    assert JpegImagePlugin.get_sampling(reference) == subsampling
    assert actual.tobytes() == reference.tobytes()


@pytest.mark.parametrize("name,value", [("4:4:4", 0), ("4:2:2", 1), ("4:2:0", 2), ("4:1:1", 2)])
def test_jpeg_subsampling_aliases(name, value):
    image = Image.new("RGB", (17, 9), "red")
    assert encode(image, subsampling=name) == encode(image, subsampling=value)


def test_jpeg_defaults_and_cached_encoder_reset():
    image = Image.frombytes("RGB", (17, 9), bytes(i % 256 for i in range(17 * 9 * 3)))
    baseline = encode(image)
    assert baseline == encode(image, subsampling=2, progressive=False, optimize=False)
    for subsampling in (0, 1, 2):
        optimized = encode(image, subsampling=subsampling, optimize=True)
        progressive = encode(image, subsampling=subsampling, progressive=True)
        plain = encode(image, subsampling=subsampling)
        assert PillowImage.open(BytesIO(optimized)).tobytes() == PillowImage.open(BytesIO(plain)).tobytes()
        assert PillowImage.open(BytesIO(progressive)).tobytes() == PillowImage.open(BytesIO(plain)).tobytes()
        assert encode(image) == baseline


@pytest.mark.parametrize("compressor", [LosslessImageCompressor(effort=3), LossyImageCompressor(effort=1)])
def test_jpeg_compressors_keep_subsampling_and_progressive(compressor):
    image = Image.frombytes("RGB", (17, 9), bytes(i * 31 % 256 for i in range(17 * 9 * 3)))
    output = encode(image, subsampling=0, progressive=True, optimize=True, compressor=compressor)
    decoded = PillowImage.open(BytesIO(output))
    assert decoded.info["progressive"]
    assert JpegImagePlugin.get_sampling(decoded) == 0


@pytest.mark.parametrize(
    "options,error",
    [
        ({"subsampling": 3}, ValueError),
        ({"subsampling": -2}, ValueError),
        ({"subsampling": "keep"}, ValueError),
        ({"subsampling": "4:2:1"}, ValueError),
        ({"subsampling": 1.5}, TypeError),
        ({"subsampling": True}, TypeError),
        ({"progressive": 1}, TypeError),
        ({"optimize": "yes"}, TypeError),
    ],
)
def test_jpeg_rejects_invalid_controls(options, error):
    image = Image.new("RGB", (2, 2))
    stream = BytesIO(b"unchanged")
    with pytest.raises(error):
        image.save(stream, "JPEG", **options)
    assert stream.getvalue() == b"unchanged"


@pytest.mark.parametrize("name", ["subsampling", "progressive", "optimize"])
def test_jpeg_options_are_format_specific(name):
    with pytest.raises(TypeError, match="unsupported PNG save option"):
        Image.new("RGB", (2, 2)).save(BytesIO(), "PNG", **{name: 0})
