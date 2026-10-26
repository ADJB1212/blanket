from __future__ import annotations

from io import BytesIO

import pytest
from PIL import Image as PillowImage

from blanket import Image, JpegImagePlugin


def jpeg(mode="RGB", size=(32, 48), *, subsampling=0, progressive=False, **metadata):
    channels = len(PillowImage.new(mode, (1, 1)).getbands())
    pixels = bytes((x * 17 + y * 31 + c * 59) % 256 for y in range(size[1]) for x in range(size[0]) for c in range(channels))
    source = PillowImage.frombytes(mode, size, pixels)
    stream = BytesIO()
    source.save(stream, "JPEG", quality=87, subsampling=subsampling, progressive=progressive, **metadata)
    return stream.getvalue()


def decoded(data):
    image = PillowImage.open(BytesIO(data))
    image.load()
    return image


@pytest.mark.parametrize("mode", ["L", "RGB", "CMYK"])
@pytest.mark.parametrize("progressive", [False, True])
@pytest.mark.parametrize("subsampling", [0, 1, 2])
@pytest.mark.parametrize("method", list(Image.Transpose))
def test_transform_inverse_preserves_decoded_source_and_quantization(mode, progressive, subsampling, method):
    data = jpeg(mode, progressive=progressive, subsampling=subsampling)
    output = JpegImagePlugin.transform(data, method)
    inverse = {Image.ROTATE_90: Image.ROTATE_270, Image.ROTATE_270: Image.ROTATE_90}.get(method, method)
    restored = JpegImagePlugin.transform(output, inverse)
    original = decoded(data)
    transformed = decoded(output)
    expected_size = original.size[::-1] if method in (Image.ROTATE_90, Image.ROTATE_270, Image.TRANSPOSE, Image.TRANSVERSE) else original.size
    assert transformed.size == expected_size
    assert transformed.mode == original.mode
    assert decoded(restored).tobytes() == original.tobytes()
    assert decoded(restored).quantization == original.quantization
    assert bool(transformed.info.get("progressive")) == progressive
    with Image.open(BytesIO(output)) as blanket:
        assert blanket.size == transformed.size
        assert blanket.mode == transformed.mode


@pytest.mark.parametrize("method", list(Image.Transpose))
def test_transform_direction_matches_pillow(method):
    data = jpeg("L")
    expected = decoded(data).transpose(method)
    actual = decoded(JpegImagePlugin.transform(data, method))
    assert actual.size == expected.size
    assert max(abs(a - b) for a, b in zip(actual.tobytes(), expected.tobytes(), strict=True)) <= 1


@pytest.mark.parametrize("mode,subsampling,origin", [("L", 0, (8, 8)), ("RGB", 0, (8, 8)), ("RGB", 1, (16, 8)), ("RGB", 2, (16, 16)), ("CMYK", 0, (8, 8))])
def test_crop_uses_source_coordinates_before_transform(mode, subsampling, origin):
    data = jpeg(mode, subsampling=subsampling)
    box = (*origin, 32, 40)
    cropped = JpegImagePlugin.transform(data, crop=box)
    actual = decoded(cropped)
    assert actual.size == (32 - origin[0], 40 - origin[1])
    assert actual.quantization == decoded(data).quantization
    if mode != "RGB" or subsampling == 0:
        assert actual.tobytes() == decoded(data).crop(box).tobytes()
    combined = JpegImagePlugin.transform(data, Image.TRANSPOSE, crop=box)
    separate = JpegImagePlugin.transform(cropped, Image.TRANSPOSE)
    assert decoded(combined).tobytes() == decoded(separate).tobytes()


def test_crop_allows_partial_right_and_bottom_edges():
    data = jpeg("L", size=(31, 39))
    box = (8, 8, 29, 37)
    assert decoded(JpegImagePlugin.transform(data, crop=box)).tobytes() == decoded(data).crop(box).tobytes()


@pytest.mark.parametrize("method", [Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM, Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270, Image.TRANSVERSE])
def test_imperfect_edges_rejected_or_explicitly_trimmed(method):
    data = jpeg("L", size=(31, 39))
    with pytest.raises(OSError):
        JpegImagePlugin.transform(data, method)
    result = decoded(JpegImagePlugin.transform(data, method, trim=True))
    box = {
        Image.FLIP_LEFT_RIGHT: (0, 0, 24, 39),
        Image.FLIP_TOP_BOTTOM: (0, 0, 31, 32),
        Image.ROTATE_90: (0, 0, 24, 39),
        Image.ROTATE_180: (0, 0, 24, 32),
        Image.ROTATE_270: (0, 0, 31, 32),
        Image.TRANSVERSE: (0, 0, 24, 32),
    }[method]
    expected = decoded(data).crop(box).transpose(method)
    assert result.size == expected.size
    assert max(abs(a - b) for a, b in zip(result.tobytes(), expected.tobytes(), strict=True)) <= 1


def test_transpose_is_perfect_with_partial_blocks():
    data = jpeg("L", size=(31, 39))
    transformed = JpegImagePlugin.transform(data, Image.TRANSPOSE)
    assert decoded(transformed).size == (39, 31)
    assert decoded(JpegImagePlugin.transform(transformed, Image.TRANSPOSE)).tobytes() == decoded(data).tobytes()


def test_metadata_policy_is_explicit_and_preserve_does_not_apply_orientation():
    exif = Image.Exif()
    exif[274] = 6
    exif[256] = 32
    exif[257] = 48
    exif.thumbnail = jpeg("L", (8, 8))
    exif_bytes = exif.tobytes()
    icc = b"test ICC profile" * 6000
    xmp = b'<x:xmpmeta xmlns:x="adobe:ns:meta/">packet</x:xmpmeta>'
    stream = BytesIO()
    Image.new("RGB", (32, 48), (50, 100, 150)).save(stream, "JPEG", exif=exif_bytes, icc_profile=icc, xmp=xmp)
    data = stream.getvalue()
    comment = b"sample comment"
    data = data[:2] + b"\xff\xfe" + (len(comment) + 2).to_bytes(2, "big") + comment + data[2:]
    preserved = JpegImagePlugin.transform(data, Image.ROTATE_90, metadata="preserve")
    stripped = JpegImagePlugin.transform(data, Image.ROTATE_90)
    assert decoded(preserved).size == (48, 32)
    assert decoded(preserved).info["exif"] == exif_bytes
    assert decoded(preserved).info["icc_profile"] == icc
    assert decoded(preserved).info["comment"] == comment
    assert xmp in preserved
    for payload in (b"Exif\0\0", b"ICC_PROFILE\0", xmp, comment):
        assert payload not in stripped
    assert decoded(stripped).tobytes() == decoded(preserved).tobytes()


@pytest.mark.parametrize("endian", ["<", ">"])
def test_preserved_exif_first_marker_allows_backend_dimension_update(endian):
    exif = Image.Exif()
    exif.endian = endian
    exif[274] = 6
    exif[34665] = {40962: 32, 40963: 48}
    exif.thumbnail = jpeg("L", (8, 8))
    data = jpeg(exif=exif.tobytes())
    assert data[2:4] == b"\xff\xe0"
    jfif_end = 4 + int.from_bytes(data[4:6], "big")
    data = data[:2] + data[jfif_end:]
    output = JpegImagePlugin.transform(data, Image.ROTATE_90, metadata="preserve")
    with Image.open(BytesIO(output)) as image:
        tags = image.getexif()
        assert tags[274] == 6
        assert tags.get_ifd(34665)[40962] == 48
        assert tags.get_ifd(34665)[40963] == 32
        assert tags.thumbnail == exif.thumbnail


def test_crop_partial_result_requires_explicit_trim_for_rotation():
    data = jpeg("L")
    with pytest.raises(OSError):
        JpegImagePlugin.transform(data, Image.ROTATE_180, crop=(8, 8, 29, 37))
    result = JpegImagePlugin.transform(data, Image.ROTATE_180, crop=(8, 8, 29, 37), trim=True)
    assert decoded(result).size == (16, 24)


@pytest.mark.parametrize("crop", [(1, 0, 16, 16), (0, 1, 16, 16), (0, 0, 33, 48), (8, 8, 8, 16), (16, 16, 8, 8)])
def test_invalid_crop_is_rejected(crop):
    with pytest.raises(ValueError):
        JpegImagePlugin.transform(jpeg(), crop=crop)


@pytest.mark.parametrize(
    "options,error",
    [
        ({"method": True}, TypeError),
        ({"method": "rotate"}, TypeError),
        ({"method": 7}, ValueError),
        ({"crop": [0, 0, 16, 16]}, TypeError),
        ({"crop": (0, 0, 16)}, TypeError),
        ({"crop": (False, 0, 16, 16)}, TypeError),
        ({"crop": (0, 0, 16.0, 16)}, TypeError),
        ({"crop": (-1, 0, 16, 16)}, ValueError),
        ({"crop": (0, 0, 2**64, 16)}, ValueError),
        ({"trim": 1}, TypeError),
        ({"metadata": "unknown"}, ValueError),
    ],
)
def test_invalid_options(options, error):
    with pytest.raises(error):
        JpegImagePlugin.transform(jpeg(), **options)


@pytest.mark.parametrize("data", [b"", b"not JPEG", b"\xff\xd8\xff\xd9", jpeg()[:100], jpeg()[:-100]])
def test_malformed_jpeg(data):
    with pytest.raises(OSError):
        JpegImagePlugin.transform(data)


def test_input_types_and_noop():
    data = jpeg()
    for value in (data, bytearray(data), memoryview(data)):
        output = JpegImagePlugin.transform(value)
        assert isinstance(output, bytes)
        assert decoded(output).tobytes() == decoded(data).tobytes()
    for value in ("photo.jpg", BytesIO(data), Image.new("RGB", (1, 1)), 123):
        with pytest.raises(TypeError):
            JpegImagePlugin.transform(value)


def test_dimension_limit_before_transform_allocation():
    data = bytearray(jpeg())
    start = data.index(b"\xff\xc0")
    data[start + 5 : start + 9] = (20000).to_bytes(2, "big") * 2
    with pytest.raises(OSError, match="limit"):
        JpegImagePlugin.transform(data)
