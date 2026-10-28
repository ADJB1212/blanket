from __future__ import annotations

import struct
from io import BytesIO

import pytest
from PIL import Image as PillowImage

from blanket import Image, UnidentifiedImageError
from blanket.Compressor import LosslessImageCompressor, LossyImageCompressor
from blanket.errors import UnsupportedCodecError


@pytest.mark.parametrize(
    "mode,format",
    [
        ("1", "PNG"),
        ("L", "PNG"),
        ("LA", "PNG"),
        ("P", "PNG"),
        ("PA", "PNG"),
        ("RGB", "PNG"),
        ("RGBA", "PNG"),
        ("RGB", "BMP"),
        ("RGB", "WEBP"),
        ("RGB", "JPEG"),
        ("CMYK", "JPEG"),
        ("YCbCr", "JPEG"),
        ("LAB", "TIFF"),
        ("I", "TIFF"),
        ("I;16", "PNG"),
        ("I;16L", "TIFF"),
        ("I;16B", "TIFF"),
        ("F", "TIFF"),
    ],
)
def test_encoded_bytes_match_streams(mode, format):
    source = Image.new(mode, (7, 5), 17)
    if mode in ("P", "PA"):
        source.putpalette(bytes(i * 31 % 256 for i in range(768)))
    stream = BytesIO()
    assert source.save(stream, format) is None
    encoded = source.encode(format)
    assert type(encoded) is bytes
    assert encoded == stream.getvalue()
    direct = Image.decode(encoded)
    reference = Image.open(BytesIO(encoded))
    assert direct.filename == ""
    assert (direct.format, direct.mode, direct.size, direct.bit_depth) == (reference.format, reference.mode, reference.size, reference.bit_depth)
    assert direct.tobytes() == reference.tobytes()
    assert direct.info == reference.info
    assert direct.getpalette() == reference.getpalette()
    with PillowImage.open(BytesIO(encoded)) as interoperable:
        assert interoperable.size == direct.size


@pytest.mark.parametrize("mode,channels,depth,format", [("L", 1, 10, "PNG"), ("RGB", 3, 12, "PNG"), ("RGBA", 4, 16, "PNG"), ("RGB;F", 3, 32, "TIFF"), ("RGBA;F", 4, 32, "TIFF")])
def test_encoded_bytes_preserve_sample_depth(mode, channels, depth, format):
    values = [i / 4 - 2 if depth == 32 else (i * 131) % (1 << depth) for i in range(6 * channels)]
    raw = struct.pack(f"<{len(values)}{'f' if depth == 32 else 'H'}", *values)
    source = Image.frombytes(mode, (3, 2), raw, bit_depth=depth)
    options = {"compression": "tiff_lzw"} if format == "TIFF" else {}
    encoded = source.encode(format, **options)
    reference = BytesIO()
    source.save(reference, format, **options)
    assert encoded == reference.getvalue()
    decoded = Image.decode(encoded)
    assert decoded.bit_depth == depth
    assert decoded.tobytes() == source.tobytes()


@pytest.mark.parametrize("format", ["PNG", "JPEG", "WEBP", "TIFF"])
def test_encoded_bytes_preserve_and_override_metadata(format):
    source = Image.new("RGB", (7, 5), "red")
    exif = Image.Exif()
    exif[305] = "Blanket bytes"
    source.info.update(exif=exif.tobytes(), xmp=b"packet", icc_profile=b"profile")
    decoded = Image.decode(source.encode(format))
    assert decoded.getexif()[305] == "Blanket bytes"
    assert decoded.info["xmp"] == b"packet"
    assert decoded.info["icc_profile"] == b"profile"
    stripped = Image.decode(source.encode(format, exif=None, xmp=None, icc_profile=None))
    assert 305 not in stripped.getexif()
    assert not stripped.info.get("xmp")
    assert not stripped.info.get("icc_profile")


@pytest.mark.parametrize("compressor", [LosslessImageCompressor(effort=1), LossyImageCompressor(effort=1)])
def test_byte_encoding_uses_jpeg_options_and_compressors(compressor):
    source = Image.new("RGB", (17, 9), (31, 73, 127))
    options = {"quality": 85, "subsampling": 0, "progressive": True, "optimize": True, "compressor": compressor}
    encoded = source.encode("jpg", **options)
    stream = BytesIO()
    source.save(stream, "JPEG", **options)
    assert encoded == stream.getvalue()
    assert Image.decode(encoded, formats=["JPEG"]).format == "JPEG"
    assert PillowImage.open(BytesIO(encoded)).info["progressive"]


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("format", ["PNG", "GIF", "WEBP"])
def test_byte_sequences_preserve_frames_controls_and_positions(format, lazy):
    frames = [Image.new("RGBA", (7, 5), color) for color in ("red", "green", "blue")]
    options = {"save_all": True, "append_images": frames[1:], "duration": [20, 40, 60], "loop": 2}
    if format == "WEBP":
        options["lossless"] = True
    encoded = frames[0].encode(format, **options)
    stream = BytesIO()
    frames[0].save(stream, format, **options)
    assert encoded == stream.getvalue()
    image = Image.decode(encoded, lazy=lazy, frame_cache_size=3)
    assert image.n_frames == 3
    image.seek(1)
    image.putpixel((0, 0), (10, 20, 30, 255))
    pixels = image.tobytes()
    current = Image.decode(image.encode("PNG"))
    assert current.n_frames == 1
    assert current.tobytes() == pixels
    exported = Image.decode(image.encode(format, save_all=True, **({"lossless": True} if format == "WEBP" else {})))
    assert image.tell() == 1
    assert image.tobytes() == pixels
    assert exported.n_frames == 3
    for i, duration in enumerate((20, 40, 60)):
        exported.seek(i)
        assert exported.info["duration"] == duration
        assert exported.info["loop"] == 2
    exported.seek(1)
    assert exported.tobytes() == pixels
    image.close()
    with pytest.raises(ValueError, match="closed"):
        image.encode(format)


def test_pdf_byte_encoding_matches_stream():
    first, second = Image.new("RGB", (7, 5), "red"), Image.new("L", (3, 9), 42)
    options = {"save_all": True, "append_images": [second]}
    stream = BytesIO()
    first.save(stream, "PDF", **options)
    assert first.encode("PDF", **options) == stream.getvalue()


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("buffer_type", [bytes, bytearray, memoryview])
def test_decode_accepts_buffers_and_owns_mutable_input(lazy, buffer_type):
    first, second = Image.new("RGBA", (3, 2), "red"), Image.new("RGBA", (3, 2), "blue")
    storage = bytearray(first.encode("PNG", save_all=True, append_images=[second], duration=20))
    image = Image.decode(buffer_type(storage), lazy=lazy)
    storage[:] = bytes(len(storage))
    image.seek(1)
    assert image.tobytes() == second.tobytes()
    assert image.filename == ""


def test_decode_accepts_strided_buffer():
    encoded = Image.new("L", (3, 2), 42).encode("PNG")
    storage = bytearray(len(encoded) * 2)
    storage[::2] = encoded
    assert Image.decode(memoryview(storage)[::2]).getpixel((0, 0)) == 42


@pytest.mark.parametrize("data", [None, 100, "photo.png", [1, 2, 3], BytesIO(b"data")])
def test_decode_rejects_non_buffers(data):
    with pytest.raises(TypeError, match="bytes-like"):
        Image.decode(data)


@pytest.mark.parametrize("data", [b"", b"not an image", b"\x89PNG\r\n\x1a\n"])
def test_decode_rejects_invalid_encoded_data(data):
    with pytest.raises(UnidentifiedImageError):
        Image.decode(data)


@pytest.mark.parametrize(
    "options,error",
    [
        ({"formats": "PNG"}, TypeError),
        ({"lazy": 1}, TypeError),
        ({"frame_cache_size": True}, TypeError),
        ({"frame_cache_size": 1.5}, TypeError),
        ({"frame_cache_size": 0}, ValueError),
        ({"frame_cache_size": -1}, ValueError),
    ],
)
def test_decode_validates_options(options, error):
    encoded = Image.new("L", (1, 1)).encode("PNG")
    with pytest.raises(error):
        Image.decode(encoded, **options)


def test_decode_enforces_format_allowlist_and_does_not_read_paths(tmp_path):
    encoded = Image.new("L", (1, 1)).encode("PNG")
    with pytest.raises(UnidentifiedImageError):
        Image.decode(encoded, formats=["JPEG"])
    path = tmp_path / "image.png"
    path.write_bytes(encoded)
    with pytest.raises(UnidentifiedImageError):
        Image.decode(bytes(path))


def test_encode_validates_format_options_and_image_state():
    image = Image.new("RGB", (3, 2))
    for format in (None, 1, b"PNG"):
        with pytest.raises(TypeError, match="format must be a string"):
            image.encode(format)
    with pytest.raises(UnsupportedCodecError):
        image.encode("unknown")
    with pytest.raises(TypeError, match="unsupported PNG save option"):
        image.encode("PNG", quality=85)
    with pytest.raises(TypeError, match="compressor must"):
        image.encode("PNG", compressor=object())
    with pytest.raises(TypeError, match="path or binary stream"):
        image.save(None, "PNG")
    image.close()
    with pytest.raises(ValueError, match="closed"):
        image.encode("PNG")


def test_sequence_encoding_restores_position_on_failure():
    first = Image.new("RGBA", (3, 2), "red")
    encoded = first.encode("PNG", save_all=True, append_images=[Image.new("RGBA", (3, 2), "blue")])
    image = Image.decode(encoded, lazy=True, frame_cache_size=2)
    image.seek(1)
    pixels = image.tobytes()
    with pytest.raises(ValueError):
        image.encode("PNG", save_all=True, duration=[1])
    assert image.tell() == 1
    assert image.tobytes() == pixels
    with pytest.raises(UnsupportedCodecError):
        image.encode("PNG", save_all=True, compressor=LosslessImageCompressor())
