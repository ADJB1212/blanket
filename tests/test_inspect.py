from __future__ import annotations

import struct
import zlib
from io import BytesIO, StringIO

import pytest
from PIL import Image as PillowImage

from blanket import Image, ImageCms, PngImagePlugin, UnidentifiedImageError, features


def encode(mode="RGB", format="PNG", depth=8, **options):
    image = Image.new(mode, (17, 9))
    if depth != 8:
        image = image.convert(mode, bit_depth=depth)
    stream = BytesIO()
    image.save(stream, format, **options)
    return stream.getvalue()


@pytest.mark.parametrize(
    "format,mode,expected",
    [
        ("PNG", "1", "1"),
        ("PNG", "L", "L"),
        ("PNG", "LA", "LA"),
        ("PNG", "RGB", "RGB"),
        ("PNG", "RGBA", "RGBA"),
        ("JPEG", "L", "L"),
        ("JPEG", "RGB", "RGB"),
        ("JPEG", "CMYK", "CMYK"),
        ("TIFF", "L", "L"),
        ("TIFF", "RGB", "RGB"),
        ("TIFF", "RGBA", "RGBA"),
        ("TIFF", "CMYK", "CMYK"),
        ("TIFF", "LAB", "LAB"),
        ("TIFF", "F", "F"),
        ("WEBP", "RGB", "RGB"),
        ("JXL", "L", "L"),
        ("JXL", "RGB", "RGB"),
        ("BMP", "RGB", "RGB"),
        ("ICO", "RGBA", "RGBA"),
        ("PPM", "1", "1"),
        ("PPM", "L", "L"),
        ("PPM", "RGB", "RGB"),
        ("PAM", "RGBA", "RGBA"),
        ("TGA", "RGB", "RGB"),
        ("TGA", "RGBA", "RGBA"),
        ("QOI", "RGB", "RGB"),
        ("JPEG2000", "RGB", "RGB"),
        ("PCX", "RGB", "RGB"),
        ("PCX", "1", "1"),
        ("SGI", "RGBA", "RGBA"),
        ("XBM", "1", "1"),
        ("XPM", "RGB", "P"),
        ("MSP", "1", "1"),
    ],
)
def test_formats_and_modes(format, mode, expected):
    data = encode(mode, format)
    header = Image.inspect(BytesIO(data), formats=[format])
    assert isinstance(header, Image.ImageHeader)
    assert (header.format, header.size, header.mode) == (format, (17, 9), expected)
    assert (header.width, header.height) == (17, 9)
    assert header.bit_depth == (1 if mode == "1" else 32 if mode == "F" else 8)
    if format not in ("JXL", "XPM", "PAM"):
        assert PillowImage.open(BytesIO(data)).size == header.size


@pytest.mark.parametrize("format", ["PNG", "TIFF", "PPM", "JPEG2000", "JXL"])
@pytest.mark.parametrize("depth", [10, 12, 16])
@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
def test_wide_samples(format, depth, mode):
    if format == "PPM" and mode == "RGBA":
        format = "PAM"
    data = encode(mode, format, depth)
    header = Image.inspect(BytesIO(data))
    assert (header.mode, header.size) == (mode, (17, 9))
    assert header.bit_depth == (depth if format == "JPEG2000" else 16)


@pytest.mark.parametrize("format,depth", [("AVIF", 8), ("HEIF", 8), ("HEIF", 10), ("HEIF", 12)])
def test_heif_and_avif_headers(format, depth):
    if format == "HEIF" and not features.check("heif_encoder"):
        pytest.skip("HEIF encoder unavailable")
    header = Image.inspect(BytesIO(encode("RGB", format, depth)))
    assert (header.format, header.size, header.mode) == (format, (17, 9), "RGB")
    assert header.bit_depth == (8 if format == "AVIF" else depth)


def png_chunk(kind, data):
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


@pytest.mark.parametrize("color,depth,mode", [(0, 1, "1"), (0, 2, "L"), (0, 4, "L"), (3, 4, "P"), (6, 16, "RGBA")])
def test_png_header_without_pixels_or_decoded_pixel_budget(color, depth, mode):
    data = b"\x89PNG\r\n\x1a\n" + png_chunk(b"IHDR", struct.pack(">IIBBBBB", 100000, 100000, depth, color, 0, 0, 0))
    header = Image.inspect(BytesIO(data))
    assert (header.size, header.mode, header.bit_depth) == ((100000, 100000), mode, depth)
    with pytest.raises(UnidentifiedImageError):
        Image.open(BytesIO(data))


@pytest.mark.parametrize("format", ["PNG", "JPEG", "TIFF", "WEBP", "JXL"])
def test_container_metadata(format):
    exif = Image.Exif()
    exif[274] = 6
    exif[270] = "header inspection"
    profile = ImageCms.createProfile("sRGB").tobytes()
    data = encode(format=format, exif=exif, xmp=b"<test/>", icc_profile=profile)
    header = Image.inspect(BytesIO(data))
    assert header.info == Image.open(BytesIO(data)).info
    assert header.info["xmp"] == b"<test/>"
    assert header.info["icc_profile"] == profile
    parsed = Image.Exif()
    parsed.load(header.info["exif"])
    assert parsed[274] == 6


def test_png_text_metadata_after_pixels_and_corrupt_pixels():
    info = PngImagePlugin.PngInfo()
    info.add_text("label", "inspection", zip=True)
    info.add_itxt("unicode", "caf\u00e9")
    data = encode(pnginfo=info)
    pos = data.index(b"IDAT") + 4
    damaged = data[:pos] + bytes([data[pos] ^ 255]) + data[pos + 1 :]
    assert Image.inspect(BytesIO(damaged)).info == Image.inspect(BytesIO(data)).info
    assert Image.inspect(BytesIO(data)).info["label"] == "inspection"
    with pytest.raises(UnidentifiedImageError):
        Image.open(BytesIO(damaged))


def test_paths_streams_and_allowlists(tmp_path):
    data = encode()
    path = tmp_path / "image.bin"
    path.write_bytes(data)
    expected = Image.inspect(path)
    assert Image.inspect(str(path)) == Image.inspect(bytes(path)) == expected
    stream = BytesIO(data)
    stream.seek(5)
    assert Image.inspect(stream, formats=("png",)) == expected
    assert stream.tell() == 5 and not stream.closed
    with pytest.raises(UnidentifiedImageError, match="not in formats"):
        Image.inspect(stream, formats=["JPEG"])
    assert stream.tell() == 5
    with pytest.raises(TypeError, match="list or tuple"):
        Image.inspect(stream, formats="PNG")
    with pytest.raises(ValueError, match="binary data"):
        Image.inspect(StringIO("image"))
    assert Image.inspect(BytesIO(encode(format="JPEG")), formats=["jpg"]).format == "JPEG"


def test_nonseekable_stream():
    class Stream:
        def read(self):
            return encode()

    assert Image.inspect(Stream()).size == (17, 9)


@pytest.mark.parametrize("data", [b"", b"garbage", b"GIF89a", b"\x89PNG\r\n\x1a\n", b"BM", b"qoif", b"II*\0", b"\xff\xd8\xff", b"\x01\xda", b"\0\0\x01\0"])
def test_invalid_or_truncated_headers(data):
    stream = BytesIO(data)
    stream.seek(1)
    with pytest.raises(UnidentifiedImageError):
        Image.inspect(stream)
    assert stream.tell() == 1


def test_animation_and_mixed_tiff_pages():
    frames = [PillowImage.new("RGB", (17, 9), "red"), PillowImage.new("RGB", (17, 9), "blue")]
    for format in ("GIF", "PNG", "WEBP", "TIFF"):
        output = BytesIO()
        frames[0].save(output, format, save_all=True, append_images=frames[1:], duration=50, loop=2)
        header = Image.inspect(output)
        assert (header.size, header.format) == ((17, 9), format)
        assert header.mode == ("P" if format == "GIF" else "RGB")
    output = BytesIO()
    frames[0].save(output, "TIFF", save_all=True, append_images=[PillowImage.new("L", (3, 5))])
    assert (Image.inspect(output).size, Image.inspect(output).mode) == ((17, 9), "RGB")


def test_eps_inspection_without_ghostscript():
    data = b"%!PS-Adobe-3.0 EPSF-3.0\n%%BoundingBox: -1 2 40 25\n"
    assert Image.inspect(BytesIO(data)).size == (41, 23)


@pytest.mark.parametrize("bits,mode", [(8, "L"), (16, "I;16"), (32, "I"), (-32, "F"), (-64, "F")])
def test_fits_header_without_samples(bits, mode):
    cards = ["SIMPLE  =                    T", f"BITPIX  = {bits:20}", "NAXIS   =                    2", "NAXIS1  =                    7", "NAXIS2  =                    3", "END"]
    data = b"".join(card.encode().ljust(80) for card in cards)
    header = Image.inspect(BytesIO(data))
    assert (header.format, header.size, header.mode, header.bit_depth) == ("FITS", (7, 3), mode, abs(bits))


@pytest.mark.parametrize("depth", [1, 2, 4, 8])
def test_indexed_png_encoded_depth(depth):
    image = PillowImage.new("P", (17, 9))
    image.putpalette([i for i in range(256) for _ in range(3)])
    stream = BytesIO()
    image.save(stream, "PNG", bits=depth, transparency=0)
    header = Image.inspect(stream)
    assert (header.mode, header.bit_depth) == ("P", depth)


def test_remaining_header_only_formats():
    spider = [0.0] * 256
    for key, value in {1: 1, 2: 3, 5: 1, 12: 4, 13: 64, 22: 1024, 23: 16}.items():
        spider[key - 1] = value
    pixar = bytearray(1024)
    pixar[:4] = b"\x80\xe8\0\0"
    struct.pack_into("<HH", pixar, 416, 3, 4)
    struct.pack_into("<HH", pixar, 424, 14, 2)
    sun = struct.pack(">8I", 0x59A66A95, 4, 3, 24, 36, 1, 0, 0)
    dds = bytearray(128)
    dds[:4] = b"DDS "
    struct.pack_into("<7I", dds, 4, 124, 0x1007, 4, 8, 16, 0, 1)
    struct.pack_into("<2I4s", dds, 76, 32, 4, b"DXT1")
    for format, data, size, mode, depth in [
        ("SPIDER", struct.pack("<256f", *spider), (4, 3), "F", 32),
        ("SPIDER", struct.pack(">256f", *spider), (4, 3), "F", 32),
        ("PIXAR", pixar, (4, 3), "RGB", 8),
        ("SUN", sun, (4, 3), "RGB", 8),
        ("DDS", dds, (8, 4), "RGB", 8),
        ("GIF", b"GIF89a" + struct.pack("<HHBBB", 65535, 65535, 0, 0, 0), (65535, 65535), "P", 8),
    ]:
        header = Image.inspect(BytesIO(data))
        assert (header.format, header.size, header.mode, header.bit_depth) == (format, size, mode, depth)


def test_ico_selects_largest_entry_and_rejects_malformed_bitmap():
    entries = []
    payloads = []
    offset = 6 + 2 * 16
    for size in (32, 16):
        stream = BytesIO()
        PillowImage.new("RGBA", (size, size)).save(stream, "PNG")
        data = stream.getvalue()
        entries.append(struct.pack("<BBBBHHII", size, size, 0, 0, 1, 32, len(data), offset))
        payloads.append(data)
        offset += len(data)
    data = b"\0\0\x01\0\x02\0" + b"".join(entries + payloads)
    assert Image.inspect(BytesIO(data)).size == Image.open(BytesIO(data)).size == (32, 32)
    invalid = b"\0\0\x01\0\x01\0" + struct.pack("<BBBBHHII", 16, 16, 0, 0, 1, 32, 0, 22)
    with pytest.raises(UnidentifiedImageError):
        Image.inspect(BytesIO(invalid))
    stream = BytesIO()
    PillowImage.new("RGBA", (16, 16)).save(stream, "ICO", bitmap_format="bmp")
    assert Image.inspect(stream).size == (16, 16)


def test_jpeg_header_precision():
    data = bytearray(encode(format="JPEG"))
    pos = data.index(b"\xff\xc0")
    data[pos + 1] = 0xC1
    data[pos + 4] = 12
    header = Image.inspect(BytesIO(data))
    assert (header.mode, header.bit_depth) == ("RGB", 12)


def test_tiff_float_and_integer_depths():
    for mode, depth, expected in [("1", 1, "1"), ("I;16", 16, "L"), ("I", 32, "I"), ("F", 32, "F")]:
        stream = BytesIO()
        PillowImage.new(mode, (4, 3)).save(stream, "TIFF")
        header = Image.inspect(stream)
        assert (header.mode, header.bit_depth) == (expected, depth)
