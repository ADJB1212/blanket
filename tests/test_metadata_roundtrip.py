from __future__ import annotations

import struct
import zlib
from fractions import Fraction
from io import BytesIO
from math import isnan

import pytest
from helpers import encode_image
from PIL import Image as PILImage, ImageCms, PngImagePlugin as PILPng
from PIL.TiffImagePlugin import IFDRational

from blanket import Image, ImageChops, ImageEnhance, ImageOps, PngImagePlugin
from blanket._exif import read_metadata
from blanket.Compressor import LosslessImageCompressor, LossyImageCompressor


@pytest.fixture
def profile() -> bytes:
    return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


@pytest.mark.parametrize("short_writes", [False, True])
def test_png_chunk_writer_interoperability(short_writes: bool) -> None:
    class ShortWriter(BytesIO):
        def write(self, data: bytes) -> int:
            return super().write(data[:3])

    output = ShortWriter() if short_writes else BytesIO()
    reference = BytesIO()
    BytesIO.write(output, PngImagePlugin._MAGIC)
    reference.write(b"\x89PNG\r\n\x1a\n")
    chunks = [(b"IHDR", PngImagePlugin.o32(2), PngImagePlugin.o32(1), b"\x08\x02\x00\x00\x00"), (b"tEXt", b"Note\0", b"generated metadata"), (b"IDAT", zlib.compress(b"\0\xff\0\0\0\0\xff")), (b"IEND", b"")]
    for cid, *data in chunks:
        PngImagePlugin.putchunk(output, cid, *data)
        PILPng.putchunk(reference, cid, *data)
    assert output.getvalue() == reference.getvalue()
    assert PngImagePlugin.o16(0x1234) == b"\x12\x34"
    for module in (Image, PILImage):
        with module.open(BytesIO(output.getvalue())) as decoded:
            assert decoded.convert("RGB").tobytes() == b"\xff\0\0\0\0\xff"
            assert decoded.info["Note"] == "generated metadata"


def test_apng_public_control_enums_interoperate_with_pillow() -> None:
    first = Image.new("RGBA", (3, 2), "red")
    second = Image.new("RGBA", first.size, "blue")
    data = first.encode("PNG", save_all=True, append_images=[second], duration=[20, 40], disposal=[PngImagePlugin.Disposal.OP_BACKGROUND, PngImagePlugin.Disposal.OP_PREVIOUS], blend=[PngImagePlugin.Blend.OP_SOURCE, PngImagePlugin.Blend.OP_OVER])
    for module in (Image, PILImage):
        with module.open(BytesIO(data)) as decoded:
            for i, frame in enumerate((first, second)):
                decoded.seek(i)
                decoded.load()
                assert decoded.convert("RGBA").tobytes() == frame.tobytes()
                assert decoded.info["disposal"] == (PILPng.Disposal.OP_BACKGROUND, PILPng.Disposal.OP_PREVIOUS)[i]
                assert decoded.info["blend"] == (PILPng.Blend.OP_SOURCE, PILPng.Blend.OP_OVER)[i]


def exif_tags() -> Image.Exif:
    exif = Image.Exif()
    exif[270] = "A description stored outside the IFD"
    exif[274] = 1
    exif[305] = "Blanket"
    exif[34665] = {36867: "2026:10:23 10:00:00", 33434: Fraction(1, 125), 37500: b"opaque maker note", 40965: {1: "R98"}}
    exif[34853] = {1: "N", 2: (IFDRational(40), IFDRational(30), IFDRational(15, 2))}
    return exif


@pytest.mark.parametrize("endian", ["<", ">"])
def test_tiff_merges_many_tags_preserving_raster_fields_and_metadata(endian: str) -> None:
    exif = Image.Exif()
    exif.endian = endian
    exif[256], exif[257] = 999, 888
    for tag in range(45000, 46000):
        exif[tag] = tag - 44999
    image = Image.new("RGB", (3, 2), (17, 83, 191))
    data = image.encode("TIFF", exif=exif, xmp=b"<title>kept</title>")
    for module in (Image, PILImage):
        with module.open(BytesIO(data)) as restored:
            assert restored.size == image.size and restored.tobytes() == image.tobytes()
            tags = restored.getexif()
            assert tags[256] == 3 and tags[257] == 2
            assert all(tags[tag] == tag - 44999 for tag in range(45000, 46000))
            assert tags[700] == b"<title>kept</title>"


@pytest.mark.parametrize("endian", ["<", ">"])
@pytest.mark.parametrize("nested", [False, True])
def test_string_user_comment_roundtrips_like_pillow(endian: str, nested: bool) -> None:
    exif = Image.Exif()
    reference = PILImage.Exif()
    for value in (exif, reference):
        value.endian = endian
        if nested:
            value[34665] = {37510: "A comment"}
        else:
            value[37510] = "A comment"
    for loader in (Image.Exif, PILImage.Exif):
        restored = loader()
        restored.load(exif.tobytes())
        expected = loader()
        expected.load(reference.tobytes())
        assert (restored.get_ifd(34665) if nested else restored)[37510] == (expected.get_ifd(34665) if nested else expected)[37510] == "A comment"


@pytest.mark.parametrize("mode", ["L", "RGB"])
@pytest.mark.parametrize("operation", ["mirror", "flip", "grayscale", "expand", "pad", "scale", "autocontrast", "equalize", "invert", "posterize", "solarize", "Brightness", "Contrast", "Color", "Sharpness", "difference", "chops_invert", "offset"])
def test_processing_preserves_metadata_and_cached_exif_edits(mode, operation, profile):
    source = Image.new(mode, (3, 2), 73)
    image = Image.decode(source.encode("PNG", exif=exif_tags(), icc_profile=profile, xmp=b"<metadata/>"))
    original = image.info.copy()
    image.getexif()[305] = "Edited"
    image.getexif().get_ifd(34665)[36867] = "2026:11:01 12:00:00"
    image.info["custom"] = "retained"
    if operation in ImageEnhance.__all__:
        result = getattr(ImageEnhance, operation)(image).enhance(1.5)
    elif operation == "difference":
        result = ImageChops.difference(image, image)
    elif operation == "chops_invert":
        result = ImageChops.invert(image)
    elif operation == "offset":
        result = ImageChops.offset(image, 1)
    else:
        args = {"expand": (1,), "pad": ((5, 2),), "scale": (1,), "posterize": (4,)}.get(operation, ())
        result = getattr(ImageOps, operation)(image, *args)
    assert result.getexif()[305] == "Edited"
    assert result.getexif().get_ifd(34665)[36867] == "2026:11:01 12:00:00"
    assert result.info["icc_profile"] == profile
    assert result.info["xmp"] == b"<metadata/>"
    assert result.info["custom"] == "retained"
    assert result.info is not image.info
    result.getexif()[305] = "result"
    result.info["custom"] = "result"
    assert image.getexif()[305] == "Edited"
    assert image.info == {**original, "custom": "retained"}


@pytest.mark.parametrize("mode", ["L", "LA", "RGB", "RGBA", "P", "PA"])
def test_pa_conversion_preserves_metadata_and_cached_exif_edits(mode, profile):
    image = Image.new(mode, (3, 2), 73)
    image.info.update(exif=exif_tags().tobytes(), icc_profile=profile, xmp=b"<metadata/>", custom="retained")
    image.getexif()[305] = "Edited"
    image.getexif().get_ifd(34665)[36867] = "2026:11:01 12:00:00"
    result = image.convert("PA")
    assert result.mode == "PA"
    assert result.info == image._metadata_info()
    assert result.info is not image.info
    reopened = Image.decode(result.encode("PNG"))
    assert reopened.info["icc_profile"] == profile
    assert reopened.info["xmp"] == b"<metadata/>"
    assert reopened.getexif()[305] == "Edited"
    assert reopened.getexif().get_ifd(34665)[36867] == "2026:11:01 12:00:00"
    result.getexif()[305] = "result"
    result.info["custom"] = "result"
    assert image.getexif()[305] == "Edited"
    assert image.info["custom"] == "retained"


@pytest.mark.parametrize("format", ["JPEG", "PNG", "WEBP", "JXL", "HEIF", "TIFF"])
def test_metadata_preserved_and_editable(format: str, profile: bytes, tmp_path) -> None:
    image = Image.new("RGB", (16, 16), (30, 70, 100))
    xmp = b'<x:xmpmeta xmlns:x="adobe:ns:meta/">test packet</x:xmpmeta>'
    data = encode_image(image, format, exif=exif_tags(), xmp=xmp, icc_profile=profile)
    loaded = Image.open(BytesIO(data))
    assert loaded.getexif()[270] == exif_tags()[270]
    assert loaded.getexif().get_ifd(34665)[33434] == pytest.approx(1 / 125)
    assert loaded.getexif().get_ifd(40965)[1] == "R98"
    assert loaded.getexif().get_ifd(34853)[2] == (40, 30, 7.5)
    assert loaded.info["xmp"] == xmp
    assert loaded.info["icc_profile"] == profile
    loaded.getexif()[305] = "Edited"
    loaded.getexif().get_ifd(34665)[36867] = "2026:10:24 10:00:00"
    path = tmp_path / "image"
    options = {"xmp": loaded.info["xmp"]} if format == "JPEG" else {}
    loaded.save(path, format, **options)
    reopened = Image.open(path)
    assert reopened.getexif()[305] == "Edited"
    assert reopened.getexif().get_ifd(34665)[36867] == "2026:10:24 10:00:00"
    assert reopened.info["xmp"] == xmp
    assert reopened.info["icc_profile"] == profile
    if format in ("JPEG", "PNG", "WEBP", "TIFF"):
        pillow = PILImage.open(path)
        assert pillow.getexif()[305] == "Edited"
        assert pillow.getexif().get_ifd(34665)[36867] == "2026:10:24 10:00:00"
        assert pillow.info["icc_profile"] == profile


@pytest.mark.parametrize("format", ["PNG", "JPEG", "JXL", "TIFF", "WEBP"])
def test_metadata_text_does_not_override_container_detection(format: str) -> None:
    image = Image.new("RGB", (17, 19), (30, 70, 100))
    expected = Image.open(BytesIO(encode_image(image, format)))
    xmp = b'<x:xmpmeta xmlns:x="adobe:ns:meta/">_width _height</x:xmpmeta>'
    actual = Image.open(BytesIO(encode_image(image, format, xmp=xmp)))
    assert (actual.format, actual.size, actual.mode) == (format, expected.size, expected.mode)
    assert actual.tobytes() == expected.tobytes()
    assert actual.info["xmp"] == xmp


@pytest.mark.parametrize("endian", ["<", ">"])
def test_exif_mapping_nested_ifds_and_thumbnail(endian: str) -> None:
    exif = exif_tags()
    exif.endian = endian
    thumb = BytesIO()
    PILImage.new("RGB", (3, 2), "red").save(thumb, "JPEG")
    exif.thumbnail = thumb.getvalue()
    raw = exif.tobytes()
    pillow = PILImage.Exif()
    pillow.load(raw)
    assert pillow[270] == exif[270]
    assert pillow.get_ifd(34665)[33434] == IFDRational(1, 125)
    parsed = Image.Exif()
    parsed.load(raw)
    assert parsed.thumbnail == thumb.getvalue()
    assert parsed.get_ifd(1)[514] == len(thumb.getvalue())
    parsed.get_ifd(34853)[1] = "S"
    del parsed[274]
    parsed[305] = "Changed"
    pillow.load(parsed.tobytes())
    assert pillow[305] == "Changed"
    assert 274 not in pillow
    assert pillow.get_ifd(34853)[1] == "S"
    again = Image.Exif()
    again.load(parsed.tobytes())
    assert again.thumbnail == thumb.getvalue()


@pytest.mark.parametrize("endian", ["<", ">"])
@pytest.mark.parametrize("count", [1, 3, 4097])
@pytest.mark.parametrize("kind,code", [(1, "B"), (3, "H"), (4, "I"), (6, "b"), (8, "h"), (9, "i"), (11, "f"), (12, "d"), (13, "I"), (5, "II"), (10, "ii")])
def test_exif_numeric_arrays_and_tiff_relocation(endian: str, count: int, kind: int, code: str) -> None:
    values = (1, 2) * count if kind in (5, 10) else (3,) * count
    payload = struct.pack(endian + code * count, *values)
    field = payload.ljust(4, b"\0") if len(payload) <= 4 else struct.pack(endian + "I", 26)
    raw = (b"II" if endian == "<" else b"MM") + struct.pack(endian + "HIH", 42, 8, 1)
    raw += struct.pack(endian + "HHI", 65000, kind, count) + field + bytes(4)
    if len(payload) > 4:
        raw += payload
    expected = 0.5 if kind in (5, 10) else 3
    expected = expected if count == 1 else (expected,) * count
    exif = Image.Exif()
    exif.load(raw)
    assert exif[65000] == expected
    exif[65000] = 2 if count == 1 else (2,) * count
    reloaded = Image.Exif()
    reloaded.load(exif.tobytes())
    assert reloaded[65000] == exif[65000]
    image = Image.open(BytesIO(encode_image(Image.new("L", (1, 1)), "TIFF", exif=raw)))
    assert image.getexif()[65000] == expected


@pytest.mark.parametrize("endian", ["<", ">"])
@pytest.mark.parametrize("signed", [False, True])
@pytest.mark.parametrize("count", [1, 3])
def test_exif_rational_preserves_zero_denominator(endian: str, signed: bool, count: int) -> None:
    numerator = -7 if signed else 7
    raw = (b"II" if endian == "<" else b"MM") + struct.pack(endian + "HIHHHII", 42, 8, 1, 65000, 10 if signed else 5, count, 26)
    raw += bytes(4) + struct.pack(endian + ("ii" if signed else "II"), numerator, 0) * count
    exif = Image.Exif()
    for _ in range(2):
        exif.load(raw)
        values = (exif[65000],) if count == 1 else exif[65000]
        for value in values:
            assert isnan(value)
            assert (value.numerator, value.denominator) == (numerator, 0)
        raw = exif.tobytes()


@pytest.mark.parametrize("format", ["JPEG", "PNG", "WEBP", "JXL", "HEIF", "TIFF"])
def test_metadata_overrides_and_removal(format: str, profile: bytes) -> None:
    image = Image.new("RGB", (8, 8))
    image.info.update(exif=exif_tags().tobytes(), xmp=b"original", icc_profile=profile)
    original = image.info.copy()
    override = Image.Exif()
    override[305] = "Replacement"
    opened = Image.open(BytesIO(encode_image(image, format, exif=override, xmp="replacement", icc_profile=None)))
    assert opened.getexif()[305] == "Replacement"
    assert opened.info["xmp"] == b"replacement"
    removed = Image.open(BytesIO(encode_image(image, format, exif=None, xmp=b"", icc_profile=b"")))
    assert 305 not in removed.getexif()
    assert 270 not in removed.getexif()
    assert "xmp" not in removed.info
    assert image.info == original


def test_tiff_getexif_includes_raster_tags(profile: bytes) -> None:
    raw = encode_image(Image.new("RGB", (8, 6)), "TIFF", exif=exif_tags(), icc_profile=profile)
    image = Image.open(BytesIO(raw))
    pillow = PILImage.open(BytesIO(raw))
    for tag in (256, 257, 258, 259, 262, 273, 277, 278, 279, 284, 305, 34675):
        assert image.getexif()[tag] == pillow.getexif()[tag]


@pytest.mark.parametrize("compressed", [False, True])
def test_png_text_interop_and_roundtrip(compressed: bool) -> None:
    info = PngImagePlugin.PngInfo()
    info.add_text("Author", "Andr\xe9", zip=compressed)
    info.add_text("Unicode", "snowman \u2603", zip=compressed)
    info.add_itxt("Title", "Translated", lang="en", tkey="Title translation", zip=compressed)
    info.add(b"vpAg", b"private chunk", after_idat=True)
    data = encode_image(Image.new("RGBA", (4, 3)), "PNG", pnginfo=info)
    pillow = PILImage.open(BytesIO(data))
    opened = Image.open(BytesIO(data))
    for key in ("Author", "Unicode", "Title"):
        assert opened.info[key] == pillow.info[key]
    assert opened.info["Title"].lang == "en"
    assert opened.info["Title"].tkey == "Title translation"
    assert data.index(b"vpAg") > data.index(b"IDAT")
    second = Image.open(BytesIO(encode_image(opened, "PNG")))
    assert second.info["Unicode"] == "snowman \u2603"
    assert second.info["Title"].tkey == "Title translation"
    assert "Author" not in Image.open(BytesIO(encode_image(opened, "PNG", pnginfo=PngImagePlugin.PngInfo()))).info
    native_info = PILPng.PngInfo()
    native_info.add_itxt("Unicode", "snowman \u2603", zip=True)
    stream = BytesIO()
    PILImage.new("L", (2, 2)).save(stream, "PNG", pnginfo=native_info)
    assert Image.open(stream).info["Unicode"] == "snowman \u2603"


@pytest.mark.parametrize("format", ["PNG", "WEBP"])
def test_animation_preserves_metadata(format: str, profile: bytes) -> None:
    source = Image.new("RGBA", (5, 4), "red")
    source.info.update(exif=exif_tags().tobytes(), xmp=b"animation", icc_profile=profile)
    data = encode_image(source, format, save_all=True, append_images=[Image.new("RGBA", source.size, "blue")], duration=[20, 40], loop=2)
    opened = Image.open(BytesIO(data))
    assert opened.n_frames == 2
    for i in range(2):
        opened.seek(i)
        assert opened.getexif()[305] == "Blanket"
        assert opened.info["xmp"] == b"animation"
        assert opened.info["icc_profile"] == profile
    pillow = PILImage.open(BytesIO(data))
    assert pillow.n_frames == 2
    assert pillow.info["icc_profile"] == profile


@pytest.mark.parametrize("mode", ["RGB", "P", "PA"])
def test_exif_edits_copy_and_orientation(mode: str) -> None:
    image = Image.new(mode, (3, 2))
    exif = image.getexif()
    assert image.getexif() is exif
    exif[305] = "new EXIF"
    copied = image.copy()
    copied.getexif()[305] = "independent"
    assert image.getexif()[305] == "new EXIF"
    assert Image.open(BytesIO(encode_image(image, "PNG"))).getexif()[305] == "new EXIF"
    image.info["exif"] = exif_tags().tobytes()
    assert image.getexif()[305] == "Blanket"
    image.getexif()[274] = 6
    result = ImageOps.exif_transpose(image)
    assert result.size == (2, 3)
    assert 274 not in result.getexif()
    del image.info["exif"]
    empty = image.getexif()
    assert not empty
    assert image.getexif() is empty
    image.close()
    with pytest.raises(ValueError, match="closed"):
        image.getexif()


@pytest.mark.parametrize("profile", [False, True])
def test_exif_cache_tracks_metadata_replacement(profile: bool) -> None:
    image = Image.new("RGB", (1, 1))
    key = "Raw profile type exif" if profile else "exif"
    first = image.getexif()
    assert image.getexif() is first
    raw = exif_tags().tobytes()
    image.info[key] = raw.hex() if profile else raw
    cached = image.getexif()
    assert cached is not first
    assert image.getexif() is cached
    cached[305] = "edited"
    assert image.getexif()[305] == "edited"
    replacement = Image.Exif()
    replacement[305] = "replacement"
    raw = replacement.tobytes()
    image.info[key] = raw.hex() if profile else raw
    assert image.getexif()[305] == "replacement"
    del image.info[key]
    assert not image.getexif()
    image.close()
    with pytest.raises(ValueError, match="closed"):
        image.getexif()


def test_jpeg_multisegment_icc_and_size_limits() -> None:
    image = Image.new("RGB", (2, 2))
    profile = bytes(range(256)) * 700
    opened = Image.open(BytesIO(encode_image(image, "JPEG", icc_profile=profile)))
    assert opened.info["icc_profile"] == profile
    with pytest.raises(ValueError, match="XMP data is too long"):
        encode_image(image, "JPEG", xmp=b"x" * 65534)


@pytest.mark.parametrize("raw", [b"bad", b"II*\0\xff\xff\xff\xff", b"II*\0\x08\0\0\0\xff\xff"])
def test_invalid_exif_is_bounded(raw: bytes) -> None:
    with pytest.raises(ValueError):
        Image.Exif().load(raw)


def test_png_text_expansion_is_bounded() -> None:
    payload = b"Comment\0\0" + zlib.compress(b"x" * (1024 * 1024 + 1))
    raw = b"\x89PNG\r\n\x1a\n" + struct.pack(">I", len(payload)) + b"zTXt" + payload + bytes(4)
    assert "Comment" not in read_metadata(raw, "PNG")


@pytest.mark.parametrize("options", [{"exif": "invalid"}, {"icc_profile": 1}, {"pnginfo": {}}])
def test_metadata_options_validated_before_write(options: dict[str, object]) -> None:
    stream = BytesIO(b"untouched")
    with pytest.raises(TypeError):
        Image.new("RGB", (2, 2)).save(stream, "PNG", **options)
    assert stream.getvalue() == b"untouched"


@pytest.mark.parametrize("format", ["JPEG", "PNG", "WEBP", "JXL", "HEIF", "TIFF"])
def test_thumbnail_embedding_in_containers(format: str) -> None:
    exif = exif_tags()
    exif.thumbnail = encode_image(Image.new("RGB", (3, 2), "red"), "JPEG")
    data = encode_image(Image.new("RGB", (8, 6)), format, exif=exif)
    image = Image.open(BytesIO(data))
    assert image.n_frames == 1
    assert image.getexif().thumbnail == exif.thumbnail
    again = Image.open(BytesIO(encode_image(image, format)))
    assert again.getexif().thumbnail == exif.thumbnail


@pytest.mark.parametrize("format", ["PNG", "TIFF", "JXL"])
@pytest.mark.parametrize(("mode", "depth"), [("1", 8), ("P", 8), ("LA", 8), ("RGB", 16), ("I;16B", 16)])
def test_metadata_does_not_change_lossless_pixels(format: str, mode: str, depth: int) -> None:
    image = Image.new(mode, (7, 5), 1 if mode in ("1", "P", "I;16B") else None)
    if depth == 16 and mode == "RGB":
        image = image.convert(mode, bit_depth=depth)
    options = {"lossless": True} if format == "JXL" else {}
    baseline = Image.open(BytesIO(encode_image(image, format, **options)))
    output = Image.open(BytesIO(encode_image(image, format, exif=exif_tags(), xmp=b"text", **options)))
    assert output.mode == baseline.mode
    assert output.bit_depth == baseline.bit_depth
    assert output.tobytes() == baseline.tobytes()
    assert output.getexif()[305] == "Blanket"


@pytest.mark.parametrize("compression", ["raw", "tiff_lzw", "tiff_adobe_deflate", "jpeg"])
def test_tiff_compression_and_external_metadata(compression: str, profile: bytes) -> None:
    stream = BytesIO()
    source = PILImage.new("RGB", (12, 9), (30, 60, 90))
    source.save(stream, "TIFF", tiffinfo={305: "Pillow metadata"}, icc_profile=profile, compression=compression)
    image = Image.open(stream)
    assert image.getexif()[305] == "Pillow metadata"
    image.getexif()[34665] = {33434: Fraction(1, 125)}
    reopened = Image.open(BytesIO(encode_image(image, "TIFF", compression=compression)))
    assert reopened.getexif()[305] == "Pillow metadata"
    assert reopened.info["icc_profile"] == profile
    assert reopened.getexif().get_ifd(34665)[33434] == pytest.approx(1 / 125)
    assert reopened.size == source.size


@pytest.mark.parametrize("format", ["JPEG", "PNG", "WEBP", "JXL", "HEIF"])
@pytest.mark.parametrize("compressor", [LosslessImageCompressor(effort=1), LossyImageCompressor(effort=1)])
def test_compression_preserves_metadata(format: str, compressor: object, profile: bytes) -> None:
    image = Image.new("RGB", (16, 16), (20, 50, 80))
    image.info.update(exif=exif_tags().tobytes(), xmp=b"packet", icc_profile=profile)
    options = {"xmp": image.info["xmp"]} if format == "JPEG" else {}
    normal = Image.open(BytesIO(encode_image(image, format, **options)))
    optimized = Image.open(BytesIO(encode_image(image, format, compressor=compressor, **options)))
    assert optimized.getexif()[305] == "Blanket"
    assert optimized.info["xmp"] == b"packet"
    assert optimized.info["icc_profile"] == profile
    if isinstance(compressor, LosslessImageCompressor):
        assert optimized.size == normal.size
        assert optimized.convert(normal.mode).tobytes() == normal.tobytes()


def test_multiframe_tiff_exif_is_frame_local() -> None:
    pages = []
    for software in ("page zero", "page one"):
        page = PILImage.new("RGB", (5, 4))
        page.encoderinfo = {"tiffinfo": {305: software}}
        pages.append(page)
    stream = BytesIO()
    pages[0].save(stream, "TIFF", save_all=True, append_images=pages[1:])
    image = Image.open(stream)
    assert image.getexif()[305] == "page zero"
    image.getexif()[305] = "edited zero"
    image.seek(1)
    assert image.getexif()[305] == "page one"
    image.getexif()[305] = "edited one"
    image.seek(0)
    assert image.getexif()[305] == "edited zero"
    assert image.convert("RGBA").getexif()[305] == "edited zero"
    image.seek(1)
    assert image.getexif()[305] == "edited one"


def test_getexif_reads_xmp_orientation_and_hidden_ifds() -> None:
    image = Image.new("RGB", (4, 3))
    image.info["xmp"] = b'<rdf tiff:Orientation="6"/>'
    assert image.getexif()[274] == 6
    image.getexif()[34665] = {33434: Fraction(1, 125)}
    image.getexif().hide_offsets()
    assert 34665 not in image.getexif()
    reopened = Image.open(BytesIO(encode_image(image, "PNG")))
    assert reopened.getexif().get_ifd(34665)[33434] == pytest.approx(1 / 125)


def test_inconsistent_jpeg_icc_is_ignored_without_losing_exif() -> None:
    exif = exif_tags().tobytes()
    raw = b"\xff\xd8"
    for number, total in ((1, 3), (2, 2)):
        payload = b"ICC_PROFILE\0" + bytes((number, total)) + b"profile"
        raw += b"\xff\xe2" + (len(payload) + 2).to_bytes(2, "big") + payload
    raw += b"\xff\xe1" + (len(exif) + 2).to_bytes(2, "big") + exif + b"\xff\xd9"
    metadata = read_metadata(raw, "JPEG")
    assert "icc_profile" not in metadata
    assert metadata["exif"] == exif
