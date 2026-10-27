from __future__ import annotations

import struct
from io import BytesIO
from xml.etree import ElementTree as ET

import pytest
from PIL import Image as PillowImage

from blanket import Image, ImageCms, ImageMetadata, PngImagePlugin, UnidentifiedImageError

XMP = b'<x:xmpmeta xmlns:x="adobe:ns:meta/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns:exif="http://ns.adobe.com/exif/1.0/" xmlns:dc="http://purl.org/dc/elements/1.1/"><rdf:RDF><rdf:Description exif:GPSLatitude="private-gps" dc:creator="artist"><exif:GPSLongitude>private-longitude</exif:GPSLongitude><dc:title>public-title</dc:title></rdf:Description></rdf:RDF></x:xmpmeta>'


def encode(format="PNG", mode="RGB", depth=8, **options):
    image = Image.new(mode, (17, 9))
    if depth != 8:
        image = image.convert(mode, bit_depth=depth)
    stream = BytesIO()
    image.save(stream, format, **options)
    return stream.getvalue()


def exif_data(thumbnail=False):
    value = Image.Exif()
    value[305] = "public-software"
    value[274] = 6
    value[34853] = {1: "N", 2: (1.0, 2.0, 3.0)}
    value[34665] = {36867: "2026:10:27 12:00:00", 37500: b"private-maker-note"}
    if thumbnail:
        value.thumbnail = encode("JPEG")
    return value


def png_chunks(data):
    result = []
    pos = 8
    while pos < len(data):
        length = int.from_bytes(data[pos : pos + 4], "big")
        result.append((data[pos + 4 : pos + 8], data[pos : pos + length + 12]))
        pos += length + 12
    return result


def riff_chunks(data):
    result = []
    pos = 12
    while pos < len(data):
        length = int.from_bytes(data[pos + 4 : pos + 8], "little")
        result.append((data[pos : pos + 4], data[pos : pos + length + 8 + (length & 1)]))
        pos += length + 8 + (length & 1)
    return result


def jxl_codestream(data):
    if data[:2] == b"\xff\x0a":
        return data
    pos = 12
    result = []
    while pos < len(data):
        length = int.from_bytes(data[pos : pos + 4], "big")
        kind = data[pos + 4 : pos + 8]
        if kind in (b"jxlc", b"jxlp"):
            result.append(data[pos + 8 : pos + length])
        pos += length
    return b"".join(result)


@pytest.mark.parametrize("format", ["JPEG", "PNG", "WEBP", "JXL"])
def test_metadata_replacement_removal_and_unchanged_pixels(format):
    profile = ImageCms.createProfile("sRGB").tobytes()
    data = encode(format, exif=exif_data(), xmp=XMP, icc_profile=profile)
    original = Image.open(BytesIO(data))
    exif = Image.Exif()
    exif[305] = "replacement"
    rewritten = ImageMetadata.rewrite(BytesIO(data), exif=exif, xmp=b"<replacement/>")
    result = Image.open(BytesIO(rewritten))
    assert (result.mode, result.size, result.tobytes()) == (original.mode, original.size, original.tobytes())
    assert result.getexif()[305] == "replacement"
    assert result.info["xmp"] == b"<replacement/>"
    assert result.info["icc_profile"] == profile
    removed = ImageMetadata.rewrite(BytesIO(rewritten), exif=None, xmp=None)
    result = Image.open(BytesIO(removed))
    assert not result.info.get("exif") and not result.info.get("xmp")
    assert b"replacement" not in removed
    if format == "JPEG":
        assert data[data.index(b"\xff\xda") :] == rewritten[rewritten.index(b"\xff\xda") :]
    elif format == "PNG":
        assert [v for k, v in png_chunks(data) if k == b"IDAT"] == [v for k, v in png_chunks(rewritten) if k == b"IDAT"]
    elif format == "WEBP":
        assert [v for k, v in riff_chunks(data) if k in (b"VP8 ", b"VP8L", b"ALPH")] == [v for k, v in riff_chunks(rewritten) if k in (b"VP8 ", b"VP8L", b"ALPH")]
    else:
        assert jxl_codestream(data) == jxl_codestream(rewritten)


@pytest.mark.parametrize("format", ["JPEG", "PNG", "WEBP"])
def test_color_profile_control_is_independent(format):
    profile = ImageCms.createProfile("sRGB").tobytes()
    data = encode(format, exif=exif_data(), xmp=XMP, icc_profile=profile)
    policy = ImageMetadata.MetadataPolicy(exif=False, xmp=False, text=False)
    rewritten = ImageMetadata.rewrite(BytesIO(data), policy=policy)
    assert Image.inspect(BytesIO(rewritten)).info["icc_profile"] == profile
    stripped = ImageMetadata.rewrite(BytesIO(data), policy=ImageMetadata.MetadataPolicy(icc_profile=False))
    info = Image.inspect(BytesIO(stripped)).info
    assert "icc_profile" not in info and "exif" in info and "xmp" in info


@pytest.mark.parametrize("format", ["JPEG", "PNG", "WEBP", "JXL"])
def test_location_policy_and_exif_allowlist(format):
    data = encode(format, exif=exif_data(thumbnail=True), xmp=XMP)
    policy = ImageMetadata.MetadataPolicy(location=False)
    rewritten = ImageMetadata.rewrite(BytesIO(data), policy=policy)
    result = Image.open(BytesIO(rewritten))
    exif = result.getexif()
    assert 34853 not in exif and 37500 not in exif.get_ifd(34665)
    assert exif[305] == "public-software" and exif.thumbnail is None
    assert b"private-maker-note" not in rewritten and b"private-gps" not in rewritten and b"private-longitude" not in rewritten
    assert b"public-title" in result.info["xmp"]
    filtered = ImageMetadata.rewrite(BytesIO(data), policy=ImageMetadata.MetadataPolicy(exif_tags=frozenset({36867})))
    tags = Image.open(BytesIO(filtered)).getexif()
    assert set(tags) == {34665} and set(tags.get_ifd(34665)) == {36867}


def test_xmp_allowlist_and_invalid_xmp():
    policy = ImageMetadata.MetadataPolicy(xmp_fields=frozenset({"{http://purl.org/dc/elements/1.1/}title"}))
    result = policy.apply({"xmp": XMP})["xmp"]
    assert b"public-title" in result and b"private-gps" not in result and b"artist" not in result
    ET.fromstring(result)
    for raw in (b"invalid XML", b'<!DOCTYPE x [<!ENTITY value "secret">]><x>&value;</x>', "<!DOCTYPE x><x/>".encode("utf-16")):
        assert policy.apply({"xmp": raw})["xmp"] == b""


def test_png_text_categories_and_hdr_preservation():
    text = PngImagePlugin.PngInfo()
    text.add_text("Author", "public-author", zip=True)
    text.add_itxt("GPSLatitude", "private-gps")
    text.add_text("Other", "other")
    data = encode(pnginfo=text, xmp=XMP, cicp=(9, 16, 0, 1))
    rewritten = ImageMetadata.rewrite(BytesIO(data), policy=ImageMetadata.MetadataPolicy(location=False, text_keys=frozenset({"Author"})))
    info = Image.inspect(BytesIO(rewritten)).info
    assert info["Author"] == "public-author" and "Other" not in info and "GPSLatitude" not in info
    assert info["cicp"] == (9, 16, 0, 1) and b"public-title" in info["xmp"]
    replaced = ImageMetadata.rewrite(BytesIO(data), text={"Unicode": "caf\u00e9"})
    info = Image.inspect(BytesIO(replaced)).info
    assert info["Unicode"] == "caf\u00e9" and "Author" not in info and info["xmp"] == XMP


def test_legacy_png_exif_is_independent_of_text_and_exposes_thumbnails():
    exif = exif_data(thumbnail=True)
    text = PngImagePlugin.PngInfo()
    text.add_text("Raw profile type exif", exif.tobytes().hex())
    text.add_text("Author", "author")
    data = encode(pnginfo=text)
    rewritten = ImageMetadata.rewrite(BytesIO(data), policy=ImageMetadata.MetadataPolicy(text=False))
    assert Image.open(BytesIO(rewritten)).getexif()[305] == "public-software"
    assert ImageMetadata.get_previews(BytesIO(rewritten))[0].data == exif.thumbnail
    assert "Author" not in Image.inspect(BytesIO(rewritten)).info
    stripped = ImageMetadata.rewrite(BytesIO(data), policy=ImageMetadata.MetadataPolicy(exif=False))
    assert not Image.open(BytesIO(stripped)).getexif()


@pytest.mark.parametrize("format", ["PNG", "WEBP"])
def test_animation_bytes_controls_and_all_frames_preserved(format):
    stream = BytesIO()
    first = Image.new("RGBA", (8, 6), (255, 0, 0, 255))
    second = Image.new("RGBA", (8, 6), (0, 255, 0, 127))
    first.save(stream, format, save_all=True, append_images=[second], duration=[40, 90], loop=3, xmp=XMP)
    data = stream.getvalue()
    rewritten = ImageMetadata.rewrite(BytesIO(data), xmp=None)
    before, after = Image.open(BytesIO(data)), Image.open(BytesIO(rewritten))
    assert before.n_frames == after.n_frames == 2
    for i in range(2):
        before.seek(i)
        after.seek(i)
        assert before.tobytes() == after.tobytes()
        assert before.info["duration"] == after.info["duration"]
        assert before.info["loop"] == after.info["loop"] == 3
    chunks = png_chunks if format == "PNG" else riff_chunks
    keys = (b"IDAT", b"fdAT", b"fcTL", b"acTL") if format == "PNG" else (b"ANIM", b"ANMF")
    assert [v for k, v in chunks(data) if k in keys] == [v for k, v in chunks(rewritten) if k in keys]


@pytest.mark.parametrize("format,mode,depth", [("PNG", "1", 8), ("PNG", "RGBA", 16), ("PNG", "L", 12), ("JXL", "RGB", 16), ("JPEG", "CMYK", 8)])
def test_modes_and_precision(format, mode, depth):
    data = encode(format, mode, depth)
    rewritten = ImageMetadata.rewrite(BytesIO(data), xmp=XMP)
    before, after = Image.open(BytesIO(data)), Image.open(BytesIO(rewritten))
    assert (before.mode, before.bit_depth, before.tobytes()) == (after.mode, after.bit_depth, after.tobytes())


@pytest.mark.parametrize("format", ["JPEG", "PNG", "WEBP", "JXL"])
def test_resolution_rewrite_and_removal(format):
    data = ImageMetadata.rewrite(BytesIO(encode(format)), dpi=(300, 150))
    resolution = ImageMetadata.get_resolution(BytesIO(data), conflict="error")
    assert resolution.dpi == pytest.approx((300, 150), rel=1e-4)
    assert resolution.pixel_aspect == pytest.approx(0.5, rel=1e-4)
    result = ImageMetadata.get_resolution(BytesIO(ImageMetadata.rewrite(BytesIO(data), dpi=None)))
    assert result.dpi is None


@pytest.mark.parametrize("format", ["JPEG", "PNG", "TIFF", "BMP"])
def test_external_resolution_and_units(format):
    stream = BytesIO()
    PillowImage.new("RGB", (17, 9)).save(stream, format, dpi=(254, 127))
    result = ImageMetadata.get_resolution(stream)
    assert result.dpi == pytest.approx((254, 127), rel=1e-4)
    assert result.pixel_aspect == pytest.approx(0.5, rel=1e-4)


def test_resolution_conflicts_and_unitless_density():
    exif = Image.Exif()
    exif[282], exif[283], exif[296] = 100.0, 50.0, 3
    stream = BytesIO()
    PillowImage.new("RGB", (17, 9)).save(stream, "JPEG", dpi=(72, 72), exif=exif.tobytes())
    result = ImageMetadata.get_resolution(stream)
    assert result.dpi == (72, 72) and result.conflicts == ("exif",)
    result = ImageMetadata.get_resolution(stream, conflict="exif")
    assert result.dpi == pytest.approx((254, 127)) and result.unit == "cm"
    with pytest.raises(ValueError, match="conflicting"):
        ImageMetadata.get_resolution(stream, conflict="error")
    data = bytearray(encode("JPEG"))
    pos = data.index(b"JFIF\0")
    data[pos + 7 : pos + 12] = struct.pack(">BHH", 0, 2, 1)
    result = ImageMetadata.get_resolution(BytesIO(data))
    assert result.dpi is None and result.pixel_aspect == 0.5 and result.unit == "relative"


@pytest.mark.parametrize("format", ["JPEG", "PNG", "WEBP", "JXL", "TIFF", "HEIF"])
def test_save_policy_filters_explicit_metadata_and_cached_exif(format):
    image = Image.new("RGB", (17, 9))
    image.getexif().update(exif_data(thumbnail=True))
    image.info["xmp"] = XMP
    profile = ImageCms.createProfile("sRGB").tobytes()
    stream = BytesIO()
    image.save(stream, format, metadata_policy=ImageMetadata.MetadataPolicy(location=False), icc_profile=profile)
    result = Image.open(stream)
    assert 34853 not in result.getexif() and b"private-gps" not in result.info["xmp"]
    assert 34853 in image.getexif() and image.info["xmp"] == XMP


@pytest.mark.parametrize("format", ["JPEG", "PNG", "WEBP", "JXL", "TIFF", "HEIF"])
def test_exif_thumbnail_access_and_policy(format):
    exif = exif_data(thumbnail=True)
    data = encode(format, exif=exif)
    previews = ImageMetadata.get_previews(BytesIO(data))
    assert len(previews) == 1
    preview = previews[0]
    assert (preview.data, preview.format, preview.size, preview.source) == (exif.thumbnail, "JPEG", (17, 9), "exif")
    assert preview.open().size == (17, 9)
    policy = ImageMetadata.MetadataPolicy(previews=False)
    if format in ("TIFF", "HEIF"):
        stripped = encode(format, exif=exif, metadata_policy=policy)
    else:
        stripped = ImageMetadata.rewrite(BytesIO(data), policy=policy)
    assert ImageMetadata.get_previews(BytesIO(stripped)) == ()
    assert ImageMetadata.get_previews(BytesIO(encode())) == ()


def test_malformed_exif_thumbnail_bounds():
    exif = exif_data(thumbnail=True)
    raw = bytearray(exif.tobytes())
    parsed = Image.Exif()
    parsed.load(raw)
    start = 6 + parsed._next
    count = struct.unpack_from(">H", raw, start)[0]
    for pos in range(start + 2, start + 2 + count * 12, 12):
        if struct.unpack_from(">H", raw, pos)[0] == 513:
            struct.pack_into(">I", raw, pos + 8, len(raw) + 100)
            break
    with pytest.raises(ValueError, match="thumbnail bounds"):
        ImageMetadata.get_previews(BytesIO(encode("JPEG", exif=bytes(raw))))


@pytest.mark.parametrize("format", ["JPEG", "PNG", "WEBP", "JXL"])
def test_rewrite_preserves_unedited_container_bytes(format):
    data = encode(format, exif=exif_data(), xmp=XMP)
    assert ImageMetadata.rewrite(BytesIO(data)) == data


def test_progressive_jpeg_and_indexed_png_interoperate():
    for mode, format, options in [("RGB", "JPEG", {"progressive": True}), ("P", "PNG", {"bits": 4, "transparency": 0})]:
        image = PillowImage.new(mode, (17, 9))
        stream = BytesIO()
        image.save(stream, format, **options)
        data = stream.getvalue()
        rewritten = ImageMetadata.rewrite(BytesIO(data), xmp=XMP, dpi=(72, 72))
        before = PillowImage.open(BytesIO(data)).convert("RGBA")
        after = PillowImage.open(BytesIO(rewritten)).convert("RGBA")
        assert before.tobytes() == after.tobytes()


def test_eps_container_preview_bounds():
    tiff = encode("TIFF")
    ps = b"%!PS-Adobe-3.0 EPSF-3.0\n%%BoundingBox: 0 0 17 9\n"
    data = struct.pack("<7IH", 0xC6D3D0C5, 30, len(ps), 0, 0, 30 + len(ps), len(tiff), 0xFFFF) + ps + tiff
    previews = ImageMetadata.get_previews(BytesIO(data))
    assert len(previews) == 1 and previews[0].data == tiff and previews[0].source == "container"
    assert previews[0].open().size == (17, 9)
    with pytest.raises(OSError, match="truncated EPS preview"):
        ImageMetadata.get_previews(BytesIO(data[:-1]))


def test_heif_container_thumbnail_interoperability():
    pillow_heif = pytest.importorskip("pillow_heif")
    stream = BytesIO()
    pillow_heif.from_pillow(PillowImage.new("RGB", (128, 96), (70, 130, 190))).save(stream, thumbnails=[32, 64])
    previews = ImageMetadata.get_previews(stream)
    assert len(previews) == 2
    assert {preview.size for preview in previews} == {(32, 24), (64, 48)}
    external = pillow_heif.open_heif(stream)
    for preview, expected in zip(previews, (external[0].get_thumbnail(i) for i in range(2)), strict=True):
        assert preview.format == "PNG" and preview.source == "container"
        assert preview.open().tobytes() == expected.to_pillow().tobytes()


@pytest.mark.parametrize("depth", [10, 12])
def test_heif_wide_thumbnails_return_sample_preserving_png(depth):
    pillow_heif = pytest.importorskip("pillow_heif")
    size = (128, 96)
    raw = struct.pack("<H", 24000) * (size[0] * size[1] * 3)
    stream = BytesIO()
    pillow_heif.from_bytes("RGB;16", size, raw).save(stream, thumbnails=[32], bit_depth=depth)
    preview = ImageMetadata.get_previews(stream)[0]
    image = preview.open()
    assert (image.size, image.mode, image.bit_depth) == ((32, 24), "RGB", depth)
    assert image.getpixel((0, 0)) == pytest.approx((24000 >> (16 - depth),) * 3, abs=5)


def test_png_save_policy_filters_explicit_xmp_text_and_sequence_position():
    text = PngImagePlugin.PngInfo()
    text.add_itxt("XML:com.adobe.xmp", XMP)
    text.add_text("GPSLatitude", "private")
    first = Image.new("RGBA", (8, 6), "red")
    second = Image.new("RGBA", (8, 6), "blue")
    stream = BytesIO()
    first.save(stream, "PNG", save_all=True, append_images=[second], pnginfo=text, metadata_policy=ImageMetadata.MetadataPolicy(location=False))
    image = Image.open(stream)
    image.seek(1)
    output = BytesIO()
    image.save(output, "PNG", save_all=True, metadata_policy=ImageMetadata.MetadataPolicy(text=False))
    assert image.tell() == 1 and image.n_frames == 2
    info = Image.inspect(output).info
    assert b"public-title" in info["xmp"] and b"private" not in info["xmp"] and "GPSLatitude" not in info


def test_paths_positions_output_validation_and_no_raster_decode(tmp_path, monkeypatch):
    source = tmp_path / "source.png"
    source.write_bytes(encode())
    output = tmp_path / "output.png"
    stream = BytesIO(source.read_bytes())
    stream.seek(4)
    monkeypatch.setattr(Image, "open", lambda *a, **k: pytest.fail("unexpected pixel decode"))
    data = ImageMetadata.rewrite(stream, output, xmp=XMP)
    assert stream.tell() == 4 and not stream.closed and output.read_bytes() == data
    assert ImageMetadata.get_resolution(stream).dpi is None
    assert ImageMetadata.get_previews(stream) == ()
    assert ImageMetadata.rewrite(source, source, xmp=XMP) == source.read_bytes()
    with pytest.raises(ValueError, match="distinct"):
        ImageMetadata.rewrite(stream, stream)
    with pytest.raises(ValueError, match="not supported"):
        ImageMetadata.rewrite(BytesIO(encode("TIFF")), output)
    with pytest.raises(ValueError, match="codestream"):
        ImageMetadata.rewrite(BytesIO(encode("JXL")), output, icc_profile=None)
    assert output.read_bytes() == data


@pytest.mark.parametrize("dpi", [(0, 10), (float("nan"), 10), (float("inf"), 10), (-1, 10), (10,), True])
def test_invalid_resolution(dpi):
    with pytest.raises((TypeError, ValueError)):
        ImageMetadata.rewrite(BytesIO(encode()), dpi=dpi)


def test_malformed_containers_and_metadata_are_rejected_without_output(tmp_path):
    output = tmp_path / "output.png"
    output.write_bytes(b"unchanged")
    data = bytearray(encode())
    pos = data.index(b"IDAT") + 4
    data[pos] ^= 255
    with pytest.raises(OSError, match="CRC"):
        ImageMetadata.rewrite(BytesIO(data), output, xmp=XMP)
    with pytest.raises((ValueError, UnidentifiedImageError)):
        ImageMetadata.rewrite(BytesIO(encode("JPEG", exif=b"invalid")), output, policy=ImageMetadata.MetadataPolicy(location=False))
    assert output.read_bytes() == b"unchanged"
