from __future__ import annotations

import struct
from datetime import UTC, datetime, timedelta, timezone
from fractions import Fraction
from io import BytesIO

import pytest
from helpers import assert_pixels_equal, encode_image
from PIL import Image as PillowImage

from blanket import Image, ImageMetadata


def encoded(format="JPEG", **options):
    return encode_image(Image.new("RGB", (20, 12), (15, 42, 67)), format, **options)


@pytest.mark.parametrize("endian", ["<", ">"])
@pytest.mark.parametrize("format", ["JPEG", "TIFF", "PNG", "WEBP"])
def test_gps_capture_time_interop(endian, format):
    exif = Image.Exif()
    exif.endian = endian
    exif[305] = "retain"
    exif[34853] = {18: "WGS-84"}
    exif[34665] = {42036: "lens"}
    gps = ImageMetadata.GPS(-33.875, 151.2111111, -12.75)
    capture = datetime(2026, 11, 15, 12, 34, 56, 123456, timezone(timedelta(hours=5, minutes=45)))
    ImageMetadata.set_gps(exif, gps)
    ImageMetadata.set_capture_time(exif, capture)
    assert exif[305] == "retain"
    assert exif.get_ifd(34853)[18] == "WGS-84"
    assert exif.get_ifd(34665)[42036] == "lens"
    assert ImageMetadata.get_capture_time(exif) == capture
    data = encoded(format, exif=exif)
    reopened = Image.open(BytesIO(data))
    actual = ImageMetadata.get_gps(reopened.getexif())
    assert actual.latitude == pytest.approx(gps.latitude, abs=1e-9)
    assert actual.longitude == pytest.approx(gps.longitude, abs=1e-9)
    assert actual.altitude == gps.altitude
    assert ImageMetadata.get_capture_time(reopened.getexif()) == capture
    other = PillowImage.open(BytesIO(data)).getexif()
    assert other.get_ifd(34665)[36881] == "+05:45"
    assert other.get_ifd(34665)[36867] == "2026:11:15 12:34:56"
    assert other.get_ifd(34853)[1] == "S"
    assert float(other.get_ifd(34853)[2][0]) == 33
    assert other.get_ifd(34853)[5] in (1, b"\x01")


def test_gps_missing_clear_and_invalid():
    exif = Image.Exif()
    assert ImageMetadata.get_gps(exif) is None
    ImageMetadata.set_gps(exif, ImageMetadata.GPS(0, -180))
    assert ImageMetadata.get_gps(exif) == ImageMetadata.GPS(0, -180)
    assert 6 not in exif.get_ifd(34853)
    before = exif.tobytes()
    for gps in (ImageMetadata.GPS(91, 0), ImageMetadata.GPS(0, float("nan")), ImageMetadata.GPS(0, 0, float("inf"))):
        with pytest.raises(ValueError):
            ImageMetadata.set_gps(exif, gps)
        assert exif.tobytes() == before
    ImageMetadata.set_gps(exif, None)
    assert 34853 not in exif
    assert ImageMetadata.get_gps(exif) is None
    exif[34853] = {1: "N", 2: (1.0, 2.0, 3.0)}
    with pytest.raises(ValueError):
        ImageMetadata.get_gps(exif)
    exif[34853] = {1: "N", 2: (Fraction(90), Fraction(0), Fraction(1)), 3: "E", 4: (Fraction(0),) * 3}
    with pytest.raises(ValueError):
        ImageMetadata.get_gps(exif)


def test_capture_edits_retain_thumbnail_and_opaque_metadata():
    exif = Image.Exif()
    exif.thumbnail = encoded()
    exif[34665] = {37500: b"opaque-maker-note", 42036: "lens"}
    ImageMetadata.set_gps(exif, ImageMetadata.GPS(1, 2, 3))
    ImageMetadata.set_capture_time(exif, datetime(2026, 11, 15, tzinfo=UTC))
    assert exif.thumbnail == encoded()
    assert exif.get_ifd(34665)[37500] == b"opaque-maker-note"
    ImageMetadata.set_gps(exif, None)
    ImageMetadata.set_capture_time(exif, None)
    assert exif.thumbnail == encoded()
    assert exif.get_ifd(34665) == {37500: b"opaque-maker-note", 42036: "lens"}


def test_capture_time_missing_offsets_and_clear():
    exif = Image.Exif()
    assert ImageMetadata.get_capture_time(exif) is None
    exif[34665] = {36867: "2024:02:29 00:00:00", 37521: "000001234", 42036: "lens"}
    with pytest.raises(ValueError, match="timezone"):
        ImageMetadata.get_capture_time(exif)
    assert ImageMetadata.get_capture_time(exif, default_timezone=UTC) == datetime(2024, 2, 29, microsecond=1, tzinfo=UTC)
    before = exif.tobytes()
    for value in (datetime(2026, 1, 1), datetime(2026, 1, 1, tzinfo=timezone(timedelta(seconds=1)))):
        with pytest.raises(ValueError):
            ImageMetadata.set_capture_time(exif, value)
        assert exif.tobytes() == before
    ImageMetadata.set_capture_time(exif, None)
    assert ImageMetadata.get_capture_time(exif) is None
    assert exif.get_ifd(34665) == {42036: "lens"}


@pytest.mark.parametrize("tag,value", [(36867, "2023:02:29 00:00:00"), (36867, "2026:01:01 24:00:00"), (36881, "+24:00"), (36881, "+00:60"), (37521, "not-digits")])
def test_capture_time_rejects_invalid_fields(tag, value):
    exif = Image.Exif()
    exif[34665] = {36867: "2026:01:01 00:00:00", 36881: "+00:00", tag: value}
    with pytest.raises(ValueError):
        ImageMetadata.get_capture_time(exif)


def iim(value=b"old", dataset=120):
    return bytes([0x1C, 2, dataset]) + len(value).to_bytes(2, "big") + value


def resource(kind, payload):
    return b"8BIM" + kind.to_bytes(2, "big") + b"\x00\x00" + len(payload).to_bytes(4, "big") + payload + (b"\0" if len(payload) & 1 else b"")


def jpeg_iptc(raw):
    image = encoded()
    payload = b"Photoshop 3.0\0" + resource(0x03EE, b"unrelated") + resource(0x0404, raw) + resource(0x0425, b"opaque")
    segment = b"\xff\xed" + (len(payload) + 2).to_bytes(2, "big") + payload
    return image[:2] + segment + image[2:]


def tiff_iptc(raw, endian, kind=1, tag=33723):
    stream = BytesIO()
    other = PillowImage.new("RGB", (20, 12), (15, 42, 67)) if endian == "<" else PillowImage.new("I;16B", (20, 12), 4095)
    other.save(stream, "TIFF")
    data = stream.getvalue()
    # Append a replacement first directory so existing pixel offsets stay valid.
    assert data[:2] == (b"II" if endian == "<" else b"MM")
    offset = struct.unpack_from(endian + "I", data, 4)[0]
    count = struct.unpack_from(endian + "H", data, offset)[0]
    entries = data[offset + 2 : offset + 2 + count * 12]
    raw = raw + b"\0" * ((-len(raw)) % 4 if kind == 4 else 0)
    value_offset = len(data)
    entry = struct.pack(endian + "HHII", tag, kind, len(raw) // (4 if kind == 4 else 1), value_offset)
    result = bytearray(data + raw)
    new_offset = len(result)
    result.extend(struct.pack(endian + "H", count + 1) + entries + entry + b"\0" * 4)
    struct.pack_into(endian + "I", result, 4, new_offset)
    return bytes(result)


@pytest.mark.parametrize("endian", ["<", ">"])
@pytest.mark.parametrize("container", ["JPEG", "TIFF", "TIFF_LONG", "TIFF_PHOTOSHOP"])
def test_iptc_repeat_binary_and_lossless_rewrite(container, endian):
    raw = iim(b"one", 25) + iim(b"two", 25) + iim(b"\xff\x00")
    if container == "JPEG":
        data = jpeg_iptc(raw)
    elif container == "TIFF_PHOTOSHOP":
        data = tiff_iptc(resource(0x03EE, b"unrelated") + resource(0x0404, raw), endian, tag=34377)
    else:
        data = tiff_iptc(raw, endian, 4 if container == "TIFF_LONG" else 1)
    fp = BytesIO(data)
    fp.seek(7)
    assert ImageMetadata.get_iptc(fp) == ((2, 25, b"one"), (2, 25, b"two"), (2, 120, b"\xff\x00"))
    assert fp.tell() == 7
    datasets = ((1, 90, b"\x1b%G"), (2, 25, b"new-one"), (2, 25, b"new-two"), (2, 120, b"\x00" * 40000))
    output = BytesIO()
    changed = ImageMetadata.rewrite_iptc(fp, datasets, output)
    assert fp.tell() == 7
    assert output.getvalue() == changed
    assert ImageMetadata.get_iptc(BytesIO(changed)) == datasets
    assert_pixels_equal(Image.open(BytesIO(changed)), Image.open(BytesIO(data)))
    other = PillowImage.open(BytesIO(changed))
    assert other.tobytes() == PillowImage.open(BytesIO(data)).tobytes()
    if container == "JPEG":
        from PIL import IptcImagePlugin

        interop = ImageMetadata.rewrite_iptc(BytesIO(data), datasets[:-1])
        assert IptcImagePlugin.getiptcinfo(PillowImage.open(BytesIO(interop)))[2, 25] == [b"new-one", b"new-two"]
        assert resource(0x03EE, b"unrelated") in changed
        assert resource(0x0425, b"opaque") in changed
        assert changed[changed.index(b"\xff\xda") :] == data[data.index(b"\xff\xda") :]
    cleared = ImageMetadata.rewrite_iptc(BytesIO(changed), [])
    assert ImageMetadata.get_iptc(BytesIO(cleared)) == ()
    if container == "TIFF_PHOTOSHOP":
        assert resource(0x03EE, b"unrelated") in cleared


def test_iptc_missing_malformed_and_limits():
    assert ImageMetadata.get_iptc(BytesIO(encoded())) == ()
    with pytest.raises(ValueError, match="existing"):
        ImageMetadata.rewrite_iptc(BytesIO(encoded()), [(2, 120, b"new")])
    for raw in (b"bad", b"\x1c\x02\x78\x80\x00", b"\x1c\x02\x78\x00\x05x"):
        destination = BytesIO(b"untouched")
        with pytest.raises(ValueError):
            ImageMetadata.rewrite_iptc(BytesIO(jpeg_iptc(raw)), [], destination)
        assert destination.getvalue() == b"untouched"
    with pytest.raises(ValueError, match="APP13"):
        ImageMetadata.rewrite_iptc(BytesIO(jpeg_iptc(iim())), [(2, 120, b"x" * 65500)])


XMP = b'<x:meta xmlns:x="adobe:ns:meta/" xmlns:exif="http://ns.adobe.com/exif/1.0/" xmlns:dc="http://purl.org/dc/elements/1.1/"><exif:GPSLatitude>private</exif:GPSLatitude><dc:title>public</dc:title></x:meta>'


def test_xmp_sidecar_policy_and_sibling(tmp_path):
    path = tmp_path / "photo.image.jpg"
    path.write_bytes(b"original-image")
    assert ImageMetadata.read_xmp_sidecar(path) is None
    sidecar = ImageMetadata.write_xmp_sidecar(path, XMP)
    assert sidecar == tmp_path / "photo.image.xmp"
    assert sidecar.read_bytes() == XMP
    assert ImageMetadata.read_xmp_sidecar(path) == XMP
    policy = ImageMetadata.MetadataPolicy(location=False)
    embedded = policy.apply({"xmp": XMP})["xmp"]
    assert ImageMetadata.read_xmp_sidecar(path, policy=policy) == embedded
    ImageMetadata.write_xmp_sidecar(path, XMP, policy=policy)
    assert sidecar.read_bytes() == embedded
    assert b"private" not in embedded and b"public" in embedded
    ImageMetadata.write_xmp_sidecar(path, XMP, policy=ImageMetadata.MetadataPolicy(xmp=False))
    assert sidecar.read_bytes() == b""
    assert ImageMetadata.read_xmp_sidecar(path) is None
    assert path.read_bytes() == b"original-image"


def test_xmp_sidecar_malformed_and_fields(tmp_path):
    path = tmp_path / "missing.jpg"
    ImageMetadata.write_xmp_sidecar(path, b"not XML")
    assert ImageMetadata.read_xmp_sidecar(path) == b"not XML"
    policy = ImageMetadata.MetadataPolicy(location=False)
    assert ImageMetadata.read_xmp_sidecar(path, policy=policy) is None
    ImageMetadata.write_xmp_sidecar(path, b"not XML", policy=policy)
    assert path.with_suffix(".xmp").read_bytes() == b""
    policy = ImageMetadata.MetadataPolicy(xmp_fields=frozenset({"{http://purl.org/dc/elements/1.1/}title"}))
    ImageMetadata.write_xmp_sidecar(path, XMP.decode(), policy=policy)
    assert b"private" not in path.with_suffix(".xmp").read_bytes()


def test_xmp_sidecar_errors_do_not_change_existing_packet(tmp_path):
    path = tmp_path / "photo.jpg"
    sidecar = ImageMetadata.write_xmp_sidecar(path, XMP)
    with pytest.raises(TypeError):
        ImageMetadata.write_xmp_sidecar(path, XMP, policy="invalid")
    assert sidecar.read_bytes() == XMP
    with pytest.raises(FileNotFoundError):
        ImageMetadata.write_xmp_sidecar(tmp_path / "missing" / "photo.jpg", XMP)
    sidecar.unlink()
    sidecar.mkdir()
    with pytest.raises(IsADirectoryError):
        ImageMetadata.read_xmp_sidecar(path)


@pytest.mark.parametrize("format", ["JPEG", "TIFF"])
def test_geometry_crop_aspect_and_no_application(format):
    exif = Image.Exif()
    exif[50829] = (1, 2, 11, 19)
    exif[50719] = (Fraction(1, 2), Fraction(3, 2))
    exif[50720] = (Fraction(12), Fraction(6))
    exif[274] = 6
    exif[282], exif[283], exif[296] = Fraction(100), Fraction(200), 2
    data = encoded(format, exif=exif)
    original_size = Image.inspect(BytesIO(data)).size
    geometry = ImageMetadata.get_geometry(BytesIO(data))
    assert geometry.clean_aperture == (2.5, 2.5, 14.5, 8.5)
    assert geometry.pixel_aspect == 2
    assert geometry.source == ("tiff:default_crop" if format == "TIFF" else "exif:default_crop")
    assert Image.inspect(BytesIO(data)).size == original_size


def box(kind, payload):
    return (len(payload) + 8).to_bytes(4, "big") + kind + payload


def heif_geometry_boxes(*, denominator=1, large=False):
    properties = [box(b"ispe", b"\0" * 4 + struct.pack(">II", 20, 12)), box(b"clap", struct.pack(">IIIIiIiI", 16, denominator, 8, 1, -1, 1, 1, 1)), box(b"pasp", struct.pack(">II", 4, 3)), box(b"pasp", struct.pack(">II", 99, 1))]
    # Primary item 7 uses properties 1, 2, 3. Thumbnail 8 uses property 4.
    version_flags = b"\1\0\0\1" if large else b"\0" * 4

    def item(value):
        return struct.pack(">I" if large else ">H", value)

    def index(value):
        return struct.pack(">H" if large else ">B", value)

    associations = version_flags + struct.pack(">I", 2) + item(7) + b"\3" + b"".join(index(v) for v in (1, 2, 3)) + item(8) + b"\1" + index(4)
    return box(b"meta", b"\0" * 4 + box(b"pitm", b"\0" * 4 + struct.pack(">H", 7)) + box(b"iprp", box(b"ipco", b"".join(properties)) + box(b"ipma", associations)))


@pytest.mark.parametrize("large", [False, True])
def test_native_heif_primary_geometry(large):
    from blanket._blanket import _get_geometry

    aperture, aspect, source = _get_geometry(heif_geometry_boxes(large=large), "HEIF")
    assert aperture == (1, 3, 17, 11)
    assert aspect == pytest.approx(4 / 3)
    assert source == "heif:clap"
    with pytest.raises(ValueError, match="denominator"):
        _get_geometry(heif_geometry_boxes(denominator=0), "HEIF")
    with pytest.raises(ValueError):
        _get_geometry(heif_geometry_boxes()[:-1], "HEIF")


@pytest.mark.parametrize("format", ["HEIF", "AVIF"])
def test_public_heif_geometry(format):
    data = encoded(format)
    original = Image.open(BytesIO(data)).tobytes()
    geometry = ImageMetadata.get_geometry(BytesIO(data))
    assert geometry.pixel_aspect == 1
    if format == "HEIF":
        assert geometry.clean_aperture == (0, 0, 20, 12)
        assert geometry.source == "heif:clap"
    else:
        assert geometry.clean_aperture is None
    assert Image.open(BytesIO(data)).tobytes() == original


def test_geometry_defaults_and_invalid_crop():
    assert ImageMetadata.get_geometry(BytesIO(encoded())) == ImageMetadata.ImageGeometry()
    exif = Image.Exif()
    exif[50720] = (0, 10)
    with pytest.raises(ValueError, match="crop"):
        ImageMetadata.get_geometry(BytesIO(encoded(exif=exif)))
