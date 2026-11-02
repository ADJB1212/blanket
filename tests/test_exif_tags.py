from __future__ import annotations

from io import BytesIO

import pytest
from PIL import ExifTags as PillowTags, Image as PillowImage

from blanket import ExifTags, Image


@pytest.mark.parametrize("tag", list(ExifTags.Base))
def test_base_tag_identifiers_match_pillow(tag: ExifTags.Base) -> None:
    assert tag.value == PillowTags.Base[tag.name].value
    assert ExifTags.TAGS[tag] == PillowTags.TAGS[tag.value]


@pytest.mark.parametrize("tag", list(ExifTags.IFD))
def test_ifd_identifiers_match_pillow(tag: ExifTags.IFD) -> None:
    assert tag.value == PillowTags.IFD[tag.name].value


@pytest.mark.parametrize("endian", ["<", ">"])
def test_named_tags_roundtrip_nested_metadata(endian: str) -> None:
    exif = Image.Exif()
    exif.endian = endian
    exif[ExifTags.Base.Orientation] = 6
    exif[ExifTags.IFD.Exif] = {ExifTags.Base.UserComment: b"ASCII\0\0\0comment", ExifTags.IFD.Interop: {1: "R98"}}
    exif[ExifTags.IFD.GPSInfo] = {1: "N"}
    for exif_type in (Image.Exif, PillowImage.Exif):
        restored = exif_type()
        restored.load(exif.tobytes())
        assert restored[ExifTags.Base.Orientation] == 6
        assert restored.get_ifd(ExifTags.IFD.Exif)[ExifTags.Base.UserComment] == b"ASCII\0\0\0comment"
        assert restored.get_ifd(ExifTags.IFD.Interop)[1] == "R98"
        assert restored.get_ifd(ExifTags.IFD.GPSInfo)[1] == "N"


def test_named_thumbnail_ifd_preserves_integer_access() -> None:
    thumbnail = BytesIO()
    Image.new("RGB", (3, 2), "red").save(thumbnail, "JPEG")
    exif = Image.Exif()
    exif.thumbnail = thumbnail.getvalue()
    restored = Image.Exif()
    restored.load(exif.tobytes())
    assert restored.get_ifd(ExifTags.IFD.IFD1) is restored.get_ifd(1)
    assert restored.get_ifd(ExifTags.IFD.IFD1)[ExifTags.Base.JpegIFByteCount] == len(thumbnail.getvalue())
    assert restored.thumbnail == thumbnail.getvalue()
