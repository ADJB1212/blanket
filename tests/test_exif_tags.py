from __future__ import annotations

from io import BytesIO

import pytest
from helpers import image_buffer
from PIL import ExifTags as PillowTags, Image as PillowImage

from blanket import ExifTags, Image


def test_base_tag_identifiers_match_pillow() -> None:
    assert {tag.name: tag.value for tag in ExifTags.Base} == {tag.name: PillowTags.Base[tag.name].value for tag in ExifTags.Base}
    assert {tag.value: ExifTags.TAGS[tag] for tag in ExifTags.Base} == {tag.value: PillowTags.TAGS[tag.value] for tag in ExifTags.Base}


def test_ifd_identifiers_match_pillow() -> None:
    assert {tag.name: tag.value for tag in ExifTags.IFD} == {tag.name: PillowTags.IFD[tag.name].value for tag in ExifTags.IFD}


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
    thumbnail = image_buffer(Image.new("RGB", (3, 2), "red"), "JPEG")
    exif = Image.Exif()
    exif.thumbnail = thumbnail.getvalue()
    restored = Image.Exif()
    restored.load(exif.tobytes())
    assert restored.get_ifd(ExifTags.IFD.IFD1) is restored.get_ifd(1)
    assert restored.get_ifd(ExifTags.IFD.IFD1)[ExifTags.Base.JpegIFByteCount] == len(thumbnail.getvalue())
    assert restored.thumbnail == thumbnail.getvalue()
