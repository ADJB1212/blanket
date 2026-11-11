"""Named EXIF tags used by Blanket's metadata adapters.

Base and TAGS cover common image and capture metadata rather than the entire
EXIF tag registry. Integer tag identifiers remain accepted by Image.Exif.
"""

from __future__ import annotations

from enum import IntEnum


class Base(IntEnum):
    """Identify common image and capture metadata tags by their EXIF numbers."""

    ImageWidth = 0x0100
    ImageLength = 0x0101
    BitsPerSample = 0x0102
    Compression = 0x0103
    PhotometricInterpretation = 0x0106
    ImageDescription = 0x010E
    Make = 0x010F
    Model = 0x0110
    Orientation = 0x0112
    SamplesPerPixel = 0x0115
    XResolution = 0x011A
    YResolution = 0x011B
    ResolutionUnit = 0x0128
    Software = 0x0131
    DateTime = 0x0132
    Artist = 0x013B
    JpegIFOffset = 0x0201
    JpegIFByteCount = 0x0202
    YCbCrPositioning = 0x0213
    Copyright = 0x8298
    ExposureTime = 0x829A
    FNumber = 0x829D
    ExifOffset = 0x8769
    ExposureProgram = 0x8822
    GPSInfo = 0x8825
    ISOSpeedRatings = 0x8827
    ExifVersion = 0x9000
    DateTimeOriginal = 0x9003
    DateTimeDigitized = 0x9004
    ComponentsConfiguration = 0x9101
    ShutterSpeedValue = 0x9201
    ApertureValue = 0x9202
    ExposureBiasValue = 0x9204
    FocalLength = 0x920A
    MakerNote = 0x927C
    UserComment = 0x9286
    FlashPixVersion = 0xA000
    ColorSpace = 0xA001
    ExifImageWidth = 0xA002
    ExifImageHeight = 0xA003
    ExifInteroperabilityOffset = 0xA005
    RelatedImageWidth = 0x1001
    RelatedImageLength = 0x1002
    ImageResources = 0x8649
    SubIFDs = 0x014A


class IFD(IntEnum):
    """Identify nested EXIF directories accepted by Image.Exif.get_ifd."""

    Exif = 0x8769
    GPSInfo = 0x8825
    MakerNote = 0x927C
    Interop = 0xA005
    IFD1 = -1


TAGS: dict[int, str] = {tag.value: tag.name for tag in Base}

__all__ = ["IFD", "TAGS", "Base"]
