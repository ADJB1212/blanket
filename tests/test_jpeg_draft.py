from __future__ import annotations

from io import BytesIO

import pytest
from helpers import assert_pixels_equal, encode_image
from PIL import Image as PillowImage

from blanket import Image, ImageOps


@pytest.mark.parametrize("source_mode", ["L", "RGB", "CMYK"])
@pytest.mark.parametrize("mode,size", [(None, size) for size in (None, (100, 100), (17, 13), (8, 7), (1, 1))] + [("RGB", (17, 13)), ("L", (8, 7)), ("YCbCr", (1, 1))])
def test_jpeg_draft_matches_decoder(source_mode: str, mode: str | None, size: tuple[int, int] | None) -> None:
    dimensions = (63, 47)
    channels = len(source_mode)
    pixels = bytes((value * 37 + value // 7) % 256 for value in range(63 * 47 * channels))
    source = PillowImage.frombytes(source_mode, dimensions, pixels)
    encoded = encode_image(source, "JPEG")
    with Image.open(BytesIO(encoded)) as actual, PillowImage.open(BytesIO(encoded)) as expected:
        assert actual.draft(mode, size) == expected.draft(mode, size)
        assert_pixels_equal(actual, expected)
        assert actual.draft(mode, size) is None
    assert Image.open(BytesIO(encoded)).tobytes() == PillowImage.open(BytesIO(encoded)).tobytes()


def test_loaded_jpeg_draft_is_noop() -> None:
    encoded = Image.new("RGB", (17, 13), "red").encode("JPEG")
    actual = Image.open(BytesIO(encoded))
    actual.load()
    assert actual.draft("L", (1, 1)) is None
    assert actual.mode == "RGB"
    assert actual.size == (17, 13)


@pytest.mark.parametrize("opener", [lambda data: Image.open(BytesIO(data)), Image.decode])
@pytest.mark.parametrize("access", ["getpixel", "putpixel", "tobytes", "copy"])
def test_pixel_access_disables_jpeg_draft(opener, access) -> None:
    encoded = Image.new("RGB", (32, 32), "red").encode("JPEG")
    with opener(encoded) as image:
        if access == "getpixel":
            image.getpixel((0, 0))
        elif access == "putpixel":
            image.putpixel((0, 0), (0, 255, 0))
        else:
            getattr(image, access)()
        assert image.draft("L", (4, 4)) is None
        assert image.mode == "RGB"
        assert image.size == (32, 32)
        if access == "putpixel":
            assert image.getpixel((0, 0)) == (0, 255, 0)


def test_eager_jpeg_metadata_does_not_disable_draft() -> None:
    with Image.decode(Image.new("RGB", (32, 32), "red").encode("JPEG")) as image:
        assert (image.mode, image.size, image.format, image.bit_depth) == ("RGB", (32, 32), "JPEG", 8)
        assert image.tell() == 0
        image.seek(0)
        assert image.draft("L", (4, 4)) == ("L", (0, 0, 4.0, 4.0))
        assert image.size == (4, 4)


@pytest.mark.parametrize("operation", ["thumbnail", "exif_transpose"])
def test_in_place_geometry_keeps_jpeg_format_and_save_settings(operation):
    source = Image.new("RGB", (32, 16), "red")
    source.getexif()[274] = 6
    with Image.open(BytesIO(source.encode("JPEG", quality=83, subsampling=0))) as image:
        if operation == "thumbnail":
            image.thumbnail((8, 8))
            assert image.size == (8, 4)
        else:
            ImageOps.exif_transpose(image, in_place=True)
            assert image.size == (16, 32)
        assert image.format == "JPEG"
        assert image.get_format_mimetype() == "image/jpeg"
        assert image.copy().format is None
        with Image.decode(image.encode("JPEG", quality="keep")) as saved:
            assert saved.size == image.size
            assert saved.quantization == image.quantization
