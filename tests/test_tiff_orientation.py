from __future__ import annotations

from copy import copy, deepcopy
from io import BytesIO

import pytest
from helpers import assert_image_properties, assert_pixels_equal, assert_raises, encode_image, image_buffer
from PIL import Image as PILImage

from blanket import Image, ImageOps


@pytest.mark.parametrize("mode,orientation", [("RGB", orientation) for orientation in range(1, 9)] + [("1", 2), ("L", 3), ("RGBA", 5), ("CMYK", 6), ("F", 8)])
@pytest.mark.parametrize("lazy", [False, True])
def test_tiff_applies_orientation_before_copy(mode: str, orientation: int, lazy: bool) -> None:
    source = PILImage.new(mode, (7, 5))
    if mode in ("1", "L", "F"):
        source.putdata([(i * 37) % 256 for i in range(35)])
    else:
        source.putdata([tuple((i * 37 + c * 19) % 256 for c in range(len(source.getbands()))) for i in range(35)])
    raw = encode_image(source, "TIFF", tiffinfo={274: orientation})
    expected = PILImage.open(BytesIO(raw))
    expected.load()
    actual = Image.open(BytesIO(raw), lazy=lazy)
    assert actual.size == expected.size
    if mode == "1":
        assert actual.convert("L").tobytes() == expected.convert("L").tobytes()
    else:
        assert_image_properties(actual, mode=expected.mode, pixels=expected.tobytes())
    assert 274 not in actual.getexif()
    for result in (actual.copy(), copy(actual), deepcopy(actual), ImageOps.exif_transpose(actual)):
        assert_pixels_equal(result, actual)
        assert 274 not in result.getexif()
    assert Image.inspect(BytesIO(raw)).size == actual.size


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
@pytest.mark.parametrize("orientation", range(2, 9))
def test_tiff_orientation_preserves_high_depth(mode: str, orientation: int) -> None:
    channels = Image.getmodebands(mode)
    source = Image.frombytes(mode, (7, 5), bytes(7 * 5 * channels * 2), bit_depth=16)
    for y in range(5):
        for x in range(7):
            value = tuple((x * 1000 + y * 300 + c * 17) % 65536 for c in range(channels))
            source.putpixel((x, y), value[0] if channels == 1 else value)
    exif = Image.Exif()
    exif[274] = orientation
    raw = source.encode("TIFF", exif=exif)
    actual = Image.open(BytesIO(raw))
    method = {2: Image.Transpose.FLIP_LEFT_RIGHT, 3: Image.Transpose.ROTATE_180, 4: Image.Transpose.FLIP_TOP_BOTTOM, 5: Image.Transpose.TRANSPOSE, 6: Image.Transpose.ROTATE_270, 7: Image.Transpose.TRANSVERSE, 8: Image.Transpose.ROTATE_90}[
        orientation
    ]
    expected = source.transpose(method)
    assert_image_properties(actual, size=expected.size, bit_depth=16, pixels=expected.tobytes())
    assert 274 not in actual.getexif()


@pytest.mark.parametrize("lazy", [False, True])
def test_tiff_oriented_pages_keep_navigation_and_local_metadata(lazy: bool) -> None:
    first = PILImage.new("RGB", (7, 5), (30, 60, 90))
    second = PILImage.new("L", (4, 3), 100)
    raw = encode_image(first, "TIFF", save_all=True, append_images=[second], tiffinfo={274: 6})
    actual = Image.open(BytesIO(raw), lazy=lazy)
    expected = PILImage.open(BytesIO(raw))
    assert actual.n_frames == 2
    for frame in (0, 1, 0):
        actual.seek(frame)
        expected.seek(frame)
        expected.load()
        assert_pixels_equal(actual, expected)
        assert 274 not in actual.getexif()
        result = actual.copy()
        assert_image_properties(result, n_frames=1, pixels=actual.tobytes())
    assert_raises(EOFError, lambda: actual.seek(2))
    assert actual.tell() == 0


def test_oriented_tiff_region_uses_display_coordinates() -> None:
    source = PILImage.new("RGB", (7, 5))
    source.putdata([(i * 7, i * 3, i) for i in range(35)])
    raw = encode_image(source, "TIFF", tiffinfo={274: 6})
    expected = PILImage.open(BytesIO(raw))
    expected.load()
    result = Image.open_region(BytesIO(raw), (1, 2, 4, 6), fallback="full")
    assert result.tobytes() == expected.crop((1, 2, 4, 6)).tobytes()
    assert result.info["decode_fallback"] is True


def test_header_metadata_retains_raw_tiff_orientation() -> None:
    source = PILImage.new("RGB", (7, 5))
    stream = image_buffer(source, "TIFF", tiffinfo={274: 6})
    header = Image.inspect(BytesIO(stream.getvalue()))
    exif = Image.Exif()
    exif.load(header.info["exif"])
    assert exif[274] == 6
    assert header.size == (5, 7)


def test_oriented_tiff_clears_xmp_orientation_without_changing_source_metadata() -> None:
    source = PILImage.new("RGB", (7, 5), (10, 20, 30))
    xmp = b'<rdf:Description tiff:Orientation="6" keep="metadata"/>'
    raw = encode_image(source, "TIFF", tiffinfo={274: 6, 700: xmp})
    image = Image.open(BytesIO(raw))
    assert 274 not in image.getexif()
    assert b"tiff:Orientation" not in image.info["xmp"]
    assert b'keep="metadata"' in image.info["xmp"]
    assert ImageOps.exif_transpose(image).size == image.size == (5, 7)
    assert Image.inspect(BytesIO(raw)).info["xmp"] == xmp
