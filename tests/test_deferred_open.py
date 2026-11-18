from __future__ import annotations

import zlib
from io import BytesIO

import pytest
from PIL import Image as PillowImage

from blanket import Image, ImageFile

CASES = [("JPEG", mode) for mode in ("L", "RGB", "CMYK")] + [("PNG", mode) for mode in ("L", "LA", "RGB", "RGBA")] + [("WEBP", mode) for mode in ("RGB", "RGBA")]


@pytest.mark.parametrize("format,mode", CASES)
def test_metadata_and_verify_do_not_decode(monkeypatch, format, mode):
    source = PillowImage.new(mode, (19, 13), 73)
    stream = BytesIO()
    source.save(stream, format)

    def unexpected_decode(*args):
        pytest.fail("metadata access decoded pixels")

    monkeypatch.setattr(Image, "_open_sequence", unexpected_decode)
    monkeypatch.setattr(Image, "_draft_jpeg", unexpected_decode)
    with Image.open(BytesIO(stream.getvalue())) as image:
        assert image.mode == mode
        assert image.size == (19, 13)
        assert (image.width, image.height, image.bit_depth) == (19, 13, 8)
        assert image.format == format
        assert image.n_frames == 1
        assert not image.is_animated
        assert image.tell() == 0
        image.seek(0)
        with pytest.raises(EOFError):
            image.seek(1)
        assert not image.getexif()
        assert image.get_format_mimetype() == {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}[format]
        assert image.getxmp() == {}
        image.verify()
        if format == "JPEG":
            image.draft("L", (5, 4))
        image.close()
        image.close()
        with pytest.raises(ValueError, match="closed"):
            image.load()
        with pytest.raises(ValueError, match="closed"):
            image.verify()
        with pytest.raises(ValueError, match="closed"):
            image.draft("L", (1, 1))


@pytest.mark.parametrize("format,mode", CASES)
def test_pixels_decode_once_and_retain_metadata(monkeypatch, format, mode):
    channels = PillowImage.getmodebands(mode)
    raw = bytes((i * 37 + i // 7) % 256 for i in range(19 * 13 * channels))
    stream = BytesIO()
    PillowImage.frombytes(mode, (19, 13), raw).save(stream, format)
    encoded = stream.getvalue()
    native_decode = Image._open_sequence
    calls = []

    def decode(*args):
        calls.append(1)
        return native_decode(*args)

    monkeypatch.setattr(Image, "_open_sequence", decode)
    stream = BytesIO(encoded)
    image = Image.open(stream)
    stream.close()
    image.info["custom"] = "retained"
    expected = PillowImage.open(BytesIO(encoded)).tobytes()
    assert image.tobytes() == expected
    assert image.copy().tobytes() == expected
    assert image.getpixel((0, 0)) == PillowImage.open(BytesIO(encoded)).getpixel((0, 0))
    assert image.info["custom"] == "retained"
    assert len(calls) == 1
    image.close()


@pytest.mark.parametrize("access", ["getpixel", "tobytes", "copy", "putpixel", "convert", "encode"])
def test_pixel_access_checks_truncated_jpeg_policy(monkeypatch, access):
    data = Image.new("RGB", (32, 32), "red").encode("JPEG")[:-2]
    image = Image.open(BytesIO(data))

    def pixels():
        if access == "getpixel":
            return image.getpixel((0, 0))
        if access == "putpixel":
            return image.putpixel((0, 0), (0, 255, 0))
        if access == "convert":
            return image.convert("L")
        if access == "encode":
            return image.encode("PNG")
        return getattr(image, access)()

    monkeypatch.setattr(ImageFile, "LOAD_TRUNCATED_IMAGES", False)
    assert image.size == (32, 32)
    with pytest.raises(OSError, match="broken data stream"):
        pixels()
    monkeypatch.setattr(ImageFile, "LOAD_TRUNCATED_IMAGES", True)
    pixels()
    image.close()


def test_truncated_jpeg_can_close_without_loading(monkeypatch):
    data = Image.new("RGB", (32, 32), "red").encode("JPEG")[:-2]
    monkeypatch.setattr(ImageFile, "LOAD_TRUNCATED_IMAGES", False)
    image = Image.open(BytesIO(data))
    image.close()
    image.close()
    with pytest.raises(ValueError, match="closed"):
        image.load()


@pytest.mark.parametrize("method", ["lens_map", "polar_map"])
def test_coordinate_map_preserves_deferred_decode(monkeypatch, method):
    data = Image.new("RGB", (63, 47), "red").encode("JPEG")

    def unexpected_decode(*args):
        pytest.fail("coordinate map decoded source pixels")

    monkeypatch.setattr(Image, "_open_sequence", unexpected_decode)
    with Image.open(BytesIO(data)) as image:
        assert getattr(image, method)().size == (63, 47)
        assert image.draft(None, (8, 7)) is not None
    image.close()
    with pytest.raises(ValueError, match="closed"):
        getattr(image, method)()


@pytest.mark.parametrize("corruption", ["checksum", "truncated", "pixels"])
def test_png_verification_and_deferred_decode_errors(corruption):
    encoded = bytearray(Image.new("RGB", (19, 13), "red").encode("PNG"))
    position = encoded.index(b"IDAT")
    length = int.from_bytes(encoded[position - 4 : position], "big")
    if corruption == "truncated":
        del encoded[-12:]
    else:
        encoded[position + 4] ^= 255
        if corruption == "pixels":
            checksum = zlib.crc32(encoded[position : position + 4 + length])
            encoded[position + 4 + length : position + 8 + length] = checksum.to_bytes(4, "big")
    image = Image.open(BytesIO(encoded))
    if corruption == "pixels":
        image.verify()
        with pytest.raises(OSError):
            image.load()
    else:
        with pytest.raises(OSError):
            image.verify()
    image.close()
