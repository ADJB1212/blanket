"""Native reduced decoding runs before allocating thumbnail rasters."""

from __future__ import annotations

from io import BytesIO
from shutil import which
from subprocess import run

import pytest
from helpers import image_buffer
from PIL import Image as PillowImage

from blanket import CorruptImageError, Image


@pytest.mark.parametrize("format,mode,lossless", [("JPEG", "L", False), ("JPEG", "RGB", False), ("JPEG", "CMYK", False)] + [("WEBP", mode, lossless) for mode in ("RGB", "RGBA") for lossless in (False, True)])
def test_thumbnail_uses_reduced_native_raster(monkeypatch, format, mode, lossless):
    size = (513, 389)
    channels = len(mode)
    raw = bytes((i * 37 + i // 7) % 256 for i in range(size[0] * size[1] * channels))
    encoded = Image.frombytes(mode, size, raw).encode(format, **({"lossless": lossless} if format == "WEBP" else {}))
    name = "_draft_jpeg" if format == "JPEG" else "_draft_webp"
    decoder = getattr(Image, name)
    decoded_sizes = []

    def decode(*args):
        native = decoder(*args)
        decoded_sizes.append(native.size)
        return native

    def unexpected_full_decode(*args):
        pytest.fail("thumbnail allocated a full raster")

    monkeypatch.setattr(Image, name, decode)
    monkeypatch.setattr(Image, "_open_sequence", unexpected_full_decode)
    stream = BytesIO(encoded)
    image = Image.open(stream)
    stream.close()
    assert image._decoded is None
    image.info["custom"] = "retained"
    image.thumbnail((31, 23))
    assert len(decoded_sizes) == 1
    assert decoded_sizes[0][0] < size[0] // 2
    assert decoded_sizes[0][1] < size[1] // 2
    assert image.size == (30, 23)
    assert image.info["custom"] == "retained"
    assert image.format == format
    assert image.mode == mode
    assert image.bit_depth == 8
    before = image.tobytes()
    assert image.tobytes() == before
    assert len(decoded_sizes) == 1


@pytest.mark.parametrize("size", [(513, 389), (64, 48), (65, 49)])
@pytest.mark.parametrize("mode", ["L", "RGB", "CMYK"])
@pytest.mark.parametrize("resample", [0, 1, 2, 3])
@pytest.mark.parametrize("gap", [None, 1.0, 2.0, 3.0])
def test_jpeg_thumbnail_matches_pillow(size, mode, resample, gap):
    raw = bytes((i * 37 + i // 7) % 256 for i in range(size[0] * size[1] * len(mode)))
    stream = BytesIO()
    PillowImage.frombytes(mode, size, raw).save(stream, "JPEG")
    data = stream.getvalue()
    with Image.open(BytesIO(data)) as image, PillowImage.open(BytesIO(data)) as reference:
        image.thumbnail((7.9, 5.2), resample, reducing_gap=gap)
        reference.thumbnail((7.9, 5.2), resample, reducing_gap=gap)
        assert image.size == reference.size
        assert image.tobytes() == reference.tobytes()


@pytest.mark.parametrize("mode", ["RGB", "RGBA"])
@pytest.mark.parametrize("lossless", [False, True])
def test_webp_draft_is_deferred_and_one_shot(monkeypatch, mode, lossless):
    raw = bytes((i * 37 + i // 7) % 256 for i in range(63 * 47 * len(mode)))
    data = Image.frombytes(mode, (63, 47), raw).encode("WEBP", lossless=lossless)
    with Image.open(BytesIO(data)) as image:
        assert image._decoded is None
        decoder = Image._draft_webp
        calls = []

        def decode(*args):
            calls.append(args[1])
            return decoder(*args)

        monkeypatch.setattr(Image, "_draft_webp", decode)
        assert image.draft("L", (8, 7)) == (mode, (0, 0, 11.0, 8.0))
        assert image.size == (11, 8)
        assert image.mode == mode
        assert image._decoded is None
        assert not calls
        assert image.draft(None, (1, 1)) is None
        assert image.tobytes() == Image.Image(decoder(data, (11, 8))).tobytes()
        assert calls == [(11, 8)]
        image.load()
        assert calls == [(11, 8)]


@pytest.mark.parametrize("format", ["JPEG", "WEBP"])
def test_loaded_images_and_disabled_reduction_use_full_pixels(monkeypatch, format):
    encoded = Image.new("RGB", (129, 97), "red").encode(format)

    def unexpected_draft(*args):
        pytest.fail("native reduction was disabled")

    monkeypatch.setattr(Image, "_draft_jpeg", unexpected_draft)
    monkeypatch.setattr(Image, "_draft_webp", unexpected_draft)
    for loaded in [False, True]:
        with Image.open(BytesIO(encoded)) as image:
            if loaded:
                image.load()
                assert image.draft(None, (1, 1)) is None
            image.thumbnail((13, 11), reducing_gap=None if not loaded else 2)
            assert image.size == (13, 10)


@pytest.mark.parametrize("lazy", [False, True])
def test_animated_webp_retains_frame_navigation(monkeypatch, lazy):
    first = Image.new("RGBA", (65, 49), (255, 0, 0, 127))
    second = Image.new("RGBA", first.size, (0, 255, 0, 255))
    stream = image_buffer(first, "WEBP", save_all=True, append_images=[second], duration=[20, 40], lossless=True)

    def unexpected_draft(*args):
        pytest.fail("animated WebP used still-image scaled decoding")

    monkeypatch.setattr(Image, "_draft_webp", unexpected_draft)
    with Image.open(stream, lazy=lazy) as image:
        image.seek(1)
        assert image.draft(None, (4, 4)) is None
        image.thumbnail((9, 9))
        assert image.size == (9, 7)
        assert image.tell() == 1
        assert image.info["duration"] == 40
        image.seek(0)
        assert image.size == (65, 49)


@pytest.mark.parametrize("depth", [8, 16])
def test_png_fallback_preserves_depth(monkeypatch, depth):
    source = Image.new("RGB", (65, 49), (17, 51, 99)).convert("RGB", bit_depth=depth)
    data = source.encode("PNG")

    def unexpected_draft(*args):
        pytest.fail("PNG used reduced JPEG/WebP decoding")

    monkeypatch.setattr(Image, "_draft_jpeg", unexpected_draft)
    monkeypatch.setattr(Image, "_draft_webp", unexpected_draft)
    with Image.open(BytesIO(data)) as image:
        assert image.draft(None, (4, 4)) is None
        if depth == 16:
            with pytest.raises(ValueError, match="requires 8-bit pixels"):
                image.thumbnail((9, 9))
            assert image.bit_depth == 16
            assert image.size == source.size
            return
        image.thumbnail((9, 9))
        assert image.size == (9, 7)
        assert image.bit_depth == depth


@pytest.mark.parametrize("format", ["JPEG", "WEBP"])
@pytest.mark.parametrize("size,error", [((0, 1), ValueError), ((1, -1), ValueError), ((1,), ValueError), ((True, 1), TypeError), ((1.5, 1), TypeError)])
def test_invalid_draft_keeps_original_header(format, size, error):
    data = Image.new("RGB", (63, 47)).encode(format)
    with Image.open(BytesIO(data)) as image:
        with pytest.raises(error):
            image.draft(None, size)
        assert image.size == (63, 47)
        assert image._decoded is None
        assert image.draft(None, (8, 7)) is not None


@pytest.mark.parametrize("gap", [0, -1, float("nan"), float("inf")])
def test_invalid_thumbnail_gap_does_not_change_draft(gap):
    data = Image.new("RGB", (63, 47)).encode("JPEG")
    with Image.open(BytesIO(data)) as image:
        with pytest.raises(ValueError, match="reducing_gap"):
            image.thumbnail((4, 4), reducing_gap=gap)
        assert image.size == (63, 47)
        assert image._decoded is None


def test_webp_close_after_draft_and_no_upscale(monkeypatch):
    data = Image.new("RGB", (63, 47)).encode("WEBP")
    image = Image.open(BytesIO(data))
    image.thumbnail((100, 100))
    assert image.size == (63, 47)
    assert image._decoded is None
    image.draft(None, (8, 7))
    image.close()
    with pytest.raises(ValueError, match="closed"):
        image.load()


@pytest.mark.parametrize("mode", ["RGB", "RGBA"])
@pytest.mark.parametrize("lossless", [False, True])
def test_webp_scaled_interoperability(tmp_path, mode, lossless):
    decoder = which("dwebp")
    if decoder is None:
        pytest.skip("dwebp is not installed")
    raw = bytes((i * 37 + i // 7) % 256 for i in range(63 * 47 * len(mode)))
    encoded = Image.frombytes(mode, (63, 47), raw).encode("WEBP", lossless=lossless)
    source = tmp_path / "source.webp"
    output = tmp_path / "scaled.png"
    source.write_bytes(encoded)
    run([decoder, str(source), "-resize", "11", "8", "-o", str(output), "-quiet"], check=True, capture_output=True)
    with Image.open(BytesIO(encoded)) as actual, PillowImage.open(output) as expected:
        actual.draft(None, (8, 7))
        assert actual.size == expected.size
        assert actual.tobytes() == expected.convert(mode).tobytes()


@pytest.mark.parametrize("data", [b"not webp", b"RIFF\x00\x00\x00\x00WEBP"])
def test_invalid_scaled_webp_preserves_codec_error(data):
    with pytest.raises(CorruptImageError) as caught:
        Image._draft_webp(data, (1, 1))
    assert caught.value.format == "WEBP"
    assert caught.value.operation == "decode"


@pytest.mark.parametrize("format", ["JPEG", "WEBP"])
def test_thumbnail_enforces_original_dimension_limits(monkeypatch, format):
    data = Image.new("RGB", (63, 47)).encode(format)
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 100)
    with pytest.raises(Image.DecompressionBombError):
        Image.open(BytesIO(data)).thumbnail((1, 1))
