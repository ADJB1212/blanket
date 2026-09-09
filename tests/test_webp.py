from __future__ import annotations

from io import BytesIO

import pytest
from PIL import Image as PillowImage

from blanket import Image, UnidentifiedImageError


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
@pytest.mark.parametrize("lossless", [False, True])
def test_webp_decode_matches_pillow(mode: str, lossless: bool) -> None:
    channels = {"L": 1, "RGB": 3, "RGBA": 4}[mode]
    raw = bytes((i * 31 + i // 7) % 256 for i in range(19 * 13 * channels))
    stream = BytesIO()
    PillowImage.frombytes(mode, (19, 13), raw).save(
        stream, "WEBP", lossless=lossless, quality=85
    )
    expected = PillowImage.open(BytesIO(stream.getvalue()))
    actual = Image.open(BytesIO(stream.getvalue()))
    assert (actual.mode, actual.size, actual.format) == (
        expected.mode, expected.size, "WEBP"
    )
    assert actual.tobytes() == expected.tobytes()
    with pytest.raises(UnidentifiedImageError):
        Image.open(BytesIO(stream.getvalue()[:30]))


@pytest.mark.parametrize("lossless", [False, True])
def test_webp_animation_first_frame(lossless: bool) -> None:
    stream = BytesIO()
    first = PillowImage.new("RGB", (19, 13), "red")
    first.save(stream, "WEBP", save_all=True, append_images=[
        PillowImage.new("RGB", first.size, "blue")
    ], lossless=lossless, duration=100)
    actual = Image.open(BytesIO(stream.getvalue()))
    expected = PillowImage.open(BytesIO(stream.getvalue()))
    assert actual.convert("RGBA").tobytes() == expected.convert("RGBA").tobytes()


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
@pytest.mark.parametrize("lossless", [False, True])
def test_webp_encode_matches_pillow_decode(mode: str, lossless: bool) -> None:
    channels = {"L": 1, "RGB": 3, "RGBA": 4}[mode]
    raw = bytes((i * 31 + i // 7) % 256 for i in range(19 * 13 * channels))
    source = Image.frombytes(mode, (19, 13), raw)
    stream = BytesIO()
    source.save(stream, "WEBP", lossless=lossless, quality=85)
    expected = PillowImage.open(BytesIO(stream.getvalue()))
    actual = Image.open(BytesIO(stream.getvalue()))
    assert (actual.mode, actual.size, actual.format) == (
        expected.mode, expected.size, "WEBP"
    )
    assert actual.tobytes() == expected.tobytes()
    if mode == "RGBA":
        assert actual.tobytes()[3::4] == raw[3::4]
    if lossless:
        # RGB values under fully transparent pixels may be discarded by WebP.
        converted = source.convert(actual.mode).tobytes()
        decoded = actual.tobytes()
        stride = len(actual.mode)
        for offset in range(0, len(converted), stride):
            if mode != "RGBA" or converted[offset + 3]:
                assert decoded[offset:offset + stride] == (
                    converted[offset:offset + stride]
                )
