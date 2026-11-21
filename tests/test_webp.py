from __future__ import annotations

from io import BytesIO

import pytest
from helpers import assert_image_properties, assert_raises, image_buffer
from PIL import Image as PillowImage

from blanket import Image, UnidentifiedImageError
from blanket.errors import CodecResourceLimitError


def test_webp_checks_pixel_limits_before_allocating_output():
    encoded = bytearray(Image.new("RGB", (1, 1), "red").encode("WEBP", lossless=True))
    assert encoded[12:16] == b"VP8L"
    assert encoded[20] == 0x2F
    encoded[21:25] = (8223 | (8223 << 14)).to_bytes(4, "little")
    with pytest.raises(CodecResourceLimitError):
        Image.decode(bytes(encoded))


def test_webp_preserves_indexed_transparency():
    source = Image.new("P", (2, 1), 0)
    source.putpalette(bytes([255, 0, 0, 0, 0, 255]))
    source.putpixel((1, 0), 1)
    source.info["transparency"] = 0
    result = Image.decode(source.encode("WEBP", lossless=True))
    assert result.getchannel("A").tobytes() == b"\x00\xff"
    assert result.getpixel((1, 0)) == (0, 0, 255, 255)


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
@pytest.mark.parametrize("lossless", [False, True])
def test_webp_decode_matches_pillow(mode: str, lossless: bool) -> None:
    channels = {"L": 1, "RGB": 3, "RGBA": 4}[mode]
    raw = bytes((i * 31 + i // 7) % 256 for i in range(19 * 13 * channels))
    stream = image_buffer(PillowImage.frombytes(mode, (19, 13), raw), "WEBP", lossless=lossless, quality=85)
    expected = PillowImage.open(BytesIO(stream.getvalue()))
    actual = Image.open(BytesIO(stream.getvalue()))
    assert_image_properties(actual, mode=expected.mode, size=expected.size, format="WEBP", pixels=expected.tobytes())
    assert_raises(UnidentifiedImageError, lambda: Image.open(BytesIO(stream.getvalue()[:30])))


@pytest.mark.parametrize("lossless", [False, True])
def test_webp_animation_first_frame(lossless: bool) -> None:
    first = PillowImage.new("RGB", (19, 13), "red")
    stream = image_buffer(first, "WEBP", save_all=True, append_images=[PillowImage.new("RGB", first.size, "blue")], lossless=lossless, duration=100)
    actual = Image.open(stream)
    expected = PillowImage.open(stream)
    assert actual.convert("RGBA").tobytes() == expected.convert("RGBA").tobytes()


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("duration", [160_000, 16_777_215])
def test_webp_animation_round_trips_long_frame_delays(lazy: bool, duration: int) -> None:
    first = Image.new("RGB", (4, 4), "red")
    second = Image.new("RGB", first.size, "blue")
    encoded = first.encode("WEBP", save_all=True, append_images=[second], lossless=True, duration=[duration, 160_000])
    with Image.decode(encoded, lazy=lazy) as actual, PillowImage.open(BytesIO(encoded)) as expected:
        for frame, delay in enumerate([duration, 160_000]):
            actual.seek(frame)
            expected.seek(frame)
            expected.load()
            assert actual.info["duration"] == expected.info["duration"] == delay
            assert actual.convert("RGBA").tobytes() == expected.convert("RGBA").tobytes()


def test_webp_rejects_duration_above_the_frame_delay_field() -> None:
    first = Image.new("RGB", (4, 4), "red")
    second = Image.new("RGB", first.size, "blue")
    with pytest.raises(ValueError, match="WebP frame duration"):
        first.encode("WEBP", save_all=True, append_images=[second], duration=[16_777_216, 0])


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
@pytest.mark.parametrize("lossless", [False, True])
@pytest.mark.parametrize("size", [(19, 13), (513, 257)])
def test_webp_encode_matches_pillow_decode(mode: str, lossless: bool, size: tuple[int, int]) -> None:
    channels = {"L": 1, "RGB": 3, "RGBA": 4}[mode]
    raw = bytes((i * 31 + i // 7) % 256 for i in range(size[0] * size[1] * channels))
    source = Image.frombytes(mode, size, raw)
    stream = image_buffer(source, "WEBP", lossless=lossless, quality=85)
    expected = PillowImage.open(BytesIO(stream.getvalue()))
    actual = Image.open(BytesIO(stream.getvalue()))
    assert_image_properties(actual, mode=expected.mode, size=expected.size, format="WEBP", pixels=expected.tobytes())
    if mode == "RGBA":
        assert actual.tobytes()[3::4] == raw[3::4]
    if lossless:
        # RGB values under fully transparent pixels may be discarded by WebP.
        converted = source.convert(actual.mode).tobytes()
        decoded = actual.tobytes()
        stride = len(actual.mode)
        for offset in range(0, len(converted), stride):
            if mode != "RGBA" or converted[offset + 3]:
                assert decoded[offset : offset + stride] == (converted[offset : offset + stride])
