from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image as PillowImage

from blanket import Image, UnidentifiedImageError


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
@pytest.mark.parametrize("writer", ["blanket", "pillow"])
def test_avif_interop(mode: str, writer: str) -> None:
    color = {"L": 120, "RGB": (60, 120, 180), "RGBA": (60, 120, 180, 150)}[mode]
    source = PillowImage.new(mode, (19, 13), color)
    output = BytesIO()
    if writer == "blanket":
        Image.frombytes(mode, source.size, source.tobytes()).save(output, "AVIF", quality=95, effort=1)
    else:
        source.save(output, "AVIF", quality=95, speed=10)
    data = output.getvalue()
    actual = Image.open(BytesIO(data), formats=["AVIF"])
    expected = PillowImage.open(BytesIO(data)).convert("RGBA")
    assert actual.format == "AVIF"
    assert actual.size == expected.size == source.size
    assert actual.mode == expected.mode == "RGBA"
    # Independent AV1 decoders can round YUV conversions differently.
    assert max(abs(a - b) for a, b in zip(actual.tobytes(), expected.tobytes(), strict=False)) <= 3
    assert max(abs(a - b) for a, b in zip(actual.tobytes(), source.convert(actual.mode).tobytes(), strict=False)) <= 5
    with pytest.raises(UnidentifiedImageError):
        Image.open(BytesIO(data), formats=["HEIF"])
    with pytest.raises(UnidentifiedImageError):
        Image.open(BytesIO(data[:32]))


def test_avif_extension(tmp_path: Path) -> None:
    path = tmp_path / "image.AVIF"
    Image.frombytes("RGB", (8, 8), bytes([255, 0, 0]) * 64).save(path, effort=1)
    assert Image.open(path).format == "AVIF"


def test_avif_requires_explicit_eight_bit_conversion() -> None:
    source = Image.frombytes("L", (1, 1), b"\xff\x03", bit_depth=10)
    with pytest.raises(ValueError, match="8"):
        source.save(BytesIO(), "AVIF")


def test_avif_nonuniform_pixels_and_alpha() -> None:
    raw = bytes(value for i in range(19 * 13) for value in (i % 256, (i * 3) % 256, (i * 7) % 256, i % 256))
    output = BytesIO()
    Image.frombytes("RGBA", (19, 13), raw).save(output, "AVIF", quality=100, effort=1)
    actual = Image.open(BytesIO(output.getvalue()))
    expected = PillowImage.open(BytesIO(output.getvalue()))
    assert actual.mode == expected.mode == "RGBA"
    assert actual.size == expected.size == (19, 13)
    assert max(abs(a - b) for a, b in zip(actual.tobytes(), expected.tobytes(), strict=False)) <= 3
    assert max(abs(a - b) for a, b in zip(actual.tobytes()[3::4], raw[3::4], strict=False)) <= 1


@pytest.mark.parametrize(
    "options,error",
    [({"quality": 0}, ValueError), ({"quality": 101}, ValueError), ({"effort": 0}, ValueError), ({"effort": 11}, ValueError), ({"quality": True}, TypeError), ({"lossless": True}, TypeError)],
)
def test_avif_invalid_options(options: dict[str, object], error: type[Exception]) -> None:
    with pytest.raises(error):
        Image.frombytes("RGB", (8, 8), bytes(8 * 8 * 3)).save(BytesIO(), "AVIF", **options)
