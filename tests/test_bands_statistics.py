"""Band extraction, integer reduction, and entropy parity."""

from __future__ import annotations

import math
import struct

import pytest
from PIL import Image as PILImage

from blanket import Image


def pair(mode: str) -> tuple[Image.Image, PILImage.Image]:
    raw = bytes((i * 37 + i // 7) % 256 for i in range(17 * 13 * len(mode)))
    return Image.frombytes(mode, (17, 13), raw), PILImage.frombytes(mode, (17, 13), raw)


@pytest.mark.parametrize("mode", ["L", "LA", "RGB", "RGBA", "PA"])
def test_split(mode: str) -> None:
    image, reference = pair(mode)
    image.info["custom"] = 42
    bands = image.split()
    assert isinstance(bands, tuple)
    assert len(bands) == len(mode)
    for band, expected in zip(bands, reference.split(), strict=False):
        assert band.mode == "L"
        assert band.size == image.size
        assert band.tobytes() == expected.tobytes()
        assert band.info == image.info
        assert band.info is not image.info
    image.close()
    assert bands[0].tobytes() == reference.split()[0].tobytes()
    if len(bands) > 1:
        bands[0].close()
        assert bands[1].tobytes() == reference.split()[1].tobytes()


@pytest.mark.parametrize("mode", ["LA", "PA"])
@pytest.mark.parametrize("count", [0, 1, 17, 1048577])
def test_two_band_split_preserves_parallel_tail(mode: str, count: int) -> None:
    raw = (bytes(range(256)) * (count * 2 // 256 + 1))[: count * 2]
    image = Image.frombytes(mode, (count, 1), raw)
    luma, alpha = image.split()
    assert luma.tobytes() == raw[::2]
    assert alpha.tobytes() == raw[1::2]
    assert image.tobytes() == raw


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
@pytest.mark.parametrize("depth", [10, 12, 16])
@pytest.mark.parametrize("count", [0, 1, 17, 131073])
def test_wide_split_and_getchannel_preserve_samples(mode: str, depth: int, count: int) -> None:
    maximum = (1 << depth) - 1
    values = [(i * 719 + i // 7) & maximum for i in range(count * len(mode))]
    raw = struct.pack(f"<{len(values)}H", *values)
    image = Image.frombytes(mode, (count, 1), raw, bit_depth=depth)
    for channel, band in enumerate(image.split()):
        expected = struct.pack(f"<{count}H", *values[channel :: len(mode)])
        assert band.bit_depth == depth
        assert band.tobytes() == expected
        assert image.getchannel(channel).tobytes() == expected
    assert image.tobytes() == raw


def test_split_serializes_edited_exif_once(monkeypatch) -> None:
    image = Image.new("RGBA", (3, 2))
    exif = image.getexif()
    exif[305] = "Edited"
    serialize = Image.Exif.tobytes
    calls = []

    def tracked(self, *args, **kwargs):
        calls.append(self)
        return serialize(self, *args, **kwargs)

    monkeypatch.setattr(Image.Exif, "tobytes", tracked)
    bands = image.split()
    assert calls == [exif]
    for band in bands:
        assert band.getexif()[305] == "Edited"
        assert band.info is not image.info


@pytest.mark.parametrize("mode", ["L", "LA", "RGB", "RGBA"])
@pytest.mark.parametrize("factor", [1, 2, 3, 4, 5, (1, 3), (1, 7), (1, 100), (3, 1), (2, 5), (100, 100)])
@pytest.mark.parametrize("box", [None, (1, 2, 16, 12), (0, 0, 16, 12)])
def test_reduce(mode: str, factor: object, box: object) -> None:
    image, reference = pair(mode)
    image.info["custom"] = 42
    result = image.reduce(factor, box)
    expected = reference.reduce(factor, box)
    assert result.size == expected.size
    assert result.mode == expected.mode
    assert result.tobytes() == expected.tobytes()
    assert result is not image
    assert result.info == image.info
    assert result.info is not image.info


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
@pytest.mark.parametrize("mask_kind", [None, "empty", "partial", "full"])
def test_entropy(mode: str, mask_kind: str | None) -> None:
    image, reference = pair(mode)
    mask = pillow_mask = None
    if mask_kind is not None:
        raw = bytes((0 if mask_kind == "empty" else 255 if mask_kind == "full" else i % 3) for i in range(17 * 13))
        mask = Image.frombytes("L", image.size, raw)
        pillow_mask = PILImage.frombytes("L", image.size, raw)
    result = image.entropy(mask, extrema=(10, 20))
    expected = reference.entropy(pillow_mask, extrema=(10, 20))
    if math.isnan(expected):
        assert math.isnan(result)
    else:
        assert result == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
def test_constant_entropy(mode: str) -> None:
    image = Image.frombytes(mode, (2, 2), bytes(4 * len(mode)))
    assert image.entropy() == pytest.approx(math.log2(len(mode)))


@pytest.mark.parametrize(("factor", "box"), [(0, None), (-1, None), ((2, 0), None), (2, (0, 0, 0, 2)), (2, (-1, 0, 2, 2)), (2, (0, 0, 100, 2))])
def test_invalid_reduce(factor: object, box: object) -> None:
    image, _ = pair("L")
    with pytest.raises(ValueError):
        image.reduce(factor, box)


def test_mask_validation_and_closed_images() -> None:
    image, _ = pair("RGB")
    for mask in (Image.frombytes("L", (1, 1), b"a"), image):
        with pytest.raises(ValueError):
            image.entropy(mask)
    image.close()
    for operation in (image.split, image.entropy, lambda: image.reduce(2)):
        with pytest.raises(ValueError):
            operation()
