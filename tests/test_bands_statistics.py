"""Band extraction, integer reduction, and entropy parity."""

from __future__ import annotations

import math

import pytest
from blanket import Image
from PIL import Image as PILImage


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
@pytest.mark.parametrize("size", [(1, 1), (2, 3), (3, 2), (3, 3), (4, 5), (99, 96), (100, 98), (803, 607)])
def test_reduce_three_edges(mode: str, size: tuple[int, int]) -> None:
    raw = bytes((i * 37 + i // 7) % 256 for i in range(size[0] * size[1] * len(mode)))
    image = Image.frombytes(mode, size, raw)
    reference = PILImage.frombytes(mode, size, raw)
    assert image.reduce(3).tobytes() == reference.reduce(3).tobytes()


def pair(mode: str) -> tuple[Image.Image, PILImage.Image]:
    raw = bytes((i * 37 + i // 7) % 256 for i in range(17 * 13 * len(mode)))
    return Image.frombytes(mode, (17, 13), raw), PILImage.frombytes(mode, (17, 13), raw)


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
def test_split(mode: str) -> None:
    image, reference = pair(mode)
    image.info["custom"] = 42
    bands = image.split()
    assert isinstance(bands, tuple) and len(bands) == len(mode)
    for band, expected in zip(bands, reference.split()):
        assert band.mode == "L" and band.size == image.size
        assert band.tobytes() == expected.tobytes()
        assert band.info == image.info and band.info is not image.info
    image.close()
    assert bands[0].tobytes() == reference.split()[0].tobytes()
    if len(bands) > 1:
        bands[0].close()
        assert bands[1].tobytes() == reference.split()[1].tobytes()


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
@pytest.mark.parametrize("factor", [1, 2, 3, 4, 5, (1, 3), (3, 1), (2, 5), (100, 100)])
@pytest.mark.parametrize("box", [None, (1, 2, 16, 12), (0, 0, 16, 12)])
def test_reduce(mode: str, factor: object, box: object) -> None:
    image, reference = pair(mode)
    image.info["custom"] = 42
    result = image.reduce(factor, box)
    expected = reference.reduce(factor, box)
    assert result.size == expected.size and result.mode == expected.mode
    assert result.tobytes() == expected.tobytes()
    assert result is not image
    assert result.info == image.info and result.info is not image.info


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


@pytest.mark.parametrize("factor,box", [(0, None), (-1, None), ((2, 0), None), (2, (0, 0, 0, 2)), (2, (-1, 0, 2, 2)), (2, (0, 0, 100, 2))])
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
