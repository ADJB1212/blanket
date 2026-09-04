from __future__ import annotations

from io import BytesIO

import pytest

PIL = pytest.importorskip("PIL")
from PIL import Image as PillowImage

from blanket import Image as BlanketImage

from test_api import pixels


@pytest.mark.parametrize("source_mode", ["L", "RGB", "RGBA"])
@pytest.mark.parametrize("target_mode", ["L", "RGB", "RGBA"])
def test_convert_matches_pillow(source_mode: str, target_mode: str) -> None:
    raw = pixels(source_mode)
    blanket = BlanketImage.frombytes(source_mode, (17, 13), raw)
    pillow = PillowImage.frombytes(source_mode, (17, 13), raw)
    assert blanket.convert(target_mode).tobytes() == pillow.convert(target_mode).tobytes()


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
def test_png_cross_library_roundtrip(mode: str) -> None:
    raw = pixels(mode)

    pillow_output = BytesIO()
    PillowImage.frombytes(mode, (17, 13), raw).save(pillow_output, "PNG")
    assert BlanketImage.open(pillow_output).tobytes() == raw

    blanket_output = BytesIO()
    BlanketImage.frombytes(mode, (17, 13), raw).save(blanket_output, "PNG")
    blanket_output.seek(0)
    with PillowImage.open(blanket_output) as loaded:
        loaded.load()
        assert loaded.mode == mode
        assert loaded.tobytes() == raw


def test_jpeg_cross_library_loads() -> None:
    raw = pixels("RGB")
    pillow_output = BytesIO()
    PillowImage.frombytes("RGB", (17, 13), raw).save(
        pillow_output, "JPEG", quality=85
    )
    assert BlanketImage.open(pillow_output).size == (17, 13)

    blanket_output = BytesIO()
    BlanketImage.frombytes("RGB", (17, 13), raw).save(
        blanket_output, "JPEG", quality=85
    )
    blanket_output.seek(0)
    with PillowImage.open(blanket_output) as loaded:
        loaded.load()
        assert (loaded.mode, loaded.size) == ("RGB", (17, 13))


def test_to_pillow_returns_real_pillow_image() -> None:
    raw = pixels("RGBA")
    converted = BlanketImage.frombytes("RGBA", (17, 13), raw).to_pillow()
    assert isinstance(converted, PillowImage.Image)
    assert converted.tobytes() == raw
