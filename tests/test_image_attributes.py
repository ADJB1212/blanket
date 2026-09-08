from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path

import pytest
from blanket import Image
from PIL import Image as PillowImage


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
def test_image_attributes(mode: str) -> None:
    raw = bytes([255] * len(mode))
    image = Image.frombytes(mode, (1, 1), raw)
    pillow = PillowImage.frombytes(mode, (1, 1), raw)
    assert image.filename == ""
    assert image.palette is pillow.palette is None
    assert image.is_animated is getattr(pillow, "is_animated", False)
    assert image.has_transparency_data is pillow.has_transparency_data
    image.info["transparency"] = pillow.info["transparency"] = None
    assert image.has_transparency_data is pillow.has_transparency_data is True
    del image.info["transparency"]
    del pillow.info["transparency"]
    assert image.has_transparency_data is pillow.has_transparency_data


@pytest.mark.parametrize("path_type", [str, bytes, Path])
def test_filename_from_path(tmp_path: Path, path_type: type) -> None:
    path = tmp_path / "image.png"
    PillowImage.new("RGB", (2, 2)).save(path)
    filename = os.fsencode(path) if path_type is bytes else path_type(path)
    image = Image.open(filename)
    with PillowImage.open(filename) as pillow:
        assert image.filename == pillow.filename == os.fspath(filename)
    assert image.crop().filename == ""
    assert image.convert("RGBA").filename == ""
    image.close()
    assert image.filename == os.fspath(filename)


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
@pytest.mark.parametrize("format", ["PNG", "JPEG", "JXL"])
def test_opened_image_attributes(mode: str, format: str) -> None:
    if mode == "RGBA" and format == "JPEG":
        pytest.skip("JPEG does not support alpha")
    stream = BytesIO()
    Image.frombytes(mode, (1, 1), bytes([255] * len(mode))).save(stream, format)
    image = Image.open(stream)
    assert image.filename == ""
    assert image.palette is None
    assert image.is_animated is False
    assert image.has_transparency_data is (mode == "RGBA")


def test_filename_from_named_stream(tmp_path: Path) -> None:
    path = tmp_path / "image.png"
    PillowImage.new("RGB", (1, 1)).save(path)
    with path.open("rb") as stream:
        assert Image.open(stream).filename == ""
