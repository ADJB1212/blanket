from __future__ import annotations

from io import BytesIO

import pytest
from PIL import Image as PillowImage

from blanket import Image


@pytest.mark.parametrize("source_mode", ["L", "RGB", "CMYK"])
@pytest.mark.parametrize("mode", [None, "RGB", "L", "YCbCr"])
@pytest.mark.parametrize("size", [None, (100, 100), (17, 13), (8, 7), (1, 1)])
def test_jpeg_draft_matches_decoder(source_mode: str, mode: str | None, size: tuple[int, int] | None) -> None:
    dimensions = (63, 47)
    channels = len(source_mode)
    pixels = bytes((value * 37 + value // 7) % 256 for value in range(63 * 47 * channels))
    source = PillowImage.frombytes(source_mode, dimensions, pixels)
    output = BytesIO()
    source.save(output, "JPEG")
    encoded = output.getvalue()
    with Image.open(BytesIO(encoded)) as actual, PillowImage.open(BytesIO(encoded)) as expected:
        assert actual.draft(mode, size) == expected.draft(mode, size)
        assert actual.mode == expected.mode
        assert actual.size == expected.size
        assert actual.tobytes() == expected.tobytes()
        assert actual.draft(mode, size) is None
    assert Image.open(BytesIO(encoded)).tobytes() == PillowImage.open(BytesIO(encoded)).tobytes()


def test_loaded_jpeg_draft_is_noop() -> None:
    encoded = Image.new("RGB", (17, 13), "red").encode("JPEG")
    actual = Image.open(BytesIO(encoded))
    actual.load()
    assert actual.draft("L", (1, 1)) is None
    assert actual.mode == "RGB"
    assert actual.size == (17, 13)
