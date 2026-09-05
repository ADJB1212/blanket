from __future__ import annotations

from io import BytesIO

import pytest

pytest.importorskip("pillow_jxl")
from PIL import Image as PillowImage

from blanket import Image as BlanketImage
from test_api import pixels


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
@pytest.mark.parametrize("lossless,effort", [(True, 1), (True, 7), (False, 3), (False, 7)])
def test_jxl_cross_library_pixels(mode: str, lossless: bool, effort: int) -> None:
    # Cross a codec group boundary and exercise alpha and grayscale as well as RGB.
    size = (257, 259)
    raw = pixels(mode, size)
    for library in (BlanketImage, PillowImage):
        output = BytesIO()
        library.frombytes(mode, size, raw).save(
            output, "JXL", lossless=lossless, quality=90, effort=effort
        )
        payload = output.getvalue()
        with BlanketImage.open(BytesIO(payload)) as blanket:
            with PillowImage.open(BytesIO(payload)) as pillow:
                pillow.load()
                assert (blanket.mode, blanket.size) == (pillow.mode, pillow.size) == (mode, size)
                assert blanket.tobytes() == pillow.tobytes()
                if lossless:
                    assert blanket.tobytes() == raw
