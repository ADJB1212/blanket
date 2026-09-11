from __future__ import annotations

from io import BytesIO

import pytest

pytest.importorskip("pillow_jxl")
from blanket import Image as BlanketImage
from PIL import Image as PillowImage
from test_api import pixels


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
@pytest.mark.parametrize("lossless,effort", [(True, 1), (True, 7), (False, 3), (False, 7)])
def test_jxl_cross_library_pixels(mode: str, lossless: bool, effort: int) -> None:
    # Cross a codec group boundary and exercise alpha and grayscale as well as RGB.
    size = (257, 259)
    raw = pixels(mode, size)
    for library in (BlanketImage, PillowImage):
        output = BytesIO()
        library.frombytes(mode, size, raw).save(output, "JXL", lossless=lossless, quality=90, effort=effort)
        payload = output.getvalue()
        with BlanketImage.open(BytesIO(payload)) as blanket, PillowImage.open(BytesIO(payload)) as pillow:
            pillow.load()
            assert (blanket.mode, blanket.size) == (pillow.mode, pillow.size) == (mode, size)
            blanket_pixels = blanket.tobytes()
            pillow_pixels = pillow.tobytes()
            if lossless:
                assert blanket_pixels == pillow_pixels == raw
            else:
                # Independent libjxl builds can round lossy color samples
                # differently. Limit decoder disagreement to four 8-bit level.
                differences = [abs(left - right) for left, right in zip(blanket_pixels, pillow_pixels, strict=True)]
                assert max(differences) <= 4, f"maximum lossy decoder difference: {max(differences)}"
                if mode == "RGBA":
                    assert blanket_pixels[3::4] == pillow_pixels[3::4] == raw[3::4]
