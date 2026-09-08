from __future__ import annotations

import random
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import pytest
from blanket import Image, UnidentifiedImageError
from PIL import Image as PillowImage


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
@pytest.mark.parametrize("level", range(10))
@pytest.mark.parametrize("random_pixels", [False, True])
def test_png_deflate_levels(mode: str, level: int, random_pixels: bool) -> None:
    size = (129, 65)
    count = size[0] * size[1] * len(mode)
    raw = random.Random(61).randbytes(count) if random_pixels else bytes(i % 251 for i in range(count))
    image = Image.frombytes(mode, size, raw)
    encoded = BytesIO()
    image.save(encoded, "PNG", compress_level=level)
    for decoder in (Image.open, PillowImage.open):
        decoded = decoder(BytesIO(encoded.getvalue()))
        assert decoded.mode == mode and decoded.size == size
        assert decoded.tobytes() == raw


def test_jxl_reused_runners_are_independent() -> None:
    cases = []
    for size in ((17, 13), (513, 129), (65, 513)):
        for mode in ("L", "RGB", "RGBA"):
            raw = random.Random(71).randbytes(size[0] * size[1] * len(mode))
            encoded = BytesIO()
            Image.frombytes(mode, size, raw).save(encoded, "JXL", lossless=True, effort=1)
            cases.append((encoded.getvalue(), mode, size, raw))

    def decode_cases(_: int) -> None:
        for payload, mode, size, raw in cases + cases[::-1]:
            with pytest.raises(UnidentifiedImageError):
                Image.open(BytesIO(b"\xff\x0a\0"))
            decoded = Image.open(BytesIO(payload))
            assert decoded.mode == mode and decoded.size == size
            assert decoded.tobytes() == raw

    decode_cases(0)
    with ThreadPoolExecutor(max_workers=3) as executor:
        list(executor.map(decode_cases, range(3)))
