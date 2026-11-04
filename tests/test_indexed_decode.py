from __future__ import annotations

from io import BytesIO
from random import Random

import pytest
from helpers import encode_image, image_pair
from PIL import Image as PillowImage

from blanket import Image


@pytest.mark.parametrize("bits", [1, 2, 4, 8])
@pytest.mark.parametrize("transparency", [None, 0, b"\x00\x7f\xff\xff"])
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_indexed_png_preserves_palette_and_alpha(bits: int, transparency: int | bytes | None) -> None:
    source = PillowImage.new("P", (13, 5))
    source.putpalette(bytes(value % 256 for value in range(768)))
    source.putdata([value % (1 << bits) for value in range(65)])
    encoded = encode_image(source, "PNG", bits=bits, transparency=transparency)
    expected = PillowImage.open(BytesIO(encoded))
    actual = Image.open(BytesIO(encoded))
    assert actual.mode == expected.mode == "P"
    assert actual.tobytes() == expected.tobytes()
    assert actual.getpalette() == expected.getpalette()
    assert actual.info.get("transparency") == expected.info.get("transparency")
    for mode in ("RGB", "RGBA", "PA"):
        assert actual.convert(mode).tobytes() == expected.convert(mode).tobytes()
    saved = BytesIO()
    actual.save(saved, "PNG")
    reloaded = PillowImage.open(BytesIO(saved.getvalue()))
    assert reloaded.convert("RGBA").tobytes() == expected.convert("RGBA").tobytes()


@pytest.mark.parametrize("transparency", [None, 0, 3])
@pytest.mark.parametrize("lazy", [False, True])
def test_single_frame_gif_preserves_indices(transparency: int | None, lazy: bool) -> None:
    source = PillowImage.new("P", (13, 5))
    source.putpalette(bytes(value % 256 for value in range(768)))
    source.putdata([value % 4 for value in range(65)])
    encoded = encode_image(source, "GIF", transparency=transparency)
    expected = PillowImage.open(BytesIO(encoded))
    actual = Image.open(BytesIO(encoded), lazy=lazy)
    assert actual.mode == expected.mode == "P"
    assert actual.tobytes() == expected.tobytes()
    assert actual.getpalette() == expected.getpalette()
    assert actual.info.get("transparency") == expected.info.get("transparency")
    assert actual.convert("RGBA").tobytes() == expected.convert("RGBA").tobytes()


@pytest.mark.parametrize("mode", ["RGB", "RGBA", "P"])
def test_gif_save_matches_pillow_quantization(mode: str) -> None:
    size = (37, 29)
    raw = Random(914).randbytes(size[0] * size[1] * len(mode))
    actual, expected = image_pair(mode, size, raw)
    if mode == "P":
        palette = Random(915).randbytes(768)
        actual.putpalette(palette)
        expected.putpalette(palette)
    output = BytesIO()
    reference = BytesIO()
    actual.save(output, "GIF")
    expected.save(reference, "GIF")
    result = PillowImage.open(BytesIO(output.getvalue()))
    baseline = PillowImage.open(BytesIO(reference.getvalue()))
    assert result.mode == baseline.mode
    assert result.convert("RGBA").tobytes() == baseline.convert("RGBA").tobytes()


@pytest.mark.parametrize("transparent", [False, True])
@pytest.mark.parametrize("lazy", [False, True])
def test_gif_frame_modes_match_pillow(transparent: bool, lazy: bool) -> None:
    frames = [PillowImage.new("RGBA", (9, 7), color) for color in ("red", "green", "blue")]
    if transparent:
        for frame in frames:
            frame.putpixel((0, 0), (0, 0, 0, 0))
    encoded = encode_image(frames[0], "GIF", save_all=True, append_images=frames[1:], duration=30)
    expected = PillowImage.open(BytesIO(encoded))
    actual = Image.open(BytesIO(encoded), lazy=lazy)
    for position in (0, 2, 1, 0):
        actual.seek(position)
        expected.seek(position)
        assert actual.mode == expected.mode
        assert actual.tobytes() == expected.tobytes()
        assert actual.convert("RGBA").tobytes() == expected.convert("RGBA").tobytes()
