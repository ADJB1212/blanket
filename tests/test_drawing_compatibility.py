from __future__ import annotations

import struct
from io import BytesIO

import pytest
from PIL import Image as PillowImage

from blanket import Image, ImageDraw, ImageFont


@pytest.mark.parametrize("matrix", [(1, 0, 5, 0, 1, 3), (0, -1, 25, 1, 0, 0), (2, 0.5, 0, 0, 1, 0), (-1, 0, 30, 0, 1, 0)])
def test_outline_transform_preserves_geometry(matrix: tuple[float, ...]) -> None:
    outline = ImageDraw.Outline()
    outline.move(3, 4)
    outline.line(9, 4)
    outline.line(9, 12)
    before = list(outline.edges)
    a, b, c, d, e, f = matrix

    def point(p: tuple[int, int]) -> tuple[int, int]:
        x, y = p
        return int(a * x + b * y + c), int(d * x + e * y + f)

    outline.transform(matrix)
    assert outline.edges == [(point(p), point(q)) for p, q in before]
    outline.close()
    assert outline.edges[-1][1] == point((3, 4))
    actual = Image.new("L", (40, 40))
    expected = Image.new("L", actual.size)
    ImageDraw.Draw(actual).shape(outline, fill=255)
    ImageDraw.Draw(expected).polygon([point(p) for p in ((3, 4), (9, 4), (9, 12))], fill=255)
    assert actual.tobytes() == expected.tobytes()


def test_outline_invalid_transform_is_atomic() -> None:
    outline = ImageDraw.Outline()
    outline.move(1, 2)
    outline.curve(4, 5, 7, 8, 10, 11)
    edges = outline.edges[:]
    for matrix in ((1, 2), (1, 0, 0, 0, float("nan"), 0)):
        with pytest.raises((TypeError, ValueError)):
            outline.transform(matrix)
        assert outline.edges == edges
    outline.transform((1, 0, 0, 0, 1, 0))
    assert outline.edges == edges


def test_default_font_has_lowercase_and_integer_scaling() -> None:
    small, large = ImageFont.load_default(8), ImageFont.load_default(16)
    assert bytes(small.getmask("a")) != bytes(small.getmask("A"))
    assert set(bytes(small.getmask("Readable 123!?"))) == {0, 255}
    mask = small.getmask("Readable 123!?")
    enlarged = Image.frombytes("L", mask.size, bytes(mask)).resize((mask.size[0] * 2, mask.size[1] * 2), Image.Resampling.NEAREST)
    assert bytes(large.getmask("Readable 123!?")) == enlarged.tobytes()


def test_apng_precision_and_explicit_pillow_view() -> None:
    frames = [Image.frombytes("RGBA", (2, 1), struct.pack("<8H", *(sample * 2)), bit_depth=16) for sample in ((12345, 23456, 34567, 65535), (32768, 0, 65535, 65535))]
    output = BytesIO()
    frames[0].save(output, "PNG", save_all=True, append_images=frames[1:], duration=[20, 40], blend=[0, 0])
    image = Image.decode(output.getvalue())
    reference = PillowImage.open(BytesIO(output.getvalue()))
    for i in range(2):
        image.seek(i)
        reference.seek(i)
        original = image.tobytes()
        view = image.convert("RGBA", bit_depth=8)
        assert view.mode == "RGBA" and view.bit_depth == 8
        assert max(abs(a - b) for a, b in zip(view.getpixel((0, 0)), reference.getpixel((0, 0)), strict=True)) <= 1
        assert image.bit_depth == 16 and image.tobytes() == original == frames[i].tobytes()
    image.seek(0)
    assert image.tobytes() == frames[0].tobytes()
