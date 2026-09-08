"""Public geometric operations compared with Pillow."""

from __future__ import annotations

import pytest
from blanket import Image
from PIL import Image as PILImage
from PIL import ImageTransform


def pair(mode: str) -> tuple[Image.Image, PILImage.Image]:
    raw = bytes((i * 37 + i // 7) % 256 for i in range(13 * 9 * len(mode)))
    return Image.frombytes(mode, (13, 9), raw), PILImage.frombytes(mode, (13, 9), raw)


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
@pytest.mark.parametrize("resample", [0, 2, 3])
@pytest.mark.parametrize(
    "method,data",
    [
        (Image.AFFINE, (0.3, 0, -0.05, 0, -1.1, 210)),
        (Image.AFFINE, (0.7, 0.2, -2, -0.1, 1.2, 1)),
        (Image.EXTENT, (-2, 1, 280, 208)),
        (Image.PERSPECTIVE, (0.9, 0.1, -1, -0.1, 1.1, 1, 0.002, -0.001)),
        (Image.QUAD, (-1, 1, 2, 211, 277, 208, 280, -2)),
        (Image.MESH, [((0, 0, 259, 193), (-1, 1, 2, 211, 277, 208, 280, -2)), ((31, 17, 241, 181), (0, 0, 0, 211, 277, 211, 277, 0))]),
    ],
)
def test_transform_parallel_rows(mode: str, resample: int, method: int, data: object) -> None:
    size = (277, 211)
    raw = bytes((i * 37 + i // 7) % 256 for i in range(size[0] * size[1] * len(mode)))
    image = Image.frombytes(mode, size, raw)
    reference = PILImage.frombytes(mode, size, raw)
    actual = image.transform((259, 193), method, data, resample, fillcolor=17)
    expected = reference.transform((259, 193), method, data, resample, fillcolor=17)
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
@pytest.mark.parametrize("method", list(Image.Transpose))
def test_transpose(mode: str, method: int) -> None:
    image, reference = pair(mode)
    image.info["test"] = 42
    result = image.transpose(method)
    expected = reference.transpose(method)
    assert result.mode == expected.mode
    assert result.size == expected.size
    assert result.tobytes() == expected.tobytes()
    assert result.info == image.info and result.info is not image.info
    image.close()
    assert result.tobytes() == expected.tobytes()


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
@pytest.mark.parametrize("resample", [0, 2, 3])
@pytest.mark.parametrize("fillcolor", [None, "#12345680", 17])
@pytest.mark.parametrize("size", [(13, 9), (19, 14)])
@pytest.mark.parametrize(
    "method,data",
    [
        (Image.AFFINE, (1, 0, 0, 0, 1, 0)),
        (Image.AFFINE, (0.7, 0.2, -2, -0.1, 1.2, 1)),
        (Image.EXTENT, (-2, 1, 15, 8)),
        (Image.PERSPECTIVE, (0.9, 0.1, -1, -0.1, 1.1, 1, 0.02, -0.01)),
        (Image.QUAD, (-1, 1, 2, 10, 12, 8, 14, -2)),
        (Image.MESH, [((0, 0, 10, 8), (0, 0, 0, 9, 13, 9, 13, 0)), ((4, 3, 17, 12), (-2, -2, -2, 10, 14, 10, 14, -2))]),
        (Image.MESH, [((-2, -3, 8, 7), (0, 0, 0, 9, 13, 9, 13, 0))]),
        (Image.MESH, []),
    ],
)
def test_transform_parity(mode: str, resample: int, fillcolor: object, size: tuple[int, int], method: int, data: object) -> None:
    image, reference = pair(mode)
    image.info["test"] = 42
    result = image.transform(size, method, data, resample, fillcolor=fillcolor)
    expected = reference.transform(size, method, data, resample, fillcolor=fillcolor)
    assert result.size == expected.size
    assert result.mode == expected.mode
    assert result.tobytes() == expected.tobytes()
    assert result.info == image.info and result.info is not image.info


def test_transform_getdata() -> None:
    image, reference = pair("RGB")
    method = ImageTransform.AffineTransform((1, 0, -2, 0, 1, 3))
    assert image.transform((8, 7), method).tobytes() == reference.transform((8, 7), method).tobytes()


def test_transform_handler() -> None:
    class Handler(Image.ImageTransformHandler):
        def transform(self, size, image, resample=0, fill=1):
            assert size == (8, 7) and resample == 2 and fill == 0
            return image.crop((0, 0, *size))

    image, _ = pair("RGB")
    assert image.transform((8, 7), Handler(), resample=2, fill=0).size == (8, 7)


@pytest.mark.parametrize("method,data,resample", [(99, (), 0), (0, None, 0), (0, (1, 2), 0), (0, (1, 0, 0, 0, 1, 0), 1), (2, (float("nan"),) * 8, 0)])
def test_invalid_transform(method: int, data: object, resample: int) -> None:
    image, _ = pair("L")
    with pytest.raises(ValueError):
        image.transform((5, 5), method, data, resample)


def test_closed_and_invalid_transpose() -> None:
    image, _ = pair("L")
    with pytest.raises(ValueError):
        image.transpose(7)
    image.close()
    with pytest.raises(ValueError):
        image.transpose(Image.TRANSPOSE)
    with pytest.raises(ValueError):
        image.transform((5, 5), Image.AFFINE, (1, 0, 0, 0, 1, 0))


@pytest.mark.parametrize("scale", [0.1, 0.2, 0.3, 1.1])
def test_nearest_fractional_boundaries(scale: float) -> None:
    image, reference = pair("L")
    data = (scale, 0, -0.05, 0, scale, 0)
    assert image.transform((80, 80), Image.AFFINE, data).tobytes() == reference.transform((80, 80), PILImage.Transform.AFFINE, data).tobytes()


@pytest.mark.parametrize("target", [(15, 17), (16, 16), (33, 31), (400, 300), (800, 600)])
@pytest.mark.parametrize("extent", [(80.0, 60.0, 720.0, 540.0), (799.9, 599.9, 0.1, 0.1), (0.1, 0.1, 10.1, 10.1), (3.5, 2.5, 3.5, 2.5), (-10.0, -10.0, 810.0, 610.0)])
def test_nearest_extent_gather(target: tuple[int, int], extent: tuple[float, float, float, float]) -> None:
    raw = bytes((i * 37 + i // 7) % 256 for i in range(800 * 600))
    image = Image.frombytes("L", (800, 600), raw)
    reference = PILImage.frombytes("L", (800, 600), raw)
    actual = image.transform(target, Image.EXTENT, extent, fillcolor=19)
    expected = reference.transform(target, PILImage.Transform.EXTENT, extent, fillcolor=19)
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("size", [(0, 5), (5, 0), (0, 0)])
def test_empty_affine_and_transpose(size: tuple[int, int]) -> None:
    image, _ = pair("L")
    result = image.transform(size, Image.AFFINE, (1, 0, 0, 0, 1, 0))
    assert result.size == size and result.tobytes() == b""
    assert result.transpose(Image.TRANSPOSE).size == size[::-1]


@pytest.mark.parametrize("resample", [0, 2, 3])
def test_singular_perspective(resample: int) -> None:
    image, reference = pair("L")
    data = (1, 0, 0, 0, 1, 0, -2, 0)
    assert image.transform((5, 5), Image.PERSPECTIVE, data, resample).tobytes() == reference.transform((5, 5), PILImage.Transform.PERSPECTIVE, data, resample).tobytes()
