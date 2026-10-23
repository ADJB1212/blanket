from __future__ import annotations

import pytest
from PIL import Image as PillowImage

from blanket import Image


@pytest.mark.parametrize("mode", ["L", "LA", "RGB", "RGBA", "HSV", "CMYK", "YCbCr", "LAB"])
@pytest.mark.parametrize("width", [1, 15, 16, 17, 33, 129])
def test_box_half_preserves_separate_pass_rounding(mode: str, width: int) -> None:
    channels = 3 if mode == "YCbCr" else len(mode)
    size = (width * 2, 34)
    raw = bytes((i * 37 + i // 7 * 19) % 256 for i in range(size[0] * size[1] * channels))
    actual = Image.frombytes(mode, size, raw)
    expected = PillowImage.frombytes(mode, size, raw)
    if mode in ("LA", "RGBA"):
        expected = expected.convert("La" if mode == "LA" else "RGBa")
    expected = expected.resize((width, 34), PillowImage.Resampling.BOX).resize((width, 17), PillowImage.Resampling.BOX)
    if mode in ("LA", "RGBA"):
        expected = expected.convert(mode)
    assert actual.resize((width, 17), Image.Resampling.BOX).tobytes() == expected.tobytes()
    assert actual.tobytes() == raw


@pytest.mark.parametrize("mode", ["L", "LA", "RGB", "RGBA"])
@pytest.mark.parametrize("method", [None, *Image.Resampling])
def test_resize_matches_pillow(mode: str, method: int | None, subtests: pytest.Subtests) -> None:
    raw = bytes((i * 37 + i // 7 * 19) % 256 for i in range(67 * 49 * len(mode)))
    image = Image.frombytes(mode, (67, 49), raw)
    reference = PillowImage.frombytes(mode, image.size, raw)
    for size, box in (((11, 7), None), ((101, 83), None), ((9, 8), (2.3, 1.7, 59.2, 43.8))):
        for gap in (None, 1.0, 2.0, 3.0):
            with subtests.test(size=size, box=box, gap=gap):
                expected = reference.resize(size, method, box, gap)
                actual = image.resize(size, method, box, gap)
                assert actual.size == expected.size
                assert actual.mode == expected.mode
                assert actual.tobytes() == expected.tobytes()
                assert image.tobytes() == raw


@pytest.mark.parametrize("mode", ["L", "LA", "RGB", "RGBA"])
@pytest.mark.parametrize("method", [Image.Resampling.BOX, Image.Resampling.BILINEAR, Image.Resampling.HAMMING, Image.Resampling.BICUBIC, Image.Resampling.LANCZOS])
def test_resize_single_axis_keeps_coefficients_and_rounding(mode: str, method: Image.Resampling, subtests: pytest.Subtests) -> None:
    raw = bytes((i * 37 + i // 7 * 19) % 256 for i in range(67 * 49 * len(mode)))
    image = Image.frombytes(mode, (67, 49), raw)
    reference = PillowImage.frombytes(mode, (67, 49), raw)
    for size in ((67, 7), (67, 83)):
        with subtests.test(size=size):
            actual = image.resize(size, method)
            expected = reference.resize(size, method)
            assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("mode", ["L", "LA", "RGB", "RGBA"])
@pytest.mark.parametrize("size", [(3, 2), (6, 4)])
def test_resize_copies_metadata_and_owns_pixels(mode: str, size: tuple[int, int]) -> None:
    raw = bytes(range(6 * len(mode)))
    image = Image.frombytes(mode, (3, 2), raw)
    image.info["custom"] = "retained"
    result = image.resize(list(size))
    assert result is not image
    assert result.info == image.info
    assert result.info is not image.info
    assert result.format is None
    image.close()
    expected = PillowImage.frombytes(mode, (3, 2), raw).resize(size)
    assert (result.mode, result.size) == (mode, size)
    assert result.tobytes() == expected.tobytes()


@pytest.mark.parametrize("mode", ["L", "LA", "RGB", "RGBA", "HSV", "CMYK", "YCbCr", "LAB"])
def test_horizontal_resize_writes_output_directly_with_full_height(mode: str, subtests: pytest.Subtests) -> None:
    channels = 3 if mode == "YCbCr" else len(mode)
    raw = bytes((i * 37 + i // 7 * 19) % 256 for i in range(67 * 49 * channels))
    actual = Image.frombytes(mode, (67, 49), raw)
    expected = PillowImage.frombytes(mode, (67, 49), raw)
    for method in Image.Resampling:
        if method == Image.Resampling.NEAREST:
            continue
        for box in (None, (1.25, 0, 61.75, 49)):
            for width in (1, 13, 513):
                with subtests.test(method=method, box=box, width=width):
                    assert actual.resize((width, 49), method, box).tobytes() == expected.resize((width, 49), method, box).tobytes()
    assert actual.tobytes() == raw


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"size": (0, 2)}, ValueError),
        ({"size": (-1, 2)}, ValueError),
        ({"size": (2.5, 2)}, TypeError),
        ({"size": (2,)}, TypeError),
        ({"resample": 6}, ValueError),
        ({"reducing_gap": 0.5}, ValueError),
        ({"box": (-1, 0, 2, 2)}, ValueError),
        ({"box": (0, 0, 4, 2)}, ValueError),
        ({"box": (2, 0, 1, 2)}, ValueError),
        ({"box": (0, 0, float("nan"), 2)}, ValueError),
        ({"box": (0, 0, 2)}, TypeError),
    ],
)
def test_resize_rejects_invalid_arguments(kwargs: dict, error: type[Exception]) -> None:
    image = Image.frombytes("L", (3, 2), bytes(6))
    with pytest.raises(error):
        image.resize(**({"size": (2, 2)} | kwargs))


def test_resize_closed_image() -> None:
    image = Image.frombytes("L", (3, 2), bytes(6))
    image.close()
    with pytest.raises(ValueError):
        image.resize((3, 2))


@pytest.mark.parametrize("mode", ["LA", "RGBA"])
@pytest.mark.parametrize(("size", "target"), [((2, 1024), (1, 5)), ((1024, 2), (5, 1)), ((67, 49), (1, 1))])
def test_alpha_resize_reduction_extreme_shapes(mode: str, size: tuple[int, int], target: tuple[int, int]) -> None:
    raw = bytes((i * 37 + i // 7 * 19) % 256 for i in range(size[0] * size[1] * len(mode)))
    actual = Image.frombytes(mode, size, raw)
    expected = PillowImage.frombytes(mode, size, raw)
    assert actual.resize(target, Image.Resampling.LANCZOS, reducing_gap=2.0).tobytes() == expected.resize(target, PillowImage.Resampling.LANCZOS, reducing_gap=2.0).tobytes()
