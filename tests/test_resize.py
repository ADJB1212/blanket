from __future__ import annotations

import pytest
from blanket import Image
from PIL import Image as PillowImage


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
@pytest.mark.parametrize("method", [None, *Image.Resampling])
@pytest.mark.parametrize("size,box", [((11, 7), None), ((101, 83), None), ((9, 8), (2.3, 1.7, 59.2, 43.8)), ((67, 49), None)])
@pytest.mark.parametrize("gap", [None, 1.0, 2.0, 3.0])
def test_resize_matches_pillow(mode: str, method: int | None, size: tuple[int, int], box: tuple[float, ...] | None, gap: float | None) -> None:
    raw = bytes((i * 37 + i // 7 * 19) % 256 for i in range(67 * 49 * len(mode)))
    image = Image.frombytes(mode, (67, 49), raw)
    expected = PillowImage.frombytes(mode, image.size, raw).resize(size, method, box, gap)
    actual = image.resize(size, method, box, gap)
    assert actual.size == expected.size
    assert actual.mode == expected.mode
    assert actual.tobytes() == expected.tobytes()
    assert image.tobytes() == raw


@pytest.mark.parametrize("size", [(3, 2), (6, 4)])
def test_resize_copies_metadata_and_owns_pixels(size: tuple[int, int]) -> None:
    image = Image.frombytes("RGBA", (3, 2), bytes(range(24)))
    image.info["custom"] = "retained"
    result = image.resize(list(size))
    assert result is not image
    assert result.info == image.info
    assert result.info is not image.info
    assert result.format is None
    image.close()
    assert len(result.tobytes()) == size[0] * size[1] * 4


@pytest.mark.parametrize(
    "kwargs,error",
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
