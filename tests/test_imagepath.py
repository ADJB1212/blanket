from __future__ import annotations

import struct
from array import array

import pytest
from PIL import ImagePath as PillowPath

from blanket import Image, ImageDraw, ImagePath


@pytest.mark.parametrize("coordinates", [[], 3, [1, 2, 3.5, -4], [(1, 2), (3.5, -4)], [[1, 2], [3.5, -4]], [(1, 2), 3.5, -4], array("f", [1, 2, 3.5, -4]), struct.pack("=ffff", 1, 2, 3.5, -4)])
def test_path_construction_and_sequences(coordinates: object) -> None:
    actual, expected = ImagePath.Path(coordinates), PillowPath.Path(coordinates)
    assert len(actual) == len(expected)
    assert list(actual) == list(expected)
    assert actual.tolist() == expected.tolist()
    assert actual.tolist(True) == expected.tolist(True)
    assert actual.getbbox() == expected.getbbox()
    assert actual[1:].tolist() == expected[1:].tolist()
    if actual:
        assert actual[-1] == expected[-1]
        actual[-1] = expected[-1] = (8, 9)
        assert actual.tolist() == expected.tolist()
    copy = ImagePath.Path(actual)
    if copy:
        copy[0] = (99, 99)
        assert actual[0] != copy[0]


@pytest.mark.parametrize("distance", [-1, 0, 1, 2, 2.5, 100])
def test_compact_matches_pillow(distance: float) -> None:
    points = [(0, 0), (0, 0), (0.5, 0.5), (1, 1), (2, 1), (4, 4)]
    actual, expected = ImagePath.Path(points), PillowPath.Path(points)
    assert actual.compact(distance) == expected.compact(distance)
    assert actual.tolist() == expected.tolist()
    assert ImagePath.Path([]).compact(distance) == 0


def test_large_path_compaction_preserves_vertices_and_bounds() -> None:
    points = [(float(i // 3), float(-(i // 3))) for i in range(30000)]
    path = ImagePath.Path(points)
    vertices = path.tolist()
    assert path.getbbox() == (0, -9999, 9999, 0)
    assert path.compact(1) == 20000
    assert path.tolist() == points[::3]
    assert all(actual is original for actual, original in zip(path, vertices[::3], strict=True))
    assert path.getbbox() == (0, -9999, 9999, 0)


@pytest.mark.parametrize("first_nan", [False, True])
def test_bounds_preserve_nonfinite_comparison_order(first_nan: bool) -> None:
    import math

    points = [(float("nan"), float("nan")), (-float("inf"), -3), (4, float("inf"))]
    if not first_nan:
        points.reverse()
    path = ImagePath.Path(points)
    xs, ys = zip(*points, strict=True)
    expected = min(xs), min(ys), max(xs), max(ys)
    for actual, value in zip(path.getbbox(), expected, strict=True):
        assert math.isnan(actual) if math.isnan(value) else actual == value


@pytest.mark.parametrize("key", [slice(None), slice(1, 3), slice(-2, None), slice(3, 1), slice(0, 0, 2)])
def test_slice_mutation_is_independent(key: slice) -> None:
    path = ImagePath.Path([(1, 2), (3, 4), (5, 6)])
    result = path[key]
    original = result.tolist()
    path[1] = (30, 40)
    assert result.tolist() == original
    result.map(lambda x, y: (x + 10, y + 20))
    assert path.tolist() == [(1, 2), (30, 40), (5, 6)]
    assert result.tolist() == [(x + 10, y + 20) for x, y in original]


@pytest.mark.parametrize("matrix", [(2, 0, 3, 0, -2, 5), (0.5, 2, -1, -2, 0.25, 4)])
@pytest.mark.parametrize("wrap", [0, 3])
def test_map_and_transform_match_pillow(matrix: tuple[float, ...], wrap: float) -> None:
    actual, expected = ImagePath.Path([(-1, 2), (3, 4)]), PillowPath.Path([(-1, 2), (3, 4)])
    for path in (actual, expected):
        assert path.map(lambda x, y: (x + 2, y * 3)) is None
        assert path.transform(matrix, wrap) is None
    assert actual.tolist() == expected.tolist()


def test_path_mutation_failures() -> None:
    path = ImagePath.Path([(1, 2), (3, 4)])
    with pytest.raises(ValueError, match="during mapping"):
        path.map(lambda x, y: (path.compact(), y))
    assert path.compact(0) == 0
    with pytest.raises(RuntimeError):
        path.map(lambda x, y: (_ for _ in ()).throw(RuntimeError()))
    assert path.compact(0) == 0
    with pytest.raises(TypeError):
        path[::2]
    with pytest.raises(TypeError):
        path[:] = [(1, 2)]
    with pytest.raises(TypeError):
        del path[0]
    with pytest.raises(IndexError):
        path[2] = (0, 0)
    with pytest.raises(ValueError):
        path.transform((1, 2, 3))


@pytest.mark.parametrize("distance", [float("nan"), float("inf"), -float("inf")])
def test_native_compaction_preserves_nonfinite_selection_and_identity(distance: float) -> None:
    path = ImagePath.Path([(0, 0), (1, 2), (float("inf"), 3), (float("nan"), 4), (-5, 6)])
    original = path.tolist()
    expected = original[:1]
    for point in original[1:]:
        if abs(point[0] - expected[-1][0]) + abs(point[1] - expected[-1][1]) >= distance:
            expected.append(point)
    assert path.compact(distance) == len(original) - len(expected)
    assert len(path) == len(expected)
    assert all(a is b for a, b in zip(path, expected, strict=True))


@pytest.mark.parametrize("wrap", [0.0, -0.0, -3.0, float("inf"), float("nan")])
def test_native_path_transform_preserves_wrapping(wrap: float) -> None:
    from math import fmod, isnan

    path = ImagePath.Path([(-0.0, 2), (-7.25, 3), (8.5, 4), (float("nan"), 5)])
    expected = []
    for x, y in path:
        tx, ty = 1 * x + 0 * y + 0, 0 * x + 1 * y + 0
        expected.append((fmod(tx, wrap) if wrap else tx, ty))
    path.transform((1, 0, 0, 0, 1, 0), wrap)
    for point, reference in zip(path, expected, strict=True):
        for actual, value in zip(point, reference, strict=True):
            assert isnan(actual) if isnan(value) else actual == value


def test_native_path_transform_preserves_partial_updates_on_domain_error() -> None:
    path = ImagePath.Path([(1, 2), (float("inf"), 3), (4, 5)])
    with pytest.raises(ValueError, match="math domain error"):
        path.transform((2, 0, 1, 0, 1, 0), wrap=3)
    assert path[0] == (0, 2)
    assert path[1] == (float("inf"), 3)
    assert path[2] == (4, 5)


@pytest.mark.parametrize("coordinates", [[1], [(1, 2, 3)], ["1", "2"], None])
def test_invalid_coordinates(coordinates: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        ImagePath.Path(coordinates)


@pytest.mark.parametrize("method", ["point", "line", "polygon", "rectangle", "ellipse", "rounded_rectangle"])
@pytest.mark.parametrize("mode", ["1", "L", "LA", "RGB", "RGBA", "P", "I", "I;16", "F", "CMYK", "YCbCr", "LAB", "HSV"])
def test_draw_accepts_paths(method: str, mode: str) -> None:
    points = [(1.5, 2.5), (6.5, 7.5)]
    path = ImagePath.Path(points)
    path.transform((1, 0, 1, 0, 1, 0))
    actual, expected = Image.new(mode, (10, 10)), Image.new(mode, (10, 10))
    getattr(ImageDraw.Draw(actual), method)(path, fill=1)
    getattr(ImageDraw.Draw(expected), method)(path.tolist(), fill=1)
    assert actual.tobytes() == expected.tobytes()
