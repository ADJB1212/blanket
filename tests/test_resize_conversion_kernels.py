from __future__ import annotations

import random
import struct

import pytest
from helpers import assert_image_properties, assert_pixels_equal, image_pair
from PIL import Image as PillowImage

from blanket import Image


@pytest.mark.parametrize("mode", ["LA", "RGBA"])
@pytest.mark.parametrize("width", [2, 32, 34, 512, 1026])
@pytest.mark.parametrize("height", [2, 34])
def test_alpha_box_half_blocks_and_tails(mode: str, width: int, height: int) -> None:
    raw = random.Random(791).randbytes(width * height * len(mode))
    actual, expected = image_pair(mode, (width, height), raw)
    target = (width // 2, height // 2)
    assert actual.resize(target, Image.Resampling.BOX).tobytes() == expected.resize(target, PillowImage.Resampling.BOX).tobytes()
    assert actual.tobytes() == raw


@pytest.mark.parametrize("source", ["L", "RGB", "RGBA"])
def test_conversion_depth_and_mode_rounding(source: str, subtests: pytest.Subtests) -> None:
    for target in ["L", "RGB", "RGBA"]:
        for source_depth in [8, 10, 12, 16]:
            for target_depth in [8, 10, 12, 16]:
                with subtests.test(source=source, target=target, source_depth=source_depth, target_depth=target_depth):
                    maximum = (1 << source_depth) - 1
                    target_max = (1 << target_depth) - 1
                    rng = random.Random(73)
                    samples = [rng.randrange(maximum + 1) for _ in range(33 * len(source))]
                    samples[: len(source)] = [maximum] * len(source)
                    samples[len(source) : 2 * len(source)] = [0] * len(source)
                    raw = bytes(samples) if source_depth == 8 else struct.pack(f"<{len(samples)}H", *samples)
                    image = Image.frombytes(source, (33, 1), raw, bit_depth=source_depth)
                    expected = []
                    for i in range(0, len(samples), len(source)):
                        pixel = samples[i : i + len(source)]
                        if target == source:
                            values = pixel
                        elif target == "L":
                            values = [(pixel[0] * 19595 + pixel[1] * 38470 + pixel[2] * 7471 + 32768) >> 16]
                        else:
                            values = pixel[:3] if source != "L" else pixel * 3
                            if target == "RGBA":
                                values.append(maximum)
                        expected.extend((value * target_max + maximum // 2) // maximum for value in values)
                    encoded = bytes(expected) if target_depth == 8 else struct.pack(f"<{len(expected)}H", *expected)
                    actual = image.convert(target, bit_depth=target_depth)
                    assert_image_properties(actual, mode=target, bit_depth=target_depth, size=(33, 1), pixels=encoded)
                    assert image.tobytes() == raw


@pytest.mark.parametrize("mode", ["L", "LA", "RGB", "RGBA", "CMYK"])
def test_filtered_resize_row_batches_and_fractional_boxes(mode: str, subtests: pytest.Subtests) -> None:
    for height in [1, 3, 4, 5, 15, 16, 17]:
        for method in [1, 2, 3, 4, 5]:
            with subtests.test(mode=mode, height=height, method=method):
                size = (39, height)
                raw = random.Random(83).randbytes(size[0] * size[1] * len(mode))
                actual, expected = image_pair(mode, size, raw)
                for target in [(17, height), (67, height), (13, 7)]:
                    for box in [None, (0.25, 0.0, 38.5, float(height))]:
                        assert actual.resize(target, method, box=box).tobytes() == expected.resize(target, method, box=box).tobytes()
                assert actual.tobytes() == raw


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
def test_nearest_gathers_and_repeated_rows(mode: str, subtests: pytest.Subtests) -> None:
    for size in [(37, 29), (38, 26), (803, 607)]:
        for target in [(19, 13), (91, 79), (7, 5)]:
            with subtests.test(mode=mode, size=size, target=target):
                raw = random.Random(42).randbytes(size[0] * size[1] * len(mode))
                image, reference = image_pair(mode, size, raw)
                for box in (None, (1.25, 2.5, size[0] - 0.75, size[1] - 1.5)):
                    actual = image.resize(target, Image.Resampling.NEAREST, box=box)
                    expected = reference.resize(target, PillowImage.Resampling.NEAREST, box=box)
                    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("mode", ["L", "LA", "RGB", "RGBA", "CMYK"])
def test_filtered_half_resize_vector_blocks_and_edges(mode: str, subtests: pytest.Subtests) -> None:
    for width in [64, 66, 96, 98, 128, 130, 1026]:
        for height in [1, 3, 17]:
            for method in [1, 2, 3, 4, 5]:
                with subtests.test(mode=mode, width=width, height=height, method=method):
                    raw = random.Random(93).randbytes(width * height * len(mode))
                    image, reference = image_pair(mode, (width, height), raw)
                    for target in [(width // 2, height), (width // 2, max(1, height // 2))]:
                        for box in [None, (0.25, 0.0, width - 0.5, float(height))]:
                            assert image.resize(target, method, box=box).tobytes() == reference.resize(target, method, box=box).tobytes()
                    assert image.tobytes() == raw


@pytest.mark.parametrize("mode", ["RGB", "HSV", "YCbCr", "LAB"])
@pytest.mark.parametrize("size", [(1, 1), (16, 16), (17, 19)])
def test_nearest_double_rgb_layout(mode: str, size: tuple[int, int]) -> None:
    raw = random.Random(42).randbytes(size[0] * size[1] * 3)
    actual, expected = image_pair(mode, size, raw)
    target = (size[0] * 2, size[1] * 2)
    assert actual.resize(target, Image.Resampling.NEAREST).tobytes() == expected.resize(target, PillowImage.Resampling.NEAREST).tobytes()


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
@pytest.mark.parametrize("target", [(101, 77), (37, 29), (13, 11)])
@pytest.mark.parametrize("size", [(803, 607)])
def test_parallel_thumbnail_reduction(mode: str, target: tuple[int, int], size: tuple[int, int]) -> None:
    raw = random.Random(81).randbytes(size[0] * size[1] * len(mode))
    image, reference = image_pair(mode, size, raw)
    options = {"resample": Image.Resampling.LANCZOS, "reducing_gap": 3.0}
    assert image.resize(target, **options).tobytes() == reference.resize(target, **options).tobytes()


@pytest.mark.parametrize("source", ["L", "RGB", "RGBA"])
def test_conversion_vector_tails_and_parallel_chunks(source: str, subtests: pytest.Subtests) -> None:
    for target in ["L", "RGB", "RGBA"]:
        for count in [15, 16, 17, 262147, 2800001]:
            with subtests.test(source=source, target=target, count=count):
                raw = random.Random(17).randbytes(count * len(source))
                image, reference = image_pair(source, (count, 1), raw)
                actual = image.convert(target)
                expected = reference.convert(target)
                assert_pixels_equal(actual, expected)
