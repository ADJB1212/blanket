"""Helpers for tests copied from Pillow 13.0.0; see PILLOW_TESTS.md."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import pytest

from blanket import Image, ImageMath, features


def upload(a: Image.Image, b: Image.Image) -> None:
    return None


PILLOW_IMAGES = Path(__file__).resolve().parent / "pillow_images"
logger = logging.getLogger(__name__)


def assert_image_equal(a: Image.Image, b: Image.Image, msg: str | None = None) -> None:
    assert a.mode == b.mode, msg or f"got mode {a.mode!r}, expected {b.mode!r}"
    assert a.size == b.size, msg or f"got size {a.size!r}, expected {b.size!r}"
    if a.tobytes() != b.tobytes():
        pytest.fail(msg or "got different content")


def hopper(mode: str | None = None) -> Image.Image:
    if mode is None:
        return Image.open(PILLOW_IMAGES / "hopper.ppm")
    return _cached_hopper(mode).copy()


@lru_cache
def _cached_hopper(mode: str) -> Image.Image:
    im = hopper("L") if mode == "F" else hopper()
    try:
        im = im.convert(mode)
    except ImportError:
        if mode == "LAB":
            im = Image.open(PILLOW_IMAGES / "hopper.Lab.tif")
        else:
            raise
    return im


def convert_to_comparable(a: Image.Image, b: Image.Image) -> tuple[Image.Image, Image.Image]:
    new_a, new_b = a, b
    if a.mode == "P":
        new_a = Image.new("L", a.size)
        new_b = Image.new("L", b.size)
        new_a.putdata(a.get_flattened_data())
        new_b.putdata(b.get_flattened_data())
    elif a.mode == "I;16":
        new_a = a.convert("I")
        new_b = b.convert("I")
    return new_a, new_b


def assert_image(im: Image.Image, mode: str, size: tuple[int, int], msg: str | None = None) -> None:
    if mode is not None:
        assert im.mode == mode, msg or f"got mode {im.mode!r}, expected {mode!r}"

    if size is not None:
        assert im.size == size, msg or f"got size {im.size!r}, expected {size!r}"


def assert_image_equal_tofile(a: Image.Image, filename: str | Path, msg: str | None = None, mode: str | None = None) -> None:
    with Image.open(filename) as im:
        converted_im = im.convert(mode) if mode else im
        assert_image_equal(a, converted_im, msg)


def assert_image_similar(a: Image.Image, b: Image.Image, epsilon: float, msg: str | None = None) -> None:
    assert a.mode == b.mode, msg or f"got mode {a.mode!r}, expected {b.mode!r}"
    assert a.size == b.size, msg or f"got size {a.size!r}, expected {b.size!r}"

    a, b = convert_to_comparable(a, b)

    diff = 0
    for ach, bch in zip(a.split(), b.split(), strict=True):
        chdiff = ImageMath.lambda_eval(lambda args: abs(args["a"] - args["b"]), a=ach, b=bch).convert("L")
        diff += sum(i * num for i, num in enumerate(chdiff.histogram()))

    ave_diff = diff / (a.size[0] * a.size[1])
    try:
        assert epsilon >= ave_diff, (msg or "") + f" average pixel value difference {ave_diff:.4f} > epsilon {epsilon:.4f}"
    except Exception:
        try:
            url = upload(a, b)
            if url:
                logger.exception("URL for test images: %s", url)
        except Exception:
            pass
        raise


def assert_image_similar_tofile(a: Image.Image, filename: str | Path, epsilon: float, msg: str | None = None) -> None:
    with Image.open(filename) as img:
        assert_image_similar(a, img, epsilon, msg)


def skip_unless_feature(feature: str) -> pytest.MarkDecorator:
    reason = f"{feature} not available"
    return pytest.mark.skipif(not features.check(feature), reason=reason)
