"""Phase-correlation geometry, confidence and input contracts."""

from dataclasses import FrozenInstanceError
from io import BytesIO
from math import cos, radians, sin

import numpy as np
import pytest

from blanket import Image, ImageChops, ImageFilter, ImageOps


def texture(size=(63, 47), seed=17):
    data = np.random.default_rng(seed).integers(0, 256, (size[1], size[0]), dtype=np.uint8)
    return Image.frombytes("L", size, data.tobytes())


@pytest.mark.parametrize("size", [(63, 47), (64, 48), (17, 19), (3, 5)])
@pytest.mark.parametrize("offset", [(0, 0), (4, -3), (-5, 2)])
def test_cyclic_translation_direction_and_confidence(size, offset):
    reference = texture(size)
    image = ImageChops.offset(reference, *offset)
    result = ImageOps.phase_correlate(reference, image, window=None)
    expected = [(-shift + length / 2) % length - length / 2 for shift, length in zip(offset, size, strict=True)]
    assert (result.dx, result.dy) == pytest.approx(expected, abs=1e-8)
    assert result.rotation == 0
    assert result.confidence > 0.9


def test_half_size_displacement_is_negative():
    reference = texture((64, 48))
    result = ImageOps.phase_correlate(reference, ImageChops.offset(reference, 32, 24), window=None)
    assert (result.dx, result.dy) == pytest.approx((-32, -24), abs=1e-8)


@pytest.mark.parametrize("mode", ["L", "LA", "RGB", "RGBA", "I;16", "I;16L", "I;16B", "I;16N", "F", "RGB;F", "RGBA;F"])
def test_modes_and_source_preservation(mode):
    reference = texture()
    if mode in ("RGB;F", "RGBA;F"):
        reference = reference.convert(mode.split(";")[0])
    reference = reference.convert(mode)
    image = reference.copy()
    image.info["test"] = b"preserved"
    before = image.tobytes()
    result = ImageOps.phase_correlate(reference, image)
    assert (result.dx, result.dy) == pytest.approx((0, 0), abs=1e-8)
    assert result.confidence > 0.9
    assert image.tobytes() == before
    assert image.info["test"] == b"preserved"
    with pytest.raises(FrozenInstanceError):
        result.dx = 10


@pytest.mark.parametrize("depth", [10, 12, 16])
@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
def test_depths_and_mixed_reference_modes(depth, mode):
    reference = texture()
    image = ImageChops.offset(reference, 4, -3).convert(mode, bit_depth=depth)
    result = ImageOps.phase_correlate(reference, image, window=None)
    assert (result.dx, result.dy) == pytest.approx((-4, 3), abs=0.01)
    assert result.confidence > 0.9


def test_float_brightness_and_positive_contrast_invariance():
    reference = texture().convert("F")
    shifted = np.asarray(reference).copy()
    shifted = np.roll(shifted, (-2, 3), axis=(0, 1)) * 1.75 + 200
    image = Image.frombytes("F", reference.size, shifted.astype("<f4").tobytes())
    result = ImageOps.phase_correlate(reference, image, window=None)
    assert (result.dx, result.dy) == pytest.approx((-3, 2), abs=1e-6)
    assert result.confidence > 0.9


def test_fractional_translation_is_approximately_refined():
    data = np.random.default_rng(9).random((63, 65))
    dx, dy = 2.3, -1.4
    phase = np.exp(-2j * np.pi * (np.fft.fftfreq(65)[None, :] * dx + np.fft.fftfreq(63)[:, None] * dy))
    shifted = np.fft.ifft2(np.fft.fft2(data) * phase).real
    reference = Image.frombytes("F", (65, 63), data.astype("<f4").tobytes())
    image = Image.frombytes("F", (65, 63), shifted.astype("<f4").tobytes())
    result = ImageOps.phase_correlate(reference, image, window=None)
    assert (result.dx, result.dy) == pytest.approx((-dx, -dy), abs=0.18)
    assert result.confidence > 0.5


def test_window_handles_zero_filled_translation():
    reference = texture((97, 83)).filter(ImageFilter.GaussianBlur(1))
    image = Image.new("L", reference.size, 128)
    image.paste(reference.crop((0, 3, 92, 83)), (5, 0))
    result = ImageOps.phase_correlate(reference, image)
    assert (result.dx, result.dy) == pytest.approx((-5, 3), abs=0.2)
    assert result.confidence > 0.2


@pytest.mark.parametrize("angle", [-4, 3, 2.5])
def test_rotation_search_and_transform_order(angle):
    reference = Image.new("L", (96, 96), 128)
    reference.paste(texture((64, 64)), (16, 16))
    image = ImageChops.offset(reference, 4, -3).rotate(angle, Image.Resampling.BILINEAR, fillcolor=128)
    result = ImageOps.phase_correlate(reference, image, max_rotation=abs(angle), rotation_step=1)
    assert result.rotation == pytest.approx(-angle, abs=0.01)
    assert (result.dx, result.dy) == pytest.approx((-4, 3), abs=0.4)
    assert result.confidence > 0.1


def test_rotation_then_translation_coordinates():
    reference = Image.new("L", (96, 96), 128)
    reference.paste(texture((64, 64)), (16, 16))
    image = ImageChops.offset(reference.rotate(4, Image.Resampling.BILINEAR, fillcolor=128), 5, -3)
    result = ImageOps.phase_correlate(reference, image, max_rotation=5)
    angle = radians(-4)
    expected = (-cos(angle) * 5 - sin(angle) * -3, sin(angle) * 5 - cos(angle) * -3)
    assert result.rotation == -4
    assert (result.dx, result.dy) == pytest.approx(expected, abs=0.4)


def test_flat_and_unrelated_confidence():
    reference = texture((64, 64))
    for flat in [Image.new("L", reference.size, 0), Image.new("L", reference.size, 128)]:
        assert ImageOps.phase_correlate(reference, flat, max_rotation=3) == ImageOps.Alignment(0, 0, 0, 0)
        assert ImageOps.phase_correlate(flat, reference).confidence == 0
    assert ImageOps.phase_correlate(reference, texture((64, 64), seed=123)).confidence < 0.15


def test_alpha_is_ignored():
    reference = texture().convert("RGBA")
    image = reference.copy()
    image.putalpha(0)
    assert ImageOps.phase_correlate(reference, image).confidence > 0.9


@pytest.mark.parametrize("mode", ["1", "P", "PA", "I", "La", "RGBa", "RGBX", "CMYK", "YCbCr", "LAB", "HSV"])
def test_unsupported_modes(mode):
    bad = Image.new(mode, (16, 16))
    good = texture((16, 16))
    with pytest.raises(ValueError, match="convert explicitly"):
        ImageOps.phase_correlate(bad, good)
    with pytest.raises(ValueError, match="convert explicitly"):
        ImageOps.phase_correlate(good, bad)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_color_is_rejected(value):
    bad = Image.new("F", (16, 16), value)
    good = texture((16, 16))
    with pytest.raises(ValueError, match="finite color"):
        ImageOps.phase_correlate(good, bad)
    with pytest.raises(ValueError, match="finite color"):
        ImageOps.phase_correlate(bad, good)


@pytest.mark.parametrize("options", [{"max_rotation": -1}, {"max_rotation": 16}, {"max_rotation": float("nan")}, {"rotation_step": 0}, {"rotation_step": float("inf")}, {"max_rotation": 15, "rotation_step": 0.01}, {"window": "invalid"}])
def test_invalid_options(options):
    image = texture()
    with pytest.raises(ValueError):
        ImageOps.phase_correlate(image, image, **options)


def test_invalid_dimensions_and_types():
    image = texture()
    for size in [(0, 0), (2, 10), (10, 2), (10, 10), (2049, 2049)]:
        bad = Image.new("L", size)
        with pytest.raises(ValueError, match="equal dimensions"):
            ImageOps.phase_correlate(image, bad)
        if size != (10, 10):
            with pytest.raises(ValueError, match="equal dimensions"):
                ImageOps.phase_correlate(bad, bad)
    with pytest.raises(TypeError, match="Blanket Images"):
        ImageOps.phase_correlate(image, None)


@pytest.mark.parametrize("lazy", [False, True])
def test_sequence_positions_and_close(lazy):
    first = texture()
    second = ImageChops.offset(first, 3, -2)
    stream = BytesIO()
    first.save(stream, "PNG", save_all=True, append_images=[second], duration=20)
    stream.seek(0)
    with Image.open(stream, lazy=lazy) as image:
        image.seek(1)
        before = image.tobytes()
        result = ImageOps.phase_correlate(first, image, window=None)
        assert (result.dx, result.dy) == pytest.approx((-3, 2), abs=1e-8)
        assert image.tell() == 1
        assert image.tobytes() == before
    image.close()
    with pytest.raises(ValueError, match="closed image"):
        ImageOps.phase_correlate(first, image)
