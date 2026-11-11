from __future__ import annotations

from io import BytesIO

import numpy as np
import pytest
from helpers import image_buffer

from blanket import Image, ImageMetrics
from blanket._blanket import image_metric

METRICS = [ImageMetrics.ssim, ImageMetrics.ms_ssim, ImageMetrics.ssimulacra2]


def reference_statistics(a, b):
    axis = np.arange(-5, 6, dtype=float)
    kernel = np.exp(-(axis[:, None] ** 2 + axis[None, :] ** 2) / 4.5)
    kernel /= kernel.sum()

    def smooth(values):
        windows = np.lib.stride_tricks.sliding_window_view(values, (11, 11), axis=(0, 1))
        return np.einsum("ijcxy,xy->ijc", windows, kernel)

    ma, mb = smooth(a), smooth(b)
    va = np.maximum(0, smooth(a * a) - ma * ma)
    vb = np.maximum(0, smooth(b * b) - mb * mb)
    covariance = smooth(a * b) - ma * mb
    cs = (2 * covariance + 0.0009) / (va + vb + 0.0009)
    ssim = (2 * ma * mb + 0.0001) / (ma * ma + mb * mb + 0.0001) * cs
    return ssim.mean(axis=(0, 1)), cs.mean(axis=(0, 1))


def reference_multiscale(a, b):
    scores = np.ones(3)
    weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    for level, weight in enumerate(weights):
        ssim, cs = reference_statistics(a, b)
        scores *= np.maximum(0, ssim if level == 4 else cs) ** weight
        if level < 4:
            images = []
            for pixels in (a, b):
                h, w, _ = pixels.shape
                padded = np.pad(pixels, ((0, h % 2), (0, w % 2), (0, 0)), mode="edge")
                images.append(padded.reshape((h + 1) // 2, 2, (w + 1) // 2, 2, 3).mean(axis=(1, 3)))
            a, b = images
    return scores.mean()


@pytest.mark.parametrize("shape", [(11, 11, 3), (13, 19, 3), (18, 24, 3)])
def test_ssim_matches_independent_gaussian_reference(shape):
    rng = np.random.default_rng(42)
    a = rng.integers(0, 256, shape, dtype=np.uint8)
    b = np.clip(a.astype(int) + rng.integers(-40, 40, shape), 0, 255).astype(np.uint8)
    expected, _ = reference_statistics(a.astype(float) / 255, b.astype(float) / 255)
    assert ImageMetrics.ssim(Image.fromarray(a), Image.fromarray(b)) == pytest.approx(expected.mean(), abs=1e-12)


@pytest.mark.parametrize("shape", [(161, 161, 3), (162, 180, 3), (177, 193, 3)])
def test_ms_ssim_matches_independent_reference(shape):
    rng = np.random.default_rng(17)
    a = rng.integers(0, 256, shape, dtype=np.uint8)
    b = np.clip(a.astype(int) + rng.integers(-75, 75, shape), 0, 255).astype(np.uint8)
    expected = reference_multiscale(a.astype(float) / 255, b.astype(float) / 255)
    assert ImageMetrics.ms_ssim(Image.fromarray(a), Image.fromarray(b)) == pytest.approx(expected, abs=1e-11)


@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize("mode", ["L", "LA", "RGB", "RGBA", "I;16", "I;16L", "I;16B", "I;16N", "F", "RGB;F", "RGBA;F"])
def test_identical_images_and_mixed_grayscale_modes(metric, mode):
    value = 1.0 if mode in ("F", "RGB;F", "RGBA;F") else 65535 if mode.startswith("I;16") else 255
    channels = Image.getmodebands(mode)
    source = Image.new(mode, (161, 163), value if channels == 1 else (value,) * channels)
    expected = 100 if metric is ImageMetrics.ssimulacra2 else 1
    assert metric(source, source.copy()) == pytest.approx(expected, abs=1e-9)
    assert metric(source, Image.new("RGB", source.size, "white")) == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
@pytest.mark.parametrize("depth", [10, 12, 16])
def test_sample_depth_preserved(metric, mode, depth):
    maximum = (1 << depth) - 1
    shape = (161, 163) if mode == "L" else (161, 163, len(mode))
    rng = np.random.default_rng(9)
    a = rng.integers(0, maximum + 1, shape, dtype=np.uint16)
    b = np.clip(a.astype(int) + rng.integers(-maximum // 10, maximum // 10, shape), 0, maximum).astype(np.uint16)
    source, distorted = Image.fromarray(a, bit_depth=depth), Image.fromarray(b, bit_depth=depth)
    fa, fb = Image.fromarray((a / maximum).astype(np.float32)), Image.fromarray((b / maximum).astype(np.float32))
    assert metric(source, distorted) == pytest.approx(metric(fa, fb), abs=2e-4)


@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize("mode,dtype", [("I;16", "<u2"), ("I;16L", "<u2"), ("I;16B", ">u2"), ("I;16N", "=u2")])
def test_unsigned_scalar_endianness(metric, mode, dtype):
    a = (np.arange(161 * 163).reshape(163, 161) * 7919 % 65536).astype(dtype)
    b = np.maximum(a.astype(int) - 219, 0).astype(dtype)
    source = Image.frombytes(mode, (161, 163), a.tobytes())
    distorted = Image.frombytes(mode, source.size, b.tobytes())
    fa, fb = Image.fromarray((a / 65535).astype(np.float32)), Image.fromarray((b / 65535).astype(np.float32))
    assert metric(source, distorted) == pytest.approx(metric(fa, fb), abs=2e-4)


@pytest.mark.parametrize("metric", METRICS)
def test_alpha_is_ignored_and_metadata_is_unchanged(metric):
    a = Image.new("RGBA", (161, 163), (90, 100, 110, 0))
    b = Image.new("RGBA", a.size, (90, 100, 110, 255))
    a.info.update({"icc_profile": b"ignored", "cicp": (9, 16, 0, 1), "orientation": 6})
    before = a.tobytes(), a.info.copy()
    assert metric(a, b) == pytest.approx(100 if metric is ImageMetrics.ssimulacra2 else 1)
    assert (a.tobytes(), a.info) == before
    b.putpixel((80, 80), (255, 0, 0, 0))
    assert metric(a, b) < metric(a, a)


@pytest.mark.parametrize("metric", METRICS)
def test_increasing_distortion_lowers_score(metric):
    rng = np.random.default_rng(23)
    a = rng.integers(64, 192, (161, 163, 3), dtype=np.uint8)
    source = Image.fromarray(a)
    mild = Image.fromarray(a + 5)
    strong = Image.fromarray(a + 50)
    assert metric(source, source) > metric(source, mild) > metric(source, strong)


def test_ssimulacra2_matches_libjxl_reference_and_argument_order():
    # Golden scores from libjxl's ssimulacra2 CLI on PNGs of these exact pixels.
    pixels = bytes((i * 37 + i // 7) % 256 for i in range(129 * 127 * 3))
    original = Image.frombytes("RGB", (129, 127), pixels)
    distorted = Image.frombytes("RGB", original.size, bytes(min(255, v + 17) for v in pixels))
    assert ImageMetrics.ssimulacra2(original, distorted) == pytest.approx(63.81938432, abs=0.1)
    assert ImageMetrics.ssimulacra2(distorted, original) == pytest.approx(66.78730933, abs=0.1)


def test_negative_ssim_and_ms_ssim_clamping():
    a = ((np.indices((161, 163)).sum(axis=0) % 2) * 255).astype(np.uint8)
    source, inverse = Image.fromarray(a), Image.fromarray(255 - a)
    assert ImageMetrics.ssim(source, inverse) < -0.9
    assert ImageMetrics.ms_ssim(source, inverse) == 0


@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize("lazy", [False, True])
def test_current_frame_and_closed_images(metric, lazy):
    frames = [Image.new("RGB", (161, 163), value) for value in ((30, 40, 50), (100, 110, 120))]
    stream = image_buffer(frames[0], "PNG", save_all=True, append_images=frames[1:])
    source = Image.open(stream, lazy=lazy)
    source.seek(1)
    before = source.tobytes(), source.info.copy()
    assert metric(source, frames[1]) == pytest.approx(100 if metric is ImageMetrics.ssimulacra2 else 1)
    assert source.tell() == 1
    assert (source.tobytes(), source.info) == before
    source.close()
    with pytest.raises(ValueError, match="closed"):
        metric(source, frames[1])


@pytest.mark.parametrize("metric", METRICS)
def test_invalid_types_dimensions_and_modes(metric):
    source = Image.new("RGB", (161, 161))
    with pytest.raises(TypeError, match="Blanket"):
        metric(None, source)
    with pytest.raises(TypeError, match="Blanket"):
        metric(source, BytesIO())
    with pytest.raises(ValueError, match="dimensions"):
        metric(source, Image.new("RGB", (161, 162)))
    for mode in ("1", "P", "PA", "I", "La", "RGBa", "CMYK", "LAB", "HSV", "YCbCr", "RGBX"):
        with pytest.raises(ValueError, match="convert explicitly"):
            metric(source.convert(mode), source)


@pytest.mark.parametrize("metric,minimum", [(ImageMetrics.ssim, 11), (ImageMetrics.ms_ssim, 161), (ImageMetrics.ssimulacra2, 8)])
def test_minimum_sizes(metric, minimum):
    for size in ((0, 0), (minimum - 1, minimum), (minimum, minimum - 1)):
        image = Image.new("RGB", size)
        with pytest.raises(ValueError, match="at least"):
            metric(image, image)
    image = Image.new("RGB", (minimum, minimum))
    assert metric(image, image) == pytest.approx(100 if metric is ImageMetrics.ssimulacra2 else 1)


@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), -0.1, 1.1])
def test_invalid_float_color_samples(metric, value):
    source = Image.new("F", (161, 161), 0.5)
    source.putpixel((80, 80), value)
    with pytest.raises(ValueError, match=r"finite.*\[0, 1\]"):
        metric(source, source)


def test_native_entry_rejects_unknown_algorithm():
    source = Image.new("RGB", (11, 11))
    with pytest.raises(ValueError, match="unknown"):
        image_metric(source._native, source._native, "unknown")
