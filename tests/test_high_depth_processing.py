"""Native processing retains unsigned sample precision and storage modes."""

from __future__ import annotations

import math
import struct
from collections import Counter
from io import BytesIO

import pytest
from PIL import Image as PillowImage, ImageFilter as PillowFilter

from blanket import Image, ImageEnhance, ImageFilter, ImageStat

FORMATS = [(mode, depth) for mode in ("L", "RGB", "RGBA") for depth in (10, 12, 16)] + [(mode, 16) for mode in ("I;16", "I;16L", "I;16B")]


def make_image(mode, depth, size=(7, 5), values=None):
    channels = {"RGB": 3, "RGBA": 4}.get(mode, 1)
    count = size[0] * size[1] * channels
    maximum = (1 << depth) - 1
    if values is None:
        values = [(i * 719 + i // 3 + 257) & maximum for i in range(count)]
    raw = struct.pack(f"{'>' if mode == 'I;16B' else '<'}{len(values)}H", *values)
    return Image.frombytes(mode, size, raw, bit_depth=depth if mode in ("L", "RGB", "RGBA") else 8)


def samples(image):
    raw = image.tobytes()
    return list(struct.unpack(f"{'>' if image.mode == 'I;16B' else '<'}{len(raw) // 2}H", raw))


@pytest.mark.parametrize(("mode", "depth"), FORMATS)
@pytest.mark.parametrize(("window", "rank"), [(1, 0), (3, 0), (3, 4), (3, 8), (5, 7), (9, 0), (9, 80)])
def test_rank_filters_use_samples_and_extend_edges(mode, depth, window, rank):
    image = make_image(mode, depth)
    image.info["test"] = "metadata"
    original = samples(image)
    channels = len(image.getbands())
    radius = window // 2
    expected = []
    for y in range(image.height):
        for x in range(image.width):
            for c in range(channels):
                neighborhood = [
                    original[(min(max(yy, 0), image.height - 1) * image.width + min(max(xx, 0), image.width - 1)) * channels + c]
                    for yy in range(y - radius, y + radius + 1)
                    for xx in range(x - radius, x + radius + 1)
                ]
                expected.append(sorted(neighborhood)[rank])
    result = image.filter(ImageFilter.RankFilter(window, rank))
    assert (result.mode, result.bit_depth, result.size, result.info) == (mode, depth, image.size, image.info)
    assert samples(result) == expected
    assert samples(image) == original


@pytest.mark.parametrize(("mode", "depth"), FORMATS)
def test_mode_filter_clips_edges_and_keeps_original_for_low_counts(mode, depth):
    channels = {"RGB": 3, "RGBA": 4}.get(mode, 1)
    values = [v for v in (301, 301, 700, 700, 700, 301, 901, 901, 901) for _ in range(channels)]
    image = make_image(mode, depth, (3, 3), values)
    expected = []
    for y in range(3):
        for x in range(3):
            for c in range(channels):
                counts = Counter(values[(yy * 3 + xx) * channels + c] for yy in range(max(0, y - 1), min(3, y + 2)) for xx in range(max(0, x - 1), min(3, x + 2)))
                count = max(counts.values())
                expected.append(min(v for v, n in counts.items() if n == count) if count > 2 else values[(y * 3 + x) * channels + c])
    assert samples(image.filter(ImageFilter.ModeFilter(3))) == expected


@pytest.mark.parametrize(("mode", "depth"), FORMATS)
@pytest.mark.parametrize("scale", [1, 4])
def test_kernel_uses_sample_offsets_saturation_and_unchanged_borders(mode, depth, scale):
    image = make_image(mode, depth)
    values = samples(image)
    channels = len(image.getbands())
    result = image.filter(ImageFilter.Kernel((3, 3), [0, 0, 0, 0, 2, 0, 0, 0, 0], scale=scale, offset=-17))
    expected = values.copy()
    for y in range(1, image.height - 1):
        for x in range(1, image.width - 1):
            for c in range(channels):
                i = (y * image.width + x) * channels + c
                expected[i] = min((1 << depth) - 1, max(0, int(values[i] * 2 / scale - 17 + 0.5)))
    assert samples(result) == expected
    flipped = image.filter(ImageFilter.Kernel((3, 3), [1, 0, 0, 0, 0, 0, 0, 0, 0], scale=1))
    assert flipped.getpixel((1, 1)) == image.getpixel((0, 2))


@pytest.mark.parametrize(("mode", "depth"), FORMATS)
def test_blur_and_unsharp_sample_reference(mode, depth):
    image = make_image(mode, depth)
    values = samples(image)
    channels = len(image.getbands())
    horizontal = values.copy()
    expected = values.copy()
    for y in range(image.height):
        for x in range(image.width):
            for c in range(channels):
                horizontal[(y * image.width + x) * channels + c] = (sum(values[(y * image.width + min(max(xx, 0), image.width - 1)) * channels + c] for xx in range(x - 1, x + 2)) + 1) // 3
    for y in range(image.height):
        for x in range(image.width):
            for c in range(channels):
                expected[(y * image.width + x) * channels + c] = (sum(horizontal[(min(max(yy, 0), image.height - 1) * image.width + x) * channels + c] for yy in range(y - 1, y + 2)) + 1) // 3
    assert samples(image.filter(ImageFilter.BoxBlur(1))) == expected
    blurred = samples(image.filter(ImageFilter.GaussianBlur((1.5, 1.5))))
    expected = [min((1 << depth) - 1, max(0, a + math.trunc((a - b) * 150 / 100))) if abs(a - b) > 17 else a for a, b in zip(values, blurred, strict=True)]
    assert samples(image.filter(ImageFilter.UnsharpMask(1.5, 150, 17))) == expected
    assert image.filter(ImageFilter.BoxBlur(0)).tobytes() == image.tobytes()


@pytest.mark.parametrize(("mode", "depth"), FORMATS)
@pytest.mark.parametrize(("radius", "percent", "threshold"), [(20, 0, -1), (20, 150, 65535), (0, 150, -1)])
def test_unsharp_identity_preserves_samples(mode, depth, radius, percent, threshold):
    image = make_image(mode, depth)
    image.info["test"] = "metadata"
    result = image.filter(ImageFilter.UnsharpMask(radius, percent, threshold))
    assert (result.mode, result.bit_depth, result.info) == (mode, depth, image.info)
    assert result.tobytes() == image.tobytes()
    result.close()
    assert samples(image)[0] == 257


@pytest.mark.parametrize(("mode", "depth"), FORMATS)
@pytest.mark.parametrize("name", ImageEnhance.__all__)
def test_enhancements_retain_precision_and_alpha(mode, depth, name):
    image = make_image(mode, depth)
    original = samples(image)
    channels = len(image.getbands())
    enhancer = getattr(ImageEnhance, name)(image)
    degenerate = samples(enhancer.degenerate)
    luma = [original[i] if channels == 1 else (original[i] * 19595 + original[i + 1] * 38470 + original[i + 2] * 7471 + 32768) >> 16 for i in range(0, len(original), channels)]
    mean = int(sum(luma) / len(luma) + 0.5)
    for i, value in enumerate(degenerate):
        if mode == "RGBA" and i % 4 == 3:
            assert value == original[i]
        elif name == "Brightness":
            assert value == 0
        elif name == "Color":
            assert value == luma[i // channels]
        elif name == "Contrast":
            assert value == mean
        elif name == "Sharpness":
            pixel, c = divmod(i, channels)
            y, x = divmod(pixel, image.width)
            if x in (0, image.width - 1) or y in (0, image.height - 1):
                assert value == original[i]
            else:
                total = sum(original[(yy * image.width + xx) * channels + c] for yy in range(y - 1, y + 2) for xx in range(x - 1, x + 2)) + 4 * original[i]
                assert value == (total + 6) // 13
    for factor in (0, 0.5, 1, 2, -1):
        result = enhancer.enhance(factor)
        assert (result.mode, result.bit_depth) == (mode, depth)
        assert samples(result) == [min((1 << depth) - 1, max(0, int(a + factor * (b - a)))) for a, b in zip(degenerate, original, strict=True)]


@pytest.mark.parametrize(("mode", "depth"), FORMATS)
@pytest.mark.parametrize("masked", [False, True])
def test_statistics_use_full_sample_values_and_mask(mode, depth, masked):
    image = make_image(mode, depth)
    mask = Image.frombytes("L", image.size, bytes(i % 3 for i in range(image.width * image.height))) if masked else None
    original = samples(image)
    channels = len(image.getbands())
    statistics = ImageStat.Stat(image, mask)
    for c in range(channels):
        values = [v for i, v in enumerate(original[c::channels]) if not masked or i % 3]
        counts = Counter(values)
        assert statistics.h[c * (1 << depth) : (c + 1) * (1 << depth)] == [counts[v] for v in range(1 << depth)]
        mean = sum(values) / len(values)
        assert statistics.count[c] == len(values)
        assert statistics.extrema[c] == (min(values), max(values))
        assert statistics.sum[c] == sum(values)
        assert statistics.sum2[c] == sum(v * v for v in values)
        assert statistics.mean[c] == mean
        assert statistics.median[c] == sorted(values)[len(values) // 2]
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        assert statistics.var[c] == pytest.approx(variance)
        assert statistics.stddev[c] == pytest.approx(math.sqrt(variance))
        assert statistics.rms[c] == pytest.approx(math.sqrt(sum(v * v for v in values) / len(values)))
    assert len(statistics.h) == (1 << depth) * channels
    image.close()
    assert statistics.count == [len(values)] * channels


@pytest.mark.parametrize("depth", [10, 12, 16])
@pytest.mark.parametrize("mask_depth", [8, None])
def test_composite_paste_blend_and_putalpha(depth, mask_depth):
    maximum = (1 << depth) - 1
    background = make_image("RGBA", depth, (3, 2))
    foreground = make_image("RGBA", depth, (3, 2), list(reversed(samples(background))))
    mask_depth = depth if mask_depth is None else mask_depth
    mask_maximum = (1 << mask_depth) - 1
    mask_values = [0, 1, mask_maximum // 2, mask_maximum - 1, mask_maximum, 0]
    mask = Image.frombytes("L", (3, 2), bytes(mask_values)) if mask_depth == 8 else make_image("L", depth, (3, 2), mask_values)
    a, b = samples(background), samples(foreground)
    result = Image.composite(foreground, background, mask)
    assert samples(result) == [(x * (mask_maximum - mask_values[i // 4]) + y * mask_values[i // 4] + mask_maximum // 2) // mask_maximum for i, (x, y) in enumerate(zip(a, b, strict=True))]
    assert samples(Image.blend(background, foreground, 0.5)) == [(x + y) // 2 for x, y in zip(a, b, strict=True)]
    pasted = background.copy()
    pasted.paste(foreground, (-1, 1), mask)
    expected = a.copy()
    for x in range(2):
        for c in range(4):
            i, j = (3 + x) * 4 + c, (x + 1) * 4 + c
            alpha = mask_values[x + 1]
            expected[i] = (a[i] * (mask_maximum - alpha) + b[j] * alpha + mask_maximum // 2) // mask_maximum
    assert samples(pasted) == expected
    background.putalpha(mask)
    assert samples(background)[3::4] == [(v * maximum + mask_maximum // 2) // mask_maximum for v in mask_values]
    background.putalpha(maximum - 17)
    assert samples(background)[3::4] == [maximum - 17] * 6
    rgb = foreground.convert("RGB")
    rgb.putalpha(mask)
    assert rgb.mode == "RGBA" and rgb.bit_depth == depth
    assert samples(rgb)[3::4] == [(v * maximum + mask_maximum // 2) // mask_maximum for v in mask_values]


@pytest.mark.parametrize("depth", [10, 12, 16])
def test_alpha_composite_uses_straight_alpha_and_wide_intermediates(depth):
    maximum = (1 << depth) - 1
    first = [maximum - 1, 715, 1, maximum, 1, 3, maximum, maximum // 2, 1, 2, 3, 0, 617, 891, 71, 0]
    second = [1, 3, maximum, 0, maximum, 1, 3, maximum // 2, 731, 1, 2, maximum, 5, 6, 7, 0]
    background = make_image("RGBA", depth, (4, 1), first)
    overlay = make_image("RGBA", depth, (4, 1), second)
    expected = []
    for i in range(0, len(first), 4):
        a, b = first[i : i + 4], second[i : i + 4]
        alpha = b[3] * maximum + a[3] * (maximum - b[3])
        if b[3] == 0:
            expected.extend(a)
        else:
            expected.extend((b[c] * b[3] * maximum + a[c] * a[3] * (maximum - b[3]) + alpha // 2) // alpha for c in range(3))
            expected.append((alpha + maximum // 2) // maximum)
    result = Image.alpha_composite(background, overlay)
    assert samples(result) == expected
    assert result.bit_depth == depth
    background.alpha_composite(overlay)
    assert background.tobytes() == result.tobytes()
    background.alpha_composite(background)
    assert background.tobytes() == Image.alpha_composite(result, result).tobytes()


@pytest.mark.parametrize("depth", [10, 12, 16])
@pytest.mark.parametrize("mode", ["RGB", "RGBA"])
def test_color_lut_and_merge_retain_low_bits(depth, mode):
    image = make_image(mode, depth)
    lut = ImageFilter.Color3DLUT.generate(3, lambda r, g, b: (b, r, g))
    result = image.filter(lut)
    expected = [v for pixel in image.getdata() for v in ((pixel[2], pixel[0], pixel[1], pixel[3]) if mode == "RGBA" else (pixel[2], pixel[0], pixel[1]))]
    assert samples(result) == expected
    assert result.bit_depth == depth
    assert Image.merge(mode, image.split()).tobytes() == image.tobytes()
    stream = BytesIO()
    result.convert(mode, bit_depth=16).save(stream, "PNG")
    reopened = Image.open(BytesIO(stream.getvalue()))
    assert reopened.tobytes() == result.convert(mode, bit_depth=16).tobytes()


@pytest.mark.parametrize(("mode", "depth"), FORMATS)
def test_empty_and_closed_images(mode, depth):
    empty = make_image(mode, depth, (0, 0))
    for operation in (ImageFilter.Kernel((3, 3), [1] * 9), ImageFilter.MedianFilter(), ImageFilter.ModeFilter(), ImageFilter.BoxBlur(2), ImageFilter.GaussianBlur(2), ImageFilter.UnsharpMask()):
        assert empty.filter(operation).tobytes() == b""
    for name in ImageEnhance.__all__:
        assert getattr(ImageEnhance, name)(empty).enhance(0.5).tobytes() == b""
    statistics = ImageStat.Stat(empty)
    channels = len(empty.getbands())
    assert statistics.extrema == [((1 << depth) - 1, 0)] * channels
    assert statistics.median == [(1 << depth) - 1] * channels
    assert statistics.mean == statistics.var == statistics.rms == [0.0] * channels
    empty.close()
    for operation in (lambda: empty.filter(ImageFilter.MedianFilter()), lambda: ImageStat.Stat(empty), lambda: ImageEnhance.Brightness(empty), lambda: Image.blend(empty, empty, 0)):
        with pytest.raises(ValueError, match="closed"):
            operation()


def test_mismatches_fail_before_mutating_destination():
    image = make_image("RGBA", 16)
    original = image.tobytes()
    other = make_image("RGBA", 12)
    for operation in (
        lambda: image.paste(other),
        lambda: Image.blend(image, other, 0),
        lambda: Image.alpha_composite(image, other),
        lambda: image.putalpha(make_image("L", 12)),
        lambda: Image.composite(image, image, Image.new("L", (1, 1))),
        lambda: Image.merge("RGBA", [image.getchannel(0)] * 3 + [make_image("L", 12)]),
    ):
        with pytest.raises(ValueError):
            operation()
        assert image.tobytes() == original


@pytest.mark.parametrize("name", ["BoxBlur", "GaussianBlur"])
def test_high_depth_blur_agrees_with_pillow_integer_samples(name):
    image = make_image("L", 16)
    pillow = PillowImage.new("L", image.size)
    pillow.putdata([v % 256 for v in samples(image)])
    image = make_image("L", 16, image.size, list(pillow.get_flattened_data()))
    actual = image.filter(getattr(ImageFilter, name)(1.5))
    expected = pillow.filter(getattr(PillowFilter, name)(1.5))
    assert samples(actual) == list(expected.get_flattened_data())


@pytest.mark.parametrize("depth", [10, 12, 16])
def test_high_depth_fills_and_rgba_masks(depth):
    maximum = (1 << depth) - 1
    image = make_image("RGBA", depth, (2, 1), [271, 301, 491, 0] * 2)
    image.paste((maximum, 0, 617), (0, 0, 2, 1), Image.new("L", (2, 1), 127))
    assert image.getdata() == [(maximum, 0, 617, (maximum * 127 + 127) // 255)] * 2
    image.paste("red", (0, 0, 2, 1))
    assert image.getdata() == [(maximum, 0, 0, maximum)] * 2
    image.paste((301, 617, maximum, 0), (0, 0, 2, 1))
    mask = make_image("RGBA", depth, (2, 1), [maximum, 0, 0, 0, 0, 0, 0, maximum])
    source = make_image("RGBA", depth, (2, 1), [maximum, 17, 301, maximum] * 2)
    image.paste(source, mask=mask)
    assert image.getdata() == [(301, 617, maximum, 0), (maximum, 17, 301, maximum)]
    gray = make_image("L", depth, (2, 1))
    gray.paste(maximum - 1, (0, 0, 2, 1))
    assert gray.getdata() == [maximum - 1] * 2
    image.paste(source.convert("RGB"), (0, 0))
    assert image.getdata() == [(maximum, 17, 301, maximum)] * 2


@pytest.mark.parametrize("mode", ["I;16", "I;16L", "I;16B"])
def test_integer16_masked_paste_uses_whole_samples(mode):
    image = make_image(mode, 16, (3, 1), [255, 65280, 65535])
    source = make_image(mode, 16, (3, 1), [256, 255, 0])
    mask = Image.frombytes("L", (3, 1), bytes([0, 128, 255]))
    image.paste(source, mask=mask)
    assert samples(image) == [255, (65280 * 127 + 255 * 128 + 127) // 255, 0]
    assert image.mode == mode


@pytest.mark.parametrize("depth", [10, 12, 16])
def test_high_depth_blur_large_radii_and_validation(depth):
    maximum = (1 << depth) - 1
    image = make_image("RGBA", depth, (1, 1), [maximum, maximum - 1, 1, 0])
    for radius in ((0, 1.5), (1.5, 0), (0.25, 2.75), (100000, 100000)):
        for cls in (ImageFilter.BoxBlur, ImageFilter.GaussianBlur):
            result = image.filter(cls(radius))
            assert max(abs(a - b) for a, b in zip(samples(result), samples(image), strict=True)) <= 1
    for radius in (math.nan, math.inf):
        for cls in (ImageFilter.BoxBlur, ImageFilter.GaussianBlur):
            with pytest.raises(ValueError, match="radius"):
                image.filter(cls(radius))
    with pytest.raises(ValueError, match="radius"):
        image.filter(ImageFilter.BoxBlur(-1))
    assert image.filter(ImageFilter.GaussianBlur(-1)).tobytes() == image.filter(ImageFilter.GaussianBlur(1)).tobytes()
    assert ImageStat.Stat(image).var == [0.0] * 4
    image.paste((maximum, maximum, maximum, maximum), (0, 0, 1, 1))
    assert ImageStat.Stat(image).sum2 == [maximum * maximum] * 4
    assert Image.blend(image, image, math.nan).tobytes() == image.tobytes()


@pytest.mark.parametrize("depth", [10, 12, 16])
def test_high_depth_lut_generates_alpha_and_clips(depth):
    maximum = (1 << depth) - 1
    image = make_image("RGB", depth, (2, 1), [0, maximum, 301, maximum, 0, maximum])
    lut = ImageFilter.Color3DLUT.generate(2, lambda r, g, b: (-1, 2, b, r), channels=4, target_mode="RGBA")
    result = image.filter(lut)
    assert (result.mode, result.bit_depth) == ("RGBA", depth)
    assert result.getdata() == [(0, maximum, 301, 0), (0, maximum, maximum, maximum)]


def test_high_depth_processing_png_interoperability():
    image = make_image("L", 16)
    result = ImageEnhance.Brightness(image).enhance(0.5).filter(ImageFilter.MedianFilter())
    stream = BytesIO()
    result.save(stream, "PNG")
    with PillowImage.open(BytesIO(stream.getvalue())) as pillow:
        assert list(pillow.get_flattened_data()) == result.getdata()


@pytest.mark.parametrize("count", [12345, 100001])
def test_high_depth_variance_avoids_cancellation(count):
    image = Image.new("I;16", (count, 1), 65535)
    statistics = ImageStat.Stat(image)
    assert statistics.var == statistics.stddev == [0.0]
    image.putpixel((0, 0), 65534)
    statistics = ImageStat.Stat(image)
    expected = (count - 1) / count**2
    assert statistics.var == pytest.approx([expected], rel=1e-12)
    assert statistics.stddev == pytest.approx([math.sqrt(expected)], rel=1e-12)


@pytest.mark.parametrize(("mode", "depth"), FORMATS)
@pytest.mark.parametrize("width", [15, 16, 17, 31, 32, 33])
def test_simd_enhancements_and_convolution_match_scalar_images(mode, depth, width):
    image = make_image(mode, depth, (width, 5))
    for cls in (ImageEnhance.Color, ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Sharpness):
        enhancer = cls(image)
        degenerate = enhancer.degenerate
        result = enhancer.enhance(0.5)
        assert samples(result) == [(a + b) // 2 for a, b in zip(samples(image), samples(degenerate), strict=True)]
    for kernel_size in (3, 5):
        weights = [1 if i % 3 == 0 else -0.5 for i in range(kernel_size**2)]
        operation = ImageFilter.Kernel((kernel_size, kernel_size), weights, scale=2, offset=0.25)
        result = image.filter(operation)
        radius = kernel_size // 2
        for x in range(radius, width - radius):
            region = image.crop((x - radius, 0, x + radius + 1, 5))
            expected = region.filter(operation)
            assert result.getpixel((x, 2)) == expected.getpixel((radius, 2))


@pytest.mark.parametrize("depth", [10, 12, 16])
@pytest.mark.parametrize("mode", ["RGB", "RGBA"])
def test_simd_lut_and_alpha_composite_match_scalar_pixels(depth, mode):
    image = make_image(mode, depth, (33, 1))
    lut = ImageFilter.Color3DLUT.generate((3, 4, 5), lambda r, g, b: (r * g, g * b, b * r, (r + g + b) / 3), channels=4, target_mode="RGBA")
    result = image.filter(lut)
    for x in range(image.width):
        assert result.getpixel((x, 0)) == image.crop((x, 0, x + 1, 1)).filter(lut).getpixel((0, 0))
    overlay = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT).convert("RGBA")
    background = image.convert("RGBA")
    result = Image.alpha_composite(background, overlay)
    for x in range(image.width):
        expected = Image.alpha_composite(background.crop((x, 0, x + 1, 1)), overlay.crop((x, 0, x + 1, 1)))
        assert result.getpixel((x, 0)) == expected.getpixel((0, 0))
