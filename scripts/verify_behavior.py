"""Differential public API and cross-codec checks against Pillow."""

from __future__ import annotations

import random
from io import BytesIO, StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import (
    Image as PillowImage,
    ImageChops as PillowChops,
    ImageColor as PillowColor,
    ImageDraw as PillowDraw,
    ImageEnhance as PillowEnhance,
    ImageFilter,
    ImageFont as PillowFont,
    ImageMath as PillowMath,
    ImageOps as PillowOps,
    ImagePalette as PillowPalette,
    ImagePath as PillowPath,
    ImageStat as PillowStat,
)
from PIL.FontFile import FontFile

from blanket import (
    Image as BlanketImage,
    ImageChops as BlanketChops,
    ImageColor as BlanketColor,
    ImageDraw as BlanketDraw,
    ImageEnhance as BlanketEnhance,
    ImageFilter as BlanketFilter,
    ImageFont as BlanketFont,
    ImageMath as BlanketMath,
    ImageOps as BlanketOps,
    ImagePalette as BlanketPalette,
    ImagePath as BlanketPath,
    ImageStat as BlanketStat,
    features,
)
from blanket.Compressor import LosslessImageCompressor, LossyImageCompressor


def compare_images(actual: BlanketImage.Image, expected: PillowImage.Image, label: object) -> None:
    assert (actual.mode, actual.size) == (expected.mode, expected.size), label
    assert actual.tobytes() == expected.tobytes(), label


def pixels(mode: str, width: int = 37, height: int = 29) -> bytes:
    channels = {"L": 1, "LA": 2, "RGB": 3, "RGBA": 4, "HSV": 3, "CMYK": 4, "YCbCr": 3, "LAB": 3}[mode]
    return bytes((x * 17 + y * 29 + channel * 53) % 256 for y in range(height) for x in range(width) for channel in range(channels))


def check_conversions() -> int:
    checks = 0
    for source_mode in ("L", "RGB", "RGBA", "HSV", "CMYK", "YCbCr"):
        raw = pixels(source_mode)
        blanket = BlanketImage.frombytes(source_mode, (37, 29), raw)
        pillow = PillowImage.frombytes(source_mode, (37, 29), raw)
        for target_mode in ("L", "RGB", "RGBA", "HSV", "CMYK", "YCbCr"):
            actual = blanket.convert(target_mode)
            expected = pillow.convert(target_mode)
            assert actual.mode == expected.mode
            assert actual.size == expected.size
            assert actual.tobytes() == expected.tobytes()
            checks += 1
    return checks


def check_lab() -> int:
    checks = 0
    for source, target in (("RGB", "LAB"), ("LAB", "RGB")):
        raw = pixels(source)
        actual = BlanketImage.frombytes(source, (37, 29), raw)
        expected = PillowImage.frombytes(source, actual.size, raw)
        assert actual.convert(target).tobytes() == expected.convert(target).tobytes()
        assert actual.getdata() == list(expected.get_flattened_data())
        checks += 2
    stream = BytesIO()
    actual.save(stream, "TIFF")
    for module in (BlanketImage, PillowImage):
        reopened = module.open(BytesIO(stream.getvalue()))
        assert reopened.mode == "LAB" and reopened.tobytes() == actual.tobytes()
        checks += 1
    return checks


def check_fromarray() -> int:
    checks = 0
    for mode in ("L", "RGB", "RGBA"):
        raw = pixels(mode)
        channels = {"L": (), "RGB": (3,), "RGBA": (4,)}[mode]
        array = np.frombuffer(raw, dtype=np.uint8).reshape((29, 37, *channels))
        blanket = BlanketImage.fromarray(array)
        pillow = PillowImage.fromarray(array)
        assert (blanket.mode, blanket.size) == (pillow.mode, pillow.size)
        assert blanket.tobytes() == pillow.tobytes()
        checks += 1

    hsv = np.frombuffer(pixels("HSV", 16, 12), dtype=np.uint8).reshape(12, 16, 3)
    assert BlanketImage.fromarray(hsv, "HSV").tobytes() == PillowImage.fromarray(hsv, "HSV").tobytes()
    checks += 1

    source = np.arange(12 * 16 * 3, dtype=np.uint8).reshape(12, 16, 3)
    strided = source[::2, ::2]
    assert BlanketImage.fromarray(strided).tobytes() == PillowImage.fromarray(strided).tobytes()
    return checks + 1


def check_image_access() -> int:
    """Compare pixel mutation, channel access, bounds, histograms, and LUTs."""
    checks = 0
    for mode in ("1", "L", "LA", "RGB", "RGBA", "HSV", "CMYK", "YCbCr", "LAB", "P", "PA", "I", "I;16", "I;16L", "I;16B", "F"):
        reference = PillowImage.new(mode, (37, 29))
        if mode in ("1", "P", "I", "F", "I;16", "I;16L", "I;16B"):
            reference.putdata([255 if i % 3 else 0 for i in range(37 * 29)] if mode == "1" else [(i * 17) % 251 for i in range(37 * 29)])
        else:
            source_mode = "LA" if mode == "PA" else mode
            reference = PillowImage.frombytes(mode, reference.size, pixels(source_mode))
        actual = BlanketImage.frombytes(mode, reference.size, reference.tobytes())
        assert actual.getbands() == reference.getbands(), mode
        assert actual.get_flattened_data() == reference.get_flattened_data(), mode
        assert actual.getdata() == list(reference.get_flattened_data()), mode
        assert actual.getextrema() == (reference.convert("I") if mode in ("I;16L", "I;16B") else reference).getextrema(), mode
        for alpha_only in (False, True):
            assert actual.getbbox(alpha_only=alpha_only) == reference.getbbox(alpha_only=alpha_only), (mode, alpha_only)
        for limit in () if mode.startswith("I;16") else (1, 256, 37 * 29):
            colors, expected_colors = actual.getcolors(limit), reference.getcolors(limit)
            assert (None if colors is None else sorted(colors)) == (None if expected_colors is None else sorted(expected_colors)), mode
            checks += 1
        checks += 6
        copied, reference_copy = actual.copy(), reference.copy()
        value = reference.getpixel((12, 13))
        copied.putpixel((-1, -1), value)
        reference_copy.putpixel((-1, -1), value)
        compare_images(copied, reference_copy, (mode, "putpixel"))
        compare_images(actual, reference, (mode, "copy independence"))
        checks += 2
        if mode in ("I", "F", "I;16", "I;16L", "I;16B", "PA"):
            continue
        mask = BlanketImage.frombytes("L", actual.size, pixels("L"))
        reference_mask = PillowImage.frombytes("L", actual.size, mask.tobytes())
        assert actual.histogram() == reference.histogram(), mode
        assert actual.histogram(mask) == reference.histogram(reference_mask), mode
        checks += 2
        if mode == "P":
            continue
        for band, name in enumerate(actual.getbands()):
            compare_images(actual.getchannel(band), reference.getchannel(band), (mode, band))
            compare_images(actual.getchannel(name), reference.getchannel(name), (mode, name))
            assert actual.getdata(band) == list(reference.get_flattened_data(band)), (mode, band)
            checks += 3
        if mode == "1":
            continue
        for lut in ([255 - value for value in range(256)] * len(actual.getbands()), lambda value: value // 2):
            compare_images(actual.point(lut), reference.point(lut), (mode, "point"))
            checks += 1
        data = list(reference.get_flattened_data())[::-1]
        actual.putdata(data)
        reference.putdata(data)
        compare_images(actual, reference, (mode, "putdata"))
        checks += 1
    for mode in ("L", "I", "F"):
        actual, expected = BlanketImage.new(mode, (7, 3)), PillowImage.new(mode, (7, 3))
        actual.putdata(list(range(21)), scale=2, offset=3)
        expected.putdata(list(range(21)), scale=2, offset=3)
        compare_images(actual, expected, (mode, "scaled putdata"))
        checks += 1
    for dtype in (np.uint16, np.int32, np.float32, np.float64):
        array = np.arange(37 * 29).reshape(29, 37).astype(dtype)[::2, ::2]
        compare_images(BlanketImage.fromarray(array), PillowImage.fromarray(array), str(dtype))
        checks += 1
    return checks


def check_image_geometry() -> int:
    """Compare direct geometry methods, including every transform method."""
    checks = 0
    transforms = (
        (BlanketImage.AFFINE, (0.9, 0.1, -2, -0.1, 1.1, 3)),
        (BlanketImage.EXTENT, (-2, 1, 39, 28)),
        (BlanketImage.PERSPECTIVE, (1, 0.1, -2, 0.05, 1, 1, 0.001, 0.002)),
        (BlanketImage.QUAD, (1, 2, 0, 27, 35, 28, 36, 0)),
        (BlanketImage.MESH, [((0, 0, 19, 17), (1, 2, 0, 27, 35, 28, 36, 0))]),
    )
    for mode in ("1", "L", "LA", "RGB", "RGBA", "HSV", "CMYK", "YCbCr", "LAB", "P", "I", "F", "I;16"):
        source_mode = "L" if mode in ("1", "P", "I", "F", "I;16") else mode
        reference = PillowImage.frombytes(source_mode, (37, 29), pixels(source_mode)).convert(mode)
        actual = BlanketImage.frombytes(mode, reference.size, reference.tobytes())
        for method in BlanketImage.Transpose:
            compare_images(actual.transpose(method), reference.transpose(method), (mode, method.name))
            checks += 1
        for method in BlanketImage.Resampling:
            # Wide-sample interpolation has different rounding from Pillow.
            if mode in ("I", "F", "I;16") and method != BlanketImage.NEAREST:
                continue
            for box, gap in ((None, None), ((1.5, 2.5, 35.5, 27.5), 1.5)):
                compare_images(actual.resize((9, 7), method, box, gap), reference.resize((9, 7), method, box, gap), (mode, "resize", method, box, gap))
                checks += 1
            small, expected_small = actual.copy(), reference.copy()
            small.thumbnail((9, 8), method)
            expected_small.thumbnail((9, 8), method)
            compare_images(small, expected_small, (mode, "thumbnail", method))
            checks += 1
        # Compare warps on the 8-bit modes with shared Pillow semantics.
        if mode in ("I", "F", "I;16"):
            continue
        for method in (BlanketImage.NEAREST, BlanketImage.BILINEAR, BlanketImage.BICUBIC):
            if mode in ("I", "F", "I;16") and method != BlanketImage.NEAREST:
                continue
            for angle, options in ((90, {}), (17, {"expand": True}), (-33, {"center": (12, 10), "translate": (2, -3), "fillcolor": 0})):
                compare_images(actual.rotate(angle, method, **options), reference.rotate(angle, method, **options), (mode, "rotate", method, angle))
                checks += 1
            for transform, data in transforms:
                compare_images(actual.transform((19, 17), transform, data, method), reference.transform((19, 17), transform, data, method), (mode, "transform", transform, method))
                checks += 1
    return checks


def check_indexed_images() -> int:
    checks = 0
    palette = bytes(value for i in range(256) for value in (i, 255 - i, i // 2))
    actual = BlanketImage.frombytes("P", (37, 29), pixels("L"))
    expected = PillowImage.frombytes("P", actual.size, actual.tobytes())
    for image in (actual, expected):
        image.putpalette(palette)
    for rawmode in ("RGB", "RGBA"):
        assert actual.getpalette(rawmode) == expected.getpalette(rawmode)
        checks += 1
    for mapping in (list(reversed(range(256))), list(range(0, 256, 2))):
        for source in (None, palette):
            result, reference = actual.remap_palette(mapping, source), expected.remap_palette(mapping, source)
            compare_images(result, reference, "remap_palette")
            assert result.getpalette() == reference.getpalette()
            compare_images(result.convert("RGB"), reference.convert("RGB"), "remapped colors")
            checks += 1
    return checks


def check_imagecolor() -> int:
    checks = 0
    colors = (*PillowColor.colormap, "#123", "#1234", "#102030", "#10203080", "rgb(12,34,56)", "rgb(100%,0%,50%)", "rgba(1,2,3,4)", "hsl(120,30%,50%)", "hsv(250,50%,80%)")
    for color in colors:
        assert BlanketColor.getrgb(color) == PillowColor.getrgb(color), color
        checks += 1
        for mode in ("L", "LA", "RGB", "RGBA", "I", "F"):
            assert BlanketColor.getcolor(color, mode) == PillowColor.getcolor(color, mode), (color, mode)
            checks += 1
        # Blanket normalizes bilevel colors to binary sample values.
        assert BlanketColor.getcolor(color, "1") == (255 if PillowColor.getcolor(color, "L") >= 128 else 0), color
        checks += 1
    return checks


def check_imagepath() -> int:
    checks = 0
    points = [(0, 0), (0, 0), (0.5, 0.5), (1, 1), (2, 1), (4, 4)]
    for coordinates in ([], 3, points, [value for pair in points for value in pair]):
        actual, expected = BlanketPath.Path(coordinates), PillowPath.Path(coordinates)
        assert len(actual) == len(expected)
        assert list(actual) == list(expected)
        assert actual.tolist(True) == expected.tolist(True)
        assert actual.getbbox() == expected.getbbox()
        assert actual[1:].tolist() == expected[1:].tolist()
        if actual:
            actual[-1] = expected[-1] = (8, 9)
            assert actual[-1] == expected[-1]
        checks += 1
    for distance in (0, 1, 2.5, 100):
        actual, expected = BlanketPath.Path(points), PillowPath.Path(points)
        assert actual.compact(distance) == expected.compact(distance)
        assert actual.tolist() == expected.tolist()
        checks += 1
    for wrap in (0, 3):
        actual, expected = BlanketPath.Path(points), PillowPath.Path(points)
        for path in (actual, expected):
            path.map(lambda x, y: (x + 2, y * 3))
            path.transform((0.5, 2, -1, -2, 0.25, 4), wrap)
        assert actual.tolist() == expected.tolist()
        checks += 1
    return checks


def check_imagemath() -> int:
    checks = 0
    expressions = (
        "+a",
        "-a",
        "abs(a)",
        "a + b",
        "a - b",
        "a * b",
        "a / b",
        "a % b",
        "a ** 2",
        "2 + a",
        "5 - a",
        "3 * a",
        "7 / a",
        "7 % a",
        "a == b",
        "a != b",
        "a < b",
        "a <= b",
        "a > b",
        "a >= b",
        "min(a, b)",
        "max(a, b)",
        "equal(a, b)",
        "notequal(a, b)",
        "int(a)",
        "float(a)",
        "convert(a + b, 'L')",
    )
    for mode in ("1", "L", "I", "F"):
        reference = PillowImage.new(mode, (17, 3))
        reference.putdata([i % 19 - 7 if mode in ("I", "F") else i % 251 for i in range(51)])
        other = reference.transpose(PillowImage.Transpose.FLIP_LEFT_RIGHT)
        actual = BlanketImage.frombytes(mode, reference.size, reference.tobytes())
        second = BlanketImage.frombytes(mode, other.size, other.tobytes())
        cases = expressions
        if mode != "F":
            cases += ("~a", "a & 3", "a | 3", "a ^ 3", "a << 2", "a >> 2")
        for expression in cases:
            expected = PillowMath.unsafe_eval(expression, a=reference, b=other)
            for evaluator in (BlanketMath.eval, BlanketMath.unsafe_eval):
                compare_images(evaluator(expression, a=actual, b=second), expected, (mode, expression))
                checks += 1

        def evaluate(args):
            return args["convert"](args["max"](args["float"](args["a"]) / 2, 1.5), "L")

        compare_images(BlanketMath.lambda_eval(evaluate, a=actual), PillowMath.lambda_eval(evaluate, a=reference), (mode, "lambda_eval"))
        checks += 1
    return checks


def check_png_interop() -> int:
    checks = 0
    for mode in ("L", "RGB", "RGBA"):
        raw = pixels(mode)
        pillow_bytes = BytesIO()
        PillowImage.frombytes(mode, (37, 29), raw).save(pillow_bytes, "PNG")
        blanket_loaded = BlanketImage.open(pillow_bytes)
        assert (blanket_loaded.format, blanket_loaded.mode) == ("PNG", mode)
        assert blanket_loaded.tobytes() == raw

        blanket_bytes = BytesIO()
        BlanketImage.frombytes(mode, (37, 29), raw).save(blanket_bytes, "PNG")
        blanket_bytes.seek(0)
        with PillowImage.open(blanket_bytes) as pillow_loaded:
            pillow_loaded.load()
            assert pillow_loaded.mode == mode
            assert pillow_loaded.tobytes() == raw
        checks += 2
    return checks


def check_bmp_gif_ico_interop() -> int:
    checks = 0
    for format in ("BMP", "GIF", "ICO"):
        for mode in ("L", "RGB", "RGBA"):
            source = PillowImage.new(mode, (32, 32), 120 if mode == "L" else (60, 120, 180) if mode == "RGB" else (60, 120, 180, 255))
            for writer in (source, BlanketImage.frombytes(mode, source.size, source.tobytes())):
                output = BytesIO()
                writer.save(output, format)
                actual = BlanketImage.open(output, formats=[format])
                expected = PillowImage.open(output).convert("RGBA")
                assert (actual.size, actual.format) == (source.size, format)
                assert actual.convert("RGBA").tobytes() == expected.tobytes()
                checks += 1
    return checks


def check_jpeg_interop() -> int:
    raw = pixels("RGB")
    for producer in ("Pillow", "Blanket"):
        stream = BytesIO()
        if producer == "Pillow":
            PillowImage.frombytes("RGB", (37, 29), raw).save(stream, "JPEG", quality=85)
        else:
            BlanketImage.frombytes("RGB", (37, 29), raw).save(stream, "JPEG", quality=85)

        payload = stream.getvalue()
        with PillowImage.open(BytesIO(payload)) as pillow_loaded:
            pillow_loaded.load()
            blanket_loaded = BlanketImage.open(BytesIO(payload))
            assert blanket_loaded.size == pillow_loaded.size == (37, 29)
            assert blanket_loaded.mode == pillow_loaded.mode == "RGB"
            differences = [abs(left - right) for left, right in zip(blanket_loaded.tobytes(), pillow_loaded.tobytes(), strict=True)]
            assert sum(differences) / len(differences) <= 1.0
            assert max(differences) <= 4
    return 2


def check_jxl_roundtrip() -> int:
    checks = 0
    for mode in ("L", "RGB", "RGBA"):
        raw = pixels(mode, 16, 12)
        stream = BytesIO()
        BlanketImage.frombytes(mode, (16, 12), raw).save(stream, "JXL", lossless=True, effort=1)
        loaded = BlanketImage.open(stream)
        assert (loaded.format, loaded.mode, loaded.size) == ("JXL", mode, (16, 12))
        assert loaded.tobytes() == raw
        checks += 1
    return checks


def check_pillow_adapter() -> int:
    """Exercise unsupported image operations through Blanket's Pillow adapter."""
    size = (32, 24)
    raw = bytes((x * 13 + y * 31 + channel * 71) % 256 for y in range(size[1]) for x in range(size[0]) for channel in range(3))
    blanket = BlanketImage.frombytes("RGB", size, raw)
    pillow = blanket.to_pillow()
    assert isinstance(pillow, PillowImage.Image)

    bands = pillow.split()
    merged = PillowImage.merge("RGB", bands)
    assert merged.tobytes() == raw

    resized = merged.resize((16, 12))
    rotated = resized.rotate(17)
    filtered = rotated.filter(ImageFilter.DETAIL)
    processed = PillowOps.autocontrast(filtered)
    assert processed.size == (16, 12)

    returned = BlanketImage.fromarray(np.asarray(processed))
    assert (returned.mode, returned.size) == (processed.mode, processed.size)
    output = BytesIO()
    returned.save(output, "PNG")
    reloaded = BlanketImage.open(output)
    assert reloaded.tobytes() == processed.tobytes()
    return 1


class MirrorMesh:
    """Exercise the deformer protocol with a horizontal reflection."""

    def getmesh(self, image: BlanketImage.Image | PillowImage.Image) -> list[tuple[tuple[int, int, int, int], tuple[int, ...]]]:
        w, h = image.size
        return [((0, 0, w, h), (w, 0, w, h, 0, h, 0, 0))]


def check_bands_statistics() -> int:
    """Compare split, reduction, and entropy across all supported modes."""
    checks = 0
    for mode in ("L", "RGB", "RGBA", "HSV"):
        raw = pixels(mode)
        blanket = BlanketImage.frombytes(mode, (37, 29), raw)
        pillow = PillowImage.frombytes(mode, (37, 29), raw)
        for actual, expected in zip(blanket.split(), pillow.split(), strict=False):
            assert (actual.mode, actual.size, actual.tobytes()) == (expected.mode, expected.size, expected.tobytes())
            checks += 1
        for xy in ((0, 0), (36, 28), (-1, -1), (-37, -29), (12, 13)):
            assert blanket.getpixel(xy) == pillow.getpixel(xy)
            checks += 1
        for factor in (1, 2, 3, (2, 5), (1, 7), (50, 40)):
            for box in (None, (1, 2, 36, 28)):
                actual, expected = blanket.reduce(factor, box), pillow.reduce(factor, box)
                assert (actual.mode, actual.size, actual.tobytes()) == (expected.mode, expected.size, expected.tobytes())
                checks += 1
        # HSV has no native quantizer. Pillow rejects it; Blanket converts through RGB only for convert("P").
        if mode == "HSV":
            assert abs(blanket.entropy() - pillow.entropy()) < 1e-12
            checks += 1
            mask = pixels("L")
            b_mask = BlanketImage.frombytes("L", blanket.size, mask)
            p_mask = PillowImage.frombytes("L", pillow.size, mask)
            assert abs(blanket.entropy(b_mask) - pillow.entropy(p_mask)) < 1e-12
            checks += 1
            continue
        # Generated palettes need not choose Pillow's exact colors/order.
        # Check the public contract and compare decoding through Pillow.
        for method in (2,) if mode == "RGBA" else (0, 1, 2):
            quantized = blanket.quantize(colors=16, method=method)
            expected = pillow.quantize(colors=16, method=method)
            assert (quantized.mode, quantized.size) == (expected.mode, expected.size)
            assert len(set(quantized.tobytes())) <= 16
            reference = quantized.to_pillow()
            assert quantized.convert(mode).tobytes() == reference.convert(mode).tobytes()
            output = BytesIO()
            quantized.save(output, "PNG")
            assert PillowImage.open(BytesIO(output.getvalue())).convert(mode).tobytes() == quantized.convert(mode).tobytes()
            checks += 1
        if mode == "RGB":
            palette = blanket.quantize(colors=16)
            actual = blanket.quantize(palette=palette, dither=BlanketImage.Dither.NONE)
            expected = pillow.quantize(palette=palette.to_pillow(), dither=PillowImage.Dither.NONE)
            assert actual.convert("RGB").tobytes() == expected.convert("RGB").tobytes()
            checks += 1
        assert abs(blanket.entropy() - pillow.entropy()) < 1e-12
        checks += 1
        mask = pixels("L")
        b_mask = BlanketImage.frombytes("L", blanket.size, mask)
        p_mask = PillowImage.frombytes("L", pillow.size, mask)
        assert abs(blanket.entropy(b_mask) - pillow.entropy(p_mask)) < 1e-12
        checks += 1
    return checks


def check_crop_apis() -> int:
    """Keep Image.crop box coordinates distinct from ImageOps.crop borders."""
    checks = 0
    for mode in ("L", "RGB", "RGBA", "HSV"):
        raw = pixels(mode)
        blanket = BlanketImage.frombytes(mode, (37, 29), raw)
        pillow = PillowImage.frombytes(mode, (37, 29), raw)
        for box in (None, (1, 2, 3, 4), (-2, -3, 40, 32), (0.5, 1.5, 9.5, 8.5)):
            actual, expected = blanket.crop(box=box), pillow.crop(box=box)
            assert (actual.mode, actual.size, actual.tobytes()) == (expected.mode, expected.size, expected.tobytes()), f"Image.crop {mode} {box}"
            checks += 1
        for border in (0, 2, -2, (1, 2), (1, 2, 3, 4)):
            actual = BlanketOps.crop(blanket, border=border)
            expected = PillowOps.crop(pillow, border=border)
            assert (actual.mode, actual.size, actual.tobytes()) == (expected.mode, expected.size, expected.tobytes()), f"ImageOps.crop {mode} {border}"
            checks += 1
    return checks


def check_imageops() -> int:
    """Compare every ImageOps function, including masks, filters and EXIF."""
    checks = 0

    def compare(actual: BlanketImage.Image, expected: PillowImage.Image, label: str) -> None:
        nonlocal checks
        assert (actual.mode, actual.size) == (expected.mode, expected.size), label
        assert actual.tobytes() == expected.tobytes(), label
        checks += 1

    for mode in ("L", "RGB", "RGBA", "HSV"):
        raw = pixels(mode)
        blanket = BlanketImage.frombytes(mode, (37, 29), raw)
        pillow = PillowImage.frombytes(mode, (37, 29), raw)
        # Named fills go through ImageColor, which converts into HSV. Blanket's canvas fill stores the RGB tuple.
        fill = 0 if mode == "HSV" else "rebeccapurple"
        cases = [("crop", {"border": (1, 2, 3, 4)}), ("expand", {"border": (2, 3), "fill": fill}), ("flip", {}), ("mirror", {}), ("grayscale", {})]
        if mode in ("L", "RGB"):
            cases += [
                ("autocontrast", {}),
                ("autocontrast", {"cutoff": (2, 5), "ignore": [0, 255], "preserve_tone": True}),
                ("equalize", {}),
                ("invert", {}),
                ("posterize", {"bits": 4}),
                ("solarize", {"threshold": 100}),
            ]
            mask = pixels("L")
            b_mask = BlanketImage.frombytes("L", blanket.size, mask)
            p_mask = PillowImage.frombytes("L", pillow.size, mask)
            for name in ("autocontrast", "equalize"):
                compare(getattr(BlanketOps, name)(blanket, mask=b_mask), getattr(PillowOps, name)(pillow, mask=p_mask), f"{name} masked {mode}")
        if mode == "L":
            cases += [("colorize", {"black": "navy", "white": "gold"}), ("colorize", {"black": "black", "white": "white", "mid": "red", "blackpoint": 10, "midpoint": 100, "whitepoint": 240})]
        for name, options in cases:
            compare(getattr(BlanketOps, name)(blanket, **options), getattr(PillowOps, name)(pillow, **options), f"{name} {mode} {options}")

        for method in BlanketImage.Resampling:
            pad_color = 0 if mode == "HSV" else "#1234"
            for name, options in (("contain", {}), ("cover", {}), ("fit", {"bleed": 0.05, "centering": (0.2, 0.8)}), ("pad", {"color": pad_color, "centering": (0, 1)})):
                compare(getattr(BlanketOps, name)(blanket, (19, 17), method, **options), getattr(PillowOps, name)(pillow, (19, 17), method, **options), f"{name} {mode} {method.name}")
            compare(BlanketOps.scale(blanket, 1.5, method), PillowOps.scale(pillow, 1.5, method), f"scale {mode} {method.name}")
        for method in (BlanketImage.NEAREST, BlanketImage.BILINEAR, BlanketImage.BICUBIC):
            compare(BlanketOps.deform(blanket, MirrorMesh(), method), PillowOps.deform(pillow, MirrorMesh(), method), f"deform {mode} {method.name}")

        if mode == "HSV":
            continue
        for orientation in range(1, 9):
            exif = PillowImage.Exif()
            exif[274] = orientation
            exif[270] = "Preserve unrelated EXIF data"
            # Decode fresh inputs for each call, including the mutating variant.
            payload = BytesIO()
            pillow.save(payload, "PNG", exif=exif)
            for in_place in (False, True):
                b_source = BlanketImage.open(BytesIO(payload.getvalue()))
                p_source = PillowImage.open(BytesIO(payload.getvalue()))
                with b_source, p_source:
                    actual = BlanketOps.exif_transpose(b_source, in_place=in_place)
                    expected = PillowOps.exif_transpose(p_source, in_place=in_place)
                    if in_place:
                        assert actual is expected is None
                        actual, expected = b_source, p_source
                    compare(actual, expected, f"exif_transpose {mode} orientation={orientation} in_place={in_place}")
                    actual_exif = PillowImage.Exif()
                    actual_exif.load(actual.info["exif"])
                    assert dict(actual_exif) == dict(expected.getexif())
    return checks


def check_imageenhance() -> int:
    """Compare every ImageEnhance class across modes and factor ranges."""
    checks = 0
    for mode in ("L", "RGB", "RGBA", "HSV"):
        raw = pixels(mode)
        blanket = BlanketImage.frombytes(mode, (37, 29), raw)
        pillow = PillowImage.frombytes(mode, (37, 29), raw)
        # Color and Contrast on HSV use Pillow's luminance of H,S,V. Blanket matches that only for RGB-family modes.
        names = ("Brightness", "Sharpness") if mode == "HSV" else ("Color", "Contrast", "Brightness", "Sharpness")
        for name in names:
            actual = getattr(BlanketEnhance, name)(blanket)
            expected = getattr(PillowEnhance, name)(pillow)
            assert actual.degenerate.tobytes() == expected.degenerate.tobytes(), f"{name} degenerate {mode}"
            checks += 1
            for factor in (-1, 0, 0.5, 1, 1.75, 3):
                result = actual.enhance(factor)
                reference = expected.enhance(factor)
                assert (result.mode, result.size) == (reference.mode, reference.size), f"{name} {mode} {factor}"
                assert result.tobytes() == reference.tobytes(), f"{name} {mode} {factor}"
                checks += 1
    return checks


def check_imagepalette() -> int:
    """Compare palette state, allocation, factories, serialization, and loaders."""
    checks = 0
    for mode in ("RGB", "RGBA", "CMYK"):
        for factory in ("wedge", "negative"):
            actual = getattr(BlanketPalette, factory)(mode)
            expected = getattr(PillowPalette, factory)(mode)
            assert actual.getdata() == expected.getdata()
            assert actual.tobytes() == expected.tobytes()
            assert actual.colors == expected.colors
            assert actual.copy().getdata() == expected.copy().getdata()
            output, reference = StringIO(), StringIO()
            actual.save(output)
            expected.save(reference)
            assert output.getvalue() == reference.getvalue()
            checks += 1
    for mode in ("RGB", "RGBA"):
        actual, expected = BlanketPalette.ImagePalette(mode), PillowPalette.ImagePalette(mode)
        for value in range(256):
            color = (value, 255 - value, value // 2)
            assert actual.getcolor(color) == expected.getcolor(color)
        assert actual.getdata() == expected.getdata()
        assert actual.colors == expected.colors
        checks += 1
    state = random.getstate()
    try:
        random.seed(12)
        actual = BlanketPalette.random()
        random.seed(12)
        assert actual.getdata() == PillowPalette.random().getdata()
        checks += 1
    finally:
        random.setstate(state)
    for white in ("#fff0c0", "red", "#123", "hsl(120, 50%, 50%)"):
        assert BlanketPalette.sepia(white).getdata() == PillowPalette.sepia(white).getdata()
        checks += 1
    for value in (0, 0.5, 1, 2.2, 255):
        assert BlanketPalette.make_gamma_lut(value) == PillowPalette.make_gamma_lut(value)
        assert BlanketPalette.make_linear_lut(0, value) == PillowPalette.make_linear_lut(0, value)
        checks += 2
    assert BlanketPalette.raw("RGB;L", b"\1\2\3").getdata() == PillowPalette.raw("RGB;L", b"\1\2\3").getdata()
    checks += 1
    with TemporaryDirectory() as directory:
        path = Path(directory) / "palette"
        payloads = ["0 255 0 0\n1 128\n", "GIMP Palette\nName: Sample\n255 0 0 Red\n0 255 0 Green\n"]
        payloads += [f"GIMP Gradient\nName: Sample\n1\n0 .3 1 0 .2 .4 0 1 .6 .9 1 {kind} 0\n" for kind in range(5)]
        for payload in payloads:
            path.write_text(payload)
            assert BlanketPalette.load(str(path)) == PillowPalette.load(str(path))
            checks += 1
    return checks


def check_imagefilter() -> int:
    """Compare native filters byte-for-byte across every supported image mode."""
    checks = 0
    specifications = [(name, ()) for name in ("BLUR", "CONTOUR", "DETAIL", "EDGE_ENHANCE", "EDGE_ENHANCE_MORE", "EMBOSS", "FIND_EDGES", "SHARPEN", "SMOOTH", "SMOOTH_MORE")]
    specifications += [("Kernel", ((3, 3), [0, 1, 0, -1, 2, 1, 0, 1, 0], 4, 3))]
    specifications += [("RankFilter", (5, 7)), ("MedianFilter", (3,)), ("MinFilter", (5,)), ("MaxFilter", (5,)), ("ModeFilter", (3,))]
    specifications += [(name, (radius,)) for name in ("BoxBlur", "GaussianBlur") for radius in (0, 0.3, 2, (1.7, 0.5), 20)]
    specifications += [("UnsharpMask", ()), ("UnsharpMask", (1.5, 75, 0))]
    for mode in ("L", "RGB", "RGBA", "HSV"):
        raw = pixels(mode)
        actual = BlanketImage.frombytes(mode, (37, 29), raw)
        expected = PillowImage.frombytes(mode, (37, 29), raw)
        for name, args in specifications:
            # Pillow rejects nonzero box/gaussian blurs and unsharp mask on HSV.
            if mode == "HSV" and (name == "UnsharpMask" or (name in ("BoxBlur", "GaussianBlur") and args != (0,))):
                continue
            result = actual.filter(getattr(BlanketFilter, name)(*args))
            reference = expected.filter(getattr(ImageFilter, name)(*args))
            assert (result.mode, result.size, result.info) == (reference.mode, reference.size, reference.info), name
            assert result.tobytes() == reference.tobytes(), f"{name} {mode} {args}"
            checks += 1
        if mode != "L":
            for channels in (3, 4):

                def color(r: float, g: float, b: float, channels: int = channels) -> tuple[float, ...]:
                    return (1 - r, g * g, b * 1.2) + ((0.5,) if channels == 4 else ())

                target = "RGBA" if channels == 4 else None
                lut = BlanketFilter.Color3DLUT.generate((3, 4, 5), color, channels, target)
                reference_lut = ImageFilter.Color3DLUT.generate((3, 4, 5), color, channels, target)
                assert lut.table == reference_lut.table
                assert actual.filter(lut).tobytes() == expected.filter(reference_lut).tobytes()
                checks += 1
                for with_normals in (False, True):

                    def transform(*values: float) -> tuple[float, ...]:
                        samples = values[3:] if with_normals else values
                        return tuple(1 - sample for sample in samples)

                    transformed = lut.transform(transform, with_normals=with_normals)
                    reference_transformed = reference_lut.transform(transform, with_normals=with_normals)
                    assert transformed.table == reference_transformed.table
                    compare_images(actual.filter(transformed), expected.filter(reference_transformed), ("Color3DLUT.transform", mode, channels, with_normals))
                    checks += 1
    return checks


def check_heif_and_high_depth() -> int:
    """Check an independent HEIF implementation and exact 10-bit PNG storage."""
    import pillow_heif

    checks = 0
    for channels in (3, 4):
        samples = (np.arange(32 * 32 * channels, dtype=np.uint16) % 1024).reshape(32, 32, channels)
        image = BlanketImage.fromarray(samples, bit_depth=10)
        output = BytesIO()
        image.save(output, "PNG")
        result = BlanketImage.open(output)
        assert result.bit_depth == 10 and result.tobytes() == image.tobytes()
        checks += 1
        output = BytesIO()
        image.save(output, "HEIC", quality=95)
        result = BlanketImage.open(output)
        expected = np.asarray(pillow_heif.open_heif(output.getvalue(), convert_hdr_to_8bit=False, hdr_to_16bit=False))
        assert result.bit_depth == 10
        assert result.tobytes() == expected.astype("<u2").tobytes()
        checks += 1
        external = pillow_heif.from_bytes("RGB;16" if channels == 3 else "RGBA;16", (32, 32), (samples << 6).tobytes())
        output = BytesIO()
        external.save(output, bit_depth=10, quality=95)
        expected = np.asarray(pillow_heif.open_heif(output.getvalue(), convert_hdr_to_8bit=False, hdr_to_16bit=False))
        result = BlanketImage.open(output)
        assert result.bit_depth == 10 and result.tobytes() == expected.astype("<u2").tobytes()
        checks += 1
    return checks


def check_avif_interop() -> int:
    checks = 0
    for mode, color in (("L", 120), ("RGB", (60, 120, 180)), ("RGBA", (60, 120, 180, 150))):
        source = PillowImage.new(mode, (19, 13), color)
        for writer in (source, BlanketImage.frombytes(mode, source.size, source.tobytes())):
            output = BytesIO()
            writer.save(output, "AVIF", quality=95)
            actual = BlanketImage.open(BytesIO(output.getvalue()))
            expected = PillowImage.open(BytesIO(output.getvalue())).convert("RGBA")
            assert (actual.mode, actual.size, actual.format) == (expected.mode, expected.size, "AVIF")
            assert max(abs(a - b) for a, b in zip(actual.tobytes(), expected.tobytes(), strict=False)) <= 3
            checks += 1
    return checks


def check_compositing() -> int:
    checks = 0
    for mode in ("L", "RGB", "RGBA", "HSV"):
        actual = BlanketImage.new(mode, (37, 29), "navy")
        expected = PillowImage.new(mode, actual.size, "navy")
        assert actual.tobytes() == expected.tobytes()
        checks += 1
        raw = pixels(mode)
        source = BlanketImage.frombytes(mode, actual.size, raw)
        reference = PillowImage.frombytes(mode, actual.size, raw)
        mask = BlanketImage.frombytes("L", actual.size, pixels("L"))
        pmask = PillowImage.frombytes("L", actual.size, pixels("L"))
        for alpha in (-0.5, 0, 0.37, 1, 1.5):
            compare_images(BlanketImage.blend(actual, source, alpha), PillowImage.blend(expected, reference, alpha), (mode, "blend", alpha))
            checks += 1
        compare_images(BlanketImage.composite(actual, source, mask), PillowImage.composite(expected, reference, pmask), (mode, "composite"))
        checks += 1
        actual.paste(source, (-3, 2), mask)
        expected.paste(reference, (-3, 2), pmask)
        assert actual.tobytes() == expected.tobytes()
        checks += 1
        assert BlanketImage.merge(mode, actual.split()).tobytes() == PillowImage.merge(mode, expected.split()).tobytes()
        checks += 1
        if mode in ("RGB", "RGBA"):
            actual.putalpha(mask)
            expected.putalpha(pmask)
            assert actual.tobytes() == expected.tobytes()
            checks += 1
        if mode == "RGBA":
            assert BlanketImage.alpha_composite(actual, source).tobytes() == PillowImage.alpha_composite(expected, reference).tobytes()
            actual.alpha_composite(source, (-2, 3), (1, 2, 30, 20))
            expected.alpha_composite(reference, (-2, 3), (1, 2, 30, 20))
            assert actual.tobytes() == expected.tobytes()
            checks += 2
    return checks


def check_imagechops() -> int:
    """Check native channel arithmetic, wraparound offsets, and helpers."""
    checks = 0
    for mode in ("L", "RGB", "RGBA", "HSV"):
        raw = pixels(mode)
        first = BlanketImage.frombytes(mode, (37, 29), raw)
        reference = PillowImage.frombytes(mode, first.size, raw)
        second = BlanketImage.frombytes(mode, first.size, raw[::-1])
        other = PillowImage.frombytes(mode, first.size, raw[::-1])
        mask = BlanketImage.frombytes("L", first.size, pixels("L"))
        reference_mask = PillowImage.frombytes("L", first.size, pixels("L"))
        for name in BlanketChops.__all__:
            if name.startswith("logical_"):
                continue
            if name in ("invert", "duplicate"):
                args, reference_args = (first,), (reference,)
            elif name in ("constant", "offset"):
                args, reference_args = (first, 73), (reference, 73)
            elif name == "blend":
                args, reference_args = (first, second, 0.37), (reference, other, 0.37)
            elif name == "composite":
                args, reference_args = (first, second, mask), (reference, other, reference_mask)
            elif name in ("add", "subtract"):
                args, reference_args = (first, second, 1.7, -13), (reference, other, 1.7, -13)
            else:
                args, reference_args = (first, second), (reference, other)
            actual = getattr(BlanketChops, name)(*args)
            expected = getattr(PillowChops, name)(*reference_args)
            assert (actual.mode, actual.size, actual.tobytes()) == (expected.mode, expected.size, expected.tobytes()), (mode, name)
            checks += 1
    first = BlanketImage.frombytes("1", (9, 1), b"\xa5\x80")
    second = BlanketImage.frombytes("1", (9, 1), b"\x3c\x00")
    reference = PillowImage.frombytes("1", first.size, first.tobytes())
    other = PillowImage.frombytes("1", second.size, second.tobytes())
    for name in ("logical_and", "logical_or", "logical_xor"):
        assert getattr(BlanketChops, name)(first, second).tobytes() == getattr(PillowChops, name)(reference, other).tobytes()
        checks += 1
    return checks


def check_imagestat() -> int:
    """Compare each native statistic for images, masks, and histograms."""
    checks = 0
    for mode in ("L", "RGB", "RGBA", "HSV"):
        image = BlanketImage.frombytes(mode, (37, 29), pixels(mode))
        reference = PillowImage.frombytes(mode, image.size, pixels(mode))
        mask = BlanketImage.frombytes("L", image.size, pixels("L"))
        reference_mask = PillowImage.frombytes("L", image.size, pixels("L"))
        pairs = [
            (BlanketStat.Stat(image), PillowStat.Stat(reference)),
            (BlanketStat.Stat(image, mask), PillowStat.Stat(reference, reference_mask)),
            (BlanketStat.Stat(image.histogram()), PillowStat.Stat(reference.histogram())),
        ]
        for actual, expected in pairs:
            for name in ("extrema", "count", "sum", "sum2", "mean", "median", "rms", "var", "stddev"):
                assert np.allclose(getattr(actual, name), getattr(expected, name), rtol=1e-14, atol=1e-12), (mode, name)
                checks += 1
    return checks


def check_imagedraw() -> int:
    checks = 0
    for mode, ink in (("L", 173), ("RGB", (37, 173, 91)), ("RGBA", (37, 173, 91, 83))):
        for width in (1, 2, 8):
            actual = BlanketImage.new(mode, (13, 11))
            expected = PillowImage.new(mode, actual.size)
            for module, image in ((BlanketDraw, actual), (PillowDraw, expected)):
                draw = module.Draw(image)
                draw.rectangle((-2, 1, 8, 7), fill=ink, outline=255, width=width)
                draw.point([(-1, 0), (3.9, 4.9), (12, 10)], fill=ink)
            assert actual.tobytes() == expected.tobytes(), (mode, width)
            checks += 1
    for mode, ink in (("L", 173), ("RGB", (37, 173, 91)), ("RGBA", (37, 173, 91, 83))):
        for name, args in (("line", ([(1, 2), (12, 9), (2, 9)],)), ("polygon", ([(2, 2), (12, 2), (12, 12), (2, 12)],)), ("ellipse", ((2, 2, 12, 12),))):
            actual, expected = BlanketImage.new(mode, (15, 15)), PillowImage.new(mode, (15, 15))
            getattr(BlanketDraw.Draw(actual), name)(*args, fill=ink)
            getattr(PillowDraw.Draw(expected), name)(*args, fill=ink)
            compare_images(actual, expected, (mode, name))
            checks += 1
        # Curves and wide strokes intentionally use different rasterization.
        # Compare representative interior/exterior pixels with Pillow.
        for name, args, options, inside in (
            ("arc", ((2, 2, 22, 22), 0, 180), {"width": 3}, (12, 22)),
            ("chord", ((2, 2, 22, 22), 0, 90), {}, (18, 18)),
            ("pieslice", ((2, 2, 22, 22), 0, 90), {}, (18, 18)),
            ("rounded_rectangle", ((2, 2, 22, 22),), {"radius": 5}, (12, 12)),
            ("regular_polygon", (((12, 12), 10), 5), {}, (12, 12)),
        ):
            actual, expected = BlanketImage.new(mode, (25, 25)), PillowImage.new(mode, (25, 25))
            getattr(BlanketDraw.Draw(actual), name)(*args, fill=ink, **options)
            getattr(PillowDraw.Draw(expected), name)(*args, fill=ink, **options)
            assert actual.getpixel(inside) == expected.getpixel(inside) == ink, (mode, name)
            assert actual.getpixel((0, 0)) == expected.getpixel((0, 0)), (mode, name)
            checks += 1
        font = PillowFont.load_default(size=13)
        for anchor in (None, "mm", "rs"):
            actual, expected = BlanketImage.new(mode, (100, 60)), PillowImage.new(mode, (100, 60))
            draw, reference = BlanketDraw.Draw(actual), PillowDraw.Draw(expected)
            draw.text((40, 20), "Hello!", font=font, fill=ink, anchor=anchor)
            reference.text((40, 20), "Hello!", font=font, fill=ink, anchor=anchor)
            compare_images(actual, expected, (mode, "text", anchor))
            assert draw.textlength("Hello!", font=font) == reference.textlength("Hello!", font=font)
            assert draw.textbbox((40, 20), "Hello!", font=font, anchor=anchor) == reference.textbbox((40, 20), "Hello!", font=font, anchor=anchor)
            checks += 3
            for align in ("left", "center", "right"):
                actual, expected = BlanketImage.new(mode, (100, 60)), PillowImage.new(mode, (100, 60))
                draw, reference = BlanketDraw.Draw(actual), PillowDraw.Draw(expected)
                options = {"font": font, "anchor": anchor, "align": align, "spacing": 6}
                draw.multiline_text((40, 25), "Hello\nWorld!", fill=ink, **options)
                reference.multiline_text((40, 25), "Hello\nWorld!", fill=ink, **options)
                compare_images(actual, expected, (mode, "multiline_text", anchor, align))
                assert draw.multiline_textbbox((40, 25), "Hello\nWorld!", **options) == reference.multiline_textbbox((40, 25), "Hello\nWorld!", **options)
                checks += 2
    for border in (None, 200):
        for threshold in (0, 3, 10):
            raw = bytes([200] * 7 + [200, 10, 11, 14, 19, 30, 200] * 5 + [200] * 7)
            actual, expected = BlanketImage.frombytes("L", (7, 7), raw), PillowImage.frombytes("L", (7, 7), raw)
            BlanketDraw.floodfill(actual, (2, 2), 100, border=border, thresh=threshold)
            PillowDraw.floodfill(expected, (2, 2), 100, border=border, thresh=threshold)
            compare_images(actual, expected, ("floodfill", border, threshold))
            checks += 1
    return checks


def check_imagefont() -> int:
    """Check native font contracts; outline coverage differs from FreeType."""
    checks = 0
    data = PillowFont.load_default(size=13).font_bytes
    font = BlanketFont.truetype(BytesIO(data), 13)
    reference = PillowFont.truetype(BytesIO(data), 13)
    assert font.getname() == reference.getname()
    assert font.getmetrics() == reference.getmetrics()
    checks += 2
    for actual in (BlanketFont.load_default(), BlanketFont.load_default(size=14), font, font.font_variant(size=26)):
        for text in ("", "Hello!", "AV fi"):
            for anchor in ("la", "mm", "rs"):
                box = actual.getbbox(text, anchor=anchor)
                mask, offset = actual.getmask2(text, anchor=anchor)
                assert offset == box[:2]
                assert mask.size == (box[2] - box[0], box[3] - box[1])
                assert len(bytes(mask)) == mask.size[0] * mask.size[1]
                assert bytes(actual.getmask(text, anchor=anchor)) == bytes(mask)
                image = BlanketImage.new("L", (100, 60))
                draw = BlanketDraw.Draw(image)
                assert draw.textlength(text, font=actual) == actual.getlength(text)
                draw.text((40, 25), text, font=actual, fill=255, anchor=anchor)
                assert bool(image.getbbox()) == bool(text)
                checks += 1
    with TemporaryDirectory() as directory:
        path = Path(directory) / "font.pil"
        compiled = FontFile()
        atlas = PillowImage.frombytes("1", (3, 2), b"\xa0\xe0")
        compiled.glyph[ord("A")] = ((4, 0), (0, 0, 3, 2), (0, 0, 3, 2), atlas)
        compiled.save(path)
        actual, expected = BlanketFont.load(path), PillowFont.load(path)
        assert actual.getlength("AA") == expected.getlength("AA")
        assert actual.getbbox("AA") == expected.getbbox("AA")
        assert bytes(actual.getmask("AA")) == bytes(expected.getmask("AA"))
        checks += 3
    return checks


def check_save_compressors() -> int:
    """Use Pillow to decode optimized saves and measure their guarantees."""
    checks = 0
    for mode in ("L", "RGB", "RGBA"):
        image = BlanketImage.frombytes(mode, (37, 29), pixels(mode))
        original = image.tobytes()
        for format, options in (("PNG", {}), ("JPEG", {"quality": 85})):
            if mode == "RGBA" and format == "JPEG":
                continue
            baseline = BytesIO()
            image.save(baseline, format, **options)
            expected = np.asarray(PillowImage.open(BytesIO(baseline.getvalue())).convert("RGBA")).astype(np.float64)
            for compressor in (LosslessImageCompressor(effort=1), LossyImageCompressor(max_rmse=2, effort=1)):
                output = BytesIO()
                image.save(output, format, compressor=compressor, **options)
                assert compressor.effort == 1
                assert len(output.getvalue()) <= len(baseline.getvalue())
                actual = np.asarray(PillowImage.open(BytesIO(output.getvalue())).convert("RGBA")).astype(np.float64)
                assert np.array_equal(actual[..., 3], expected[..., 3])
                if isinstance(compressor, LosslessImageCompressor):
                    assert np.array_equal(actual, expected), (mode, format)
                else:
                    assert np.sqrt(np.mean((actual[..., :3] - expected[..., :3]) ** 2)) <= compressor.max_rmse, (mode, format)
                assert image.tobytes() == original
                checks += 1
    return checks


def check_image_utilities() -> int:
    checks = 0
    actual = BlanketImage.frombytes("L", (37, 29), pixels("L")).convert("1")
    expected = PillowImage.frombytes("1", actual.size, actual.tobytes())
    with PillowImage.open(BytesIO(actual.tobitmap("sample"))) as decoded:
        assert decoded.size == expected.size and decoded.tobytes() == expected.tobytes()
    checks += 1
    for mode in ("L", "RGB", "RGBA"):
        image = BlanketImage.frombytes(mode, (37, 29), pixels(mode))
        original = image.tobytes()
        compare_images(image.effect_spread(0), image.to_pillow(), (mode, "effect_spread identity"))
        # Randomized spread has no shared seed API; a constant image stays constant.
        constant = BlanketImage.new(mode, image.size, 73)
        compare_images(constant.effect_spread(5), constant.to_pillow(), (mode, "effect_spread constant"))
        assert image.tobytes() == original
        checks += 2
        image.load()
        assert image.getim() is image.im
        assert image.draft(None, (8, 8)) is None
        assert image.verify() is None
        checks += 3
    for name in features.get_supported():
        assert features.check(name)
        assert features.version(name) is None or isinstance(features.version(name), str)
        checks += 1
    for alias, name in (("jpeg", "jpg"), ("jpegxl", "jxl"), ("tiff", "libtiff"), ("heic", "heif")):
        assert features.check(alias) == features.check(name)
        assert features.version(alias) == features.version(name)
        checks += 1
    report = StringIO()
    features.blanketinfo(report)
    assert "Blanket " in report.getvalue() and "PNG: open, save" in report.getvalue()
    return checks + 1


def main() -> None:
    checks = check_conversions() + check_fromarray() + check_png_interop() + check_jpeg_interop() + check_jxl_roundtrip() + check_pillow_adapter() + check_crop_apis() + check_bands_statistics()
    imageops_checks = check_imageops()
    checks += check_lab()
    checks += check_heif_and_high_depth()
    checks += check_avif_interop()
    checks += check_bmp_gif_ico_interop()
    checks += check_compositing()
    checks += check_imagechops()
    checks += check_imagestat()
    checks += check_imagedraw()
    checks += check_image_access()
    checks += check_image_geometry()
    checks += check_indexed_images()
    checks += check_imagecolor()
    checks += check_imagepath()
    checks += check_imagemath()
    checks += check_imagefont()
    checks += check_save_compressors()
    checks += check_image_utilities()
    imageenhance_checks = check_imageenhance()
    imagepalette_checks = check_imagepalette()
    imagefilter_checks = check_imagefilter()
    print(
        f"behavior verification passed: {checks + imageops_checks + imageenhance_checks + imagepalette_checks + imagefilter_checks} checks "
        f"({imageops_checks} ImageOps, {imageenhance_checks} ImageEnhance, {imagepalette_checks} ImagePalette, {imagefilter_checks} ImageFilter)"
    )


if __name__ == "__main__":
    main()
