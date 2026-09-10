"""Differential Image, ImageOps, and cross-codec checks against Pillow."""

from __future__ import annotations

import random
from io import BytesIO, StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from blanket import Image as BlanketImage
from blanket import ImageEnhance as BlanketEnhance
from blanket import ImageFilter as BlanketFilter
from blanket import ImageOps as BlanketOps
from blanket import ImagePalette as BlanketPalette
from PIL import Image as PillowImage
from PIL import ImageEnhance as PillowEnhance
from PIL import ImageFilter
from PIL import ImageOps as PillowOps
from PIL import ImagePalette as PillowPalette


def pixels(mode: str, width: int = 37, height: int = 29) -> bytes:
    channels = {"L": 1, "RGB": 3, "RGBA": 4}[mode]
    return bytes((x * 17 + y * 29 + channel * 53) % 256 for y in range(height) for x in range(width) for channel in range(channels))


def check_conversions() -> int:
    checks = 0
    for source_mode in ("L", "RGB", "RGBA"):
        raw = pixels(source_mode)
        blanket = BlanketImage.frombytes(source_mode, (37, 29), raw)
        pillow = PillowImage.frombytes(source_mode, (37, 29), raw)
        for target_mode in ("L", "RGB", "RGBA"):
            actual = blanket.convert(target_mode)
            expected = pillow.convert(target_mode)
            assert actual.mode == expected.mode
            assert actual.size == expected.size
            assert actual.tobytes() == expected.tobytes()
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

    source = np.arange(12 * 16 * 3, dtype=np.uint8).reshape(12, 16, 3)
    strided = source[::2, ::2]
    assert BlanketImage.fromarray(strided).tobytes() == PillowImage.fromarray(strided).tobytes()
    return checks + 1


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
    for mode in ("L", "RGB", "RGBA"):
        raw = pixels(mode)
        blanket = BlanketImage.frombytes(mode, (37, 29), raw)
        pillow = PillowImage.frombytes(mode, (37, 29), raw)
        for actual, expected in zip(blanket.split(), pillow.split()):
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
        # Generated palettes need not choose Pillow's exact colors/order.
        # Check the public contract and compare decoding through Pillow.
        for method in ((2,) if mode == "RGBA" else (0, 1, 2)):
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
    for mode in ("L", "RGB", "RGBA"):
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

    for mode in ("L", "RGB", "RGBA"):
        raw = pixels(mode)
        blanket = BlanketImage.frombytes(mode, (37, 29), raw)
        pillow = PillowImage.frombytes(mode, (37, 29), raw)
        cases = [("crop", {"border": (1, 2, 3, 4)}), ("expand", {"border": (2, 3), "fill": "rebeccapurple"}), ("flip", {}), ("mirror", {}), ("grayscale", {})]
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
            for name, options in (("contain", {}), ("cover", {}), ("fit", {"bleed": 0.05, "centering": (0.2, 0.8)}), ("pad", {"color": "#1234", "centering": (0, 1)})):
                compare(getattr(BlanketOps, name)(blanket, (19, 17), method, **options), getattr(PillowOps, name)(pillow, (19, 17), method, **options), f"{name} {mode} {method.name}")
            compare(BlanketOps.scale(blanket, 1.5, method), PillowOps.scale(pillow, 1.5, method), f"scale {mode} {method.name}")
        for method in (BlanketImage.NEAREST, BlanketImage.BILINEAR, BlanketImage.BICUBIC):
            compare(BlanketOps.deform(blanket, MirrorMesh(), method), PillowOps.deform(pillow, MirrorMesh(), method), f"deform {mode} {method.name}")

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
    for mode in ("L", "RGB", "RGBA"):
        raw = pixels(mode)
        blanket = BlanketImage.frombytes(mode, (37, 29), raw)
        pillow = PillowImage.frombytes(mode, (37, 29), raw)
        for name in ("Color", "Contrast", "Brightness", "Sharpness"):
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
    for mode in ("L", "RGB", "RGBA"):
        for factory in ("wedge", "negative"):
            actual = getattr(BlanketPalette, factory)(mode)
            expected = getattr(PillowPalette, factory)(mode)
            assert actual.getdata() == expected.getdata()
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
    for mode in ("L", "RGB", "RGBA"):
        raw = pixels(mode)
        actual = BlanketImage.frombytes(mode, (37, 29), raw)
        expected = PillowImage.frombytes(mode, (37, 29), raw)
        for name, args in specifications:
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


def main() -> None:
    checks = check_conversions() + check_fromarray() + check_png_interop() + check_jpeg_interop() + check_jxl_roundtrip() + check_pillow_adapter() + check_crop_apis() + check_bands_statistics()
    imageops_checks = check_imageops()
    checks += check_heif_and_high_depth()
    imageenhance_checks = check_imageenhance()
    imagepalette_checks = check_imagepalette()
    imagefilter_checks = check_imagefilter()
    print(
        f"behavior verification passed: {checks + imageops_checks + imageenhance_checks + imagepalette_checks + imagefilter_checks} checks "
        f"({imageops_checks} ImageOps, {imageenhance_checks} ImageEnhance, {imagepalette_checks} ImagePalette, {imagefilter_checks} ImageFilter)"
    )


if __name__ == "__main__":
    main()
