"""Differential Image, ImageOps, and cross-codec checks against Pillow."""

from __future__ import annotations

from io import BytesIO

import numpy as np
from blanket import Image as BlanketImage
from blanket import ImageOps as BlanketOps
from PIL import Image as PillowImage
from PIL import ImageOps as PillowOps


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


class MirrorMesh:
    """Exercise the deformer protocol with a horizontal reflection."""

    def getmesh(self, image: BlanketImage.Image | PillowImage.Image) -> list[tuple[tuple[int, int, int, int], tuple[int, ...]]]:
        w, h = image.size
        return [((0, 0, w, h), (w, 0, w, h, 0, h, 0, 0))]


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


def main() -> None:
    checks = check_conversions() + check_fromarray() + check_png_interop() + check_jpeg_interop() + check_jxl_roundtrip()
    imageops_checks = check_imageops()
    print(f"behavior verification passed: {checks + imageops_checks} checks ({imageops_checks} ImageOps)")


if __name__ == "__main__":
    main()
