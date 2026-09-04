#!/usr/bin/env python3
"""Differential and cross-codec checks against Pillow 12.2."""

from __future__ import annotations

from io import BytesIO

from PIL import Image as PillowImage

from blanket import Image as BlanketImage


def pixels(mode: str, width: int = 37, height: int = 29) -> bytes:
    channels = {"L": 1, "RGB": 3, "RGBA": 4}[mode]
    return bytes(
        (x * 17 + y * 29 + channel * 53) % 256
        for y in range(height)
        for x in range(width)
        for channel in range(channels)
    )


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
            PillowImage.frombytes("RGB", (37, 29), raw).save(
                stream, "JPEG", quality=85
            )
        else:
            BlanketImage.frombytes("RGB", (37, 29), raw).save(
                stream, "JPEG", quality=85
            )

        payload = stream.getvalue()
        with PillowImage.open(BytesIO(payload)) as pillow_loaded:
            pillow_loaded.load()
            blanket_loaded = BlanketImage.open(BytesIO(payload))
            assert blanket_loaded.size == pillow_loaded.size == (37, 29)
            assert blanket_loaded.mode == pillow_loaded.mode == "RGB"
            differences = [
                abs(left - right)
                for left, right in zip(
                    blanket_loaded.tobytes(), pillow_loaded.tobytes(), strict=True
                )
            ]
            assert sum(differences) / len(differences) <= 1.0
            assert max(differences) <= 4
    return 2


def check_jxl_roundtrip() -> int:
    checks = 0
    for mode in ("L", "RGB", "RGBA"):
        raw = pixels(mode, 16, 12)
        stream = BytesIO()
        BlanketImage.frombytes(mode, (16, 12), raw).save(
            stream, "JXL", lossless=True, effort=1
        )
        loaded = BlanketImage.open(stream)
        assert (loaded.format, loaded.mode, loaded.size) == ("JXL", mode, (16, 12))
        assert loaded.tobytes() == raw
        checks += 1
    return checks


def main() -> None:
    checks = (
        check_conversions()
        + check_png_interop()
        + check_jpeg_interop()
        + check_jxl_roundtrip()
    )
    print(f"behavior verification passed: {checks} checks")


if __name__ == "__main__":
    main()
