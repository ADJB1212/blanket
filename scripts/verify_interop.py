"""Exercise unsupported image operations through Blanket's Pillow adapter."""

from __future__ import annotations

from io import BytesIO

from blanket import Image as BlanketImage
from PIL import Image as PillowImage
from PIL import ImageFilter, ImageOps


def main() -> None:
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
    processed = ImageOps.autocontrast(filtered)
    assert processed.size == (16, 12)

    returned = BlanketImage.frombytes(processed.mode, processed.size, processed.tobytes())
    output = BytesIO()
    returned.save(output, "PNG")
    reloaded = BlanketImage.open(output)
    assert reloaded.tobytes() == processed.tobytes()
    print("Pillow interoperability passed: merge, resize, rotate, filter, ImageOps")


if __name__ == "__main__":
    main()
