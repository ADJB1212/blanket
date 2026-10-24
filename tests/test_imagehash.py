from __future__ import annotations

from dataclasses import FrozenInstanceError
from io import BytesIO

import numpy as np
import pytest
from PIL import Image as PillowImage

from blanket import Image, ImageHash

HASHERS = [ImageHash.average_hash, ImageHash.dhash, ImageHash.phash]


def bit_value(bits: np.ndarray) -> int:
    return int("".join("1" if bit else "0" for bit in bits.flat), 2)


@pytest.mark.parametrize("size", [2, 3, 8, 9, 16, 64])
@pytest.mark.parametrize("algorithm", ["average", "difference", "perceptual"])
def test_hashes_match_independent_reference(size: int, algorithm: str) -> None:
    factor = 2
    width, height = (size + 1, size) if algorithm == "difference" else (size * factor, size * factor) if algorithm == "perceptual" else (size, size)
    pixels = np.random.default_rng(42).integers(0, 256, (height, width), dtype=np.uint8)
    image = Image.frombytes("L", (width, height), pixels.tobytes())
    if algorithm == "average":
        expected = pixels > pixels.mean()
        actual = ImageHash.average_hash(image, size)
    elif algorithm == "difference":
        expected = pixels[:, 1:] > pixels[:, :-1]
        actual = ImageHash.dhash(image, size)
    else:
        basis = np.cos(np.pi * np.arange(size)[:, None] * (2 * np.arange(width) + 1) / (2 * width))
        coefficients = basis @ pixels.astype(float) @ basis.T
        expected = coefficients > np.median(coefficients)
        actual = ImageHash.phash(image, size, factor)
    assert actual.value == bit_value(expected)
    assert len(actual) == size**2
    assert ImageHash.hex_to_hash(str(actual)) == actual


@pytest.mark.parametrize("hasher", HASHERS)
@pytest.mark.parametrize("value", [0, 1, 127, 255])
def test_constant_hashes_and_ties(hasher, value: int) -> None:
    image = Image.new("L", (33, 27), value)
    expected = 1 << 63 if hasher is ImageHash.phash and value else 0
    assert hasher(image).value == expected


@pytest.mark.parametrize("hasher", HASHERS)
def test_grayscale_resize_matches_pillow_preprocessing(hasher) -> None:
    pixels = np.random.default_rng(12).integers(0, 256, (39, 43, 3), dtype=np.uint8)
    source = Image.frombytes("RGB", (43, 39), pixels.tobytes())
    pillow = PillowImage.frombytes("RGB", source.size, pixels.tobytes()).convert("L")
    size = (9, 8) if hasher is ImageHash.dhash else (32, 32) if hasher is ImageHash.phash else (8, 8)
    thumbnail = pillow.resize(size, PillowImage.Resampling.LANCZOS)
    assert hasher(source) == hasher(Image.frombytes("L", size, thumbnail.tobytes()))


@pytest.mark.parametrize("hasher", HASHERS)
def test_uniform_brightness_shift_preserves_visual_fingerprint(hasher) -> None:
    pixels = np.random.default_rng(8).integers(32, 96, (48, 64), dtype=np.uint8)
    source = Image.frombytes("L", (64, 48), pixels.tobytes())
    brighter = Image.frombytes("L", (64, 48), (pixels + 40).tobytes())
    assert source.tobytes() != brighter.tobytes()
    assert hasher(source) - hasher(brighter) == 0


@pytest.mark.parametrize("hasher", HASHERS)
def test_alpha_is_discarded(hasher) -> None:
    pixels = np.random.default_rng(9).integers(0, 256, (19, 23, 4), dtype=np.uint8)
    transparent = Image.frombytes("RGBA", (23, 19), pixels.tobytes())
    pixels[:, :, 3] = 255
    opaque = Image.frombytes("RGBA", (23, 19), pixels.tobytes())
    assert hasher(transparent) == hasher(opaque)


@pytest.mark.parametrize("hasher", HASHERS)
@pytest.mark.parametrize("mode", ["1", "L", "LA", "P", "PA", "RGB", "RGBA", "HSV", "CMYK", "YCbCr", "LAB", "I", "F", "I;16", "I;16L", "I;16B"])
def test_modes_use_explicit_eight_bit_grayscale(hasher, mode: str) -> None:
    source = Image.frombytes("RGB", (17, 13), bytes((i * 37 + i // 7) % 256 for i in range(17 * 13 * 3))).convert(mode)
    before = source.tobytes()
    assert hasher(source) == hasher(source.convert("L", bit_depth=8))
    assert source.tobytes() == before


@pytest.mark.parametrize("depth", [10, 12, 16])
@pytest.mark.parametrize("hasher", HASHERS)
def test_high_depth_is_normalized_to_eight_bits(hasher, depth: int) -> None:
    image = Image.frombytes("RGB", (17, 13), bytes(i % 256 for i in range(17 * 13 * 3))).convert("RGB", bit_depth=depth)
    assert hasher(image) == hasher(image.convert("L", bit_depth=8))


@pytest.mark.parametrize("hasher", HASHERS)
def test_current_frame_and_source_state(hasher) -> None:
    frames = [Image.new("L", (9, 8), 0), Image.frombytes("L", (9, 8), bytes(range(72)))]
    stream = BytesIO()
    frames[0].save(stream, "PNG", save_all=True, append_images=frames[1:])
    image = Image.open(stream)
    image.seek(1)
    image.info["custom"] = "keep"
    before = image.tobytes(), image.info.copy()
    assert hasher(image) == hasher(frames[1])
    assert hasher(image) != hasher(frames[0])
    assert image.tell() == 1
    assert (image.tobytes(), image.info) == before
    image.close()
    with pytest.raises(ValueError, match="closed"):
        hasher(image)


def test_hash_values_serialization_distance_and_keys() -> None:
    first = ImageHash.ImageHash(0x0102)
    second = ImageHash.hex_to_hash("000000000000010F")
    assert str(first) == "0000000000000102"
    assert first - second == second - first == 3
    assert first - first == 0
    assert {first: "found"}[ImageHash.hex_to_hash(str(first))] == "found"
    assert first != 0x0102
    assert first != ImageHash.ImageHash(first.value, 9)
    with pytest.raises(FrozenInstanceError):
        first.value = 0
    with pytest.raises(ValueError, match="sizes"):
        _ = first - ImageHash.ImageHash(0, 9)
    with pytest.raises(TypeError):
        _ = first - 1
    assert str(ImageHash.ImageHash(1 << 8, 3)) == "100"
    assert ImageHash.hex_to_hash("100", 3) == ImageHash.ImageHash(256, 3)


@pytest.mark.parametrize("text", ["", "xyz", " 00", "+00", "0x00", "0_0", "00000", "fff"])
def test_invalid_hex(text: str) -> None:
    with pytest.raises(ValueError):
        ImageHash.hex_to_hash(text)


@pytest.mark.parametrize("size", [0, 1, -1, 65, True, 8.0, "8"])
@pytest.mark.parametrize("hasher", HASHERS)
def test_invalid_hash_sizes(hasher, size) -> None:
    with pytest.raises(TypeError if isinstance(size, (bool, float, str)) else ValueError):
        hasher(Image.new("L", (2, 2)), size)


@pytest.mark.parametrize("factor", [0, -1, 33, True, 1.5, "4"])
def test_invalid_phash_factor(factor) -> None:
    with pytest.raises(TypeError if isinstance(factor, (bool, float, str)) else ValueError):
        ImageHash.phash(Image.new("L", (2, 2)), highfreq_factor=factor)


@pytest.mark.parametrize("value", [-1, 1 << 64, True, 1.5])
def test_invalid_hash_value(value) -> None:
    with pytest.raises(TypeError if isinstance(value, (bool, float)) else ValueError):
        ImageHash.ImageHash(value)


@pytest.mark.parametrize("hasher", HASHERS)
def test_empty_images_and_wrong_types(hasher) -> None:
    with pytest.raises(ValueError, match="empty"):
        hasher(Image.new("L", (0, 2)))
    with pytest.raises(TypeError, match="Blanket Image"):
        hasher(None)
