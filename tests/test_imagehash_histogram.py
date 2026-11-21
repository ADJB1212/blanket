"""Color and local-texture fingerprints, persistence, and crop similarity."""

from __future__ import annotations

from io import BytesIO

import numpy as np
import pytest
from helpers import image_buffer
from PIL import Image as PillowImage

from blanket import Image, ImageHash
from blanket._blanket import image_histogram_hash_native

HASHERS = [ImageHash.colorhash, ImageHash.crop_resistant_hash]


@pytest.mark.parametrize("color,expected", [("black", "5550733c80e85264"), ("white", "284ed2f3979f5163"), ("red", "5c8ee640f54f7e53"), ("lime", "bd641e482e0e96f5"), ("blue", "01cf686211d6481c")])
def test_color_stored_vectors(color: str, expected: str) -> None:
    assert str(ImageHash.colorhash(Image.new("RGB", (1, 1), color))) == expected


def test_crop_stored_vector() -> None:
    assert str(ImageHash.crop_resistant_hash(Image.new("L", (3, 3), 0))) == "e384735bbcbfdfa6"


@pytest.mark.parametrize("hasher", HASHERS)
@pytest.mark.parametrize("size", [2, 3, 8, 9, 64])
def test_serialization_and_hamming_distance(hasher, size: int) -> None:
    source = Image.frombytes("RGB", (17, 13), bytes((i * 37 + i // 7) % 256 for i in range(17 * 13 * 3)))
    result = hasher(source, size)
    other = hasher(Image.new("RGB", source.size, "red"), size)
    assert isinstance(result, ImageHash.ImageHash)
    assert len(result) == size**2
    assert len(str(result)) == (size**2 + 3) // 4
    assert ImageHash.hex_to_hash(str(result), size) == result
    assert result - other == other - result == (result.value ^ other.value).bit_count()
    assert result - result == 0
    assert {result: "found"}[ImageHash.hex_to_hash(str(result), size)] == "found"
    with pytest.raises(ValueError, match="sizes"):
        _ = result - hasher(source, 4 if size == 8 else 8)


def test_color_tracks_color_and_ignores_layout_and_proportional_replication() -> None:
    pixels = np.random.default_rng(42).integers(0, 256, (19, 23, 3), dtype=np.uint8)
    source = Image.frombytes("RGB", (23, 19), pixels.tobytes())
    shuffled = pixels.reshape(-1, 3).copy()
    np.random.default_rng(21).shuffle(shuffled)
    replicated = np.repeat(np.repeat(pixels, 2, axis=0), 2, axis=1)
    assert ImageHash.colorhash(source) == ImageHash.colorhash(Image.frombytes("RGB", source.size, shuffled.tobytes()))
    assert ImageHash.colorhash(source) == ImageHash.colorhash(Image.frombytes("RGB", (46, 38), replicated.tobytes()))
    red = ImageHash.colorhash(Image.new("RGB", (1, 1), "red"))
    green = ImageHash.colorhash(Image.new("RGB", (1, 1), "lime"))
    blue = ImageHash.colorhash(Image.new("RGB", (1, 1), "blue"))
    assert red - green >= 16
    assert red - blue >= 16
    assert green - blue >= 16


def test_color_quantization_boundaries() -> None:
    assert ImageHash.colorhash(Image.new("RGB", (1, 1), (0, 64, 128))) == ImageHash.colorhash(Image.new("RGB", (1, 1), (63, 127, 191)))
    assert ImageHash.colorhash(Image.new("RGB", (1, 1), (64, 64, 128))) != ImageHash.colorhash(Image.new("RGB", (1, 1), (63, 64, 128)))


@pytest.mark.parametrize("repeat", [1, 2, 3, 4, 6])
def test_color_projection_ties_survive_proportional_replication(repeat: int) -> None:
    raw = bytes((0, 0, 0)) + bytes((0, 0, 64)) * 4 + bytes((0, 0, 128)) * 9
    image = Image.frombytes("RGB", (14 * repeat, 1), raw * repeat)
    assert str(ImageHash.colorhash(image)) == "94000792288d4050"


@pytest.mark.parametrize("box", [(0, 0, 96, 96), (32, 16, 128, 112), (40, 40, 88, 88)])
def test_crop_similarity_and_texture_discrimination(box: tuple[int, int, int, int]) -> None:
    y, x = np.indices((128, 128))
    pixels = np.where((x // 4 + y // 7) % 2, 40, 200).astype(np.uint8)
    source = Image.frombytes("L", (128, 128), pixels.tobytes())
    fingerprint = ImageHash.crop_resistant_hash(source)
    cropped = ImageHash.crop_resistant_hash(source.crop(box))
    unrelated = ImageHash.crop_resistant_hash(Image.new("L", source.size, 40))
    assert fingerprint - cropped <= 8
    assert fingerprint - cropped < fingerprint - unrelated


def test_crop_hash_is_invariant_to_strictly_increasing_intensity_mapping() -> None:
    pixels = np.random.default_rng(17).integers(0, 64, (41, 53), dtype=np.uint8)
    source = Image.frombytes("L", (53, 41), pixels.tobytes())
    shifted = Image.frombytes("L", source.size, (pixels * 3 + 12).tobytes())
    assert ImageHash.crop_resistant_hash(source) == ImageHash.crop_resistant_hash(shifted)
    assert ImageHash.crop_resistant_hash(Image.new("L", (3, 3), 0)) == ImageHash.crop_resistant_hash(Image.new("L", (13, 17), 255))


@pytest.mark.parametrize("hasher", HASHERS)
@pytest.mark.parametrize("mode", ["1", "L", "LA", "P", "PA", "RGB", "RGBA", "HSV", "CMYK", "YCbCr", "LAB", "I", "F", "I;16", "I;16L", "I;16B", "I;16N", "RGB;F", "RGBA;F"])
def test_modes_use_explicit_eight_bit_conversion(hasher, mode: str) -> None:
    source = Image.frombytes("RGB", (17, 13), bytes((i * 37 + i // 7) % 256 for i in range(17 * 13 * 3))).convert(mode)
    before = source.tobytes()
    target = "RGB" if hasher is ImageHash.colorhash else "L"
    expected_source = source.convert("RGB", bit_depth=8) if mode in ("RGB;F", "RGBA;F") else source
    assert hasher(source) == hasher(expected_source.convert(target, bit_depth=8))
    assert source.tobytes() == before


@pytest.mark.parametrize("hasher", HASHERS)
@pytest.mark.parametrize("depth", [10, 12, 16])
def test_high_depth_normalization(hasher, depth: int) -> None:
    source = Image.frombytes("RGB", (17, 13), bytes(i % 256 for i in range(17 * 13 * 3))).convert("RGB", bit_depth=depth)
    assert hasher(source) == hasher(source.convert("RGB", bit_depth=8))


@pytest.mark.parametrize("hasher", HASHERS)
def test_alpha_is_discarded(hasher) -> None:
    pixels = np.random.default_rng(9).integers(0, 256, (19, 23, 4), dtype=np.uint8)
    transparent = Image.frombytes("RGBA", (23, 19), pixels.tobytes())
    pixels[:, :, 3] = 255
    assert hasher(transparent) == hasher(Image.frombytes("RGBA", (23, 19), pixels.tobytes()))


@pytest.mark.parametrize("hasher", HASHERS)
def test_current_frame_source_state_and_lifetime(hasher) -> None:
    pixels = bytes((i * 37 + i // 7) % 256 for i in range(17 * 13 * 3))
    stream = image_buffer(PillowImage.new("RGB", (17, 13), "red"), "TIFF", save_all=True, append_images=[PillowImage.frombytes("RGB", (17, 13), pixels)])
    with Image.open(stream) as source:
        source.seek(1)
        source.info["custom"] = "retained"
        before = source.info.copy(), source.tobytes()
        result = hasher(source)
        assert result == hasher(Image.frombytes("RGB", source.size, pixels))
        assert source.tell() == 1
        assert (source.info, source.tobytes()) == before
    assert ImageHash.hex_to_hash(str(result)) == result
    source.close()
    with pytest.raises(ValueError, match="closed"):
        hasher(source)


@pytest.mark.parametrize("hasher", HASHERS)
@pytest.mark.parametrize("size", [0, 1, -1, 65, True, 8.0, "8"])
def test_invalid_hash_size(hasher, size) -> None:
    with pytest.raises(TypeError if isinstance(size, (bool, float, str)) else ValueError):
        hasher(Image.new("L", (3, 3)), size)


@pytest.mark.parametrize("hasher", HASHERS)
def test_invalid_images(hasher) -> None:
    with pytest.raises(TypeError, match="Blanket Image"):
        hasher(None)
    with pytest.raises(ValueError, match="empty"):
        hasher(Image.new("L", (0, 3)))


@pytest.mark.parametrize("size", [(1, 1), (2, 5), (5, 2)])
def test_crop_hash_requires_neighborhoods(size: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match="3x3"):
        ImageHash.crop_resistant_hash(Image.new("L", size))


@pytest.mark.parametrize(
    "mode,depth,size,hash_size,algorithm",
    [
        ("RGB", 8, (3, 3), 0, "color"),
        ("RGB", 8, (3, 3), 65, "color"),
        ("L", 8, (3, 3), 8, "color"),
        ("RGB", 8, (3, 3), 8, "crop-resistant"),
        ("L", 8, (2, 3), 8, "crop-resistant"),
        ("RGB", 16, (3, 3), 8, "color"),
        ("RGB", 8, (0, 3), 8, "color"),
        ("L", 8, (3, 3), 8, "unknown"),
    ],
)
def test_native_validation(mode: str, depth: int, size: tuple[int, int], hash_size: int, algorithm: str) -> None:
    source = Image.new(mode, size).convert(mode, bit_depth=depth)
    with pytest.raises(ValueError):
        image_histogram_hash_native(source._native, hash_size, algorithm)
