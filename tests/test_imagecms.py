from __future__ import annotations

import os
import struct
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from typing import TYPE_CHECKING

import pytest
from helpers import assert_image_properties, assert_raises, image_buffer
from PIL import Image as PillowImage, ImageCms as PillowCms

from blanket import Image, ImageCms

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("space", ["sRGB", "Adobe RGB", "Display P3", "LAB", "XYZ"])
def test_profile_creation_serialization_and_metadata(space: str, tmp_path: Path) -> None:
    profile = ImageCms.createProfile(space)
    data = profile.tobytes()
    pillow = PillowCms.ImageCmsProfile(BytesIO(data))
    path = tmp_path / "profile.icc"
    path.write_bytes(data)
    for source in (profile, BytesIO(data), path, str(path), os.fsencode(path)):
        reopened = ImageCms.getOpenProfile(source)
        assert reopened.tobytes() == data
        assert ImageCms.getProfileDescription(reopened) == PillowCms.getProfileDescription(pillow)
        assert ImageCms.getProfileName(reopened) == PillowCms.getProfileName(pillow)
        assert ImageCms.getProfileInfo(reopened) == PillowCms.getProfileInfo(pillow)
    if space in ("Adobe RGB", "Display P3"):
        assert space in ImageCms.getProfileDescription(profile)


@pytest.mark.parametrize("intent", list(ImageCms.Intent))
@pytest.mark.parametrize(("source_mode", "target_mode"), [("RGB", "RGB"), ("RGB", "LAB"), ("LAB", "RGB"), ("RGBA", "RGBA"), ("RGBA", "RGB"), ("RGB", "RGBA")])
def test_transform_matches_pillow(source_mode: str, target_mode: str, intent: int) -> None:
    source_profile = ImageCms.createProfile("LAB" if source_mode == "LAB" else "Display P3")
    target_profile = ImageCms.createProfile("LAB" if target_mode == "LAB" else "sRGB")
    channels = len(source_mode)
    raw = bytes((i * 37 + 19) % 256 for i in range(37 * 3 * channels))
    image = Image.frombytes(source_mode, (37, 3), raw)
    image.info["custom"] = "preserved"
    pillow = PillowImage.frombytes(source_mode, image.size, raw)
    expected = PillowCms.profileToProfile(pillow, PillowCms.ImageCmsProfile(BytesIO(source_profile.tobytes())), PillowCms.ImageCmsProfile(BytesIO(target_profile.tobytes())), intent, target_mode)
    transform = ImageCms.buildTransform(source_profile, target_profile, source_mode, target_mode, intent)
    for _ in range(2):
        actual = ImageCms.applyTransform(image, transform)
        assert actual is not None
        assert_image_properties(actual, pixels=expected.tobytes(), mode=target_mode)
        assert actual.info["icc_profile"] == target_profile.tobytes()
        assert actual.info["custom"] == "preserved"
    assert image.tobytes() == raw


@pytest.mark.parametrize("intent", list(ImageCms.Intent))
def test_proof_transform_matches_pillow(intent: int) -> None:
    source, output, proof = (ImageCms.createProfile(space) for space in ("Display P3", "sRGB", "Adobe RGB"))
    image = Image.new("RGB", (3, 1))
    image.putdata([(80, 140, 220), (250, 15, 80), (35, 200, 25)])
    flags = ImageCms.Flags.SOFTPROOFING | ImageCms.Flags.BLACKPOINTCOMPENSATION
    transform = ImageCms.buildProofTransform(source, output, proof, "RGB", "RGB", intent, intent, flags)
    profiles = [PillowCms.ImageCmsProfile(BytesIO(profile.tobytes())) for profile in (source, output, proof)]
    expected_transform = PillowCms.buildProofTransform(*profiles, "RGB", "RGB", intent, intent, flags)
    expected = PillowCms.applyTransform(PillowImage.frombytes("RGB", image.size, image.tobytes()), expected_transform)
    actual = ImageCms.applyTransform(image, transform)
    assert actual is not None
    assert actual.tobytes() == expected.tobytes()


def test_in_place_and_embedded_output_profile() -> None:
    source, output = ImageCms.createProfile("Display P3"), ImageCms.createProfile("sRGB")
    image = Image.new("RGBA", (3, 1))
    image.putdata([(80, 140, 220, 0), (250, 15, 80, 127), (35, 200, 25, 255)])
    original = image.tobytes()
    expected = ImageCms.profileToProfile(image, source, output)
    assert expected is not None
    core = image.getim()
    assert ImageCms.profileToProfile(image, source, output, inPlace=True) is None
    assert image.getim() is core
    assert image.tobytes() == expected.tobytes()
    assert image.tobytes() != original
    assert image.getchannel("A").tobytes() == bytes([0, 127, 255])
    stream = image_buffer(image, "PNG")
    reopened = Image.open(BytesIO(stream.getvalue()))
    assert reopened.info["icc_profile"] == output.tobytes()


@pytest.mark.parametrize("depth", [8, 16])
def test_transform_only_changes_current_frame(depth) -> None:
    first_image = Image.new("RGB", (2, 1), (80, 140, 220)).convert("RGB", bit_depth=depth)
    second_image = Image.new("RGB", (2, 1), (35, 200, 25)).convert("RGB", bit_depth=depth)
    stream = image_buffer(first_image, "PNG", save_all=True, append_images=[second_image])
    image = Image.open(BytesIO(stream.getvalue()))
    first = image.tobytes()
    format = image.format
    image.seek(1)
    source, output = ImageCms.createProfile("Display P3"), ImageCms.createProfile("sRGB")
    expected = ImageCms.profileToProfile(image, source, output)
    assert expected is not None
    ImageCms.profileToProfile(image, source, output, inPlace=True)
    assert image.format == format
    assert image.tell() == 1
    image.seek(0)
    assert image.tobytes() == first
    image.seek(1)
    assert image.tobytes() == expected.tobytes()
    assert image.info["icc_profile"] == output.tobytes()


@pytest.mark.parametrize("mode", ["1", "LA", "P", "PA", "I", "I;16", "I;16L", "I;16B", "F", "HSV", "YCbCr"])
def test_unsupported_modes_are_rejected(mode: str) -> None:
    profile = ImageCms.createProfile("sRGB")
    assert_raises(ImageCms.PyCMSError, lambda: ImageCms.profileToProfile(Image.new(mode, (1, 1)), profile, profile))


def test_invalid_arguments_leave_pixels_unchanged(tmp_path: Path) -> None:
    profile = ImageCms.createProfile("sRGB")
    image = Image.new("RGB", (2, 1), (80, 140, 220))
    raw = image.tobytes()
    for intent in (-1, 4, 1.5, "perceptual"):
        assert_raises(ImageCms.PyCMSError, lambda: ImageCms.profileToProfile(image, profile, profile, intent, inPlace=True))  # type: ignore[arg-type]
    for flags in (-1, 2**32, 0.5):
        assert_raises(ImageCms.PyCMSError, lambda: ImageCms.buildTransform(profile, profile, "RGB", "RGB", flags=flags))  # type: ignore[arg-type]
    assert_raises(ImageCms.PyCMSError, lambda: ImageCms.ImageCmsProfile(BytesIO(b"invalid ICC")))
    assert_raises(ImageCms.PyCMSError, lambda: ImageCms.getOpenProfile(tmp_path / "missing.icc"))
    assert_raises(ImageCms.PyCMSError, lambda: ImageCms.createProfile("unknown"))
    assert_raises(ImageCms.PyCMSError, lambda: ImageCms.profileToProfile(image, profile, ImageCms.createProfile("LAB"), outputMode="LAB", inPlace=True))
    transform = ImageCms.buildTransform(profile, profile, "RGB", "RGB")
    assert_raises(ImageCms.PyCMSError, lambda: ImageCms.applyTransform(Image.new("RGBA", (1, 1)), transform))
    assert image.tobytes() == raw
    image.close()
    assert_raises(ImageCms.PyCMSError, lambda: ImageCms.applyTransform(image, transform))
    assert_raises(ImageCms.PyCMSError, lambda: ImageCms.applyTransform(Image.new("RGB", (1, 1)).convert("RGB", bit_depth=12), transform))


@pytest.mark.parametrize("mode,depth", [("RGBA", 8), ("RGBA", 16), ("RGBA;F", 32)])
def test_reused_transform_across_threads(mode, depth) -> None:
    profile = ImageCms.createProfile("sRGB")
    transform = ImageCms.buildTransform(profile, profile, mode, mode)
    image = Image.new("RGBA", (37, 11), (10, 20, 30, 40)).convert(mode, bit_depth=16 if depth == 16 else None)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(transform.apply, [image] * 8))
    assert all(result.tobytes() == image.tobytes() for result in results)


@pytest.mark.parametrize(("space", "expected"), [("Display P3", (95, 128, 156)), ("Adobe RGB", (89, 129, 155))])
def test_created_rgb_profile_colorimetry(space: str, expected: tuple[int, int, int]) -> None:
    image = Image.new("RGB", (1, 1), (102, 128, 153))
    result = ImageCms.profileToProfile(image, ImageCms.createProfile(space), ImageCms.createProfile("sRGB"))
    assert result is not None
    actual = result.getpixel((0, 0))
    assert isinstance(actual, tuple)
    assert all(abs(a - b) <= 1 for a, b in zip(actual, expected, strict=True))


@pytest.mark.parametrize("temperature", [0, 5000, 6500])
def test_lab_white_temperature(temperature: int) -> None:
    profile = ImageCms.createProfile("LAB", temperature)
    pillow = PillowCms.ImageCmsProfile(PillowCms.createProfile("LAB", temperature))
    source = ImageCms.createProfile("sRGB")
    image = Image.new("RGB", (1, 1), (70, 130, 210))
    actual = ImageCms.profileToProfile(image, source, profile, outputMode="LAB")
    expected = PillowCms.profileToProfile(image.to_pillow(), PillowCms.ImageCmsProfile(BytesIO(source.tobytes())), pillow, outputMode="LAB")
    assert actual is not None
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("temperature", [-100, 3000, 30000, float("nan"), float("inf")])
def test_invalid_lab_temperature(temperature: float) -> None:
    assert_raises(ImageCms.PyCMSError, lambda: ImageCms.createProfile("LAB", temperature))


@pytest.mark.parametrize("flags", [ImageCms.Flags.NOCACHE, ImageCms.Flags.NOOPTIMIZE, ImageCms.Flags.COPY_ALPHA])
def test_flags_and_output_image(flags: int) -> None:
    profile = ImageCms.createProfile("sRGB")
    transform = ImageCms.buildTransformFromOpenProfiles(profile, profile, "RGBA", "RGBA", flags=flags)
    image = Image.new("RGBA", (7, 1), (40, 80, 120, 160))
    output = Image.new("RGBA", image.size)
    assert transform.apply(image, output) is output
    assert output.tobytes() == image.tobytes()
    assert_raises(ImageCms.PyCMSError, lambda: transform.apply(image, Image.new("RGBA", (1, 1))))
    assert_raises(ImageCms.PyCMSError, lambda: transform.apply(image, Image.new("RGB", image.size)))


@pytest.mark.parametrize("mode", ["L", "CMYK", "LAB"])
def test_profile_and_mode_must_match(mode: str) -> None:
    profile = ImageCms.createProfile("sRGB")
    assert_raises(ImageCms.PyCMSError, lambda: ImageCms.buildTransform(profile, profile, mode, "RGB"))
    assert_raises(ImageCms.PyCMSError, lambda: ImageCms.buildTransform(profile, profile, "RGB", mode))


def test_empty_image_and_truncated_profile() -> None:
    profile = ImageCms.createProfile("sRGB")
    for size in ((0, 0), (0, 3), (3, 0)):
        result = ImageCms.profileToProfile(Image.new("RGB", size), profile, profile)
        assert result is not None
        assert_image_properties(result, size=size, pixels=b"")
    for data in (profile.tobytes()[:64], profile.tobytes()[:128]):
        assert_raises(ImageCms.PyCMSError, lambda: ImageCms.ImageCmsProfile(BytesIO(data)))


@pytest.mark.parametrize("mode", ["RGB", "RGBA"])
@pytest.mark.parametrize("proof", [False, True])
def test_wide_integer_precision_alpha_and_reuse(mode, proof):
    profile = ImageCms.createProfile("sRGB")
    transform = ImageCms.buildProofTransform(profile, profile, profile, mode, mode) if proof else ImageCms.buildTransform(profile, profile, mode, mode)
    samples = [(i * 7919 + 11) % 65536 for i in range(37 * len(mode))]
    raw = struct.pack(f"<{len(samples)}H", *samples)
    image = Image.frombytes(mode, (37, 1), raw, bit_depth=16)
    image.info["custom"] = "preserved"
    for source in (Image.new(mode, image.size), image, image):
        result = transform.apply(source)
        assert result.bit_depth == source.bit_depth
        assert result.tobytes() == source.tobytes()
    output = Image.new(mode, image.size).convert(mode, bit_depth=16)
    assert transform.apply(image, output) is output
    assert output.tobytes() == raw
    assert_raises(ImageCms.PyCMSError, lambda: transform.apply(image, Image.new(mode, image.size)))
    core = image.getim()
    transform.apply_in_place(image)
    assert image.getim() is core
    assert image.tobytes() == raw
    assert image.info["custom"] == "preserved"
    assert image.info["icc_profile"] == profile.tobytes()


def test_wide_integer_transfer_curve_precision():
    source, output = ImageCms.createProfile("Adobe RGB"), ImageCms.createProfile("sRGB")
    values = list(range(1, 65536, 73))
    samples = [value for value in values for _ in range(3)]
    image = Image.frombytes("RGB", (len(values), 1), struct.pack(f"<{len(samples)}H", *samples), bit_depth=16)
    result = ImageCms.profileToProfile(image, source, output, flags=ImageCms.Flags.NOOPTIMIZE)
    assert result is not None
    for value, pixel in zip(values, result.getdata(), strict=True):
        linear = (value / 65535) ** (563 / 256)
        expected = 12.92 * linear if linear <= 0.0031308 else 1.055 * linear ** (1 / 2.4) - 0.055
        assert pixel == pytest.approx((expected * 65535,) * 3, abs=2)
    assert len({pixel[0] for pixel in result.getdata()}) > 256


@pytest.mark.parametrize("mode", ["RGB;F", "RGBA;F"])
@pytest.mark.parametrize("proof", [False, True])
def test_float_transform_extended_range_and_transfer_curve(mode, proof):
    source, output = ImageCms.createProfile("Adobe RGB"), ImageCms.createProfile("sRGB")
    values = [0.001 + i / 53 for i in range(100)]
    samples = [v for value in values for v in ((value,) * 3 + ((value - 0.3,) if mode == "RGBA;F" else ()))]
    raw = struct.pack(f"<{len(samples)}f", *samples)
    image = Image.frombytes(mode, (len(values), 1), raw)
    transform = ImageCms.buildProofTransform(source, output, output, mode, mode) if proof else ImageCms.buildTransform(source, output, mode, mode)
    result = transform.apply(image)
    assert (result.mode, result.bit_depth) == (mode, 32)
    if not proof:
        for value, pixel in zip(values, result.getdata(), strict=True):
            linear = value ** (563 / 256)
            expected = 12.92 * linear if linear <= 0.0031308 else 1.055 * linear ** (1 / 2.4) - 0.055
            assert pixel[:3] == pytest.approx((expected,) * 3, abs=0.0003)
    if mode == "RGBA;F":
        assert result.getchannel("A").tobytes() == image.getchannel("A").tobytes()
    destination = Image.frombytes(mode, image.size, bytes(len(raw)))
    assert transform.apply(image, destination) is destination
    assert destination.tobytes() == result.tobytes()
    assert image.tobytes() == raw
    core = image.getim()
    transform.apply_in_place(image)
    assert image.getim() is core
    assert image.tobytes() == result.tobytes()


@pytest.mark.parametrize("depth,source_mode,target_mode", [(16, "RGB", "RGBA"), (16, "RGBA", "RGB"), (32, "RGB;F", "RGBA;F"), (32, "RGBA;F", "RGB;F"), (16, "RGBA", "RGBA;F"), (32, "RGBA;F", "RGBA")])
def test_wide_mode_conversion(depth, source_mode, target_mode):
    profile = ImageCms.createProfile("sRGB")
    values = [0.2, 0.4, 0.6, 0.3][: Image.getmodebands(source_mode)]
    samples = [round(v * 65535) for v in values] if depth == 16 else values
    image = Image.frombytes(source_mode, (1, 1), struct.pack("<" + ("H" if depth == 16 else "f") * len(samples), *samples), bit_depth=depth if depth == 16 else 8)
    result = ImageCms.profileToProfile(image, profile, profile, outputMode=target_mode, flags=ImageCms.Flags.COPY_ALPHA)
    assert result is not None
    expected_depth = 32 if ";F" in target_mode else (8 if depth == 32 else depth)
    assert result.bit_depth == expected_depth
    maximum = 1 if expected_depth == 32 else (1 << expected_depth) - 1
    expected = values[:3] + ([values[3] if len(values) == 4 else 1] if Image.getmodebands(target_mode) == 4 else [])
    assert result.getpixel((0, 0)) == pytest.approx([v * maximum for v in expected], abs=1 if expected_depth != 32 else 0.0001)


@pytest.mark.parametrize("mode,depth", [("RGB", 16), ("RGBA", 16), ("RGB;F", 32), ("RGBA;F", 32)])
def test_wide_empty_images(mode, depth):
    profile = ImageCms.createProfile("sRGB")
    for size in ((0, 0), (0, 3), (3, 0)):
        image = Image.frombytes(mode, size, b"", bit_depth=depth if depth == 16 else 8)
        result = ImageCms.profileToProfile(image, profile, profile)
        assert result is not None
        assert_image_properties(result, mode=mode, size=size, bit_depth=depth, pixels=b"")


@pytest.mark.parametrize("depth,mode,raw", [(16, "RGBA", struct.pack("<4H", 12345, 23456, 34567, 45678)), (32, "RGBA;F", struct.pack("<4f", 0.12345, 0.45678, 1.2345, -0.5))])
def test_wide_transform_chunk_boundary(depth, mode, raw):
    profile = ImageCms.createProfile("sRGB")
    image = Image.frombytes(mode, (65537, 1), raw * 65537, bit_depth=depth if depth == 16 else 8)
    result = ImageCms.profileToProfile(image, profile, profile)
    assert result is not None
    assert result.tobytes() == image.tobytes()
