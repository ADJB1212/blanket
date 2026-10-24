from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image as PillowImage, ImageCms as PillowCms

from blanket import Image, ImageCms


@pytest.mark.parametrize("space", ["sRGB", "Adobe RGB", "Display P3", "LAB", "XYZ"])
def test_profile_creation_serialization_and_metadata(space: str, tmp_path: Path) -> None:
    profile = ImageCms.createProfile(space)
    data = profile.tobytes()
    pillow = PillowCms.ImageCmsProfile(BytesIO(data))
    path = tmp_path / "profile.icc"
    path.write_bytes(data)
    for source in (profile, BytesIO(data), path, str(path)):
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
        assert actual.tobytes() == expected.tobytes()
        assert actual.mode == target_mode
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
    stream = BytesIO()
    image.save(stream, "PNG")
    reopened = Image.open(BytesIO(stream.getvalue()))
    assert reopened.info["icc_profile"] == output.tobytes()


def test_transform_only_changes_current_frame() -> None:
    stream = BytesIO()
    Image.new("RGB", (2, 1), (80, 140, 220)).save(stream, "PNG", save_all=True, append_images=[Image.new("RGB", (2, 1), (35, 200, 25))])
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
    with pytest.raises(ImageCms.PyCMSError):
        ImageCms.profileToProfile(Image.new(mode, (1, 1)), profile, profile)


def test_invalid_arguments_leave_pixels_unchanged(tmp_path: Path) -> None:
    profile = ImageCms.createProfile("sRGB")
    image = Image.new("RGB", (2, 1), (80, 140, 220))
    raw = image.tobytes()
    for intent in (-1, 4, 1.5, "perceptual"):
        with pytest.raises(ImageCms.PyCMSError):
            ImageCms.profileToProfile(image, profile, profile, intent, inPlace=True)  # type: ignore[arg-type]
    for flags in (-1, 2**32, 0.5):
        with pytest.raises(ImageCms.PyCMSError):
            ImageCms.buildTransform(profile, profile, "RGB", "RGB", flags=flags)  # type: ignore[arg-type]
    with pytest.raises(ImageCms.PyCMSError):
        ImageCms.ImageCmsProfile(BytesIO(b"invalid ICC"))
    with pytest.raises(ImageCms.PyCMSError):
        ImageCms.getOpenProfile(tmp_path / "missing.icc")
    with pytest.raises(ImageCms.PyCMSError):
        ImageCms.createProfile("unknown")
    with pytest.raises(ImageCms.PyCMSError):
        ImageCms.profileToProfile(image, profile, ImageCms.createProfile("LAB"), outputMode="LAB", inPlace=True)
    transform = ImageCms.buildTransform(profile, profile, "RGB", "RGB")
    with pytest.raises(ImageCms.PyCMSError):
        ImageCms.applyTransform(Image.new("RGBA", (1, 1)), transform)
    assert image.tobytes() == raw
    image.close()
    with pytest.raises(ImageCms.PyCMSError):
        ImageCms.applyTransform(image, transform)
    with pytest.raises(ImageCms.PyCMSError):
        ImageCms.applyTransform(Image.new("RGB", (1, 1)).convert("RGB", bit_depth=16), transform)


def test_reused_transform_across_threads() -> None:
    profile = ImageCms.createProfile("sRGB")
    transform = ImageCms.buildTransform(profile, profile, "RGBA", "RGBA")
    image = Image.new("RGBA", (37, 11), (10, 20, 30, 40))
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
    with pytest.raises(ImageCms.PyCMSError):
        ImageCms.createProfile("LAB", temperature)


@pytest.mark.parametrize("flags", [ImageCms.Flags.NOCACHE, ImageCms.Flags.NOOPTIMIZE, ImageCms.Flags.COPY_ALPHA])
def test_flags_and_output_image(flags: int) -> None:
    profile = ImageCms.createProfile("sRGB")
    transform = ImageCms.buildTransformFromOpenProfiles(profile, profile, "RGBA", "RGBA", flags=flags)
    image = Image.new("RGBA", (7, 1), (40, 80, 120, 160))
    output = Image.new("RGBA", image.size)
    assert transform.apply(image, output) is output
    assert output.tobytes() == image.tobytes()
    with pytest.raises(ImageCms.PyCMSError):
        transform.apply(image, Image.new("RGBA", (1, 1)))
    with pytest.raises(ImageCms.PyCMSError):
        transform.apply(image, Image.new("RGB", image.size))


@pytest.mark.parametrize("mode", ["L", "CMYK", "LAB"])
def test_profile_and_mode_must_match(mode: str) -> None:
    profile = ImageCms.createProfile("sRGB")
    with pytest.raises(ImageCms.PyCMSError):
        ImageCms.buildTransform(profile, profile, mode, "RGB")
    with pytest.raises(ImageCms.PyCMSError):
        ImageCms.buildTransform(profile, profile, "RGB", mode)


def test_empty_image_and_truncated_profile() -> None:
    profile = ImageCms.createProfile("sRGB")
    for size in ((0, 0), (0, 3), (3, 0)):
        result = ImageCms.profileToProfile(Image.new("RGB", size), profile, profile)
        assert result is not None
        assert result.size == size
        assert result.tobytes() == b""
    for data in (profile.tobytes()[:64], profile.tobytes()[:128]):
        with pytest.raises(ImageCms.PyCMSError):
            ImageCms.ImageCmsProfile(BytesIO(data))
