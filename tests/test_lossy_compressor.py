from __future__ import annotations

import io
import math
import random
from pathlib import Path

import pytest
from PIL import Image as PillowImage

from blanket import Image
from blanket.Compressor import LosslessImageCompressor, LossyImageCompressor


def encode(image: Image.Image, format: str, **options: object) -> bytes:
    output = io.BytesIO()
    image.save(output, format, **options)
    return output.getvalue()


def assert_bounded(baseline: bytes, output: bytes, bound: float) -> None:
    assert len(output) <= len(baseline)
    expected = Image.open(io.BytesIO(baseline)).convert("RGBA", bit_depth=16)
    actual = Image.open(io.BytesIO(output)).convert("RGBA", bit_depth=16)
    assert actual.size == expected.size
    before_raw, after_raw = expected.tobytes(), actual.tobytes()
    before = [int.from_bytes(before_raw[i : i + 2], "little") for i in range(0, len(before_raw), 2)]
    after = [int.from_bytes(after_raw[i : i + 2], "little") for i in range(0, len(after_raw), 2)]
    assert before[3::4] == after[3::4]
    error = sum(((a - b) / 257) ** 2 for i, (a, b) in enumerate(zip(before, after)) if i % 4 != 3)
    assert math.sqrt(error / (actual.width * actual.height * 3)) <= bound + 0.004


@pytest.mark.parametrize("format", ["PNG", "JPEG", "WEBP", "JXL", "AVIF", "HEIF", "HEIC"])
@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
def test_bounded_save(format: str, mode: str) -> None:
    if format == "JPEG" and mode == "RGBA":
        pytest.skip("JPEG does not support alpha")
    raw = random.Random(49).randbytes(33 * 19 * len(mode))
    source = Image.frombytes(mode, (33, 19), raw)
    source.info["custom"] = "retained"
    compressor = LossyImageCompressor(max_rmse=6, effort=3)
    baseline = encode(source, format)
    output = encode(source, format, compressor=compressor)
    assert_bounded(baseline, output, compressor.max_rmse)
    assert source.tobytes() == raw
    assert source.info == {"custom": "retained"}
    assert encode(source, format, compressor=compressor) == output
    if format in {"PNG", "JPEG", "WEBP"}:
        with PillowImage.open(io.BytesIO(output)) as external:
            assert external.convert("RGBA").tobytes() == Image.open(io.BytesIO(output)).convert("RGBA").tobytes()


@pytest.mark.parametrize("format", ["PNG", "JPEG", "WEBP"])
def test_additional_loss_reduces_size(format: str) -> None:
    raw = random.Random(37).randbytes(129 * 67 * 3)
    source = Image.frombytes("RGB", (129, 67), raw)
    baseline = encode(source, format)
    output = encode(source, format, compressor=LossyImageCompressor(max_rmse=12, effort=7))
    lossless = encode(source, format, compressor=LosslessImageCompressor(effort=7))
    assert len(output) < len(lossless)
    assert Image.open(io.BytesIO(output)).convert("RGB").tobytes() != Image.open(io.BytesIO(baseline)).convert("RGB").tobytes()
    assert_bounded(baseline, output, 12)


@pytest.mark.parametrize("format", ["PNG", "JPEG", "WEBP", "JXL", "AVIF", "HEIF"])
def test_zero_error_retains_lossless_behavior(format: str) -> None:
    source = Image.frombytes("RGB", (19, 17), random.Random(21).randbytes(19 * 17 * 3))
    output = encode(source, format, compressor=LossyImageCompressor(max_rmse=0, effort=1))
    assert output == encode(source, format, compressor=LosslessImageCompressor(effort=1))


@pytest.mark.parametrize("quality", [1, 2, 100])
def test_quality_endpoints_and_maximum_effort(quality: int) -> None:
    source = Image.frombytes("RGB", (17, 19), random.Random(9).randbytes(17 * 19 * 3))
    baseline = encode(source, "JPEG", quality=quality)
    output = encode(source, "JPEG", quality=quality, compressor=LossyImageCompressor(max_rmse=4, effort=10))
    assert_bounded(baseline, output, 4)


def test_looser_bound_allows_more_png_loss() -> None:
    source = Image.frombytes("RGBA", (65, 33), random.Random(27).randbytes(65 * 33 * 4))
    baseline = encode(source, "PNG")
    tight = encode(source, "PNG", compressor=LossyImageCompressor(max_rmse=0.01, effort=10))
    loose = encode(source, "PNG", compressor=LossyImageCompressor(max_rmse=20, effort=10))
    assert len(loose) < len(tight)
    assert_bounded(baseline, tight, 0.01)
    assert_bounded(baseline, loose, 20)


@pytest.mark.parametrize("format", ["WEBP", "JXL", "HEIF"])
def test_explicit_lossless_disables_additional_loss(format: str) -> None:
    source = Image.frombytes("RGB", (19, 17), random.Random(21).randbytes(19 * 17 * 3))
    output = encode(source, format, lossless=True, compressor=LossyImageCompressor(max_rmse=255, effort=1))
    assert output == encode(source, format, lossless=True, compressor=LosslessImageCompressor(effort=1))


@pytest.mark.parametrize("format,depth", [("JXL", 10), ("JXL", 12), ("JXL", 16), ("HEIF", 10), ("HEIF", 12)])
def test_high_depth(format: str, depth: int) -> None:
    rng = random.Random(91)
    raw = b"".join(rng.randrange(1 << depth).to_bytes(2, "little") for _ in range(17 * 19 * 3))
    source = Image.frombytes("RGB", (17, 19), raw, bit_depth=depth)
    baseline = encode(source, format)
    output = encode(source, format, compressor=LossyImageCompressor(max_rmse=8, effort=2))
    assert_bounded(baseline, output, 8)
    assert source.tobytes() == raw
    assert source.bit_depth == depth


@pytest.mark.parametrize("format", ["PNG", "TIFF", "PDF"])
def test_fallback_formats(format: str) -> None:
    source = Image.new("RGB", (7, 9), (13, 51, 99))
    if format == "PNG":
        source = source.convert("RGB", bit_depth=16)
    assert encode(source, format, compressor=LossyImageCompressor()) == encode(source, format)


@pytest.mark.parametrize("format", ["PNG", "WEBP"])
def test_indexed_input(format: str) -> None:
    source = Image.frombytes("RGB", (17, 19), random.Random(17).randbytes(17 * 19 * 3)).quantize()
    raw, palette = source.tobytes(), source.palette.tobytes()
    baseline = encode(source, format)
    output = encode(source, format, compressor=LossyImageCompressor(effort=2))
    if format == "PNG":
        assert output == baseline
    assert_bounded(baseline, output, 2)
    assert source.mode == "P"
    assert source.tobytes() == raw
    assert source.palette.tobytes() == palette


def test_path_and_reuse(tmp_path: Path) -> None:
    compressor = LossyImageCompressor(max_rmse=4, effort=2)
    assert compressor.max_rmse == 4
    assert compressor.effort == 2
    for mode in ("L", "RGB", "RGBA"):
        image = Image.new(mode, (7, 9))
        path = tmp_path / f"{mode}.png"
        image.save(path, compressor=compressor)
        assert path.read_bytes() == encode(image, "PNG", compressor=compressor)
    for property in ("effort", "max_rmse"):
        with pytest.raises(AttributeError):
            setattr(compressor, property, 1)


@pytest.mark.parametrize("value", [True, "2", None, object()])
def test_error_bound_type(value: object) -> None:
    with pytest.raises(TypeError, match="max_rmse"):
        LossyImageCompressor(max_rmse=value)


@pytest.mark.parametrize("value", [-1, 256, float("nan"), float("inf"), -float("inf"), 10**1000])
def test_error_bound_range(value: float) -> None:
    with pytest.raises(ValueError, match="max_rmse"):
        LossyImageCompressor(max_rmse=value)


@pytest.mark.parametrize("value,error", [(True, TypeError), (1.5, TypeError), ("7", TypeError), (None, TypeError), (0, ValueError), (11, ValueError)])
def test_effort_validation(value: object, error: type[Exception]) -> None:
    with pytest.raises(error, match="effort"):
        LossyImageCompressor(effort=value)


def test_failed_save_does_not_write(tmp_path: Path) -> None:
    path = tmp_path / "existing.jpg"
    path.write_bytes(b"keep")
    source = Image.new("RGBA", (7, 9))
    with pytest.raises(OSError):
        source.save(path, compressor=LossyImageCompressor())
    assert path.read_bytes() == b"keep"
    source.close()
    with pytest.raises(ValueError, match="closed"):
        encode(source, "PNG", compressor=LossyImageCompressor())
