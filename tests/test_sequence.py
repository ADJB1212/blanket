from __future__ import annotations

import struct
import zlib
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image as PillowImage, PdfParser

from blanket import Image, ImageSequence, UnidentifiedImageError
from blanket.Compressor import LosslessImageCompressor


def animation(format: str = "PNG", **options: object) -> Image.Image:
    frames = [PillowImage.new("RGBA", (9, 7), color) for color in ("red", "green", "blue")]
    stream = BytesIO()
    frames[0].save(stream, format, save_all=True, append_images=frames[1:], duration=[20, 40, 60], loop=2, **options)
    return Image.open(stream)


@pytest.mark.parametrize("format", ["GIF", "WEBP", "PNG"])
def test_navigation_metadata_mutation_and_copy(format: str) -> None:
    image = animation(format, **({"lossless": True} if format == "WEBP" else {}))
    assert image.n_frames == 3
    assert image.is_animated
    assert image.tell() == 0
    for i, duration in enumerate((20, 40, 60)):
        image.seek(i)
        assert image.tell() == i
        assert image.info["duration"] == duration
        assert image.info["loop"] == 2
    image.putpixel((0, 0), (50, 60, 70, 255))
    image.info["custom"] = "third"
    copied = image.copy()
    assert copied.n_frames == 1
    assert not copied.is_animated
    assert copied.tell() == 0
    image.seek(0)
    assert "custom" not in image.info
    image.seek(2)
    assert image.getpixel((0, 0)) == (50, 60, 70, 255)
    assert image.info["custom"] == "third"
    for frame in (-1, 3, 100):
        with pytest.raises(EOFError):
            image.seek(frame)
        assert image.tell() == 2
    with pytest.raises(TypeError):
        image.seek(1.5)
    image.close()
    for operation in (lambda: image.seek(0), image.tell, image.load):
        with pytest.raises(ValueError, match="closed"):
            operation()
    assert copied.getpixel((0, 0)) == (50, 60, 70, 255)


def test_static_navigation_and_sequence_protocol() -> None:
    still = Image.new("L", (2, 3), 7)
    assert (still.n_frames, still.is_animated, still.tell()) == (1, False, 0)
    still.seek(0)
    with pytest.raises(EOFError):
        still.seek(1)
    image = animation()
    iterator = ImageSequence.Iterator(image)
    assert iter(iterator) is iterator
    assert iterator[1] is image
    assert image.tell() == 1
    assert [frame.tell() for frame in iterator] == [0, 1, 2]
    assert image.tell() == 2
    frames = ImageSequence.all_frames([still, image], lambda frame: frame.convert("RGB"))
    assert len(frames) == 4
    assert image.tell() == 2
    assert all(frame.n_frames == 1 and frame.mode == "RGB" for frame in frames)
    frames[-1].putpixel((0, 0), (1, 2, 3))
    assert image.getpixel((0, 0)) != (1, 2, 3, 255)

    def fail(frame: Image.Image) -> Image.Image:
        raise RuntimeError("callback")

    with pytest.raises(RuntimeError, match="callback"):
        ImageSequence.all_frames(image, fail)
    assert image.tell() == 2


@pytest.mark.parametrize("format", ["GIF", "WEBP", "PNG"])
@pytest.mark.parametrize("writer", ["pillow", "blanket"])
def test_animation_interoperability(format: str, writer: str) -> None:
    module = Image if writer == "blanket" else PillowImage
    frames = [module.new("RGBA", (11, 9), (0, 0, 0, 0)) for _ in range(3)]
    for i, frame in enumerate(frames):
        for y in range(2, 6):
            for x in range(i * 3, i * 3 + 3):
                frame.putpixel((x, y), ((255, 0, 0, 255), (0, 255, 0, 255), (0, 0, 255, 255))[i])
    stream = BytesIO()
    options = {"lossless": True} if format == "WEBP" else {"disposal": 2} if format == "GIF" and writer == "pillow" else {}
    frames[0].save(stream, format, save_all=True, append_images=frames[1:], duration=[20, 40, 60], loop=3, **options)
    actual = Image.open(stream)
    expected = PillowImage.open(stream)
    assert actual.n_frames == expected.n_frames == 3
    assert actual.info["loop"] == expected.info["loop"] == 3
    for i in range(3):
        actual.seek(i)
        expected.seek(i)
        assert actual.convert("RGBA").tobytes() == expected.convert("RGBA").tobytes()
        expected.load()
        assert actual.info["duration"] == expected.info["duration"]
        if writer == "blanket":
            assert actual.tobytes() == frames[i].tobytes()


@pytest.mark.parametrize(("format", "disposal"), [("GIF", 1), ("GIF", 2), ("GIF", 3), ("PNG", 0), ("PNG", 1), ("PNG", 2)])
def test_partial_frames_and_disposal(format: str, disposal: int) -> None:
    frames = [PillowImage.new("RGBA", (13, 7), (0, 0, 0, 0)) for _ in range(3)]
    frames[0].paste("red", (0, 0, 5, 5))
    frames[1].paste("green", (4, 2, 9, 7))
    frames[2].paste("blue", (8, 0, 13, 5))
    stream = BytesIO()
    frames[0].save(stream, format, save_all=True, append_images=frames[1:], disposal=[disposal] * 3, duration=30)
    actual, expected = Image.open(stream), PillowImage.open(stream)
    for i in range(3):
        actual.seek(i)
        expected.seek(i)
        assert actual.tobytes() == expected.convert("RGBA").tobytes()


@pytest.mark.parametrize("writer", ["pillow", "blanket"])
def test_apng_separate_default_image(writer: str) -> None:
    module = Image if writer == "blanket" else PillowImage
    frames = [module.new("RGBA", (5, 3), color) for color in ("red", "green", "blue")]
    stream = BytesIO()
    frames[0].save(stream, "PNG", save_all=True, append_images=frames[1:], default_image=True, duration=[40, 60], loop=4)
    actual = Image.open(stream)
    expected = PillowImage.open(stream)
    assert actual.n_frames == 3
    assert actual.info["default_image"] is True
    assert "duration" not in actual.info
    for i in range(3):
        actual.seek(i)
        expected.seek(i)
        assert actual.tobytes() == expected.tobytes() == frames[i].tobytes()
        if i:
            assert actual.info["duration"] == [40, 60][i - 1]


@pytest.mark.parametrize("depth", [8, 10, 12, 16])
def test_apng_sample_precision(depth: int) -> None:
    frames = []
    for i in (1, 2):
        maximum = (1 << depth) - 1
        raw = bytes([17 * i, 29 * i, 51 * i, 255]) if depth == 8 else struct.pack("<4H", 17 * i, maximum // 2, maximum - 1, maximum)
        frames.append(Image.frombytes("RGBA", (1, 1), raw, bit_depth=depth))
    stream = BytesIO()
    frames[0].save(stream, "PNG", save_all=True, append_images=frames[1:], duration=30)
    reopened = Image.open(stream)
    assert reopened.bit_depth == (8 if depth == 8 else 16)
    for i, frame in enumerate(frames):
        reopened.seek(i)
        expected = frame.tobytes() if depth == 8 else struct.pack("<4H", *[(v * 65535 + maximum // 2) // maximum for (v,) in struct.iter_unpack("<H", frame.tobytes())])
        assert reopened.tobytes() == expected


@pytest.mark.parametrize("compression", ["raw", "tiff_lzw", "tiff_adobe_deflate", "jpeg"])
def test_tiff_pages_preserve_modes_and_dimensions(compression: str) -> None:
    frames = [PillowImage.new("RGB", (9, 7), "red"), PillowImage.new("L", (5, 3), 123), PillowImage.new("CMYK", (3, 5), (10, 20, 30, 40))]
    if compression == "jpeg":
        frames[-1] = frames[-1].convert("RGB")
    stream = BytesIO()
    frames[0].save(stream, "TIFF", save_all=True, append_images=frames[1:], compression=compression)
    actual, expected = Image.open(stream), PillowImage.open(stream)
    assert actual.n_frames == 3
    assert actual.is_animated
    for i in range(3):
        actual.seek(i)
        expected.seek(i)
        assert actual.size == expected.size
        assert actual.mode == expected.mode
        assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("mode", ["LAB", "F", "I;16"])
def test_tiff_special_mode_on_later_page(mode: str) -> None:
    first = PillowImage.new("RGB", (3, 2), "red")
    second = PillowImage.new(mode, (2, 4), (60, 120, 140) if mode == "LAB" else 123)
    stream = BytesIO()
    first.save(stream, "TIFF", save_all=True, append_images=[second])
    actual = Image.open(stream)
    actual.seek(1)
    if mode == "I;16":
        assert actual.bit_depth == 16
    else:
        assert actual.mode == mode
    assert actual.size == second.size
    assert actual.tobytes() == second.tobytes()


def test_multipage_pdf_and_save_positions() -> None:
    source = animation()
    source.seek(2)
    extra = Image.new("CMYK", (2, 3), (10, 20, 30, 40))
    stream = BytesIO()
    source.save(stream, "PDF", save_all=True, append_images=[extra])
    assert source.tell() == 2
    assert extra.tell() == 0
    pdf = PdfParser.PdfParser(buf=stream.getvalue())
    assert len(pdf.pages) == 4
    for i, reference in enumerate(pdf.pages):
        page = pdf.read_indirect(reference)
        assert page[b"MediaBox"] == ([0, 0, 9, 7] if i < 3 else [0, 0, 2, 3])
        raster = pdf.read_indirect(page[b"Resources"][b"XObject"][b"Im0"])
        if i < 3:
            source.seek(i)
            assert raster.decode() == source.convert("RGB").tobytes()
            assert pdf.read_indirect(raster.dictionary[b"SMask"]).decode() == source.getchannel("A").tobytes()
        else:
            assert raster.dictionary[b"ColorSpace"] == b"DeviceCMYK"
            assert raster.decode() == extra.tobytes()


def test_save_defaults_and_path(tmp_path: Path) -> None:
    source = animation()
    source.seek(1)
    path = tmp_path / "animated.apng"
    source.save(bytes(path), save_all=True)
    reopened = Image.open(path)
    assert reopened.n_frames == 3
    assert source.tell() == 1
    assert [frame.info["duration"] for frame in ImageSequence.Iterator(reopened)] == [20, 40, 60]
    stream = BytesIO()
    source.save(stream, "PNG", save_all=False, append_images=[Image.new("RGB", (2, 2))])
    still = Image.open(stream)
    assert still.n_frames == 1
    assert still.tobytes() == source.tobytes()


@pytest.mark.parametrize("format", ["GIF", "WEBP", "PNG", "PDF"])
def test_save_repeated_sources_preserves_frame_edits_and_metadata(format: str) -> None:
    source = animation()
    source.seek(1)
    source.putpixel((0, 0), (17, 29, 51, 255))
    source.info.update(duration=90, custom="edited")
    expected = ImageSequence.all_frames(source)
    stream = BytesIO()
    source.save(stream, format, save_all=True, append_images=(frame for frame in [source]), **({"lossless": True} if format == "WEBP" else {}))
    assert source.tell() == 1
    assert source.info["custom"] == "edited"
    assert source.info["duration"] == 90
    assert source.getpixel((0, 0)) == (17, 29, 51, 255)
    if format == "PDF":
        pdf = PdfParser.PdfParser(buf=stream.getvalue())
        assert len(pdf.pages) == 6
        for reference, frame in zip(pdf.pages, expected * 2, strict=True):
            page = pdf.read_indirect(reference)
            assert pdf.read_indirect(page[b"Resources"][b"XObject"][b"Im0"]).decode() == frame.convert("RGB").tobytes()
    else:
        reopened = Image.open(stream)
        assert reopened.n_frames == 6
        for actual, frame in zip(ImageSequence.Iterator(reopened), expected * 2, strict=True):
            assert actual.tobytes() == frame.tobytes()
            assert actual.info["duration"] == frame.info["duration"]
    for actual, frame in zip(ImageSequence.Iterator(source), expected, strict=True):
        assert actual.tobytes() == frame.tobytes()
        assert actual.info == frame.info


def test_pdf_save_includes_replaced_current_raster() -> None:
    source = animation()
    source.seek(1)
    source.thumbnail((4, 4))
    size, pixels = source.size, source.convert("RGB").tobytes()
    stream = BytesIO()
    source.save(stream, "PDF", save_all=True)
    assert source.tell() == 1
    assert source.size == size
    pdf = PdfParser.PdfParser(buf=stream.getvalue())
    page = pdf.read_indirect(pdf.pages[1])
    assert page[b"MediaBox"] == [0, 0, *size]
    assert pdf.read_indirect(page[b"Resources"][b"XObject"][b"Im0"]).decode() == pixels


@pytest.mark.parametrize("format", ["GIF", "WEBP", "PNG", "PDF"])
def test_closed_appended_source_does_not_write_or_change_position(format: str) -> None:
    source, closed = animation(), animation()
    source.seek(1)
    closed.close()
    stream = BytesIO()
    with pytest.raises(ValueError, match="closed"):
        source.save(stream, format, save_all=True, append_images=[closed])
    assert source.tell() == 1
    assert not stream.getvalue()


@pytest.mark.parametrize(
    ("format", "options", "error"),
    [
        ("JPEG", {"save_all": True}, ValueError),
        ("PNG", {"save_all": 1}, TypeError),
        ("PNG", {"save_all": True, "append_images": [1]}, TypeError),
        ("PNG", {"save_all": True, "duration": [20]}, ValueError),
        ("PNG", {"save_all": True, "duration": float("nan")}, ValueError),
        ("PNG", {"save_all": True, "disposal": 3}, ValueError),
        ("PNG", {"save_all": True, "blend": 2}, ValueError),
        ("GIF", {"save_all": True, "duration": 700000}, ValueError),
        ("GIF", {"save_all": True, "loop": 65536}, ValueError),
        ("PNG", {"save_all": True, "loop": 0x80000000}, ValueError),
        ("WEBP", {"save_all": True, "disposal": 1}, TypeError),
        ("PDF", {"save_all": True, "duration": 20}, TypeError),
        ("PNG", {"save_all": True, "compressor": LosslessImageCompressor()}, ValueError),
    ],
)
def test_invalid_sequence_options_do_not_write(format: str, options: dict[str, object], error: type[Exception]) -> None:
    image = animation()
    image.seek(1)
    stream = BytesIO()
    with pytest.raises(error):
        image.save(stream, format, **options)
    assert stream.getvalue() == b""
    assert image.tell() == 1


@pytest.mark.parametrize("format", ["GIF", "WEBP", "PNG"])
def test_animation_requires_matching_dimensions(format: str) -> None:
    output = BytesIO()
    with pytest.raises(ValueError, match="matching dimensions"):
        Image.new("RGB", (3, 2)).save(output, format, save_all=True, append_images=[Image.new("RGB", (2, 3))])
    assert not output.getvalue()


@pytest.mark.parametrize("format", ["GIF", "WEBP", "PNG", "TIFF"])
def test_truncated_later_frame_is_rejected_at_open(format: str) -> None:
    first = PillowImage.new("RGB", (17, 13), "red")
    stream = BytesIO()
    first.save(stream, format, save_all=True, append_images=[PillowImage.new("RGB", first.size, "blue")])
    data = stream.getvalue()
    with pytest.raises(UnidentifiedImageError):
        Image.open(BytesIO(data[: len(data) * 3 // 4]))


@pytest.mark.parametrize("depth", [8, 16])
def test_apng_over_blending_retains_sample_precision(depth: int) -> None:
    maximum = (1 << depth) - 1
    samples = [(maximum, 0, 0, maximum), (0, 0, maximum, (maximum + 1) // 2)]
    frames = [Image.frombytes("RGBA", (1, 1), bytes(values) if depth == 8 else struct.pack("<4H", *values), bit_depth=depth) for values in samples]
    stream = BytesIO()
    frames[0].save(stream, "PNG", save_all=True, append_images=frames[1:], duration=20, blend=[0, 1])
    image = Image.open(stream)
    image.seek(1)
    assert image.info["blend"] == 1
    expected = (maximum // 2, 0, (maximum + 1) // 2, maximum)
    assert image.tobytes() == (bytes(expected) if depth == 8 else struct.pack("<4H", *expected))


@pytest.mark.parametrize("depth", [8, 16])
def test_apng_over_blending_alpha_boundaries(depth: int) -> None:
    maximum = (1 << depth) - 1
    alphas = (0, 1, maximum // 2, maximum)
    background = [(maximum, maximum // 3, 0, da) for da in alphas for _ in alphas]
    foreground = [(0, maximum // 5, maximum, sa) for _ in alphas for sa in alphas]

    def pack(pixels: list[tuple[int, ...]]) -> bytes:
        samples = [value for pixel in pixels for value in pixel]
        return bytes(samples) if depth == 8 else struct.pack(f"<{len(samples)}H", *samples)

    frames = [Image.frombytes("RGBA", (16, 1), pack(samples), bit_depth=depth) for samples in (background, foreground)]
    stream = BytesIO()
    frames[0].save(stream, "PNG", save_all=True, append_images=frames[1:], blend=[0, 1])
    reopened = Image.open(stream)
    assert reopened.tobytes() == pack(background)
    reopened.seek(1)
    expected = []
    for dst, src in zip(background, foreground, strict=True):
        alpha = src[3] * maximum + dst[3] * (maximum - src[3])
        rgb = tuple((src[c] * src[3] * maximum + dst[c] * dst[3] * (maximum - src[3]) + alpha // 2) // alpha if alpha else 0 for c in range(3))
        expected.append((*rgb, (alpha + maximum // 2) // maximum))
    assert reopened.tobytes() == pack(expected)


def test_apng_declared_frame_budget_is_checked_before_decoding() -> None:
    first = PillowImage.new("RGBA", (2, 2), "red")
    stream = BytesIO()
    first.save(stream, "PNG", save_all=True, append_images=[PillowImage.new("RGBA", first.size, "blue")])
    data = bytearray(stream.getvalue())
    start = data.index(b"acTL")
    data[start + 4 : start + 8] = struct.pack(">I", 0x7FFFFFFF)
    data[start + 12 : start + 16] = struct.pack(">I", zlib.crc32(data[start : start + 12]))
    with pytest.raises(UnidentifiedImageError, match="limit"):
        Image.open(BytesIO(data))


def test_tiff_cyclic_page_chain_is_rejected() -> None:
    stream = BytesIO()
    PillowImage.new("RGB", (2, 2), "red").save(stream, "TIFF")
    data = bytearray(stream.getvalue())
    offset = struct.unpack_from("<I", data, 4)[0]
    count = struct.unpack_from("<H", data, offset)[0]
    struct.pack_into("<I", data, offset + 2 + count * 12, offset)
    with pytest.raises(UnidentifiedImageError, match="cycl"):
        Image.open(BytesIO(data))


def test_gif_frame_outside_canvas_is_rejected() -> None:
    stream = BytesIO()
    PillowImage.new("RGB", (2, 2), "red").save(stream, "GIF")
    data = bytearray(stream.getvalue())
    descriptor = data.index(b",")
    struct.pack_into("<H", data, descriptor + 1, 65535)
    with pytest.raises(UnidentifiedImageError):
        Image.open(BytesIO(data))


@pytest.mark.parametrize("mode", ["1", "L", "LA", "P", "PA", "RGB", "RGBA", "I", "I;16", "I;16L", "I;16B"])
@pytest.mark.parametrize("format", ["GIF", "WEBP", "PNG", "PDF"])
def test_sequence_save_mode_adaptation(mode: str, format: str) -> None:
    frames = [Image.new(mode, (3, 2), 0), Image.new(mode, (3, 2), 1)]
    stream = BytesIO()
    frames[0].save(stream, format, save_all=True, append_images=frames[1:], **({"duration": 20} if format != "PDF" else {}))
    if format == "PDF":
        assert len(PdfParser.PdfParser(buf=stream.getvalue()).pages) == 2
    else:
        actual, expected = Image.open(stream), PillowImage.open(stream)
        assert actual.n_frames == expected.n_frames
        for i in range(actual.n_frames):
            actual.seek(i)
            expected.seek(i)
            assert actual.convert("RGBA", bit_depth=8).tobytes() == expected.convert("RGBA").tobytes()


@pytest.mark.parametrize("mode", ["HSV", "LAB", "YCbCr", "CMYK", "F"])
def test_sequence_save_respects_mode_restrictions(mode: str) -> None:
    stream = BytesIO()
    with pytest.raises(OSError, match="cannot write mode"):
        Image.new(mode, (3, 2)).save(stream, "PNG", save_all=True)
    assert not stream.getvalue()


@pytest.mark.parametrize("mode", ["I", "I;16", "I;16L", "I;16B"])
def test_integer_apng_retains_values_above_255(mode: str) -> None:
    values = [0x1234, 0xABCD]
    frames = [Image.new(mode, (1, 1), value) for value in values]
    output = BytesIO()
    frames[0].save(output, "PNG", save_all=True, append_images=frames[1:])
    reopened = Image.open(output)
    assert reopened.bit_depth == 16
    for i, value in enumerate(values):
        reopened.seek(i)
        assert reopened.tobytes() == struct.pack("<4H", value, value, value, 65535)
