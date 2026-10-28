from __future__ import annotations

import struct
import zlib
from io import BytesIO

import pytest
from PIL import Image as PillowImage, PdfParser

from blanket import Image, ImageSequence, UnidentifiedImageError
from blanket.errors import CodecResourceLimitError, CorruptImageError, UnsupportedCodecError


def sequence_data(format="PNG", **options):
    frames = [PillowImage.new("RGBA", (13, 7), (0, 0, 0, 0)) for _ in range(3)]
    if format == "WEBP":
        frames = [PillowImage.new("RGBA", (13, 7), "white") for _ in range(3)]
    for i, color in enumerate(("red", "green", "blue")):
        frames[i].paste(color, (i * 4, 0, i * 4 + 5, 5))
    stream = BytesIO()
    frames[0].save(stream, format, save_all=True, append_images=frames[1:], duration=[20, 40, 60], loop=2, **options)
    return stream.getvalue()


@pytest.mark.parametrize(
    ("format", "options"),
    [
        ("GIF", {"disposal": 1}),
        ("GIF", {"disposal": 2}),
        ("GIF", {"disposal": 3}),
        ("PNG", {"disposal": 0}),
        ("PNG", {"disposal": 1}),
        ("PNG", {"disposal": 2}),
        ("PNG", {"blend": 1}),
        ("WEBP", {"lossless": True}),
    ],
)
def test_lazy_composition_matches_eager_and_pillow(format, options):
    data = sequence_data(format, **options)
    lazy = Image.open(BytesIO(data), lazy=True)
    eager = Image.open(BytesIO(data))
    pillow = PillowImage.open(BytesIO(data))
    assert lazy.n_frames == 3 and lazy.is_animated
    for i in (2, 0, 1, 2, 1, 0):
        lazy.seek(i)
        eager.seek(i)
        pillow.seek(i)
        assert lazy.tobytes() == eager.tobytes()
        assert lazy.tobytes() == pillow.convert("RGBA").tobytes()
        assert lazy.info == eager.info
        assert lazy.tell() == i
        assert list(lazy._lazy_frames.cache) == [i]
    for i in (-1, 3):
        with pytest.raises(EOFError):
            lazy.seek(i)
        assert lazy.tell() == 0


def test_lazy_cache_lru_edits_and_close(monkeypatch):
    from blanket import _lazy_frames

    calls = []
    decode = _lazy_frames._open_sequence

    def tracked(data, formats, frame):
        calls.append(frame)
        return decode(data, formats, frame)

    monkeypatch.setattr(_lazy_frames, "_open_sequence", tracked)
    image = Image.open(BytesIO(sequence_data()), lazy=True, frame_cache_size=2)
    image.putpixel((0, 0), (17, 29, 51, 255))
    image.info["custom"] = "edited"
    image.getexif()[270] = "edited"
    image.seek(1)
    image.seek(0)
    assert calls == [0, 1]
    assert image.getpixel((0, 0)) == (17, 29, 51, 255)
    assert image.info["custom"] == "edited"
    assert image.getexif()[270] == "edited"
    copied = image.copy()
    image.seek(2)
    assert list(image._lazy_frames.cache) == [0, 2]
    image.seek(1)
    image.seek(0)
    assert calls == [0, 1, 2, 1, 0]
    assert "custom" not in image.info
    assert 270 not in image.getexif()
    assert image.getpixel((0, 0)) == (255, 0, 0, 255)
    image.close()
    assert image._lazy_frames.data == b"" and not image._lazy_frames.cache
    for operation in (image.load, image.tell, lambda: image.seek(1)):
        with pytest.raises(ValueError, match="closed"):
            operation()
    assert copied.n_frames == 1
    assert copied.getpixel((0, 0)) == (17, 29, 51, 255)


@pytest.mark.parametrize("mode", ["1", "L", "RGB", "RGBA", "CMYK", "F", "I;16"])
@pytest.mark.parametrize("compression", ["raw", "tiff_lzw", "tiff_adobe_deflate"])
def test_lazy_tiff_mixed_pages_and_metadata(mode, compression):
    first = PillowImage.new("RGB", (3, 2), "red")
    second = PillowImage.new(mode, (5, 4))
    stream = BytesIO()
    first.save(stream, "TIFF", save_all=True, append_images=[second], compression=compression)
    image = Image.open(stream, lazy=True)
    eager = Image.open(stream)
    for i in (1, 0, 1):
        image.seek(i)
        eager.seek(i)
        assert (image.mode, image.size, image.bit_depth) == (eager.mode, eager.size, eager.bit_depth)
        assert image.tobytes() == eager.tobytes()
        assert dict(image.getexif()) == dict(eager.getexif())


def test_lazy_tiff_jpeg_pages():
    frames = [PillowImage.new("RGB", (9, 7), "red"), PillowImage.new("L", (5, 3), 123)]
    stream = BytesIO()
    frames[0].save(stream, "TIFF", save_all=True, append_images=frames[1:], compression="jpeg")
    lazy = Image.open(stream, lazy=True)
    pillow = PillowImage.open(stream)
    for i in (1, 0, 1):
        lazy.seek(i)
        pillow.seek(i)
        assert (lazy.mode, lazy.size, lazy.tobytes()) == (pillow.mode, pillow.size, pillow.tobytes())


def test_lazy_webp_alpha_matches_eager():
    frames = [PillowImage.new("RGBA", (13, 7), (0, 0, 0, 0)) for _ in range(3)]
    for i, frame in enumerate(frames):
        frame.paste((17, 29, 51, 127 + i), (i * 4, 0, i * 4 + 5, 5))
    stream = BytesIO()
    frames[0].save(stream, "WEBP", save_all=True, append_images=frames[1:], duration=30, lossless=True)
    lazy = Image.open(stream, lazy=True)
    eager = Image.open(stream)
    for i in (2, 0, 1):
        lazy.seek(i)
        eager.seek(i)
        assert lazy.mode == eager.mode == "RGBA"
        assert lazy.tobytes() == eager.tobytes()
        assert lazy.getchannel("A").getextrema()[0] == 0


@pytest.mark.parametrize("depth", [8, 10, 12, 16])
@pytest.mark.parametrize("default", [False, True])
def test_lazy_apng_precision_and_default_image(depth, default):
    maximum = (1 << depth) - 1
    frames = [
        Image.frombytes("RGBA", (2, 3), bytes([i * 17, 29, 51, 255]) * 6 if depth == 8 else struct.pack("<4H", i * 17, maximum // 2, maximum - 1, maximum) * 6, bit_depth=depth) for i in (1, 2, 3)
    ]
    stream = BytesIO()
    frames[0].save(stream, "PNG", save_all=True, append_images=frames[1:], default_image=default, duration=40)
    lazy = Image.open(stream, lazy=True)
    eager = Image.open(stream)
    for i in (2, 0, 1):
        lazy.seek(i)
        eager.seek(i)
        assert lazy.bit_depth == eager.bit_depth
        assert lazy.tobytes() == eager.tobytes()
        assert lazy.info == eager.info


@pytest.mark.parametrize("format", ["GIF", "PNG", "WEBP", "PDF"])
def test_lazy_iteration_saving_and_position_restoration(format):
    data = sequence_data()
    image = Image.open(BytesIO(data), lazy=True)
    image.seek(1)
    frames = ImageSequence.all_frames(image)
    assert image.tell() == 1
    assert len(frames) == 3 and all(frame.n_frames == 1 for frame in frames)
    stream = BytesIO()
    image.save(stream, format, save_all=True, **({"lossless": True} if format == "WEBP" else {}))
    assert image.tell() == 1
    if format == "PDF":
        assert len(PdfParser.PdfParser(buf=stream.getvalue()).pages) == 3
    else:
        reopened = Image.open(stream)
        assert [frame.tobytes() for frame in ImageSequence.Iterator(reopened)] == [frame.tobytes() for frame in frames]

    def fail(frame):
        raise RuntimeError("callback")

    with pytest.raises(RuntimeError, match="callback"):
        ImageSequence.all_frames(image, fail)
    assert image.tell() == 1


def png_chunk(kind, data):
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


def test_lazy_defers_bad_compressed_frame_and_preserves_position():
    data = sequence_data()
    offset = data.index(b"fdAT") - 4
    size = struct.unpack_from(">I", data, offset)[0]
    payload = data[offset + 8 : offset + 8 + size]
    data = data[:offset] + png_chunk(b"fdAT", payload[:4] + b"invalid deflate") + data[offset + size + 12 :]
    with pytest.raises(CorruptImageError):
        Image.open(BytesIO(data))
    image = Image.open(BytesIO(data), lazy=True)
    original = image.tobytes()
    image.info["custom"] = True
    for _ in range(2):
        with pytest.raises(CorruptImageError):
            image.seek(1)
        assert image.tell() == 0
        assert image.tobytes() == original
        assert image.info["custom"] is True
        assert list(image._lazy_frames.cache) == [0]


def test_lazy_tiff_defers_bad_page_pixels():
    stream = BytesIO()
    first = PillowImage.new("RGB", (3, 2), "red")
    first.save(stream, "TIFF", save_all=True, append_images=[PillowImage.new("L", (4, 5), 17)], compression="tiff_adobe_deflate")
    data = bytearray(stream.getvalue())
    pillow = PillowImage.open(stream)
    pillow.seek(1)
    offset = pillow.tag_v2[273][0]
    count = pillow.tag_v2[279][0]
    data[offset : offset + count] = b"\xff" * count
    with pytest.raises(CorruptImageError):
        Image.open(BytesIO(data))
    image = Image.open(BytesIO(data), lazy=True)
    original = image.tobytes()
    with pytest.raises(CorruptImageError):
        image.seek(1)
    assert image.tell() == 0 and image.tobytes() == original


def test_lazy_gif_defers_bad_lzw_pixels():
    data = bytearray(sequence_data("GIF"))
    position = 13 + (3 * (2 << (data[10] & 7)) if data[10] & 128 else 0)
    frame = 0
    while data[position] != 0x3B:
        marker = data[position]
        position += 1
        if marker == 0x21:
            position += 1
        else:
            assert marker == 0x2C
            flags = data[position + 8]
            position += 9 + (3 * (2 << (flags & 7)) if flags & 128 else 0)
            position += 1
            frame += 1
        while data[position]:
            count = data[position]
            if marker == 0x2C and frame == 2:
                data[position + 1 : position + 1 + count] = b"\xff" * count
            position += count + 1
        position += 1
    with pytest.raises(CorruptImageError):
        Image.open(BytesIO(data))
    image = Image.open(BytesIO(data), lazy=True)
    original = image.tobytes()
    with pytest.raises(CorruptImageError):
        image.seek(1)
    assert image.tell() == 0 and image.tobytes() == original


def test_lazy_animation_edits_do_not_change_other_composited_frames():
    data = sequence_data("PNG", blend=1)
    image = Image.open(BytesIO(data), lazy=True, frame_cache_size=2)
    image.putpixel((0, 0), (17, 29, 51, 255))
    image.seek(1)
    eager = Image.open(BytesIO(data))
    eager.seek(1)
    assert image.tobytes() == eager.tobytes()
    image.seek(0)
    assert image.getpixel((0, 0)) == (17, 29, 51, 255)


@pytest.mark.parametrize("mode", ["1", "L", "LA", "P", "RGB", "RGBA", "I;16"])
def test_lazy_static_png_modes(mode):
    stream = BytesIO()
    PillowImage.new(mode, (3, 2)).save(stream, "PNG")
    lazy = Image.open(stream, lazy=True)
    eager = Image.open(stream)
    assert (lazy.mode, lazy.bit_depth, lazy.tobytes()) == (eager.mode, eager.bit_depth, eager.tobytes())
    assert lazy.convert("RGBA").tobytes() == eager.convert("RGBA").tobytes()


def test_lazy_save_current_frame_and_replaced_cached_raster():
    image = Image.open(BytesIO(sequence_data()), lazy=True, frame_cache_size=2)
    image.seek(1)
    image.thumbnail((4, 4))
    expected = image.copy()
    image.seek(0)
    image.seek(1)
    assert image.size == expected.size
    assert image.tobytes() == expected.tobytes()
    output = BytesIO()
    image.save(output, "PNG")
    reopened = Image.open(output)
    assert reopened.n_frames == 1 and reopened.tobytes() == expected.tobytes()


def test_lazy_cumulative_limit_before_frame_decode():
    data = sequence_data()
    offset = data.index(b"acTL") - 4
    data = data[:offset] + png_chunk(b"acTL", struct.pack(">II", 3_000_000, 0)) + data[offset + 20 :]
    with pytest.raises(CodecResourceLimitError):
        Image.open(BytesIO(data), lazy=True)


@pytest.mark.parametrize("format", ["GIF", "WEBP", "TIFF"])
def test_lazy_other_formats_cumulative_limit_before_decode(format, monkeypatch):
    from blanket import _lazy_frames

    def unexpected_decode(*args):
        pytest.fail("pixel decoding started before cumulative limit validation")

    monkeypatch.setattr(_lazy_frames, "_open_sequence", unexpected_decode)
    if format == "TIFF":
        stream = BytesIO()
        first = PillowImage.new("RGB", (3, 2), "red")
        first.save(stream, "TIFF", save_all=True, append_images=[PillowImage.new("L", (4, 5), 17)])
        data = bytearray(stream.getvalue())
        offset = struct.unpack_from("<I", data, 4)[0]
        while offset:
            count = struct.unpack_from("<H", data, offset)[0]
            for i in range(count):
                entry = offset + 2 + i * 12
                tag, kind = struct.unpack_from("<HH", data, entry)
                if tag in (256, 257, 278):
                    struct.pack_into("<H" if kind == 3 else "<I", data, entry + 8, 10_000)
            offset = struct.unpack_from("<I", data, offset + 2 + count * 12)[0]
    else:
        data = bytearray(sequence_data(format))
        if format == "GIF":
            struct.pack_into("<HH", data, 6, 10_000, 10_000)
        else:
            offset = data.index(b"VP8X")
            data[offset + 12 : offset + 15] = (9999).to_bytes(3, "little")
            data[offset + 15 : offset + 18] = (9999).to_bytes(3, "little")
    with pytest.raises(CodecResourceLimitError):
        Image.open(BytesIO(data), lazy=True)


@pytest.mark.parametrize("options, error", [({"lazy": 1}, TypeError), ({"frame_cache_size": True}, TypeError), ({"frame_cache_size": 1.5}, TypeError), ({"frame_cache_size": 0}, ValueError)])
def test_lazy_option_validation(options, error):
    with pytest.raises(error):
        Image.open(BytesIO(sequence_data()), **options)


def test_lazy_format_allowlist_and_unsupported_format():
    with pytest.raises(UnidentifiedImageError):
        Image.open(BytesIO(sequence_data()), formats=["GIF"], lazy=True)
    stream = BytesIO()
    Image.new("RGB", (2, 2)).save(stream, "JPEG")
    with pytest.raises(UnsupportedCodecError):
        Image.open(stream, lazy=True)


@pytest.mark.parametrize("format", ["GIF", "PNG", "WEBP", "TIFF"])
def test_lazy_static_images_and_stream_ownership(format):
    stream = BytesIO()
    PillowImage.new("L", (3, 2), 17).save(stream, format)
    lazy = Image.open(stream, lazy=True)
    eager = Image.open(stream)
    stream.close()
    assert lazy.n_frames == 1 and not lazy.is_animated
    assert (lazy.mode, lazy.tobytes()) == (eager.mode, eager.tobytes())
    lazy.close()
