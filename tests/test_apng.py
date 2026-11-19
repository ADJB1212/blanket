"""Copied passing Pillow 13.0.0 contracts; see PILLOW_TESTS.md."""

from __future__ import annotations

import struct
import tracemalloc
import zlib
from io import BytesIO

import pytest
from helpers import assert_raises, png_chunk
from pillow_helpers import PILLOW_IMAGES

from blanket import CorruptImageError, Image, ImageSequence, PngImagePlugin
from blanket.PngImagePlugin import Blend, Disposal

TYPE_CHECKING = False
if TYPE_CHECKING:
    from pathlib import Path


def assert_frame_pixels(image, first, second=None, *, frame=None):
    if frame is not None:
        image.seek(frame)
    assert image.getpixel((0, 0)) == first
    assert image.getpixel((64, 32)) == (first if second is None else second)


@pytest.mark.parametrize("name", ["mode_palette.png", "mode_palette_1bit_alpha.png", "mode_palette_alpha.png"])
@pytest.mark.parametrize("lazy", [False, True])
def test_packed_apng_palette_decodes_with_and_without_progress(name, lazy):
    data = (PILLOW_IMAGES / "apng" / name).read_bytes()
    with Image.decode(data, lazy=lazy) as expected, Image.decode(data, lazy=lazy, progress=lambda _: None) as actual:
        assert actual.mode == expected.mode == "P"
        for index in range(expected.n_frames):
            expected.seek(index)
            actual.seek(index, progress=lambda _: None)
            assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("progress", [None, lambda _: None])
def test_interlaced_packed_apng_does_not_reuse_bits_from_previous_frame(progress):
    width, height = 13, 9
    passes = [(0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8), (2, 0, 4, 4), (0, 2, 2, 4), (1, 0, 2, 2), (0, 1, 1, 2)]

    def interlace(index):
        rows = []
        for left, top, dx, dy in passes:
            count = len(range(left, width, dx))
            row = bytes((0xFF if index else 0,)) * ((count + 7) // 8)
            rows.extend([b"\0" + row] * len(range(top, height, dy)))
        return zlib.compress(b"".join(rows))

    def control(sequence):
        return png_chunk(b"fcTL", struct.pack(">5I2H2B", sequence, width, height, 0, 0, 1, 10, 0, 0))

    data = (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", struct.pack(">2I5B", width, height, 1, 3, 0, 0, 1))
        + png_chunk(b"PLTE", b"\x00\x00\x00\xff\xff\xff")
        + png_chunk(b"acTL", struct.pack(">2I", 2, 0))
        + control(0)
        + png_chunk(b"IDAT", interlace(1))
        + control(1)
        + png_chunk(b"fdAT", struct.pack(">I", 2) + interlace(0))
        + png_chunk(b"IEND", b"")
    )
    with Image.decode(data, progress=progress) as image:
        assert image.tobytes() == b"\x01" * (width * height)
        image.seek(1)
        assert image.tobytes() == bytes(width * height)


@pytest.mark.parametrize("lazy", [False, True])
def test_indexed_apng_over_expands_blended_palette_colors(lazy):
    first = Image.new("P", (4, 4), 0)
    first.putpalette(bytes([255, 0, 0, 0, 0, 255]))
    first.info["transparency"] = b"\x80\x80"
    second = first.copy()
    second.putpixel((2, 2), 1)
    data = first.encode("PNG", save_all=True, append_images=[second], blend=[0, 1])
    with Image.decode(data, lazy=lazy) as image:
        assert image.mode == "P"
        image.seek(1)
        assert image.mode == "RGBA"
        assert "transparency" not in image.info
        assert image.getpixel((2, 2)) == (85, 0, 170, 192)
        assert image.getpixel((0, 0)) == (255, 0, 0, 192)


@pytest.mark.parametrize("key", [0, 1])
def test_bilevel_apng_transparency(key):
    first = Image.frombytes("1", (2, 1), b"\x40")
    first.info["transparency"] = key
    second = Image.frombytes("1", (2, 1), b"\x80")
    second.info["transparency"] = key
    data = first.encode("PNG", save_all=True, append_images=[second])
    with Image.decode(data) as image:
        expected = bytes([255 if key == 1 else 0, 0 if key == 1 else 255])
        assert first.convert("LA").getchannel("A").tobytes() == expected
        assert image.convert("RGBA").getchannel("A").tobytes() == expected
        image.seek(1)
        assert image.convert("RGBA").getchannel("A").tobytes() == expected[::-1]


def test_apng_canvas_preserves_transparent_padding_and_larger_frame_content():
    first = Image.new("RGB", (3, 3), "red")
    second = Image.new("RGB", (5, 5), "blue")
    second.paste("red", (0, 0, 3, 3))
    second.putpixel((0, 0), (0, 255, 0))
    data = first.encode("PNG", save_all=True, append_images=[second])
    with Image.decode(data) as image:
        assert image.getpixel((4, 4)) == (0, 0, 0, 0)
        image.seek(1)
        assert image.convert("RGB").tobytes() == second.tobytes()


def test_apng_rejects_empty_frame_before_padding():
    empty = Image.new("RGBA", (0, 3))
    full = Image.new("RGBA", (3, 3))
    with pytest.raises(ValueError, match="empty image"):
        empty.encode("PNG", save_all=True, append_images=[full])


def test_apng_background_disposal_clears_the_full_submitted_frame():
    first = Image.new("RGBA", (5, 4), "red")
    second = first.copy()
    second.putpixel((2, 2), (0, 255, 0, 255))
    third = Image.new("RGBA", first.size)
    data = first.encode("PNG", save_all=True, append_images=[second, third], disposal=[0, 1, 0], blend=[0, 0, 1])
    with Image.decode(data) as image:
        image.seek(1)
        assert image.tobytes() == second.tobytes()
        image.seek(2)
        assert image.tobytes() == third.tobytes()


@pytest.mark.parametrize("lazy", [False, True])
def test_apng_resave_preserves_composited_frames_from_any_position(lazy):
    first = Image.new("RGBA", (5, 4), (255, 0, 0, 128))
    second = Image.new("RGBA", (3, 2), (0, 0, 255, 128))
    data = first.encode("PNG", save_all=True, append_images=[second], blend=[0, 1], duration=[20, 40])
    with Image.decode(data, lazy=lazy) as image:
        expected = ImageSequence.all_frames(image)
        for position in (0, 1):
            image.seek(position)
            with Image.decode(image.encode("PNG", save_all=True)) as result:
                assert [frame.tobytes() for frame in ImageSequence.all_frames(result)] == [frame.tobytes() for frame in expected]
            assert image.tell() == position


def invalid_apng_without_pixels(width: int, height: int) -> bytes:
    stream = BytesIO(PngImagePlugin._MAGIC)
    stream.seek(0, 2)
    PngImagePlugin.putchunk(stream, b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
    PngImagePlugin.putchunk(stream, b"acTL", struct.pack(">II", 0, 0))
    PngImagePlugin.putchunk(stream, b"IEND")
    return stream.getvalue()


def test_invalid_apng_recovery_bounds_temporary_memory() -> None:
    data = invalid_apng_without_pixels(512, 512)
    tracemalloc.start()
    try:
        with pytest.warns(UserWarning, match="Invalid APNG"):
            recovered, error, count = PngImagePlugin.sanitize_apng(data)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 512 * 1024, f"recovery allocated {peak} bytes"
    assert isinstance(error, OSError) and count is None
    with Image.decode(recovered) as image:
        assert image.size == (512, 512)


def test_invalid_apng_recovery_checks_dimensions_before_allocation(monkeypatch) -> None:
    def checked_bytes(value):
        if isinstance(value, int) and value > 64 * 1024:
            raise AssertionError("oversized recovery allocation")
        return bytes(value)

    data = invalid_apng_without_pixels(20000, 20000)
    monkeypatch.setattr(PngImagePlugin, "bytes", checked_bytes, raising=False)
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", None)
    with pytest.raises(CorruptImageError, match="IDAT or fdAT"):
        Image.decode(data)


@pytest.mark.parametrize("mode,depth", [("L", 8), ("LA", 8), ("RGB", 8), ("RGBA", 8), ("L", 16), ("RGB", 16), ("RGBA", 16)])
@pytest.mark.parametrize("default_image", [False, True])
def test_terminal_previous_disposal_preserves_lazy_and_eager_samples(mode: str, depth: int, default_image: bool) -> None:
    channels = len(mode)
    frames = []
    for i in range(3):
        values = [(17 + i * 37 + c * 53) * (257 if depth == 16 else 1) for c in range(channels)]
        raw = bytes(values) if depth == 8 else struct.pack(f"<{channels}H", *values)
        frames.append(Image.frombytes(mode, (17, 3), raw * 51, bit_depth=depth))
    disposal = [2, 2] if default_image else [0, 2, 2]
    duration = [20, 30] if default_image else [10, 20, 30]
    data = frames[0].encode("PNG", save_all=True, append_images=frames[1:], default_image=default_image, disposal=disposal, duration=duration, blend=[0] * len(disposal))
    assert all(frame.tell() == 0 for frame in frames)
    for lazy in [False, True]:
        with Image.open(BytesIO(data), lazy=lazy) as image:
            assert image.n_frames == 3 and image.is_animated
            for index in [2, 0, 1, 2, 1, 0]:
                image.seek(index)
                expected = frames[index] if depth == 8 else frames[index].convert("RGBA", bit_depth=16)
                assert image.mode == expected.mode and image.bit_depth == depth
                assert image.tobytes() == expected.tobytes()
                if index > 0:
                    assert image.info["disposal"] == 2 and image.info["duration"] == duration[index - int(default_image)]


# APNG browser support tests and fixtures via:
# https://philip.html5.org/tests/apng/tests.html
# (referenced from https://wiki.mozilla.org/APNG_Specification)
def test_apng_basic() -> None:
    with Image.open(PILLOW_IMAGES / "apng/single_frame.png") as im:
        assert isinstance(im, PngImagePlugin.PngImageFile)
        assert not im.is_animated
        assert im.n_frames == 1
        assert im.get_format_mimetype() == "image/apng"
        assert im.info.get("default_image") is None
        assert_frame_pixels(im, (0, 255, 0, 255))

    with Image.open(PILLOW_IMAGES / "apng/single_frame_default.png") as im:
        assert isinstance(im, PngImagePlugin.PngImageFile)
        assert im.is_animated
        assert im.n_frames == 2
        assert im.get_format_mimetype() == "image/apng"
        assert im.info.get("default_image")
        assert_frame_pixels(im, (255, 0, 0, 255))
        assert_frame_pixels(im, (0, 255, 0, 255), frame=1)

        # test out of bounds seek
        assert_raises(EOFError, lambda: im.seek(2))

        im.seek(0)
        assert_raises(ValueError, lambda: im._seek(2), match="cannot seek to frame 2")

        # test rewind support
        assert_frame_pixels(im, (255, 0, 0, 255))
        assert_frame_pixels(im, (0, 255, 0, 255), frame=1)


@pytest.mark.parametrize("filename", (PILLOW_IMAGES / "apng/split_fdat.png", PILLOW_IMAGES / "apng/split_fdat_zero_chunk.png"))
def test_apng_fdat(filename: str) -> None:
    with Image.open(filename) as im:
        assert isinstance(im, PngImagePlugin.PngImageFile)
        assert_frame_pixels(im, (0, 255, 0, 255), frame=im.n_frames - 1)


@pytest.mark.parametrize(
    "filename,first,second",
    [
        ("dispose_op_none.png", (0, 255, 0, 255), (0, 255, 0, 255)),
        ("dispose_op_background.png", (0, 0, 0, 0), (0, 0, 0, 0)),
        ("dispose_op_background_final.png", (0, 255, 0, 255), (0, 255, 0, 255)),
        ("dispose_op_previous.png", (0, 255, 0, 255), (0, 255, 0, 255)),
        ("dispose_op_previous_final.png", (0, 255, 0, 255), (0, 255, 0, 255)),
        ("dispose_op_previous_first.png", (0, 0, 0, 0), (0, 0, 0, 0)),
    ],
)
def test_apng_dispose(filename: str, first: tuple[int, ...], second: tuple[int, ...]) -> None:
    with Image.open(PILLOW_IMAGES / "apng" / filename) as im:
        assert isinstance(im, PngImagePlugin.PngImageFile)
        assert_frame_pixels(im, first, second, frame=im.n_frames - 1)


@pytest.mark.parametrize(
    "filename,first,second",
    [
        ("dispose_op_none_region.png", (0, 255, 0, 255), (0, 255, 0, 255)),
        ("dispose_op_background_before_region.png", (0, 0, 0, 0), (0, 0, 0, 0)),
        ("dispose_op_background_region.png", (0, 0, 255, 255), (0, 0, 0, 0)),
        ("dispose_op_previous_region.png", (0, 255, 0, 255), (0, 255, 0, 255)),
    ],
)
def test_apng_dispose_region(filename: str, first: tuple[int, ...], second: tuple[int, ...]) -> None:
    with Image.open(PILLOW_IMAGES / "apng" / filename) as im:
        assert isinstance(im, PngImagePlugin.PngImageFile)
        assert_frame_pixels(im, first, second, frame=im.n_frames - 1)


def test_apng_dispose_op_previous_frame() -> None:
    # Test that the dispose settings being used are from the previous frame
    #
    # Image created with:
    # red = Image.new("RGBA", (128, 64), (255, 0, 0, 255))
    # green = red.copy()
    # green.paste(Image.new("RGBA", (64, 32), (0, 255, 0, 255)))
    # blue = red.copy()
    # blue.paste(Image.new("RGBA", (64, 32), (0, 255, 0, 255)), (64, 32))
    #
    # red.save(
    #     PILLOW_IMAGES / "apng/dispose_op_previous_frame.png",
    #     save_all=True,
    #     append_images=[green, blue],
    #     disposal=[
    #         PngImagePlugin.Disposal.OP_NONE,
    #         PngImagePlugin.Disposal.OP_PREVIOUS,
    #         PngImagePlugin.Disposal.OP_PREVIOUS
    #     ],
    # )
    with Image.open(PILLOW_IMAGES / "apng/dispose_op_previous_frame.png") as im:
        assert isinstance(im, PngImagePlugin.PngImageFile)
        im.seek(im.n_frames - 1)
        assert im.getpixel((0, 0)) == (255, 0, 0, 255)


def test_apng_dispose_op_background_p_mode() -> None:
    with Image.open(PILLOW_IMAGES / "apng/dispose_op_background_p_mode.png") as im:
        im.seek(1)
        im.load()
        assert im.size == (128, 64)


def test_apng_dispose_op_background_decompression_bomb() -> None:
    o16 = PngImagePlugin.o16
    o32 = PngImagePlugin.o32

    b = BytesIO()
    b.write(PngImagePlugin._MAGIC)
    PngImagePlugin.putchunk(b, b"IHDR", o32(2**31 - 1), o32(1), b"\x08\x06\x00\x00\x00")
    PngImagePlugin.putchunk(b, b"acTL", o32(2), o32(0))
    PngImagePlugin.putchunk(b, b"fcTL", o32(0), o32(1), o32(1), o32(0), o32(0), o16(1), o16(10), bytes([PngImagePlugin.Disposal.OP_BACKGROUND, PngImagePlugin.Blend.OP_SOURCE]))
    PngImagePlugin.putchunk(b, b"IDAT")
    PngImagePlugin.putchunk(b, b"IEND")

    assert_raises(Image.DecompressionBombError, lambda: Image.open(b))


@pytest.mark.parametrize(
    "filename,first,second",
    [
        ("blend_op_source_solid.png", (0, 255, 0, 255), (0, 255, 0, 255)),
        ("blend_op_source_transparent.png", (0, 0, 0, 0), (0, 0, 0, 0)),
        ("blend_op_source_near_transparent.png", (0, 255, 0, 2), (0, 255, 0, 2)),
        ("blend_op_over.png", (0, 255, 0, 255), (0, 255, 0, 255)),
        ("blend_op_over_near_transparent.png", (0, 255, 0, 255), (0, 255, 0, 255)),
    ],
)
def test_apng_blend(filename: str, first: tuple[int, ...], second: tuple[int, ...]) -> None:
    with Image.open(PILLOW_IMAGES / "apng" / filename) as im:
        assert isinstance(im, PngImagePlugin.PngImageFile)
        assert_frame_pixels(im, first, second, frame=im.n_frames - 1)


def test_apng_blend_transparency() -> None:
    with Image.open(PILLOW_IMAGES / "blend_transparency.png") as im:
        im.seek(1)
        assert im.getpixel((0, 0)) == (255, 0, 0)


def test_apng_chunk_order() -> None:
    with Image.open(PILLOW_IMAGES / "apng/fctl_actl.png") as im:
        assert isinstance(im, PngImagePlugin.PngImageFile)
        assert_frame_pixels(im, (0, 255, 0, 255), frame=im.n_frames - 1)


@pytest.mark.parametrize(
    "filename,durations", [("delay.png", [500.0, 1000.0, 500.0, 1000.0]), ("delay_round.png", [500.0, 1000.0]), ("delay_short_max.png", [500.0, 1000.0]), ("delay_zero_denom.png", [500.0, 1000.0]), ("delay_zero_numer.png", [0.0, 0.0, 500.0, 1000.0])]
)
def test_apng_delay(filename: str, durations: list[float]) -> None:
    with Image.open(PILLOW_IMAGES / "apng" / filename) as im:
        for frame, duration in enumerate(durations, 1):
            im.seek(frame)
            assert im.info.get("duration") == duration


def test_apng_num_plays() -> None:
    with Image.open(PILLOW_IMAGES / "apng/num_plays.png") as im:
        assert im.info.get("loop") == 0

    with Image.open(PILLOW_IMAGES / "apng/num_plays_1.png") as im:
        assert im.info.get("loop") == 1


def test_apng_save_different_palettes(tmp_path: Path) -> None:
    # APNG does not support multiple palettes
    test_file = tmp_path / "temp.png"
    red = Image.new("P", (1, 1), (255, 0, 0))
    green = Image.new("P", (1, 1), (0, 255, 0))
    red.save(test_file, save_all=True, append_images=[green])

    with Image.open(test_file) as reloaded:
        # So the frames must be converted to another mode
        assert reloaded.mode == "RGB"
        assert reloaded.getpixel((0, 0)) == (255, 0, 0)

        reloaded.seek(1)
        assert reloaded.mode == "RGB"
        assert reloaded.getpixel((0, 0)) == (0, 255, 0)

    # Test that RGBA palettes result in an RGBA image
    green = Image.new("L", (1, 1))
    green.putpalette([], "RGBA")
    red.save(test_file, save_all=True, append_images=[green])
    with Image.open(test_file) as reloaded:
        assert reloaded.mode == "RGBA"


@pytest.mark.parametrize("filename", ["chunk_no_actl.png", "chunk_actl_after_idat.png"])
def test_apng_nonanimated_chunks(filename) -> None:
    with Image.open(PILLOW_IMAGES / "apng" / filename) as im:
        assert isinstance(im, PngImagePlugin.PngImageFile)
        assert not im.is_animated


@pytest.mark.parametrize("filename", ["chunk_no_fctl.png", "chunk_repeat_fctl.png", "chunk_no_fdat.png"])
def test_apng_missing_frame_chunks(filename) -> None:
    with Image.open(PILLOW_IMAGES / "apng" / filename) as im:
        assert isinstance(im, PngImagePlugin.PngImageFile)
        assert_raises(SyntaxError, lambda: im.seek(im.n_frames - 1))


@pytest.mark.parametrize("filename,error", [("chunk_multi_actl.png", None), ("syntax_num_frames_zero.png", OSError), ("syntax_num_frames_zero_default.png", None), ("syntax_num_frames_invalid.png", None)])
def test_apng_invalid_animation_control(filename, error) -> None:
    with pytest.warns(UserWarning, match="Invalid APNG"):
        im = Image.open(PILLOW_IMAGES / "apng" / filename)
    with im:
        assert isinstance(im, PngImagePlugin.PngImageFile)
        assert not im.is_animated
        if error:
            assert_raises(error, im.load)
        elif filename != "chunk_multi_actl.png":
            im.load()


def test_apng_low_frame_count() -> None:
    with Image.open(PILLOW_IMAGES / "apng/syntax_num_frames_low.png") as im:
        assert isinstance(im, PngImagePlugin.PngImageFile)
        im.seek(im.n_frames - 1)


def test_apng_high_frame_count() -> None:
    with pytest.raises(OSError), Image.open(PILLOW_IMAGES / "apng/syntax_num_frames_high.png") as im:
        assert isinstance(im, PngImagePlugin.PngImageFile)
        im.seek(im.n_frames - 1)
        im.load()


@pytest.mark.parametrize("test_file", ("sequence_start.png", "sequence_gap.png", "sequence_repeat.png", "sequence_repeat_chunk.png", "sequence_reorder.png", "sequence_reorder_chunk.png", "sequence_fdat_fctl.png"))
@pytest.mark.parametrize("lazy", [False, True])
def test_apng_sequence_errors(test_file: str, lazy: bool) -> None:
    with Image.open(PILLOW_IMAGES / f"apng/{test_file}", lazy=lazy) as im:
        assert isinstance(im, PngImagePlugin.PngImageFile)
        original = im.tobytes()
        assert im.n_frames > 1
        with pytest.raises(SyntaxError):
            im.seek(im.n_frames - 1)
        assert im.tell() == 0
        assert im.tobytes() == original


def test_apng_save(tmp_path: Path) -> None:
    with Image.open(PILLOW_IMAGES / "apng/single_frame.png") as im:
        test_file = tmp_path / "temp.png"
        im.save(test_file, save_all=True)

    with Image.open(test_file) as im:
        assert isinstance(im, PngImagePlugin.PngImageFile)
        im.load()
        assert not im.is_animated
        assert im.n_frames == 1
        assert im.get_format_mimetype() == "image/png"
        assert im.info.get("default_image") is None
        assert_frame_pixels(im, (0, 255, 0, 255))

    with Image.open(PILLOW_IMAGES / "apng/single_frame_default.png") as im:
        frames = [frame_im.copy() for frame_im in ImageSequence.Iterator(im)]
        frames[0].save(test_file, save_all=True, default_image=True, append_images=frames[1:])

    with Image.open(test_file) as im:
        assert isinstance(im, PngImagePlugin.PngImageFile)
        im.load()
        assert im.is_animated
        assert im.n_frames == 2
        assert im.get_format_mimetype() == "image/apng"
        assert im.info.get("default_image")
        assert_frame_pixels(im, (0, 255, 0, 255), frame=1)


def test_apng_save_alpha(tmp_path: Path) -> None:
    test_file = tmp_path / "temp.png"

    im = Image.new("RGBA", (1, 1), (255, 0, 0, 255))
    im2 = Image.new("RGBA", (1, 1), (255, 0, 0, 127))
    im.save(test_file, save_all=True, append_images=[im2])

    with Image.open(test_file) as reloaded:
        assert reloaded.getpixel((0, 0)) == (255, 0, 0, 255)

        reloaded.seek(1)
        assert reloaded.getpixel((0, 0)) == (255, 0, 0, 127)


@pytest.mark.parametrize("mode, key, other", [("L", 12, 48), ("RGB", (12, 24, 36), (48, 60, 72))])
def test_apng_save_shared_transparency_key(mode, key, other):
    first = Image.new(mode, (4, 4), key)
    second = Image.new(mode, first.size, other)
    second.putpixel((0, 0), key)
    first.info["transparency"] = second.info["transparency"] = key
    encoded = first.encode("PNG", save_all=True, append_images=[second])
    with Image.decode(encoded) as decoded:
        for position, frame in enumerate((first, second)):
            decoded.seek(position)
            assert decoded.convert("RGBA").tobytes() == frame.convert("RGBA").tobytes()


@pytest.mark.parametrize("lazy", [False, True])
def test_consecutive_previous_disposals_restore_the_same_canvas(lazy):
    frames = [Image.new("RGBA", (4, 4), color) for color in ("red", "green", "blue", "green")]
    frames[-1].putpixel((0, 0), (150, 10, 50, 255))
    encoded = frames[0].encode("PNG", save_all=True, append_images=frames[1:], disposal=[0, 2, 2, 0])
    with Image.decode(encoded, lazy=lazy) as decoded:
        for position, frame in enumerate(frames):
            decoded.seek(position)
            assert decoded.tobytes() == frame.tobytes()


def test_apng_over_composites_repeated_semitransparent_frames():
    background = Image.new("RGBA", (4, 4), "blue")
    frame = Image.new("RGBA", (4, 4), (255, 0, 0, 128))
    clear = Image.new("RGBA", frame.size)
    encoded = background.encode("PNG", save_all=True, append_images=[frame, frame.copy(), clear], duration=[100] * 4, blend=[0, 1, 1, 0])
    with Image.decode(encoded) as decoded:
        assert decoded.n_frames == 4
        assert decoded.tobytes() == background.tobytes()
        decoded.seek(1)
        assert decoded.getpixel((0, 0)) == (128, 0, 127, 255)
        decoded.seek(2)
        assert decoded.getpixel((0, 0)) == (192, 0, 63, 255)
        decoded.seek(3)
        assert decoded.tobytes() == clear.tobytes()


def test_apng_background_disposal_does_not_merge_with_cleared_frame():
    frame = Image.new("RGBA", (4, 4), "red")
    clear = Image.new("RGBA", frame.size)
    encoded = frame.encode("PNG", save_all=True, append_images=[clear], duration=[100, 100], disposal=1)
    with Image.decode(encoded) as decoded:
        assert decoded.n_frames == 2
        assert decoded.tobytes() == frame.tobytes()
        decoded.seek(1)
        assert decoded.tobytes() == clear.tobytes()


def test_apng_does_not_merge_frames_with_identical_overlap_but_different_sizes():
    first = Image.new("RGBA", (4, 4), "red")
    second = Image.new("RGBA", (4, 6), "red")
    data = first.encode("PNG", save_all=True, append_images=[second], duration=[100, 100])
    with Image.decode(data) as decoded:
        assert decoded.n_frames == 2
        assert decoded.getpixel((0, 5)) == (0, 0, 0, 0)
        decoded.seek(1)
        assert decoded.tobytes() == second.tobytes()


@pytest.mark.parametrize("depths", [(8, 12), (12, 8), (10, 16)])
@pytest.mark.parametrize("second_size", [(4, 4), (5, 6)])
def test_apng_promotes_mixed_depths_without_losing_samples(depths, second_size):
    sources = [Image.frombytes("RGB", (4, 4), bytes((27, 81, 179)) * 16), Image.frombytes("RGB", second_size, bytes((47, 101, 207)) * (second_size[0] * second_size[1]))]
    first, second = [source.convert("RGB", bit_depth=depth) for source, depth in zip(sources, depths, strict=True)]
    data = first.encode("PNG", save_all=True, append_images=[second])
    with Image.decode(data) as decoded:
        for position, frame in enumerate((first, second)):
            decoded.seek(position)
            assert decoded.bit_depth == 16
            assert decoded.size == second_size
            assert decoded.tobytes() == frame.convert("RGBA", bit_depth=16).crop((0, 0, *second_size)).tobytes()


def test_apng_save_split_fdat(tmp_path: Path) -> None:
    # test to make sure we do not generate sequence errors when writing
    # frames with image data spanning multiple fdAT chunks (in this case
    # both the default image and first animation frame will span multiple
    # data chunks)
    test_file = tmp_path / "temp.png"
    with Image.open(PILLOW_IMAGES / "old-style-jpeg-compression.png") as im:
        frames = [im.copy(), Image.new("RGBA", im.size, (255, 0, 0, 255))]
        im.save(test_file, save_all=True, default_image=True, append_images=frames)
    with Image.open(test_file) as im:
        assert isinstance(im, PngImagePlugin.PngImageFile)
        im.seek(im.n_frames - 1)
        im.load()


def test_apng_save_duration_loop(tmp_path: Path) -> None:
    test_file = tmp_path / "temp.png"
    with Image.open(PILLOW_IMAGES / "apng/delay.png") as im:
        frames = []
        durations = []
        loop = im.info.get("loop")
        default_image = im.info.get("default_image")
        for i, frame_im in enumerate(ImageSequence.Iterator(im)):
            frames.append(frame_im.copy())
            if i != 0 or not default_image:
                durations.append(frame_im.info.get("duration", 0))
        frames[0].save(test_file, save_all=True, default_image=default_image, append_images=frames[1:], duration=durations, loop=loop)

    with Image.open(test_file) as im:
        im.load()
        assert im.info.get("loop") == loop
        im.seek(1)
        assert im.info.get("duration") == 500.0
        im.seek(2)
        assert im.info.get("duration") == 1000.0
        im.seek(3)
        assert im.info.get("duration") == 500.0
        im.seek(4)
        assert im.info.get("duration") == 1000.0

    # test removal of duplicated frames
    frame = Image.new("RGBA", (128, 64), (255, 0, 0, 255))
    frame.save(test_file, save_all=True, append_images=[frame, frame], duration=[500, 100, 150])
    with Image.open(test_file) as im:
        assert isinstance(im, PngImagePlugin.PngImageFile)
        assert im.n_frames == 1
        assert "duration" not in im.info

    different_frame = Image.new("RGBA", (128, 64))
    frame.save(test_file, save_all=True, append_images=[frame, different_frame], duration=[500, 100, 150])
    with Image.open(test_file) as im:
        assert isinstance(im, PngImagePlugin.PngImageFile)
        assert im.n_frames == 2
        assert im.info["duration"] == 600

        im.seek(1)
        assert im.info["duration"] == 150

    # test info duration
    frame.info["duration"] = 300
    frame.save(test_file, save_all=True, append_images=[frame, different_frame])
    with Image.open(test_file) as im:
        assert isinstance(im, PngImagePlugin.PngImageFile)
        assert im.n_frames == 2
        assert im.info["duration"] == 600


def test_apng_save_duration_float(tmp_path: Path) -> None:
    test_file = tmp_path / "temp.png"
    im = Image.new("1", (1, 1))
    im2 = Image.new("1", (1, 1), 1)
    im.save(test_file, save_all=True, append_images=[im2], duration=0.5)

    with Image.open(test_file) as reloaded:
        assert reloaded.info["duration"] == 0.5


def test_apng_save_large_duration(tmp_path: Path) -> None:
    test_file = tmp_path / "temp.png"
    im = Image.new("1", (1, 1))
    im2 = Image.new("1", (1, 1), 1)
    assert_raises(ValueError, lambda: im.save(test_file, save_all=True, append_images=[im2], duration=65536000), match="cannot write duration")

    # Do not merge identical frames if the combined duration cannot be written
    im.save(test_file, save_all=True, append_images=[im.copy()], duration=[32768000, 32768000])

    with Image.open(test_file) as reloaded:
        assert isinstance(reloaded, PngImagePlugin.PngImageFile)
        assert reloaded.n_frames == 2
        assert reloaded.info["duration"] == 32768000

        reloaded.seek(1)
        assert reloaded.info["duration"] == 32768000


@pytest.mark.parametrize(
    "colors,default_image,disposal,blend,expected",
    [
        (["red", (0, 255, 0, 255), (0, 0, 0, 0)], False, Disposal.OP_NONE, Blend.OP_OVER, [(2, (0, 255, 0, 255))]),
        (["red", "red", (0, 0, 0, 0)], False, [Disposal.OP_NONE, Disposal.OP_BACKGROUND, Disposal.OP_NONE], Blend.OP_OVER, [(2, (0, 0, 0, 0))]),
        (["red", (0, 255, 0, 255)], False, [Disposal.OP_NONE, Disposal.OP_BACKGROUND], Blend.OP_OVER, [(1, (0, 255, 0, 255))]),
        (["red", (0, 255, 0, 255), "red", (0, 0, 0, 0)], True, [Disposal.OP_NONE, Disposal.OP_PREVIOUS, Disposal.OP_NONE], Blend.OP_OVER, [(3, (0, 255, 0, 255))]),
        (["red", (0, 255, 0, 255)], False, [Disposal.OP_NONE, Disposal.OP_PREVIOUS], Blend.OP_OVER, [(1, (0, 255, 0, 255))]),
        (["red", "red", (0, 255, 0, 255)], True, Disposal.OP_NONE, [Blend.OP_OVER, Blend.OP_SOURCE], [(2, (0, 255, 0, 255))]),
        (["red", "red", (0, 0, 0, 0)], True, Disposal.OP_NONE, [Blend.OP_OVER, Blend.OP_SOURCE], [(2, (0, 0, 0, 0))]),
        (["red", (0, 255, 0, 255), (0, 0, 0, 0)], True, Disposal.OP_NONE, Blend.OP_OVER, [(1, (0, 255, 0, 255)), (2, (0, 255, 0, 255))]),
    ],
    ids=["none", "background", "background-final", "previous-default", "previous-final", "source-solid", "source-transparent", "over"],
)
def test_apng_save_frame_controls(tmp_path, colors, default_image, disposal, blend, expected) -> None:
    test_file = tmp_path / "temp.png"
    frames = [Image.new("RGBA", (128, 64), color) for color in colors]
    frames[0].save(test_file, save_all=True, append_images=frames[1:], default_image=default_image, disposal=disposal, blend=blend)
    with Image.open(test_file) as im:
        for index, color in expected:
            assert_frame_pixels(im, color, frame=index)


def test_apng_save_disposal_info(tmp_path: Path) -> None:
    test_file = tmp_path / "temp.png"
    red = Image.new("RGBA", (128, 64), (255, 0, 0, 255))
    red.info["disposal"] = PngImagePlugin.Disposal.OP_BACKGROUND
    red.save(test_file, save_all=True, append_images=[Image.new("RGBA", (10, 10), (0, 255, 0, 255))])
    with Image.open(test_file) as im:
        im.seek(1)
        assert im.getpixel((64, 32)) == (0, 0, 0, 0)


def test_apng_save_disposal_previous(tmp_path: Path) -> None:
    test_file = tmp_path / "temp.png"
    size = (128, 64)
    blue = Image.new("RGBA", size, (0, 0, 255, 255))
    red = Image.new("RGBA", size, (255, 0, 0, 255))
    green = Image.new("RGBA", size, (0, 255, 0, 255))

    # test OP_NONE
    blue.save(test_file, save_all=True, append_images=[red, green], disposal=PngImagePlugin.Disposal.OP_PREVIOUS)
    with Image.open(test_file) as im:
        assert im.getpixel((0, 0)) == (0, 0, 255, 255)

        assert_frame_pixels(im, (0, 255, 0, 255), frame=2)


def test_apng_save_blend_info(tmp_path: Path) -> None:
    test_file = tmp_path / "temp.png"
    red, green, transparent = [Image.new("RGBA", (128, 64), color) for color in ((255, 0, 0, 255), (0, 255, 0, 255), (0, 0, 0, 0))]
    red.info["blend"] = PngImagePlugin.Blend.OP_OVER
    red.save(test_file, save_all=True, append_images=[green, transparent])
    with Image.open(test_file) as im:
        im.seek(2)
        assert im.getpixel((0, 0)) == (0, 255, 0, 255)


def test_apng_save_size(tmp_path: Path) -> None:
    test_file = tmp_path / "temp.png"

    im = Image.new("L", (100, 100))
    im.save(test_file, save_all=True, append_images=[Image.new("L", (200, 200))])

    with Image.open(test_file) as reloaded:
        assert reloaded.size == (200, 200)


def test_compress_level() -> None:
    compress_level_sizes = {}
    for compress_level in (0, 9):
        out = BytesIO()

        im = Image.new("L", (100, 100))
        im.save(out, "PNG", save_all=True, append_images=[Image.new("L", (200, 200))], compress_level=compress_level)

        compress_level_sizes[compress_level] = len(out.getvalue())

    assert compress_level_sizes[0] > compress_level_sizes[9]


def test_seek_after_close() -> None:
    im = Image.open(PILLOW_IMAGES / "apng/delay.png")
    im.seek(1)
    im.close()

    assert_raises(ValueError, lambda: im.seek(0))


@pytest.mark.parametrize("mode", ("RGBA", "RGB", "P"))
@pytest.mark.parametrize("default_image", (True, False))
@pytest.mark.parametrize("duplicate", (True, False))
def test_different_modes_in_later_frames(mode: str, default_image: bool, duplicate: bool, tmp_path: Path) -> None:
    test_file = tmp_path / "temp.png"

    im = Image.new("L", (1, 1))
    second = im.convert(mode) if duplicate else Image.new(mode, (1, 1), 1)
    expected = [frame.convert("RGBA").tobytes() for frame in (im, second)]
    im.save(test_file, save_all=True, default_image=default_image, append_images=[second])
    with Image.open(test_file) as reloaded:
        assert reloaded.mode == ("RGB" if mode == "P" else mode)
        for position, pixels in enumerate(expected):
            reloaded.seek(position)
            assert reloaded.convert("RGBA").tobytes() == pixels


def test_different_durations(tmp_path: Path) -> None:
    test_file = tmp_path / "temp.png"

    with Image.open(PILLOW_IMAGES / "apng/different_durations.png") as im:
        for _ in range(3):
            im.seek(0)
            assert im.info["duration"] == 4000

            im.seek(1)
            assert im.info["duration"] == 1000

        im.save(test_file, save_all=True)

    with Image.open(test_file) as reloaded:
        assert reloaded.info["duration"] == 4000

        reloaded.seek(1)
        assert reloaded.info["duration"] == 1000
