from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from struct import pack
from threading import get_ident
from zlib import compress, crc32

import pytest
from PIL import Image as Pillow

from blanket import Compressor, Image, OperationCancelledError


@pytest.mark.parametrize("mode", ["1", "P", "RGB", "I;16"])
def test_png_progress_preserves_pixels_and_can_cancel_deferred_load(mode):
    size = (41, 49)
    source = Pillow.frombytes("L", size, bytes(i % 256 for i in range(size[0] * size[1]))).convert(mode)
    stream = BytesIO()
    source.save(stream, format="PNG")
    data = stream.getvalue()
    expected = Image.decode(data)
    seen = []
    assert Image.decode(data, progress=seen.append).tobytes() == expected.tobytes()
    assert seen == list(range(len(seen))) and len(seen) > 3
    assert Image.open(BytesIO(data), progress=lambda _: True).tobytes() == expected.tobytes()
    with pytest.raises(OperationCancelledError):
        Image.decode(data, progress=lambda completed: completed < 1)

    if mode == "RGB":
        image = Image.open(BytesIO(data))
        with pytest.raises(OperationCancelledError):
            image.load(progress=lambda completed: completed < 1)
        assert image.tobytes() == expected.tobytes()


def test_interlaced_png_progress_preserves_adam7_output():
    width, height = 29, 37
    pixels = bytes(i % 251 for i in range(width * height))
    passes = [(0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8), (2, 0, 4, 4), (0, 2, 2, 4), (1, 0, 2, 2), (0, 1, 1, 2)]
    rows = b"".join(b"\0" + pixels[y * width + x : (y + 1) * width : dx] for x, top, dx, dy in passes for y in range(top, height, dy))

    def chunk(kind, payload):
        return pack(">I", len(payload)) + kind + payload + pack(">I", crc32(kind + payload))

    data = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", pack(">IIBBBBB", width, height, 8, 0, 0, 0, 1)) + chunk(b"IDAT", compress(rows)) + chunk(b"IEND", b"")
    seen = []
    assert Image.decode(data, progress=seen.append).tobytes() == pixels
    assert seen == list(range(len(seen))) and len(seen) > 3


@pytest.mark.parametrize("format", ["PNG", "GIF", "WEBP"])
def test_sequence_progress_and_cancelled_lazy_seek(format):
    source = Image.new("RGB", (41, 49), "red")
    following = [Image.new("RGB", source.size, color) for color in ("green", "blue")]
    data = source.encode(format, save_all=True, append_images=following, duration=100)
    seen = []
    eager = Image.decode(data, progress=seen.append)
    assert seen == list(range(len(seen))) and len(seen) >= 5
    lazy = Image.decode(data, lazy=True)
    first = lazy.tobytes()
    with pytest.raises(OperationCancelledError):
        lazy.seek(2, progress=lambda completed: completed < 1)
    assert lazy.tell() == 0 and lazy.tobytes() == first
    lazy.seek(2, progress=lambda _: True)
    eager.seek(2)
    assert lazy.tobytes() == eager.tobytes()


@pytest.mark.parametrize(
    ("mode", "depth", "method", "linear_light"),
    [("RGB", 8, Image.Resampling.NEAREST, False), ("RGBA", 8, Image.Resampling.LANCZOS, False), ("RGBA", 8, Image.Resampling.BICUBIC, True), ("RGB", 12, Image.Resampling.BILINEAR, False), ("RGB;F", 32, Image.Resampling.BICUBIC, False)],
)
def test_resize_progress_is_monotonic_and_preserves_output(mode, depth, method, linear_light):
    image = Image.new(mode, (71, 83))
    if depth == 12:
        image = image.convert(mode, bit_depth=depth)
    seen = []
    result = image.resize((59, 67), method, linear_light=linear_light, progress=seen.append)
    assert result.tobytes() == image.resize((59, 67), method, linear_light=linear_light).tobytes()
    assert seen == list(range(len(seen))) and len(seen) > 3
    with pytest.raises(OperationCancelledError):
        image.resize((59, 67), method, linear_light=linear_light, progress=lambda completed: completed < 2)
    assert image.size == (71, 83)


def test_resize_progress_covers_pre_reduction_and_identity():
    image = Image.new("RGBA", (192, 192), "orange")
    seen = []
    result = image.resize((21, 25), reducing_gap=2, progress=seen.append)
    assert result.tobytes() == image.resize((21, 25), reducing_gap=2).tobytes()
    assert seen == list(range(len(seen))) and len(seen) > 12
    with pytest.raises(OperationCancelledError):
        image.resize(image.size, progress=lambda completed: completed == 0)


def test_seam_carving_progress_and_worker_thread():
    image = Image.new("RGB", (37, 41), "blue")
    seen = []
    caller = get_ident()

    def progress(completed):
        assert get_ident() != caller
        seen.append(completed)
        return completed < 2

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(image.seam_carve, (45, 39), progress=progress)
        with pytest.raises(OperationCancelledError):
            future.result(timeout=10)
    assert seen == [0, 1, 2]
    seen.clear()
    assert image.seam_carve((45, 39), progress=seen.append).tobytes() == image.seam_carve((45, 39)).tobytes()
    assert seen == list(range(len(seen))) and len(seen) > 3
    assert image.size == (37, 41)


@pytest.mark.parametrize(("format", "compressor"), [("PNG", Compressor.LosslessImageCompressor(effort=3)), ("JPEG", Compressor.LossyImageCompressor(effort=3)), ("WEBP", Compressor.LosslessImageCompressor(effort=2))])
def test_cancelled_compressor_preserves_destination_and_can_retry(format, compressor, tmp_path):
    source = Image.new("RGB", (401, 337), "orange")
    path = tmp_path / "existing"
    path.write_bytes(b"original")
    stream = BytesIO(b"original")
    seen = []

    def cancel(completed):
        seen.append(completed)
        return completed < 2

    for target in (path, stream):
        seen.clear()
        with pytest.raises(OperationCancelledError):
            source.save(target, format, compressor=compressor, progress=cancel)
        assert seen == [0, 1, 2]
    assert path.read_bytes() == stream.getvalue() == b"original"
    seen.clear()
    result = source.encode(format, compressor=compressor, progress=seen.append)
    assert result == source.encode(format, compressor=compressor)
    assert seen == list(range(len(seen))) and len(seen) > 3


def test_callback_exceptions_propagate_without_codec_wrapping():
    image = Image.new("RGB", (41, 49), "red")
    data = image.encode("PNG")
    failure = OSError("callback failed")

    def fail(completed):
        if completed == 1:
            raise failure

    with pytest.raises(OSError) as decode_error:
        Image.decode(data, progress=fail)
    assert decode_error.value is failure
    with pytest.raises(OSError) as encode_error:
        image.encode("PNG", compressor=Compressor.LosslessImageCompressor(), progress=fail)
    assert encode_error.value is failure
    with pytest.raises(TypeError, match="progress must be callable"):
        image.resize(image.size, progress=False)
    with pytest.raises(ValueError, match="requires a compressor"):
        image.encode("PNG", progress=fail)


@pytest.mark.parametrize("format", ["PNG", "JPEG", "WEBP"])
def test_closing_from_progress_keeps_deferred_image_closed(format):
    data = Image.new("RGB", (64, 48), "red").encode(format)
    image = Image.open(BytesIO(data))
    if format != "PNG":
        assert image.draft(None, (12, 10)) is not None

    def close(completed):
        assert completed == 0
        image.close()

    with pytest.raises(ValueError, match="closed"):
        image.load(progress=close)
    with pytest.raises(ValueError, match="closed"):
        image.load()


def test_draft_from_progress_can_be_loaded_on_retry():
    data = Image.new("RGB", (64, 48), "red").encode("JPEG")
    image = Image.open(BytesIO(data))

    def draft(completed):
        if completed == 0:
            assert image.draft("L", (12, 10)) is not None

    with pytest.raises(ValueError, match="draft"):
        image.load(progress=draft)
    assert image.mode == "L"
    assert image.size == (16, 12)
    assert isinstance(image.load()[0, 0], int)


def test_deferred_load_keeps_replacement_raster_created_during_progress():
    source = Image.new("RGB", (16, 16), "red")
    image = Image.open(BytesIO(source.encode("PNG")))

    def edit(completed):
        if completed == 0:
            image.thumbnail((8, 8))

    image.load(progress=edit)
    assert image.size == (8, 8)
    assert image.tobytes() == source.resize((8, 8)).tobytes()


@pytest.mark.parametrize("action", ["close", "seek"])
def test_lazy_seek_progress_cannot_overwrite_callback_action(action):
    first = Image.new("RGB", (16, 16), "red")
    frames = [Image.new("RGB", first.size, color) for color in ("green", "blue")]
    image = Image.decode(first.encode("PNG", save_all=True, append_images=frames), lazy=True)

    def change(completed):
        assert completed == 0
        if action == "close":
            image.close()
        else:
            image.seek(2)

    with pytest.raises(ValueError, match="closed" if action == "close" else "frame changed"):
        image.seek(1, progress=change)
    if action == "close":
        with pytest.raises(ValueError, match="closed"):
            image.load()
    else:
        assert image.tell() == 2
        assert image.tobytes() == frames[1].tobytes()
        image.seek(1)
        assert image.tobytes() == frames[0].tobytes()


def test_lazy_seek_retains_changes_to_current_frame_made_during_progress():
    first = Image.new("RGB", (16, 16), "red")
    second = Image.new("RGB", first.size, "blue")
    image = Image.decode(first.encode("PNG", save_all=True, append_images=[second]), lazy=True, frame_cache_size=2)

    def edit(completed):
        if completed == 0:
            image.thumbnail((8, 8))

    image.seek(1, progress=edit)
    image.seek(0)
    assert image.size == (8, 8)
    assert image.tobytes() == first.resize((8, 8)).tobytes()
