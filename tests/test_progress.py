import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from io import BytesIO
from struct import pack
from threading import get_ident
from types import SimpleNamespace
from unittest.mock import ANY, Mock, call
from zlib import compress

import pytest
from helpers import assert_pixels_equal, encode_image, png_chunk
from PIL import Image as Pillow

from blanket import Compressor, Image, OperationCancelledError


@pytest.mark.parametrize("mode", ["1", "P", "RGB", "I;16"])
def test_png_progress_preserves_pixels_and_can_cancel_deferred_load(mode):
    size = (41, 49)
    source = Pillow.frombytes("L", size, bytes(i % 256 for i in range(size[0] * size[1]))).convert(mode)
    data = encode_image(source, "PNG")
    expected = Image.decode(data)
    seen = []
    assert_pixels_equal(Image.decode(data, progress=seen.append), expected)
    assert seen == list(range(len(seen))) and len(seen) > 3
    assert_pixels_equal(Image.open(BytesIO(data), progress=lambda _: True), expected)
    with pytest.raises(OperationCancelledError):
        Image.decode(data, progress=lambda completed: completed < 1)

    if mode == "RGB":
        image = Image.open(BytesIO(data))
        with pytest.raises(OperationCancelledError):
            image.load(progress=lambda completed: completed < 1)
        assert_pixels_equal(image, expected)


def test_interlaced_png_progress_preserves_adam7_output():
    width, height = 29, 37
    pixels = bytes(i % 251 for i in range(width * height))
    passes = [(0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8), (2, 0, 4, 4), (0, 2, 2, 4), (1, 0, 2, 2), (0, 1, 1, 2)]
    rows = b"".join(b"\0" + pixels[y * width + x : (y + 1) * width : dx] for x, top, dx, dy in passes for y in range(top, height, dy))

    data = b"\x89PNG\r\n\x1a\n" + png_chunk(b"IHDR", pack(">IIBBBBB", width, height, 8, 0, 0, 0, 1)) + png_chunk(b"IDAT", compress(rows)) + png_chunk(b"IEND", b"")
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
    assert_pixels_equal(lazy, eager)


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


@pytest.mark.parametrize("operation", ["identity_resize", "resize", "seam_carve"])
def test_geometric_progress_uses_pixels_and_metadata_from_the_starting_frame(operation):
    first = Image.new("RGB", (16, 16), "red")
    second = Image.new("RGB", first.size, "blue")
    data = first.encode("PNG", save_all=True, append_images=[second], duration=[30, 50])
    with Image.decode(data, lazy=True) as source:

        def change_frame(completed):
            if completed == 0:
                source.seek(1)

        if operation == "identity_resize":
            result = source.resize(source.size, progress=change_frame)
        elif operation == "resize":
            result = source.resize((12, 12), progress=change_frame)
        else:
            result = source.seam_carve((12, 12), progress=change_frame)
        assert result.getpixel((0, 0)) == (255, 0, 0)
        assert result.info["duration"] == 30
        assert source.tell() == 1


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


@pytest.fixture
def mock_tqdm(monkeypatch):
    bar = Mock()

    @contextmanager
    def display(**kwargs):
        try:
            yield bar
        finally:
            bar.close()

    tqdm = Mock(side_effect=display)
    monkeypatch.setitem(sys.modules, "tqdm", SimpleNamespace(tqdm=tqdm))
    return tqdm, bar


@pytest.mark.parametrize("compressor_type", [Compressor.LosslessImageCompressor, Compressor.LossyImageCompressor])
def test_compressor_tqdm_progress(compressor_type, mock_tqdm):
    tqdm, bar = mock_tqdm
    frames = []
    bar.update.side_effect = lambda _: frames.append(bar.bar_format)
    image = Image.new("RGB", (17, 19), "red")
    baseline = image.encode("PNG", compressor=compressor_type(effort=1))
    tqdm.assert_not_called()

    compressor = compressor_type(effort=1, show_progress=True)
    seen = []
    assert image.encode("PNG", compressor=compressor, progress=seen.append) == baseline
    tqdm.assert_called_once_with(desc="Compressing", unit="checkpoint", bar_format=ANY)
    assert seen == list(range(len(seen))) and len(seen) > 2
    assert bar.update.call_args_list == [call(1)] * (len(seen) - 1)
    assert "| ===                |" in frames[0]
    assert "|  ===               |" in frames[1]
    assert "|====================|" in bar.bar_format
    bar.close.assert_called_once_with()

    tqdm.reset_mock()
    bar.reset_mock()
    assert encode_image(image, "PNG", compressor=compressor) == baseline
    tqdm.assert_called_once()
    assert bar.update.call_args_list == [call(1)] * (len(seen) - 1)
    bar.close.assert_called_once_with()

    tqdm.reset_mock()
    bar.reset_mock()
    output = BytesIO(b"original")
    with pytest.raises(OperationCancelledError):
        image.save(output, "PNG", compressor=compressor, progress=lambda completed: completed < 1)
    assert output.getvalue() == b"original"
    tqdm.assert_called_once()
    bar.update.assert_called_once_with(1)
    assert "| ===                |" in bar.bar_format
    bar.close.assert_called_once_with()


def test_compressor_tqdm_renders_bar(capsys):
    pytest.importorskip("tqdm")
    image = Image.new("RGB", (17, 19), "red")
    image.encode("PNG", compressor=Compressor.LosslessImageCompressor(effort=1, show_progress=True))
    display = capsys.readouterr().err
    assert "Compressing: |===                 | 0 [" in display
    assert "Compressing: |====================|" in display


def test_compressor_progress_when_tqdm_is_missing(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "tqdm", None)
    image = Image.new("RGB", (17, 19), "red")
    seen = []
    compressor = Compressor.LosslessImageCompressor(effort=1)
    baseline = image.encode("PNG", compressor=compressor, progress=seen.append)
    reported = []
    assert encode_image(image, "PNG", compressor=Compressor.LosslessImageCompressor(effort=1, show_progress=True), progress=reported.append) == baseline
    assert reported == seen
    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize(
    ("operation", "description"),
    [
        (lambda image, **options: Image.open(BytesIO(image.encode("PNG")), **options), "Decoding"),
        (lambda image, **options: Image.decode(image.encode("PNG"), **options), "Decoding"),
        (lambda image, **options: Image.open_region(BytesIO(image.encode("TIFF")), (1, 1, 15, 17), **options), "Decoding region"),
        (lambda image, **options: Image.open_region(BytesIO(image.encode("PNG")), (1, 1, 15, 17), fallback="full", **options), "Decoding region"),
        (lambda image, **options: image.resize((15, 17), **options), "Resizing"),
        (lambda image, **options: image.seam_carve((15, 17), **options), "Seam carving"),
    ],
    ids=["open", "decode", "region", "region-fallback", "resize", "seam-carve"],
)
def test_image_tqdm_progress(operation, description, mock_tqdm):
    tqdm, bar = mock_tqdm
    image = Image.new("RGB", (17, 19), "red")
    expected = operation(image)
    tqdm.assert_not_called()

    seen = []
    result = operation(image, progress=seen.append, show_progress=True)
    assert result.size == expected.size
    assert_pixels_equal(result, expected)
    tqdm.assert_called_once_with(desc=description, unit="checkpoint", bar_format=ANY)
    assert seen == list(range(len(seen))) and len(seen) >= 2
    assert bar.update.call_args_list == [call(1)] * (len(seen) - 1)
    bar.close.assert_called_once_with()

    tqdm.reset_mock()
    bar.reset_mock()
    with pytest.raises(OperationCancelledError):
        operation(image, progress=lambda _: False, show_progress=True)
    bar.close.assert_called_once_with()


def test_tqdm_only_displays_load_and_seek_when_decoding(mock_tqdm):
    tqdm, _ = mock_tqdm
    source = Image.new("RGB", (17, 19), "red")
    image = Image.open(BytesIO(source.encode("PNG")))
    assert image.load(show_progress=True)[0, 0] == (255, 0, 0)
    tqdm.assert_called_once_with(desc="Decoding", unit="checkpoint", bar_format=ANY)
    image.load(show_progress=True)
    mapped = Image.frombuffer("L", (2, 2), bytearray(4))
    assert mapped.load(show_progress=True)[0, 0] == 0
    tqdm.assert_called_once()

    tqdm.reset_mock()
    second = Image.new("RGB", source.size, "blue")
    sequence = Image.decode(source.encode("PNG", save_all=True, append_images=[second]), lazy=True, frame_cache_size=2)
    sequence.seek(1, show_progress=True)
    assert_pixels_equal(sequence, second)
    tqdm.assert_called_once_with(desc="Decoding frame", unit="checkpoint", bar_format=ANY)
    sequence.seek(0, show_progress=True)
    sequence.seek(1, show_progress=True)
    tqdm.assert_called_once()


def test_open_show_progress_without_tqdm_keeps_deferred_decode(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "tqdm", None)
    source = Image.new("RGB", (17, 19), "red")
    image = Image.open(BytesIO(source.encode("PNG")), show_progress=True)
    seen = []
    assert image.load(progress=seen.append, show_progress=True)[0, 0] == (255, 0, 0)
    assert seen == list(range(len(seen))) and len(seen) >= 2
    assert capsys.readouterr() == ("", "")


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
