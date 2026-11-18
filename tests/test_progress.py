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
