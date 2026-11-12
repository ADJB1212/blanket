from __future__ import annotations

import struct
import sys
from io import BytesIO
from random import Random
from threading import Event, Thread
from time import monotonic
from typing import TYPE_CHECKING

import pytest
from helpers import assert_image_properties

from blanket import Image, ImageCms, _blanket

if TYPE_CHECKING:
    from collections.abc import Callable


def assert_python_progress[T](operation: Callable[[], T], *, timeout: float = 5.0) -> T:
    """Repeat a native call until the worker runs within the timeout."""
    if not getattr(sys, "_is_gil_enabled", lambda: True)():
        pytest.skip("requires a GIL-enabled interpreter")
    ready, start, progressed = Event(), Event(), Event()

    def worker() -> None:
        ready.set()
        start.wait()
        progressed.set()

    thread = Thread(target=worker)
    thread.start()
    interval = sys.getswitchinterval()
    try:
        assert ready.wait(5)
        # Only the native call may yield Python execution during this check.
        sys.setswitchinterval(60)
        start.set()
        deadline = monotonic() + timeout
        result = operation()
        while not progressed.is_set() and monotonic() < deadline:
            result = operation()
        assert progressed.is_set(), "native work held the GIL"
        return result
    finally:
        sys.setswitchinterval(interval)
        start.set()
        thread.join(5)
        assert not thread.is_alive()


def test_small_copy_releases_gil() -> None:
    image = Image.new("L", (1, 1), 73).getim()
    result = assert_python_progress(image.copy)
    assert result.getpixel((0, 0)) == 73


@pytest.mark.parametrize("mode", ["RGB;F", "RGBA;F"])
@pytest.mark.parametrize("operation", ["convert", "getchannel"])
def test_float_color_work_releases_gil(mode: str, operation: str) -> None:
    channels = 4 if mode == "RGBA;F" else 3
    pixel = struct.pack("<" + "f" * channels, *([0.25] * channels))
    image = Image.frombytes(mode, (1024, 1024), pixel * (1024 * 1024)).getim()
    if operation == "convert":
        result = assert_python_progress(lambda: _blanket.image_convert(image, "RGBA", 8))
        expected = (64, 64, 64, 64 if channels == 4 else 255)
    else:
        result = assert_python_progress(lambda: image.getchannel(channels - 1))
        expected = 0.25
    assert result.getpixel((1023, 1023)) == expected


@pytest.mark.parametrize("format", ["JPEG", "JXL"])
def test_streaming_decode_releases_gil(format: str) -> None:
    source = Image.frombytes("RGB", (1024, 1024), Random(19).randbytes(1024 * 1024 * 3))
    data = source.encode(format)
    decoder = _blanket.StreamingDecoder(format)
    assert_python_progress(lambda: decoder.feed(data))
    # Snapshot copies and close can finish before another thread is scheduled.
    preview = decoder.image
    result = decoder.close()
    assert preview is not None
    assert _blanket.image_tobytes(preview) == _blanket.image_tobytes(result) == Image.open(BytesIO(data)).tobytes()


@pytest.mark.parametrize("proof", [False, True])
def test_cms_creation_releases_gil(proof: bool) -> None:
    source = ImageCms.createProfile("Display P3").tobytes()
    output = ImageCms.createProfile("sRGB").tobytes()
    proof_profile = ImageCms.createProfile("Adobe RGB").tobytes() if proof else None
    flags = (65 << 16) | (int(ImageCms.Flags.SOFTPROOFING) if proof else 0)
    transform = assert_python_progress(lambda: _blanket._CmsTransform(source, output, "RGBA", "RGBA", 0, flags, proof_profile, 0))
    image = Image.new("RGBA", (257, 19), (40, 80, 120, 73))
    result = transform.apply(image.getim())
    assert _blanket.image_tobytes(result.getchannel(3)) == bytes([73]) * (257 * 19)


@pytest.mark.parametrize("floating", [False, True])
@pytest.mark.parametrize("operation", ["weights", "lut"])
def test_gpu_preparation_releases_gil(floating: bool, operation: str) -> None:
    if not hasattr(_blanket, "_gpu_weights"):
        pytest.skip("GPU bindings are not built")
    if operation == "weights":
        stride, values = assert_python_progress(lambda: _blanket._gpu_weights(131072, 262144, 1, floating))
        assert len(values) == stride * 262144
        for offset in (0, stride, len(values) - stride):
            assert sum(values[offset + 2 : offset + stride]) == pytest.approx(1.0)
    else:
        table = [0.25] * (65 * 65 * 65 * 4)
        values = assert_python_progress(lambda: _blanket._gpu_lut_controls([65, 65, 65], 4, table, floating))
        assert len(values) == len(table) + 6
        assert values[6:] == [0.25 if floating else 4080.0] * len(table)


@pytest.mark.parametrize("depth", [8, 10, 12, 16])
def test_png_compression_releases_gil(depth: int) -> None:
    output = BytesIO()
    pixel = b"\x7f" if depth == 8 else ((1 << depth) - 1).to_bytes(2, "little")
    data = pixel * (1024 * 1024)
    writer = _blanket.PngStreamWriter(output, "L", (1024, 1024), depth, 6)
    pending = data

    def write_pending() -> None:
        nonlocal pending
        writer.write(pending)
        pending = b""

    assert_python_progress(write_pending)
    writer.close()
    image = Image.open(BytesIO(output.getvalue()))
    assert_image_properties(image, bit_depth=depth, pixels=data)
