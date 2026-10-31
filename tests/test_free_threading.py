from __future__ import annotations

import gc
import os
import subprocess
import sys
import sysconfig
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from threading import Barrier, Event

import numpy as np
import pytest

from blanket import Image, ImageCms, ImageFile, ImageFilter, _blanket

pytestmark = pytest.mark.skipif(not sysconfig.get_config_var("Py_GIL_DISABLED"), reason="requires free-threaded CPython")


def test_import_keeps_gil_disabled() -> None:
    environment = os.environ.copy()
    environment.pop("PYTHON_GIL", None)
    subprocess.run(
        [sys.executable, "-W", "error::RuntimeWarning", "-c", "import sys; assert not sys._is_gil_enabled(); import blanket; assert not sys._is_gil_enabled()"],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )


@pytest.mark.parametrize("format", ["PNG", "JPEG", "JXL", "TIFF", "WEBP", "GIF", "AVIF", "HEIF", "JPEG2000"])
def test_concurrent_codec_instances(format: str) -> None:
    barrier = Barrier(4)
    source = Image.frombytes("RGB", (37, 29), bytes((i * 37 + i // 7) % 256 for i in range(37 * 29 * 3)))
    expected = Image.open(BytesIO(source.encode(format))).tobytes()

    def run(index: int) -> None:
        barrier.wait(timeout=30)
        for _ in range(8):
            encoded = source.encode(format)
            image = Image.open(BytesIO(encoded))
            assert image.tobytes() == expected
            assert image.copy().resize((19 + index, 13)).size == (19 + index, 13)
            if format in ("JPEG", "JXL"):
                parser = ImageFile.Parser()
                for start in range(0, len(encoded), 97):
                    parser.feed(encoded[start : start + 97])
                assert parser.close().tobytes() == expected

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(run, range(4)))
    assert not sys._is_gil_enabled()


@pytest.mark.parametrize("dtype", ["<u2", ">u2", "<i4", "<f4", ">f4", "<f8", ">f8"])
def test_concurrent_array_snapshots(dtype: str) -> None:
    values = np.arange(67 * 37, dtype=dtype).reshape(37, 67)
    barrier = Barrier(4)

    def run(index: int) -> None:
        view = values[::-1, ::2] if index % 2 else values
        expected = view.astype("<f4").tobytes() if view.dtype.kind == "f" else view.tobytes()
        barrier.wait(timeout=30)
        for _ in range(32):
            assert Image.fromarray(view).tobytes() == expected

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(run, range(4)))


def test_shared_cms_transform_during_gc() -> None:
    source, target = (ImageCms.createProfile(space) for space in ("Display P3", "sRGB"))
    transform = ImageCms.buildTransform(source, target, "RGBA", "RGBA")
    image = Image.new("RGBA", (257, 129), (40, 80, 120, 73))
    expected = transform.apply(image).tobytes()
    stop = Event()
    barrier = Barrier(5)

    def collect() -> None:
        barrier.wait(timeout=30)
        while not stop.is_set():
            gc.collect()

    def run(index: int) -> None:
        barrier.wait(timeout=30)
        for _ in range(16):
            result = transform.apply(image)
            assert result.tobytes() == expected
            assert result.filter(ImageFilter.BoxBlur(index)).size == image.size

    with ThreadPoolExecutor(max_workers=5) as pool:
        collector = pool.submit(collect)
        try:
            list(pool.map(run, range(4)))
        finally:
            stop.set()
        collector.result(timeout=30)


def test_native_writer_rejects_concurrent_mutation() -> None:
    class CheckingSink(BytesIO):
        writer = None
        attempts = 0

        def write(self, data: bytes) -> int:
            if self.writer is not None:
                self.attempts += 1
                with ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(self.writer.abort)
                    with pytest.raises(RuntimeError, match="borrowed"):
                        future.result(timeout=10)
            return super().write(data)

    sink = CheckingSink()
    writer = _blanket.PngStreamWriter(sink, "L", (257, 37), 8, 0)
    sink.writer = writer
    raw = bytes(range(256)) * 37 + bytes(37)
    writer.write(raw)
    writer.close()
    sink.writer = None
    assert sink.attempts > 0
    assert Image.open(BytesIO(sink.getvalue())).tobytes() == raw
