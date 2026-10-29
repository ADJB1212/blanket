from concurrent.futures import CancelledError, ThreadPoolExecutor
from threading import Event as ThreadEvent

import pytest

from blanket import Image, ImageGpu


@pytest.fixture
def metal():
    if not ImageGpu.capabilities()["available"]:
        pytest.skip("native Metal required")


@pytest.mark.parametrize("unified", [False, True])
def test_async_transfers_snapshot_and_lifetime(metal, unified):
    source = Image.new("RGBA", (257, 129), (30, 60, 90, 120))
    expected = source.tobytes()
    upload = ImageGpu.from_image_async(source, unified_memory=unified)
    source.putpixel((0, 0), (0, 0, 0, 0))
    source.close()
    resident = upload.result(timeout=10)
    assert resident.unified_memory == unified
    completion = resident.record_event()
    result = resident.convert("RGBA")
    download = result.to_image_async()
    resident.close()
    result.close()
    completion.wait()
    assert completion.done()
    assert download.result(timeout=10).tobytes() == expected
    assert upload.done() and download.done()
    assert not upload.cancel()


def test_transfer_cancellation_before_execution(metal, monkeypatch):
    from blanket import _gpu_async

    gate = ThreadEvent()
    with ThreadPoolExecutor(max_workers=1) as executor:
        blocker = executor.submit(gate.wait)
        monkeypatch.setattr(_gpu_async, "_executor", lambda: executor)
        try:
            transfer = ImageGpu.from_image_async(Image.new("L", (17, 9), 57))
            assert transfer.cancel()
            assert transfer.cancelled() and transfer.done()
            with pytest.raises(CancelledError):
                transfer.result()
        finally:
            gate.set()
            blocker.result(timeout=10)


def test_async_errors_and_closed_images(metal):
    failed = ImageGpu.from_image_async(Image.new("L", (3, 5)), backend="invalid")
    with pytest.raises(ValueError, match="backend"):
        failed.result(timeout=10)
    gpu = ImageGpu.from_image(Image.new("L", (3, 5)))
    gpu.close()
    with pytest.raises(ValueError, match="closed"):
        gpu.to_image_async()
    with pytest.raises(ValueError, match="closed"):
        gpu.record_event()


def test_native_stream_ordering_and_event_ownership(metal):
    producer, consumer = ImageGpu.Stream(), ImageGpu.Stream()
    with producer:
        gpu = ImageGpu.from_image(Image.new("L", (257, 129), 57)).invert()
        completion = gpu.record_event()
    with consumer:
        consumer.wait_event(completion)
        result = gpu.invert()
    gpu.close()
    consumer.synchronize()
    completion.wait()
    assert completion.done()
    assert result.to_image().tobytes() == bytes([57]) * (257 * 129)


def test_stream_context_validation(metal):
    with pytest.raises(ValueError):
        ImageGpu.Stream("metal", device=1)
    with pytest.raises(ValueError):
        ImageGpu.Stream("invalid")
    source = ImageGpu.from_image(Image.new("L", (3, 5)))
    event = source.record_event()
    event.device = "cuda:0"
    with pytest.raises(ValueError, match="context"):
        ImageGpu.Stream().wait_event(event)


def test_async_unsigned_roundtrip(metal):
    source = Image.frombytes("L", (3, 1), b"\x00\x00\x80\x00\xff\x03", bit_depth=10)
    resident = ImageGpu.from_image_async(source).result(timeout=10)
    event = resident.record_event()
    transfer = resident.to_image_async()
    resident.close()
    event.wait()
    assert transfer.result(timeout=10).tobytes() == source.tobytes()


def test_async_reduction_retains_closed_inputs(metal):
    gpu = ImageGpu.from_image(Image.new("L", (17, 9), 57))
    histogram = gpu.histogram()
    statistics = gpu.statistics()
    event = statistics.record_event()
    counts = histogram.to_cpu_async()
    moments = statistics.to_cpu_async()
    gpu.close()
    histogram.close()
    statistics.close()
    event.wait()
    assert counts.result(timeout=10)[0][57] == 17 * 9
    assert moments.result(timeout=10)[0]["mean"] == 57


def test_running_transfer_cannot_be_cancelled(metal, monkeypatch):
    started, release = ThreadEvent(), ThreadEvent()
    original = ImageGpu.from_image

    def blocked(*args, **kwargs):
        started.set()
        assert release.wait(timeout=10)
        return original(*args, **kwargs)

    monkeypatch.setattr(ImageGpu, "from_image", blocked)
    transfer = ImageGpu.from_image_async(Image.new("L", (17, 9), 57))
    try:
        assert started.wait(timeout=10)
        assert transfer.running()
        assert not transfer.cancel()
        with pytest.raises(TimeoutError):
            transfer.result(timeout=0)
    finally:
        release.set()
    assert transfer.result(timeout=10).to_image().tobytes() == bytes([57]) * (17 * 9)


def test_cuda_async_transfers_and_stream_events():
    if not ImageGpu.capabilities("cuda")["available"]:
        pytest.skip("CUDA feature and device required")
    source = Image.new("RGBA", (257, 129), (30, 60, 90, 120))
    producer = ImageGpu.Stream("cuda")
    consumer = ImageGpu.Stream("cuda")
    with producer:
        upload = ImageGpu.from_image_async(source, backend="cuda")
    resident = upload.result(timeout=30)
    with producer:
        result = resident.invert()
        event = result.record_event()
    with consumer:
        consumer.wait_event(event)
        result = result.invert()
        download = result.to_image_async()
    resident.close()
    result.close()
    assert download.result(timeout=30).tobytes() == source.tobytes()
    consumer.synchronize()
    event.wait()
    assert event.done()


def test_cuda_pinned_transfer_ordering_without_device(monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace

    import numpy as np

    from blanket import _gpu_cuda

    pending = []
    allocations = []
    calls = []

    class FakeStream:
        def wait_event(self, event):
            calls.append("wait")

        def synchronize(self):
            while pending:
                pending.pop(0)()

    stream = FakeStream()

    class FakeEvent:
        def record(self, selected):
            assert selected is stream
            calls.append("record")

        def synchronize(self):
            calls.append("complete")
            stream.synchronize()

    class FakeArray:
        def __init__(self, values):
            self.values = values.copy()
            self.device = SimpleNamespace(id=0)
            self.shape, self.size = values.shape, values.size
            self.dtype, self.nbytes = values.dtype, values.nbytes

        def get(self, *, out, stream, blocking):
            assert not blocking
            calls.append("get")
            pending.append(lambda: np.copyto(out, self.values))

    def allocate(count):
        allocations.append(count)
        return bytearray(count)

    def asarray(host, *, blocking):
        assert not blocking
        return FakeArray(host)

    cp = SimpleNamespace(cuda=SimpleNamespace(Device=lambda _: nullcontext(), Event=FakeEvent, get_current_stream=lambda: stream, alloc_pinned_memory=allocate), asarray=asarray)
    monkeypatch.setattr(_gpu_cuda, "cupy", lambda: cp)
    monkeypatch.setattr(_gpu_cuda, "require", lambda _: None)
    resource = _gpu_cuda.upload(bytes(range(24)), (2, 3, 4), False, 0)
    try:
        assert len(resource.owners) == 2
        assert resource.download() == bytes(range(24))
        assert allocations == [24, 24]
        assert calls.index("wait") < calls.index("get") < calls.index("complete")
        assert not pending
    finally:
        resource.synchronize()
        del resource.event
