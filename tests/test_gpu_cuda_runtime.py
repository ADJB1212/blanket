import gc
import weakref
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from blanket import _gpu_cuda
from blanket._gpu_cuda_plan import _Cache, _ScratchLease


@pytest.fixture
def cuda_events(monkeypatch):
    events = []
    stream = SimpleNamespace(ptr=1)

    class Event:
        def __init__(self):
            self.complete = False
            self.waits = 0
            self.queries = 0
            events.append(self)

        def record(self, stream):
            pass

        def query(self):
            self.queries += 1
            return self.complete

        def synchronize(self):
            self.waits += 1
            self.complete = True

    cuda = SimpleNamespace(Device=lambda device: nullcontext(), Event=Event, get_current_stream=lambda: stream)
    monkeypatch.setattr(_gpu_cuda, "cupy", lambda: SimpleNamespace(cuda=cuda))
    monkeypatch.setattr(_gpu_cuda, "_PENDING", {})
    monkeypatch.setattr(_gpu_cuda, "_PENDING_REGISTERED", True)
    return events, stream


class Array:
    device = SimpleNamespace(id=0)

    def __init__(self, nbytes=16):
        self.nbytes = nbytes


def test_cuda_close_does_not_wait_and_retains_pending_storage(cuda_events):
    events, _ = cuda_events
    image = _gpu_cuda.CudaImage(Array())
    pixels = weakref.ref(image.array)
    del image
    gc.collect()
    assert pixels() is not None
    assert events[0].waits == 0
    _gpu_cuda._collect_pending(0)
    assert pixels() is not None
    events[0].complete = True
    _gpu_cuda._collect_pending(0)
    assert pixels() is None


def test_cuda_releases_intermediates_without_retaining_image_chains(cuda_events):
    events, _ = cuda_events
    source = _gpu_cuda.CudaImage(Array())
    intermediate = _gpu_cuda.CudaImage(Array(), owners=(source,))
    source_pixels = weakref.ref(source.array)
    intermediate_pixels = weakref.ref(intermediate.array)
    intermediate_handle = weakref.ref(intermediate)
    result = _gpu_cuda.CudaImage(Array(), owners=(intermediate,))
    del source, intermediate
    assert intermediate_handle() is None
    events[0].complete = events[1].complete = True
    _gpu_cuda._collect_pending(0)
    assert source_pixels() is None
    assert intermediate_pixels() is not None
    result.synchronize()
    assert intermediate_pixels() is None
    assert [event.waits for event in events] == [0, 0, 1]


def test_cuda_completion_queues_release_independent_streams(cuda_events):
    events, stream = cuda_events
    first = _gpu_cuda.CudaImage(Array())
    stream.ptr = 2
    second = _gpu_cuda.CudaImage(Array())
    second_pixels = weakref.ref(second.array)
    del second
    events[1].complete = True
    _gpu_cuda._collect_pending(0)
    assert second_pixels() is None
    assert not events[0].complete
    assert first.array is not None
    assert all(event.waits == 0 for event in events)


def test_cuda_pending_reader_checks_are_linear(cuda_events):
    events, _ = cuda_events
    for _ in range(128):
        _gpu_cuda.CudaImage(Array())
    assert sum(event.queries for event in events) == 127
    for event in events:
        event.complete = True
    _gpu_cuda._collect_pending(0)
    assert sum(event.queries for event in events) == 255
    assert not _gpu_cuda._PENDING


def test_cuda_submissions_do_not_rescan_unrelated_pending_streams(cuda_events):
    events, stream = cuda_events
    for i in range(128):
        stream.ptr = i
        _gpu_cuda.CudaImage(Array())
    assert sum(event.queries for event in events) == 0
    for event in events:
        event.complete = True
    _gpu_cuda._collect_pending(0)
    assert sum(event.queries for event in events) == 128
    assert not _gpu_cuda._PENDING


def test_cuda_scratch_returns_after_completion_without_waiting(cuda_events):
    events, _ = cuda_events
    cache = _Cache(0)
    scratch = (Array(32), Array(32))
    event = _gpu_cuda.cupy().cuda.Event()
    lease = _ScratchLease(cache, scratch)
    result = _gpu_cuda.CudaImage(Array(), owners=(lease,), completion=event)
    del lease, result
    assert cache.info()[0] == 0
    event.complete = True
    assert cache.info()[0] == 64
    assert events[0].waits == 0
    assert cache.take(32) in scratch
    assert cache.info()[0] == 32
    cache.configure(16, 0)
    assert cache.info() == (0, 16, 0, 0)
    cache.put(scratch)
    assert cache.info()[0] == 0


def test_cuda_scratch_cache_accounts_for_eviction(cuda_events):
    cache = _Cache(0)
    cache.configure(64, 0)
    cache.put((Array(32), Array(32)))
    replacement = Array(48)
    cache.put((replacement,))
    assert cache.info()[0] == 48
    assert cache.take(48) is replacement
    assert cache.info()[0] == 0


def test_cuda_shutdown_waits_before_releasing_pending_storage(cuda_events):
    events, _ = cuda_events
    image = _gpu_cuda.CudaImage(Array())
    pixels = weakref.ref(image.array)
    del image
    assert pixels() is not None
    _gpu_cuda._finish_pending()
    assert events[0].waits == 1
    assert pixels() is None
    assert not _gpu_cuda._PENDING
