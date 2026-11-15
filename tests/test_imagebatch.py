from __future__ import annotations

from contextlib import closing
from dataclasses import FrozenInstanceError
from functools import partial
from threading import Barrier, Event, Lock, current_thread
from weakref import ref

import pytest

from blanket import Image, ImageBatch, ImageOps


@pytest.mark.parametrize("mode", ["1", "L", "LA", "RGB", "RGBA", "P", "PA", "I", "I;16", "I;16B", "F", "RGB;F", "RGBA;F"])
def test_pipeline_matches_serial_operations(mode):
    source = Image.frombytes("RGB", (17, 13), bytes((offset * 37 + offset // 7) % 256 for offset in range(17 * 13 * 3))).convert(mode)
    pipeline = ImageBatch.Pipeline([partial(ImageOps.crop, border=1), ImageOps.flip])
    expected = ImageOps.flip(ImageOps.crop(source, 1))
    results = pipeline.map([source, source], max_workers=2)
    for result in results:
        assert (result.mode, result.size, result.bit_depth) == (expected.mode, expected.size, expected.bit_depth)
        assert result.tobytes() == expected.tobytes()


@pytest.mark.parametrize("depth", [10, 12, 16])
def test_sample_depth_preserved(depth):
    raw = b"".join(((offset * 37 + 13) % (1 << depth)).to_bytes(2, "little") for offset in range(11 * 7 * 4))
    source = Image.frombytes("RGBA", (11, 7), raw, bit_depth=depth)
    result = ImageBatch.Pipeline([ImageOps.flip]).map([source], max_workers=1)[0]
    assert result.bit_depth == depth
    assert result.tobytes() == ImageOps.flip(source).tobytes()
    assert source.tobytes() == raw


def test_in_place_steps_work_on_independent_copies():
    source = Image.new("L", (3, 2), 10)
    source.info["label"] = "source"

    def edit(image):
        image.putpixel((0, 0), 99)
        image.info["label"] = "result"
        return image

    results = ImageBatch.Pipeline([edit]).map([source] * 4, max_workers=2)
    assert source.getpixel((0, 0)) == 10
    assert source.info["label"] == "source"
    assert len({id(image) for image in results}) == 4
    assert all(image.getpixel((0, 0)) == 99 for image in results)
    results[0].close()
    assert results[1].getpixel((0, 0)) == 99
    assert source.getpixel((0, 0)) == 10


def test_nested_metadata_is_isolated_between_batch_items():
    source = Image.new("L", (3, 2), 10)
    source.info["custom"] = {"history": []}

    def edit(image):
        image.info["custom"]["history"].append("processed")
        return image

    results = ImageBatch.Pipeline([edit]).map([source] * 4, max_workers=2)
    assert source.info["custom"] == {"history": []}
    assert all(image.info["custom"] == {"history": ["processed"]} for image in results)
    assert len({id(image.info["custom"]) for image in results}) == 4


def test_configuration_is_captured_and_reusable():
    steps = [ImageOps.flip]
    pipeline = ImageBatch.Pipeline(steps)
    steps.clear()
    assert pipeline.steps == (ImageOps.flip,)
    with pytest.raises(FrozenInstanceError):
        pipeline.steps = ()
    source = Image.new("L", (2, 2), 3)
    assert pipeline(source).tobytes() == pipeline.map([source], max_workers=1)[0].tobytes()
    assert ImageBatch.Pipeline().map([], max_workers=1) == []


def test_parallel_execution_and_order():
    barrier = Barrier(2, timeout=10)
    second_finished = Event()

    def step(image):
        barrier.wait()
        if image.getpixel((0, 0)) == 0:
            assert second_finished.wait(10)
        else:
            second_finished.set()
        image.info["worker"] = current_thread().name
        return image

    sources = [Image.new("L", (1, 1), value) for value in (0, 1)]
    results = ImageBatch.Pipeline([step]).map(sources, max_workers=2)
    assert [image.getpixel((0, 0)) for image in results] == [0, 1]
    assert all(image.info["worker"].startswith("blanket-batch") for image in results)


def test_streaming_is_lazy_bounded_and_closes_early():
    consumed = []

    def sources():
        for value in range(20):
            consumed.append(value)
            yield Image.new("L", (1, 1), value)

    iterator = ImageBatch.Pipeline().iter(sources(), max_workers=2)
    assert consumed == []
    with closing(iterator):
        first = next(iterator)
        assert consumed == [0, 1]
        second = next(iterator)
        assert consumed == [0, 1, 2]
    assert consumed == [0, 1, 2]
    assert list(iterator) == []
    assert first.getpixel((0, 0)) == 0
    assert second.getpixel((0, 0)) == 1


@pytest.mark.parametrize("count", [20, 200])
def test_streaming_does_not_retain_consumed_results(count):
    references = []
    lock = Lock()

    def track(image):
        with lock:
            references.append(ref(image))
        return image

    source = Image.new("RGB", (64, 64), "red")
    pipeline = ImageBatch.Pipeline([track])
    with closing(pipeline.iter((source for _ in range(count)), max_workers=3)) as results:
        for result in results:
            with lock:
                assert sum(reference() is not None for reference in references) <= 4
            assert result.size == source.size
    assert sum(reference() is not None for reference in references) == 1


def test_close_waits_for_running_work_without_consuming_more_inputs(monkeypatch):
    started = Event()
    shutting_down = Event()
    finished = Event()
    consumed = []
    original_shutdown = ImageBatch.ThreadPoolExecutor.shutdown

    def shutdown(executor, wait=True, *, cancel_futures=False):
        shutting_down.set()
        return original_shutdown(executor, wait=wait, cancel_futures=cancel_futures)

    monkeypatch.setattr(ImageBatch.ThreadPoolExecutor, "shutdown", shutdown)

    def step(image):
        if image.getpixel((0, 0)) == 1:
            started.set()
            assert shutting_down.wait(10)
            finished.set()
        return image

    def sources():
        for value in range(10):
            consumed.append(value)
            yield Image.new("L", (1, 1), value)

    iterator = ImageBatch.Pipeline([step]).iter(sources(), max_workers=2)
    assert next(iterator).getpixel((0, 0)) == 0
    assert started.wait(10)
    iterator.close()
    assert finished.is_set()
    assert consumed == [0, 1]


@pytest.mark.parametrize("suffix", ["png", "tiff", "webp"])
def test_paths_and_images_can_be_mixed(tmp_path, suffix, monkeypatch):
    source = Image.new("RGB", (7, 5), (19, 87, 211))
    path = tmp_path / f"source.{suffix}"
    source.save(path, **({"lossless": True} if suffix == "webp" else {}))
    opened = []
    closed = []
    original_open = Image.open
    original_close = Image.Image.close

    def track_open(path):
        image = original_open(path)
        opened.append(image)
        return image

    def track_close(image):
        closed.append(image)
        original_close(image)

    monkeypatch.setattr(Image, "open", track_open)
    monkeypatch.setattr(Image.Image, "close", track_close)
    pipeline = ImageBatch.Pipeline()
    results = pipeline.map([path, str(path), source], max_workers=1)
    for result in results:
        assert result.size == source.size
        assert result.tobytes() == source.tobytes()
    assert len(opened) == 2
    assert closed == opened
    assert pipeline(path).tobytes() == source.tobytes()


@pytest.mark.parametrize("lazy", [False, True])
def test_current_frame_and_navigation_are_preserved(tmp_path, lazy):
    path = tmp_path / "animation.png"
    frames = [Image.new("RGBA", (3, 2), color) for color in ("red", "blue")]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=20)
    with Image.open(path, lazy=lazy) as source:
        source.seek(1)
        results = ImageBatch.Pipeline().map([source, source], max_workers=2)
        assert source.tell() == 1
        assert all(result.n_frames == 1 for result in results)
        assert all(result.tobytes() == source.tobytes() for result in results)
    assert results[0].tobytes() == frames[1].tobytes()
    assert ImageBatch.Pipeline()(path).tobytes() == frames[0].tobytes()


def test_worker_exception_is_preserved_in_input_order():
    failure = RuntimeError("step failed")

    def step(image):
        if image.getpixel((0, 0)) == 1:
            raise failure
        return image

    sources = [Image.new("L", (1, 1), value) for value in (0, 1, 2)]
    iterator = ImageBatch.Pipeline([step]).iter(sources, max_workers=2)
    assert next(iterator).getpixel((0, 0)) == 0
    with pytest.raises(RuntimeError) as caught:
        next(iterator)
    assert caught.value is failure
    assert list(iterator) == []


def test_input_iterator_error_is_preserved():
    def sources():
        yield Image.new("L", (1, 1))
        raise LookupError("input failed")

    with pytest.raises(LookupError, match="input failed"):
        ImageBatch.Pipeline().map(sources(), max_workers=2)


@pytest.mark.parametrize("workers", [0, -1])
def test_invalid_worker_counts(workers):
    with pytest.raises(ValueError, match="max_workers must be positive"):
        ImageBatch.Pipeline().iter([], max_workers=workers)


def test_invalid_configuration_and_inputs(tmp_path):
    with pytest.raises(TypeError, match="step 0 must be callable"):
        ImageBatch.Pipeline([None])
    with pytest.raises(TypeError):
        ImageBatch.Pipeline().iter([], max_workers=1.5)
    with pytest.raises(TypeError, match="expected an image or filesystem path"):
        ImageBatch.Pipeline().map([123], max_workers=1)
    with pytest.raises(FileNotFoundError):
        ImageBatch.Pipeline().map([tmp_path / "missing.png"], max_workers=1)
    with pytest.raises(TypeError, match="step 0 must return an image"):
        ImageBatch.Pipeline([lambda image: None]).map([Image.new("L", (1, 1))], max_workers=1)
