"""Apply reusable pipelines to independent images in parallel."""

from __future__ import annotations

__lazy_modules__ = {"os", "collections", "collections.abc", "concurrent.futures", "contextlib", "copy", "operator"}

import os
from collections import deque
from collections.abc import Callable, Generator, Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import closing
from copy import deepcopy
from dataclasses import dataclass
from operator import index

from . import Image
from ._blanket import _deterministic_seed

type Source = Image.Image | str | os.PathLike[str]
type Step = Callable[[Image.Image], Image.Image]


def _prepare(source: Source) -> Source:
    if isinstance(source, Image.Image):
        return deepcopy(source)
    if not isinstance(source, (str, os.PathLike)):
        raise TypeError(f"expected an image or filesystem path, got {type(source).__name__}")
    return source


@dataclass(frozen=True, init=False)
class Pipeline:
    """Reuse an ordered sequence of image transforms across a batch.

    Each step receives an independent current-frame image and must return an
    image. Steps may mutate that working image. Configuration is captured as a
    tuple; callable state must be safe for concurrent use. Native Blanket pixel
    operations release the GIL; pure Python steps remain subject to the GIL
    on GIL-enabled interpreters.
    """

    steps: tuple[Step, ...]

    def __init__(self, steps: Iterable[Step] = ()) -> None:
        """Capture the transforms, rejecting non-callable steps.

        Args:
            steps: Callables that each accept an image and return an image.

        Raises:
            TypeError: A step is not callable.
        """
        configured = tuple(steps)
        for position, step in enumerate(configured):
            if not callable(step):
                raise TypeError(f"pipeline step {position} must be callable")
        object.__setattr__(self, "steps", configured)

    def __call__(self, source: Source) -> Image.Image:
        """Process one image or path synchronously, preserving the source image."""
        return self._run(_prepare(source))

    def _run(self, source: Source) -> Image.Image:
        if isinstance(source, Image.Image):
            image = source
        else:
            with closing(Image.open(source)) as opened:
                image = deepcopy(opened)
        for position, step in enumerate(self.steps):
            image = step(image)
            if not isinstance(image, Image.Image):
                raise TypeError(f"pipeline step {position} must return an image, got {type(image).__name__}")
        return image

    def map(self, sources: Iterable[Source], *, max_workers: int | None = None) -> list[Image.Image]:
        """Return processed images in input order, propagating worker exceptions.

        See :meth:`iter` for concurrency, ownership, and failure semantics. This
        method retains all output images; use the iterator for large batches.

        Args:
            sources: Images or filesystem paths, processed in order.
            max_workers: Maximum concurrent workers. Defaults to the CPU count, capped at 32.

        Returns:
            Processed images in input order.

        Raises:
            ValueError: ``max_workers`` is less than 1.
            TypeError: A source or step result is not an image.
        """
        return list(self.iter(sources, max_workers=max_workers))

    def iter(self, sources: Iterable[Source], *, max_workers: int | None = None) -> Generator[Image.Image, None, None]:
        """Yield processed images once, in input order, with bounded pending work.

        At most ``max_workers`` items are submitted at a time. The default is
        the CPU count, capped at 32. Source images are copied on the consuming
        thread before submission, including nested metadata; paths are opened
        and closed by workers. Custom metadata must support deep copying.
        With `BLANKET_DETERMINISTIC=1`, each item is processed on the consuming
        thread before the next source is read.
        Only the current frame is processed. Callers must not mutate source
        images during iteration, and own each yielded result.

        Exceptions propagate unchanged at the corresponding input position.
        Input iteration and copying errors propagate immediately.
        On failure or iterator close, queued work is cancelled and running
        calls finish before the worker pool closes. Side effects in steps
        are not rolled back. Close the iterator when stopping early (for
        example, using ``contextlib.closing``).

        Args:
            sources: Images or filesystem paths, processed in order.
            max_workers: Maximum concurrent workers. Defaults to the CPU count, capped at 32.

        Returns:
            A generator of processed images. It can be consumed only once.

        Raises:
            ValueError: ``max_workers`` is less than 1.
            TypeError: A source or step result is not an image.
        """
        workers = min(32, os.cpu_count() or 1) if max_workers is None else index(max_workers)
        if workers < 1:
            raise ValueError(f"max_workers must be positive, got {workers}")
        return self._iterate(sources, workers)

    def _iterate(self, sources: Iterable[Source], workers: int) -> Generator[Image.Image, None, None]:
        if _deterministic_seed() is not None:
            for source in sources:
                yield self(source)
            return
        inputs = iter(sources)
        pending: deque[Future[Image.Image]] = deque()
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="blanket-batch") as executor:
            try:
                for _ in range(workers):
                    try:
                        source = next(inputs)
                    except StopIteration:
                        break
                    pending.append(executor.submit(self._run, _prepare(source)))
                while pending:
                    yield pending.popleft().result()
                    try:
                        source = next(inputs)
                    except StopIteration:
                        continue
                    pending.append(executor.submit(self._run, _prepare(source)))
            finally:
                for future in pending:
                    future.cancel()


__all__ = ["Pipeline"]
