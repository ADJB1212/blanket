"""Adapt operation checkpoints to an optional tqdm display."""

from __future__ import annotations

__lazy_modules__ = {"importlib", "collections.abc"}

from contextlib import ExitStack, contextmanager, suppress
from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator

    from ._typing import ProgressCallback


def validate_progress(progress: ProgressCallback | None, *, show_progress: bool = False) -> None:
    """Validate the public callback and display options."""
    if progress is not None and not callable(progress):
        raise TypeError("progress must be callable or None")
    if not isinstance(show_progress, bool):
        raise TypeError("show_progress must be a boolean")


@contextmanager
def progress_display(progress: ProgressCallback | None, *, show_progress: bool, description: str) -> Generator[ProgressCallback | None, None, None]:
    """Yield a callback that reports checkpoints and close the display on exit."""
    tqdm = None
    if show_progress:
        with suppress(ImportError):
            tqdm = import_module("tqdm").tqdm
    if tqdm is None:
        yield progress
        return

    bar = None
    with ExitStack() as stack:

        def report(completed: int) -> bool | None:
            nonlocal bar
            if bar is None:
                bar = stack.enter_context(tqdm(desc=description, unit="checkpoint"))
            if completed:
                bar.update(1)
            return progress(completed) if progress is not None else None

        yield report
