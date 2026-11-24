"""Adapt operation checkpoints to an optional tqdm display."""

from __future__ import annotations

__lazy_modules__ = {"importlib", "collections.abc"}

from contextlib import ExitStack, contextmanager, suppress
from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator
    from typing import Protocol

    from ._typing import ProgressCallback

    class _ProgressDisplay(Protocol):
        bar_format: str

        def update(self, amount: int, /) -> object: ...


_BAR_WIDTH = 20
_BAR_MARKER = "==="


def validate_progress(progress: ProgressCallback | None, *, show_progress: bool = False) -> None:
    """Validate the public callback and display options."""
    if progress is not None and not callable(progress):
        raise TypeError("progress must be callable or None")
    if not isinstance(show_progress, bool):
        raise TypeError("show_progress must be a boolean")


def _bar_format(indicator: str) -> str:
    return "{desc}: |" + indicator + "| {n_fmt} [{elapsed}, {rate_fmt}]"


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

        @contextmanager
        def display_bar() -> Generator[_ProgressDisplay, None, None]:
            with tqdm(desc=description, unit="checkpoint", bar_format=_bar_format(_BAR_MARKER.ljust(_BAR_WIDTH))) as display:
                yield display
                display.bar_format = _bar_format("=" * _BAR_WIDTH)

        def report(completed: int) -> bool | None:
            nonlocal bar
            if bar is None:
                bar = stack.enter_context(display_bar())
            if completed:
                offset = completed % (_BAR_WIDTH - len(_BAR_MARKER) + 1)
                bar.bar_format = _bar_format((" " * offset + _BAR_MARKER).ljust(_BAR_WIDTH))
                bar.update(1)
            return progress(completed) if progress is not None else None

        yield report
