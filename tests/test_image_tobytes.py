"""Passing Pillow 13.0.0 contracts; see PILLOW_TESTS.md."""

from __future__ import annotations

from pillow_helpers import hopper


def test_sanity() -> None:
    data = hopper().tobytes()
    assert isinstance(data, bytes)
