"""Benchmark font layout and disk strokes after a release extension build.

Run `uv run --no-sync maturin develop --release --uv`, then
`uv run --no-sync scripts/benchmark_font.py`. Pillow supplies only the
in-memory Aileron font fixture; timed operations all run through Blanket.
"""

from __future__ import annotations

from io import BytesIO
from statistics import median
from timeit import repeat
from typing import TYPE_CHECKING

from PIL import ImageFont as PILFont

from blanket import ImageFont
from blanket._blanket import _font_stroke

if TYPE_CHECKING:
    from collections.abc import Callable


def main() -> None:
    font = ImageFont.truetype(BytesIO(PILFont.load_default().font_bytes), 32)
    text = "The quick brown fox jumps over the lazy dog. " * 8
    mask = font.getmask(text)
    cases: dict[str, tuple[Callable[[], object], int]] = {
        "getlength": (lambda: font.getlength(text), 300),
        "getbbox": (lambda: font.getbbox(text), 100),
        "getmask": (lambda: font.getmask(text), 20),
        "stroke_2": (lambda: _font_stroke(bytes(mask), mask.size, 2), 10),
        "stroke_8": (lambda: _font_stroke(bytes(mask), mask.size, 8), 3),
        "stroke_24": (lambda: _font_stroke(bytes(mask), mask.size, 24), 1),
    }
    print(f"Aileron 32 px, {len(text)} characters, {mask.size[0]} x {mask.size[1]} mask; median of five batches")
    for name, (run, count) in cases.items():
        elapsed = median(repeat(run, number=count, repeat=5)) / count
        print(f"{name}: {elapsed * 1e6:.1f} us", flush=True)


if __name__ == "__main__":
    main()
