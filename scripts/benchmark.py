#!/usr/bin/env python3
"""Benchmark Blanket's supported operations against Pillow."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from collections.abc import Callable
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image as PillowImage

from blanket import Image as BlanketImage


def make_rgb(width: int, height: int) -> bytes:
    return bytes((index * 37 + (index // 3) * 11) & 0xFF for index in range(width * height * 3))


def pillow_payload(image: PillowImage.Image, format: str, **options: object) -> bytes:
    output = BytesIO()
    image.save(output, format, **options)
    return output.getvalue()


def measure(operation: Callable[[], object], warmups: int, iterations: int) -> float:
    for _ in range(warmups):
        operation()
    samples: list[float] = []
    gc.disable()
    try:
        for _ in range(iterations):
            started = time.perf_counter()
            operation()
            samples.append(time.perf_counter() - started)
    finally:
        gc.enable()
    return statistics.median(samples)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--skip-jxl", action="store_true")
    parser.add_argument("--json", type=Path, dest="json_path")
    arguments = parser.parse_args()
    for name in ("width", "height", "iterations"):
        if getattr(arguments, name) < 1:
            parser.error(f"--{name} must be positive")
    if arguments.warmups < 0:
        parser.error("--warmups cannot be negative")
    return arguments


def main() -> None:
    args = parse_args()
    size = (args.width, args.height)
    raw = make_rgb(*size)
    blanket = BlanketImage.frombytes("RGB", size, raw)
    pillow = PillowImage.frombytes("RGB", size, raw)
    png = pillow_payload(pillow, "PNG", compress_level=6)
    jpeg = pillow_payload(pillow, "JPEG", quality=85)

    def pillow_load(payload: bytes) -> None:
        with PillowImage.open(BytesIO(payload)) as image:
            image.load()

    comparisons: list[tuple[str, Callable[[], object], Callable[[], object]]] = [
        (
            "load PNG",
            lambda: BlanketImage.open(BytesIO(png)),
            lambda: pillow_load(png),
        ),
        (
            "load JPEG",
            lambda: BlanketImage.open(BytesIO(jpeg)),
            lambda: pillow_load(jpeg),
        ),
        (
            "save PNG",
            lambda: blanket.save(BytesIO(), "PNG", compress_level=6),
            lambda: pillow.save(BytesIO(), "PNG", compress_level=6),
        ),
        (
            "save JPEG",
            lambda: blanket.save(BytesIO(), "JPEG", quality=85),
            lambda: pillow.save(BytesIO(), "JPEG", quality=85),
        ),
        ("convert RGB-L", lambda: blanket.convert("L"), lambda: pillow.convert("L")),
        (
            "convert RGB-RGBA",
            lambda: blanket.convert("RGBA"),
            lambda: pillow.convert("RGBA"),
        ),
    ]

    results: list[dict[str, Any]] = []
    for name, blanket_operation, pillow_operation in comparisons:
        blanket_seconds = measure(blanket_operation, args.warmups, args.iterations)
        pillow_seconds = measure(pillow_operation, args.warmups, args.iterations)
        results.append(
            {
                "operation": name,
                "blanket_ms": blanket_seconds * 1000,
                "pillow_ms": pillow_seconds * 1000,
                "blanket_speedup": pillow_seconds / blanket_seconds,
            }
        )

    if not args.skip_jxl:
        jxl_output = BytesIO()
        blanket.save(jxl_output, "JXL", lossless=True, effort=1)
        jxl = jxl_output.getvalue()
        for name, operation in (
            ("load JXL", lambda: BlanketImage.open(BytesIO(jxl))),
            (
                "save JXL",
                lambda: blanket.save(BytesIO(), "JXL", lossless=True, effort=1),
            ),
        ):
            blanket_seconds = measure(operation, args.warmups, args.iterations)
            results.append(
                {
                    "operation": name,
                    "blanket_ms": blanket_seconds * 1000,
                    "pillow_ms": None,
                    "blanket_speedup": None,
                }
            )

    print(f"image: {args.width}x{args.height}; median of {args.iterations} runs")
    print(f"{'operation':<18} {'Blanket ms':>12} {'Pillow ms':>12} {'speedup':>10}")
    for result in results:
        pillow_ms = result["pillow_ms"]
        speedup = result["blanket_speedup"]
        print(
            f"{result['operation']:<18} {result['blanket_ms']:>12.3f} "
            f"{pillow_ms if pillow_ms is not None else '-':>12.3f} "
            f"{speedup if speedup is not None else '-':>10.2f}"
            if pillow_ms is not None
            else f"{result['operation']:<18} {result['blanket_ms']:>12.3f} {'-':>12} {'-':>10}"
        )

    if args.json_path is not None:
        args.json_path.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
