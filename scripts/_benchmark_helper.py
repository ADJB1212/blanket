"""Benchmark fixtures, measurements, reporting, and baseline helpers."""

from __future__ import annotations

import gc
import json
import math
import os
import platform
import struct
import subprocess
import sys
from datetime import UTC, datetime
from functools import partial
from io import BytesIO
from pathlib import Path
from statistics import geometric_mean, median
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import argparse
    from collections.abc import Callable
    from types import ModuleType
    from typing import Any

    from PIL import Image as PillowImage


def memory_profiler() -> ModuleType:
    """Load the development profiler without adding a runtime dependency."""
    if sys.platform not in ("darwin", "linux"):
        raise OSError(f"allocation tracking is unavailable on {sys.platform}; Memray supports macOS and Linux")
    try:
        import memray
    except ImportError as error:
        raise OSError("allocation tracking requires Memray; run uv sync --group test") from error
    return memray


def measure_memory(operation: Callable[[], object], iterations: int) -> dict[str, float]:
    """Return median, p25, and p75 peak allocated bytes over independent runs.

    Track Python and native allocations, including returned data and transient
    buffers. Input fixtures and allocations made before each run are excluded.
    """
    if iterations < 1:
        raise ValueError("iterations must be positive")
    profiler = memory_profiler()
    samples: list[int] = []
    with TemporaryDirectory(prefix="blanket-memory-") as directory:
        destination = profiler.FileDestination(Path(directory) / "allocations.bin", overwrite=True, compress_on_exit=False)
        for _ in range(iterations):
            gc.collect()
            with profiler.Tracker(destination=destination, trace_python_allocators=True, file_format=profiler.FileFormat.AGGREGATED_ALLOCATIONS):
                result = operation()
            del result
            with profiler.FileReader(destination.path) as reader:
                samples.append(reader.metadata.peak_memory)
    samples.sort()
    return {"median_bytes": median(samples), "p25_bytes": samples[len(samples) // 4], "p75_bytes": samples[(len(samples) * 3) // 4]}


Console = Panel = Table = Text = None

NO_RICH = os.environ.get("NO_RICH")

if not NO_RICH:
    try:
        from rich import console as rich_console
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text

        Console = rich_console.Console
    except ImportError:
        NO_RICH = True


def make_rgb(width: int, height: int) -> bytes:
    return bytes((index * 37 + (index // 3) * 11) & 0xFF for index in range(width * height * 3))


def make_rgba(width: int, height: int) -> bytes:
    return bytes((index * 37 + (index // 3) * 11) & 0xFF for index in range(width * height * 4))


def make_gray(width: int, height: int) -> bytes:
    return bytes((index * 37) & 0xFF for index in range(width * height))


def make_one(width: int, height: int) -> bytes:
    return bytes((index * 37 + 0xA5) & 0xFF for index in range(((width + 7) // 8) * height))


def make_la(width: int, height: int) -> bytes:
    return bytes(value for index in range(width * height) for value in ((index * 37) & 0xFF, (index * 53) & 0xFF))


def make_pa(width: int, height: int) -> bytes:
    return bytes(value for index in range(width * height) for value in (index & 0xFF, (index * 53) & 0xFF))


def make_integer(width: int, height: int, mode: str) -> bytes:
    samples = np.arange(width * height, dtype=np.int32) * 37
    if mode == "I":
        return ((samples % 131072) - 32768).astype("<i4").tobytes()
    return (samples % 65536).astype(">u2" if mode == "I;16B" else "<u2").tobytes()


def make_float(width: int, height: int) -> np.ndarray:
    samples = (np.arange(width * height, dtype=np.uint32) * 37 % 1024).astype(np.float32)
    return (samples - 128.5).reshape(height, width)


def pillow_payload(image: PillowImage.Image, fmt: str, **options: object) -> bytes:
    output = BytesIO()
    image.save(output, fmt, **options)
    return output.getvalue()


def legacy_load_payloads(size: tuple[int, int], rgb: bytes, gray: bytes, formats: set[str] | None = None) -> list[tuple[str, bytes]]:
    """Build read-only raster fixtures outside the timed operations."""
    w, h = size
    payloads: list[tuple[str, bytes]] = []
    if formats is None or "FITS" in formats:
        for bits, mode, samples in (
            (8, "L", gray),
            (16, "I;16", np.frombuffer(make_integer(w, h, "I;16"), dtype="<u2").astype(">u2").tobytes()),
            (32, "I", np.frombuffer(make_integer(w, h, "I"), dtype="<i4").astype(">i4").tobytes()),
            (-32, "F", make_float(w, h).astype(">f4").tobytes()),
            (-64, "F 64-bit source", make_float(w, h).astype(">f8").tobytes()),
        ):
            cards = ["SIMPLE  =                    T", f"BITPIX  = {bits:20}", "NAXIS   =                    2", f"NAXIS1  = {w:20}", f"NAXIS2  = {h:20}", "END"]
            header = b"".join(card.encode("ascii").ljust(80) for card in cards).ljust(2880)
            payloads.append((f"FITS {mode}", header + samples.ljust(((len(samples) + 2879) // 2880) * 2880, b"\0")))

    if formats is None or "PIXAR" in formats:
        header = bytearray(1024)
        header[:4] = b"\x80\xe8\0\0"
        struct.pack_into("<HH", header, 416, h, w)
        struct.pack_into("<HH", header, 424, 14, 2)
        payloads.append(("PIXAR RGB", bytes(header) + rgb))

    if formats is None or "SUN" in formats:
        for depth, mode, samples in ((1, "1", make_one(w, h)), (8, "L", gray), (24, "RGB", rgb)):
            row_bytes = (w * depth + 7) // 8
            stride = ((w * depth + 15) // 16) * 2
            raw = b"".join(samples[y * row_bytes : (y + 1) * row_bytes].ljust(stride, b"\0") for y in range(h))
            header = struct.pack(">8I", 0x59A66A95, w, h, depth, len(raw), 3, 0, 0)
            payloads.append((f"SUN {mode}", header + raw))
    return payloads


def measure(operation: Callable[[], object], warmups: int, iterations: int) -> dict[str, float]:
    """Return median, p25, and p75 timings in seconds."""
    for _ in range(warmups):
        operation()
    samples: list[float] = []
    gc.disable()
    try:
        for _ in range(iterations):
            started = perf_counter()
            operation()
            samples.append(perf_counter() - started)
    finally:
        gc.enable()
    samples.sort()
    return {"median": median(samples), "p25": samples[len(samples) // 4], "p75": samples[(len(samples) * 3) // 4]}


def _speedup_label(value: float | None) -> str:
    if value is None:
        return "---"
    return f"{value:.2f}x"


def _speedup_text(value: float | None) -> Text:
    """Return a colored speedup cell."""
    if value is None:
        return Text("---", style="dim")
    label = f"{value:.2f}x"
    if value >= 2.0:
        return Text(label, style="bold green")
    if value >= 1.0:
        return Text(label, style="green")
    return Text(label, style="red")


def format_memory(value: float | None, *, signed: bool = False) -> str:
    if value is None:
        return "---"
    precision = 3 if 0 < abs(value) < 2**20 / 100 else 2
    return format(value / 2**20, f"{'+' if signed else ''}.{precision}f")


def memory_cell(result: dict[str, Any], library: str) -> str:
    return format_memory(result.get(f"{library}_memory_bytes"))


def memory_difference(blanket: float | None, pillow: float | None) -> str:
    return format_memory(blanket - pillow if blanket is not None and pillow is not None else None, signed=True)


def memory_difference_text(blanket: float | None, pillow: float | None) -> Text:
    label = memory_difference(blanket, pillow)
    if blanket is None or pillow is None or blanket == pillow:
        return Text(label, style="dim")
    return Text(label, style="green" if blanket < pillow else "red")


def section_memory_value(results: list[dict[str, Any]], section: str, library: str) -> float | None:
    values = [r[f"{library}_memory_bytes"] for r in results if r["section"] == section and r.get(f"{library}_memory_bytes") is not None]
    return median(values) if values else None


def section_memory(results: list[dict[str, Any]], section: str, library: str) -> str:
    return format_memory(section_memory_value(results, section, library))


def make_detail_table(results: list[dict[str, Any]]) -> Table:
    """Build a table containing every benchmark result."""
    table = Table(show_lines=False, pad_edge=False, box=None)
    table.add_column("Operation", style="cyan", no_wrap=True)
    table.add_column("Blanket ms", justify="right")
    table.add_column("Pillow ms", justify="right")
    table.add_column("Speedup", justify="right")
    table.add_column("Blanket MiB", justify="right")
    table.add_column("Pillow MiB", justify="right")
    table.add_column("Diff MiB (B-P)", justify="right")

    last_section: str | None = None
    for r in results:
        if r["section"] != last_section:
            table.add_row(Text(r["section"], style="bold underline"))
            last_section = r["section"]
        spread = (r["blanket_p75_ms"] - r["blanket_p25_ms"]) / 2 if r["blanket_ms"] is not None else None
        blanket_cell = f"{r['blanket_ms']:.2f} ±{spread:.2f}" if spread is not None else "---"
        pillow_cell = f"{r['pillow_ms']:.2f}" if r["pillow_ms"] is not None else "---"
        table.add_row(r["operation"], blanket_cell, pillow_cell, _speedup_text(r["blanket_speedup"]), memory_cell(r, "blanket"), memory_cell(r, "pillow"), memory_difference_text(r.get("blanket_memory_bytes"), r.get("pillow_memory_bytes")))

    return available_columns(table, results, 3)


def make_summary_table(results: list[dict[str, Any]]) -> Table:
    """Summarize benchmark results with one row per section."""
    table = Table(show_lines=False, pad_edge=False, box=None)
    table.add_column("Section", style="cyan", no_wrap=True)
    table.add_column("Faster", justify="right")
    table.add_column("Geo mean", justify="right")
    table.add_column("Best", justify="right")
    table.add_column("Worst", justify="right")
    table.add_column("Median Blanket MiB", justify="right")
    table.add_column("Median Pillow MiB", justify="right")
    table.add_column("Diff MiB (B-P)", justify="right")

    sections = dict.fromkeys(r["section"] for r in results)
    for section in sections:
        memory_values = [section_memory_value(results, section, library) for library in ("blanket", "pillow")]
        memory = [*(format_memory(value) for value in memory_values), memory_difference_text(*memory_values)]
        paired = [r for r in results if r["section"] == section and r["blanket_speedup"] is not None]
        if not paired:
            table.add_row(section, "---", "---", "---", "---", *memory)
            continue
        speedups = [r["blanket_speedup"] for r in paired]
        wins = sum(speedup > 1.0 for speedup in speedups)
        table.add_row(section, f"{wins}/{len(paired)}", _speedup_text(geometric_mean(speedups)), _speedup_text(max(speedups)), _speedup_text(min(speedups)), *memory)

    return available_columns(table, results, 4)


def measurement_visibility(results: list[dict[str, Any]]) -> tuple[bool, bool]:
    timing = any(r.get(f"{library}_ms") is not None for r in results for library in ("blanket", "pillow"))
    memory = any(r.get(f"{library}_memory_bytes") is not None for r in results for library in ("blanket", "pillow"))
    return timing, memory


def available_columns(table: Table, results: list[dict[str, Any]], timing_columns: int) -> Table:
    timing, memory = measurement_visibility(results)
    table.columns = [column for index, column in enumerate(table.columns) if index == 0 or (timing if index <= timing_columns else memory)]
    return table


def print_measurement_line(line: str, results: list[dict[str, Any]], label_width: int, timing_width: int) -> None:
    timing, memory = measurement_visibility(results)
    print(line[:label_width] + (line[label_width : label_width + timing_width] if timing else "") + (line[label_width + timing_width :] if memory else ""))


def print_detail_plain(results: list[dict[str, Any]]) -> None:
    """Print every benchmark result as plain text lines."""
    header = f"{'Operation':<55} {'Blanket ms':>18} {'Pillow ms':>12} {'Speedup':>10} {'Blanket MiB':>17} {'Pillow MiB':>17} {'Diff MiB (B-P)':>17}"
    output = partial(print_measurement_line, results=results, label_width=55, timing_width=43)
    output(header)
    output("-" * len(header))
    last_section: str | None = None
    for r in results:
        if r["section"] != last_section:
            print(f"\n== {r['section']} ==")
            last_section = r["section"]
        spread = (r["blanket_p75_ms"] - r["blanket_p25_ms"]) / 2 if r["blanket_ms"] is not None else None
        blanket_cell = f"{r['blanket_ms']:.2f} ±{spread:.2f}" if spread is not None else "---"
        pillow_cell = f"{r['pillow_ms']:.2f}" if r["pillow_ms"] is not None else "---"
        difference = memory_difference(r.get("blanket_memory_bytes"), r.get("pillow_memory_bytes"))
        output(f"{r['operation']:<55} {blanket_cell:>18} {pillow_cell:>12} {_speedup_label(r['blanket_speedup']):>10} {memory_cell(r, 'blanket'):>17} {memory_cell(r, 'pillow'):>17} {difference:>17}")


def print_summary_plain(results: list[dict[str, Any]]) -> None:
    """Print section summaries as plain text lines."""
    header = f"{'Section':<20} {'Faster':>10} {'Geo mean':>10} {'Best':>10} {'Worst':>10} {'Median Blanket MiB':>18} {'Median Pillow MiB':>18} {'Diff MiB (B-P)':>17}"
    output = partial(print_measurement_line, results=results, label_width=20, timing_width=44)
    output(header)
    output("-" * len(header))
    sections = dict.fromkeys(r["section"] for r in results)
    for section in sections:
        blanket_memory, pillow_memory = (section_memory_value(results, section, library) for library in ("blanket", "pillow"))
        memory = f"{format_memory(blanket_memory):>18} {format_memory(pillow_memory):>18} {memory_difference(blanket_memory, pillow_memory):>17}"
        paired = [r for r in results if r["section"] == section and r["blanket_speedup"] is not None]
        if not paired:
            output(f"{section:<20} {'---':>10} {'---':>10} {'---':>10} {'---':>10} {memory}")
            continue
        speedups = [r["blanket_speedup"] for r in paired]
        wins = sum(speedup > 1.0 for speedup in speedups)
        output(f"{section:<20} {f'{wins}/{len(paired)}':>10} {_speedup_label(geometric_mean(speedups)):>10} {_speedup_label(max(speedups)):>10} {_speedup_label(min(speedups)):>10} {memory}")


def slower_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return paired results where Blanket trails Pillow (rounded to 2 decimal places)."""
    return [r for r in results if r["blanket_speedup"] is not None and round(r["blanket_speedup"], 2) < 1.0]


def result_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(row[field].replace("→", " to ") if field == "operation" and isinstance(row[field], str) else row[field] for field in ("section", "operation", "size", "width", "height"))


def load_baseline(path: Path) -> dict[str, Any]:
    """Accept saved runs and result lists, using median Blanket times for repeated cases."""
    document = json.loads(path.read_text())
    if isinstance(document, list):
        document = {"results": document, "metadata": {}}
    if not isinstance(document, dict) or not isinstance(document.get("results"), list):
        raise ValueError("expected a run object or a list of results")
    if not isinstance(document.get("metadata", {}), dict):
        raise ValueError("metadata must be an object")
    cases: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in document["results"]:
        try:
            key = result_key(row)
            value = row["blanket_ms"]
            valid = not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value) and value > 0
            if not valid:
                raise ValueError(f"timing for {key!r} must be positive and finite")
            cases.setdefault(key, []).append(row)
        except (KeyError, TypeError) as error:
            raise ValueError("invalid benchmark result") from error
    document["results"] = [{**rows[0], "blanket_ms": median(row["blanket_ms"] for row in rows)} for rows in cases.values()]
    return document


def run_metadata(args: argparse.Namespace) -> dict[str, Any]:
    def git(*arguments: str) -> str | None:
        try:
            return subprocess.check_output(["git", *arguments], cwd=Path(__file__).resolve().parents[1], stderr=subprocess.DEVNULL, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = git("status", "--porcelain")
    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(status) if status is not None else None,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "iterations": args.iterations,
        "warmups": args.warmups,
        "codec_formats": args.formats,
        "memory": None if args.no_memory else {"method": "Memray peak allocations", "aggregation": "single", "runs": 1},
    }


def compare_results(results: list[dict[str, Any]], baseline: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int, int]:
    previous = {result_key(row): row for row in baseline}
    current_keys = {result_key(row) for row in results}
    matched = []
    for row in results:
        old = previous.get(result_key(row))
        if old is not None:
            matched.append({**row, "baseline_ms": old["blanket_ms"], "change_pct": (row["blanket_ms"] / old["blanket_ms"] - 1) * 100})
    return matched, len(current_keys - previous.keys()), len(previous.keys() - current_keys)


def print_comparison(console: Console, results: list[dict[str, Any]], baseline: dict[str, Any], threshold: float, verbose: bool) -> None:
    matched, added, missing = compare_results(results, baseline["results"])
    console.rule("Blanket vs baseline")
    console.print(f"{len(matched)} matched; {added} new; {missing} baseline cases not run. Negative time change = faster.")
    console.print(f"Changes within ±{threshold:g}% are below threshold (not a statistical significance test).")
    if not matched:
        console.print("No matching cases; use the same sections and sizes as the baseline.")
        return

    def change_cell(change: float) -> Text:
        return Text(f"{change:+.1f}%", style="green" if change < -threshold else "red" if change > threshold else "dim")

    table = Table("Section / size", "Time change", "Improved", "Regressed", "Within threshold")
    groups = dict.fromkeys((row["section"], row["size"]) for row in matched)
    for section, size in [*groups, ("All matched cases", "")]:
        rows = matched if not size else [row for row in matched if (row["section"], row["size"]) == (section, size)]
        improved = sum(row["change_pct"] < -threshold for row in rows)
        regressed = sum(row["change_pct"] > threshold for row in rows)
        change = (geometric_mean(row["blanket_ms"] / row["baseline_ms"] for row in rows) - 1) * 100
        table.add_row(f"{section} {size}".strip(), change_cell(change), str(improved), str(regressed), str(len(rows) - improved - regressed))
    console.print(table)
    console.print("Section and overall changes use equally weighted geometric mean time ratios.")
    details = Table("Section / size / operation", "Before ms", "Now ms", "Delta ms", "Time change")
    for row in sorted(matched, key=lambda row: abs(row["change_pct"]), reverse=True):
        if verbose or abs(row["change_pct"]) > threshold:
            details.add_row(f"{row['section']} / {row['size']} / {row['operation']}", f"{row['baseline_ms']:.3f}", f"{row['blanket_ms']:.3f}", f"{row['blanket_ms'] - row['baseline_ms']:+.3f}", change_cell(row["change_pct"]))
    if details.row_count:
        console.print(details)


def print_comparison_plain(results: list[dict[str, Any]], baseline: dict[str, Any], threshold: float, verbose: bool) -> None:
    matched, added, missing = compare_results(results, baseline["results"])
    print("\n--- Blanket vs baseline ---")
    print(f"{len(matched)} matched; {added} new; {missing} baseline cases not run. Negative time change = faster.")
    print(f"Changes within ±{threshold:g}% are below threshold (not a statistical significance test).")
    if not matched:
        print("No matching cases; use the same sections and sizes as the baseline.")
        return

    def change_label(change: float) -> str:
        return f"{change:+.1f}%"

    header = f"{'Section / size':<35} {'Time change':>12} {'Improved':>10} {'Regressed':>10} {'Within':>10}"
    print(header)
    print("-" * len(header))
    groups = dict.fromkeys((row["section"], row["size"]) for row in matched)
    for section, size in [*groups, ("All matched cases", "")]:
        rows = matched if not size else [row for row in matched if (row["section"], row["size"]) == (section, size)]
        improved = sum(row["change_pct"] < -threshold for row in rows)
        regressed = sum(row["change_pct"] > threshold for row in rows)
        change = (geometric_mean(row["blanket_ms"] / row["baseline_ms"] for row in rows) - 1) * 100
        print(f"{f'{section} {size}'.strip():<35} {change_label(change):>12} {improved:>10} {regressed:>10} {len(rows) - improved - regressed:>10}")
    print("Section and overall changes use equally weighted geometric mean time ratios.")

    detail_header = f"{'Section / size / operation':<60} {'Before ms':>10} {'Now ms':>10} {'Delta ms':>10} {'Change':>10}"
    printed_header = False
    for row in sorted(matched, key=lambda row: abs(row["change_pct"]), reverse=True):
        if verbose or abs(row["change_pct"]) > threshold:
            if not printed_header:
                print(detail_header)
                print("-" * len(detail_header))
                printed_header = True
            print(f"{row['section']} / {row['size']} / {row['operation']:<60s}"[:60] + f" {row['baseline_ms']:>10.3f} {row['blanket_ms']:>10.3f} {row['blanket_ms'] - row['baseline_ms']:>+10.3f} {change_label(row['change_pct']):>10}")
