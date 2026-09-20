"""Measure save compression size, time, and exact decoded-pixel preservation.

Run with: uv run --no-sync scripts/benchmark_compression.py --help
Loads real test images from the ``test_images/`` directory.
"""

from __future__ import annotations

import argparse
import io
import json
import platform
from collections.abc import Iterator
from pathlib import Path
from statistics import median
from time import perf_counter

from blanket import Image
from blanket.Compressor import LosslessImageCompressor

TEST_IMAGES_DIR = Path(__file__).resolve().parent.parent / "test_images"


def fixtures() -> Iterator[tuple[str, Image.Image]]:
    """Yield ``(stem, image)`` pairs from the ``test_images/`` directory.

    Each unique stem (e.g. ``hdr_16b``, ``manning``) is loaded once from
    whichever source file is found first (PNG preferred).
    """
    if not TEST_IMAGES_DIR.is_dir():
        raise SystemExit(f"test_images directory not found: {TEST_IMAGES_DIR}")
    seen: set[str] = set()
    # Prefer PNG sources since they are lossless.
    paths = sorted(TEST_IMAGES_DIR.iterdir(), key=lambda p: (p.suffix != ".png", p.name))
    for path in paths:
        if path.suffix.lstrip(".") not in ("png", "jpg", "jpeg", "jxl", "heic") or path.name.startswith("."):
            continue
        if path.stem in seen:
            continue
        seen.add(path.stem)
        yield path.stem, Image.open(path)


def decoded_samples(encoded: bytes) -> tuple[tuple[int, int], bytes]:
    # PNG may legally reduce its stored mode. RGBA expansion retains hidden
    # RGB samples and alpha, unlike compositing or a perceptual comparison.
    with Image.open(io.BytesIO(encoded)) as decoded:
        return decoded.size, decoded.convert("RGBA").tobytes()


def benchmark(image: Image.Image, fmt: str, effort: int, repeats: int, warmups: int) -> dict[str, object]:
    options: dict[str, object] = {"compress_level": 6} if fmt == "PNG" else {"quality": 90}
    if fmt in ("JXL", "HEIF"):
        options = {"lossless": True}
    if fmt == "JXL":
        options["effort"] = 7
    compressor = LosslessImageCompressor(effort=effort)

    def save(optimized: bool) -> tuple[bytes, float]:
        output = io.BytesIO()
        start = perf_counter()
        image.save(output, fmt, **options, **({"compressor": compressor} if optimized else {}))
        elapsed = perf_counter() - start
        return output.getvalue(), elapsed

    for _ in range(warmups):
        save(False)
        save(True)
    times: dict[bool, list[float]] = {False: [], True: []}
    sizes: dict[bool, list[int]] = {False: [], True: []}
    lossless, non_growing = True, True
    source = (image.size, image.convert("RGBA").tobytes())
    for iteration in range(repeats):
        outputs = {}
        # Alternate order to reduce systematic first/second timing bias.
        for optimized in (False, True) if iteration % 2 == 0 else (True, False):
            encoded, elapsed = save(optimized)
            outputs[optimized] = encoded
            times[optimized].append(elapsed)
            sizes[optimized].append(len(encoded))
        reference = decoded_samples(outputs[False])
        lossless &= decoded_samples(outputs[True]) == reference
        # The compressor promises no additional loss over the normal save.
        # HEIF's codec color conversion may change samples even in lossless
        # mode. PNG additionally guarantees exact preservation of the source.
        if fmt == "PNG":
            lossless &= reference == source
        non_growing &= len(outputs[True]) <= len(outputs[False])
    baseline_bytes, optimized_bytes = (median(sizes[key]) for key in (False, True))
    baseline_time, optimized_time = (median(times[key]) for key in (False, True))
    return {
        "baseline_bytes": baseline_bytes,
        "optimized_bytes": optimized_bytes,
        "saved_bytes": baseline_bytes - optimized_bytes,
        "reduction_percent": 100 * (1 - optimized_bytes / baseline_bytes),
        "baseline_ms": baseline_time * 1000,
        "optimized_ms": optimized_time * 1000,
        "time_ratio": optimized_time / baseline_time,
        "lossless": lossless,
        "non_growing": non_growing,
        "baseline_seconds": times[False],
        "optimized_seconds": times[True],
        "save_options": options,
    }


def positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def _row_key(row: dict[str, object]) -> tuple[str, str, int]:
    return (str(row["fixture"]), str(row["format"]), int(row["effort"]))


def _delta(old: float, new: float) -> str:
    """Format a relative change as a colored string."""
    if old == 0:
        return "N/A"
    pct = 100 * (new - old) / abs(old)
    sign = "+" if pct >= 0 else ""
    # Green for improvement (negative size / negative time), red for regression.
    color = "green" if pct < -0.5 else ("red" if pct > 0.5 else "dim")
    return f"[{color}]{sign}{pct:.2f}%[/{color}]"


def compare_results(path_a: Path, path_b: Path) -> int:
    """Load two JSON result files and print a rich table comparing overlapping tests."""
    from rich.console import Console
    from rich.table import Table

    data_a = json.loads(path_a.read_text())
    data_b = json.loads(path_b.read_text())

    lookup_a: dict[tuple[str, str, int], dict[str, object]] = {_row_key(r): r for r in data_a["results"] if "error" not in r}
    lookup_b: dict[tuple[str, str, int], dict[str, object]] = {_row_key(r): r for r in data_b["results"] if "error" not in r}
    overlap = sorted(lookup_a.keys() & lookup_b.keys())
    if not overlap:
        print("No overlapping tests found between the two files.")
        return 1

    label_a = path_a.stem
    label_b = path_b.stem

    table = Table(
        title=f"Comparison: [bold]{label_a}[/bold] vs [bold]{label_b}[/bold]",
        caption=(f"A: {data_a.get('platform', '?')} / Python {data_a.get('python', '?')} ({data_a.get('repeats', '?')} repeats)\nB: {data_b.get('platform', '?')} / Python {data_b.get('python', '?')} ({data_b.get('repeats', '?')} repeats)"),
        show_lines=True,
    )
    table.add_column("Fixture", style="cyan")
    table.add_column("Fmt")
    table.add_column("Eff", justify="right")
    table.add_column(f"Opt B ({label_a})", justify="right")
    table.add_column(f"Opt B ({label_b})", justify="right")
    table.add_column("Δ Size", justify="right")
    table.add_column(f"Saved% ({label_a})", justify="right")
    table.add_column(f"Saved% ({label_b})", justify="right")
    table.add_column(f"Opt ms ({label_a})", justify="right")
    table.add_column(f"Opt ms ({label_b})", justify="right")
    table.add_column("Δ Time", justify="right")
    table.add_column("Check", justify="center")

    for key in overlap:
        a, b = lookup_a[key], lookup_b[key]
        a_opt = float(a["optimized_bytes"])
        b_opt = float(b["optimized_bytes"])
        a_pct = float(a["reduction_percent"])
        b_pct = float(b["reduction_percent"])
        a_ms = float(a["optimized_ms"])
        b_ms = float(b["optimized_ms"])
        a_pass = a.get("lossless", False) and a.get("non_growing", False)
        b_pass = b.get("lossless", False) and b.get("non_growing", False)
        check_a = "[green]✓[/green]" if a_pass else "[red]✗[/red]"
        check_b = "[green]✓[/green]" if b_pass else "[red]✗[/red]"

        table.add_row(key[0], key[1], str(key[2]), f"{a_opt:.0f}", f"{b_opt:.0f}", _delta(a_opt, b_opt), f"{a_pct:.2f}", f"{b_pct:.2f}", f"{a_ms:.2f}", f"{b_ms:.2f}", _delta(a_ms, b_ms), f"{check_a} {check_b}")

    console = Console()
    console.print(table)
    console.print(f"\n[bold]{len(overlap)}[/bold] overlapping tests compared.")
    skipped_a = len(lookup_a) - len(overlap)
    skipped_b = len(lookup_b) - len(overlap)
    if skipped_a or skipped_b:
        console.print(f"[dim]Skipped {skipped_a} test(s) only in A, {skipped_b} only in B.[/dim]")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-f", "--formats", nargs="+", choices=["PNG", "JPEG", "JXL", "HEIF"], default=["PNG", "JPEG", "JXL", "HEIF"])
    parser.add_argument("-e", "--efforts", nargs="+", type=int, choices=range(1, 11), default=[7])
    parser.add_argument("--repeats", type=positive, default=3)
    parser.add_argument("--warmups", type=int, choices=range(11), default=1)
    parser.add_argument("--json", type=Path, help="write machine-readable results and individual timing samples")
    parser.add_argument("--compare", nargs=2, type=Path, metavar=("A.json", "B.json"), help="compare two JSON result files and exit (no benchmark is run)")
    args = parser.parse_args()
    if args.compare:
        return compare_results(args.compare[0], args.compare[1])
    print("Times are median in-memory saves; decoding/verification is excluded.")
    print("Lossless means exact RGBA samples, including hidden colors; JPEG compares to the normal lossy save.")
    print(f"{'Fixture':<12} {'Size':<11} {'Format':<8} {'Effort':>6} {'Bytes Before':>15} {'Bytes After':>15} {'Saved %':>8} {'Base ms':>10} {'Opt ms':>10} {'Time x':>8} {'Was Lossless':>13}")
    rows: list[dict[str, object]] = []
    failed = False
    for name, original in fixtures():
        w, h = original.size
        size_label = f"{w}×{h}"
        for fmt in dict.fromkeys(args.formats):
            if "16b" in name and fmt in ("JPEG", "HEIF"):
                continue
            image = original.convert("RGB") if fmt == "JPEG" else original
            for effort in dict.fromkeys(args.efforts):
                row: dict[str, object] = {"fixture": name, "width": w, "height": h, "format": fmt, "effort": effort}
                try:
                    result = benchmark(image, fmt, effort, args.repeats, args.warmups)
                    row.update(result)
                    passed = result["lossless"] and result["non_growing"]
                    failed |= not passed
                    print(f"{name:<12} {size_label:<11} {fmt:<8} {effort:>6} {result['baseline_bytes']:>15.0f} {result['optimized_bytes']:>15.0f} {result['reduction_percent']:>8.2f} {result['baseline_ms']:>10.2f} {result['optimized_ms']:>10.2f} {result['time_ratio']:>8.2f} {'YES' if passed else 'NO':>13}", flush=True)
                except Exception as error:
                    failed = True
                    row["error"] = str(error)
                    print(f"{name} {fmt} effort={effort}: ERROR: {error}", flush=True)
                rows.append(row)
    if args.json:
        args.json.write_text(json.dumps({"platform": platform.platform(), "python": platform.python_version(), "repeats": args.repeats, "warmups": args.warmups, "results": rows}, indent=2) + "\n")
    print(f"{len(rows)} cases: {'FAILED (pixel mismatch, size increase, or codec error)' if failed else 'all checks passed'}")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
