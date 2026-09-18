"""Measure save compression size, time, and exact decoded-pixel preservation.

Run with: uv run --no-sync scripts/benchmark_compression.py --help
Uses deterministic 8-bit fixtures; requires no Pillow or external image assets.
"""

from __future__ import annotations

import argparse
import io
import json
import platform
import random
from collections.abc import Callable, Iterator
from pathlib import Path
from statistics import median
from time import perf_counter

from blanket import Image
from blanket.Compressor import LosslessImageCompressor


def fixtures(size: tuple[int, int]) -> Iterator[tuple[str, Image.Image]]:
    width, height = size
    rng = random.Random(42)
    noise = rng.randbytes(width * height * 3)
    builders: dict[str, tuple[str, Callable[[int, int], bytes]]] = {
        "binary": ("L", lambda x, y: bytes([255 * ((x // 8 + y // 8) % 2)])),
        "gray-gradient": ("L", lambda x, y: bytes([(x + y) % 256])),
        "palette": ("RGB", lambda x, y: bytes(((x // 8 % 4) * 85, (y // 8 % 4) * 85, 85))),
        "rgb-gradient": ("RGB", lambda x, y: bytes((x % 256, y % 256, (x + y) % 256))),
        "gray-alpha": ("RGBA", lambda x, y: bytes([(x + y) % 256] * 3 + [(x * 17 + y * 31) % 256])),
        "color-key": ("RGBA", lambda x, y: bytes((19, 73, 151, 0)) if x % 8 == 0 else bytes((x % 256, y % 256, 0, 255))),
    }
    for name, (mode, build) in builders.items():
        yield name, Image.frombytes(mode, size, b"".join(build(x, y) for y in range(height) for x in range(width)))
    yield "rgb-noise", Image.frombytes("RGB", size, noise)
    yield "opaque-rgba", Image.frombytes("RGBA", size, b"".join(noise[i : i + 3] + b"\xff" for i in range(0, len(noise), 3)))


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formats", nargs="+", choices=["PNG", "JPEG", "JXL", "HEIF"], default=["PNG", "JPEG", "JXL", "HEIF"])
    parser.add_argument("--efforts", nargs="+", type=int, choices=range(1, 11), default=[7])
    parser.add_argument("--width", type=positive, default=1920)
    parser.add_argument("--height", type=positive, default=1080)
    parser.add_argument("--repeats", type=positive, default=3)
    parser.add_argument("--warmups", type=int, choices=range(11), default=1)
    parser.add_argument("--json", type=Path, help="write machine-readable results and individual timing samples")
    args = parser.parse_args()
    print("Times are median in-memory saves; decoding/verification is excluded.")
    print("Lossless means exact RGBA samples, including hidden colors; JPEG compares to the normal lossy save.")
    print(f"{'Fixture':<15} {'Format':<6} {'Effort':>6} {'Before B':>10} {'After B':>10} {'Saved %':>8} {'Base ms':>10} {'Opt ms':>10} {'Time x':>8} {'Check':>6}")
    rows: list[dict[str, object]] = []
    failed = False
    for name, original in fixtures((args.width, args.height)):
        for fmt in dict.fromkeys(args.formats):
            image = original.convert("RGB") if fmt == "JPEG" else original
            for effort in dict.fromkeys(args.efforts):
                row: dict[str, object] = {"fixture": name, "format": fmt, "effort": effort}
                try:
                    result = benchmark(image, fmt, effort, args.repeats, args.warmups)
                    row.update(result)
                    passed = result["lossless"] and result["non_growing"]
                    failed |= not passed
                    print(f"{name:<15} {fmt:<6} {effort:>6} {result['baseline_bytes']:>10.0f} {result['optimized_bytes']:>10.0f} {result['reduction_percent']:>8.2f} {result['baseline_ms']:>10.2f} {result['optimized_ms']:>10.2f} {result['time_ratio']:>8.2f} {'PASS' if passed else 'FAIL':>6}", flush=True)
                except Exception as error:
                    failed = True
                    row["error"] = str(error)
                    print(f"{name} {fmt} effort={effort}: ERROR: {error}", flush=True)
                rows.append(row)
    if args.json:
        args.json.write_text(json.dumps({"platform": platform.platform(), "python": platform.python_version(), "width": args.width, "height": args.height, "repeats": args.repeats, "warmups": args.warmups, "results": rows}, indent=2) + "\n")
    print(f"{len(rows)} cases: {'FAILED (pixel mismatch, size increase, or codec error)' if failed else 'all checks passed'}")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
