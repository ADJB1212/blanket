"""Measure batch throughput and process peak RSS in isolated subprocesses."""

from __future__ import annotations

import argparse
import json
import random
import resource
import subprocess
import sys
from contextlib import closing
from functools import partial
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory
from time import perf_counter

from blanket import Image, ImageBatch, ImageFilter


def encode(image: Image.Image) -> Image.Image:
    """Exercise native encoding without retaining the encoded buffer."""
    image.encode("JPEG", quality=85)
    return image


def measure(args: argparse.Namespace) -> dict[str, object]:
    """Measure one worker configuration using a bounded result consumer."""
    source = Image.frombytes("RGB", (args.width, args.height), random.Random(42).randbytes(args.width * args.height * 3))
    steps = [partial(Image.Image.resize, size=(max(1, args.width // 2), max(1, args.height // 2)), resample=Image.Resampling.LANCZOS), partial(Image.Image.filter, filter=ImageFilter.GaussianBlur(2))]
    if args.workload == "codec":
        steps.append(encode)
    pipeline = ImageBatch.Pipeline(steps)
    with TemporaryDirectory(prefix="blanket-batch-") as temporary:
        path = Path(temporary) / "input.jpg"
        if args.workload == "codec":
            source.save(path, quality=90)
        item = path if args.workload == "codec" else source
        pipeline(item).close()
        times = []
        for _ in range(args.repeats):
            inputs = (item for _ in range(args.images))
            started = perf_counter()
            if args.child_workers == 0:
                for image in inputs:
                    pipeline(image).close()
            else:
                with closing(pipeline.iter(inputs, max_workers=args.child_workers)) as results:
                    for image in results:
                        image.close()
            times.append(perf_counter() - started)
    source.close()
    seconds = median(times)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "workload": args.workload,
        "workers": args.child_workers,
        "images": args.images,
        "seconds": seconds,
        "images_per_second": round(args.images / seconds, 2),
        "peak_rss_mib": round(rss / (1024 * 1024 if sys.platform == "darwin" else 1024), 1),
    }


def main() -> None:
    """Compare serial execution with explicit batch worker counts on Unix."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=int, default=32)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--workload", choices=["pixels", "codec"], default="pixels")
    parser.add_argument("--child-workers", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.images, args.width, args.height, args.repeats, *args.workers) < 1:
        parser.error("image count, dimensions, repeats, and worker counts must be positive")
    if args.child_workers is not None:
        print(json.dumps(measure(args)))
        return
    baseline = None
    for workers in [0, *args.workers]:
        command = [sys.executable, str(Path(__file__).resolve()), "--images", str(args.images), "--width", str(args.width), "--height", str(args.height), "--repeats", str(args.repeats), "--workload", args.workload, "--child-workers", str(workers)]
        row = json.loads(subprocess.run(command, check=True, capture_output=True, text=True).stdout)
        if baseline is None:
            baseline = row["seconds"]
        row["speedup"] = round(baseline / row["seconds"], 2)
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
