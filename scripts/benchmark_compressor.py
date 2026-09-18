"""Reproducible save-compressor timings; build with maturin develop -r first."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
import random
from statistics import median
from time import perf_counter

from blanket import Image
from blanket.Compressor import LosslessImageCompressor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[32, 512])
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--formats", nargs="+", default=["PNG", "JPEG", "JXL", "HEIF"])
    args = parser.parse_args()
    if args.repeat < 1 or any(size < 1 for size in args.sizes):
        parser.error("sizes and repeat must be positive")
    results = []
    for size in args.sizes:
        rng = random.Random(42)
        noise = rng.randbytes(size * size * 3)
        gray = bytes(i % 16 * 17 for i in range(size * size))
        palette = b"".join(bytes((i, i * 71 % 256, i * 133 % 256, i % 3 * 127)) for i in range(256))
        indexed = b"".join(palette[i * 4:i * 4 + 4] for i in rng.randbytes(size * size))
        cases = [("rgb", "RGB", noise), ("gray", "L", gray), ("palette", "RGBA", indexed)]
        for fmt in args.formats:
            for name, mode, pixels in cases if fmt == "PNG" else cases[:1]:
                source = Image.frombytes(mode, (size, size), pixels)
                options = {"compressor": LosslessImageCompressor(effort=7 if fmt == "PNG" else 3)}
                if fmt in {"JXL", "HEIF"}:
                    options["lossless"] = True
                times = []
                digest = None
                for repetition in range(args.repeat + 1):
                    output = io.BytesIO()
                    start = perf_counter()
                    source.save(output, fmt, **options)
                    elapsed = perf_counter() - start
                    payload = output.getvalue()
                    current = hashlib.sha256(payload).hexdigest()
                    if digest is not None and current != digest:
                        raise AssertionError("non-deterministic compressed output")
                    digest = current
                    if repetition:
                        times.append(elapsed)
                results.append({"format": fmt, "case": name, "size": size, "seconds": median(times), "bytes": len(payload), "sha256": digest})
    print(json.dumps({"machine": platform.machine(), "results": results}, indent=2))


if __name__ == "__main__":
    main()
