"""Compare Blanket CPU operations with resident and transfer-inclusive GPU execution."""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import TYPE_CHECKING

from blanket import Image, ImageChops, ImageEnhance, ImageFilter, ImageGpu, __version__

NO_RICH = os.environ.get("NO_RICH")

if not NO_RICH:
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True)
class Operation:
    cpu: Callable[[Image.Image, Image.Image], Image.Image]
    gpu: Callable[[ImageGpu.GpuImage, ImageGpu.GpuImage], ImageGpu.GpuImage]
    binary: bool = False


def operations(size: tuple[int, int]) -> dict[str, Operation]:
    target = (max(1, size[0] // 2), max(1, size[1] // 2))
    coefficients = [1.0] * 9
    kernel = ImageFilter.Kernel((3, 3), coefficients, scale=9)
    blur = ImageFilter.GaussianBlur(2)
    return {
        "resize": Operation(lambda a, _: a.resize(target, Image.Resampling.BICUBIC), lambda a, _: a.resize(target, Image.Resampling.BICUBIC)),
        "gaussian_blur": Operation(lambda a, _: a.filter(blur), lambda a, _: a.gaussian_blur(2)),
        "convolve": Operation(lambda a, _: a.filter(kernel), lambda a, _: a.convolve(coefficients, scale=9)),
        "brightness": Operation(lambda a, _: ImageEnhance.Brightness(a).enhance(1.2), lambda a, _: a.brightness(1.2)),
        "invert": Operation(lambda a, _: ImageChops.invert(a), lambda a, _: a.invert()),
        "transpose": Operation(lambda a, _: a.transpose(Image.Transpose.ROTATE_90), lambda a, _: a.transpose(Image.Transpose.ROTATE_90)),
        "blend": Operation(lambda a, b: Image.blend(a, b, 0.4), lambda a, b: a.blend(b, 0.4), binary=True),
        "pipeline": Operation(
            lambda a, _: ImageEnhance.Brightness(a.resize(target, Image.Resampling.BICUBIC).filter(blur)).enhance(1.2),
            lambda a, _: a.resize(target, Image.Resampling.BICUBIC).gaussian_blur(2).brightness(1.2),
        ),
    }


def measure(run: Callable[[], Image.Image | ImageGpu.GpuImage], warmups: int, repeats: int, iterations: int) -> list[float]:
    def sample(count: int) -> float:
        elapsed = 0.0
        for _ in range(count):
            start = perf_counter()
            result = run()
            if isinstance(result, ImageGpu.GpuImage):
                result.synchronize()
            elapsed += perf_counter() - start
            result.close()
        return elapsed / count

    for _ in range(warmups):
        sample(1)
    enabled = gc.isenabled()
    gc.disable()
    try:
        return [sample(iterations) * 1000 for _ in range(repeats)]
    finally:
        if enabled:
            gc.enable()


def positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def dimensions(value: str) -> tuple[int, int]:
    try:
        width, height = (positive(part) for part in value.lower().split("x"))
    except (ValueError, argparse.ArgumentTypeError) as error:
        raise argparse.ArgumentTypeError("size must be WIDTHxHEIGHT with positive dimensions") from error
    if width * height > 178_956_970:
        raise argparse.ArgumentTypeError("size exceeds Blanket's image pixel limit")
    return width, height


def fixture(mode: str, size: tuple[int, int], seed: int) -> Image.Image:
    channels = {"L": 1, "LA": 2, "RGB": 3, "RGBA": 4}[mode]
    data = bytes((i * 37 + i // channels * 11 + seed) % 256 for i in range(size[0] * size[1] * channels))
    return Image.frombytes(mode, size, data)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("metal", "cuda"), default="metal")
    parser.add_argument("--runtime", choices=("native", "torch"), default="native")
    parser.add_argument("--unified-memory", action="store_true", help="Share native Metal output storage with CPU images")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--sizes", type=dimensions, nargs="+", default=[(800, 600), (1920, 1080)])
    parser.add_argument("--modes", choices=("L", "LA", "RGB", "RGBA"), nargs="+", default=["RGB", "RGBA"])
    parser.add_argument("--operations", choices=tuple(operations((1, 1))), nargs="+", default=list(operations((1, 1))))
    parser.add_argument("-w", "--warmups", type=positive, default=3)
    parser.add_argument("-r", "--repeats", type=positive, default=10)
    parser.add_argument("-i", "--iterations", type=positive, default=1)
    parser.add_argument("--json", type=Path, help="Write samples, medians, speedups, and environment metadata")
    args = parser.parse_args()
    if args.device < 0 or (args.backend == "metal" and args.device != 0):
        parser.error("device must be nonnegative; Metal supports only device 0")
    capability = ImageGpu.capabilities(args.backend, device=args.device)
    if not capability["available"]:
        parser.error(str(capability.get("diagnostic") or f"{args.backend} unavailable; build the matching GPU feature"))

    def upload(image: Image.Image) -> ImageGpu.GpuImage:
        result = ImageGpu.from_image(image, backend=args.backend, runtime=args.runtime, device=args.device, unified_memory=args.unified_memory)
        if not isinstance(result, ImageGpu.GpuImage):
            raise TypeError("benchmark fixtures require uint8 GPU images")
        return result

    rows = []
    console = None if NO_RICH else Console()

    def emit(message: str) -> None:
        if console is None:
            print(message, flush=True)
        else:
            console.print(message, markup=False)

    emit(f"Blanket {__version__} | {platform.platform()} | {args.backend}, runtime={args.runtime}, device={args.device}")
    emit("Median wall time, milliseconds; speedup = CPU / GPU (greater than 1 means GPU is faster).")
    emit("Resident: pre-uploaded inputs, operation + synchronization. With transfers: uploads + operation + CPU export.")
    if args.unified_memory:
        emit("Unified memory: CPU exports share Metal storage; CPU writes copy on demand.")
    if console is None:
        emit(f"{'Mode / size / operation':44} {'CPU ms':>10} {'GPU resident':>13} {'speedup':>9} {'GPU + transfers':>16} {'speedup':>9}")
    for size in args.sizes:
        for mode in args.modes:
            table = None
            if console is not None:
                table = Table(title=f"{mode} / {size[0]}x{size[1]}")
                table.add_column("Operation", style="cyan")
                for heading in ("CPU ms", "GPU resident ms", "Speedup", "GPU + transfers ms", "Speedup"):
                    table.add_column(heading, justify="right")
            first, second = fixture(mode, size, 0), fixture(mode, size, 73)
            resident_first, resident_second = upload(first), upload(second)
            try:
                resident_first.synchronize()
                resident_second.synchronize()
                for name in args.operations:
                    operation = operations(size)[name]

                    def cpu() -> Image.Image:
                        return operation.cpu(first, second)

                    def resident() -> ImageGpu.GpuImage:
                        return operation.gpu(resident_first, resident_second)

                    def transferred() -> Image.Image:
                        source = upload(first)
                        other = upload(second) if operation.binary else source
                        try:
                            result = operation.gpu(source, other)
                            try:
                                return result.to_image()
                            finally:
                                result.close()
                        finally:
                            source.close()
                            if other is not source:
                                other.close()

                    cpu_samples = measure(cpu, args.warmups, args.repeats, args.iterations)
                    resident_samples = measure(resident, args.warmups, args.repeats, args.iterations)
                    transferred_samples = measure(transferred, args.warmups, args.repeats, args.iterations)
                    cpu_ms, resident_ms, transferred_ms = median(cpu_samples), median(resident_samples), median(transferred_samples)
                    label = f"{mode} / {size[0]}x{size[1]} / {name}"
                    if table is None:
                        emit(f"{label:44} {cpu_ms:10.3f} {resident_ms:13.3f} {cpu_ms / resident_ms:8.2f}x {transferred_ms:16.3f} {cpu_ms / transferred_ms:8.2f}x")
                    else:
                        table.add_row(
                            name,
                            f"{cpu_ms:.3f}",
                            f"{resident_ms:.3f}",
                            Text(f"{cpu_ms / resident_ms:.2f}x", style="green" if cpu_ms > resident_ms else "red"),
                            f"{transferred_ms:.3f}",
                            Text(f"{cpu_ms / transferred_ms:.2f}x", style="green" if cpu_ms > transferred_ms else "red"),
                        )
                    rows.append(
                        {
                            "mode": mode,
                            "size": size,
                            "operation": name,
                            "input_uploads": 2 if operation.binary else 1,
                            "gpu_runtime": resident_first.runtime,
                            "gpu_device": resident_first.device,
                            "cpu_ms": cpu_ms,
                            "gpu_resident_ms": resident_ms,
                            "gpu_with_transfers_ms": transferred_ms,
                            "resident_speedup": cpu_ms / resident_ms,
                            "with_transfers_speedup": cpu_ms / transferred_ms,
                            "cpu_samples_ms": cpu_samples,
                            "gpu_resident_samples_ms": resident_samples,
                            "gpu_with_transfers_samples_ms": transferred_samples,
                        }
                    )
                if console is not None and table is not None:
                    console.print(table)
            finally:
                resident_first.close()
                resident_second.close()
                first.close()
                second.close()
    if args.json:
        report = {
            "blanket": __version__,
            "python": sys.version,
            "platform": platform.platform(),
            "backend": args.backend,
            "runtime": args.runtime,
            "device": args.device,
            "unified_memory": args.unified_memory,
            "warmups": args.warmups,
            "repeats": args.repeats,
            "iterations": args.iterations,
            "timing": "warm synchronized wall time; resident excludes transfers; transfer-inclusive uploads inputs and exports output",
            "results": rows,
        }
        args.json.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
