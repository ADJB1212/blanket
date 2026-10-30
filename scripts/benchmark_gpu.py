"""Compare Blanket CPU operations with resident and transfer-inclusive GPU execution."""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import sys
from array import array
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import TYPE_CHECKING

from blanket import Image, ImageChops, ImageEnhance, ImageFilter, ImageGpu, ImageStat, __version__

NO_RICH = os.environ.get("NO_RICH")

if not NO_RICH:
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text

if TYPE_CHECKING:
    from collections.abc import Callable


MODES = ("L", "LA", "RGB", "RGBA")
CpuResult = Image.Image | tuple[Image.Image, ...] | list[list[int]] | list[dict[str, object]]
GpuResult = ImageGpu.GpuImage | tuple[ImageGpu.GpuImage, ...] | ImageGpu.GpuReduction


def unary_inputs(first: Image.Image, _second: Image.Image) -> tuple[Image.Image, ...]:
    return (first,)


def binary_inputs(first: Image.Image, second: Image.Image) -> tuple[Image.Image, ...]:
    return first, second


def alpha_inputs(first: Image.Image, second: Image.Image) -> tuple[Image.Image, ...]:
    return first, second.getchannel(0)


def paste_inputs(first: Image.Image, second: Image.Image) -> tuple[Image.Image, ...]:
    return first, second, second.getchannel(0)


def merge_inputs(first: Image.Image, _second: Image.Image) -> tuple[Image.Image, ...]:
    return first.split()


def remap_inputs(first: Image.Image, _second: Image.Image) -> tuple[Image.Image, ...]:
    width, height = first.size
    coordinates = array("f", (value for y in range(height) for x in range(width) for value in (x + 1.75, y + 1.25, 0.0)))
    return first, Image.frombytes("RGB;F", first.size, coordinates.tobytes())


def putalpha(image: Image.Image, alpha: Image.Image) -> Image.Image:
    if image.mode in ("L", "LA"):
        return Image.merge("LA", (image if image.mode == "L" else image.getchannel(0), alpha))
    result = image.copy()
    result.putalpha(alpha)
    return result


def paste(image: Image.Image, source: Image.Image, mask: Image.Image) -> Image.Image:
    result = image.copy()
    result.paste(source, (3, 5), mask)
    return result


def alpha_composite(image: Image.Image, source: Image.Image) -> Image.Image:
    if image.mode == "LA":
        return Image.alpha_composite(image.convert("RGBA"), source.convert("RGBA")).convert("LA")
    return Image.alpha_composite(image, source)


def statistics(image: Image.Image) -> list[dict[str, object]]:
    histogram = image.histogram()
    stats = ImageStat.Stat(histogram)
    result: list[dict[str, object]] = []
    for band, count in enumerate(stats.count):
        bins = histogram[band * 256 : (band + 1) * 256]
        moments = (stats.mean[band], stats.sum2[band] / count, *(sum(n * value**order for value, n in enumerate(bins)) / count for order in (3, 4)))
        result.append(
            {
                "count": count,
                "extrema": stats.extrema[band],
                "sum": stats.sum[band],
                "sum2": stats.sum2[band],
                "moments": moments,
                "mean": stats.mean[band],
                "variance": stats.var[band],
                "stddev": stats.stddev[band],
            }
        )
    return result


def histogram(image: Image.Image) -> list[list[int]]:
    values = image.histogram()
    return [values[start : start + 256] for start in range(0, len(values), 256)]


@dataclass(frozen=True)
class Operation:
    cpu: Callable[..., CpuResult]
    gpu: Callable[..., GpuResult]
    inputs: Callable[[Image.Image, Image.Image], tuple[Image.Image, ...]] = unary_inputs
    modes: tuple[str, ...] = MODES
    cpu_reference: str | None = None


def operations(size: tuple[int, int], mode: str = "RGB") -> dict[str, Operation]:
    target = (max(1, size[0] // 2), max(1, size[1] // 2))
    coefficients = [1.0] * 9
    kernel = ImageFilter.Kernel((3, 3), coefficients, scale=9)
    blur = ImageFilter.GaussianBlur(2)
    crop = (size[0] // 4, size[1] // 4, size[0] // 4 + target[0], size[1] // 4 + target[1])
    lookup = [min(255, int(value * 1.1)) for value in range(256)]
    lut = ImageFilter.Color3DLUT.generate(17, lambda r, g, b: (r * 0.8, g * 0.9, b * 0.7))
    affine = (1.0, 0.05, 0.0, -0.05, 1.0, 0.0)
    conversion = "RGBA" if mode != "RGBA" else "RGB"
    return {
        "resize": Operation(lambda a: a.resize(target, Image.Resampling.BICUBIC), lambda a: a.resize(target, Image.Resampling.BICUBIC)),
        "convolve": Operation(lambda a: a.filter(kernel), lambda a: a.convolve(coefficients, scale=9)),
        "crop": Operation(lambda a: a.crop(crop), lambda a: a.crop(crop)),
        "transpose": Operation(lambda a: a.transpose(Image.Transpose.ROTATE_90), lambda a: a.transpose(Image.Transpose.ROTATE_90)),
        "offset": Operation(lambda a: ImageChops.offset(a, 7, -5), lambda a: a.offset(7, -5)),
        "brightness": Operation(lambda a: ImageEnhance.Brightness(a).enhance(1.2), lambda a: a.brightness(1.2)),
        "invert": Operation(ImageChops.invert, lambda a: a.invert()),
        "point": Operation(lambda a: a.point(lookup * len(a.getbands())), lambda a: a.point(lookup)),
        "solarize": Operation(lambda a: a.point([v if v < 128 else 255 - v for v in range(256)] * len(a.getbands())), lambda a: a.solarize(128)),
        "posterize": Operation(lambda a: a.point([v & 0xF0 for v in range(256)] * len(a.getbands())), lambda a: a.posterize(4)),
        "difference": Operation(ImageChops.difference, lambda a, b: a.difference(b), binary_inputs),
        "multiply": Operation(ImageChops.multiply, lambda a, b: a.multiply(b), binary_inputs),
        "screen": Operation(ImageChops.screen, lambda a, b: a.screen(b), binary_inputs),
        "lighter": Operation(ImageChops.lighter, lambda a, b: a.lighter(b), binary_inputs),
        "darker": Operation(ImageChops.darker, lambda a, b: a.darker(b), binary_inputs),
        "add": Operation(lambda a, b: ImageChops.add(a, b, 2.0, 10), lambda a, b: a.add(b, 2.0, 10), binary_inputs),
        "subtract": Operation(lambda a, b: ImageChops.subtract(a, b, 2.0, 10), lambda a, b: a.subtract(b, 2.0, 10), binary_inputs),
        "blend": Operation(lambda a, b: Image.blend(a, b, 0.4), lambda a, b: a.blend(b, 0.4), binary_inputs),
        "alpha_composite": Operation(alpha_composite, lambda a, b: a.alpha_composite(b), binary_inputs, ("LA", "RGBA"), "Image.alpha_composite; LA via RGBA"),
        "convert": Operation(lambda a: a.convert(conversion), lambda a: a.convert(conversion)),
        "grayscale": Operation(lambda a: a.convert("L"), lambda a: a.grayscale()),
        "getchannel": Operation(lambda a: a.getchannel(0), lambda a: a.getchannel(0)),
        "split": Operation(lambda a: a.split(), lambda a: a.split()),
        "merge": Operation(lambda *bands: Image.merge(mode, bands), lambda *bands: ImageGpu.merge(mode, bands), merge_inputs),
        "putalpha": Operation(putalpha, lambda a, mask: a.putalpha(mask), alpha_inputs, cpu_reference="Image.putalpha; L/LA via Image.merge"),
        "box_blur": Operation(lambda a: a.filter(ImageFilter.BoxBlur(2)), lambda a: a.box_blur(2)),
        "gaussian_blur": Operation(lambda a: a.filter(blur), lambda a: a.gaussian_blur(2)),
        "unsharp_mask": Operation(lambda a: a.filter(ImageFilter.UnsharpMask(2, 150, 3)), lambda a: a.unsharp_mask(2, 150, 3)),
        "rank_filter": Operation(lambda a: a.filter(ImageFilter.RankFilter(3, 2)), lambda a: a.rank_filter(3, 2)),
        "min_filter": Operation(lambda a: a.filter(ImageFilter.MinFilter(3)), lambda a: a.min_filter(3)),
        "max_filter": Operation(lambda a: a.filter(ImageFilter.MaxFilter(3)), lambda a: a.max_filter(3)),
        "median_filter": Operation(lambda a: a.filter(ImageFilter.MedianFilter(3)), lambda a: a.median_filter(3)),
        "color": Operation(lambda a: ImageEnhance.Color(a).enhance(1.2), lambda a: a.color(1.2)),
        "contrast": Operation(lambda a: ImageEnhance.Contrast(a).enhance(1.2), lambda a: a.contrast(1.2)),
        "sharpness": Operation(lambda a: ImageEnhance.Sharpness(a).enhance(1.2), lambda a: a.sharpness(1.2)),
        "color_lut": Operation(lambda a: a.filter(lut), lambda a: a.color_lut(lut), modes=("RGB", "RGBA")),
        "linear_light_resize": Operation(
            lambda a: a.resize(target, Image.Resampling.BICUBIC, linear_light=True), lambda a: a.resize(target, Image.Resampling.BICUBIC, linear_light=True), modes=("RGB", "RGBA")
        ),
        "linear_light_composite": Operation(lambda a, b: Image.alpha_composite(a, b, linear_light=True), lambda a, b: a.alpha_composite(b, linear_light=True), binary_inputs, ("RGBA",)),
        "transform": Operation(
            lambda a: a.transform(size, Image.Transform.AFFINE, affine, Image.Resampling.BILINEAR), lambda a: a.transform(size, Image.Transform.AFFINE, affine, Image.Resampling.BILINEAR)
        ),
        "rotate": Operation(lambda a: a.rotate(17, Image.Resampling.BILINEAR), lambda a: a.rotate(17, Image.Resampling.BILINEAR)),
        "remap": Operation(
            lambda a, _map: a.transform(size, Image.Transform.AFFINE, (1, 0, 1.25, 0, 1, 0.75), Image.Resampling.BILINEAR),
            lambda a, mapping: a.remap(mapping, Image.Resampling.BILINEAR),
            remap_inputs,
            cpu_reference="Image.transform with the equivalent affine coordinate map",
        ),
        "paste": Operation(paste, lambda a, b, mask: a.paste(b, (3, 5), mask), paste_inputs),
        "statistics": Operation(statistics, lambda a: a.statistics(), cpu_reference="ImageStat plus histogram moments of orders 3 and 4"),
        "histogram": Operation(histogram, lambda a: a.histogram()),
        "copy": Operation(lambda a: a.copy(), lambda a: a.convert(a.mode)),
        "pipeline": Operation(
            lambda a: ImageEnhance.Brightness(a.resize(target, Image.Resampling.BICUBIC).filter(blur)).enhance(1.2),
            lambda a: a.resize(target, Image.Resampling.BICUBIC).gaussian_blur(2).brightness(1.2),
        ),
    }


def close_result(result: CpuResult | GpuResult) -> None:
    if isinstance(result, tuple):
        for image in result:
            image.close()
    elif isinstance(result, (Image.Image, ImageGpu.GpuImage, ImageGpu.GpuReduction)):
        result.close()


def download(result: GpuResult) -> CpuResult:
    if isinstance(result, tuple):
        return tuple(image.to_image() for image in result)
    return result.to_cpu() if isinstance(result, ImageGpu.GpuReduction) else result.to_image()


def measure(run: Callable[[], CpuResult | GpuResult], warmups: int, repeats: int, iterations: int) -> list[float]:
    def sample(count: int) -> float:
        elapsed = 0.0
        for _ in range(count):
            start = perf_counter()
            result = run()
            try:
                for item in result if isinstance(result, tuple) else (result,):
                    if isinstance(item, (ImageGpu.GpuImage, ImageGpu.GpuReduction)):
                        item.synchronize()
                elapsed += perf_counter() - start
            finally:
                close_result(result)
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
    parser.add_argument("-um", "--unified-memory", action="store_true", help="Share native Metal output storage with CPU images")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--sizes", type=dimensions, nargs="+", default=[(800, 600), (3840, 2160)])
    parser.add_argument("--modes", choices=MODES, nargs="+", default=["RGB", "RGBA"])
    parser.add_argument("--operations", choices=tuple(operations((1, 1))), nargs="+", default=list(operations((1, 1))))
    parser.add_argument("-w", "--warmups", type=positive, default=3)
    parser.add_argument("-r", "--repeats", type=positive, default=10)
    parser.add_argument("-i", "--iterations", type=positive, default=1)
    parser.add_argument("--json", type=Path, help="Write samples, medians, speedups, and environment metadata")
    args = parser.parse_args()
    if args.device < 0 or (args.backend == "metal" and args.device != 0):
        parser.error("device must be nonnegative; Metal supports only device 0")
    if args.unified_memory and (args.backend != "metal" or args.runtime != "native"):
        parser.error("unified memory requires native Metal")
    capability = ImageGpu.capabilities(args.backend, device=args.device)
    if not capability["available"]:
        parser.error(str(capability.get("diagnostic") or f"{args.backend} unavailable; build the matching GPU feature"))

    def upload(image: Image.Image) -> ImageGpu.GpuImage:
        result = ImageGpu.from_image(image, backend=args.backend, runtime=args.runtime, device=args.device, unified_memory=args.unified_memory)
        if not isinstance(result, ImageGpu.GpuImage):
            raise TypeError("benchmark fixtures require uint8 or float32 GPU images")
        return result

    rows = []
    skipped = []
    console = None if NO_RICH else Console()

    def emit(message: str) -> None:
        if console is None:
            print(message, flush=True)
        else:
            console.print(message, markup=False)

    emit(f"Blanket {__version__} | {platform.platform()} | {args.backend}, runtime={args.runtime}, device={args.device}")
    emit("Median wall time, milliseconds; speedup = CPU / GPU (greater than 1 means GPU is faster).")
    emit("Resident: pre-uploaded inputs, operation + synchronization. With transfers: uploads + operation + CPU export.")
    emit("Auxiliary inputs are prepared before timing; transfer-inclusive runs upload masks, bands, and coordinate maps.")
    emit("Remap CPU reference: affine transform equivalent to the prepared coordinate map. Statistics include raw moments of orders 1-4.")
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
            with ExitStack() as fixtures:
                first, second = fixture(mode, size, 0), fixture(mode, size, 73)
                fixtures.callback(first.close)
                fixtures.callback(second.close)
                catalog = operations(size, mode)
                for name in args.operations:
                    operation = catalog[name]
                    if mode not in operation.modes:
                        reason = f"requires {'/'.join(operation.modes)}"
                        emit(f"SKIP {mode} / {size[0]}x{size[1]} / {name}: {reason}")
                        skipped.append({"mode": mode, "size": size, "operation": name, "reason": reason})
                        continue
                    with ExitStack() as resources:
                        inputs = operation.inputs(first, second)
                        input_modes = [image.mode for image in inputs]
                        for image in inputs:
                            if image is not first and image is not second:
                                resources.callback(image.close)
                        resident_inputs = []
                        for image in inputs:
                            gpu = upload(image)
                            resources.callback(gpu.close)
                            resident_inputs.append(gpu)
                        for gpu in resident_inputs:
                            gpu.synchronize()
                        gpu_runtime, gpu_device = resident_inputs[0].runtime, resident_inputs[0].device

                        def cpu() -> CpuResult:
                            return operation.cpu(*inputs)

                        def resident() -> GpuResult:
                            return operation.gpu(*resident_inputs)

                        def transferred() -> CpuResult:
                            with ExitStack() as transfers:
                                uploaded = []
                                for image in inputs:
                                    gpu = upload(image)
                                    transfers.callback(gpu.close)
                                    uploaded.append(gpu)
                                result = operation.gpu(*uploaded)
                                transfers.callback(close_result, result)
                                return download(result)

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
                            "input_uploads": len(inputs),
                            "input_modes": input_modes,
                            "cpu_reference": operation.cpu_reference,
                            "gpu_runtime": gpu_runtime,
                            "gpu_device": gpu_device,
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
            "timing": "warm synchronized wall time; fixture preparation excluded; resident excludes transfers; transfer-inclusive uploads all inputs and exports all outputs",
            "results": rows,
            "skipped": skipped,
        }
        args.json.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
