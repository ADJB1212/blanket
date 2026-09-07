"""Benchmark Blanket's supported operations against Pillow."""

from __future__ import annotations

import argparse
import gc
import json
from collections.abc import Callable
from functools import partial
from io import BytesIO
from pathlib import Path
from statistics import geometric_mean, median
from time import perf_counter
from typing import Any

import numpy as np
import pillow_jxl
from blanket import Image as BlanketImage
from blanket import ImageEnhance as BlanketEnhance
from blanket import ImageOps as BlanketOps
from PIL import Image as PillowImage
from PIL import ImageEnhance as PillowEnhance
from PIL import ImageOps as PillowOps
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# ── image sizes ──────────────────────────────────────────────────────────

SIZES: dict[str, tuple[int, int]] = {"web": (800, 600), "square": (1024, 1024), "photo": (1920, 1280), "4K": (3840, 2160)}


# ── helpers ──────────────────────────────────────────────────────────────


def make_rgb(width: int, height: int) -> bytes:
    return bytes((index * 37 + (index // 3) * 11) & 0xFF for index in range(width * height * 3))


def make_rgba(width: int, height: int) -> bytes:
    return bytes((index * 37 + (index // 3) * 11) & 0xFF for index in range(width * height * 4))


def make_gray(width: int, height: int) -> bytes:
    return bytes((index * 37) & 0xFF for index in range(width * height))


def pillow_payload(image: PillowImage.Image, fmt: str, **options: object) -> bytes:
    output = BytesIO()
    image.save(output, fmt, **options)
    return output.getvalue()


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


# ── comparison builders ──────────────────────────────────────────────────

Comparison = tuple[str, Callable[[], object], Callable[[], object] | None]


def codec_comparisons(size: tuple[int, int], *, skip_jxl: bool, jxl_only: bool = False) -> list[Comparison]:
    """Build load/save benchmarks for every codec × relevant mode."""
    w, h = size
    raw_rgb = make_rgb(w, h)
    raw_rgba = make_rgba(w, h)
    raw_gray = make_gray(w, h)

    b_rgb = BlanketImage.frombytes("RGB", size, raw_rgb)
    p_rgb = PillowImage.frombytes("RGB", size, raw_rgb)
    b_rgba = BlanketImage.frombytes("RGBA", size, raw_rgba)
    p_rgba = PillowImage.frombytes("RGBA", size, raw_rgba)
    b_gray = BlanketImage.frombytes("L", size, raw_gray)
    p_gray = PillowImage.frombytes("L", size, raw_gray)

    def pillow_load(payload: bytes) -> None:
        with PillowImage.open(BytesIO(payload)) as img:
            img.load()

    comps: list[Comparison] = []

    if not jxl_only:
        # ── PNG ───────────────────────────────────────────────────────
        for label, b_img, p_img, mode in (("RGB", b_rgb, p_rgb, "RGB"), ("RGBA", b_rgba, p_rgba, "RGBA"), ("L", b_gray, p_gray, "L")):
            png = pillow_payload(p_img, "PNG", compress_level=6)
            comps.append((f"load PNG {label}", lambda p=png: BlanketImage.open(BytesIO(p)), lambda p=png: pillow_load(p)))
        for level in (1, 6, 9):
            comps.append((f"save PNG clvl={level}", lambda b=b_rgb, lv=level: b.save(BytesIO(), "PNG", compress_level=lv), lambda p=p_rgb, lv=level: p.save(BytesIO(), "PNG", compress_level=lv)))

        # ── JPEG ──────────────────────────────────────────────────────
        for label, b_img, p_img in (("RGB", b_rgb, p_rgb), ("L", b_gray, p_gray)):
            jpeg = pillow_payload(p_img, "JPEG", quality=85)
            comps.append((f"load JPEG {label}", lambda p=jpeg: BlanketImage.open(BytesIO(p)), lambda p=jpeg: pillow_load(p)))
        for quality in (50, 85, 95):
            comps.append((f"save JPEG q={quality}", lambda b=b_rgb, q=quality: b.save(BytesIO(), "JPEG", quality=q), lambda p=p_rgb, q=quality: p.save(BytesIO(), "JPEG", quality=q)))

    # ── JXL (Pillow support via pillow-jxl-plugin) ────────────────────
    if not skip_jxl:
        jxl_ll = BytesIO()
        b_rgb.save(jxl_ll, "JXL", lossless=True, effort=1)
        jxl_ll_bytes = jxl_ll.getvalue()
        jxl_lossy = BytesIO()
        b_rgb.save(jxl_lossy, "JXL", quality=90, effort=3)
        jxl_lossy_bytes = jxl_lossy.getvalue()

        comps += [
            ("load JXL lossless", lambda p=jxl_ll_bytes: BlanketImage.open(BytesIO(p)), lambda p=jxl_ll_bytes: pillow_load(p)),
            ("load JXL lossy", lambda p=jxl_lossy_bytes: BlanketImage.open(BytesIO(p)), lambda p=jxl_lossy_bytes: pillow_load(p)),
            ("save JXL lossless eff=1", lambda b=b_rgb: b.save(BytesIO(), "JXL", lossless=True, effort=1), lambda p=p_rgb: p.save(BytesIO(), "JXL", lossless=True, effort=1)),
            ("save JXL q90 eff=3", lambda b=b_rgb: b.save(BytesIO(), "JXL", quality=90, effort=3), lambda p=p_rgb: p.save(BytesIO(), "JXL", quality=90, effort=3)),
            ("save JXL q90 eff=7", lambda b=b_rgb: b.save(BytesIO(), "JXL", quality=90, effort=7), lambda p=p_rgb: p.save(BytesIO(), "JXL", quality=90, effort=7)),
        ]

    return comps


def conversion_comparisons(size: tuple[int, int]) -> list[Comparison]:
    """Build benchmarks for all six non-identity mode conversions."""
    raw_rgb = make_rgb(*size)
    raw_rgba = make_rgba(*size)
    raw_gray = make_gray(*size)

    b_rgb = BlanketImage.frombytes("RGB", size, raw_rgb)
    p_rgb = PillowImage.frombytes("RGB", size, raw_rgb)
    b_rgba = BlanketImage.frombytes("RGBA", size, raw_rgba)
    p_rgba = PillowImage.frombytes("RGBA", size, raw_rgba)
    b_gray = BlanketImage.frombytes("L", size, raw_gray)
    p_gray = PillowImage.frombytes("L", size, raw_gray)

    return [
        ("cvt RGB→L", lambda b=b_rgb: b.convert("L"), lambda p=p_rgb: p.convert("L")),
        ("cvt RGB→RGBA", lambda b=b_rgb: b.convert("RGBA"), lambda p=p_rgb: p.convert("RGBA")),
        ("cvt RGBA→RGB", lambda b=b_rgba: b.convert("RGB"), lambda p=p_rgba: p.convert("RGB")),
        ("cvt RGBA→L", lambda b=b_rgba: b.convert("L"), lambda p=p_rgba: p.convert("L")),
        ("cvt L→RGB", lambda b=b_gray: b.convert("RGB"), lambda p=p_gray: p.convert("RGB")),
        ("cvt L→RGBA", lambda b=b_gray: b.convert("RGBA"), lambda p=p_gray: p.convert("RGBA")),
    ]


def resize_comparisons(size: tuple[int, int]) -> list[Comparison]:
    """Benchmark resize filters, source boxes, and integer reduction by mode."""
    w, h = size
    down = (max(1, w // 2), max(1, h // 2))
    up = (w * 2, h * 2)
    thumbnail = (max(1, w // 8), max(1, h // 8))
    box = (w * 0.1, h * 0.1, w * 0.9, h * 0.9)
    comps: list[Comparison] = []
    for mode, make_pixels in (("L", make_gray), ("RGB", make_rgb), ("RGBA", make_rgba)):
        raw = make_pixels(w, h)
        blanket = BlanketImage.frombytes(mode, size, raw)
        pillow = PillowImage.frombytes(mode, size, raw)
        for method in BlanketImage.Resampling:
            for direction, target in (("down", down), ("up", up)):
                comps.append((f"resize {direction} {method.name} {mode}", partial(blanket.resize, target, resample=method), partial(pillow.resize, target, resample=int(method))))
        cases = [("box BICUBIC", down, {"box": box}), ("thumb LANCZOS", thumbnail, {"resample": BlanketImage.Resampling.LANCZOS}), ("thumb LANCZOS gap=3", thumbnail, {"resample": BlanketImage.Resampling.LANCZOS, "reducing_gap": 3.0})]
        for label, target, options in cases:
            comps.append((f"resize {label} {mode}", partial(blanket.resize, target, **options), partial(pillow.resize, target, **options)))
    return comps


def geometry_comparisons(size: tuple[int, int]) -> list[Comparison]:
    """Benchmark crop, rotation, every transpose, and every transform by mode."""
    w, h = size
    target = (max(1, w // 2), max(1, h // 2))
    inset = (w * 0.1, h * 0.1, w * 0.9, h * 0.9)
    quad = (w * 0.05, h * 0.1, w * 0.1, h * 0.95,
            w * 0.9, h * 0.9, w * 0.95, h * 0.05)
    mesh = [((0, 0, *target), quad)]
    transforms = [
        (BlanketImage.Transform.AFFINE, (1.1, 0.2, -w * 0.05, -0.1, 1.2, h * 0.05)),
        (BlanketImage.Transform.EXTENT, inset),
        (BlanketImage.Transform.PERSPECTIVE, (1.1, 0.1, 0, -0.1, 1.2, 0, 0.1 / w, -0.1 / h)),
        (BlanketImage.Transform.QUAD, quad),
        (BlanketImage.Transform.MESH, mesh),
    ]
    comps: list[Comparison] = []
    for mode, make_pixels in (("L", make_gray), ("RGB", make_rgb), ("RGBA", make_rgba)):
        raw = make_pixels(w, h)
        blanket = BlanketImage.frombytes(mode, size, raw)
        pillow = PillowImage.frombytes(mode, size, raw)
        for label, box in (("copy", None), ("inset", inset), ("padded", (-w // 10, -h // 10, w, h))):
            comps.append((f"Image.crop {label} {mode}", partial(blanket.crop, box=box), partial(pillow.crop, box=box)))
        for method in BlanketImage.Transpose:
            comps.append((f"transpose {method.name} {mode}", partial(blanket.transpose, method), partial(pillow.transpose, int(method))))
        comps.append((f"rotate 90 expand {mode}", partial(blanket.rotate, 90, expand=True), partial(pillow.rotate, 90, expand=True)))
        for resample in (BlanketImage.Resampling.NEAREST, BlanketImage.Resampling.BILINEAR, BlanketImage.Resampling.BICUBIC):
            for label, options in (("17", {}), ("17 expand fill", {"expand": True, "fillcolor": "navy"})):
                comps.append((f"rotate {label} {resample.name} {mode}", partial(blanket.rotate, 17, resample=resample, **options), partial(pillow.rotate, 17, resample=int(resample), **options)))
            for method, data in transforms:
                comps.append((f"transform {method.name} {resample.name} {mode}", partial(blanket.transform, target, method, data, resample=resample), partial(pillow.transform, target, int(method), data, resample=int(resample))))
    return comps


def band_statistics_comparisons(size: tuple[int, int]) -> list[Comparison]:
    """Benchmark band copies, integer reduction, and masked entropy."""
    w, h = size
    mask_raw = bytes(i % 2 for i in range(w * h))
    b_mask = BlanketImage.frombytes("L", size, mask_raw)
    p_mask = PillowImage.frombytes("L", size, mask_raw)
    comps: list[Comparison] = []
    for mode, make_pixels in (("L", make_gray), ("RGB", make_rgb), ("RGBA", make_rgba)):
        raw = make_pixels(w, h)
        blanket = BlanketImage.frombytes(mode, size, raw)
        pillow = PillowImage.frombytes(mode, size, raw)
        comps.append((f"split {mode}", blanket.split, pillow.split))
        comps.append((f"entropy {mode}", blanket.entropy, pillow.entropy))
        comps.append((f"entropy masked {mode}", partial(blanket.entropy, b_mask), partial(pillow.entropy, p_mask)))
        for factor in (2, 3, (2, 3)):
            comps.append((f"reduce {factor} {mode}", partial(blanket.reduce, factor), partial(pillow.reduce, factor)))
        box = (w // 10, h // 10, w - w // 10, h - h // 10)
        comps.append((f"reduce box {mode}", partial(blanket.reduce, 3, box=box), partial(pillow.reduce, 3, box=box)))
    return comps


def memory_comparisons(size: tuple[int, int]) -> list[Comparison]:
    """Benchmark array/byte construction, extraction, and Pillow conversion."""
    raw = make_rgb(*size)
    array = np.frombuffer(raw, dtype=np.uint8).reshape(size[1], size[0], 3)

    b_img = BlanketImage.frombytes("RGB", size, raw)
    p_img = PillowImage.frombytes("RGB", size, raw)

    return [
        ("fromarray RGB", lambda a=array: BlanketImage.fromarray(a), lambda a=array: PillowImage.fromarray(a)),
        ("frombytes RGB", lambda r=raw, s=size: BlanketImage.frombytes("RGB", s, r), lambda r=raw, s=size: PillowImage.frombytes("RGB", s, r)),
        ("tobytes RGB", lambda b=b_img: b.tobytes(), lambda p=p_img: p.tobytes()),
        ("to_pillow()", lambda b=b_img: b.to_pillow(), None),
    ]


class InsetMesh:
    """Map an inset source rectangle onto the full destination."""

    def getmesh(self, image: BlanketImage.Image | PillowImage.Image) -> list[tuple[tuple[int, int, int, int], tuple[float, ...]]]:
        w, h = image.size
        x, y = w * 0.05, h * 0.05
        return [((0, 0, w, h), (x, y, x, h - y, w - x, h - y, w - x, y))]


def imageops_comparisons(size: tuple[int, int]) -> list[Comparison]:
    """Benchmark all ImageOps functions on their supported Blanket modes.

    Inputs and EXIF payloads are prepared outside the timed calls.
    Operations return new images so every iteration sees the same source.
    """
    w, h = size
    target = (max(1, w // 2), max(1, h // 3))
    border = max(0, min(w, h) // 20)
    comps: list[Comparison] = []
    exif = PillowImage.Exif()
    exif[274] = 6
    exif_bytes = exif.tobytes()
    for mode, make_pixels in (("L", make_gray), ("RGB", make_rgb), ("RGBA", make_rgba)):
        raw = make_pixels(w, h)
        blanket = BlanketImage.frombytes(mode, size, raw)
        pillow = PillowImage.frombytes(mode, size, raw)
        blanket.info["exif"] = pillow.info["exif"] = exif_bytes
        cases = [
            ("contain", {"size": target}),
            ("cover", {"size": target}),
            ("crop", {"border": border}),
            ("deform", {"deformer": InsetMesh()}),
            ("exif_transpose", {}),
            ("expand", {"border": border, "fill": "navy"}),
            ("fit", {"size": target, "bleed": 0.05}),
            ("flip", {}),
            ("grayscale", {}),
            ("mirror", {}),
            ("pad", {"size": target, "color": "navy"}),
            ("scale", {"factor": 0.5}),
        ]
        if mode in ("L", "RGB"):
            cases += [("autocontrast", {"cutoff": 1}), ("equalize", {}), ("invert", {}), ("posterize", {"bits": 4}), ("solarize", {"threshold": 128})]
        if mode == "L":
            cases.append(("colorize", {"black": "navy", "white": "gold"}))
        for name, options in cases:
            label = f"ImageOps.crop border={border}" if name == "crop" else name
            comps.append((f"{label} {mode}", partial(getattr(BlanketOps, name), blanket, **options), partial(getattr(PillowOps, name), pillow, **options)))
    return comps


def imageenhance_comparisons(size: tuple[int, int]) -> list[Comparison]:
    """Benchmark construction and enhancement for every class and mode."""
    comps: list[Comparison] = []
    for mode, make_pixels in (("L", make_gray), ("RGB", make_rgb), ("RGBA", make_rgba)):
        raw = make_pixels(*size)
        blanket = BlanketImage.frombytes(mode, size, raw)
        pillow = PillowImage.frombytes(mode, size, raw)
        for name in ("Color", "Contrast", "Brightness", "Sharpness"):
            blanket_class = getattr(BlanketEnhance, name)
            pillow_class = getattr(PillowEnhance, name)
            comps.append((f"{name} {mode}", lambda cls=blanket_class, image=blanket: cls(image).enhance(1.5), lambda cls=pillow_class, image=pillow: cls(image).enhance(1.5)))
    return comps


# ── Rich output ─────────────────────────────────────────────────────────


def _speedup_text(value: float | None) -> Text:
    """Return a colored speedup cell."""
    if value is None:
        return Text("—", style="dim")
    label = f"{value:.2f}x"
    if value >= 2.0:
        return Text(label, style="bold green")
    if value >= 1.0:
        return Text(label, style="green")
    return Text(label, style="red")


def make_table(results: list[dict[str, Any]]) -> Table:
    """Build one compact Rich table spanning all benchmark sections."""
    table = Table(show_lines=False, pad_edge=False, box=None)
    table.add_column("Operation", style="cyan", no_wrap=True)
    table.add_column("Blanket ms", justify="right")
    table.add_column("Pillow ms", justify="right")
    table.add_column("Speedup", justify="right")

    last_section: str | None = None
    for r in results:
        if r["section"] != last_section:
            table.add_row(Text(r["section"], style="bold underline"))
            last_section = r["section"]
        spread = (r["blanket_p75_ms"] - r["blanket_p25_ms"]) / 2
        blanket_cell = f"{r['blanket_ms']:.2f} ±{spread:.2f}"
        pillow_cell = f"{r['pillow_ms']:.2f}" if r["pillow_ms"] is not None else "—"
        table.add_row(r["operation"], blanket_cell, pillow_cell, _speedup_text(r["blanket_speedup"]))

    return table


# ── main ─────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", choices=list(SIZES.keys()), default=list(SIZES.keys()), help="image size presets to benchmark (default: all)")
    parser.add_argument("-i", "--iterations", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--skip-jxl", action="store_true", help="skip JXL benchmarks (pillow-jxl-plugin required otherwise)")
    parser.add_argument("--jxl-only", action="store_true", help="only run JXL codec benchmarks (pillow-jxl-plugin required)")
    parser.add_argument("--json", type=Path, dest="json_path")
    arguments = parser.parse_args()
    if arguments.iterations < 1:
        parser.error("--iterations must be positive")
    if arguments.warmups < 0:
        parser.error("--warmups cannot be negative")
    if arguments.skip_jxl and arguments.jxl_only:
        parser.error("--skip-jxl and --jxl-only are mutually exclusive")
    return arguments


def run_section(comparisons: list[Comparison], warmups: int, iterations: int) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for name, blanket_op, pillow_op in comparisons:
        b = measure(blanket_op, warmups, iterations)
        if pillow_op is not None:
            p = measure(pillow_op, warmups, iterations)
            results.append({"operation": name, "blanket_ms": b["median"] * 1000, "blanket_p25_ms": b["p25"] * 1000, "blanket_p75_ms": b["p75"] * 1000, "pillow_ms": p["median"] * 1000, "blanket_speedup": p["median"] / b["median"]})
        else:
            results.append({"operation": name, "blanket_ms": b["median"] * 1000, "blanket_p25_ms": b["p25"] * 1000, "blanket_p75_ms": b["p75"] * 1000, "pillow_ms": None, "blanket_speedup": None})
    return results


def main() -> None:
    args = parse_args()
    console = Console()
    all_results: list[dict[str, Any]] = []

    for size_name in args.sizes:
        size = SIZES[size_name]
        w, h = size
        mpx = (w * h) / 1_000_000

        console.rule(f"[bold]{size_name}[/bold]  {w}×{h}  ({mpx:.2f} Mpx)  —  median of {args.iterations} runs")

        sections: list[tuple[str, list[Comparison]]] = [("Codec I/O", codec_comparisons(size, skip_jxl=args.skip_jxl, jxl_only=args.jxl_only))]
        if not args.jxl_only:
            sections += [("Conversions", conversion_comparisons(size)), ("Resize", resize_comparisons(size)), ("Geometry", geometry_comparisons(size)), ("Bands and statistics", band_statistics_comparisons(size)), ("Memory", memory_comparisons(size)), ("ImageOps", imageops_comparisons(size)), ("ImageEnhance", imageenhance_comparisons(size))]

        size_results: list[dict[str, Any]] = []
        for section_name, comparisons in sections:
            results = run_section(comparisons, args.warmups, args.iterations)
            for r in results:
                r["section"] = section_name
                r["size"] = size_name
                r["width"] = w
                r["height"] = h
            size_results.extend(results)

        console.print(make_table(size_results))
        console.print()
        all_results.extend(size_results)

    # ── summary ──────────────────────────────────────────────────────
    paired = [r for r in all_results if r["blanket_speedup"] is not None]
    if paired:
        speedups = [r["blanket_speedup"] for r in paired]
        wins = sum(1 for s in speedups if s > 1.0)
        geo_mean = geometric_mean(speedups)
        best = max(paired, key=lambda r: r["blanket_speedup"])
        worst = min(paired, key=lambda r: r["blanket_speedup"])

        summary = (
            f"[bold]{wins}[/bold]/{len(paired)} operations faster than Pillow\nGeometric mean speedup: [bold]{geo_mean:.2f}x[/bold]\nBest:  [green]{best['operation']}[/green] @ {best['size']} ({best['blanket_speedup']:.2f}x)\nWorst: [red]{worst['operation']}[/red] @ {worst['size']} ({worst['blanket_speedup']:.2f}x)"
        )
        console.print(Panel(summary, title="Summary", border_style="bold"))

    if args.json_path is not None:
        args.json_path.write_text(json.dumps(all_results, indent=2) + "\n")


if __name__ == "__main__":
    main()
