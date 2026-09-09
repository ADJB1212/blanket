from __future__ import annotations

import importlib.util
from io import BytesIO
import runpy

import pytest
from blanket import Image
import sys
from pathlib import Path
from typing import Any


def load_benchmark() -> Any:
    path = Path(__file__).parents[1] / "scripts" / "benchmark.py"
    spec = importlib.util.spec_from_file_location("benchmark", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_summary_table_has_one_row_per_section() -> None:
    benchmark = load_benchmark()
    results = [{"section": "Codec I/O", "blanket_speedup": 2.0}, {"section": "Codec I/O", "blanket_speedup": 0.5}, {"section": "Memory", "blanket_speedup": None}]

    table = benchmark.make_summary_table(results)

    assert table.row_count == 2
    assert [column.header for column in table.columns] == ["Section", "Faster", "Geo mean", "Best", "Worst"]


def test_slower_only_argument(monkeypatch: Any) -> None:
    benchmark = load_benchmark()
    monkeypatch.setattr(sys, "argv", ["benchmark.py", "--slower-only"])

    arguments = benchmark.parse_args()

    assert arguments.slower_only is True


def test_slower_results_excludes_faster_and_unpaired_operations() -> None:
    benchmark = load_benchmark()
    slower = {"operation": "slower", "blanket_speedup": 0.5}
    results = [slower, {"operation": "equal", "blanket_speedup": 1.0}, {"operation": "faster", "blanket_speedup": 2.0}, {"operation": "unpaired", "blanket_speedup": None}]

    assert benchmark.slower_results(results) == [slower]


def test_sections_default_to_all(monkeypatch: Any) -> None:
    benchmark = load_benchmark()
    monkeypatch.setattr(sys, "argv", ["benchmark.py"])
    assert benchmark.parse_args().sections == list(benchmark.SECTION_NAMES)


def test_sections_select_multiple_names(monkeypatch: Any) -> None:
    benchmark = load_benchmark()
    monkeypatch.setattr(sys, "argv", ["benchmark.py", "--sections", "ImagePalette", "Codec I/O"])
    assert benchmark.parse_args().sections == ["ImagePalette", "Codec I/O"]


def test_sections_reject_invalid_arguments(monkeypatch: Any) -> None:
    import pytest

    benchmark = load_benchmark()
    for arguments in (["--sections", "unknown"], ["--sections"], ["--sections", "Resize", "--filter-only"]):
        monkeypatch.setattr(sys, "argv", ["benchmark.py", *arguments])
        with pytest.raises(SystemExit) as error:
            benchmark.parse_args()
        assert error.value.code == 2


def test_sections_only_build_selected_comparisons(monkeypatch: Any, tmp_path: Path) -> None:
    import json

    benchmark = load_benchmark()
    calls: list[str] = []

    def builder(name: str) -> Any:
        def build(*args: Any, **kwargs: Any) -> list[Any]:
            calls.append(name)
            return [(name, lambda: None, lambda: None)]
        return build

    for name in ("codec", "conversion", "resize", "geometry", "band_statistics", "memory", "imageops", "imageenhance", "imagefilter", "imagepalette"):
        monkeypatch.setattr(benchmark, f"{name}_comparisons", builder(name))
    for sections, expected in ((["ImagePalette"], ["imagepalette"]), (["Memory", "ImageFilter"], ["memory", "imagefilter"])):
        calls.clear()
        output = tmp_path / "results.json"
        monkeypatch.setattr(sys, "argv", ["benchmark.py", "--sections", *sections, "--sizes", "4K", "-i", "1", "-w", "0", "--json", str(output)])
        benchmark.main()
        assert calls == expected
        results = json.loads(output.read_text())
        assert {row["section"] for row in results} == set(sections)


@pytest.fixture(scope="module")
def benchmark() -> dict[str, Any]:
    return runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts" / "benchmark.py"))


def test_new_codec_benchmarks(benchmark: dict[str, Any]) -> None:
    comparisons = benchmark["codec_comparisons"]((32, 24), skip_jxl=True)
    names = {name for name, _, _ in comparisons}
    for mode in ("L", "RGB", "RGBA"):
        for action in ("load", "save"):
            assert f"{action} TIFF {mode}" in names
            for setting in ("lossless", "q=50", "q=85", "q=95"):
                assert f"{action} WEBP {mode} {setting}" in names
    assert {"load DNG LinearRaw", "load DNG CFA"} <= names
    assert not any("SVG" in name for name in names)
    for name, blanket, pillow in comparisons:
        if not any(fmt in name for fmt in ("TIFF", "WEBP", "DNG")):
            continue
        result = blanket()
        if name.startswith("load"):
            assert result.size == (32, 24)
        if "DNG" in name:
            assert pillow is None
        else:
            assert pillow is not None
            pillow()


@pytest.mark.parametrize("cfa", [False, True])
def test_benchmark_dng_dimensions(benchmark: dict[str, Any], cfa: bool) -> None:
    payload = benchmark["make_dng"]((32, 24), cfa=cfa)
    result = Image.open(BytesIO(payload))
    assert (result.format, result.mode, result.size) == ("DNG", "RGB", (32, 24))


def test_jxl_only_excludes_other_codecs(benchmark: dict[str, Any]) -> None:
    comparisons = benchmark["codec_comparisons"]((32, 24), skip_jxl=False, jxl_only=True)
    assert comparisons
    assert all("JXL" in name for name, _, _ in comparisons)
