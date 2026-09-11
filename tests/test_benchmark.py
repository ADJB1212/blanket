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


def timing(operation: str, milliseconds: float, width: int = 800) -> dict[str, Any]:
    return {"section": "Resize", "operation": operation, "size": "web", "width": width, "height": 600, "blanket_ms": milliseconds}


def test_baseline_matching_and_change_direction() -> None:
    benchmark = load_benchmark()
    old = [timing("faster", 10), timing("slower", 10), timing("resized", 10)]
    new = [timing("slower", 12), timing("faster", 8), timing("resized", 10, 900)]
    matched, added, missing = benchmark.compare_results(new, old)
    assert [row["change_pct"] for row in matched] == pytest.approx([20, -20])
    assert (added, missing) == (1, 1)


def test_conversion_result_labels_use_to(monkeypatch: Any) -> None:
    import json

    benchmark = load_benchmark()
    monkeypatch.setattr(benchmark, "measure", lambda *args: {"median": .01, "p25": .009, "p75": .011})
    results = benchmark.run_section([("cvt RGB→L", lambda: None, lambda: None), ("cvt L 10→8-bit", lambda: None, None)], 0, 1)
    assert [row["operation"] for row in results] == ["cvt RGB to L", "cvt L 10 to 8-bit"]
    assert "\\u2192" not in json.dumps(results)
    matched, added, missing = benchmark.compare_results([timing("cvt RGB to L", 10)], [timing("cvt RGB→L", 10)])
    assert len(matched) == 1
    assert (added, missing) == (0, 0)


@pytest.mark.parametrize("include_unpaired", [False, True])
def test_unpaired_cases_are_opt_in(monkeypatch: Any, include_unpaired: bool) -> None:
    benchmark = load_benchmark()
    calls: list[str] = []

    def measure(operation: Any, *args: Any) -> dict[str, float]:
        operation()
        return {"median": .01, "p25": .009, "p75": .011}

    monkeypatch.setattr(benchmark, "measure", measure)
    results = benchmark.run_section([
        ("paired", lambda: calls.append("blanket"), lambda: calls.append("pillow")),
        ("unpaired", lambda: calls.append("unpaired"), None),
    ], 0, 1, include_unpaired=include_unpaired)
    assert calls == ["blanket", "pillow"] + (["unpaired"] if include_unpaired else [])
    assert len(results) == (2 if include_unpaired else 1)


def test_paired_codecs_skip_unpaired_setup(monkeypatch: Any) -> None:
    benchmark = load_benchmark()

    def unexpected(*args: Any, **kwargs: Any) -> None:
        pytest.fail("unpaired codec setup must be skipped")

    monkeypatch.setattr(benchmark, "make_dng", unexpected)
    monkeypatch.setattr(benchmark, "make_ten_bit", unexpected)
    comparisons = benchmark.codec_comparisons((32, 24), skip_jxl=False, include_unpaired=False)
    assert comparisons
    assert all(pillow is not None for _, _, pillow in comparisons)


def test_baseline_formats_and_validation(tmp_path: Path) -> None:
    import json

    benchmark = load_benchmark()
    path = tmp_path / "baseline.json"
    rows = [timing("resize", 10)]
    for document in (rows, {"metadata": {}, "results": rows}):
        path.write_text(json.dumps(document))
        assert benchmark.load_baseline(path)["results"] == rows
    for document in ({}, [*rows, *rows], [timing("resize", 0)], [timing("resize", float("nan"))], [{}]):
        path.write_text(json.dumps(document))
        with pytest.raises(ValueError):
            benchmark.load_baseline(path)


def test_comparison_output_threshold_and_unpaired_cases() -> None:
    from io import StringIO
    from rich.console import Console

    benchmark = load_benchmark()
    stream = StringIO()
    old = [timing(name, 10) for name in ("faster", "slower", "quiet")]
    new = [timing("faster", 8), timing("slower", 12), timing("quiet", 10.1)]
    benchmark.print_comparison(Console(file=stream, width=160), new, {"results": old}, 5, False)
    output = stream.getvalue()
    assert "3 matched" in output
    assert "-20.0%" in output and "+20.0%" in output
    assert "quiet" not in output
    assert "All matched cases" in output


def test_save_and_compare_baseline_cli(monkeypatch: Any, tmp_path: Path, capsys: Any) -> None:
    import json

    benchmark = load_benchmark()
    monkeypatch.setattr(benchmark, "imagepalette_comparisons", lambda _: [("case", lambda: None, lambda: None)])
    monkeypatch.setattr(benchmark, "measure", lambda *args: {"median": .01, "p25": .009, "p75": .011})
    path = tmp_path / "baseline.json"
    monkeypatch.setattr(benchmark, "RESULTS_DIRECTORY", tmp_path)
    monkeypatch.setattr(sys, "argv", ["benchmark.py", "--sections", "ImagePalette", "--slower-only", "--iterations", "10", "--save-baseline", str(path)])
    benchmark.main()
    document = json.loads(path.read_text())
    assert len(document["results"]) == 1
    assert document["metadata"]["iterations"] == 10
    monkeypatch.setattr(sys, "argv", ["benchmark.py", "--sections", "ImagePalette", "--compare"])
    benchmark.main()
    assert "1 matched" in capsys.readouterr().out
    for extra in (["--threshold", "nan"], ["--json", str(path)], ["--save-baseline", str(path)]):
        monkeypatch.setattr(sys, "argv", ["benchmark.py", "--compare", *extra])
        with pytest.raises(SystemExit):
            benchmark.parse_args()


def test_summary_table_has_one_row_per_section() -> None:
    benchmark = load_benchmark()
    results = [{"section": "Codec I/O", "blanket_speedup": 2.0}, {"section": "Codec I/O", "blanket_speedup": 0.5}, {"section": "Memory", "blanket_speedup": None}]

    table = benchmark.make_summary_table(results)

    assert table.row_count == 2
    assert [column.header for column in table.columns] == ["Section", "Faster", "Geo mean", "Best", "Worst"]


def test_automatic_baseline_and_run_history(monkeypatch: Any, tmp_path: Path, capsys: Any) -> None:
    import json

    benchmark = load_benchmark()
    directory = tmp_path / ".benchmarks"
    monkeypatch.setattr(benchmark, "RESULTS_DIRECTORY", directory)
    monkeypatch.setattr(benchmark, "imagepalette_comparisons", lambda _: [("case", lambda: None, lambda: None)])
    monkeypatch.setattr(benchmark, "measure", lambda *args: {"median": .01, "p25": .009, "p75": .011})
    monkeypatch.setattr(sys, "argv", ["benchmark.py", "--sections", "ImagePalette"])
    benchmark.main()
    baseline = directory / "baseline.json"
    original = baseline.read_text()
    benchmark.main()
    assert baseline.read_text() == original
    assert len(list(directory.glob("run-*.json"))) == 2
    assert "matched" not in capsys.readouterr().out
    for run in directory.glob("run-*.json"):
        assert json.loads(run.read_text())["baseline"] is None
    monkeypatch.setattr(sys, "argv", ["benchmark.py", "--compare"])
    assert benchmark.parse_args().compare is True
    assert benchmark.parse_args().compare_path == baseline
    monkeypatch.setattr(sys, "argv", ["benchmark.py", "--sections", "ImagePalette", "--save-baseline"])
    monkeypatch.setattr(benchmark, "measure", lambda *args: {"median": .02, "p25": .019, "p75": .021})
    benchmark.main()
    assert json.loads(baseline.read_text())["results"][0]["blanket_ms"] == 20
    assert len(list(directory.glob("run-*.json"))) == 3


def test_compare_requires_fixed_baseline(monkeypatch: Any, tmp_path: Path) -> None:
    benchmark = load_benchmark()
    monkeypatch.setattr(benchmark, "RESULTS_DIRECTORY", tmp_path)
    for argv in (["--compare"], ["--compare", str(tmp_path / "other.json")]):
        monkeypatch.setattr(sys, "argv", ["benchmark.py", *argv])
        with pytest.raises(SystemExit):
            benchmark.parse_args()
    (tmp_path / "baseline.json").write_text("invalid json")
    monkeypatch.setattr(sys, "argv", ["benchmark.py"])
    args = benchmark.parse_args()
    assert args.compare is False
    assert args.baseline is None
    assert args.save_baseline is None


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


def test_sections_default_to_core_operations(monkeypatch: Any) -> None:
    benchmark = load_benchmark()
    monkeypatch.setattr(sys, "argv", ["benchmark.py"])
    assert benchmark.parse_args().sections == ["Codec I/O", "Conversions", "Resize", "Memory"]


def test_all_restores_full_suite(monkeypatch: Any) -> None:
    benchmark = load_benchmark()
    monkeypatch.setattr(sys, "argv", ["benchmark.py", "--all"])
    assert benchmark.parse_args().sections == list(benchmark.SECTION_NAMES)
    monkeypatch.setattr(sys, "argv", ["benchmark.py", "--all", "--sizes", "web", "--no-palette", "--skip-jxl"])
    args = benchmark.parse_args()
    assert args.sections == [name for name in benchmark.SECTION_NAMES if name != "ImagePalette"]
    assert args.sizes == ["web"]
    assert args.skip_jxl


def test_sections_select_multiple_names(monkeypatch: Any) -> None:
    benchmark = load_benchmark()
    monkeypatch.setattr(sys, "argv", ["benchmark.py", "--sections", "ImagePalette", "Codec I/O"])
    assert benchmark.parse_args().sections == ["ImagePalette", "Codec I/O"]


def test_sections_reject_invalid_arguments(monkeypatch: Any) -> None:
    import pytest

    benchmark = load_benchmark()
    for arguments in (["--sections", "unknown"], ["--sections"], ["--sections", "Resize", "--filter-only"], ["--all", "--sections", "Resize"], ["--all", "--jxl-only"]):
        monkeypatch.setattr(sys, "argv", ["benchmark.py", *arguments])
        with pytest.raises(SystemExit) as error:
            benchmark.parse_args()
        assert error.value.code == 2


def test_sections_only_build_selected_comparisons(monkeypatch: Any, tmp_path: Path) -> None:
    import json

    benchmark = load_benchmark()
    monkeypatch.setattr(benchmark, "RESULTS_DIRECTORY", tmp_path / "runs")
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
        if "DNG" in name or "10-bit" in name:
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


def test_heif_and_ten_bit_codec_benchmarks(benchmark: dict[str, Any]) -> None:
    comparisons = benchmark["codec_comparisons"]((32, 24), skip_jxl=False)
    names = {name for name, _, _ in comparisons}
    for mode in ("L", "RGB", "RGBA"):
        assert f"save HEIC/HEIF {mode} lossless" in names
        for fmt in ("PNG", "TIFF", "JXL", "HEIC/HEIF"):
            assert f"save {fmt} {mode} 10-bit" in names
            assert f"load {fmt} {mode} 10-bit source" in names
    for name, blanket_op, pillow_op in comparisons:
        if "HEIC/HEIF" in name or "10-bit" in name:
            result = blanket_op()
            if name.startswith("load"):
                assert result.size == (32, 24)
                if "10-bit" in name:
                    assert result.bit_depth >= 10
            if "10-bit" in name:
                assert pillow_op is None
            elif pillow_op is not None:
                pillow_op()


def test_ten_bit_operation_benchmarks(benchmark: dict[str, Any]) -> None:
    for mode in ("L", "RGB", "RGBA"):
        samples = benchmark["make_ten_bit"]((32, 24), mode)
        assert samples.min() == 0
        assert samples.max() == 1023
    comparisons = benchmark["ten_bit_comparisons"]((32, 24))
    assert len(comparisons) == 39
    for name, operation, baseline in comparisons:
        assert baseline is None
        result = operation()
        if isinstance(result, Image.Image):
            assert result.bit_depth == (8 if "10→8" in name else 10)
            assert result.size == ((16, 12) if name.startswith("resize") else (32, 24))
