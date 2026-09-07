from __future__ import annotations

import importlib.util
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
    results = [
        {"section": "Codec I/O", "blanket_speedup": 2.0},
        {"section": "Codec I/O", "blanket_speedup": 0.5},
        {"section": "Memory", "blanket_speedup": None},
    ]

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
    results = [
        slower,
        {"operation": "equal", "blanket_speedup": 1.0},
        {"operation": "faster", "blanket_speedup": 2.0},
        {"operation": "unpaired", "blanket_speedup": None},
    ]

    assert benchmark.slower_results(results) == [slower]
