from __future__ import annotations

import json
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

from blanket import Image


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_compression.py"


def test_compression_benchmark_cli(tmp_path: Path) -> None:
    report = tmp_path / "compression.json"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--width", "17", "--height", "9",
         "--efforts", "1", "--repeats", "1", "--warmups", "0", "--json", str(report)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    rows = json.loads(report.read_text())["results"]
    assert len(rows) == 9
    assert all(row["lossless"] and row["non_growing"] for row in rows)
    assert all(row["baseline_ms"] > 0 and row["optimized_ms"] > 0 for row in rows)


def test_benchmark_detects_hidden_color_loss(monkeypatch: pytest.MonkeyPatch) -> None:
    benchmark = runpy.run_path(str(SCRIPT))["benchmark"]
    save = Image.Image.save

    def corrupt_save(self: Image.Image, *args: object, **kwargs: object) -> None:
        if "compressor" in kwargs:
            kwargs.pop("compressor")
            self = Image.frombytes("RGBA", self.size, bytes((0, 0, 0, 0)) * (self.width * self.height))
        save(self, *args, **kwargs)

    monkeypatch.setattr(Image.Image, "save", corrupt_save)
    image = Image.frombytes("RGBA", (2, 2), bytes((19, 73, 151, 0)) * 4)
    result = benchmark(image, "PNG", effort=1, repeats=2, warmups=0)
    assert result["lossless"] is False
