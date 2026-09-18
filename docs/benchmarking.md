# Comparing performance across changes

[Back to the README](../README.md) · [All modules](index.md)

The `Bands` section includes pixel access and mutation, histograms, extrema,
bounding boxes, channel extraction, lookup tables, thumbnails, blending, and
masked compositing in L, RGB, and RGBA modes:

```sh
maturin develop -r --extras test --uv
uv run --no-sync scripts/benchmark.py --sections Bands --sizes web --no-baseline
```

Mutating cases include a fresh copy for both libraries on every iteration.
Pixel input lists and lookup tables are prepared outside timing. The `getdata`
comparison materializes Pillow's sequence into a list to match Blanket's
snapshot; `get_flattened_data` compares tuple snapshots directly.

Run the benchmark before editing, then rebuild and run the same command again:

```sh
maturin develop -r --extras test --uv
uv run --no-sync scripts/benchmark.py --sections Resize ImageOps --sizes web -i 30
# Make your changes, then rebuild the native extension.
maturin develop -r --extras test --uv
uv run --no-sync scripts/benchmark.py --sections Resize ImageOps --sizes web -i 30 --compare
```

Filenames are automatic. The first run creates `.benchmarks/baseline.json`;
later runs compare against it only when `--compare` is passed, without replacing it. Every run is saved under
`.benchmarks/` with a timestamp and Git commit in its filename. This directory
is ignored by Git and is always relative to the repository, regardless of your
working directory. Add `--save-baseline` to make the current run the new reference.
`--save-baseline` also accepts an explicit output path. `--compare` is a boolean
flag and always reads `.benchmarks/baseline.json`; it fails if that baseline is missing.
Use `--no-baseline` in CI to disable baseline saving, comparison, and run history.
Explicit `--json` output still works; `--no-baseline` cannot be combined with
`--compare` or `--save-baseline`. CI excludes the benchmark tests with
`pytest --ignore=tests/test_benchmark.py`.

The comparison shows Blanket's percentage time change by section and size, followed
by operations ranked by absolute percentage change with before/after times and
millisecond deltas. Negative percentages mean faster; positive means slower.
Section and overall changes are equally weighted geometric means of matched time
ratios. The default 5% threshold hides small changes in the operation list;
use `--threshold 2` to adjust it or `--verbose` to show every matched operation.
This threshold is a display filter, not a statistical significance test.

Cases match by section, operation, size preset, and dimensions. New cases and
baseline cases not run are counted separately. Existing `--json` files also work
with `--compare`. `--save-baseline` always saves every measured case, even with
`--slower-only`, along with the Git commit, working-tree dirty flag, timestamp,
platform, Python version, and sampling settings. The Git metadata describes the
checkout; rebuild to ensure the installed extension reflects it.

Use the same machine, release build, selections, and sampling settings for both
runs, with other heavy workloads stopped. Repeat runs to check that a change is
consistent. Recorded environment or sampling differences produce a warning.
Baseline comparisons include operations without a Pillow counterpart and are
independent of the existing Pillow comparison tables.
