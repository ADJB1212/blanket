# Development

[Back to the README](../README.md) · [All modules](index.md)

JPEG XL encoding statically builds libjxl. Install Rust, CMake, Ninja, and a
C++ compiler before building. AVIF requires system dav1d 1.3+ (`brew install dav1d`
on macOS or `apt-get install libdav1d-dev` on Ubuntu), also present at runtime.
HEIF requires system libheif 1.17+ with an HEVC
decoder (libde265) and encoder (x265). On macOS use `brew install libheif`;
on Ubuntu install `libheif-dev`, `libheif-plugin-libde265`, and
`libheif-plugin-x265`. The libheif shared library and codec plugins must also
be available at runtime. Pillow and pillow-heif are test dependencies only.

```console
uv sync --extra test
cargo test
uv run pytest
uv run python scripts/verify_behavior.py
uv run python scripts/benchmark.py
```

By default, benchmarks cover Codec I/O, Conversions, Resize, and Memory. Use
`--all` for the full suite, including codecs, or `--sections` to select specific
sections. For a smaller run, use `--sizes web`; `--all --sizes web` runs every
section at web size.
Cases without a Pillow equivalent, including 10-bit I/O, require
`--all` and are otherwise skipped even with an explicit section selection.

The benchmark uses equal in-memory inputs, forces Pillow to decode eagerly,
and summarizes each section using Blanket's speed ratio. Pass `--verbose` to
see timings for every operation, or `--slower-only` to show only operations
where Blanket trails Pillow. JPEG XL results are reported separately because
Pillow has no built-in JPEG XL codec.
