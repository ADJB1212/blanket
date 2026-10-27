# Native Rust benchmarks

Run from the repository root using the pinned nightly toolchain:

```sh
cargo bench -p blanket-ops --bench resize
cargo bench -p blanket-codec-jxl --bench codec
```

Defaults favor quick iteration: 50 samples, 200 ms warm-up, a 1000 ms
measurement target, and 10,000 bootstrap resamples per case. The default matrix
uses 800x600, 8-bit RGB (RGBA for alpha compositing), with patterned inputs for
codecs and file I/O. Processing kernels retain their random input pixels.
Core conversion also includes 16-bit RGB in the default matrix.
Slow operations can exceed the measurement target to collect all samples.
These short measurements have more noise than the full configuration.

Restore the complete matrix and Criterion's standard sampling settings with:

```sh
BLANKET_BENCH_FULL=1 cargo bench -p blanket-codec-jxl --bench codec
```

Criterion's command-line options still override timing defaults, for example
`-- --measurement-time 2 --warm-up-time 1 --sample-size 50`. Use `-- --test`
to execute each case once without statistical measurement. Compilation uses
the same optimized bench profile in both configurations.

This Criterion suite calls Rust functions directly, separately from
`scripts/benchmark.py` and its Pillow comparisons. It does not require Pillow,
Maturin, or an installed Blanket extension. A Python development installation is
still required to link PyO3. The interpreter is initialized and attached outside
processing measurements because existing Rust entry points accept `Python`;
their normal detach/reattach overhead remains included. No Python API calls or
Python argument conversion are measured.

Benchmarks live in the owning crate's `benches/` directory. Select a crate with
`-p` and a workload with `--bench`; only that crate's dependency graph is built.

| Crate                    | Targets                          | Workloads                                                                                                                    |
| ------------------------ | -------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `blanket-core`           | `conversion`                     | All L/RGB/RGBA mode pairs at 8/10/12/16 bits, plus combined mode/depth conversions and same-mode depth scaling.              |
| `blanket-ops`            | `resize`, `histogram`            | Nearest/bilinear/Lanczos downsampling, upscaling, and fractional crops; random, constant-color, and masked 8-bit histograms. |
| `blanket-filter`         | `blur`                           | Gaussian blur at 8/16 bits.                                                                                                  |
| `blanket-enhance`        | `blend`                          | Blending at 8/16 bits.                                                                                                       |
| `blanket-composite`      | `alpha_composite`                | RGBA alpha compositing at 8/16 bits.                                                                                         |
| `blanket-codec-png`      | `codec`                          | PNG levels 1/6, encoding and decoding; compressed text and ICC decode overhead.                                              |
| `blanket-codec-jpeg`     | `codec`, `streaming`             | JPEG quality 85; small-image context reuse; baseline/progressive chunked decoding.                                           |
| `blanket-codec-jxl`      | `codec`, `settings`, `streaming` | JXL lossy/lossless encoding and decoding; float-to-8-bit decoding; efforts and thread counts; chunked decoding.              |
| `blanket-codec-webp`     | `codec`                          | WebP lossy/lossless encoding and decoding.                                                                                   |
| `blanket-codec-jpeg2000` | `codec`                          | Lossless JPEG 2000 encoding and decoding.                                                                                    |
| `blanket-codec-avif`     | `codec`                          | AVIF quality 85, effort 3.                                                                                                   |
| `blanket-codec-heif`     | `codec`                          | HEIF quality 85.                                                                                                             |
| `blanket-codecs`         | `io`                             | Encoded file reads/writes, memory open with format detection, read-and-open, and encode-and-write for PNG/JPEG/JXL/WebP.     |

The dev-only `blanket-bench-support` crate shares fixtures and timing helpers;
it depends on the core and Criterion, without pulling in other codec backends.

Filter benchmarks, smoke test all cases, or save and compare a baseline:

```sh
BLANKET_BENCH_FULL=1 cargo bench -p blanket-ops --bench resize -- 'resize/RGBA-16'
cargo bench -p blanket-codec-jxl --features jxl-vendored --bench codec -- 'codecs/jxl-lossless'
BLANKET_BENCH_FULL=1 cargo bench -p blanket-codec-jxl --features jxl-vendored --bench settings -- 'jxl-settings/RGB-16'
cargo bench -p blanket-codec-jpeg --bench streaming -- 'streaming/jpeg-progressive'
cargo bench -p blanket-codecs --bench io --features no-vendored-webp
cargo bench -p blanket-codec-jxl --features jxl-vendored --bench codec -- --test
cargo bench -p blanket-codec-jxl --features jxl-vendored --bench codec -- --save-baseline before
cargo bench -p blanket-codec-jxl --features jxl-vendored --bench codec -- --baseline before
```

Criterion uses Cargo's optimized bench profile and writes results and HTML
reports under `target/criterion/`. Keep build settings, hardware, and Rayon
thread counts consistent across comparisons. `RAYON_NUM_THREADS=1` limits
Blanket's Rayon kernels; external codecs manage their own workers.
