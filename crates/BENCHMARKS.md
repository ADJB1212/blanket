# Native Rust benchmarks

Run from the repository root using the pinned nightly toolchain:

```sh
cargo bench -p blanket-ops --bench resize
cargo bench -p blanket-codec-jxl --bench codec
```

Defaults favor quick iteration: 20 samples, 100 ms warm-up, a 300 ms
measurement target, and 10,000 bootstrap resamples per case. The default matrix
uses 512x512, 8-bit RGB (RGBA for alpha compositing), with patterned inputs for
codecs and file I/O. Processing kernels retain their random input pixels.
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

| Crate                    | Targets                          | Workloads                                                                                                                |
| ------------------------ | -------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `blanket-core`           | `conversion`                     | L/RGB/RGBA conversions at 8/16 bits.                                                                                     |
| `blanket-ops`            | `resize`, `histogram`            | Nearest/bilinear/Lanczos downsampling, upscaling, and fractional crops; 8-bit histograms.                                  |
| `blanket-filter`         | `blur`                           | Gaussian blur at 8/16 bits.                                                                                              |
| `blanket-enhance`        | `blend`                          | Blending at 8/16 bits.                                                                                                   |
| `blanket-composite`      | `alpha_composite`                | RGBA alpha compositing at 8/16 bits.                                                                                     |
| `blanket-codec-png`      | `codec`                          | PNG levels 1/6, encoding and decoding; compressed text and ICC decode overhead.                                           |
| `blanket-codec-jpeg`     | `codec`, `streaming`             | JPEG quality 85; small-image context reuse; baseline/progressive chunked decoding.                                       |
| `blanket-codec-jxl`      | `codec`, `settings`, `streaming` | JXL lossy/lossless encoding and decoding; float-to-8-bit decoding; efforts and thread counts; chunked decoding.           |
| `blanket-codec-webp`     | `codec`                          | WebP lossy/lossless encoding and decoding.                                                                               |
| `blanket-codec-jpeg2000` | `codec`                          | Lossless JPEG 2000 encoding and decoding.                                                                                |
| `blanket-codec-avif`     | `codec`                          | AVIF quality 85, effort 3.                                                                                               |
| `blanket-codec-heif`     | `codec`                          | HEIF quality 85.                                                                                                         |
| `blanket-codecs`         | `io`                             | Encoded file reads/writes, memory open with format detection, read-and-open, and encode-and-write for PNG/JPEG/JXL/WebP. |

The dev-only `blanket-bench-support` crate shares fixtures and timing helpers;
it depends on the core and Criterion, without pulling in other codec backends.

The full matrix uses 512x512 and 2048x1536. Codec cases exercise L/RGB/RGBA for PNG,
JXL, and WebP; JPEG uses L/RGB, and other formats use RGB. AVIF and HEIF stop
at 512x512 to keep their slower encoders practical. PNG, JXL, and JPEG 2000
also cover L/RGB/RGBA at 10/12/16 bits on patterned 512x512 images.

JXL's main codec cases use effort 3 and quality 85 or lossless encoding. The
`jxl-float` cases decode floating-point RGB input to the normal 8-bit display
output at each selected size, with encoding outside the measurement. The
`jxl-settings` group compares efforts 1/3/7, lossy/lossless encoding, and one,
two, or four threads versus libjxl's resizable worker pool, at 8/16 bits on medium and large
RGB images. These measurements include sample preparation on each encode.
WebP uses quality 85 for lossy encoding and its normal lossless settings.
AVIF uses quality 85 and effort 3; HEIF uses quality 85 and its normal preset.

The `jpeg-context` and `jxl-context` groups compare 32x32 and 64x64 operations
against fresh codec instances with the original settings. The `png-text` and
`png-icc` cases isolate native raster decoding with 1 MiB of compressed metadata;
public `Image.open` still parses that metadata separately. The `io-metadata`
group compares native metadata attachment and file writing against the prior
Python-bytes pipeline. Its PNG case uses level 0 to expose buffer-copy costs.

In the full matrix, codec and file workloads use both a compressible gradient/checker pattern and
deterministic random pixels. These synthetic fixtures bracket compressibility;
neither is a substitute for a photographic corpus. Fixtures and encoded decode
inputs are prepared outside timing. Normal output allocation and destruction
are included. Inputs are reused between iterations, so results represent warm
processing, including warmed resize coefficient caches and codec thread pools.

File benchmarks use an automatically cleaned temporary directory. `read-bytes`
and `write-bytes` measure filesystem calls and allocation without codec work;
`open-memory` measures detection and decoding without file reads. `read-and-open`
and `encode-and-write` include both parts. Writes reopen and truncate a file on
each iteration. They do not call `sync_all`; reads normally hit the OS page
cache. These measure cached file I/O, not durable storage or cold disk speed.

Streaming benchmarks feed baseline JPEG, progressive JPEG, and JXL in 1 KiB,
16 KiB, and 64 KiB chunks. Each iteration constructs a new native decoder,
feeds the entire encoded input, and retrieves the final image. They include
decoder destruction and final image copying, without intermediate preview
requests. These chunked APIs have their own thread behavior; they do not use
the eager JXL decoder's resizable worker pool.

Throughput counts source pixels (including for resize) except raw file
reads/writes, which count encoded bytes. Backend benchmarks call each codec's
native entry points, including required sample normalization. File pipelines
use the normal format detection and dispatch paths in `blanket-codecs`.

Install the selected crate's native build dependencies. For example, JXL needs
libjxl (or `--features jxl-vendored`), HEIF needs libheif with an HEVC encoder
(or `--features embedded-libheif` plus the codec libraries), and AVIF needs
dav1d. WebP can use `--features no-vendored-webp` for system libraries.
`blanket-codecs` I/O requires the full codec dependency graph; use
`--features no-vendored-webp` there to match the extension's default WebP setup.
Backend failures fail the run rather than silently omitting cases.

Filter benchmarks, smoke test all cases, or save and compare a baseline:

```sh
BLANKET_BENCH_FULL=1 cargo bench -p blanket-ops --bench resize -- 'resize/RGBA-16'
cargo bench -p blanket-codec-jxl --bench codec -- 'codecs/jxl-lossless'
BLANKET_BENCH_FULL=1 cargo bench -p blanket-codec-jxl --bench settings -- 'jxl-settings/RGB-16'
cargo bench -p blanket-codec-jpeg --bench streaming -- 'streaming/jpeg-progressive'
cargo bench -p blanket-codecs --bench io --features no-vendored-webp
cargo bench -p blanket-codec-jxl --bench codec -- --test
cargo bench -p blanket-codec-jxl --bench codec -- --save-baseline before
cargo bench -p blanket-codec-jxl --bench codec -- --baseline before
```

Criterion uses Cargo's optimized bench profile and writes results and HTML
reports under `target/criterion/`. Keep build settings, hardware, and Rayon
thread counts consistent across comparisons. `RAYON_NUM_THREADS=1` limits
Blanket's Rayon kernels; external codecs manage their own workers. Use the
JXL settings group's `single` cases to measure JXL encoding with one thread.
