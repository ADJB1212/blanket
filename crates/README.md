# Native crates

The `blanket` extension assembles the Python API from focused library crates.
Each operation crate owns its implementations, SIMD helpers, and unit tests.

| Crate | Responsibility | Internal dependencies |
| --- | --- | --- |
| `core` | Image storage, sample modes, pixel conversion, parallel helpers | None |
| `ops` | ImageOps transforms, resampling, LUTs | core |
| `composite` | Compositing and channel arithmetic | core |
| `enhance` | Brightness, contrast, color, sharpness | core |
| `filter` | Convolution, rank, blur filters | core |
| `palette` | Palettes, quantization, dithering | core |
| `stat` | Histogram reductions and moments | None |
| `codecs` | Format detection, decoding, encoding | core |
| `compressor` | Lossless and lossy compression search | core, codecs, ops |
| `blanket` | Python registration and save argument adaptation | All library crates |

Operation crates do not depend on each other. Shared image representations and
pixel layout helpers belong in core. Algorithm-specific SIMD stays beside the
algorithm. Statistics operates on histograms and only needs PyO3.

The extension's `src/encode.rs` handles Python save dispatch and mode adaptation.
Codec implementations and compression search remain independently owned.
Codec build features are forwarded through the extension.

Run commands from the repository root. For example:

```sh
cargo test -p blanket-palette
cargo check --workspace
cargo fmt --check
```

Public Python behavior is tested in `tests/`. Rebuild the extension with
`maturin develop -G dev --uv` before running Python tests after Rust changes.
