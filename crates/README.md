# Native crates

The `blanket` extension assembles the Python API from focused library crates.
Each operation crate owns its implementations, SIMD helpers, and unit tests.

| Crate           | Responsibility                                                  | Internal dependencies   |
| --------------- | --------------------------------------------------------------- | ----------------------- |
| `core`          | Image storage, conversion dispatch, Python image methods        | kernels, modes, convert-kernels |
| `modes`         | Shared pixel modes and mode parsing                             | None                    |
| `convert-kernels` | Sample, mode, alpha, and float conversion kernels             | kernels, modes          |
| `ops`           | ImageOps transforms, histogram operations, LUTs                 | core, kernels, math     |
| `kernels`       | Shared pixel kernels                                            | None                    |
| `resample-kernels` | Resize and transform SIMD kernels                            | kernels                 |
| `resample`      | Resize, transforms, resampling                                  | core, kernels, resample-kernels |
| `math`          | ImageMath operations                                            | core                    |
| `draw`          | Geometry, masks, flood fill                                     | core                    |
| `font`          | Font loading, shaping, rasterization                            | kernels                 |
| `cms`           | LittleCMS profiles and color transforms                         | core                    |
| `metadata`      | Encoded-container metadata operations                           | codec-common            |
| `composite`     | Compositing and channel arithmetic                              | core                    |
| `enhance`       | Brightness, contrast, color, sharpness                          | core                    |
| `filter`        | Filter registration, color LUTs, band merging                   | core, filter-blur, filter-neighborhood |
| `filter-blur`   | Box/Gaussian blur and unsharp masking across sample depths      | core                    |
| `filter-neighborhood` | Convolution, rank, extrema, and mode filters across sample depths | core              |
| `palette`       | Palettes, quantization, dithering                               | core                    |
| `stat`          | Histogram reductions and moments                                | core                    |
| `codecs`        | Format detection, decoding, encoding                            | core                    |
| `compressor`    | Lossless and lossy compression search                           | core, codecs, ops       |
| `blanket`       | Python registration and save argument adaptation                | All library crates      |
| `gpu`           | GPU descriptors and processing plans                            | None                    |
| `gpu-metal`     | Metal execution and resource caches                             | gpu                     |
| `gpu-cuda`      | CUDA kernels and fatbinary builds                               | None                    |
| `gpu-python`    | GPU bindings and image exchange                                 | core, gpu, GPU backends |

The table lists primary internal dependencies. Codec backend crates are listed
in [Codec build layout](codecs/README.md). Shared image representations belong
in core, and shared pixel modes belong in modes. Pixel layout helpers and LUTs
stay in kernels so core does not depend on resampling algorithms. Conversion
kernels depend on kernels and modes, without depending on core. Core preserves
its existing mode and conversion exports. Algorithm-specific SIMD stays beside
the algorithm or in its kernel crate. Statistics operates on histograms and
shares core helpers.

Filtering is split by algorithm: blur and neighborhood implementations compile
independently on top of core. Each owns its supported sample depths, SIMD, and
unit tests. The filter crate registers their Python functions and preserves its
existing Rust entry points while implementing color LUTs and band merging.

The extension's `src/encode.rs` handles Python save dispatch and mode adaptation.
Codec implementations and compression search remain independently owned.
Codec build features are forwarded through the extension.

Run commands from the repository root. For example:

```sh
cargo test -p blanket-palette
cargo check --workspace
```

Public Python behavior is tested in `tests/`. Rebuild the extension with
`maturin develop -G dev --uv` before running Python tests after Rust changes.
