# Native crates

The `blanket` extension assembles the Python API from focused library crates.
Each operation crate owns its implementations, SIMD helpers, and unit tests.

| Crate           | Responsibility                                                  | Internal dependencies   |
| --------------- | --------------------------------------------------------------- | ----------------------- |
| `core`          | Image storage, sample modes, pixel conversion, parallel helpers | kernels                 |
| `ops`           | ImageOps transforms, histogram operations, LUTs                 | core, kernels, math     |
| `kernels`       | Shared pixel kernels                                            | None                    |
| `resample`      | Resize, transforms, resampling                                  | core, kernels           |
| `math`          | ImageMath operations                                            | core                    |
| `draw`          | Geometry, masks, flood fill                                     | core                    |
| `font`          | Font loading, shaping, rasterization                            | kernels                 |
| `cms`           | LittleCMS profiles and color transforms                         | core                    |
| `metadata`      | Encoded-container metadata operations                           | codec-common            |
| `composite`     | Compositing and channel arithmetic                              | core                    |
| `enhance`       | Brightness, contrast, color, sharpness                          | core                    |
| `filter`        | Convolution, rank, blur filters                                 | core                    |
| `palette`       | Palettes, quantization, dithering                               | core                    |
| `stat`          | Histogram reductions and moments                                | core                    |
| `codecs`        | Format detection, decoding, encoding                            | core                    |
| `compressor`    | Lossless and lossy compression search                           | core, codecs, ops       |
| `blanket`       | Python registration and save argument adaptation                | All library crates      |
| `gpu`           | GPU descriptors and processing plans                            | None                    |
| `gpu-metal`     | Metal execution and resource caches                             | gpu                     |
| `gpu-cuda`      | CUDA kernels and fatbinary builds                               | None                    |
| `gpu-python`    | GPU bindings and image exchange                                 | core, gpu, GPU backends |
| `bench-support` | Shared benchmark fixtures                                       | core                    |

The table lists primary internal dependencies. Codec backend crates are listed
in [Codec build layout](codecs/README.md). Shared image representations and
pixel layout helpers belong in core. Algorithm-specific SIMD stays beside the
algorithm. Statistics operates on histograms and shares core helpers.

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
