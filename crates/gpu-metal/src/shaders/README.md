# Metal shaders

`src/lib.rs` concatenates these files in dependency order. Runtime compilation
uses the assembled strings rather than filesystem includes.

| File               | Responsibility                                                |
| ------------------ | ------------------------------------------------------------- |
| `common.metal`     | Operation IDs, precision constants, and sample access         |
| `color.metal`      | Luminance, sRGB transfer, color LUTs, and mode conversion     |
| `geometry.metal`   | Scalar crop, transpose, wrap, and resampling                  |
| `filters.metal`    | Scalar filters and packed byte-filter helpers                 |
| `reductions.metal` | Scalar histogram, moment, and luminance fallbacks             |
| `arithmetic.metal` | Sample arithmetic, alpha math, and enhancement                |
| `kernels.metal`    | Shared scalar dispatcher and image entry points               |
| `packed.metal`     | Native packed-byte dispatch                                   |
| `pixels.metal`     | Native packed geometry, float filters/warps, alpha, transpose |
| `blur.metal`       | Native horizontal scans and vertical sliding box blurs        |
| `statistics.metal` | Native parallel statistics and luminance reductions           |
| `paste.metal`      | Native clipped, masked paste                                  |
| `tiles.metal`      | Native packed tile copies                                     |

`SHADERS` supplies the scalar kernels to native Metal and PyTorch MPS.
`NATIVE_SHADERS` adds optimized entry points after those shared helpers.
`BLANKET_METAL_SPECIALIZE` enables the native operation and channel constants.

The common buffer slots are source (0), destination (1), other image (2),
parameters (3), and values (4). Paste also binds its mask at slot 5. Image
parameters are `[source_width, source_height, channels, output_width,
output_height, operation, kernel_or_x, axis_or_y]`; tile copies use their
separate eleven-word rectangle layout.
