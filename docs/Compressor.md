# Compressor

`blanket.Compressor` provides optional lossless and quality-bounded lossy
optimization for individual `Image.save()` calls. Both search alternative
encodings and keep the smallest qualifying result, including the normal save.

```python
from blanket import Image
from blanket.Compressor import LosslessImageCompressor

compressor = LosslessImageCompressor(effort=7)

with Image.open("input.png") as image:
    image.save("optimized.png", compressor=compressor)
    image.convert("RGB").save("optimized.jpg", quality=90, compressor=compressor)
    image.save("optimized.jxl", lossless=True, compressor=compressor)
    image.save("optimized.heic", lossless=True, compressor=compressor)
```

See [Image](Image.md) for supported image modes, bit depths, and save options.

## API

### `LosslessImageCompressor(*, effort=7)`

Construct a reusable compressor configuration. `effort` must be an integer
from **1 through 10**, inclusive. Non-integers, including booleans, raise
`TypeError`; integers outside that range raise `ValueError`.

The read-only `.effort` property returns the configured value. The compressor
does not expose a separate `compress()` method: pass it through the
`compressor` keyword to `Image.save()`.

```python
from io import BytesIO
from blanket import Image
from blanket.Compressor import LosslessImageCompressor

image = Image.new("RGB", (128, 128), (12, 31, 79))
output = BytesIO()
image.save(output, "PNG", compress_level=6, compressor=LosslessImageCompressor(effort=3))
encoded = output.getvalue()
```

An explicit format is needed for buffers. Files can use their filename
extension to select the format. `compressor=None` uses normal saving; other
objects passed as `compressor` raise `TypeError`.

The same compressor can be used for multiple images and formats. Optimization
does not change the input image's pixels, mode, bit depth, or metadata.

### `LossyImageCompressor(*, max_rmse=2.0, effort=7)`

Search for smaller encodings while bounding **additional** sample error
relative to the normal save with the same options. The normal save's existing
loss is not included in this bound. This measures pixel differences, not
perceptual similarity or error relative to an opened image's original file.

```python
from blanket.Compressor import LossyImageCompressor

compressor = LossyImageCompressor(max_rmse=2.0, effort=7)
image.save("smaller.png", compressor=compressor)
image.convert("RGB").save("smaller.jpg", quality=90, compressor=compressor)
```

`max_rmse` must be a finite number from 0 through 255. Booleans and nonnumeric
values raise `TypeError`; out-of-range values raise `ValueError`. `effort`
accepts integers from 1 through 10 with the same validation as the lossless
compressor. Both properties are read-only.

For each candidate, decoded samples are normalized to the 0–255 scale,
grayscale is expanded to RGB, and the root mean square error is computed over
all RGB samples. Hidden RGB beneath transparent pixels is included. Alpha
must match exactly; it is excluded from the error average. Each candidate is
compared to the fixed normal-save reference, never to the previous winner.
Dimensions are preserved. High-bit-depth codec samples use the same normalized
error scale without first rounding to eight bits.

| Format | Lossy search |
| --- | --- |
| 8-bit non-indexed PNG | Round color samples to progressively coarser steps, then apply lossless PNG optimization; alpha is unchanged |
| JPEG, WebP, JPEG XL, AVIF, HEIF/HEIC | Try lower quality settings, decode smaller candidates, and check the error bound |
| Indexed or high-bit-depth PNG, other formats | Retain the lossless compressor's behavior |

The search starts with lossless optimization. At most `effort` additional
candidates are tried. Higher efforts extend a fixed set of sample steps or
quality reductions; they do not promise a global optimum. PNG sample steps
are 2, 3, 4, 6, 8, 12, 16, 24, 32, and 64. Codec quality reductions are
1, 2, 4, 8, 12, 20, 30, 45, 65, and 99 points, clamped to quality 1.
Codec `save(..., effort=...)` retains its ordinary meaning independently of
the compressor's search effort.

`max_rmse=0` uses lossless optimization only. Explicit `lossless=True` also
disables additional loss for codecs supporting that save option. The smallest
qualifying result wins, so output is never larger than the normal save.
Results may remain identical when no smaller candidate meets the bound.
As with the lossless compressor, the source is unchanged, metadata follows
ordinary save behavior, and the configuration is reusable across saves.

## What “lossless” means

The reference is a **normal save with the same save options**. That encoding
always remains a candidate, so optimization cannot increase its encoded size.
Only strictly smaller candidates replace the current result; ties retain the
earlier encoding.

This guarantee applies to the requested save, not to the original file:

- JPEG still applies the requested quality and subsampling during the normal
  save. Optimization preserves that save's DCT coefficients.
- Lossy JXL and HEIF saves still introduce their normal loss. Optimization adds
  no further sample changes.
- HEIF's RGB/YUV conversion can change color samples even with
  `lossless=True`; candidates must match the normal save's decoded samples.
- Saving an opened image encodes its current pixels. The compressor does not
  reuse or transcode the original file's bitstream, and does not guarantee a
  smaller file than that original.

Metadata follows ordinary save behavior. The compressor does not add metadata
retention; Blanket currently saves pixels without carrying through source
EXIF, XMP, or ICC metadata.

## Format-specific behavior

| Format | Optimization | Input depths |
| --- | --- | --- |
| PNG | Filter and DEFLATE searches, exact channel reductions, packed grayscale, transparency keys, and exact palettes | 8-bit `L`, `RGB`, and `RGBA` |
| JPEG | Optimized Huffman tables; progressive scans at effort 3 and above | 8-bit `L` and `RGB` |
| JPEG XL | Encoder effort search, with exact decoded-sample verification | 8, 10, 12, and 16 bits |
| HEIF / HEIC | x265 preset search, with exact decoded-sample verification | 8, 10, and 12 bits |

High-bit-depth PNG, indexed PNG, and other output formats use their normal
encoders. Passing a compressor does not expand a format's supported modes or
bit depths; for example, JPEG still requires conversion from RGBA to RGB.

### PNG

PNG optimization preserves every sample, including RGB values beneath fully
transparent pixels. It can remove opaque alpha channels, reduce gray RGB
images to grayscale, and store gray RGBA images as grayscale-plus-alpha.
Reopening an optimized PNG may therefore report a different mode; convert it
to the original mode to compare pixels.

Exact grayscale packing uses 1-bit samples for values 0/255, 2-bit samples for
multiples of 85, and 4-bit samples for multiples of 17. Wider representations
remain candidates because filtering can make them smaller.

An exact palette can represent up to 256 distinct colors, including alpha.
The search tries packed indices and, at higher effort, alternative palette
orders. Transparent entries can be moved earlier to shorten the transparency
chunk. Efforts 8–10 also try wider index depths; this adds encoding work but
can improve compression of images with long runs of palette colors. No colors
are quantized or discarded.

A single transparency key is considered only when alpha is binary, all
transparent pixels have the same RGB color, and no opaque pixel has that
color. Nonzero keys in 2-/4-bit grayscale are avoided for Pillow compatibility.

Higher efforts try more row filters, including minimum-entropy filtering.
Efforts 9–10 also search additional DEFLATE levels.

### JPEG

All efforts try optimized Huffman coding. Effort 3 and above also tries
progressive scans. These transformations preserve DCT coefficients,
quantization tables, subsampling, dimensions, and partial edge blocks.
Arithmetic coding is not used.

### JPEG XL and HEIF/HEIC

JXL searches encoder efforts 1 through the compressor's effort, skipping the
setting already used by the normal save. The codec's `save(..., effort=...)`
option controls that normal save; the compressor's `.effort` controls the
additional search. The normal save remains eligible even when its codec
effort exceeds the compressor's effort.

HEIF searches x265 presets from `ultrafast` through the selected effort:
`medium` is 6, `slow` is 7, and `placebo` is 10. The normal save uses `medium`,
which is not repeated. Other HEVC plugins retain their own preset defaults.

Each smaller candidate is decoded and checked against the normal save's
dimensions, mode, bit depth, and every sample, including alpha and hidden
colors. A candidate with any difference is discarded. For lossy saves, the
search also tries lossless encoding of the normal save's decoded pixels;
those candidates must pass the same checks.

The implementation defers verification decoding until needed. JXL reuses a
thread pool sized for each frame, while HEIF reuses prepared pixel planes
across preset trials with fresh encoders and containers.

## Choosing effort and measuring results

Lower effort performs fewer trials; higher effort spends more time searching.
The best tradeoff depends on image content and format. A higher setting is
not a promise of further size reduction, and JPEG uses the same two
candidates at efforts 3–10. Benchmark representative images before choosing
settings for latency-sensitive saves.

The [compression benchmark](https://github.com/ADJB1212/blanket/blob/main/scripts/benchmark_compression.py) measures size,
save time, and decoded-pixel error using images from `test_images/`.
It benchmarks both compressors by default; `--compressor lossless` or
`--compressor lossy` selects one. `--max-rmse` sets the lossy error limit
(default 2.0).
Run it from the repository after the development setup. Use a release build
for meaningful timing measurements:

```sh
maturin develop -r --extras test --uv
uv run --no-sync scripts/benchmark_compression.py
uv run --no-sync scripts/benchmark_compression.py --efforts 1 7 8 10 --repeats 3 --json compression.json
uv run --no-sync scripts/benchmark_compression.py --formats JPEG JXL HEIF --compressor lossless
uv run --no-sync scripts/benchmark_compression.py --compressor lossy --max-rmse 4
```

The report includes baseline and optimized byte sizes, percentage saved,
median save times, and the optimized/baseline time ratio. A ratio above 1 means
optimization took longer. JSON output also includes individual timing samples
and the save options used. Each row identifies the compressor, measured RGB
RMSE, alpha preservation, and validation outcome. JSON comparisons distinguish
compressor types and lossy error limits; older JSON rows are treated as lossless.

`--warmups` controls untimed saves. Timings exclude image loading, decoding,
and verification. JPEG uses quality 90. JXL/HEIF use lossless saves for the
lossless compressor and quality 90 for the lossy compressor, so lossy searches
are enabled. Each result is checked against its own normal save: exact RGBA
for lossless, or bounded RGB RMSE plus exact alpha for lossy. High-bit-depth
samples retain their precision during verification. Normal PNG encoding is
also checked against the input.

A failed quality check, size increase, or codec error produces a nonzero exit
status. Benchmark results are informational and do not represent all images
or hardware.
