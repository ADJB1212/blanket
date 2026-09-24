# Blanket

**Image processing in Python. Pixel work in Rust.**

[![CI](https://github.com/ADJB1212/blanket/actions/workflows/CI.yml/badge.svg)](https://github.com/ADJB1212/blanket/actions/workflows/CI.yml)
![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Blanket is a Rust-backed image library with a familiar, Pillow-shaped Python
API. Open, transform, and save images using native pixel kernels, with no
Pillow or other Python packages required at runtime.

```python
from blanket import Image, ImageOps

with Image.open("photo.jpg") as image:
    upright = ImageOps.exif_transpose(image)
    thumbnail = ImageOps.fit(upright, (256, 256), method=Image.Resampling.LANCZOS)
    thumbnail.save("thumbnail.webp", quality=85)
```

[Installation](#installation) · [Examples](#examples) ·
[Formats](#supported-formats) · [API notes](https://adjb1212.github.io/blanket-docs/) ·
[Contributing](#contributing)

## Why Blanket?

- **Familiar image APIs.** Work with `Image`, `ImageOps`, `ImageChops`, `ImageEnhance`,
  `ImageFilter`, `ImagePalette`, and `ImageStat`.
- **Native processing.** Rust kernels use SIMD and parallel execution for
  supported operations, with small-image fast paths.
- **Modern formats.** Read and write JPEG XL, WebP, AVIF, and HEIC/HEIF alongside
  PNG, JPEG, TIFF, BMP, GIF, and ICO. Export single-page PDFs.
- **Explicit precision.** Keep 10-, 12-, and 16-bit samples in supported
  workflows, and choose when to convert to 8 bits.
- **Python interoperability.** Import array data, work with binary streams,
  and convert explicitly to Pillow when you need it.

Blanket is currently **alpha software**. It implements a focused subset of
Pillow's API, centered on `L`, `RGB`, and `RGBA` images, with limited indexed
`P` support. It is not a drop-in replacement for `PIL`.

See [ImageChops](https://adjb1212.github.io/blanket-docs/ImageChops/) for
arithmetic and blend modes, and
[ImageStat](https://adjb1212.github.io/blanket-docs/ImageStat/) for per-band
statistics.

## Installation

Blanket requires **Python 3.12+**. Install from this repository after setting
up the native build prerequisites below:

```sh
pip install pyblanket
```

See the [development notes](https://adjb1212.github.io/blanket-docs/development/) for information on building from source.

## Examples

### Convert between formats

```python
from blanket import Image

with Image.open("input.png") as image:
    image.convert("RGB").save("output.jpg", quality=85)
    image.save("output.jxl", lossless=True)
```

File extensions select the output format. For binary streams, pass it explicitly:

```python
from io import BytesIO
from blanket import Image

with Image.open("input.png") as image:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    png_bytes = buffer.getvalue()
```

### Optimize a save without additional loss

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

The compressor keeps the smallest encoding that preserves the normal save's
decoded pixels. It supports PNG, JPEG, JPEG XL, and HEIF/HEIC; higher effort
tries more settings and takes longer. JPEG optimization preserves DCT
coefficients. This optimizes the requested save, not the original source file:
lossy save settings still introduce their normal loss. See the
[Compressor module](https://adjb1212.github.io/blanket-docs/Compressor/) for the API, format-specific behavior,
and compression benchmarks.

For smaller output with bounded additional pixel error, pass
`LossyImageCompressor(max_rmse=2.0, effort=7)` from `blanket.Compressor`.
It searches PNG color precision or lower JPEG, WebP, JPEG XL, AVIF, and
HEIF/HEIC quality settings, preserving alpha and checking decoded RGB error
against the normal save. The result never exceeds the normal save's size.

### Resize, enhance, and filter

```python
from blanket import Image, ImageEnhance, ImageFilter, ImageOps

with Image.open("photo.jpg") as image:
    resized = ImageOps.contain(image, (1200, 800))
    enhanced = ImageEnhance.Contrast(resized).enhance(1.2)
    sharpened = enhanced.filter(ImageFilter.UnsharpMask(radius=2, percent=150))
    sharpened.save("edited.png")
```

`resize()` supports all six `Image.Resampling` filters. `ImageOps` includes
cropping, padding, flips, color adjustments, and EXIF orientation correction.
`ImageFilter` provides blurs, convolution kernels, rank filters, and 3D color LUTs.

### Build an image with transparency

```python
from blanket import Image

background = Image.new("RGBA", (640, 480), "white")
overlay = Image.new("RGBA", (160, 160), (30, 100, 220, 128))
background.alpha_composite(overlay, dest=(40, 40))
background.save("composite.png")
```

Use `paste()` for masked placement, `putalpha()` to set transparency, and
`split()` / `Image.merge()` to work with individual channels.

### Work with NumPy and high-bit-depth images

NumPy is optional. `fromarray()` accepts array-interface objects, including
strided arrays:

```python
import numpy as np
from blanket import Image

samples = np.full((64, 64, 3), 713, dtype=np.uint16)
image = Image.fromarray(samples, bit_depth=10)
image.save("exact.png")
image.convert("RGB", bit_depth=8).save("preview.jpg")
```

`image.bit_depth` reports sample precision. Copying, conversion, pixel access,
splitting, cropping, transposition, and resizing preserve high-bit-depth
samples. Other processing operations require conversion to 8 bits.
See [high-bit-depth images](https://adjb1212.github.io/blanket-docs/high-bit-depth/) for storage
and format-specific behavior.

## Supported formats

| Format      | Read | Write | Notes                                                                     |
| ----------- | :--: | :---: | ------------------------------------------------------------------------- |
| PNG         | Yes  |  Yes  | Lossless; high-bit-depth and indexed output                               |
| JPEG        | Yes  |  Yes  | 8-bit output; convert RGBA to RGB before saving                           |
| JPEG XL     | Yes  |  Yes  | Lossy or lossless; high-bit-depth support                                 |
| TIFF        | Yes  |  Yes  | First image only; uncompressed output                                     |
| WebP        | Yes  |  Yes  | Lossy or lossless; first frame only                                       |
| AVIF        | Yes  |  Yes  | Primary image only; 8-bit, lossy output                                   |
| HEIC / HEIF | Yes  |  Yes  | HEVC; retains 8-, 10-, or 12-bit source depth                             |
| PDF         |  —   |  Yes  | Single page; lossless 8-bit output with transparency                      |
| BMP         | Yes  |  Yes  | 8-bit output; uncompressed                                                |
| GIF         | Yes  |  Yes  | First frame only; single-frame output, 256 colors and binary transparency |
| ICO         | Yes  |  Yes  | Largest icon on read; one PNG icon on write, 1–256 pixels per dimension   |

Encoder settings and their defaults are listed in the
[format guide](https://adjb1212.github.io/blanket-docs/formats/). HEIF lossless compression can still
change RGB values during RGB/YUV conversion.

## Compatibility and scope

- **Image modes:** general processing uses `L`, `RGB`, and `RGBA`. Quantization
  produces indexed `P` images with a smaller supported operation set.
- **Metadata:** PNG, JPEG, and uncompressed JPEG XL metadata boxes supply
  EXIF/XMP for orientation handling. Saving writes pixels only; metadata is
  not preserved.
- **Multiple frames:** animation and multipage editing are not supported.
- **High bit depth:** sample retention is supported for selected operations
  and formats; HDR tone mapping and HEIF HDR/ICC metadata retention are not.
- **Pillow interop:** `image.to_pillow()` creates a Pillow image when Pillow is
  installed. High-bit-depth images must first be converted to 8 bits.

See the [usage and API notes](https://adjb1212.github.io/blanket-docs/) for operation-specific restrictions,
palette support, and interoperability examples.

## Performance

The benchmark suite compares Blanket and Pillow using equal in-memory inputs
and eager decoding. After completing the development setup below, run it with
a release build:

```sh
maturin develop -r --extras test --uv
uv run --no-sync scripts/benchmark.py --sizes web
```

Use `--all` for the full suite or `--sections Resize ImageOps` to select
operations. Results depend on image size, operation, codec settings, and
hardware. The [benchmark guide](https://adjb1212.github.io/blanket-docs/benchmarking/)
explains baseline comparisons and reproducible runs.

## Contributing

With the [build prerequisites](#installation) installed, set up a development
environment and run the checks:

```sh
uv sync --no-install-project --extra test
uv tool install maturin
maturin develop --extras test --uv
cargo fmt --check
cargo clippy -- -D warnings
cargo test
uv run --no-sync pytest -q
uv run --no-sync scripts/verify_behavior.py
```

The native core lives in [`crates/`](crates/), the Python API in
[`python/blanket/`](python/blanket/), and compatibility tests in [`tests/`](tests/).
Include a focused regression test with bug fixes and new features.
[open an issue](https://github.com/ADJB1212/blanket/issues) to report a problem.

## License

Blanket is licensed under the [MIT License](LICENSE). See
[third-party licenses](THIRD_PARTY_LICENSES.md) for native dependency notices.
