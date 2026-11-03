# Blanket

**Image processing in Python. Pixel work in Rust.**

[![CI](https://github.com/ADJB1212/blanket/actions/workflows/CI.yml/badge.svg)](https://github.com/ADJB1212/blanket/actions/workflows/CI.yml)
![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)
[![Source license: MIT](https://img.shields.io/badge/source_license-MIT-green)](LICENSE)

Blanket is a Rust-backed image library with a Pillow-shaped Python API.
Open, transform, and save images with native SIMD and parallel processing,
with no runtime Python dependencies for CPU use.

```python
from blanket import Image, ImageOps

with Image.open("photo.jpg") as image:
    upright = ImageOps.exif_transpose(image)
    thumbnail = ImageOps.fit(upright, (256, 256), method=Image.Resampling.LANCZOS)
    thumbnail.save("thumbnail.webp", quality=85)
```

[Documentation](https://adjb1212.github.io/blanket-docs/) ·
[API reference](https://adjb1212.github.io/blanket-docs/Image/) ·
[Roadmap](ROADMAP.md) ·
[Issues](https://github.com/ADJB1212/blanket/issues)

## Installation

Requires **Python 3.12+**:

```sh
pip install pyblanket
```

Release wheels include CPU processing and the platform's GPU backend:
Metal on macOS and CUDA on Linux. CPU use needs no GPU dependencies.
GPU processing requires compatible hardware and runtime support; explicit
GPU requests raise an error when unavailable. See the
[GPU guide](https://adjb1212.github.io/blanket-docs/gpu/) for setup and diagnostics.

## Capabilities

- **Familiar APIs:** resize, crop, draw, filter, enhance, composite, quantize,
  and compute image statistics with Pillow-shaped modules.
- **Sample precision:** selected operations support 10/12/16-bit samples,
  float color, and linear-light resize and compositing.
- **Frames and pages:** read GIF, WebP, APNG, and TIFF with eager or lazy
  access; write animations and multi-page PDFs.
- **Color and metadata:** ICC transforms, HDR tone mapping, editable EXIF,
  XMP parsing, and metadata-only container rewrites.
- **Encoding:** JPEG controls, coefficient transforms without recompression,
  and save-time compression searches.
- **Interoperability:** array imports, binary streams, perceptual hashes,
  an optional Pillow bridge, and explicit Metal/CUDA pipelines.

Blanket is **alpha software**, with operation-specific mode and precision
restrictions. It implements a subset of Pillow's API and is not yet a drop-in
replacement for `PIL`. GPU operations do not guarantee byte-exact Pillow results.
See [getting started](https://adjb1212.github.io/blanket-docs/usage/),
[sample precision](https://adjb1212.github.io/blanket-docs/high-bit-depth/), and
[Pillow interoperability](https://adjb1212.github.io/blanket-docs/pillow-interop/).

## Supported formats

| Support | Formats |
| ------- | ------- |
| Read and write | PNG/APNG, JPEG, JPEG XL, TIFF, WebP, AVIF, HEIC/HEIF, BMP, GIF, ICO, Netpbm/PAM, TGA, QOI, JPEG 2000, PCX, SGI, XBM, XPM, MSP |
| Read only | DDS, EPS, FITS, SPIDER, Pixar, Sun raster |
| Write only | PDF, PALM |

Support varies by mode, depth, and installed codec backend. EPS requires
Ghostscript. TIFF writes single pages; PDF supports multiple pages.
Use `features.blanketinfo()` to inspect an installed build and see the
[format guide](https://adjb1212.github.io/blanket-docs/formats/) for encoder
settings and restrictions.

JPEG saves retain EXIF and ICC profiles; pass `xmp=image.info.get("xmp")`
explicitly to retain XMP. See the
[metadata guide](https://adjb1212.github.io/blanket-docs/metadata/) for retention
rules and editing without recompressing pixels.

## Development

Source builds use the nightly Rust toolchain pinned in `rust-toolchain.toml`
for portable SIMD. Install the
[native prerequisites](https://adjb1212.github.io/blanket-docs/development/), then:

```sh
uv sync --no-install-project --dev
uv run --no-sync maturin develop -G dev --uv
uv run --no-sync pytest tests/test_api.py
uv run --no-sync scripts/verify_behavior.py
```

The native core lives in [`crates/`](crates/), the Python API in
[`python/blanket/`](python/blanket/), and tests in [`tests/`](tests/).
Include a focused regression test for fixes and features. See the
[development guide](https://adjb1212.github.io/blanket-docs/development/) for
build options and checks, and the
[benchmark guide](https://adjb1212.github.io/blanket-docs/benchmarking/) for
performance comparisons. Benchmarks require a release build.

## License

Blanket's original source is licensed under the [MIT License](LICENSE).
Official binary wheels include GPL components and are distributed under
GPL-3.0-or-later with their third-party notices and corresponding source.
See [distribution licenses](THIRD_PARTY_LICENSES.md) for details.
