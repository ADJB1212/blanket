# Image

[Back to the README](../README.md) · [All modules](index.md)

Blanket is a deliberately focused, Rust-backed image package with a familiar
Pillow-shaped Python API. Its supported surface is loading, saving, and
converting and processing 8-bit `L`, `RGB`, and `RGBA` images in PNG, JPEG,
JPEG XL, TIFF, WebP, AVIF, and HEIC/HEIF files, plus developing DNG files.
High-bit-depth images retain their samples when opened and saved.

Blanket is not the `PIL` package and does not depend on Pillow at runtime.
Optional interoperability helpers can create Pillow images when Pillow is
installed.

```python
from blanket import Image

with Image.open("input.png") as image:
    image.convert("RGB").save("output.jpg", quality=85)
```

## Supported API

`Image.open()` accepts paths and binary streams. `Image.fromarray()` accepts
8-bit array-interface objects with grayscale, RGB, or RGBA shapes, including
strided NumPy arrays. `Image.Image.save()` infers PNG, JPEG, or JPEG XL from a
path extension, or accepts an explicit format for streams. TIFF (`.tif`/`.tiff`)
WebP (`.webp`), and AVIF (`.avif`) also support opening and saving. HEIC/HEIF (`.heic`/`.heif`)
supports opening and saving HEVC images; both format names report `image.format == "HEIF"`.
HEIF opens the primary image and applies container rotations and crops.
DNG supports opening:
DNG raw data develops to 8-bit RGB. DNG rendering applies raw development,
so its appearance can differ from the camera's embedded preview.
Only the first image/frame is opened for TIFF and WebP; animation and multipage
editing are not supported. New formats do not currently retain EXIF/XMP metadata.
`Image.Image.convert()`
supports all conversions among `L`, `RGB`, and `RGBA`.

`image.split()` returns independent `L` images for each channel.
`image.reduce(factor, box=None)` averages integer pixel blocks; `factor` can
be an integer or a `(horizontal, vertical)` pair. Output dimensions round up,
and RGBA reduction accounts for alpha.
`image.entropy(mask=None, extrema=None)` returns Shannon entropy over channel
histogram bins, optionally selecting pixels with a nonzero `L` mask. `extrema`
is accepted for Pillow compatibility and ignored for 8-bit images.

`image.resize((width, height), resample=None, box=None, reducing_gap=None)`
returns a new image and copies its metadata. The default filter is BICUBIC;
all six `Image.Resampling` filters are supported. Use `box` to select a source
rectangle or `reducing_gap` (at least `1.0`) to enable integer reduction before
resampling.

`image.crop((left, upper, right, lower))` returns a rectangular copy, padding
areas outside the image with zero-valued pixels. Omitting the box copies the
whole image.

`image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)` returns a flipped or rotated
copy. All seven Pillow transpose operations are supported.

`image.transform(size, method, data=None, resample=Image.Resampling.NEAREST,
fill=1, fillcolor=None)` supports `Image.Transform.AFFINE`, `EXTENT`,
`PERSPECTIVE`, `QUAD`, and `MESH`, with NEAREST, BILINEAR, or BICUBIC sampling.
It also accepts objects with `getdata()` and `Image.ImageTransformHandler`
subclasses. Both methods copy image metadata.

`image.rotate(angle, resample=Image.Resampling.NEAREST, expand=False,
center=None, translate=None, fillcolor=None)` returns a counterclockwise rotated
copy and preserves metadata. Rotation supports NEAREST, BILINEAR, and BICUBIC,
with optional canvas expansion, rotation center, translation, and fill color.
Expansion assumes the default center and no translation.

## Image assembly

Image assembly APIs include `Image.new(mode, size, color=0)`,
`image.paste(source_or_color, box=None, mask=None)`,
`Image.alpha_composite(background, overlay)`,
`image.alpha_composite(overlay, dest=(0, 0), source=(0, 0))`,
`image.putalpha(alpha)`, and `Image.merge(mode, bands)`.
These operate on 8-bit images. Paste accepts L or RGBA masks and clips at
image boundaries; alpha compositing requires RGBA images. `putalpha` accepts
an L image or integer, promoting RGB images to RGBA. Grayscale and palette
`putalpha` are unsupported because Blanket does not implement LA or PA modes.
Merge accepts L bands and produces L, RGB, or RGBA images.

## Pixel access

`image.getpixel((x, y))` reads a pixel directly from native storage and accepts
negative coordinates. `image.reduce(factor, box=None)` averages integer blocks;
the factor may be an integer or an `(x, y)` pair.

## Quantization

`image.quantize(colors=256, method=None, kmeans=0, palette=None,
dither=Image.Dither.FLOYDSTEINBERG)` returns an indexed `P` image. Native Rust
implementations provide `Image.Quantize.MEDIANCUT`, `MAXCOVERAGE`, and
`FASTOCTREE`; RGBA defaults to FASTOCTREE. Generated colors and palette ordering
may differ from Pillow. LIBIMAGEQUANT is unavailable and raises `RuntimeError`.
Supplied palettes support nearest-color mapping and Floyd–Steinberg dithering.
Palette results support `getpalette`, `putpalette`, conversion, copying,
cropping, nearest-neighbor resizing, PNG saving (including palette alpha),
and `to_pillow()`. `reduce` rejects indexed images unless it is a full-size copy.
General pixel processing remains focused on `L`, `RGB`, and `RGBA`; convert
indexed images to one of these modes before filtering or color operations.

Quantization uses NEON on AArch64 and runtime-detected SSE2 on x86 for palette
distance searches, with a scalar fallback. Large images use parallel histogram
construction and independent pixel mapping. Floyd–Steinberg error diffusion
stays sequential to preserve its results, using cached SIMD palette searches.
Grayscale 2×2 reduction uses NEON/SSE2 alongside the existing parallel reduction
paths. Single-pixel access avoids extra Python work rather than starting workers.

## PDF output

Save an 8-bit image as a single-page PDF with `image.save("output.pdf")`, or
`image.save(stream, format="PDF")`. PDF output preserves grayscale, RGB pixels,
and RGBA transparency using uncompressed, lossless image streams. Page size is
one point per pixel (72 DPI). PDF reading, multipage output, metadata, and PDF
save options are not supported; convert high-bit-depth images to 8-bit first.

## Encoder options

- PNG: `compress_level=0..9` (default `6`)
- TIFF: uncompressed output, no save options
- AVIF (`.avif`): `quality=1..100` (default `90`), `effort=1..10` (default `7`).
  Higher effort encodes more slowly. Uses the `image` crate for reading and writing;
  decoded images use RGBA, and grayscale is stored as RGB. Saving supports 8-bit images and does not offer a
  lossless option. Opens the primary image; animation is not supported.
- WebP: `quality=1..100` (default `90`), `lossless=True|False` (default `False`)
  Grayscale images are stored as RGB, as required by WebP.
- JPEG: `quality=1..100` (default `75`)
- JPEG XL: `quality=1..100` (default `90`), `lossless=True|False`, and
  `effort=1..10` (default `7`)
- HEIF: `quality=1..100` (default `90`), `lossless=True|False` (default `False`).
  The x265 encoder uses its `medium` preset for faster saves; file sizes and
  lossy pixels can differ from its `slow` default. Grayscale is stored as
  monochrome HEVC and opens as RGB in Blanket. Lossless selects lossless
  HEVC compression; RGB/YUV conversion can still change color sample values.

JPEG does not accept `RGBA`; call `image.convert("RGB")` before saving.

## EXIF and metadata

Opening PNG and JPEG retains EXIF/XMP in `image.info` for `exif_transpose`;
uncompressed EXIF/XMP boxes in JPEG XL containers are also read. Other metadata
and compressed JPEG XL metadata boxes are not retained. Transposition removes
the orientation while preserving other EXIF entries. Saving still writes pixel
data only and does not preserve metadata.

## High-bit-depth images

`image.bit_depth` reports 8, 10, 12, or 16 significant bits per channel.
Modes remain `L`, `RGB`, and `RGBA`; 10-bit samples range from 0 to 1023.
`getpixel()` returns these full values, and `tobytes()` uses tightly packed,
little-endian unsigned 16-bit samples for depths above 8.

```python
import numpy as np
from blanket import Image

samples = np.full((64, 64, 3), 713, dtype=np.uint16)
image = Image.fromarray(samples, bit_depth=10)
image.save("photo.heic", quality=95)
image.save("exact.png")
assert Image.open("exact.png").getpixel((0, 0)) == (713, 713, 713)
image.convert("RGB", bit_depth=8).save("preview.jpg")
```

`fromarray()` accepts uint16 arrays (including strided and big-endian arrays),
defaulting to 16 bits; specify `bit_depth=10` for unscaled 10-bit samples.
`frombytes(..., bit_depth=10)` accepts little-endian uint16 data.
Values outside the selected range are rejected.

HEIF retains the source 8-, 10-, or 12-bit depth. PNG stores high-bit-depth
pixels in 16-bit channels with an `sBIT` chunk so Blanket can recover the
original depth and exact samples. TIFF and JPEG XL use normalized 16-bit
channels; convert back to the desired depth to recover the original range.
High-bit-depth PNG, TIFF, and integer JPEG XL inputs are never reduced to 8 bits.

Copy, conversion, pixel access, channel splitting, crop, transpose, and all six
resize filters preserve high-bit-depth samples. Other processing operations,
JPEG/WebP saving, and `to_pillow()` require an explicit conversion to 8 bits
and raise an error instead of silently discarding precision. DNG development
continues to produce 8-bit RGB. Retaining sample precision does not implement
HDR tone mapping or retain HEIF HDR/ICC metadata.
