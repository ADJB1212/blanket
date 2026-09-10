# Blanket

Blanket is a deliberately focused, Rust-backed image package with a familiar
Pillow-shaped Python API. Its supported surface is loading, saving, and
converting and processing 8-bit `L`, `RGB`, and `RGBA` images in PNG, JPEG,
JPEG XL, TIFF, WebP, and HEIC/HEIF files, plus developing DNG files.
High-bit-depth images retain their samples when opened and saved.

```python
from blanket import Image

with Image.open("input.png") as image:
    image.convert("RGB").save("output.jpg", quality=85)
```

Blanket is not the `PIL` package and does not depend on Pillow at runtime.
Optional interoperability helpers can create Pillow images when Pillow is
installed.

## Supported API

`Image.open()` accepts paths and binary streams. `Image.fromarray()` accepts
8-bit array-interface objects with grayscale, RGB, or RGBA shapes, including
strided NumPy arrays. `Image.Image.save()` infers PNG, JPEG, or JPEG XL from a
path extension, or accepts an explicit format for streams. TIFF (`.tif`/`.tiff`)
and WebP (`.webp`) also support opening and saving. HEIC/HEIF (`.heic`/`.heif`)
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
`ImageOps.crop(image, border=2)` instead removes two pixels from each side;
its argument is a border width, not a rectangle.

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

Supported encoder options are:

- PNG: `compress_level=0..9` (default `6`)
- TIFF: uncompressed output, no save options
- WebP: `quality=1..100` (default `90`), `lossless=True|False` (default `False`)
  Grayscale images are stored as RGB, as required by WebP.
- JPEG: `quality=1..100` (default `75`)
- JPEG XL: `quality=1..100` (default `90`), `lossless=True|False`, and
  `effort=1..10` (default `7`)
- HEIF: `quality=1..100` (default `90`), `lossless=True|False` (default `False`).
  Grayscale is stored as RGB. Lossless selects lossless HEVC compression;
  RGB/YUV conversion can still change sample values.

JPEG does not accept `RGBA`; call `image.convert("RGB")` before saving.

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

## ImageOps

`from blanket import ImageOps` provides all 18 functions in Pillow's
[`ImageOps` submodule](https://pillow.readthedocs.io/en/stable/reference/ImageOps.html):

- Tone and color: `autocontrast`, `colorize`, `equalize`, `grayscale`, `invert`,
  `posterize`, `solarize`.
- Geometry: `contain`, `cover`, `crop`, `deform`, `expand`, `fit`, `flip`,
  `mirror`, `pad`, `scale`.
- Orientation: `exif_transpose`, including `in_place=True`.

```python
from blanket import Image, ImageOps

with Image.open("input.jpg") as image:
    upright = ImageOps.exif_transpose(image)
    thumbnail = ImageOps.fit(upright, (256, 256), method=Image.Resampling.LANCZOS)
    ImageOps.autocontrast(thumbnail).save("thumbnail.png")
```

Pixel processing runs in Rust, without a Pillow runtime dependency. Geometry
supports all three Blanket modes; histogram and lookup operations accept `L`
and `RGB`, while `colorize` requires `L`. `grayscale` accepts all three modes.
The signatures, defaults, border forms, masks, CSS color arguments, and
`SupportsGetMesh` protocol follow Pillow. Resizing supports all six
`Image.Resampling` filters; mesh deformation supports nearest, bilinear, and
bicubic sampling. Filtered RGBA operations use premultiplied alpha.

## ImageEnhance

`from blanket import ImageEnhance` provides Pillow-compatible `Color`,
`Contrast`, `Brightness`, and `Sharpness` enhancer classes for all three
Blanket modes. Enhancement factors are unrestricted, and RGBA enhancement
preserves alpha for finite factors. Degenerate-image construction and blending
run in native parallel kernels.

```python
from blanket import ImageEnhance

enhanced = ImageEnhance.Contrast(image).enhance(1.5)
```

Opening PNG and JPEG retains EXIF/XMP in `image.info` for `exif_transpose`;
uncompressed EXIF/XMP boxes in JPEG XL containers are also read. Other metadata
and compressed JPEG XL metadata boxes are not retained. Transposition removes
the orientation while preserving other EXIF entries. Saving still writes pixel
data only and does not preserve metadata.

## ImageFilter

`from blanket import ImageFilter` provides Pillow-compatible built-in convolution
filters, custom `Kernel` filters, `RankFilter`, `MedianFilter`, `MinFilter`,
`MaxFilter`, `ModeFilter`, `BoxBlur`, `GaussianBlur`, `UnsharpMask`, and
`Color3DLUT` (including `generate` and `transform`). Filtering supports Blanket's
`L`, `RGB`, and `RGBA` images; color LUTs require RGB or RGBA input. Pixel kernels
run in Rust and release the GIL, with parallel output partitions for large images.
Python handles filter configuration and user callbacks for generating LUT tables.

```python
from blanket import ImageFilter

blurred = image.filter(ImageFilter.GaussianBlur(radius=2))
sharpened = image.filter(ImageFilter.SHARPEN)
```

`Image.filter` accepts filter instances, classes, and custom `Filter` or
`MultibandFilter` subclasses. Convolution preserves the image border; rank filters
extend edge pixels, while mode filters use the available neighborhood. Blurs
accept a scalar radius or separate `(x, y)` radii. Non-finite or excessively large
blur radii are rejected. Run benchmarks with
`uv run --no-sync python scripts/benchmark.py --filter-only --sizes web`.

## ImagePalette

`from blanket import ImagePalette` provides Pillow-compatible palette objects,
including color allocation, copying, raw data, serialization, and the `wedge`,
`negative`, `random`, and `sepia` factories. `load` reads text palettes, GIMP
palettes, and GIMP RGB gradients. LUT generation, color indexing, palette ramps,
sepia generation, and gradient rendering use the native Rust backend. Python
preserves mutable palette storage and handles file parsing; `random` uses Python's
random generator to match Pillow's seeded behavior.

```python
from blanket import ImagePalette

palette = ImagePalette.ImagePalette("RGBA")
red_index = palette.getcolor((255, 0, 0, 128))
mode, data = palette.getdata()
```

`image.getpixel((x, y))` reads a pixel directly from native storage and accepts
negative coordinates. `image.reduce(factor, box=None)` averages integer blocks;
the factor may be an integer or an `(x, y)` pair.

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

Run palette object benchmarks
with `uv run --no-sync python scripts/benchmark.py --palette-only -i 100 -w 10`.

## Pillow interoperability

Install Pillow through the test extra, then convert explicitly:

```python
from blanket import Image as BlanketImage
from PIL import Image as PillowImage

blanket_image = BlanketImage.open("input.png")
pillow_image = blanket_image.to_pillow()
channels = pillow_image.split()
merged = PillowImage.merge("RGB", channels[:3])

blanket_result = BlanketImage.frombytes(merged.mode, merged.size, merged.tobytes())
blanket_result.save("result.jxl", lossless=True)
```

The adapter is explicit because Pillow operations require Pillow's private C
image core. Pillow is not a Blanket runtime dependency.

## Development

JPEG XL encoding statically builds libjxl. Install Rust, CMake, Ninja, and a
C++ compiler before building. HEIF requires system libheif 1.17+ with an HEVC
decoder (libde265) and encoder (x265). On macOS use `brew install libheif`;
on Ubuntu install `libheif-dev`, `libheif-plugin-libde265`, and
`libheif-plugin-x265`. The libheif shared library and codec plugins must also
be available at runtime. Pillow and pillow-heif are test dependencies only.

```console
uv sync --extra test
cargo test
uv run pytest
uv run python scripts/verify_behavior.py
uv run python scripts/benchmark.py
```

The benchmark uses equal in-memory inputs, forces Pillow to decode eagerly,
and summarizes each section using Blanket's speed ratio. Pass `--verbose` to
see timings for every operation, or `--slower-only` to show only operations
where Blanket trails Pillow. JPEG XL results are reported separately because
Pillow has no built-in JPEG XL codec.
