# Blanket

Blanket is a deliberately focused, Rust-backed image package with a familiar
Pillow-shaped Python API. Its supported surface is loading, saving, and
converting and processing 8-bit `L`, `RGB`, and `RGBA` images in PNG, JPEG,
and JPEG XL files.

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
path extension, or accepts an explicit format for streams. `Image.Image.convert()`
supports all conversions among `L`, `RGB`, and `RGBA`.

Supported encoder options are:

- PNG: `compress_level=0..9` (default `6`)
- JPEG: `quality=1..100` (default `75`)
- JPEG XL: `quality=1..100` (default `90`), `lossless=True|False`, and
  `effort=1..10` (default `7`)

JPEG does not accept `RGBA`; call `image.convert("RGB")` before saving.

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

Opening PNG and JPEG retains EXIF/XMP in `image.info` for `exif_transpose`;
uncompressed EXIF/XMP boxes in JPEG XL containers are also read. Other metadata
and compressed JPEG XL metadata boxes are not retained. Transposition removes
the orientation while preserving other EXIF entries. Saving still writes pixel
data only and does not preserve metadata.

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
C++ compiler before building.

```console
uv sync --extra test
cargo test
uv run pytest
uv run python scripts/verify_behavior.py
uv run python scripts/verify_interop.py
uv run python scripts/benchmark.py
```

The benchmark uses equal in-memory inputs, forces Pillow to decode eagerly,
and reports median duration plus Blanket's speed ratio. JPEG XL results are
reported separately because Pillow has no built-in JPEG XL codec.
