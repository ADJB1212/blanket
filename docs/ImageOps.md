# ImageOps

[Back to the README](../README.md) · [All modules](index.md)

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

Note that `ImageOps.crop(image, border=2)` removes pixels from each side by
a border width—its argument is not a rectangle. Use `image.crop()` for
rectangle-based cropping.
