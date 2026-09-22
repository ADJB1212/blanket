# ImageFilter

[Back to the README](https://github.com/ADJB1212/blanket/blob/main/README.md) · [All modules](index.md)

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
