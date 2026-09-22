# ImagePalette

[Back to the README](https://github.com/ADJB1212/blanket/blob/main/README.md) · [All modules](index.md)

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

Run palette object benchmarks
with `uv run --no-sync python scripts/benchmark.py --palette-only -i 100 -w 10`.
