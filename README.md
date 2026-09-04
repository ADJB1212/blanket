# Blanket

Blanket is a deliberately focused, Rust-backed image package with a familiar
Pillow-shaped Python API. Its supported surface is loading, saving, and
converting 8-bit `L`, `RGB`, and `RGBA` images in PNG, JPEG, and JPEG XL files.

```python
from blanket import Image

with Image.open("input.png") as image:
    image.convert("RGB").save("output.jpg", quality=85)
```

Blanket is not the `PIL` package and does not depend on Pillow at runtime.
Optional interoperability helpers can create Pillow images when Pillow is
installed.
