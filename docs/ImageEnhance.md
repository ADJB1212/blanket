# ImageEnhance

[Back to the README](https://github.com/ADJB1212/blanket/blob/main/README.md) · [All modules](index.md)

`from blanket import ImageEnhance` provides Pillow-compatible `Color`,
`Contrast`, `Brightness`, and `Sharpness` enhancer classes for all three
Blanket modes. Enhancement factors are unrestricted, and RGBA enhancement
preserves alpha for finite factors. Degenerate-image construction and blending
run in native parallel kernels.

```python
from blanket import ImageEnhance

enhanced = ImageEnhance.Contrast(image).enhance(1.5)
```
