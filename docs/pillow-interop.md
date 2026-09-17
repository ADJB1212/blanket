# Pillow interoperability

[Back to the README](../README.md) · [All modules](index.md)

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
