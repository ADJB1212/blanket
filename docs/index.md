<div class="blanket-hero">
  <div class="blanket-hero__content">
    <span class="blanket-eyebrow">THE IMAGE LIBRARY FOR PYTHON</span>
    <h1>Make every pixel count.</h1>
    <p>Familiar Python image workflows, powered by Rust. Open, transform, and save images with a focused Pillow-shaped API.</p>
    <div class="blanket-hero__actions">
      <a class="blanket-button blanket-button--primary" href="usage/">Get started <span aria-hidden="true">&#8594;</span></a>
      <a class="blanket-button blanket-button--secondary" href="Image/">Explore the API</a>
    </div>
  </div>
  <div class="blanket-hero__art" aria-hidden="true">
    <div class="blanket-art__frame">
      <img src="assets/blanket_icon.png" alt="" width="1254" height="1254">
    </div>
    <span class="blanket-art__label">L &nbsp; / &nbsp; RGB &nbsp; / &nbsp; RGBA</span>
  </div>
</div>

<div class="blanket-section-intro">
  <span class="blanket-eyebrow">BUILT FOR REAL WORK</span>
  <h2>Everything you need to move from source to output.</h2>
</div>

<div class="blanket-card-grid">
  <a class="blanket-card" href="usage/">
    <span class="blanket-card__number">01 / START HERE</span>
    <h3>Work the way you know</h3>
    <p>Open images, apply transforms, and save the result with an API that feels familiar.</p>
    <span class="blanket-card__link">Read the usage guide &#8594;</span>
  </a>
  <a class="blanket-card" href="Image/">
    <span class="blanket-card__number">02 / FORMATS &amp; PRECISION</span>
    <h3>More room for your images</h3>
    <p>Work with JPEG XL, AVIF, WebP, HEIF, and established formats. Preserve higher bit depths where supported.</p>
    <span class="blanket-card__link">Explore Image &#8594;</span>
  </a>
  <a class="blanket-card" href="ImageOps/">
    <span class="blanket-card__number">03 / PIXEL TOOLS</span>
    <h3>Make the next edit</h3>
    <p>Resize, crop, colorize, enhance, and filter with native kernels behind a Python interface.</p>
    <span class="blanket-card__link">Browse operations &#8594;</span>
  </a>
</div>

## A small example, start to finish

```python
from blanket import Image, ImageOps

with Image.open("photo.jpg") as image:
    upright = ImageOps.exif_transpose(image)
    thumbnail = ImageOps.fit(upright, (256, 256), method=Image.Resampling.LANCZOS)
    thumbnail.save("thumbnail.webp", quality=85)
```

Blanket is currently alpha software. It supports a focused subset of Pillow's API for `L`, `RGB`, and `RGBA` images, with limited indexed `P` support. See [Pillow interoperability](pillow-interop.md) for the supported boundaries.

## Find your next step

Explore the [Image API](Image.md), [ImageOps](ImageOps.md), [ImageEnhance](ImageEnhance.md), [ImageFilter](ImageFilter.md), [ImagePalette](ImagePalette.md), and [Compressor](Compressor.md). For source builds and contributing, see the [development guide](development.md); for performance work, see [benchmarking](benchmarking.md).
