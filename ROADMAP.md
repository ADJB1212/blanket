# Roadmap

Blanket's path from a focused Pillow-shaped library to a full drop-in
replacement — and beyond.

Each phase builds on the previous one. Items within a phase are roughly ordered
by priority but can ship independently.

---

## Phase 0 — Current State

The foundation is in place: native Rust pixel kernels with SIMD and parallel
execution, a Pillow-shaped Python API across seven modules, and read/write
support for eleven image formats.

### Modules

- [x] `Image` — constructors, conversions, pixel access, transforms, compositing, save
- [x] `ImageOps` — all 18 operations including `exif_transpose`
- [x] `ImageChops` — 18 channel operations (arithmetic, blend modes, utility)
- [x] `ImageEnhance` — Color, Contrast, Brightness, Sharpness
- [x] `ImageFilter` — convolution kernels, rank filters, blurs, unsharp mask, `Color3DLUT`
- [x] `ImageStat` — per-band statistics (extrema, mean, median, stddev, etc.)
- [x] `ImagePalette` — palette creation, loading, and manipulation
- [x] `Compressor` — lossless and lossy save-time optimization (Blanket extra)

### Modes

- [x] `L` (8-bit grayscale)
- [x] `RGB` (8-bit truecolor)
- [x] `RGBA` (8-bit truecolor with alpha)
- [x] `P` (indexed palette, via quantization)
- [x] High bit depth (10, 12, 16-bit) for storage, conversion, crop, transpose, resize

### Formats

- [x] PNG, JPEG, JPEG XL, TIFF, WebP, AVIF, HEIC/HEIF, BMP, GIF, ICO (read/write)
- [x] PDF (write-only)

### Core Capabilities

- [x] Resize with all six `Resampling` filters
- [x] Affine, perspective, quad, extent, and mesh transforms
- [x] Arbitrary-angle rotation
- [x] Alpha compositing, paste with mask, blend
- [x] Quantization (median cut, max coverage, fast octree) with dithering
- [x] EXIF/XMP orientation detection and correction
- [x] `fromarray` / `tobytes` / NumPy interop
- [x] `to_pillow()` explicit bridge

---

## Phase 1 — Mode Parity

Add the missing Pillow image modes so that code written for Pillow can work
unchanged. Each new mode needs storage, conversion to/from existing modes,
pixel access, and compatibility across existing operations.

- [x] `1` — bilevel (1-bit per pixel, packed or unpacked)
- [x] `LA` — grayscale with alpha
- [x] `PA` — indexed palette with alpha
- [ ] `I` — 32-bit signed integer pixels
- [ ] `I;16` / `I;16L` / `I;16B` — 16-bit integer (consolidate with existing high-bit-depth)
- [ ] `F` — 32-bit floating-point pixels
- [ ] `CMYK` — 4-channel print color (no ICC transform required; raw channel storage)
- [ ] `YCbCr` — luma/chroma (used internally by JPEG; expose for user code)
- [ ] `LAB` — CIE L\*a\*b\*
- [ ] `HSV` — hue, saturation, value

### Depends on

- `ImageChops.logical_and`, `logical_or`, `logical_xor` require mode `1`

---

## Phase 2 — ImageDraw

Drawing primitives are the most-requested missing module for real-world Pillow
replacement. This is a substantial feature that warrants its own phase.

- [ ] `ImageDraw.Draw(image, mode=None)` context
- [ ] Geometric primitives: `line`, `rectangle`, `rounded_rectangle`, `ellipse`, `arc`, `chord`, `pieslice`, `polygon`, `regular_polygon`, `point`
- [ ] Fill and outline with color, width control
- [ ] Anti-aliased drawing
- [ ] `textbbox`, `textlength`, `multiline_textbbox`
- [ ] `text`, `multiline_text` (requires `ImageFont`)
- [ ] Flood fill (`floodfill`)

---

## Phase 3 — ImageFont & Text Rendering

Text rendering depends on `ImageDraw` and a font backend. The goal is to
support TrueType/OpenType via a Rust font rasterizer (e.g. `ab_glyph` or
`cosmic-text`) without linking FreeType.

- [ ] `ImageFont.truetype(font, size, ...)` — load `.ttf`/`.otf` files
- [ ] `ImageFont.load_default(size=None)` — built-in fallback font
- [ ] Glyph rasterization and layout (horizontal, basic vertical)
- [ ] Font metrics: `getbbox`, `getlength`, `getmetrics`
- [ ] Bitmap font support (`ImageFont.load`)
- [ ] Unicode and multi-script rendering
- [ ] Text anchors and alignment

---

## Phase 4 — Missing Image Methods & Module Gaps

Fill in the remaining `Image` methods and small utility modules that Pillow
users expect.

### Image Methods

- [ ] `show()` — display via system viewer or configurable backend
- [ ] `effect_spread(distance)` — randomly displace pixels
- [ ] `remap_palette(dest_map, source_palette=None)` — reorder palette indices
- [ ] `verify()` — check file integrity without full decode
- [ ] `draft(mode, size)` — configure decoder for reduced-resolution reads
- [ ] `getim()` / `im` — internal image core access (compatibility shim)
- [ ] `tobitmap(name="image")` — X11 bitmap string (mode `1`)
- [ ] `entropy(mask, extrema)` on high-bit-depth images

### ImageColor Module

- [ ] `getrgb(color)` — public API (currently internal `_color.py`)
- [ ] `getcolor(color, mode)` — convert color string to mode-specific value
- [ ] Full CSS4 color spec compliance

### ImagePath Module

- [ ] `Path` object with `compact`, `getbbox`, `map`, `tolist`, `transform`
- [ ] Integration with `ImageDraw` for path-based drawing

### ImageMath Module

- [ ] `eval(expression, **operands)` — per-pixel math expressions
- [ ] `lambda_eval(function, **operands)` — lambda-based evaluation
- [ ] Operator support: arithmetic, bitwise, logical, comparison
- [ ] Mode coercion rules matching Pillow

### ExifTags & TiffTags

- [ ] `ExifTags.Base`, `ExifTags.GPS`, `ExifTags.IFD` — tag name/number lookups
- [ ] `TiffTags` — TIFF tag constants and type info

### features Module

- [ ] `features.check(feature)` — runtime feature detection
- [ ] `features.version(feature)` — version strings for codec backends
- [ ] `features.pilinfo()` — diagnostic output

---

## Phase 5 — Animation & Multi-Frame Support

Support reading and writing animated and multi-page images. This is a large
architectural change that touches codecs, the `Image` struct, and the Python
API.

- [ ] `Image.seek(frame)` / `Image.tell()` — frame navigation
- [ ] `Image.n_frames` property, `is_animated` reflecting actual frame count
- [ ] `ImageSequence.Iterator(image)` — frame iteration protocol
- [ ] `ImageSequence.all_frames(image, func=None)`
- [ ] Animated GIF read/write (frame delays, disposal, loop count)
- [ ] Animated WebP read/write
- [ ] APNG read/write
- [ ] Multi-page TIFF read
- [ ] Multi-page PDF write

---

## Phase 6 — Metadata Preservation

Currently, metadata is read for orientation handling and exposed via
`image.info`, but saving writes pixels only. Full metadata round-tripping is
required for a drop-in replacement.

- [ ] Preserve EXIF on save (JPEG, PNG, WebP, JXL, HEIF, TIFF)
- [ ] Preserve XMP on save
- [ ] Preserve ICC profiles on save
- [ ] `Exif` class for reading and modifying EXIF IFDs (matching `PIL.Image.Exif`)
- [ ] `image.getexif()` / `image.info["exif"]` parity
- [ ] Thumbnail embedding in EXIF
- [ ] `PngInfo` for custom PNG text chunks

---

## Phase 7 — ICC Color Management (ImageCms)

ICC color management enables accurate color conversion between devices and
color spaces.

- [ ] `ImageCms.profileToProfile(image, inputProfile, outputProfile, ...)`
- [ ] `ImageCms.buildTransform` / `buildProofTransform`
- [ ] `ImageCms.createProfile(colorSpace)` — sRGB, Adobe RGB, display P3
- [ ] `ImageCms.getProfileInfo`, `getProfileName`, `getProfileDescription`
- [ ] `ImageCms.applyTransform(image, transform)` — in-place transform
- [ ] Intent support (perceptual, relative colorimetric, saturation, absolute)
- [ ] Rust-native ICC parsing (e.g. via `lcms2` bindings or pure Rust)

---

## Phase 8 — Extended Format Support

Add formats that Pillow supports but Blanket does not, ordered by real-world
usage.

### High Priority

- [ ] PPM / PGM / PBM / PAM (Netpbm family) — read/write
- [ ] TGA — read/write
- [ ] DDS — read (DirectDraw Surface, game assets)
- [ ] EPS — read (via Ghostscript or rasterization fallback)
- [ ] QOI — read/write (simple lossless format)
- [ ] JPEG 2000 — read/write

### Medium Priority

- [ ] TIFF compressed (LZW, Deflate, JPEG) — read/write
- [ ] PCX — read/write
- [ ] SGI — read/write
- [ ] XBM / XPM — read/write
- [ ] FITS — read

### Lower Priority

- [ ] IM — read
- [ ] SPIDER — read
- [ ] Pixar — read
- [ ] MSP — read/write
- [ ] PALM — write
- [ ] Sun raster — read

---

## Phase 9 — Platform Utilities

Modules providing system integration. Lower priority for a processing-focused
library, but needed for full parity.

- [ ] `ImageGrab.grab(bbox=None, include_layered_windows=False, all_screens=False)` — screenshot
- [ ] `ImageGrab.grabclipboard()` — clipboard image
- [ ] `Image.show()` backend infrastructure (viewer registry)
- [ ] `ImageWin` — Windows-specific display (DIB, HDC) — low priority
- [ ] `PSDraw` — PostScript generation — low priority

---

## Phase 10 — Blanket Extras

Features unique to Blanket that go beyond Pillow's capabilities, leveraging the
Rust backend for performance and functionality Pillow doesn't offer.

### Already Shipped

- [x] **Save-time compression** — `LosslessImageCompressor` and `LossyImageCompressor`
      for automatic encoding optimization with bounded error
- [x] **High bit depth** — 10, 12, and 16-bit sample retention across
      operations and formats
- [x] **Modern format defaults** — JPEG XL, AVIF, HEIC as first-class citizens,
      not plugin extras
- [x] **SIMD-accelerated pixel kernels** — NEON, SSSE3, and portable SIMD
      paths for critical operations
- [x] **Parallel execution** — automatic rayon-based parallelism with
      small-buffer fast paths
- [x] **3D color LUTs** — `Color3DLUT` with trilinear interpolation,
      generation, and chained transforms

### Planned

- [ ] **Batch processing API** — process multiple images in parallel with
      shared pipeline configuration
- [ ] **GPU acceleration** — optional compute shader backend for large-image
      operations (resize, convolution, compositing)
- [ ] **Streaming decode/encode** — progressive JPEG/JXL decode, incremental
      PNG write for memory-constrained workflows
- [ ] **Image hashing** — perceptual hashing (pHash, dHash, aHash) for
      deduplication and similarity search
- [ ] **Smart cropping** — content-aware crop using saliency detection or
      face detection hints
- [ ] **Advanced quantization** — neural-network-based color quantization,
      perceptual palette optimization
- [ ] **Raw photo support** — decode camera raw formats (CR2, NEF, ARW, DNG)
      via `rawler` or `libraw` bindings
- [ ] **HDR pipeline** — tone mapping operators (Reinhard, ACES, filmic),
      HDR10/PQ/HLG metadata handling
- [ ] **SIMD filter kernels** — vectorized convolution, median, and morphological
      operations
- [ ] **WebAssembly target** — compile to Wasm for browser-side image processing

---

## Non-Goals

These items are explicitly out of scope:

- **Pillow plugin system** — Blanket uses compiled Rust codecs, not runtime
  Python decoder/encoder plugins
- **PIL.ImageTk / PIL.ImageQt** — GUI toolkit integration is left to wrapper
  libraries
- **Backwards compatibility with PIL 1.x** — only modern Pillow (10+) API is
  targeted
- **Python < 3.12** — Blanket targets current Python releases
