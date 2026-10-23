# Roadmap

Blanket's path from a focused Pillow-shaped library to a full drop-in
replacement — and beyond.

Each phase builds on the previous one. Items within a phase are roughly ordered
by priority but can ship independently.

---

## Phase 0 — Complete

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

## Phase 1 — Mode Parity — Complete

Add the missing Pillow image modes so that code written for Pillow can work
unchanged. Each new mode needs storage, conversion to/from existing modes,
pixel access, and compatibility across existing operations.

- [x] `1` — bilevel (1-bit per pixel, packed or unpacked)
- [x] `LA` — grayscale with alpha
- [x] `PA` — indexed palette with alpha
- [x] `I` — 32-bit signed integer pixels
- [x] `I;16` / `I;16L` / `I;16B` — 16-bit integer (consolidate with existing high-bit-depth)
- [x] `F` — 32-bit floating-point pixels
- [x] `CMYK` — 4-channel print color (no ICC transform required; raw channel storage)
- [x] `YCbCr` — luma/chroma (used internally by JPEG; expose for user code)
- [x] `LAB` — CIE L\*a\*b\*
- [x] `HSV` — hue, saturation, value

### Depends on

- `ImageChops.logical_and`, `logical_or`, `logical_xor` require mode `1`

---

## Phase 2 — ImageDraw — Complete

Drawing primitives are the most-requested missing module for real-world Pillow
replacement. This is a substantial feature that warrants its own phase.

The public `ImageDraw` module and `blanket-draw` crate provide in-place geometry,
optional supersampled edges, bitmap text, font-object interoperability, and
four-connected flood filling. Default text uses a small built-in bitmap font;
font loading and advanced text rendering remain in Phase 3. Drawing uses
Blanket's rasterization rules; curved edges and wide strokes can differ from
Pillow. See the ImageDraw reference for supported modes and text limitations.

- [x] `ImageDraw.Draw(image, mode=None)` context
- [x] `point`, `rectangle`, with color fills and rectangle outline widths
- [x] Remaining geometric primitives: `line`, `rounded_rectangle`, `ellipse`, `arc`, `chord`, `pieslice`, `polygon`, `regular_polygon`
- [x] Fill, outline, and width control for remaining primitives
- [x] Anti-aliased drawing
- [x] `textbbox`, `textlength`, `multiline_textbbox`
- [x] `text`, `multiline_text` (bitmap fallback and supplied font objects)
- [x] Flood fill (`floodfill`)

---

## Phase 3 — ImageFont & Text Rendering — Complete

The public `ImageFont` module uses `ab_glyph` rasterization, `rustybuzz`
OpenType shaping, and Unicode bidirectional/script runs without linking
FreeType. Fonts integrate with `ImageDraw` masks, strokes, anchors, and
multiline alignment. See the ImageFont reference for layout limitations.

- [x] `ImageFont.truetype(font, size, ...)` — load `.ttf`/`.otf` files
- [x] `ImageFont.load_default(size=None)` — built-in fallback font
- [x] Glyph rasterization and layout (horizontal, basic vertical)
- [x] Font metrics: `getbbox`, `getlength`, `getmetrics`
- [x] Bitmap font support (`ImageFont.load`)
- [x] Unicode and multi-script rendering
- [x] Text anchors and alignment

---

## Phase 4 — Missing Image Methods & Module Gaps — Complete

Fill in the remaining `Image` methods and small utility modules that Pillow
users expect.

### Image Methods

- [x] `show()` — display via system viewer or configurable backend
- [x] `effect_spread(distance)` — randomly displace pixels
- [x] `remap_palette(dest_map, source_palette=None)` — reorder palette indices
- [x] `verify()` — validate eagerly decoded pixels
- [x] `draft(mode, size)` — no-op after eager decoding
- [x] `getim()` / `im` — internal image core access (compatibility shim)
- [x] `tobitmap(name="image")` — X11 bitmap string (mode `1`)
- [x] `entropy(mask, extrema)` on high-bit-depth images

### ImageColor Module

- [x] `getrgb(color)` — public API (currently internal `_color.py`)
- [x] `getcolor(color, mode)` — convert color string to mode-specific value
- [x] CSS4 absolute color literals, including modern syntax, Lab/OKLab, predefined spaces, and sRGB gamut mapping (see the ImageColor reference for contextual CSS exclusions and Pillow's integer-alpha extension)

### ImagePath Module

- [x] `Path` object with `compact`, `getbbox`, `map`, `tolist`, `transform`
- [x] Integration with `ImageDraw` for path-based drawing

### ImageMath Module

- [x] `eval(expression, **operands)` and `unsafe_eval` — per-pixel math expressions
- [x] `lambda_eval(function, **operands)` — lambda-based evaluation
- [x] Operator support: arithmetic, bitwise, logical, comparison
- [x] Mode coercion rules matching Pillow

### features Module

- [x] `features.check(feature)` — runtime feature detection
- [x] `features.version(feature)` — version strings for codec backends
- [x] `features.blanketinfo()` — diagnostic output

---

## Phase 5 — Animation & Multi-Frame Support — Complete

Support reading and writing animated and multi-page images. This is a large
architectural change that touches codecs, the `Image` struct, and the Python
API.

Frames decode eagerly into independent rasters. Animated GIFs expose composited
RGBA canvases. Navigation retains frame edits; copies and transforms operate on
the current frame. Sequence saving uses `save_all` and `append_images`, with
format-specific timing and composition controls. See the formats guide for
memory limits, APNG sample precision and blending, and compatibility details.

- [x] `Image.seek(frame)` / `Image.tell()` — frame navigation
- [x] `Image.n_frames` property, `is_animated` reflecting actual frame count
- [x] `ImageSequence.Iterator(image)` — frame iteration protocol
- [x] `ImageSequence.all_frames(image, func=None)`
- [x] Animated GIF read/write (frame delays, disposal, loop count)
- [x] Animated WebP read/write
- [x] APNG read/write
- [x] Multi-page TIFF read
- [x] Multi-page PDF write

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

- [x] PPM / PGM / PBM / PAM (Netpbm family) — read/write
- [x] TGA — read/write
- [x] DDS — read (DXT1/DXT3/DXT5 textures, first surface)
- [x] EPS — read (requires Ghostscript)
- [x] QOI — read/write (simple lossless format)
- [x] JPEG 2000 — read/write (grayscale/RGB, lossless JP2 output)

### Medium Priority

- [x] TIFF compressed (LZW, Deflate, JPEG) — read/write
- [x] PCX — read/write
- [x] SGI — read/write
- [x] XBM / XPM — read/write
- [x] FITS — read (primary image, first plane; uncompressed samples)

### Lower Priority

- [x] SPIDER — read (2D images and first stack image)
- [x] Pixar — read (uncompressed RGB)
- [x] MSP — read/write
- [x] PALM — write
- [x] Sun raster — read

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
