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

## Phase 6 — Metadata Preservation — Complete

JPEG, PNG/APNG, WebP, JPEG XL, HEIF, and TIFF preserve EXIF, XMP, and ICC
profiles when saving. Explicit save arguments override or remove metadata.
EXIF IFDs and JPEG thumbnails are editable without Pillow. PNG text supports
Latin-1, Unicode, compression, language, and translated keywords. See the
metadata guide for container limits and opaque MakerNote behavior.

- [x] Preserve EXIF on save (JPEG, PNG, WebP, JXL, HEIF, TIFF)
- [x] Preserve XMP on save
- [x] Preserve ICC profiles on save
- [x] `Exif` class for reading and modifying EXIF IFDs (matching `PIL.Image.Exif`)
- [x] `image.getexif()` / `image.info["exif"]` parity
- [x] Thumbnail embedding in EXIF
- [x] `PngInfo` for custom PNG text chunks

---

## Phase 7 — ICC Color Management (ImageCms) — Complete

ICC color management enables accurate color conversion between devices and
color spaces.

The public `ImageCms` module uses LittleCMS through Rust bindings for profile
parsing, reusable transforms, and soft proofing. Transforms support 8-bit `L`,
`RGB`, `RGBA`, `CMYK`, and `LAB` with compatible profiles, preserve RGBA alpha,
and embed the output profile in image metadata. In-place transforms affect only
the current frame and require matching input/output modes. Profiles load from
paths or binary streams and serialize with `ImageCmsProfile.tobytes()`.

- [x] `ImageCms.profileToProfile(image, inputProfile, outputProfile, ...)`
- [x] `ImageCms.buildTransform` / `buildProofTransform`
- [x] `ImageCms.createProfile(colorSpace)` — sRGB, Adobe RGB, display P3 (plus LAB and XYZ)
- [x] `ImageCms.getProfileInfo`, `getProfileName`, `getProfileDescription`
- [x] `ImageCms.applyTransform(image, transform, inPlace=True)` — in-place transform
- [x] Intent support (perceptual, relative colorimetric, saturation, absolute)
- [x] Rust-native ICC parsing through `lcms2` bindings

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
- [x] **SIMD filter kernels** — portable SIMD 3x3/5x5 convolution,
      3x3 median, and 3x3/5x5 min/max morphology, with scalar edges and tails;
      larger rank and morphology windows retain histogram and sliding-window paths
- [x] **Streaming decode/encode** — progressive JXL input through
      `ImageFile.Parser` and incremental PNG row output through `ImageFile.PngWriter`
- [x] **Progressive JPEG streaming decode** — consume 8-bit Huffman JPEG chunks
      with bounded encoded input storage and independent previews after completed scans
- [x] **Image hashing** — `ImageHash.average_hash`, `dhash`, and `phash` with
      configurable sizes, hexadecimal serialization, and Hamming distance
- [x] **GPU acceleration foundation** — optional Apple Silicon Metal backend
      with resident image operations and PyTorch MPS tensor pipelines.
      See [GPU Roadmap](#gpu-roadmap) for shipped coverage and subsequent phases,
      and [GPU images](GPU.md) for the current API and restrictions.

### Planned

- [ ] **Batch processing API** — process multiple images in parallel with
      shared pipeline configuration
- [ ] **Smart cropping** — content-aware crop using saliency detection or
      face detection hints
- [ ] **Advanced quantization** — neural-network-based color quantization,
      perceptual palette optimization
- [ ] **Raw photo support** — decode camera raw formats (CR2, NEF, ARW, DNG)
      via `rawler` or `libraw` bindings
- [x] **HDR pipeline** — `ImageHdr.tone_map` with Reinhard, ACES, and filmic
      operators, PQ/HLG transfer decoding, BT.2020-to-sRGB rendering, and
      PNG/APNG HDR10/PQ/HLG metadata (`cICP`, `mDCV`, `cLLI`) handling
- [ ] **WebAssembly target** — compile to Wasm for browser-side image processing

### Future Ideas

Exploratory candidates beyond the planned work above. These are not release
commitments; scope and priority depend on use cases, backend support, and
measured benefits. Existing Pillow-shaped APIs should retain their behavior,
with new processing semantics exposed explicitly.

#### Precision and Color

- [x] **High-bit-depth processing coverage** — filters (including 3D LUTs),
      enhancement, blending, masked paste, alpha compositing, and full-range
      statistics retain 10/12/16-bit samples without an 8-bit conversion;
      unsigned `I;16` variants retain their storage mode and byte order.
      Portable SIMD covers sample packing, enhancement, convolution, small
      ranks, blur, unsharp masking, LUTs, compositing, and statistical moments
- [x] **Floating-point color images** — `RGB;F`/`RGBA;F` float32 storage,
      array exchange, conversion, crop/transpose, resize, paste, source-over,
      and TIFF I/O, with defined negative and nonfinite sample handling
- [x] **Linear-light processing** — opt-in resizing and source-over compositing
      with sRGB/linear transfer selection and explicit resize alpha premultiplication

#### Large Images and Resource Control

- [x] **Region and tile decoding** — `Image.open_region` reads intersecting
      TIFF strips/tiles for unsigned 8/16-bit chunky L/RGB/straight RGBA;
      other layouts/codecs require explicit `fallback="full"`
- [x] **Lazy frame access** — `Image.open(..., lazy=True, frame_cache_size=1)`
      decodes animation frames and TIFF pages on demand with a bounded LRU cache;
      evicted frame edits are discarded and animation cache misses replay originals
- [ ] **Processing budgets** — configurable memory, decoded-pixel, frame-count,
      and worker limits for applications handling many images concurrently
- [ ] **Reusable processing plans** — combine compatible operations to reduce
      intermediate allocations while preserving rounding and operation order

#### Format and Sequence Extensions

- [ ] **Multi-page TIFF writing** — save pages with independent dimensions,
      supported modes, sample depths, metadata, and compression settings
- [ ] **AVIF and JPEG XL animation** — expose frame timing and loop controls
      through the existing sequence API where codec backends support them
- [ ] **Multi-resolution containers** — select and write multiple ICO sizes,
      and expose TIFF pyramid levels without treating them as animation frames
- [ ] **OpenEXR support** — read and write half/float RGB and alpha samples,
      building on floating-point color storage

#### Drawing and Analysis

- [ ] **Font fallback and variable fonts** — explicit fallback chains,
      variation-axis selection, and metrics consistent with rendered glyphs
- [ ] **Paragraph layout** — width-constrained wrapping, Unicode line breaking,
      and justification integrated with shaping and bidirectional text
- [ ] **Morphology and segmentation** — explicit erosion, dilation, opening,
      closing, connected components, and adaptive thresholding APIs
- [ ] **Image comparison metrics** — sample-aware error metrics and difference
      maps with documented alpha and color-space handling

#### Interoperability

- [ ] **Buffer views and array exchange** — opt-in zero-copy access for compatible
      layouts, with explicit ownership, lifetime, mutability, and stride rules
- [x] **Compressed-domain JPEG transforms** — lossless rotation, flips, and
      aligned cropping of encoded JPEGs with explicit metadata handling

#### Codec Controls and Inspection

- [x] **Header-only inspection** — `Image.inspect` reports dimensions, mode,
      sample depth, and available container metadata without allocating decoded
      image rasters
- [ ] **Decode-time thumbnails** — request reduced-resolution decoding where
      supported, making `draft` useful before full raster allocation
- [x] **Encoding capability queries** — report supported modes, sample depths,
      compression methods, and animation options for each installed backend
- [x] **Structured codec errors** — distinguish unsupported features, corrupt
      input, missing backends, and resource-limit failures for callers
- [ ] **Additional TIFF layouts** — BigTIFF and tiled output with configurable
      tile sizes, predictors, and explicit compression compatibility checks
- [ ] **Texture container access** — expose DDS mip levels, array layers, and
      cubemap faces through a dedicated surface API

#### Metadata Workflows

- [x] **Metadata-only rewrites** — `ImageMetadata.rewrite` edits JPEG, PNG/APNG,
      WebP, and JPEG XL metadata while retaining compressed image data
- [x] **Metadata policies** — `MetadataPolicy` selects EXIF, XMP, PNG text, and
      known location fields, with independent ICC profile control
- [x] **Resolution and physical units** — `ImageMetadata.get_resolution`
      normalizes DPI and pixel aspect with explicit units and conflict rules
- [x] **Embedded preview access** — `ImageMetadata.get_previews` extracts EXIF
      thumbnails, HEIF/AVIF thumbnails, and binary EPS previews

#### Drawing and Compositing Extensions

- [ ] **Bezier paths** — quadratic and cubic curves with configurable stroke
      joins, caps, dash patterns, and fill rules
- [ ] **Gradient and pattern fills** — linear and radial gradients, repeated
      image patterns, and reusable clipping masks for drawing operations
- [ ] **Extended blend modes** — compositing operators such as multiply, screen,
      overlay, and soft light with defined alpha behavior
- [ ] **Color glyph rendering** — support selected color-font formats with
      explicit fallback for unsupported glyph representations
- [ ] **Distance-field masks** — signed distance fields from bilevel or alpha
      masks for scalable outlines, glows, and feathered selections

#### Filtering and Restoration

- [ ] **Edge-preserving smoothing** — bilateral or guided filters with explicit
      radius, range, and boundary behavior
- [ ] **Local contrast adjustment** — adaptive histogram equalization with
      configurable tiles and contrast limits
- [ ] **Edge detection** — Sobel and Scharr gradients plus Canny edges with
      explicit threshold and output conventions
- [ ] **Lens correction and remapping** — radial distortion correction and
      reusable coordinate maps with configurable interpolation and borders

#### Compression and Animation Optimization

- [ ] **Target-size encoding** — search supported encoder settings for a byte
      budget and report when the requested size cannot be reached
- [ ] **Compression reports** — expose chosen settings, candidate sizes,
      measured error, and search cost from compressor runs
- [ ] **Animation delta optimization** — choose frame rectangles and disposal
      or blending controls while preserving rendered frames and timing
- [ ] **Shared animation palettes** — optimize a palette across frames with
      temporal stability and explicit transparency constraints
- [ ] **Sequence editing helpers** — trim, concatenate, and retime animations
      while preserving source positions and validating frame compatibility

---

## GPU Roadmap

GPU development follows a separate track from the Pillow compatibility phases.
Apple Silicon is the initial platform; NVIDIA support is the next backend.
Keep backend-specific allocation, synchronization, and kernels in dedicated
workspace crates, with common operation contracts and Python validation shared
across backends. Existing CPU APIs retain their behavior.

GPU processing remains explicit: upload CPU images when requested, keep GPU
inputs and intermediate results resident, and download only through an explicit
CPU export. Unsupported operations must report their limitations rather than
silently staging pixels through CPU memory. Zero-copy sharing requires compatible
devices, layouts, ownership, and synchronization; GPU copies may be needed.

CPU-only builds must remain usable without GPU drivers or Python GPU packages.

### GPU Phase 0 — Apple Silicon Foundation — Complete

- [x] Separate `blanket-gpu`, `blanket-gpu-metal`, and `blanket-gpu-python`
      crates, with optional Metal support through `objc2`/`objc2-metal`
- [x] Explicit `ImageGpu` upload/download, resident results, capability queries,
      synchronization, and resource ownership
- [x] Uint8 L/LA/RGB/RGBA and float32 F/RGB;F/RGBA;F storage
- [x] All six resize filters, crop, transpose, wrapped offset, 3x3/5x5
      convolution, brightness, inversion, channel LUTs, solarize, posterize,
      channel arithmetic, blending, and straight-alpha source-over
- [x] Optional PyTorch MPS import/export with HW/HWC/CHW layouts, retained
      tensor ownership, and GPU packing of views without CPU pixel staging
- [x] Native Metal and MPS hardware tests for CPU parity, documented float
      semantics, geometry, lifetime handling, validation, and residency

Native Metal and PyTorch allocations currently use separate execution contexts.
Native Metal buffer export to PyTorch is not implemented; CPU images intended
for tensor output can be uploaded with `runtime="torch"`.

### GPU Phase 1 — Operation and Precision Coverage — Complete

- [x] Mode conversion between supported uint8/float32 GPU modes, channel
      extraction/merge, grayscale, and alpha replacement with resident results
- [x] Gaussian/box blur, unsharp masking, and min/max morphology and median/rank
      filters with odd 1x1 through 7x7 windows, on native Metal and PyTorch MPS
- [x] Color, contrast, and sharpness enhancement with resident contrast reduction;
      3D LUTs and explicit linear-light resize/compositing on native Metal and MPS
- [x] Masked paste/compositing, affine and perspective transforms, arbitrary
      rotation, and reusable coordinate remapping
- [x] Resident reductions for histograms, extrema, moments, and statistics;
      return small CPU results only when explicitly requested
- [x] Unsigned 10/12/16-bit processing with retained sample precision and
      documented rounding, saturation, alpha, and nonfinite behavior

Geometry and reductions run on native Metal and MPS. Histograms cover uint8;
statistics cover all working formats. Unsigned L/RGB/RGBA use normalized
float32 storage and quantize resident public results to their original depth.
GPU.md and `ImageGpu.capabilities()` specify the unsigned operation subset
and numerical contracts; unsupported operations never use a CPU fallback.

Add operations through shared contracts, then implement and validate them on
each backend. Capability queries must distinguish missing kernels from missing
hardware; partial coverage must not trigger an implicit CPU fallback.

### GPU Phase 2 — NVIDIA and CUDA Tensor Pipelines

- [x] Add an optional CUDA backend in dedicated workspace crates, reusing
      descriptors, validation, and operation contracts
- [ ] Reach the shipped Metal operation and sample-format coverage on NVIDIA
      hardware, with explicit backend and device selection
- [x] Upload CPU images once, retain intermediate allocations on the selected
      CUDA device, and make CPU downloads explicit
- [x] Import/export compatible PyTorch CUDA tensors through DLPack, honoring
      device identity, storage offsets, strides, ownership, and producer/consumer
      stream ordering without CPU pixel staging
- [x] Report missing drivers, unsupported devices, allocation failures, and
      backend-specific capabilities through actionable errors and diagnostics
- [ ] Add NVIDIA hardware CI and cross-backend parity tests, including tensor
      views, source lifetimes, chained operations, and transfer boundaries

The optional `gpu-cuda` build feature supplies CUDA kernels through
`blanket-gpu-cuda`, sharing Metal's pixel contracts. CuPy handles CUDA
allocations, precompiled fatbinary loading, and DLPack stream negotiation. Native CUDA
execution needs CuPy but does not need PyTorch. Shared numeric contracts and
CUDA ownership/stream tests are available for manual NVIDIA runs. The CUDA
workflow compiles CUDA kernels with nvcc, embeds them in a dev-profile wheel
on hosted Ubuntu, and uploads it as a
downloadable artifact. Hardware CI and coverage remain unchecked until that
suite runs on NVIDIA; compilation cannot establish CUDA parity. See GPU.md
for setup.

### GPU Phase 3 — Pipeline Efficiency and Resource Control

- [ ] Release-build benchmarks for cold/warm execution, CPU upload plus download,
      and already-resident tensor pipelines; publish measured crossover sizes
- [ ] Reuse allocations, compiled pipelines, coefficient buffers, and LUTs with
      bounded caches and explicit memory limits
- [ ] Batch execution and reusable processing plans that reduce command
      overhead and intermediate allocations
- [ ] Fuse compatible operations only when their rounding, saturation, alpha,
      and evaluation-order contracts remain satisfied
- [ ] Stream/event synchronization and asynchronous transfers with explicit
      completion, cancellation, and error propagation rules
- [ ] Tile processing for images exceeding working-memory budgets, with correct
      filter halos and no unnecessary CPU round trips

### GPU Phase 4 — Broader Interoperability and Backends

- [ ] Native Metal/PyTorch buffer exchange where supported, with safe ownership
      and synchronization; use explicit GPU copies when aliasing is unavailable
- [ ] Extend tensor exchange to other DLPack producers with per-device layout
      and stream compatibility checks
- [ ] Add further backends, such as Vulkan/WebGPU or ROCm, behind the shared
      operation contracts without requiring changes to CPU image storage
- [ ] Define explicit cross-device transfers and capability negotiation for
      applications using multiple GPUs or backend types

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
