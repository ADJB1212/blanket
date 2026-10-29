# GPU images

For PyTorch interoperability, install the optional GPU runtime group, then
build on Apple Silicon:

```sh
uv sync --no-default-groups --no-install-project --group gpu
uv run --no-sync maturin develop --uv --features gpu-metal
```

The `test` group includes `gpu` for GPU testing:

```sh
uv sync --no-default-groups --no-install-project --group test
uv run --no-sync maturin develop --uv --features gpu-metal
uv run --no-sync pytest tests/test_gpu.py tests/test_gpu_operations.py -n 0
```

Native Metal execution requires no additional Python runtime packages; build
it with `maturin develop --uv --features gpu-metal`.
The feature is optional; normal Blanket installations need neither Metal nor
PyTorch. GPU operations never silently fall back to CPU processing.

```python
from blanket import Image, ImageGpu

image = Image.open("input.png")
gpu = ImageGpu.from_image(image)
result = gpu.resize((1024, 768)).convolve([1] * 9)
result.to_image().save("output.png")
gpu.close()
result.close()
```

`from_image` uploads only the active frame, copies metadata, and leaves the
source position unchanged. Intermediate results stay resident. `to_image`
waits and downloads an independent CPU raster. `synchronize` waits explicitly;
`close` releases the handle while submitted commands retain their resources.
Native Metal transfers copy directly between Rust image storage and Metal,
without staging a full raster in Python `bytes`. Downloads own their CPU pixels.
Operations return independent pixel storage. Native Metal caches closed buffers
up to 128 MiB and 16 allocations, and reuses them only after all GPU writers and
readers have finished. Closing an image does not wait for its queued consumers.
Metal compiles and caches pipelines per operation on first use. Packed byte
kernels handle copy, blend, brightness, and invert; RGBA resize and box-blur
passes share coordinate calculations across all four channels. PyTorch MPS and
CUDA continue to use the shared portable kernels.

Supported modes are uint8 L/LA/RGB/RGBA and float32 F/RGB;F/RGBA;F, plus
unsigned 10/12/16-bit L/RGB/RGBA as described below. Resize
supports all six `Image.Resampling` filters and premultiplies alpha for filtered
resizing. Convolution accepts 3x3/5x5 kernels, scale and offset, and retains
border samples. `blend(other, alpha)` and `alpha_composite(source)` require
matching modes, dimensions, and execution contexts. Source-over requires LA,
RGBA, or RGBA;F. Float32 inputs are not automatically normalized.

Additional resident operations work in both native and PyTorch execution
contexts:

- `crop((left, top, right, bottom))` accepts integer coordinates and fills
  out-of-bounds pixels with zero. Bounds must fit signed 32-bit integers;
  empty or reversed output dimensions are rejected.
- `transpose(method)` supports all seven `Image.Transpose` flips, rotations,
  and transposes. `offset(x, y=None)` wraps around image edges, using `x` for
  both axes when `y` is omitted. Geometry copies preserve sample bits.
- `brightness(factor)` matches CPU enhancement for uint8: clip and truncate,
  preserve RGBA alpha, and scale LA alpha. Float color values remain unclipped,
  and RGBA;F alpha is preserved. Factors must be finite float32 values.
- `invert()` operates on every channel, including alpha, as `255 - value` or
  float `1 - value`, following ImageChops rather than preserving alpha.
- `point(lut)` accepts uint8 lookup tables with 256 shared entries or 256
  entries per channel. Values are rounded and clipped like `Image.point`.
  `solarize(threshold=128)` and `posterize(bits)` also affect alpha and require
  uint8 samples. Posterize accepts 0 through 8 bits.
- `difference`, `multiply`, `screen`, `lighter`, `darker`, `add`, and `subtract`
  operate on matching resident images. All channels, including alpha,
  participate. Uint8 results match ImageChops clipping and integer arithmetic.
  Float multiply/screen use 1 as white; float arithmetic remains unclipped,
  and NaNs propagate, including through lighter/darker.

`add(other, scale=1, offset=0)` and `subtract(other, scale=1, offset=0)` divide
the sum/difference by scale, then add offset. Scale must be a nonzero normal
float32 value; offset must be finite and representable as float32. Unlike CPU
ImageChops, binary GPU operations require matching sizes instead of cropping
to the overlap. Metadata is copied from the first input.

```python
result = gpu.crop((20, 30, 900, 700)).transpose(Image.Transpose.ROTATE_90).brightness(1.25).solarize(192)
```

These operations allocate resident results without downloading pixels; lookup
tables and scalar controls are small CPU-to-GPU control-data transfers.

## Conversion, channels, and alpha

`convert(mode)` converts between the supported uint8 and float32 modes without
leaving the GPU. Uint8 grayscale uses the CPU's fixed-point RGB luminance rule.
F represents raw grayscale sample units, conventionally 0 through 255;
RGB;F/RGBA;F color samples conventionally use 0 through 1. Converting F to color
float divides by 255; converting color float to F computes luminance and
multiplies by 255. Negative and nonfinite values remain representable in float
outputs. `getchannel` and `merge` preserve raw samples without rescaling.

Converting F to uint8 truncates and clips to 0 through 255. Converting color
float to uint8 rounds after scaling by 255; negative values become zero,
positive infinity becomes 255, and NaN becomes zero. Grayscale conversion drops
alpha without compositing. `grayscale()` chooses L for uint8 input and F for
float input. Conversions involving other modes or sample depths remain unsupported.

`getchannel(index_or_name)` and `split()` return independent resident L or F
images. `ImageGpu.merge(mode, bands)` combines matching single-channel images
with the same dimensions, dtype, and execution context. Returned channels are
independent of their source, and merged images remain valid after bands close.
Metadata comes from the first input.

`putalpha(alpha)` returns a new image with replaced alpha. Accepts a matching
resident L/F band, or an integer scalar for uint8 / finite scalar for float.
Scalar alpha is clipped to 0 through 255 or 0 through 1. L/RGB promote to
LA/RGBA; float images promote to RGBA;F, since there is no float LA mode.
Use `convert("L")`, `convert("RGB")`, or `convert("RGB;F")` to discard alpha.

```python
red, green, blue = gpu.convert("RGB;F").split()
result = ImageGpu.merge("RGB;F", (blue, green, red)).putalpha(0.75)
```

## Blur and rank filters

`box_blur(radius)` accepts a nonnegative scalar or `(horizontal, vertical)`
radius, including fractions. `gaussian_blur(radius)` uses the same three
fractional box passes per axis as the CPU approximation. Both replicate edge
samples and process every channel, including alpha, without premultiplication.
Uint8 passes preserve the CPU's 24-bit fixed-point weighting and rounding;
float passes retain float32 samples without clipping.

`unsharp_mask(radius=2, percent=150, threshold=3)` sharpens against Gaussian
blur. Percent is a signed 32-bit integer; threshold uses raw sample units, so
normalized float pipelines typically need a threshold such as `0.01`. Uint8
corrections truncate like the CPU and saturate; float corrections remain
unclipped. Channels whose difference does not exceed the threshold retain their
original samples, including NaN differences.

`min_filter(size=3)`, `max_filter(size=3)`, and `median_filter(size=3)` support odd
windows from 1x1 through 7x7, including windows larger than the image.
`rank_filter(size, rank)` exposes any zero-based rank in those windows. Borders
replicate edge samples, alpha participates, and a float window containing NaN
produces NaN. These are per-channel operations; they do not compare luminance.

```python
result = gpu.gaussian_blur((1.5, 0.5)).unsharp_mask(1, 150, 0.01).median_filter()
```

## Enhancement and 3D LUTs

`color(factor)`, `contrast(factor)`, and `sharpness(factor)` return resident
results for all supported modes. Factors must be finite float32 values; 1
returns an independent unchanged copy, 0 selects the degenerate image, and
values outside 0 through 1 extrapolate. Uint8 outputs truncate and saturate;
float outputs remain unclipped.

Color enhancement blends against RGB luminance and leaves grayscale inputs
unchanged. Contrast uses mean luminance, excluding alpha: uint8 rounds the mean
like the CPU, while float uses an unrounded float32 mean in the input's sample
units. The reduction stays on the GPU, including all intermediate partial sums;
byte sums retain 64-bit integer precision. Float NaNs and infinities participate
in the mean normally. Color and contrast preserve LA/RGBA alpha.

Sharpness blends against the CPU's 3x3 smooth kernel, retaining border pixels
and RGBA alpha. Like CPU enhancement, LA alpha participates in smoothing.
Float smoothing retains float32 precision rather than quantizing to uint8.

`color_lut(lut)` accepts an `ImageFilter.Color3DLUT` with three or four output
channels and grid dimensions from 2 through 65, including noncubic tables.
Input and target modes must be RGB/RGBA variants with matching dtype. A
three-channel table preserves an existing alpha channel when the target has
alpha; a four-channel table supplies alpha and may promote RGB to RGBA.
Set `target_mode` on the table when changing channel count.

Uint8 LUTs match CPU table quantization and trilinear interpolation. Float LUTs
clamp input coordinates to 0 through 1, treat NaN coordinates as zero, and
retain unclipped table output values. Existing alpha passes through unchanged
for three-channel float tables. Tables must contain the expected number of
finite samples. Each call snapshots the mutable table into GPU control data;
image pixels remain resident.

```python
from blanket import ImageFilter

lut = ImageFilter.Color3DLUT.generate(17, lambda r, g, b: (b, g, r))
result = gpu.color(0.8).contrast(1.2).sharpness(1.5).color_lut(lut)
```

## Linear-light processing

`resize(..., linear_light=True, transfer="srgb")` and
`alpha_composite(source, linear_light=True, transfer="srgb")` decode sRGB color,
process resident float32 pixels, and encode the result back to sRGB while
retaining the original mode/dtype. `transfer="linear"` skips transfer decoding
and encoding for inputs already in linear light. Defaults retain processing in
the existing encoded sample space. These options support RGB/RGBA variants;
source-over additionally requires RGBA and matching size/mode/context.

Transfer functions use signed extensions for negative float color samples;
negative and HDR values remain representable in float results. Alpha is never
transfer-encoded. Float processing clamps alpha to 0 through 1 and treats NaN
alpha as zero; premultiplication/unpremultiplication clears zero-alpha colors.

Resize defaults to `premultiply_alpha=True`. Setting it false filters straight
color channels instead, while still clamping float alpha. This option applies
to float or linear-light resizing; ordinary uint8 resize retains its existing
premultiplied-alpha behavior. Source-over always uses alpha-weighted color.

```python
result = gpu.resize((512, 512), linear_light=True, premultiply_alpha=True)
result = result.alpha_composite(overlay, linear_light=True)
```

GPU arithmetic can differ from CPU results: the target tolerance is one uint8
level, or float32 `rtol=1e-5, atol=1e-6` for finite values. Float alpha is clamped
to [0, 1], NaN alpha becomes zero, and zero-alpha float colors become zero when
premultiplying/unpremultiplying. Negative and nonfinite color samples otherwise
participate in floating-point arithmetic. No color-space conversion is implicit.

## Warping and masked paste

`transform(size, method, data, resample=NEAREST, fillcolor=0)` supports inverse
AFFINE (six coefficients) and PERSPECTIVE (eight coefficients) mappings.
NEAREST, BILINEAR, and BICUBIC sampling run independently per channel, including
alpha. Sampling coordinates address pixel centers at `(x + 0.5, y + 0.5)`.
Filtered samples replicate border pixels; coordinates outside the source, or
nonfinite coordinates from a singular perspective denominator, use fillcolor.
Uint8 interpolation clips and truncates; float output remains unclipped.
Evaluation uses float32, so boundary decisions and filtered samples can differ
slightly from CPU float64 evaluation.

`rotate(angle, resample=NEAREST, expand=False, fillcolor=0)` rotates about the
image center, optionally expanding its canvas. `remap(coordinates, resample,
fillcolor=0)` reuses a resident RGB;F map whose red/green samples specify source
x/y coordinates; blue is ignored. Map dimensions determine output dimensions.
NaN/infinite coordinates use fillcolor. Maps and images must share a runtime.

`paste(source, box=(0, 0), mask=None)` returns an independent image. Source
mode/runtime must match; its size may differ. The source and optional mask are
clipped to the destination canvas. Masks must match source dimensions and use
L coverage 0..255 or F coverage 0..1. Coverage is clipped, with NaN treated as
zero. Every channel, including alpha, is interpolated independently; this is
masked paste rather than source-over. Computation uses resident float32
intermediates, with conversion rounding at the final output.

## Resident reductions

`statistics()` returns a `GpuReduction` containing per-channel extrema, four
raw power sums, and an exact integer pixel count. `histogram()` returns exact
256-bin counts per channel for uint8 L/LA/RGB/RGBA; float histograms are
unsupported. Both keep results on the GPU until `to_cpu()` is called.

```python
pending = gpu.statistics()
stats = pending.to_cpu()  # Explicit download of seven words per channel.
print(stats[0]["mean"], stats[0]["extrema"])
counts = gpu.histogram().to_cpu()  # One list of 256 integer counts per channel.
```

Statistics use a sequential float32 accumulation per channel, including alpha,
without histogram quantization. Export derives mean, raw moments of orders
1..4, population variance, and standard deviation from the small result.
Counts retain uint32 precision. Sums and higher powers can round or overflow;
variance derived from raw moments can lose precision for nearly constant data.
NaNs propagate through extrema and sums; infinities participate normally.
`synchronize()` waits without downloading; `close()` releases a reduction.
Results remain valid after their source closes.

## Unsigned 10/12/16-bit images

`from_image` accepts unsigned L/RGB/RGBA at depths 10, 12, and 16, returning a
`GpuUnsignedImage` with the original `mode` and `bit_depth`. CPU upload decodes
little-endian samples in Rust and normalizes to float32 working storage; every
valid unsigned sample round-trips exactly. Public operation results quantize
on the GPU to their original depth: clip to 0..1, multiply by the unsigned
maximum, round half upward, then normalize. NaN becomes zero, positive infinity
becomes the maximum, and negative infinity becomes zero. CPU export packs
little-endian unsigned samples in Rust. Intermediate pixels stay resident.

Supported methods are listed in `capabilities()["unsigned_operations"]`:
geometry, resize, convolution, blur/rank filters, unsharp masking, enhancement,
inversion, blend, source-over, masked paste, channel extraction, and statistics.
Depth-changing conversion, LUTs, histogram, channel merge, and alpha replacement
are not exposed for unsigned images. CPU high-depth filters can differ because
this path quantizes once per public operation rather than at every internal
filter pass. Alpha participates in blur/rank/paste; enhancement and resize
follow the corresponding float color alpha rules. Source-over normalizes
alpha; RGB/RGBA resize/compositing can opt into linear-light processing.
Convolution offsets and unsharp thresholds use unsigned sample units.
Statistics use normalized working units.

With MPS, `from_torch(tensor, layout=..., bit_depth=10|12|16)` accepts uint16
storage, including views, converts and packs entirely on the GPU, and retains
independent normalized float32 working storage. `mode` may be L/RGB/RGBA or
inferred. Unsigned `to_torch()` shares those normalized float32 working samples;
it does not export uint16 storage. CPU-uploaded unsigned images require
`runtime="torch"` for tensor export.

`capabilities()` reports unsigned depths, modes, and operation coverage, the
histogram modes, and `compiled_operations` separately from device availability.

## PyTorch MPS

The `gpu` group installs PyTorch 2.13 or newer. It must expose
`torch.mps.compile_shader` and a working MPS device. The adapter uses the public
shader interface and the
PyTorch command stream, without linking libtorch into Blanket.

```python
gpu = ImageGpu.from_torch(tensor, layout="CHW")
output = gpu.resize((512, 512)).to_torch(layout="CHW")
```

Accepts HW, HWC, and CHW uint8/float32 tensors without gradients. Imports retain
the tensor owner and share storage by default; `copy=True` makes a GPU copy.
Views and noncontiguous layouts are packed on the GPU when processing starts.
Mutations through shared tensors are visible; callers must avoid concurrent
mutation and use the same MPS stream for producer, processing, and consumer.
There is no autograd integration. `to_torch` shares tensor-backed storage;
imported tensors have no image metadata. Processing does not call `.cpu()` or
stage pixels through NumPy.

Native Metal and PyTorch allocations are separate execution contexts. For CPU
images that must produce tensors, use `from_image(image, runtime="torch")`.
Native Metal `to_torch()` and mixing the two contexts are explicitly rejected.

`ImageGpu.capabilities()` reports compiled support, native device availability,
supported modes and operations, and the PyTorch interface requirement without
importing torch.

## NVIDIA CUDA

Build the optional CUDA backend with the CUDA 13 toolkit (`nvcc` on PATH,
or `NVCC` set to its executable). Building does not require a GPU or NVIDIA
driver; execution requires a compatible NVIDIA driver. The `gpu-cuda`
dependency group installs CuPy, its CUDA runtime/NVRTC
components, and PyTorch for interoperability/testing:

```sh
uv sync --no-install-project --group gpu-cuda --group test
uv run --no-sync maturin develop --uv --features gpu-cuda
```

Use the platform's codec build features and native dependencies as described
in the main build instructions. The CUDA toolkit is needed at build time;
libtorch is not linked into the Rust extension. CuPy is an optional runtime dependency for
CUDA, including its native image path; PyTorch is needed only for tensor
exchange. CPU-only installations do not import either package. CUDA and Metal
features may be enabled together.

```python
status = ImageGpu.capabilities("cuda", device=0)
if not status["available"]:
    raise RuntimeError(status["diagnostic"])

gpu = ImageGpu.from_image(image, backend="cuda", device=0)
result = gpu.resize((1024, 768)).contrast(1.1).sharpness(1.2)
tensor = result.to_torch()  # Shares CUDA storage via DLPack.
cpu_image = result.to_image()  # Explicit download.
```

Device IDs are CUDA ordinals. `device` reports `cuda:N`; binary operations
reject mismatched devices. CUDA resources use `runtime="cuda"` internally;
both `runtime="native"` and `runtime="torch"` uploads select this shared
runtime when `backend="cuda"`. The default backend remains Metal.

CUDA compiles the shared Metal operation contracts to CUDA C++ in the
`blanket-gpu-cuda` workspace crate, with explicit vector/bit-cast helpers and
64-bit integer accumulators. Its build script invokes nvcc to generate a
fatbinary embedded in the extension and wheel. Default native targets are
SM 75, 80, 86, 89, 90, 100, and 120. Set `BLANKET_CUDA_ARCHS` to a
comma-separated list of numeric compute capabilities to customize the build.
PTX for the lowest selected capability is also included for forward
compatibility; the NVIDIA driver may JIT that PTX on an architecture without
a matching native binary. Blanket's CUDA source is not compiled at runtime.
CuPy can still compile its own internal array utilities.
CuPy loads the embedded binary and launches those compiled pixel kernels;
Python manages shapes, controls, allocation, and exchange. All uint8, float32,
and unsigned working formats and resident operations have the same advertised
coverage. Float math is single precision with implicit FMA contraction disabled;
backend math-library and boundary rounding differences still need hardware
parity validation. There is no CPU fallback.

`from_torch` detects CUDA tensors automatically, validates HW/HWC/CHW layouts,
uint8/float32 dtype and disabled gradients, and imports through the DLPack
protocol. Storage offsets and strides are retained. `copy=True` makes an
independent GPU copy; noncontiguous inputs are packed on the GPU when an
operation needs them. Unsigned tensor import uses the existing `bit_depth`
argument and normalized float32 working storage.

Import while the producing PyTorch stream is current, or establish producer
completion before importing. DLPack negotiates a dependency onto the current
CuPy stream. Every operation records an event; subsequent operations wait on
their inputs even when the processing stream changes. Export enters the
recorded producer stream and lets DLPack establish the dependency onto the
consumer's current PyTorch stream. The protocol retains allocation ownership
without CPU pixel staging. Subsequent external mutations require application
synchronization or re-import on the new producer stream. Avoid concurrent
mutation while kernels read shared input storage. See the
[CuPy DLPack contract](https://docs.cupy.dev/en/stable/reference/generated/cupy.from_dlpack.html).

Results retain their asynchronous inputs and controls; dropping the last CUDA
resource waits for its event before releasing those owners. Closing a source
does not invalidate submitted work or exported tensors. This initial runtime
retains intermediate owners until results are released; bounded reuse and
fully asynchronous reclamation remain Phase 3 work. Downloads are explicit
and blocking. Driver initialization, unavailable devices, binary/architecture
failures, allocation errors, and execution failures report actionable diagnostics.
The first CUDA capability probe loads the precompiled kernels; later probes use a bounded
per-device module cache.

Run the shared Metal numeric contracts on NVIDIA hardware:

```sh
BLANKET_REQUIRE_CUDA=1 uv run --no-sync pytest tests/test_gpu_cuda.py -n 0 -ra -x
```

The CUDA wheel workflow runs on hosted Ubuntu for pushes, pull requests, and
manual dispatch. It installs the CUDA compiler, compiles the kernels, builds
the extension in the dev profile, and uploads the
Python 3.14 wheel as `blanket-cuda-dev-linux-x86_64-py314`, downloadable from
the run's artifacts for seven days. Kernel compilation errors fail the wheel
build. Building requires no NVIDIA GPU or driver. The workflow does not execute GPU tests.

For manual hardware testing, `BLANKET_REQUIRE_CUDA=1` makes missing GPU support
a failure rather than a skip. Local tests without CUDA report explicit skips;
host C++ compilation tests check source portability but cannot certify CUDA
device compilation or NVIDIA execution. GPU Phase 2's hardware parity and hardware CI
checkboxes remain pending NVIDIA execution coverage.

## Native Rust benchmarks

Compare Blanket CPU and GPU public operations with:

```sh
uv run --no-sync maturin develop --uv --release --features gpu-metal
uv run --no-sync scripts/benchmark_gpu.py --backend metal --sizes 256x256 1024x1024 --json gpu-results.json
uv run --no-sync scripts/benchmark_gpu.py --backend metal --runtime torch --operations resize blend pipeline
```

For NVIDIA, build with `--release --features gpu-cuda` using the CUDA toolkit,
then select `--backend cuda --device 0`. Use release builds for performance
measurements. The script supports uint8 L/LA/RGB/RGBA and reports median CPU
time, resident GPU time, transfer-inclusive GPU time, and CPU/GPU speedups for
both GPU paths. Ratios above 1 mean the GPU was faster.

Resident inputs are uploaded and synchronized before timing. Resident timings
include the public operation's control preparation, allocations, GPU execution,
and synchronization, without pixel upload/download. Transfer-inclusive timings
upload all required inputs, run the operation, download the output, and release
temporary GPU handles; binary operations upload both inputs. Pipeline results
remain on the GPU between operations and download only the final image.
Returned CPU images and resident GPU results are released outside their timing.
Fixtures and initialization are excluded; each path is warmed before sampling.
`--warmups`, `--repeats`, and `--iterations` control sampling; `--json` retains
samples, medians, speedups, actual GPU device/runtime, and environment metadata.
Output uses Rich tables with colored speedups by default; set `NO_RICH=1`
for plain text without importing Rich. Rendering happens outside timed samples.
These are synchronized wall-clock timings rather than isolated kernel timings.

Criterion suites exercise control preparation without a GPU and native Metal
execution without Python or PyTorch:

```sh
cargo bench -p blanket-gpu --bench preparation
cargo bench -p blanket-gpu-metal --bench metal
BLANKET_BENCH_FULL=1 cargo bench -p blanket-gpu-metal --bench metal
```

## Extending the backend

`blanket-gpu` defines descriptors, backend operations, resize/blur preparation,
and LUT control-data validation and quantization.
`blanket-gpu-metal` owns Metal allocations, command queues, and shaders.
`blanket-gpu-python` exposes native resources and shader preparation to Python.
Future backends implement the `Backend` trait and add Python device selection;
they do not need to change CPU raster storage. The CUDA crate lowers the shared
kernel contracts for build-time nvcc compilation, with CuPy loading embedded
fatbinaries and managing driver resources and DLPack.
Generic tensor exchange beyond the PyTorch adapter is not exposed yet.
