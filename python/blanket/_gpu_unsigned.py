"""Unsigned resident processing using normalized float32 working samples."""

from __future__ import annotations

__lazy_modules__ = {"torch", "collections.abc", "concurrent.futures"}

from typing import TYPE_CHECKING

from . import Image, _blanket

if TYPE_CHECKING:
    from collections.abc import Sequence
    from concurrent.futures import Future

    from torch import Tensor

    from ._gpu_async import Event
    from .ImageGpu import GpuImage, GpuReduction


class GpuUnsignedImage:
    """A resident 10-, 12-, or 16-bit L, RGB, or RGBA image.

    Create unsigned images with ImageGpu.from_image or from_torch(bit_depth=...).
    Working samples are normalized float32 values, preserving every input
    unsigned integer exactly. Each public image operation clips to 0..1 and
    quantizes once to the original depth, rounding half upward. Filters process
    alpha as documented by their underlying GpuImage operation.

    Results are independent resident images. CPU export restores unsigned
    sample values; tensor export and statistics use normalized working units.
    """

    def __init__(self, image: GpuImage, mode: str, depth: int) -> None:
        """Wrap normalized working storage with an unsigned mode and depth.

        Args:
            image: F, RGB;F, or RGBA;F working image matching the unsigned mode.
            mode: Public image mode: L, RGB, or RGBA.
            depth: Original unsigned sample depth: 10, 12, or 16.

        Use ImageGpu.from_image or from_torch for validated public construction.
        """
        self._image = image
        self.mode = mode
        self.bit_depth = depth
        self.dtype = "uint16"

    @property
    def size(self) -> tuple[int, int]:
        """Image dimensions as (width, height)."""
        return self._image.size

    @property
    def info(self) -> dict[str, object]:
        """Mutable metadata dictionary associated with this image handle."""
        return self._image.info

    @property
    def runtime(self) -> str:
        """Execution runtime: 'native', 'torch', or 'cuda'."""
        return self._image.runtime

    @property
    def device(self) -> str:
        """Execution device identifier: 'metal', 'mps', or 'cuda:<index>'."""
        return self._image.device

    def _result(self, image: GpuImage, mode: str | None = None) -> GpuUnsignedImage:
        """Quantize working samples once and retain this image's unsigned depth."""
        image = image._run(30, values=[float((1 << self.bit_depth) - 1)])
        return GpuUnsignedImage(image, self.mode if mode is None else mode, self.bit_depth)

    def _other(self, image: GpuUnsignedImage) -> GpuImage:
        """Return working storage after checking a peer's unsigned mode and depth."""
        if (image.mode, image.bit_depth) != (self.mode, self.bit_depth):
            raise ValueError("unsigned GPU images must match mode and depth")
        return image._image

    def to_image(self) -> Image.Image:
        """Wait for completion and export original-depth unsigned CPU samples.

        Returns:
            An independent Image in the original mode and 10/12/16-bit depth,
            with copied metadata. Normalized working values are clipped and
            rounded half upward to the original integer range.
        """
        if self.runtime == "native":
            self._image._check()
            result = Image.Image(self._image._resource.download_unsigned_image(self.mode, self.bit_depth))
            result.info.update(self.info)
            return result
        working = self._image.to_image()
        data = _blanket._gpu_unsigned_bytes(working.tobytes(), self.bit_depth, True)
        result = Image.frombytes(self.mode, self.size, data, bit_depth=self.bit_depth)
        result.info.update(self.info)
        return result

    def synchronize(self) -> None:
        """Wait for this image's pending GPU work without downloading pixels."""
        self._image.synchronize()

    def record_event(self) -> Event:
        """Retain this image's completion point without blocking the CPU.

        Returns:
            An Event that remains valid after this image handle closes.
        """
        return self._image.record_event()

    def to_image_async(self) -> Future[Image.Image]:
        """Export original-depth CPU pixels on a transfer worker.

        Returns:
            A Future resolving to an Image as described by to_image(). The
            operation retains storage after this handle closes; result() waits
            for completion and reports errors.

        Raises:
            ValueError: The image uses the PyTorch MPS runtime.
        """
        from ._gpu_async import submit

        if self.runtime == "torch":
            raise ValueError("asynchronous transfers require native Metal or CUDA")
        retained = GpuUnsignedImage(self._image.share(), self.mode, self.bit_depth)
        return submit(retained.to_image, device=self.device)

    def close(self) -> None:
        """Release this handle; queued operations and retained transfers stay valid."""
        self._image.close()

    def copy(self) -> GpuUnsignedImage:
        """Copy quantized samples into independent resident storage.

        Returns:
            A GpuUnsignedImage with the same mode, size, depth, and metadata.
        """
        return self._result(self._image._run(6))

    def resize(self, size: tuple[int, int], resample: int = Image.Resampling.BICUBIC, *, linear_light: bool = False) -> GpuUnsignedImage:
        """Resize normalized samples and quantize to the original unsigned depth.

        Args:
            size: Positive output (width, height).
            resample: Any Image.Resampling filter; defaults to BICUBIC.
            linear_light: Decode sRGB color before filtering and encode afterward.
                Requires RGB or RGBA input.

        Returns:
            A resident image in the original mode/depth. RGBA filtering uses
            premultiplied alpha; float working precision can differ from CPU
            integer filtering.
        """
        return self._result(self._image.resize(size, resample, linear_light=linear_light))

    def crop(self, box: tuple[int, int, int, int]) -> GpuUnsignedImage:
        """Copy a resident rectangle, filling outside-source samples with zero.

        Args:
            box: (left, top, right, bottom), with right/bottom exclusive.
                Bounds must fit signed 32-bit integers and give positive dimensions.

        Returns:
            An independent unsigned image of the requested size and original depth.
        """
        return self._result(self._image.crop(box))

    def transpose(self, method: int) -> GpuUnsignedImage:
        """Flip or rotate resident samples without interpolation.

        Args:
            method: An Image.Transpose constant.

        Returns:
            An independent unsigned image; quarter-turn rotations and diagonal
            transposes swap its width and height.
        """
        return self._result(self._image.transpose(method))

    def offset(self, x: int, y: int = 0) -> GpuUnsignedImage:
        """Translate resident pixels with wraparound.

        Args:
            x: Integer horizontal offset, positive to the right.
            y: Integer vertical offset, positive downward; defaults to zero.

        Returns:
            An independent unsigned image with unchanged size, mode, and depth.
        """
        return self._result(self._image.offset(x, y))

    def invert(self) -> GpuUnsignedImage:
        """Invert every sample, including alpha, in the unsigned sample range.

        Returns:
            A resident image containing (2**bit_depth - 1) minus each original
            unsigned value, with mode and depth preserved.
        """
        return self._result(self._image.invert())

    def brightness(self, factor: float) -> GpuUnsignedImage:
        """Scale normalized color samples and quantize to the original depth.

        Args:
            factor: Finite multiplier; 0 gives black and 1 preserves color.

        Returns:
            An independent unsigned image. RGBA alpha is preserved; color samples
            saturate to the original unsigned range.
        """
        return self._result(self._image.brightness(factor))

    def color(self, factor: float) -> GpuUnsignedImage:
        """Adjust saturation around per-pixel luminance.

        Args:
            factor: Finite blend factor; 0 gives grayscale color, 1 preserves
                color, and larger values increase saturation.

        Returns:
            A quantized unsigned image with alpha preserved. L input is copied.
        """
        return self._result(self._image.color(factor))

    def contrast(self, factor: float) -> GpuUnsignedImage:
        """Adjust contrast around mean luminance computed on the GPU.

        Args:
            factor: Finite blend factor; 0 gives the mean, 1 preserves color,
                and larger values increase contrast.

        Returns:
            A quantized unsigned image with alpha preserved, in the original depth.
        """
        return self._result(self._image.contrast(factor))

    def sharpness(self, factor: float) -> GpuUnsignedImage:
        """Blend against a 3x3 smooth kernel and quantize the result.

        Args:
            factor: Finite blend factor; 0 smooths, 1 preserves the original,
                and values above 1 sharpen.

        Returns:
            An unsigned image retaining border samples and RGBA alpha. Images
            smaller than 3 pixels on either axis are copied.
        """
        return self._result(self._image.sharpness(factor))

    def box_blur(self, radius: float | tuple[float, float]) -> GpuUnsignedImage:
        """Apply a fractional box blur to all channels, including alpha.

        Args:
            radius: Finite nonnegative radius, or (horizontal, vertical) radii.

        Returns:
            A resident unsigned image with replicated edges. Working passes use
            float32; quantization occurs after the complete blur.
        """
        return self._result(self._image.box_blur(radius))

    def gaussian_blur(self, radius: float | tuple[float, float]) -> GpuUnsignedImage:
        """Approximate Gaussian blur with three box passes per axis.

        Args:
            radius: Finite nonnegative Gaussian radius, or separate axis radii.

        Returns:
            A resident unsigned image with replicated edges and filtered alpha.
            Quantization occurs after all float32 passes.
        """
        return self._result(self._image.gaussian_blur(radius))

    def rank_filter(self, size: int, rank: int) -> GpuUnsignedImage:
        """Select a sorted neighborhood sample in every channel.

        Args:
            size: Square window width: 1, 3, 5, or 7.
            rank: Zero-based rank smaller than size*size.

        Returns:
            A resident unsigned image with replicated edges, including alpha.
        """
        return self._result(self._image.rank_filter(size, rank))

    def min_filter(self, size: int = 3) -> GpuUnsignedImage:
        """Select each channel's neighborhood minimum, including alpha.

        Args:
            size: Odd square window width from 1 through 7.

        Returns:
            An unsigned image with replicated edges and unchanged size/depth.
        """
        return self.rank_filter(size, 0)

    def max_filter(self, size: int = 3) -> GpuUnsignedImage:
        """Select each channel's neighborhood maximum, including alpha.

        Args:
            size: Odd square window width from 1 through 7.

        Returns:
            An unsigned image with replicated edges and unchanged size/depth.
        """
        return self.rank_filter(size, size * size - 1)

    def median_filter(self, size: int = 3) -> GpuUnsignedImage:
        """Select each channel's neighborhood median, including alpha.

        Args:
            size: Odd square window width from 1 through 7.

        Returns:
            An unsigned image with replicated edges and unchanged size/depth.
        """
        return self.rank_filter(size, size * size // 2)

    def unsharp_mask(self, radius: float = 2, percent: int = 150, threshold: float = 3) -> GpuUnsignedImage:
        """Sharpen against a Gaussian blur using an unsigned-unit threshold.

        Args:
            radius: Finite nonnegative Gaussian blur radius.
            percent: Signed 32-bit integer percentage of the difference to add.
            threshold: Finite cutoff in original unsigned sample units.

        Returns:
            A resident image quantized to the original depth after sharpening.
            Alpha is processed as a channel.
        """
        return self._result(self._image.unsharp_mask(radius, percent, threshold / ((1 << self.bit_depth) - 1)))

    def convolve(self, kernel: Sequence[float], *, scale: float | None = None, offset: float = 0) -> GpuUnsignedImage:
        """Apply a 3x3 convolution and quantize to the original depth.

        Args:
            kernel: Nine finite coefficients in ImageFilter.Kernel order.
            scale: Finite nonzero divisor; None uses sum(kernel), or 1 if zero.
            offset: Finite addition in original unsigned sample units.

        Returns:
            A resident unsigned image. All channels, including alpha, are filtered;
            border samples are retained.
        """
        return self._result(self._image.convolve(kernel, scale=scale, offset=offset / ((1 << self.bit_depth) - 1)))

    def blend(self, other: GpuUnsignedImage, alpha: float) -> GpuUnsignedImage:
        """Interpolate two matching unsigned images.

        Args:
            other: Image matching mode, depth, size, runtime, and device.
            alpha: Finite weight in self + (other-self)*alpha; may extrapolate.

        Returns:
            A resident image with every channel blended and quantized to the
            original depth, including alpha.
        """
        return self._result(self._image.blend(self._other(other), alpha))

    def alpha_composite(self, source: GpuUnsignedImage, *, linear_light: bool = False) -> GpuUnsignedImage:
        """Composite an unsigned source over this background using straight alpha.

        Args:
            source: RGBA image matching mode, depth, size, runtime, and device.
            linear_light: Decode sRGB color for compositing and encode afterward.

        Returns:
            A resident RGBA image quantized to the original unsigned depth.
        """
        return self._result(self._image.alpha_composite(self._other(source), linear_light=linear_light))

    def paste(self, source: GpuUnsignedImage, box: tuple[int, int] = (0, 0), mask: GpuImage | None = None) -> GpuUnsignedImage:
        """Paste a matching unsigned source through an optional resident mask.

        Args:
            source: Image matching mode, depth, runtime, and device; size may differ.
            box: Integer (x, y) destination for the source's top-left corner.
            mask: L mask in 0..255 units or F mask in 0..1 units, matching the
                source size and context. None uses full coverage. NaN coverage
                is zero; source alpha does not automatically provide coverage.

        Returns:
            An image of the background size, with every band interpolated and
            quantized to the original depth. Out-of-bounds source regions clip.
        """
        return self._result(self._image.paste(self._other(source), box, mask))

    def to_torch(self, *, layout: str = "HWC") -> Tensor:
        """Export normalized working samples to PyTorch without a CPU download.

        Args:
            layout: "HWC", "CHW", or "HW"; HW requires L mode.

        Returns:
            A float32 tensor in 0..1 units, rather than uint16 sample storage.
            Native Metal makes an independent synchronized GPU copy. Tensor
            runtimes share storage, which must remain immutable during GPU reads.

        Raises:
            ValueError: The layout is invalid or the image is closed.
        """
        return self._image.to_torch(layout=layout)

    def getchannel(self, channel: int | str) -> GpuUnsignedImage:
        """Copy a resident band without changing the unsigned sample range.

        Args:
            channel: Zero-based index or band name present in the image.

        Returns:
            An independent L-mode GpuUnsignedImage with the original depth.
        """
        return self._result(self._image.getchannel(channel), "L")

    def split(self) -> tuple[GpuUnsignedImage, ...]:
        """Copy every band into independent unsigned grayscale images.

        Returns:
            A tuple of L-mode GpuUnsignedImage objects in band order, all retaining
            the original size and sample depth.
        """
        return tuple(self._result(band, "L") for band in self._image.split())

    def transform(self, size: tuple[int, int], method: int, data: Sequence[float], resample: int = Image.Resampling.NEAREST) -> GpuUnsignedImage:
        """Apply inverse affine or perspective mapping to unsigned samples.

        Args:
            size: Positive output (width, height).
            method: Image.Transform.AFFINE or Image.Transform.PERSPECTIVE.
            data: Six affine coefficients or eight perspective coefficients,
                with the same pixel-center convention as GpuImage.transform.
            resample: NEAREST, BILINEAR, or BICUBIC.

        Returns:
            A resident image quantized to the original depth. Out-of-bounds
            coordinates fill with zero; alpha is sampled independently.
        """
        return self._result(self._image.transform(size, method, data, resample))

    def rotate(self, angle: float, resample: int = Image.Resampling.NEAREST, *, expand: bool = False) -> GpuUnsignedImage:
        """Rotate unsigned samples around the image center.

        Args:
            angle: Finite counterclockwise angle in degrees.
            resample: NEAREST, BILINEAR, or BICUBIC.
            expand: Enlarge the canvas to contain the rotated image.

        Returns:
            A resident unsigned image with zero fill outside the source.
            Interpolated samples, including alpha, quantize to the original depth.
        """
        return self._result(self._image.rotate(angle, resample, expand=expand))

    def remap(self, coordinates: GpuImage, resample: int = Image.Resampling.NEAREST) -> GpuUnsignedImage:
        """Sample unsigned pixels through a reusable resident coordinate map.

        Args:
            coordinates: RGB;F image in the same context. Red/green contain
                source x/y pixel-center coordinates; blue is ignored.
            resample: NEAREST, BILINEAR, or BICUBIC.

        Returns:
            An unsigned image sized like the map and quantized to the original
            depth. Nonfinite/out-of-bounds coordinates fill with zero.
        """
        return self._result(self._image.remap(coordinates, resample))

    def statistics(self) -> GpuReduction:
        """Compute resident statistics in normalized working sample units.

        Returns:
            A GpuReduction with per-band extrema and moments, including alpha.
            to_cpu() reports values in 0..1 units, not original unsigned integer
            units; accumulation uses float32.
        """
        return self._image.statistics()
