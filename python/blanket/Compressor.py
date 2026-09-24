"""Optional optimization settings for Image.save()."""

from __future__ import annotations

import math

from ._blanket import _LosslessImageCompressor, _LossyImageCompressor

__all__ = ["LosslessImageCompressor", "LossyImageCompressor"]


class LossyImageCompressor:
    """Search for smaller saves within an additional decoded-pixel error limit.

    ``max_rmse`` is the maximum RGB root-mean-square error relative to the
    normal save, on a 0-255 scale (including for high-bit-depth images).
    Alpha must match exactly. Hidden RGB values count toward the error.
    This is a sample error bound, not a perceptual quality guarantee.

    ``effort`` from 1 through 10 controls the number of candidate trials.
    PNG uses histogram-guided color quantization for 8-bit non-indexed images.
    JPEG, WebP, JPEG XL, AVIF, and HEIF/HEIC adaptively search lower codec
    quality settings using the measured error of each trial.
    Explicit ``lossless=True`` disables additional loss. Other formats and
    indexed or high-bit-depth PNG retain lossless optimization behavior.
    The normal save remains eligible, so output cannot grow relative to it.
    The source image is unchanged and this configuration is reusable.
    """

    def __init__(self, *, max_rmse: float = 2.0, effort: int = 7) -> None:
        """Configure LossyImageCompressor.

        Args:
            max_rmse: Maximum additional RGB root mean square error relative to the normal save, from 0 through 255. Alpha must match exactly.
            effort: Integer search effort from 1 (fastest) through 10 (most thorough); booleans are rejected.

        Examples:
            ```python
            from blanket import Image

            image = Image.new("RGB", (8, 8), (40, 100, 180))
            from io import BytesIO
            from blanket.Compressor import LossyImageCompressor

            compressor = LossyImageCompressor(effort=1)
            output = BytesIO()
            image.save(output, format="PNG", compressor=compressor)
            ```

        """
        if isinstance(max_rmse, bool) or not isinstance(max_rmse, (int, float)):
            raise TypeError("max_rmse must be a number")
        if not 0 <= max_rmse <= 255 or not math.isfinite(max_rmse):
            raise ValueError("max_rmse must be finite and between 0 and 255")
        if isinstance(effort, bool) or not isinstance(effort, int):
            raise TypeError("effort must be an integer")
        if not 1 <= effort <= 10:
            raise ValueError("effort must be between 1 and 10")
        self._native = _LossyImageCompressor(max_rmse=float(max_rmse), effort=effort)

    @property
    def max_rmse(self) -> float:
        """Configured maximum additional RGB error.

        Examples:
            ```python
            from blanket import Image

            image = Image.new("RGB", (8, 8), (40, 100, 180))
            from io import BytesIO
            from blanket.Compressor import LossyImageCompressor

            compressor = LossyImageCompressor(effort=1)
            print(compressor.max_rmse)
            ```

        """
        return self._native.max_rmse

    @property
    def effort(self) -> int:
        """Configured search effort.

        Examples:
            ```python
            from blanket import Image

            image = Image.new("RGB", (8, 8), (40, 100, 180))
            from io import BytesIO
            from blanket.Compressor import LossyImageCompressor

            compressor = LossyImageCompressor(effort=1)
            print(compressor.effort)
            ```

        """
        return self._native.effort


class LosslessImageCompressor:
    """Configure optimization for ``image.save(..., compressor=...)``.

    Args:
        effort: Optimization effort from 1 (fastest) to 10 (most thorough).
            Controls PNG filter search, JPEG scan coding, JXL effort search,
            and HEIF/HEIC encoder preset search. Higher settings can be slow.

    Eight-bit PNG saves try grayscale and opaque-alpha reductions and multiple
    row filters, including minimum-entropy filtering. Exact palettes use packed
    indices and try frequency and color ordering. Grayscale samples try packed
    1-, 2-, and 4-bit storage when exact; compatible binary alpha tries a PNG
    transparency key. Efforts 9-10 also search DEFLATE levels. The smallest
    encoding wins, including the normal baseline, without quantizing colors or
    discarding RGB values in transparent pixels.
    The image's mode and pixels stay unchanged; reopening a saved PNG may report
    a reduced mode. Size relative to the original source file is not guaranteed.
    JPEG saves optimize Huffman tables and, at effort 3 or higher, also try
    progressive scans without changing the normal save's DCT coefficients.
    JXL and HEIF/HEIC saves search encoder settings, including high-bit-depth
    images. Smaller candidates are accepted only when their decoded samples
    exactly match the normal save. For lossy saves, this also tries lossless
    encoding of the normal save's decoded pixels.

    Optimization introduces no additional loss relative to the requested save;
    it does not turn a lossy save into a pixel-exact copy of the input image.
    Opened images are encoded from their current pixels, not their source
    bitstream. Quality and lossless settings retain their normal meaning.
    High-bit-depth and indexed PNG and other formats keep normal encoding.
    Metadata follows normal save behavior. The object is reusable and affects
    only the save it is passed to, without modifying the image.

    """

    def __init__(self, *, effort: int = 7) -> None:
        """Configure LosslessImageCompressor.

        Args:
            effort: Integer search effort from 1 (fastest) through 10 (most thorough); booleans are rejected.

        Examples:
            ```python
            from blanket import Image

            image = Image.new("RGB", (8, 8), (40, 100, 180))
            from io import BytesIO
            from blanket.Compressor import LosslessImageCompressor

            compressor = LosslessImageCompressor(effort=1)
            output = BytesIO()
            image.save(output, format="PNG", compressor=compressor)
            ```

        """
        if isinstance(effort, bool) or not isinstance(effort, int):
            raise TypeError("effort must be an integer")
        if not 1 <= effort <= 10:
            raise ValueError("effort must be between 1 and 10")
        self._native = _LosslessImageCompressor(effort=effort)

    @property
    def effort(self) -> int:
        """Configured search effort.

        Examples:
            ```python
            from blanket import Image

            image = Image.new("RGB", (8, 8), (40, 100, 180))
            from io import BytesIO
            from blanket.Compressor import LosslessImageCompressor

            compressor = LosslessImageCompressor(effort=1)
            print(compressor.effort)
            ```

        """
        return self._native.effort
