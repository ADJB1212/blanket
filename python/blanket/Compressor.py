"""Optional optimization settings for Image.save()."""

from __future__ import annotations

__lazy_modules__ = {"math", "re"}

import math
import re

from ._blanket import _LosslessImageCompressor, _LossyImageCompressor

__all__ = ["LosslessImageCompressor", "LossyImageCompressor"]


def _validate_target_size(target_size: int | str | None) -> int | None:
    if isinstance(target_size, str):
        match = re.fullmatch(r"([0-9]+)(?:\.([0-9]+))?\s*(b|[kmgt]i?b)?", target_size.strip(), re.IGNORECASE)
        if match is None:
            raise ValueError("target_size must be a byte size such as '50kb' or '10MiB'")
        whole, fraction, unit = match.groups()
        unit = (unit or "b").lower()
        exponent = "kmgt".index(unit[0]) + 1 if unit != "b" else 0
        multiplier = (1024 if "i" in unit else 1000) ** exponent
        fraction = fraction or ""
        target_size = int(whole + fraction) * multiplier // (10 ** len(fraction))
    if target_size is not None:
        if isinstance(target_size, bool) or not isinstance(target_size, int):
            raise TypeError("target_size must be an integer, size string, or None")
        if target_size <= 0:
            raise ValueError("target_size must be positive")
    return target_size


class LossyImageCompressor:
    """Search for smaller saves within an additional decoded-pixel error limit.

    ``max_rmse`` is the maximum RGB root-mean-square error relative to the
    normal save, on a 0-255 scale (including for high-bit-depth images).
    Alpha must match exactly. Hidden RGB values count toward the error.
    ``metric`` optionally adds a perceptual score floor via ``min_score``.
    Choose ``"ssim"`` (-1 to 1), ``"ms_ssim"`` (0 to 1), or
    ``"ssimulacra2"`` (up to 100); higher scores mean greater similarity.
    Both settings must be supplied together. Scores use ImageMetrics' SDR
    sRGB assumptions and minimum dimensions (11, 161, and 8 respectively),
    relative to the normal save. Alpha still matches exactly, hidden RGB
    still counts, and ``max_rmse`` remains an independent limit. Use 255
    to allow the perceptual score to control color distortion alone.

    ``effort`` from 1 through 10 controls the number of candidate trials.
    PNG uses histogram-guided color quantization for 8-bit non-indexed images.
    JPEG, WebP, JPEG XL, AVIF, and HEIF/HEIC adaptively search lower codec
    quality settings using the measured error of each trial.
    Explicit ``lossless=True`` disables additional loss. Other formats and
    indexed, color-keyed, or high-bit-depth PNG retain lossless behavior.
    The normal save remains eligible, so output cannot grow relative to it.
    The source image is unchanged and this configuration is reusable.
    Metadata is always stripped, including explicit metadata options.

    ``target_size`` optionally caps the complete encoded output in bytes,
    after stripping metadata. The search still respects ``max_rmse``,
    ``min_score``, and ``effort``.
    If no searched candidate fits, saving raises ``ValueError`` before writing
    output, reporting the smallest qualifying candidate's size. This does not
    guarantee that unsearched settings could not meet the budget.
    """

    def __init__(self, *, max_rmse: float = 2.0, effort: int = 7, target_size: int | str | None = None, metric: str | None = None, min_score: float | None = None) -> None:
        """Configure LossyImageCompressor.

        Args:
            max_rmse: Maximum additional RGB root mean square error relative to the normal save, from 0 through 255. Alpha must match exactly.
            effort: Integer search effort from 1 (fastest) through 10 (most thorough); booleans are rejected.
            target_size: Positive byte budget as an integer or size string (such as "50kb" or "10MiB"), or None for no budget. Metadata is stripped.
            metric: Optional perceptual metric name: "ssim", "ms_ssim", or "ssimulacra2".
            min_score: Minimum acceptable perceptual score; required together with metric.

        Raises:
            TypeError: A numeric or string argument has the wrong type.
            ValueError: An effort, error limit, metric, score, or size budget is outside its allowed range.

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
        if metric is not None and not isinstance(metric, str):
            raise TypeError("metric must be a string or None")
        if metric is not None and metric not in {"ssim", "ms_ssim", "ssimulacra2"}:
            raise ValueError("metric must be ssim, ms_ssim, or ssimulacra2")
        if (metric is None) != (min_score is None):
            raise ValueError("metric and min_score must be specified together")
        if min_score is not None:
            if isinstance(min_score, bool) or not isinstance(min_score, (int, float)):
                raise TypeError("min_score must be a number")
            try:
                min_score = float(min_score)
            except OverflowError:
                raise ValueError("min_score must be finite and within the metric's score range") from None
            maximum = 100 if metric == "ssimulacra2" else 1
            minimum = -1 if metric == "ssim" else 0 if metric == "ms_ssim" else -math.inf
            if not minimum <= min_score <= maximum or not math.isfinite(min_score):
                raise ValueError("min_score must be finite and within the metric's score range")
        self._target_size = _validate_target_size(target_size)
        self._native = _LossyImageCompressor(max_rmse=float(max_rmse), effort=effort, metric=metric, min_score=min_score)

    @property
    def metric(self) -> str | None:
        """Configured perceptual metric, or None.

        Returns:
            ``ssim``, ``ms_ssim``, ``ssimulacra2``, or None when no metric is set.
        """
        return self._native.metric

    @property
    def min_score(self) -> float | None:
        """Configured minimum perceptual score, or None.

        Returns:
            The minimum accepted score for ``metric``, or None when no metric is set.
        """
        return self._native.min_score

    @property
    def target_size(self) -> int | None:
        """Configured byte budget after stripping metadata, or None.

        Returns:
            The maximum encoded size in bytes, or None when no budget is set.
        """
        return self._target_size

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

        Returns:
            The maximum additional decoded RGB error accepted by the search.
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

        Returns:
            The search effort, from 1 through 10.
        """
        return self._native.effort


class LosslessImageCompressor:
    """Configure optimization for ``image.save(..., compressor=...)``.

    Args:
        effort: Optimization effort from 1 (fastest) to 10 (most thorough).
            Controls PNG filter search, JPEG scan coding, JXL effort search,
            WebP method search, and HEIF/HEIC encoder preset search. Higher settings can be slow.

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
    WebP, JXL, and HEIF/HEIC saves search encoder settings. JXL and HEIF/HEIC
    also support high-bit-depth images. Smaller candidates are accepted only
    when their decoded samples
    exactly match the normal save. For lossy saves, this also tries lossless
    encoding of the normal save's decoded pixels.
    SVG saves remove formatting whitespace while preserving the traced paths.

    Optimization introduces no additional loss relative to the requested save;
    it does not turn a lossy save into a pixel-exact copy of the input image.
    Opened images are encoded from their current pixels, not their source
    bitstream. Quality and lossless settings retain their normal meaning.
    High-bit-depth, indexed, and color-keyed PNG keep normal encoding.
    Metadata is always stripped, including explicit metadata options. The object is reusable and affects
    only the save it is passed to, without modifying the image.

    ``target_size`` optionally caps the complete encoded output in bytes,
    after stripping metadata, without relaxing the fidelity guarantee. If no searched
    candidate fits, saving raises ``ValueError`` before writing output and
    reports the smallest candidate's size. Search coverage depends on ``effort``
    and format; formats without optimization try only the normal save.

    """

    def __init__(self, *, effort: int = 7, target_size: int | str | None = None) -> None:
        """Configure LosslessImageCompressor.

        Args:
            effort: Integer search effort from 1 (fastest) through 10 (most thorough); booleans are rejected.
            target_size: Positive byte budget as an integer or size string (such as "50kb" or "10MiB"), or None for no budget. Metadata is stripped.

        Raises:
            TypeError: ``effort`` is not an integer.
            ValueError: ``effort`` or ``target_size`` is outside its allowed range.

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
        self._target_size = _validate_target_size(target_size)
        self._native = _LosslessImageCompressor(effort=effort)

    @property
    def target_size(self) -> int | None:
        """Configured byte budget after stripping metadata, or None.

        Returns:
            The maximum encoded size in bytes, or None when no budget is set.
        """
        return self._target_size

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

        Returns:
            The search effort, from 1 through 10.
        """
        return self._native.effort
