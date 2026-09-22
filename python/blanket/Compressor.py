"""Optional lossless optimization settings for Image.save()."""

from __future__ import annotations

from ._blanket import _LosslessImageCompressor

__all__ = ["LosslessImageCompressor"]


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
        if isinstance(effort, bool) or not isinstance(effort, int):
            raise TypeError("effort must be an integer")
        if not 1 <= effort <= 10:
            raise ValueError("effort must be between 1 and 10")
        self._native = _LosslessImageCompressor(effort=effort)

    @property
    def effort(self) -> int:
        return self._native.effort
