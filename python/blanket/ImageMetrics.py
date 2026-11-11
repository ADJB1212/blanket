"""Native full-reference perceptual distortion metrics for SDR images.

Compare current frames with matching dimensions. Supported modes are L, LA,
RGB, RGBA (8/10/12/16-bit where available), I;16 variants, F, RGB;F, and RGBA;F.
Integer samples are normalized by their depth; float color samples must be
finite and in [0, 1]. Grayscale is replicated to RGB. Inputs may differ in mode
and depth. Other modes require explicit conversion.

Alpha is ignored, including hidden color samples. Composite both images onto
the same background first to score their displayed appearance. Samples are
assumed to be SDR sRGB; no ICC, cICP, gamma, or EXIF orientation is applied.
SSIM and MS-SSIM operate on encoded channels; SSIMULACRA2 linearizes sRGB and
uses XYB internally. Pixels, metadata, and frame positions are unchanged.
"""

from __future__ import annotations

from . import Image
from ._blanket import image_metric

__all__ = ["ms_ssim", "ssim", "ssimulacra2"]


def _score(reference: Image.Image, distorted: Image.Image, algorithm: str) -> float:
    if not isinstance(reference, Image.Image) or not isinstance(distorted, Image.Image):
        raise TypeError("reference and distorted must be Blanket Images")
    return image_metric(reference._native, distorted._native, algorithm)


def ssim(reference: Image.Image, distorted: Image.Image) -> float:
    """Return mean structural similarity, from -1 to 1 (higher is better).

    Use an 11x11 Gaussian window with sigma 1.5, population covariance,
    K1=0.01 and K2=0.03. Average valid window centers, then RGB channels.
    Both dimensions must be at least 11. Identical images score 1.
    No automatic pre-downsampling is applied.
    """
    return _score(reference, distorted, "ssim")


def ms_ssim(reference: Image.Image, distorted: Image.Image) -> float:
    """Return five-scale structural similarity, from 0 to 1 (higher is better).

    Use SSIM's window/statistics at each scale and weights
    (0.0448, 0.2856, 0.3001, 0.2363, 0.1333). Multiply contrast-structure
    terms at the first four scales and SSIM at the last, per channel, then
    average RGB. Negative terms are clamped to zero before exponentiation.
    Downsample with 2x2 averages, replicating odd right/bottom edges.
    Both dimensions must be at least 161. Identical images score 1.
    """
    return _score(reference, distorted, "ms_ssim")


def ssimulacra2(reference: Image.Image, distorted: Image.Image) -> float:
    """Return SSIMULACRA2 quality (higher is better; identical images score 100).

    The reference is the original and distorted is its modified/encoded copy;
    this metric is asymmetric. Very poor matches can have negative scores.
    Both dimensions must be at least 8. Uses the native Rust ssimulacra2
    implementation; floating-point results may vary slightly by platform.
    """
    return _score(reference, distorted, "ssimulacra2")
