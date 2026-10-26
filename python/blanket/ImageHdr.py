"""HDR rendering with explicit transfer functions and SDR sRGB output."""

from __future__ import annotations

from . import Image
from ._blanket import hdr_tone_map

__all__ = ["tone_map"]


def tone_map(
    image: Image.Image,
    operator: str = "reinhard",
    *,
    transfer: str | None = None,
    primaries: str | None = None,
    exposure: float = 0.0,
    reference_white: float = 203.0,
    peak_luminance: float = 1000.0,
    bit_depth: int = 8,
) -> Image.Image:
    """Render the current frame to SDR using Reinhard, ACES, or filmic curves.

    Args:
        image: L, LA, RGB, RGBA, F, or I;16 image. Integer samples are
            normalized by their depth; F samples use 1.0 as reference white.
        operator: ``reinhard``, ``aces`` (Narkowicz fit), or ``filmic``
            (Hable curve with an 11.2 white point), applied per color channel.
        transfer: ``linear``, ``srgb``, ``pq``, or ``hlg``. Defaults to cICP
            metadata, or linear for untagged images. PQ uses absolute nits;
            HLG uses the BT.2100 display OOTF and peak-dependent system gamma.
        primaries: ``srgb`` or ``bt2020``; defaults to cICP or sRGB.
            BT.2020 is transformed to linear sRGB before tone mapping.
        exposure: Exposure adjustment in stops, applied in linear light.
        reference_white: Reference white in nits for PQ/HLG normalization.
        peak_luminance: HLG display peak in nits (400 to 10000).
        bit_depth: Output depth, 8 or 16 (LA output supports only 8).

    Negative and NaN color samples become black; positive infinity becomes
    white. Out-of-gamut negative channels are clipped. Alpha is scaled to
    output depth without a transfer function. Output uses sRGB encoding;
    stale ICC, HDR, and gamma metadata is removed. No ICC transform is applied.
    Source pixels, metadata, and frame position are unchanged.
    """
    if image.mode not in ("L", "LA", "RGB", "RGBA", "F", "I;16", "I;16L", "I;16B"):
        raise ValueError("tone mapping requires L, LA, RGB, RGBA, F, or I;16 pixels")
    cicp = image.info.get("cicp")
    if cicp is not None:
        if not isinstance(cicp, (tuple, list)) or len(cicp) != 4 or cicp[2:] not in ((0, 1), [0, 1]):
            raise ValueError("tone mapping requires full-range RGB cICP metadata")
        if transfer is None:
            transfer = {8: "linear", 13: "srgb", 16: "pq", 18: "hlg"}.get(cicp[1])
            if transfer is None:
                raise ValueError("unsupported cICP transfer; specify transfer explicitly")
        if primaries is None:
            primaries = {1: "srgb", 9: "bt2020"}.get(cicp[0])
            if primaries is None:
                raise ValueError("unsupported cICP primaries; specify primaries explicitly")
    if not 400 <= peak_luminance <= 10000:
        raise ValueError("peak_luminance must be between 400 and 10000 nits")
    result = Image.Image(
        hdr_tone_map(image._native, operator, "linear" if transfer is None else transfer, "srgb" if primaries is None else primaries, exposure, reference_white, peak_luminance, bit_depth)
    )
    stale = {"cicp", "mastering_display", "content_light_level", "icc_profile", "gamma", "chromaticity", "srgb"}
    result.info.update({key: value for key, value in image._metadata_info().items() if key not in stale})
    result.info["cicp"] = (1, 13, 0, 1)
    return result
