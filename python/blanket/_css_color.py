"""Parse absolute CSS Color 4 functions into 8-bit sRGB values."""

from __future__ import annotations

__lazy_modules__ = {"colorsys"}

import colorsys
import re
from math import cbrt, copysign, cos, dist, isfinite, pi, sin

# sRGB transfer-function breakpoints, linear slope, offset, and exponent.
_SRGB_ENCODED_BREAK = 0.04045
_SRGB_LINEAR_BREAK = 0.0031308
_SRGB_SLOPE = 12.92
_SRGB_OFFSET = 0.055
_SRGB_SCALE = 1.055
_SRGB_GAMMA = 2.4
# Rec.2020 uses a separate display exponent with the same numeric value.
_REC2020_GAMMA = 2.4
# CSS local MINDE stops below a just-noticeable Oklab color difference.
_GAMUT_JND = 0.02
_GAMUT_EPSILON = 0.0001
# Allow conversion roundoff at gamut edges and bound binary-search work.
_GAMUT_EDGE_EPSILON = 1e-7
_GAMUT_MAX_ITERATIONS = 64

_NUMBER = re.compile(r"([+-]?(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+)(?:e[+-]?[0-9]+)?)(%|deg|grad|rad|turn)?")
_FUNCTION = re.compile(r"([a-z]+)\((.*)\)", re.DOTALL)
_XYZ_TO_RGB = ((3.2409699419, -1.5373831776, -0.4986107603), (-0.9692436363, 1.8759675015, 0.0415550574), (0.0556300797, -0.2039769589, 1.0569715142))
_D50_TO_D65 = ((0.9554734215, -0.0230984549, 0.0632592432), (-0.0283697093, 1.0099953981, 0.0210414412), (0.0123140149, -0.0205076493, 1.3303659262))
_TO_XYZ = {
    "display-p3": ((0.4865709486, 0.2656676932, 0.1982172852), (0.2289745641, 0.6917385218, 0.0792869141), (0.0, 0.0451133819, 1.0439443689)),
    "a98-rgb": ((0.5766690429, 0.1855582379, 0.1882286462), (0.2973449753, 0.6273635663, 0.0752914585), (0.0270313614, 0.0706888525, 0.9913375368)),
    "prophoto-rgb": ((0.7977666449, 0.1351812974, 0.0313477341), (0.2880748288, 0.7118352342, 0.0000899369), (0.0, 0.0, 0.8251046025)),
    "rec2020": ((0.6369580483, 0.1446169036, 0.1688809752), (0.2627002120, 0.6779980715, 0.0593017165), (0.0, 0.0280726930, 1.0609850577)),
}


def _clamp(value: float, upper: float = 1) -> float:
    """Clamp a value to the inclusive range from zero to upper."""
    return max(0.0, min(upper, value))


def _byte(value: float) -> int:
    """Clamp and round a normalized sample to an 8-bit integer."""
    return int(_clamp(value) * 255 + 0.5)


def _number(component: str, scale: float = 1, *, hue: bool = False) -> float:
    """Parse a CSS component, scaling percentages or normalizing hue to degrees."""
    if component == "none":
        return 0.0
    match = _NUMBER.fullmatch(component)
    if match is None:
        raise ValueError("invalid color component")
    value = float(match[1])
    if not isfinite(value):
        raise ValueError("color components must be finite")
    unit = match[2]
    if hue:
        if unit == "%":
            raise ValueError("hue cannot be a percentage")
        period = {None: 360, "deg": 360, "grad": 400, "rad": 2 * pi, "turn": 1}[unit]
        return (value % period) / period * 360
    if unit not in (None, "%"):
        raise ValueError("unexpected angle unit")
    return value / 100 * scale if unit == "%" else value


def _matrix(matrix: tuple[tuple[float, ...], ...], values: tuple[float, ...]) -> tuple[float, ...]:
    """Multiply a color conversion matrix by a component vector."""
    return tuple(sum(a * b for a, b in zip(row, values, strict=True)) for row in matrix)


def _linear(value: float) -> float:
    """Decode an sRGB sample to linear light, preserving its sign."""
    return value / _SRGB_SLOPE if abs(value) <= _SRGB_ENCODED_BREAK else copysign(((abs(value) + _SRGB_OFFSET) / _SRGB_SCALE) ** _SRGB_GAMMA, value)


def _encoded(value: float) -> float:
    """Encode a linear-light sample as sRGB, preserving its sign."""
    return _SRGB_SLOPE * value if abs(value) <= _SRGB_LINEAR_BREAK else copysign(_SRGB_SCALE * abs(value) ** (1 / _SRGB_GAMMA) - _SRGB_OFFSET, value)


def _xyz_rgb(xyz: tuple[float, ...], *, d50: bool = False) -> tuple[float, ...]:
    """Convert XYZ to encoded sRGB, optionally adapting D50 to D65."""
    if d50:
        xyz = _matrix(_D50_TO_D65, xyz)
    return tuple(_encoded(v) for v in _matrix(_XYZ_TO_RGB, xyz))


def _oklab_rgb(lab: tuple[float, ...]) -> tuple[float, ...]:
    """Convert Oklab components to encoded sRGB without gamut clipping."""
    light, a, b = lab
    long = (light + 0.3963377774 * a + 0.2158037573 * b) ** 3
    m = (light - 0.1055613458 * a - 0.0638541728 * b) ** 3
    s = (light - 0.0894841775 * a - 1.2914855480 * b) ** 3
    return tuple(
        _encoded(v)
        for v in (4.0767416621 * long - 3.3077115913 * m + 0.2309699292 * s, -1.2684380046 * long + 2.6097574011 * m - 0.3413193965 * s, -0.0041960863 * long - 0.7034186147 * m + 1.7076147010 * s)
    )


def _rgb_oklab(rgb: tuple[float, ...]) -> tuple[float, ...]:
    """Convert encoded sRGB components to Oklab."""
    r, g, b = (_linear(v) for v in rgb)
    long = cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b)
    m = cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b)
    s = cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b)
    return 0.2104542553 * long + 0.7936177850 * m - 0.0040720468 * s, 1.9779984951 * long - 2.4285922050 * m + 0.4505937099 * s, 0.0259040371 * long + 0.7827717662 * m - 0.8086757660 * s


def _gamut_map(rgb: tuple[float, ...]) -> tuple[float, ...]:
    """Reduce Oklab chroma using CSS local MINDE before quantizing to sRGB."""
    if not all(isfinite(v) for v in rgb):
        raise ValueError("color conversion overflow")
    if all(-_GAMUT_EDGE_EPSILON <= v <= 1 + _GAMUT_EDGE_EPSILON for v in rgb):
        return rgb
    light, a, b = _rgb_oklab(rgb)
    if light >= 1:
        return 1.0, 1.0, 1.0
    if light <= 0:
        return 0.0, 0.0, 0.0
    clipped = tuple(_clamp(v) for v in rgb)
    if dist(_rgb_oklab(clipped), (light, a, b)) < _GAMUT_JND:
        return clipped
    low, high = 0.0, 1.0
    in_gamut = True
    chroma = (a * a + b * b) ** 0.5
    for _ in range(_GAMUT_MAX_ITERATIONS):
        if (high - low) * chroma <= _GAMUT_EPSILON:
            break
        fraction = (low + high) / 2
        lab = (light, a * fraction, b * fraction)
        candidate = _oklab_rgb(lab)
        if in_gamut and all(0 <= v <= 1 for v in candidate):
            low = fraction
            continue
        clipped = tuple(_clamp(v) for v in candidate)
        error = dist(_rgb_oklab(clipped), lab)
        if error < _GAMUT_JND:
            if _GAMUT_JND - error < _GAMUT_EPSILON:
                return clipped
            in_gamut = False
            low = fraction
        else:
            high = fraction
    return clipped


def _predefined(space: str, tokens: list[str]) -> tuple[float, ...]:
    """Convert CSS color() components from a predefined space to encoded sRGB."""
    values = tuple(_number(v) for v in tokens)
    if space == "srgb":
        return values
    if space == "srgb-linear":
        return tuple(_encoded(v) for v in values)
    if space in ("xyz", "xyz-d65", "xyz-d50"):
        return _xyz_rgb(values, d50=space == "xyz-d50")
    if space in ("display-p3", "display-p3-linear"):
        linear = tuple(_linear(v) for v in values) if space == "display-p3" else values
        return _xyz_rgb(_matrix(_TO_XYZ["display-p3"], linear))
    if space == "a98-rgb":
        linear = tuple(copysign(abs(v) ** (563 / 256), v) for v in values)
    elif space == "prophoto-rgb":
        linear = tuple(v / 16 if abs(v) <= 16 / 512 else copysign(abs(v) ** 1.8, v) for v in values)
    elif space == "rec2020":
        linear = tuple(copysign(abs(v) ** _REC2020_GAMMA, v) for v in values)
    else:
        raise ValueError("unknown color space")
    return _xyz_rgb(_matrix(_TO_XYZ[space], linear), d50=space == "prophoto-rgb")


def _lab_rgb(model: str, tokens: list[str]) -> tuple[float, ...]:
    """Convert CSS Lab, LCH, Oklab, or Oklch components to encoded sRGB."""
    ok = model.startswith("ok")
    polar = model.endswith("lch")
    light = _clamp(_number(tokens[0], 1 if ok else 100), 1 if ok else 100)
    a = _number(tokens[1], 0.4 if ok else (150 if polar else 125))
    b = _number(tokens[2], hue=True) if polar else _number(tokens[2], 0.4 if ok else 125)
    if polar:
        chroma, angle = max(0, a), b * pi / 180
        a, b = chroma * cos(angle), chroma * sin(angle)
    if ok:
        return _oklab_rgb((light, a, b))
    fy = (light + 16) / 116
    f = (fy + a / 500, fy, fy - b / 200)
    xyz = tuple((v**3 if v > 6 / 29 else (116 * v - 16) * 27 / 24389) * white for v, white in zip(f, (0.9642956764, 1.0, 0.8251046025), strict=True))
    return _xyz_rgb(xyz, d50=True)


def functional_rgb(value: str) -> tuple[int, ...]:
    """Parse a CSS color function into gamut-mapped 8-bit RGB or RGBA samples."""
    match = _FUNCTION.fullmatch(value)
    if match is None:
        raise ValueError("invalid color function")
    model, body = match.groups()
    legacy = "," in body
    alpha: str | None = None
    if legacy:
        if "/" in body or model not in ("rgb", "rgba", "hsl", "hsla", "hsv", "hsb"):
            raise ValueError("invalid comma syntax")
        tokens = [v.strip() for v in body.split(",")]
        if len(tokens) == 4 and model in ("rgb", "rgba", "hsl", "hsla"):
            alpha = tokens.pop()
        if "none" in tokens or alpha == "none":
            raise ValueError("none requires modern syntax")
    else:
        parts = body.split("/")
        if len(parts) > 2:
            raise ValueError("invalid alpha separator")
        tokens = parts[0].split()
        if len(parts) == 2:
            alpha = parts[1].strip()
    space = tokens.pop(0) if model == "color" and tokens else ""
    if len(tokens) != 3:
        raise ValueError("expected three color components")
    if model in ("rgb", "rgba"):
        if legacy and len({v.endswith("%") for v in tokens}) != 1:
            raise ValueError("legacy RGB components must use the same units")
        rgb = tuple(_clamp(_number(v, 255) / 255) for v in tokens)
    elif model in ("hsl", "hsla", "hwb", "hsv", "hsb"):
        if legacy and any(not v.endswith("%") for v in tokens[1:]):
            raise ValueError("expected percentage components")
        hue = _number(tokens[0], hue=True) / 360
        x, y = (max(0, _number(v, 100) / 100) for v in tokens[1:])
        if model in ("hsl", "hsla"):
            rgb = _gamut_map(colorsys.hls_to_rgb(hue, _clamp(y), x))
        elif model == "hwb":
            rgb = (x / (x + y),) * 3 if x + y >= 1 else tuple(v * (1 - x - y) + x for v in colorsys.hsv_to_rgb(hue, 1, 1))
        else:
            rgb = colorsys.hsv_to_rgb(hue, _clamp(x), _clamp(y))
    elif model in ("lab", "lch", "oklab", "oklch"):
        rgb = _gamut_map(_lab_rgb(model, tokens))
    elif model == "color":
        rgb = _gamut_map(_predefined(space, tokens))
    else:
        raise ValueError("unknown color function")
    result = tuple(_byte(v) for v in rgb)
    if alpha is not None:
        # Preserve Pillow's integer alpha extension in comma-separated rgba().
        if model == "rgba" and legacy and all(re.fullmatch(r"[0-9]+", v) for v in [*tokens, alpha]):
            result += (int(_clamp(float(alpha), 255)),)
        else:
            result += (_byte(_number(alpha)),)
    return result
