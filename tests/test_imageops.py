from __future__ import annotations

import inspect
import subprocess
import sys
from io import BytesIO

import numpy as np
import pytest
from blanket import Image, ImageOps
from PIL import Image as PILImage
from PIL import ImageOps as PILOps
from PIL.PngImagePlugin import PngInfo


def pair(mode: str = "RGB", size: tuple[int, int] = (37, 29)) -> tuple[Image.Image, PILImage.Image]:
    raw = np.random.default_rng(42).integers(0, 256, size[0] * size[1] * len(mode), dtype=np.uint8).tobytes()
    return Image.frombytes(mode, size, raw), PILImage.frombytes(mode, size, raw)


def same(actual: Image.Image, expected: PILImage.Image) -> None:
    assert isinstance(actual, Image.Image)
    assert (actual.mode, actual.size) == (expected.mode, expected.size)
    assert actual.tobytes() == expected.tobytes()


def test_public_functions_match_pillow() -> None:
    expected = {name for name, value in vars(PILOps).items() if not name.startswith("_") and inspect.isfunction(value) and value.__module__ == PILOps.__name__}
    actual = {name for name in ImageOps.__all__ if inspect.isfunction(getattr(ImageOps, name))}
    assert actual == expected


@pytest.mark.parametrize("mode", ["L", "RGB"])
@pytest.mark.parametrize(
    "name,kwargs",
    [
        ("autocontrast", {}),
        ("autocontrast", {"cutoff": 4}),
        ("autocontrast", {"cutoff": (3, 9), "ignore": [0, 128, 255]}),
        ("autocontrast", {"cutoff": 2.5, "ignore": 17, "preserve_tone": True}),
        ("autocontrast", {"cutoff": (50, 50)}),
        ("equalize", {}),
        ("invert", {}),
        ("posterize", {"bits": 0}),
        ("posterize", {"bits": 4}),
        ("posterize", {"bits": 8}),
        ("solarize", {}),
        ("solarize", {"threshold": 0}),
        ("solarize", {"threshold": 256}),
    ],
)
def test_tone_operations(mode: str, name: str, kwargs: dict) -> None:
    b, p = pair(mode)
    same(getattr(ImageOps, name)(b, **kwargs), getattr(PILOps, name)(p, **kwargs))


@pytest.mark.parametrize("mode", ["L", "RGB"])
@pytest.mark.parametrize("name", ["autocontrast", "equalize"])
@pytest.mark.parametrize("mask_value", [None, 0, 1, 255])
def test_histogram_masks(mode: str, name: str, mask_value: int | None) -> None:
    b, p = pair(mode)
    bm, pm = pair("L") if mask_value is None else (Image.frombytes("L", b.size, bytes([mask_value]) * (b.width * b.height)), PILImage.new("L", p.size, mask_value))
    same(getattr(ImageOps, name)(b, mask=bm), getattr(PILOps, name)(p, mask=pm))


@pytest.mark.parametrize("mode", ["L", "RGB"])
@pytest.mark.parametrize("value", [0, 70, 255])
@pytest.mark.parametrize("name", ["autocontrast", "equalize"])
def test_constant_histograms(mode: str, value: int, name: str) -> None:
    p = PILImage.new(mode, (10, 10), value)
    b = Image.frombytes(mode, p.size, p.tobytes())
    same(getattr(ImageOps, name)(b), getattr(PILOps, name)(p))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"black": "navy", "white": "gold"},
        {"black": (250, 30, 17), "white": (20, 200, 55), "blackpoint": 20, "whitepoint": 240},
        {"black": "#123", "white": "rgb(90%, 50%, 20%)", "mid": "rebeccapurple"},
        {"black": "black", "white": "white", "mid": "red", "blackpoint": 30, "midpoint": 80, "whitepoint": 210},
        {"black": "red", "white": "blue", "blackpoint": 127, "whitepoint": 127},
        {"black": "red", "white": "blue", "mid": "green", "blackpoint": 127, "midpoint": 127, "whitepoint": 127},
    ],
)
def test_colorize(kwargs: dict) -> None:
    b = Image.frombytes("L", (256, 1), bytes(range(256)))
    p = PILImage.frombytes("L", b.size, b.tobytes())
    same(ImageOps.colorize(b, **kwargs), PILOps.colorize(p, **kwargs))


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
@pytest.mark.parametrize("name", ["flip", "mirror", "grayscale"])
def test_flips_and_grayscale(mode: str, name: str) -> None:
    b, p = pair(mode)
    b.info["custom"] = p.info["custom"] = "kept"
    result = getattr(ImageOps, name)(b)
    same(result, getattr(PILOps, name)(p))
    assert result.info == p.info


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
@pytest.mark.parametrize("border", [0, 2, (1, 3), (1, 2, 3, 4), -2, (-1, 3, 2, -4), (18, 14, 19, 15)])
@pytest.mark.parametrize("name", ["crop", "expand"])
def test_borders(mode: str, border: object, name: str) -> None:
    b, p = pair(mode)
    same(getattr(ImageOps, name)(b, border), getattr(PILOps, name)(p, border))


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
@pytest.mark.parametrize(
    "color", [0, 0x12345678, "red", "RebeccaPurple", "#123", "#abcd", "#12345678", "rgb(12, 34, 56)", "rgb(20%, 30%, 50%)", "rgba(1, 2, 3, 4)", "hsl(120, 30%, 50%)", "hsv(250, 50%, 80%)"]
)
def test_fill_colors(mode: str, color: object) -> None:
    b, p = pair(mode)
    same(ImageOps.expand(b, (1, 2), color), PILOps.expand(p, (1, 2), color))


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
@pytest.mark.parametrize("method", list(Image.Resampling))
@pytest.mark.parametrize(
    "name,kwargs",
    [
        ("contain", {"size": (19, 17)}),
        ("contain", {"size": (62, 65)}),
        ("cover", {"size": (19, 17)}),
        ("cover", {"size": (62, 65)}),
        ("fit", {"size": (17, 19)}),
        ("fit", {"size": (67, 43), "bleed": 0.07, "centering": (0.2, 0.8)}),
        ("fit", {"size": (13, 9), "bleed": 0.5, "centering": (-1, 2)}),
        ("pad", {"size": (20, 30), "color": "red", "centering": (0, 1)}),
        ("pad", {"size": (70, 50), "color": "#1234", "centering": (2, -1)}),
    ],
)
def test_resize_operations(mode: str, method: int, name: str, kwargs: dict) -> None:
    b, p = pair(mode)
    same(getattr(ImageOps, name)(b, method=method, **kwargs), getattr(PILOps, name)(p, method=method, **kwargs))


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
@pytest.mark.parametrize("method", list(Image.Resampling))
@pytest.mark.parametrize("factor", [0.5, 1, 1.75])
def test_scale(mode: str, method: int, factor: float) -> None:
    b, p = pair(mode)
    result = ImageOps.scale(b, factor, method)
    assert result is not b
    same(result, PILOps.scale(p, factor, method))


class Deformer:
    def __init__(self, mesh: list) -> None:
        self.mesh = mesh
        self.image = None

    def getmesh(self, image: object) -> list:
        self.image = image
        return self.mesh


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
@pytest.mark.parametrize("method", [0, 2, 3])
@pytest.mark.parametrize(
    "mesh",
    [
        [],
        [((0, 0, 37, 29), (0, 0, 0, 29, 37, 29, 37, 0))],
        [((2, 3, 32, 26), (1.2, 2.7, -1, 25, 33, 30, 31, 5))],
        [((0, 0, 37, 29), (37, 0, 37, 29, 0, 29, 0, 0))],
        [((0, 0, 18, 29), (0, 0, 0, 29, 37, 29, 37, 0)), ((18, 0, 37, 29), (0, 0, 0, 29, 37, 29, 37, 0))],
    ],
)
def test_deform(mode: str, method: int, mesh: list) -> None:
    b, p = pair(mode)
    deformer = Deformer(mesh)
    result = ImageOps.deform(b, deformer, method)
    assert deformer.image is b
    same(result, PILOps.deform(p, Deformer(mesh), method))


@pytest.mark.parametrize("orientation", range(1, 10))
@pytest.mark.parametrize("format", ["PNG", "JPEG"])
@pytest.mark.parametrize("in_place", [False, True])
def test_exif_transpose(orientation: int, format: str, in_place: bool) -> None:
    _, p = pair()
    exif = PILImage.Exif()
    exif[274] = orientation
    exif[270] = "An unrelated description with an external TIFF offset"
    exif[305] = "blanket test"
    stream = BytesIO()
    p.save(stream, format, exif=exif)
    b = Image.open(stream)
    p = PILImage.open(stream)
    p.load()
    original = b.tobytes()
    result = ImageOps.exif_transpose(b, in_place=in_place)
    expected = PILOps.exif_transpose(p, in_place=in_place)
    if in_place:
        assert result is expected is None
        result, expected = b, p
    else:
        assert result is not b
        assert b.tobytes() == original
    same(result, expected)
    actual_exif = PILImage.Exif()
    actual_exif.load(result.info["exif"])
    assert dict(actual_exif) == dict(expected.getexif())
    same(ImageOps.exif_transpose(result), expected)


@pytest.mark.parametrize("key", ["XML:com.adobe.xmp", "xmp"])
@pytest.mark.parametrize("kind", [str, bytes, tuple])
def test_exif_cleans_xmp(key: str, kind: type) -> None:
    b, p = pair()
    exif = PILImage.Exif()
    exif[274] = 6
    b.info["exif"] = p.info["exif"] = exif.tobytes()
    value = '<rdf tiff:Orientation="6"><tiff:Orientation>6</tiff:Orientation><keep/></rdf>'
    if kind is bytes:
        value = value.encode()
    elif kind is tuple:
        value = (value.encode(), b"unrelated")
    b.info[key] = p.info[key] = value
    actual, expected = ImageOps.exif_transpose(b), PILOps.exif_transpose(p)
    same(actual, expected)
    assert actual.info[key] == expected.info[key]
    assert b.info[key] == value


def test_xmp_only_png() -> None:
    _, p = pair()
    metadata = PngInfo()
    metadata.add_itxt("XML:com.adobe.xmp", '<rdf tiff:Orientation="8"/>', zip=True)
    stream = BytesIO()
    p.save(stream, "PNG", pnginfo=metadata)
    b, p = Image.open(stream), PILImage.open(stream)
    same(ImageOps.exif_transpose(b), PILOps.exif_transpose(p))


@pytest.mark.parametrize("name,kwargs", [("autocontrast", {}), ("equalize", {}), ("invert", {}), ("posterize", {"bits": 3}), ("solarize", {})])
def test_unsupported_lut_mode(name: str, kwargs: dict) -> None:
    b, p = pair("RGBA")
    for ops, image in ((ImageOps, b), (PILOps, p)):
        with pytest.raises(OSError):
            getattr(ops, name)(image, **kwargs)


@pytest.mark.parametrize(
    "name,args",
    [
        ("scale", (0,)),
        ("scale", (-1,)),
        ("crop", (100,)),
        ("crop", ((1, 2, 3),)),
        ("expand", (-100,)),
        ("expand", ((1,),)),
        ("expand", (1, "invalid-color")),
        ("contain", ((0, 10),)),
        ("fit", ((0, 10),)),
        ("scale", (2, 99)),
    ],
)
def test_invalid_arguments(name: str, args: tuple) -> None:
    b, p = pair()
    with pytest.raises(Exception) as expected:
        getattr(PILOps, name)(p, *args)
    with pytest.raises(type(expected.value)):
        getattr(ImageOps, name)(b, *args)


@pytest.mark.parametrize("mode,size", [("RGB", (2, 2)), ("L", (1, 1))])
def test_invalid_masks(mode: str, size: tuple[int, int]) -> None:
    b, _ = pair()
    mask, _ = pair(mode, size)
    for op in (ImageOps.autocontrast, ImageOps.equalize):
        with pytest.raises(ValueError):
            op(b, mask=mask)


def test_closed_image() -> None:
    b, _ = pair()
    b.close()
    for op in (ImageOps.invert, ImageOps.flip, ImageOps.equalize, ImageOps.exif_transpose):
        with pytest.raises(ValueError, match="closed"):
            op(b)


def test_no_pillow_runtime_dependency() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
sys.modules['PIL'] = None
from blanket import Image, ImageOps
im = Image.frombytes('RGB', (2, 2), bytes(range(12)))
assert ImageOps.fit(ImageOps.invert(im), (4, 3)).size == (4, 3)
assert ImageOps.expand(im, 1, 'rebeccapurple').size == (4, 4)
""",
        ],
        check=True,
    )


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
def test_processed_png_interop(mode: str) -> None:
    b, p = pair(mode)
    b = ImageOps.fit(ImageOps.mirror(b), (21, 17))
    p = PILOps.fit(PILOps.mirror(p), (21, 17))
    stream = BytesIO()
    b.save(stream, "PNG")
    loaded = PILImage.open(stream)
    assert loaded.tobytes() == p.tobytes()


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
@pytest.mark.parametrize("method", list(Image.Resampling))
@pytest.mark.parametrize("source,target", [((1, 1), (7, 9)), ((1, 7), (3, 11)), ((7, 1), (11, 3)), ((4, 4), (2, 4)), ((4, 4), (4, 2)), ((4, 4), (3, 3)), ((10, 10), (15, 15)), ((2, 205), (5, 61))])
def test_resize_thin_and_small_images(mode: str, method: int, source: tuple, target: tuple) -> None:
    b, p = pair(mode, source)
    same(ImageOps.fit(b, target, method), PILOps.fit(p, target, method))


@pytest.mark.parametrize("method", [0, 2, 3])
def test_deform_clipped_and_overlapping_boxes(method: int) -> None:
    b, p = pair()
    mesh = [((-5, -3, 20, 24), (0, 0, 0, 29, 37, 29, 37, 0)), ((10, 5, 40, 30), (3, 4, -2, 25, 28, 35, 37, 0))]
    same(ImageOps.deform(b, Deformer(mesh), method), PILOps.deform(p, Deformer(mesh), method))


@pytest.mark.parametrize("endian", ["<", ">"])
def test_exif_byte_order_and_profile(endian: str) -> None:
    import struct

    b, p = pair()
    raw = (b"II" if endian == "<" else b"MM") + struct.pack(endian + "HIHHHIH", 42, 8, 1, 274, 3, 1, 8) + bytes(6)
    b.info["Raw profile type exif"] = p.info["Raw profile type exif"] = f"\nexif\n{len(raw)}\n{raw.hex()}"
    actual, expected = ImageOps.exif_transpose(b), PILOps.exif_transpose(p)
    same(actual, expected)
    exif = PILImage.Exif()
    exif.load(bytes.fromhex(actual.info["Raw profile type exif"]))
    assert 274 not in exif


@pytest.mark.parametrize("compressed", [False, True])
def test_png_exif_profile(compressed: bool) -> None:
    _, p = pair()
    exif = PILImage.Exif()
    exif[274] = 3
    raw = exif.tobytes()
    metadata = PngInfo()
    metadata.add_text("Raw profile type exif", f"\nexif\n{len(raw)}\n{raw.hex()}", zip=compressed)
    stream = BytesIO()
    p.save(stream, "PNG", pnginfo=metadata)
    same(ImageOps.exif_transpose(Image.open(stream)), PILOps.exif_transpose(PILImage.open(stream)))


def test_jxl_exif_container() -> None:
    b, p = pair()
    stream = BytesIO()
    b.save(stream, "JXL", lossless=True)
    encoded = stream.getvalue()
    exif = PILImage.Exif()
    exif[274] = 6
    p.info["exif"] = exif.tobytes()
    payload = bytes(4) + exif.tobytes()[6:]
    box = (len(payload) + 8).to_bytes(4, "big") + b"Exif" + payload
    if encoded.startswith(b"\xff\x0a"):
        signature = b"\0\0\0\x0cJXL \r\n\x87\n"
        filetype = b"\0\0\0\x14ftypjxl \0\0\0\0jxl "
        encoded = signature + filetype + box + (len(encoded) + 8).to_bytes(4, "big") + b"jxlc" + encoded
    else:
        # Insert after the signature so a final zero-length codestream box
        # cannot swallow the metadata box.
        encoded = encoded[:12] + box + encoded[12:]
    same(ImageOps.exif_transpose(Image.open(BytesIO(encoded))), PILOps.exif_transpose(p))


@pytest.mark.parametrize("raw", [b"", b"garbage", b"Exif\0\0II", b"II*\0\xff\xff\xff\xff"])
def test_malformed_exif_does_not_break_pixels(raw: bytes) -> None:
    b, p = pair()
    b.info["exif"] = raw
    same(ImageOps.exif_transpose(b), p)


def test_deform_zero_area_box() -> None:
    b, p = pair()
    deformer = Deformer([((1, 0, 1, 10), (0, 0, 0, 10, 10, 10, 10, 0))])
    for ops, image in ((ImageOps, b), (PILOps, p)):
        with pytest.raises(ZeroDivisionError):
            ops.deform(image, deformer)


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
@pytest.mark.parametrize(
    "name,kwargs",
    [
        ("crop", {"border": (7, 9, 11, 13)}),
        ("expand", {"border": (3, 5), "fill": "#1234"}),
        ("flip", {}),
        ("mirror", {}),
        ("grayscale", {}),
        ("fit", {"size": (377, 311), "bleed": 0.07}),
        ("scale", {"factor": 1.5, "resample": Image.Resampling.NEAREST}),
    ],
)
def test_parallel_imageops_match_pillow(mode: str, name: str, kwargs: dict) -> None:
    # Large enough for worker partitions, with unaligned row and SIMD tails.
    b, p = pair(mode, (521, 509))
    same(getattr(ImageOps, name)(b, **kwargs), getattr(PILOps, name)(p, **kwargs))


@pytest.mark.parametrize("mode", ["L", "RGB"])
@pytest.mark.parametrize("name,kwargs", [("autocontrast", {"cutoff": (2, 7), "preserve_tone": True}), ("equalize", {}), ("invert", {}), ("posterize", {"bits": 3}), ("solarize", {"threshold": 173})])
def test_parallel_luts_and_histograms(mode: str, name: str, kwargs: dict) -> None:
    b, p = pair(mode, (521, 509))
    b_options, p_options = kwargs.copy(), kwargs.copy()
    if name in ("autocontrast", "equalize"):
        data = bytes(0 if i % 3 == 0 else 255 for i in range(521 * 509))
        b_options["mask"] = Image.frombytes("L", b.size, data)
        p_options["mask"] = PILImage.frombytes("L", p.size, data)
    same(getattr(ImageOps, name)(b, **b_options), getattr(PILOps, name)(p, **p_options))


def test_parallel_colorize() -> None:
    b, p = pair("L", (521, 509))
    same(ImageOps.colorize(b, "navy", "gold", "red"), PILOps.colorize(p, "navy", "gold", "red"))


@pytest.mark.parametrize("orientation", range(2, 9))
@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
def test_parallel_exif_tiles(mode: str, orientation: int) -> None:
    b, p = pair(mode, (521, 509))
    exif = PILImage.Exif()
    exif[274] = orientation
    b.info["exif"] = p.info["exif"] = exif.tobytes()
    same(ImageOps.exif_transpose(b), PILOps.exif_transpose(p))


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
def test_parallel_mesh_overwrite_order(mode: str) -> None:
    b, p = pair(mode, (521, 509))
    mesh = [((0, 0, 521, 509), (521, 0, 521, 509, 0, 509, 0, 0)), ((1, 1, 520, 508), (0, 0, 0, 507, 519, 507, 519, 0))]
    same(ImageOps.deform(b, Deformer(mesh)), PILOps.deform(p, Deformer(mesh)))
