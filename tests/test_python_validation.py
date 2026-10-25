from __future__ import annotations

import builtins
from io import BytesIO, StringIO
from typing import TYPE_CHECKING

import pytest

from blanket import Image, ImageChops, ImageCms, ImageFile, ImageFilter, ImageFont, ImageHash, ImageOps, ImagePath, PngImagePlugin
from blanket._color import color_pixel
from blanket._draw_font import BitmapFont

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    ("operation", "error"),
    [
        (lambda im: Image.ImageTransformHandler().transform((1, 1), im), NotImplementedError),
        (lambda im: im.remap_palette([0]), ValueError),
        (lambda im: im.putpalette(b""), ValueError),
        (lambda im: im.paste(im, im, im), ValueError),
        (lambda im: im.paste(0), ValueError),
        (lambda im: im.paste(0, (0, 0, 1)), TypeError),
        (lambda im: im.paste(0, (1, 0, 0, 1)), ValueError),
        (lambda im: im.paste(Image.new("RGB", (1, 1)), (0, 0, 2, 2)), ValueError),
        (lambda im: im.point(list(range(256)) * 3, "L"), ValueError),
        (lambda im: im.thumbnail((0, 1)), ValueError),
        (lambda im: im.reduce((1,)), TypeError),
        (lambda im: im.reduce(2, (0, 0)), TypeError),
        (lambda im: im.reduce(2, (0, 0, 3, 1)), ValueError),
        (lambda im: im.reduce(2, (0, 0, 0, 1)), ValueError),
        (lambda im: im.transform((1,), Image.Transform.AFFINE, (1, 0, 0, 0, 1, 0)), TypeError),
        (lambda im: im.transform((-1, 1), Image.Transform.AFFINE, (1, 0, 0, 0, 1, 0)), ValueError),
        (lambda im: im.transform((1, 1), Image.Transform.MESH, [((0, 0), (0,) * 8)]), ValueError),
        (lambda im: im.alpha_composite(im, source=(0,)), ValueError),
        (lambda im: im.alpha_composite(im, dest=(0,)), ValueError),
        (lambda im: im.alpha_composite(im, source=(-1, 0)), ValueError),
        (lambda im: im.rotate(45, Image.Resampling.LANCZOS), ValueError),
        (lambda im: im.rotate(45, center=(float("nan"), 0)), ValueError),
        (lambda im: im.save(BytesIO(), "PNG", lossless=1), TypeError),
        (lambda im: im.save(BytesIO(), "TIFF", compression=1), ValueError),
        (lambda im: im.save("unknown.extension"), ValueError),
        (lambda im: Image.new("RGB", 2), ValueError),
        (lambda im: Image.open(BytesIO(), formats="PNG"), TypeError),
        (lambda im: Image.open(StringIO("text")), ValueError),
        (lambda im: Image.frombytes("L", (1, 1), object()), TypeError),
        (lambda im: Image.composite(im, Image.new("RGB", (1, 1)), Image.new("L", (1, 1))), ValueError),
        (lambda im: Image.blend(Image.new("P", (1, 1)), im, 0.5), ValueError),
        (lambda im: ImageChops.logical_and(im, im), ValueError),
        (lambda im: Image.new("F", (1, 1)).point([0] * 256), ValueError),
        (lambda im: Image.new("F", (1, 1)).save(BytesIO(), "PNG"), OSError),
        (lambda im: Image._save_options("JXL", {"lossless": 1}), TypeError),
    ],
)
def test_image_argument_errors(operation, error) -> None:
    with pytest.raises(error):
        operation(Image.new("RGB", (2, 2)))


@pytest.mark.parametrize("mode", ["LAB", "CMYK", "F"])
def test_rejects_compressor_for_special_modes(mode: str) -> None:
    from blanket.Compressor import LosslessImageCompressor

    with pytest.raises(ValueError, match="compression optimization"):
        Image.new(mode, (1, 1)).save(BytesIO(), "TIFF", compressor=LosslessImageCompressor())


@pytest.mark.parametrize(
    ("mode", "color", "expected", "error"),
    [
        ("L", [], None, TypeError),
        ("L", (1, 2), None, TypeError),
        ("LA", (7,), [7, 255], None),
        ("RGB", (1, 2, 3, 4), [1, 2, 3], None),
        ("RGB", (1,), None, TypeError),
        ("RGB", (1, 2, 0.5), None, TypeError),
        ("1", (300,), [255], None),
    ],
)
def test_fill_color_validation(mode, color, expected, error) -> None:
    if error:
        with pytest.raises(error):
            color_pixel(color, mode)
    else:
        assert color_pixel(color, mode) == expected


@pytest.mark.parametrize(
    "operation",
    [
        lambda: ImageOps.colorize(Image.new("RGB", (1, 1)), "black", "white"),
        lambda: ImageOps.colorize(Image.new("L", (1, 1)), "black", "white", blackpoint=256),
        lambda: ImageOps.colorize(Image.new("L", (1, 1)), "black", "white", mid="red", midpoint=256),
        lambda: ImageFilter.Kernel((3, 3), [1]),
        lambda: ImageFilter.Color3DLUT((2, 2), []),
        lambda: ImageFilter.Color3DLUT(2, [(1, 2)] * 8),
        lambda: ImageFilter.Color3DLUT.generate(2, lambda *args: args, channels=2),
        lambda: ImageFilter.Color3DLUT.generate(2, lambda *args: args).transform(lambda *args: args, channels=2),
        lambda: BitmapFont(0),
        lambda: BitmapFont().getbbox("a", anchor="bad"),
        lambda: ImageFont.load_default().getlength(42),
        lambda: PngImagePlugin.PngInfo().add(b"IDAT", b""),
        lambda: PngImagePlugin.PngInfo().add_text("", "value"),
    ],
)
def test_operation_validation(operation) -> None:
    with pytest.raises((ValueError, TypeError)):
        operation()


def test_abstract_filter_contract() -> None:
    with pytest.raises(NotImplementedError):
        ImageFilter.Filter.filter(None, Image.new("L", (1, 1))._native)


def test_palette_access_and_remapping() -> None:
    image = Image.new("RGB", (1, 1))
    assert image.getpalette() is None
    image = Image.new("P", (1, 1))
    image.putpalette(bytes([1, 2, 3, 4, 5, 6]))
    assert image.getpalette("RGBA") == [1, 2, 3, 255, 4, 5, 6, 255]
    with pytest.raises(ValueError, match="raw mode"):
        image.getpalette("L")
    with pytest.raises(ValueError, match="unique"):
        image.remap_palette([0, 0])
    with pytest.raises(ValueError, match="too few"):
        image.remap_palette([2])
    image.info["transparency"] = 1
    assert "transparency" not in image.remap_palette([0]).info
    image.putpalette(bytes([1, 2, 3, 4]), "RGBA")
    assert image.getpalette("RGB") == [1, 2, 3]
    with pytest.raises(ValueError, match="index"):
        image.putpixel((0, 0), (1, 2, 3))
    with pytest.raises(TypeError, match="color"):
        image.paste("red", (0, 0, 1, 1))
    image.putpixel([0, 0], 1)
    assert image.getpixel((0, 0)) == 1
    image.palette.palette = b"\x01\x02"
    with pytest.raises(ValueError, match="invalid palette"):
        image.getpalette("RGBA")


def test_empty_image_geometry() -> None:
    image = Image.new("L", (0, 2))
    image.thumbnail((1, 1))
    assert image.size == (0, 2)
    assert ImageOps.contain(Image.new("L", (1, 0)), (2, 2)).size == (2, 0)
    assert ImageOps.fit(Image.new("L", (1, 0)), (2, 2)).size == (2, 2)
    image = Image.new("L", (4, 2), 7)
    assert ImageOps.cover(image, (1, 2)).size == (4, 2)
    assert ImageOps.cover(image, (4, 1)).size == (4, 2)
    assert ImageOps.pad(image, image.size).tobytes() == image.tobytes()


def test_deform_rejects_unsupported_filter() -> None:
    class Mesh:
        def getmesh(self, image):
            return [((0, 0, 1, 1), (0, 0, 0, 1, 1, 1, 1, 0))]

    with pytest.raises(ValueError, match="mesh transforms"):
        ImageOps.deform(Image.new("L", (1, 1)), Mesh(), Image.Resampling.LANCZOS)


def test_path_allocation_assignment_and_bounds() -> None:
    with pytest.raises(MemoryError):
        ImagePath.Path(-1)
    path = ImagePath.Path([(1, 0), (0, 2)])
    with pytest.raises(IndexError):
        path[2] = (0, 0)
    with pytest.raises(TypeError):
        path[0] = [0, 0]
    assert path.getbbox() == (0, 0, 1, 2)


def test_hash_rejects_empty_serialization() -> None:
    with pytest.raises(ValueError, match="hexadecimal"):
        ImageHash.hex_to_hash("")
    with pytest.raises(TypeError, match="string"):
        ImageHash.hex_to_hash(1)


def test_parser_preview_and_failed_close() -> None:
    stream = BytesIO()
    Image.new("RGB", (3, 2), "red").save(stream, "JXL", lossless=True)
    parser = ImageFile.Parser()
    parser.feed(stream.getvalue())
    preview = parser.image
    assert preview is not None
    assert preview.tobytes() == parser.close().tobytes()
    parser = ImageFile.Parser()
    with pytest.raises(OSError):
        parser.feed(b"not an image")
    with pytest.raises(ValueError, match="closed"):
        parser.close()


@pytest.mark.parametrize("platform", ["darwin", "win32", "linux"])
def test_show_system_viewers(platform: str, monkeypatch, tmp_path: Path) -> None:
    paths = []
    monkeypatch.delenv("BLANKET_IMAGE_VIEWER", raising=False)
    monkeypatch.setattr(Image.sys, "platform", platform)
    monkeypatch.setattr(Image.tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(Image.os, "startfile", paths.append, raising=False)
    monkeypatch.setattr(Image.subprocess, "Popen", lambda command, **kwargs: paths.append(command[-1]))
    Image.new("RGB", (1, 1)).show()
    assert len(paths) == 1
    assert Image.open(paths[0]).size == (1, 1)


@pytest.mark.parametrize("failure", ["save", "viewer"])
def test_show_cleans_up_after_failure(failure: str, monkeypatch, tmp_path: Path) -> None:
    def fail(*args, **kwargs):
        raise OSError("failed")

    monkeypatch.setattr(Image.tempfile, "tempdir", str(tmp_path))
    monkeypatch.setenv("BLANKET_IMAGE_VIEWER", "viewer")
    monkeypatch.setattr(Image.Image if failure == "save" else Image.subprocess, "save" if failure == "save" else "Popen", fail)
    with pytest.raises(OSError, match="failed"):
        Image.new("RGB", (1, 1)).show()
    assert list(tmp_path.iterdir()) == []


def test_to_pillow_without_pillow(monkeypatch) -> None:
    original = builtins.__import__

    def import_module(name, *args, **kwargs):
        if name == "PIL":
            raise ImportError("unavailable")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_module)
    with pytest.raises(ImportError, match="Pillow is required"):
        Image.new("RGB", (1, 1)).to_pillow()


def test_cms_profile_accessors_and_in_place_errors() -> None:
    profile = ImageCms.createProfile("sRGB")
    assert profile.profile_description == ImageCms.getProfileDescription(profile).rstrip("\n")
    assert ImageCms.getProfileCopyright(profile).endswith("\n")
    assert ImageCms.getProfileManufacturer(profile).endswith("\n")
    assert ImageCms.getProfileModel(profile).endswith("\n")
    for model, manufacturer, expected in [("model", "maker", "model - maker\n"), ("x" * 31, "maker", "x" * 31 + "\n")]:
        profile._model, profile._manufacturer = model, manufacturer
        assert ImageCms.getProfileName(profile) == expected
    transform = ImageCms.buildTransform(profile, profile, "RGB", "RGB")
    image = Image.new("RGB", (1, 1))
    assert transform.apply(image, image) is image
    image.close()
    with pytest.raises(ImageCms.PyCMSError):
        transform.apply_in_place(image)


@pytest.mark.parametrize("mode", ["1", "I", "F", "I;16", "I;16L", "I;16B"])
def test_scalar_split_is_independent(mode: str) -> None:
    image = Image.new(mode, (1, 1), 1)
    (band,) = image.split()
    band.putpixel((0, 0), 0)
    assert image.getpixel((0, 0)) != band.getpixel((0, 0))
    assert repr(image) == f"<blanket.Image.Image image mode={mode} size=1x1>"


def test_indexed_geometry_uses_nearest_sampling() -> None:
    image = Image.new("P", (3, 2), 7)
    assert image.transform((3, 2), Image.Transform.AFFINE, (1, 0, 0, 0, 1, 0), Image.Resampling.BICUBIC).tobytes() == image.tobytes()
    assert image.rotate(45, Image.Resampling.BICUBIC).mode == "P"
    lut = ImageFilter.Color3DLUT(2, [(1, 0, 0)] * 8)
    assert Image.new("RGB", (1, 1)).filter(lut).getpixel((0, 0)) == (255, 0, 0)


@pytest.mark.parametrize("mode", ["I;16", "I;16L", "I;16B"])
def test_integer_saves_convert_for_eight_bit_formats(mode: str) -> None:
    output = BytesIO()
    Image.new(mode, (1, 1), 7).save(output, "BMP")
    assert Image.open(output).getpixel((0, 0)) == (7, 7, 7)
