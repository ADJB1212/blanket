from __future__ import annotations

import os
import re
from io import BytesIO
from typing import TYPE_CHECKING, Any, cast

import pytest
from PIL import Image as PillowImage, JpegImagePlugin as PillowJpegImagePlugin
from pillow_helpers import PILLOW_IMAGES, assert_image, assert_image_equal, assert_image_similar, hopper

from blanket import Image, JpegImagePlugin, features
from blanket.Compressor import LosslessImageCompressor, LossyImageCompressor

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("mode", ["L", "RGB", "CMYK"])
@pytest.mark.parametrize("progressive", [False, True])
def test_jpeg_header_attributes_match_pillow(mode: str, progressive: bool) -> None:
    from blanket import JpegImagePlugin as BlanketJpeg

    encoded = Image.new(mode, (17, 13), 37).encode("JPEG", quality=85, progressive=progressive)
    actual = Image.open(BytesIO(encoded))
    expected = PillowImage.open(BytesIO(encoded))
    assert isinstance(actual, BlanketJpeg.JpegImageFile)
    assert actual.__dict__["applist"] == expected.applist
    assert actual.__dict__["app"] == expected.app
    assert actual.__dict__["layer"] == expected.layer
    assert actual.__dict__["quantization"] == expected.quantization
    assert BlanketJpeg.get_sampling(actual) == PillowJpegImagePlugin.get_sampling(expected)
    for key in ("jfif", "jfif_version", "jfif_unit", "jfif_density", "adobe", "adobe_transform", "progressive", "progression"):
        assert actual.info.get(key) == expected.info.get(key)


@pytest.mark.parametrize("comment", [b"", None, "sample comment", b"sample comment", b"x" * 65533])
@pytest.mark.parametrize("dpi", [(0, 0), (72.2, 300.4)])
def test_jpeg_comment_and_density_roundtrip(comment: str | bytes | None, dpi: tuple[float, float]) -> None:
    encoded = Image.new("RGB", (17, 13), "red").encode("JPEG", comment=comment, dpi=dpi)
    actual = Image.open(BytesIO(encoded))
    expected = PillowImage.open(BytesIO(encoded))
    assert actual.info.get("comment") == expected.info.get("comment")
    assert actual.info.get("dpi") == expected.info.get("dpi")
    if comment:
        output = actual.encode("JPEG")
        assert PillowImage.open(BytesIO(output)).info["comment"] == expected.info["comment"]
    with pytest.raises(ValueError, match="Comment is too long"):
        Image.new("RGB", (1, 1)).encode("JPEG", comment=b"x" * 65534)


def encode(image, **options):
    stream = BytesIO()
    image.save(stream, "JPEG", quality=85, **options)
    return stream.getvalue()


@pytest.mark.parametrize("width", [1, 3, 17, 33, 257])
@pytest.mark.parametrize("subsampling", [0, 1, 2])
def test_ycbcr_subsampling_preserves_odd_edges(width, subsampling):
    size = (width, 3)
    raw = bytes((i * 137 + i // 7) % 256 for i in range(width * 3 * 3))
    actual = encode(Image.frombytes("YCbCr", size, raw), subsampling=subsampling)
    expected = encode(PillowImage.frombytes("YCbCr", size, raw), subsampling=subsampling)
    assert PillowImage.open(BytesIO(actual)).tobytes() == PillowImage.open(BytesIO(expected)).tobytes()


@pytest.mark.parametrize("subsampling", [0, 1, 2])
def test_ycbcr_even_edge_padding_interoperates(subsampling):
    size = (2, 3)
    raw = bytes((i * 137 + i // 7) % 256 for i in range(18))
    encoded = encode(Image.frombytes("YCbCr", size, raw), subsampling=subsampling)
    reference = PillowImage.open(BytesIO(encoded))
    actual = Image.open(BytesIO(encoded))
    assert reference.size == actual.size == size
    assert PillowJpegImagePlugin.get_sampling(reference) == subsampling
    assert actual.tobytes() == reference.tobytes()


def test_jpeg_defaults_and_cached_encoder_reset():
    image = Image.frombytes("RGB", (17, 9), bytes(i % 256 for i in range(17 * 9 * 3)))
    baseline = encode(image)
    assert baseline == encode(image, subsampling=2, progressive=False, optimize=False)
    for subsampling in (0, 1, 2):
        optimized = encode(image, subsampling=subsampling, optimize=True)
        progressive = encode(image, subsampling=subsampling, progressive=True)
        plain = encode(image, subsampling=subsampling)
        assert PillowImage.open(BytesIO(optimized)).tobytes() == PillowImage.open(BytesIO(plain)).tobytes()
        assert PillowImage.open(BytesIO(progressive)).tobytes() == PillowImage.open(BytesIO(plain)).tobytes()
        assert encode(image) == baseline


@pytest.mark.parametrize("compressor", [LosslessImageCompressor(effort=3), LossyImageCompressor(effort=1)])
def test_jpeg_compressors_keep_subsampling_and_progressive(compressor):
    image = Image.frombytes("RGB", (17, 9), bytes(i * 31 % 256 for i in range(17 * 9 * 3)))
    output = encode(image, subsampling=0, progressive=True, optimize=True, compressor=compressor)
    decoded = PillowImage.open(BytesIO(output))
    assert decoded.info["progressive"]
    assert PillowJpegImagePlugin.get_sampling(decoded) == 0


@pytest.mark.parametrize("options,error", [({"subsampling": "keep"}, ValueError), ({"subsampling": "4:2:1"}, TypeError), ({"subsampling": 1.5}, TypeError), ({"optimize": "yes"}, TypeError)])
def test_jpeg_rejects_invalid_controls(options, error):
    image = Image.new("RGB", (2, 2))
    stream = BytesIO(b"unchanged")
    with pytest.raises(error):
        image.save(stream, "JPEG", **options)
    assert stream.getvalue() == b"unchanged"


@pytest.mark.parametrize("name", ["subsampling", "progressive", "progression", "optimize", "keep_rgb", "smooth", "qtables", "restart_marker_blocks", "restart_marker_rows", "streamtype"])
def test_jpeg_options_are_format_specific(name):
    with pytest.raises(TypeError, match="unsupported PNG save option"):
        Image.new("RGB", (2, 2)).save(BytesIO(), "PNG", **{name: 0})


@pytest.mark.parametrize("mode", ["YCbCr", "CMYK"])
@pytest.mark.parametrize("width", [1, 3, 16, 17, 33])
@pytest.mark.parametrize("options", [{}, {"progressive": True}, {"optimize": True}])
def test_jpeg_preset_component_edges(mode: str, width: int, options: dict[str, bool]) -> None:
    size = (width, 3)
    channels = len(PillowImage.new(mode, size).getbands())
    raw = bytes(i * 137 % 256 for i in range(width * 3 * channels))
    actual = Image.frombytes(mode, size, raw).encode("JPEG", quality="web_medium", **options)
    output = BytesIO()
    PillowImage.frombytes(mode, size, raw).save(output, "JPEG", quality="web_medium", **options)
    decoded = PillowImage.open(BytesIO(actual)).tobytes()
    reference = PillowImage.open(BytesIO(output.getvalue())).tobytes()
    assert max(abs(a - b) for a, b in zip(decoded, reference, strict=True)) <= 3


@pytest.mark.parametrize("quality", ["web_high", "web_low", "keep"])
def test_jpeg_quality_with_keep_rgb(quality: str) -> None:
    raw = bytes(i * 37 % 256 for i in range(17 * 9 * 3))
    source = Image.frombytes("RGB", (17, 9), raw).encode("JPEG", keep_rgb=True)
    if quality == "web_low":
        with pytest.raises(OSError):
            Image.open(BytesIO(source)).encode("JPEG", quality=quality, keep_rgb=True, subsampling=0)
        with pytest.raises(OSError):
            PillowImage.open(BytesIO(source)).save(BytesIO(), "JPEG", quality=quality, keep_rgb=True, subsampling=0)
        return
    actual = Image.open(BytesIO(source)).encode("JPEG", quality=quality, keep_rgb=True, subsampling=0)
    output = BytesIO()
    PillowImage.open(BytesIO(source)).save(output, "JPEG", quality=quality, keep_rgb=True, subsampling=0)
    assert PillowImage.open(BytesIO(actual)).tobytes() == PillowImage.open(BytesIO(output.getvalue())).tobytes()


@pytest.mark.parametrize("size", [(1, 1), (1, 3), (3, 1), (2, 2), (16, 16), (33, 17)])
@pytest.mark.parametrize("options", [{}, {"progressive": True}, {"optimize": True}, {"keep_rgb": True}])
def test_jpeg_smooth_edges_and_controls(size: tuple[int, int], options: dict[str, bool]) -> None:
    raw = bytes(i * 37 % 256 for i in range(size[0] * size[1] * 3))
    image = Image.frombytes("RGB", size, raw)
    reference = PillowImage.frombytes("RGB", size, raw)
    actual = encode(image, smooth=100, **options)
    expected = encode(reference, smooth=100, **options)
    with PillowImage.open(BytesIO(actual)) as decoded, PillowImage.open(BytesIO(expected)) as expected_image:
        assert decoded.layer == expected_image.layer
        assert bool(decoded.info.get("progressive")) == bool(expected_image.info.get("progressive"))
        assert max(abs(a - b) for a, b in zip(decoded.tobytes(), expected_image.tobytes(), strict=True)) <= 3
    assert encode(image, smooth=0) == encode(image)
    assert image.tobytes() == raw


@pytest.mark.parametrize("smooth,error", [(1.5, TypeError), ("100", TypeError), (None, TypeError), (2**31, OverflowError)])
def test_jpeg_smooth_rejects_invalid_arguments(smooth: object, error: type[Exception]) -> None:
    output = BytesIO(b"unchanged")
    with pytest.raises(error):
        Image.new("RGB", (1, 1)).save(output, "JPEG", smooth=smooth)
    assert output.getvalue() == b"unchanged"


def test_jpeg_smooth_lossless_compressor_preserves_pixels() -> None:
    raw = bytes(i * 137 % 256 for i in range(17 * 9 * 3))
    image = Image.frombytes("RGB", (17, 9), raw)
    options = {"smooth": 100, "quality": "web_medium", "progressive": True}
    baseline = image.encode("JPEG", **options)
    optimized = image.encode("JPEG", **options, compressor=LosslessImageCompressor(effort=3))
    with PillowImage.open(BytesIO(baseline)) as expected, PillowImage.open(BytesIO(optimized)) as actual:
        assert actual.tobytes() == expected.tobytes()
        assert actual.quantization == expected.quantization
        assert actual.info["progressive"]


class TestFileJpeg:
    def roundtrip_with_bytes(self, im: Image.Image, **options: Any) -> tuple[JpegImagePlugin.JpegImageFile, int]:
        out = BytesIO()
        im.save(out, "JPEG", **options)
        test_bytes = out.tell()
        out.seek(0)
        reloaded = cast("JpegImagePlugin.JpegImageFile", Image.open(out))
        return reloaded, test_bytes

    def roundtrip(self, im: Image.Image, **options: Any) -> JpegImagePlugin.JpegImageFile:
        return self.roundtrip_with_bytes(im, **options)[0]

    def gen_random_image(self, size: tuple[int, int], mode: str = "RGB") -> Image.Image:
        """Generates a very hard to compress file
        :param size: tuple
        :param mode: optional image mode

        """
        return Image.frombytes(mode, size, os.urandom(size[0] * size[1] * len(mode)))

    def test_rgb(self) -> None:
        def getchannels(im: JpegImagePlugin.JpegImageFile) -> tuple[int, ...]:
            return tuple(v[0] for v in im.layer)

        im = hopper()
        im_ycbcr = self.roundtrip(im)
        assert getchannels(im_ycbcr) == (1, 2, 3)
        assert_image_similar(im, im_ycbcr, 17)

        im_rgb = self.roundtrip(im, keep_rgb=True)
        assert getchannels(im_rgb) == (ord("R"), ord("G"), ord("B"))
        assert_image_similar(im, im_rgb, 12)

    def test_optimize(self) -> None:
        im1, im1_bytes = self.roundtrip_with_bytes(hopper())
        im2, im2_bytes = self.roundtrip_with_bytes(hopper(), optimize=0)
        im3, im3_bytes = self.roundtrip_with_bytes(hopper(), optimize=1)
        assert_image_equal(im1, im2)
        assert_image_equal(im1, im3)
        assert im1_bytes >= im2_bytes
        assert im1_bytes >= im3_bytes

    def test_progressive_compat(self) -> None:
        im1 = self.roundtrip(hopper())
        assert not im1.info.get("progressive")
        assert not im1.info.get("progression")

        im2 = self.roundtrip(hopper(), progressive=0)
        im3 = self.roundtrip(hopper(), progression=0)  # compatibility
        assert not im2.info.get("progressive")
        assert not im2.info.get("progression")
        assert not im3.info.get("progressive")
        assert not im3.info.get("progression")

        im2 = self.roundtrip(hopper(), progressive=1)
        im3 = self.roundtrip(hopper(), progression=1)  # compatibility
        if features.check_feature("mozjpeg"):
            assert_image_similar(im1, im2, 9.39)
            assert_image_similar(im1, im3, 9.39)
        else:
            assert_image_equal(im1, im2)
            assert_image_equal(im1, im3)
        assert im2.info.get("progressive")
        assert im2.info.get("progression")
        assert im3.info.get("progressive")
        assert im3.info.get("progression")

    def test_quality(self) -> None:
        im1, im1_bytes = self.roundtrip_with_bytes(hopper())
        im2, im2_bytes = self.roundtrip_with_bytes(hopper(), quality=50)
        assert_image(im1, im2.mode, im2.size)
        assert im1_bytes >= im2_bytes

        im3, im3_bytes = self.roundtrip_with_bytes(hopper(), quality=0)
        assert_image(im1, im3.mode, im3.size)
        assert im2_bytes > im3_bytes

    def test_smooth(self) -> None:
        im1 = self.roundtrip(hopper())
        im2 = self.roundtrip(hopper(), smooth=100)
        assert_image(im1, im2.mode, im2.size)

    def test_subsampling(self) -> None:
        def getsampling(im: JpegImagePlugin.JpegImageFile) -> tuple[int, int, int, int, int, int]:
            layer = im.layer
            return layer[0][1:3] + layer[1][1:3] + layer[2][1:3]

        # experimental API
        for subsampling in (-1, 3):  # (default, invalid)
            im = self.roundtrip(hopper(), subsampling=subsampling)
            assert getsampling(im) == (2, 2, 1, 1, 1, 1)
        for subsampling1 in (0, "4:4:4"):
            im = self.roundtrip(hopper(), subsampling=subsampling1)
            assert getsampling(im) == (1, 1, 1, 1, 1, 1)
        for subsampling1 in (1, "4:2:2"):
            im = self.roundtrip(hopper(), subsampling=subsampling1)
            assert getsampling(im) == (2, 1, 1, 1, 1, 1)
        for subsampling1 in (2, "4:2:0", "4:1:1"):
            im = self.roundtrip(hopper(), subsampling=subsampling1)
            assert getsampling(im) == (2, 2, 1, 1, 1, 1)

        # RGB colorspace
        for subsampling1 in (-1, 0, "4:4:4"):
            # "4:4:4" doesn't really make sense for RGB, but the conversion
            # to an integer happens at a higher level
            im = self.roundtrip(hopper(), keep_rgb=True, subsampling=subsampling1)
            assert getsampling(im) == (1, 1, 1, 1, 1, 1)
        for subsampling1 in (1, "4:2:2", 2, "4:2:0", 3):
            with pytest.raises(OSError):
                self.roundtrip(hopper(), keep_rgb=True, subsampling=subsampling1)

        with pytest.raises(TypeError):
            self.roundtrip(hopper(), subsampling="1:1:1")

    def test_quality_keep(self, tmp_path: Path) -> None:
        # RGB
        with Image.open(PILLOW_IMAGES / "hopper.jpg") as im:
            f = tmp_path / "temp.jpg"
            im.save(f, quality="keep")
        # Grayscale
        with Image.open(PILLOW_IMAGES / "hopper_gray.jpg") as im:
            f = tmp_path / "temp.jpg"
            im.save(f, quality="keep")
        # CMYK
        with Image.open(PILLOW_IMAGES / "pil_sample_cmyk.jpg") as im:
            f = tmp_path / "temp.jpg"
            im.save(f, quality="keep")

    def test_qtables(self) -> None:
        def _n_qtables_helper(n: int, test_file: str) -> None:
            b = BytesIO()
            with Image.open(test_file) as im:
                im.save(b, "JPEG", qtables=[[n] * 64] * n)
            with Image.open(b) as im:
                assert isinstance(im, JpegImagePlugin.JpegImageFile)
                assert len(im.quantization) == n
                reloaded = self.roundtrip(im, qtables="keep")
                assert im.quantization == reloaded.quantization
                assert max(reloaded.quantization[0]) <= 255

        with Image.open(PILLOW_IMAGES / "hopper.jpg") as im:
            assert isinstance(im, JpegImagePlugin.JpegImageFile)
            qtables = im.quantization
            reloaded = self.roundtrip(im, qtables=qtables, subsampling=0)
            assert im.quantization == reloaded.quantization
            assert_image_similar(im, self.roundtrip(im, qtables="web_low"), 30)
            assert_image_similar(im, self.roundtrip(im, qtables="web_high"), 30)
            assert_image_similar(im, self.roundtrip(im, qtables="keep"), 30)

            # valid bounds for baseline qtable
            bounds_qtable = [int(s) for s in ("255 1 " * 32).split(None)]
            im2 = self.roundtrip(im, qtables=[bounds_qtable])
            assert im2.quantization == {0: bounds_qtable}

            # values from wizard.txt in jpeg9-a src package.
            standard_l_qtable = [
                int(s)
                for s in [
                    "16",
                    "11",
                    "10",
                    "16",
                    "24",
                    "40",
                    "51",
                    "61",
                    "12",
                    "12",
                    "14",
                    "19",
                    "26",
                    "58",
                    "60",
                    "55",
                    "14",
                    "13",
                    "16",
                    "24",
                    "40",
                    "57",
                    "69",
                    "56",
                    "14",
                    "17",
                    "22",
                    "29",
                    "51",
                    "87",
                    "80",
                    "62",
                    "18",
                    "22",
                    "37",
                    "56",
                    "68",
                    "109",
                    "103",
                    "77",
                    "24",
                    "35",
                    "55",
                    "64",
                    "81",
                    "104",
                    "113",
                    "92",
                    "49",
                    "64",
                    "78",
                    "87",
                    "103",
                    "121",
                    "120",
                    "101",
                    "72",
                    "92",
                    "95",
                    "98",
                    "112",
                    "100",
                    "103",
                    "99",
                ]
            ]

            standard_chrominance_qtable = [
                int(s)
                for s in [
                    "17",
                    "18",
                    "24",
                    "47",
                    "99",
                    "99",
                    "99",
                    "99",
                    "18",
                    "21",
                    "26",
                    "66",
                    "99",
                    "99",
                    "99",
                    "99",
                    "24",
                    "26",
                    "56",
                    "99",
                    "99",
                    "99",
                    "99",
                    "99",
                    "47",
                    "66",
                    "99",
                    "99",
                    "99",
                    "99",
                    "99",
                    "99",
                    "99",
                    "99",
                    "99",
                    "99",
                    "99",
                    "99",
                    "99",
                    "99",
                    "99",
                    "99",
                    "99",
                    "99",
                    "99",
                    "99",
                    "99",
                    "99",
                    "99",
                    "99",
                    "99",
                    "99",
                    "99",
                    "99",
                    "99",
                    "99",
                    "99",
                    "99",
                    "99",
                    "99",
                    "99",
                    "99",
                    "99",
                    "99",
                ]
            ]

            for quality in range(101):
                qtable_from_qtable_quality = self.roundtrip(im, qtables={0: standard_l_qtable, 1: standard_chrominance_qtable}, quality=quality).quantization

                qtable_from_quality = self.roundtrip(im, quality=quality).quantization

                if features.check_feature("libjpeg_turbo"):
                    assert qtable_from_qtable_quality == qtable_from_quality
                else:
                    assert qtable_from_qtable_quality[0] == qtable_from_quality[0]
                    assert qtable_from_qtable_quality[1][1:] == qtable_from_quality[1][1:]

            # list of qtable lists
            assert_image_similar(im, self.roundtrip(im, qtables=[standard_l_qtable, standard_chrominance_qtable]), 30)

            # tuple of qtable lists
            assert_image_similar(im, self.roundtrip(im, qtables=(standard_l_qtable, standard_chrominance_qtable)), 30)

            # dict of qtable lists
            assert_image_similar(im, self.roundtrip(im, qtables={0: standard_l_qtable, 1: standard_chrominance_qtable}), 30)

            _n_qtables_helper(1, PILLOW_IMAGES / "hopper_gray.jpg")
            _n_qtables_helper(1, PILLOW_IMAGES / "pil_sample_rgb.jpg")
            _n_qtables_helper(2, PILLOW_IMAGES / "pil_sample_rgb.jpg")
            _n_qtables_helper(3, PILLOW_IMAGES / "pil_sample_rgb.jpg")
            _n_qtables_helper(1, PILLOW_IMAGES / "pil_sample_cmyk.jpg")
            _n_qtables_helper(2, PILLOW_IMAGES / "pil_sample_cmyk.jpg")
            _n_qtables_helper(3, PILLOW_IMAGES / "pil_sample_cmyk.jpg")
            _n_qtables_helper(4, PILLOW_IMAGES / "pil_sample_cmyk.jpg")

            # not a sequence
            with pytest.raises(ValueError):
                self.roundtrip(im, qtables="a")
            # sequence wrong length
            with pytest.raises(ValueError):
                self.roundtrip(im, qtables=[])
            # sequence wrong length
            with pytest.raises(ValueError):
                self.roundtrip(im, qtables=[1, 2, 3, 4, 5])

            # qtable entry not a sequence
            with pytest.raises(ValueError):
                self.roundtrip(im, qtables=[1])
            # qtable entry has wrong number of items
            with pytest.raises(ValueError):
                self.roundtrip(im, qtables=[[1, 2, 3, 4]])

    def test_qtables_incorrect_length(self) -> None:
        class IncorrectLength(list[list[int]]):
            def __len__(self) -> int:
                return 4

        im = Image.new("1", (1, 1))
        im2 = self.roundtrip(im, qtables=IncorrectLength([[1] * 64]))
        assert len(im2.quantization) == 1

    def test_save_multiple_16bit_qtables(self) -> None:
        with Image.open(PILLOW_IMAGES / "hopper_16bit_qtables.jpg") as im:
            assert isinstance(im, JpegImagePlugin.JpegImageFile)
            im2 = self.roundtrip(im, qtables="keep")
            assert im.quantization == im2.quantization

    def test_save_single_16bit_qtable(self) -> None:
        with Image.open(PILLOW_IMAGES / "hopper_16bit_qtables.jpg") as im:
            assert isinstance(im, JpegImagePlugin.JpegImageFile)
            im2 = self.roundtrip(im, qtables={0: im.quantization[0]})
            assert len(im2.quantization) == 1
            assert im2.quantization[0] == im.quantization[0]

    @pytest.mark.parametrize("blocks, rows, markers", ((0, 0, 0), (1, 0, 15), (3, 0, 5), (8, 0, 1), (0, 1, 3), (0, 2, 1)))
    def test_restart_markers(self, blocks: int, rows: int, markers: int) -> None:
        im = Image.new("RGB", (32, 32))  # 16 MCUs
        out = BytesIO()
        im.save(
            out,
            format="JPEG",
            restart_marker_blocks=blocks,
            restart_marker_rows=rows,
            # force 8x8 pixel MCUs
            subsampling=0,
        )
        assert len(re.findall(b"\xff[\xd0-\xd7]", out.getvalue())) == markers

    def test_MAXBLOCK_scaling(self, tmp_path: Path) -> None:
        im = self.gen_random_image((512, 512))
        f = tmp_path / "temp.jpeg"
        im.save(f, quality=100, optimize=True)

        with Image.open(f) as reloaded:
            # none of these should crash
            reloaded.save(f, quality="keep")
            reloaded.save(f, quality="keep", progressive=True)
            reloaded.save(f, quality="keep", optimize=True)

    @pytest.mark.parametrize("mode", ("1", "L", "RGB", "RGBX", "CMYK", "YCbCr"))
    def test_save_correct_modes(self, mode: str) -> None:
        out = BytesIO()
        img = Image.new(mode, (20, 20))
        img.save(out, "JPEG")

    def test_save_xmp(self, tmp_path: Path) -> None:
        f = tmp_path / "temp.jpg"
        im = hopper()
        im.save(f, xmp=b"XMP test")
        with Image.open(f) as reloaded:
            assert reloaded.info["xmp"] == b"XMP test"

            # Check that XMP is not saved from image info
            reloaded.save(f)

        with Image.open(f) as reloaded:
            assert "xmp" not in reloaded.info

        im.save(f, xmp=b"1" * 65504)
        with Image.open(f) as reloaded:
            assert reloaded.info["xmp"] == b"1" * 65504

        with pytest.raises(ValueError):
            im.save(f, xmp=b"1" * 65505)

    def test_separate_tables(self) -> None:
        im = hopper()
        data = []  # [interchange, tables-only, image-only]
        for streamtype in range(3):
            out = BytesIO()
            im.save(out, format="JPEG", streamtype=streamtype)
            data.append(out.getvalue())

        # SOI, EOI
        for marker in b"\xff\xd8", b"\xff\xd9":
            assert marker in data[1]
            assert marker in data[2]

        # DQT
        markers = [b"\xff\xdb"]
        if features.check_feature("libjpeg_turbo"):
            # DHT
            markers.append(b"\xff\xc4")
        for marker in markers:
            assert marker in data[1]
            assert marker not in data[2]

        # SOF0, SOS, APP0 (JFIF header)
        for marker in b"\xff\xc0", b"\xff\xda", b"\xff\xe0":
            assert marker not in data[1]
            assert marker in data[2]

        with Image.open(BytesIO(data[0])) as interchange_im, Image.open(BytesIO(data[1] + data[2])) as combined_im:
            assert_image_equal(interchange_im, combined_im)
