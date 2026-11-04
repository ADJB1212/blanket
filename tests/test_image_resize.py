# Copied Pillow 13.0.0 contracts; see PILLOW_TESTS.md.

"""
Tests for resize functionality.
"""

from __future__ import annotations

import pytest
from helpers import assert_raises
from pillow_helpers import PILLOW_IMAGES, assert_image_equal, assert_image_equal_tofile, assert_image_similar, hopper, skip_unless_feature

from blanket import Image, ImageFile

TYPE_CHECKING = False
if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path


@pytest.fixture
def gradients_image() -> Generator[ImageFile.ImageFile]:
    with Image.open(PILLOW_IMAGES / "radial_gradients.png") as im:
        im.load()
    try:
        yield im
    finally:
        im.close()


class TestReducingGapResize:
    def test_reducing_gap_values(self, gradients_image: ImageFile.ImageFile) -> None:
        ref = gradients_image.resize((52, 34), Image.Resampling.BICUBIC, reducing_gap=None)
        im = gradients_image.resize((52, 34), Image.Resampling.BICUBIC)
        assert_image_equal(ref, im)

        assert_raises(ValueError, lambda: gradients_image.resize((52, 34), Image.Resampling.BICUBIC, reducing_gap=0))

        assert_raises(ValueError, lambda: gradients_image.resize((52, 34), Image.Resampling.BICUBIC, reducing_gap=0.99))

    @pytest.mark.parametrize("box, epsilon", ((None, 4), ((1.1, 2.2, 510.8, 510.9), 4), ((3, 10, 410, 256), 10)))
    def test_reducing_gap_1(self, gradients_image: ImageFile.ImageFile, box: tuple[float, float, float, float], epsilon: float) -> None:
        ref = gradients_image.resize((52, 34), Image.Resampling.BICUBIC, box=box)
        im = gradients_image.resize((52, 34), Image.Resampling.BICUBIC, box=box, reducing_gap=1.0)

        assert_image_similar(ref, im, epsilon)
        assert ref.tobytes() != im.tobytes()

    @pytest.mark.parametrize("box, epsilon", ((None, 1.5), ((1.1, 2.2, 510.8, 510.9), 1.5), ((3, 10, 410, 256), 1)))
    def test_reducing_gap_2(self, gradients_image: ImageFile.ImageFile, box: tuple[float, float, float, float], epsilon: float) -> None:
        ref = gradients_image.resize((52, 34), Image.Resampling.BICUBIC, box=box)
        im = gradients_image.resize((52, 34), Image.Resampling.BICUBIC, box=box, reducing_gap=2.0)

        assert ref.tobytes() != im.tobytes()
        assert_image_similar(ref, im, epsilon)

    @pytest.mark.parametrize("box, epsilon", ((None, 1), ((1.1, 2.2, 510.8, 510.9), 1), ((3, 10, 410, 256), 0.5)))
    def test_reducing_gap_3(self, gradients_image: ImageFile.ImageFile, box: tuple[float, float, float, float], epsilon: float) -> None:
        ref = gradients_image.resize((52, 34), Image.Resampling.BICUBIC, box=box)
        im = gradients_image.resize((52, 34), Image.Resampling.BICUBIC, box=box, reducing_gap=3.0)

        assert ref.tobytes() != im.tobytes()
        assert_image_similar(ref, im, epsilon)

    @pytest.mark.parametrize("box", (None, (1.1, 2.2, 510.8, 510.9), (3, 10, 410, 256)))
    def test_reducing_gap_8(self, gradients_image: ImageFile.ImageFile, box: tuple[float, float, float, float]) -> None:
        ref = gradients_image.resize((52, 34), Image.Resampling.BICUBIC, box=box)
        im = gradients_image.resize((52, 34), Image.Resampling.BICUBIC, box=box, reducing_gap=8.0)

        assert_image_equal(ref, im)

    @pytest.mark.parametrize("mode", ("RGBA", "LA"))
    def test_reducing_gap_alpha(self, mode: str) -> None:
        with Image.open(PILLOW_IMAGES / "transparent.png") as im:
            im_converted = im.convert(mode)

        # Premultiply the alpha manually
        premultiplied = im_converted.convert({"RGBA": "RGBa", "LA": "La"}[mode])
        expected = premultiplied.resize((16, 16), reducing_gap=2).convert(mode)

        # Test that resizing the image applies the alpha in the same way
        assert_image_equal(im_converted.resize((16, 16), reducing_gap=2), expected)

    @pytest.mark.parametrize("box, epsilon", (((0, 0, 512, 512), 5.5), ((0.9, 1.7, 128, 128), 9.5)))
    def test_box_filter(self, gradients_image: ImageFile.ImageFile, box: tuple[float, float, float, float], epsilon: float) -> None:
        ref = gradients_image.resize((52, 34), Image.Resampling.BOX, box=box)
        im = gradients_image.resize((52, 34), Image.Resampling.BOX, box=box, reducing_gap=1.0)

        assert_image_similar(ref, im, epsilon)


class TestImagingCoreResize:
    def test_cross_platform(self, tmp_path: Path) -> None:
        # This test is intended for only check for consistent behaviour across
        # platforms. So if a future Pillow change requires that the test file
        # be updated, that is okay.
        im = hopper().resize((64, 64))
        temp_file = tmp_path / "temp.gif"
        im.save(temp_file)

        with Image.open(temp_file) as reloaded:
            assert_image_equal_tofile(reloaded, PILLOW_IMAGES / "hopper_resized.gif")


class TestImageResize:
    def test_resize(self) -> None:
        def resize(mode: str, size: tuple[int, int] | list[int]) -> None:
            out = hopper(mode).resize(size)
            assert out.mode == mode
            assert out.size == tuple(size)

        for mode in "1", "P", "L", "RGB", "I", "F":
            resize(mode, (112, 103))
            resize(mode, [188, 214])

        # Test unknown resampling filter
        with hopper() as im, pytest.raises(ValueError):
            im.resize((10, 10), -1)

    @skip_unless_feature("libtiff")
    def test_transposed(self) -> None:
        with Image.open(PILLOW_IMAGES / "g4_orientation_5.tif") as im:
            im_resized = im.resize((64, 64))
            assert im_resized.size == (64, 64)

    @pytest.mark.parametrize("mode", ("L", "RGB", "I", "I;16", "I;16L", "I;16B", "I;16N", "F"))
    def test_default_filter_bicubic(self, mode: str) -> None:
        im = hopper(mode)
        assert im.resize((20, 20), Image.Resampling.BICUBIC) == im.resize((20, 20))

    @pytest.mark.parametrize("mode", ("1", "P"))
    def test_default_filter_nearest(self, mode: str) -> None:
        im = hopper(mode)
        assert im.resize((20, 20), Image.Resampling.NEAREST) == im.resize((20, 20))
