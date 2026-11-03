from __future__ import annotations

import pytest
from PIL import Image as PillowImage

from blanket import Image

PACKED_MODES = ["I;16S", "I;16LS", "I;16BS", "I;16NS", "I;32", "I;32L", "I;32B", "I;32S", "I;32LS", "I;32BS"]


def test_mode_catalog_matches_pillow() -> None:
    assert Image.MODES == PillowImage.MODES


@pytest.mark.parametrize("mode", [*PillowImage.MODES, *PACKED_MODES])
def test_mode_metadata_matches_pillow(mode: str, subtests: pytest.Subtests) -> None:
    for helper in ["getmodebase", "getmodetype", "getmodebands", "getmodebandnames"]:
        with subtests.test(mode=mode, helper=helper):
            assert getattr(Image, helper)(mode) == getattr(PillowImage, helper)(mode)


@pytest.mark.parametrize("mode", ["", "invalid", "RGB;16"])
@pytest.mark.parametrize("helper", ["getmodebase", "getmodetype", "getmodebands", "getmodebandnames"])
def test_unknown_mode_metadata_raises_key_error(mode: str, helper: str) -> None:
    with pytest.raises(KeyError) as actual:
        getattr(Image, helper)(mode)
    assert actual.value.args == (mode,)


@pytest.mark.parametrize("mode", ["1", "L", "LA", "P", "PA", "RGB", "RGBA", "CMYK", "HSV", "YCbCr", "LAB", "I", "F", "I;16", "I;16L", "I;16B", "RGB;F", "RGBA;F"])
def test_mode_band_metadata_matches_native_image(mode: str) -> None:
    image = Image.new(mode, (3, 2))
    assert Image.getmodebandnames(mode) == image.getbands()
    assert Image.getmodebands(mode) == len(image.getbands())


@pytest.mark.parametrize("mode", ["RGB;F", "RGBA;F"])
def test_float_color_mode_metadata(mode: str) -> None:
    assert Image.getmodebase(mode) == "RGB"
    assert Image.getmodetype(mode) == "F"
