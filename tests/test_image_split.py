"""Pillow 13.0.0 contracts; see PILLOW_TESTS.md."""

from __future__ import annotations

import pytest
from pillow_helpers import assert_image_equal, hopper

from blanket import Image, features

TYPE_CHECKING = False
if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("mode", ("1", "L", "I", "F", "P", "RGB", "RGBA", "CMYK", "YCbCr"))
def test_split_merge(mode: str) -> None:
    source = hopper(mode)
    bands = source.split()
    assert len(bands) == len(source.getbands())
    assert all(band.mode == (mode if len(bands) == 1 else "L") and band.size == source.size for band in bands)
    assert_image_equal(source, Image.merge(mode, bands))


def test_split_open(tmp_path: Path) -> None:
    if features.check("zlib"):
        test_file = tmp_path / "temp.png"
    else:
        test_file = tmp_path / "temp.pcx"

    def split_open(mode: str) -> int:
        hopper(mode).save(test_file)
        with Image.open(test_file) as im:
            return len(im.split())

    assert split_open("1") == 1
    assert split_open("L") == 1
    assert split_open("P") == 1
    assert split_open("RGB") == 3
    if features.check("zlib"):
        assert split_open("RGBA") == 4
